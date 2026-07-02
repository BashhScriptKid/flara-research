//! Character-level language model — step-1 proof of concept.
//! Validates that SharedMonarchMatmul can learn before scaling to 1B.
//!
//! Change the four CONFIG constants to run at full Fydel-1B scale.

use std::time::Instant;
use std::sync::atomic::{AtomicBool, Ordering};
use std::cell::RefCell;
use std::collections::BTreeMap;
use fydel::kernels::monarch::{SharedMonarchMatmul, FwdCache, Grads};
use fydel::kernels::norm;
use fydel::kernels::fastmath;

// ---------------------------------------------------------------------------
// Tree-shaped phase profiling (investigation only — PROFILE=1 env var to
// enable). RAII spans: `let _s = span("name");` times until the guard drops.
// Nested spans accumulate under their caller's path, so the printed tree
// shows inclusive time per node (a parent's total includes its children's).
// Single-threaded harness, so plain thread_local + RefCell, no locking.
// ---------------------------------------------------------------------------
static PROFILE_ON: AtomicBool = AtomicBool::new(false);

thread_local! {
    static SPAN_STACK: RefCell<Vec<&'static str>> = RefCell::new(Vec::new());
    static SPAN_TIMES: RefCell<BTreeMap<Vec<&'static str>, u64>> = RefCell::new(BTreeMap::new());
}

struct Span { active: bool, name: &'static str, t0: Instant }

#[inline]
fn span(name: &'static str) -> Span {
    if PROFILE_ON.load(Ordering::Relaxed) {
        SPAN_STACK.with(|s| s.borrow_mut().push(name));
        Span { active: true, name, t0: Instant::now() }
    } else {
        Span { active: false, name, t0: Instant::now() }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if !self.active { return; }
        let elapsed = self.t0.elapsed().as_nanos() as u64;
        SPAN_STACK.with(|s| {
            let mut stack = s.borrow_mut();
            stack.pop(); // remove self
            let mut key: Vec<&'static str> = stack.clone();
            key.push(self.name);
            drop(stack);
            SPAN_TIMES.with(|m| *m.borrow_mut().entry(key).or_insert(0) += elapsed);
        });
    }
}

fn enable_profiling() {
    PROFILE_ON.store(true, Ordering::Relaxed);
}

fn print_profile_summary() {
    struct Node { own_ns: u64, children: BTreeMap<&'static str, Node> }
    impl Node {
        fn new() -> Self { Node { own_ns: 0, children: BTreeMap::new() } }
    }
    let mut root = Node::new();
    SPAN_TIMES.with(|m| {
        for (path, ns) in m.borrow().iter() {
            let mut cur = &mut root;
            for seg in path {
                cur = cur.children.entry(seg).or_insert_with(Node::new);
            }
            cur.own_ns += ns;
        }
    });
    fn print_node(name: &str, node: &Node, depth: usize, parent_ns: u64) {
        let ns = node.own_ns;
        let ms = ns as f64 / 1e6;
        let pct = if parent_ns > 0 { 100.0 * ns as f64 / parent_ns as f64 } else { 100.0 };
        let indent = "  ".repeat(depth);
        eprintln!("{indent}{name:<24} {ms:>10.1} ms   {pct:>5.1}%");
        for (cname, child) in &node.children {
            print_node(cname, child, depth + 1, ns);
        }
    }
    eprintln!("\n─── phase profile (tree, inclusive time, sum over all layers/steps) ───");
    let total: u64 = root.children.values().map(|n| n.own_ns).sum();
    for (name, node) in &root.children {
        print_node(name, node, 0, total);
    }
    eprintln!("{:<26} {:>10.1} ms   100.0%", "total (tracked)", total as f64 / 1e6);
}

// ---------------------------------------------------------------------------
// Config — the only lines you need to change for scale experiments
// ---------------------------------------------------------------------------
const HIDDEN:   usize = 256;   // 896 for full scale  (must be multiple of 64)
const FFN_DIM:  usize = 1024;  // 3072 for full scale (must be multiple of 64)
const N_HEADS:  usize = 4;     // HIDDEN / 64         (14 for full scale)
const N_LAYERS: usize = 2;     // 96 for full scale
const ACCUM_STEPS: usize = 4;  // sequences per opt-step — reduces gradient variance 4×
const LR_WARMUP:   usize = 100; // linear warmup opt-steps before cosine decay

const VOCAB:    usize = 128;   // printable ASCII
const SEQ_LEN:  usize = 128;
const HEAD_DIM: usize = HIDDEN / N_HEADS; // 64 — AVX2 path optimised for this
const B:        usize = 64;
const ND:       usize = 8;
const M:        usize = 8; // sqrt(B)

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------
fn encode(text: &str) -> Vec<usize> {
    text.bytes().map(|b| (b as usize).min(VOCAB - 1)).collect()
}

struct Dataset { data: Vec<usize> }

impl Dataset {
    fn load(path: &str) -> Self {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|_| {
                eprintln!("warn: {path} not found, using fallback text");
                "To be, or not to be, that is the question: \
                 Whether 'tis nobler in the mind to suffer \
                 The slings and arrows of outrageous fortune, \
                 Or to take arms against a sea of troubles. ".repeat(100)
            });
        Self { data: encode(&text) }
    }

    fn sample(&self, s: &mut u64) -> (Vec<usize>, Vec<usize>) {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let max_start = self.data.len().saturating_sub(SEQ_LEN + 1).max(1);
        let start = ((*s >> 20) as usize) % max_start;
        (self.data[start..start + SEQ_LEN].to_vec(),
         self.data[start + 1..start + SEQ_LEN + 1].to_vec())
    }
}

// ---------------------------------------------------------------------------
// Loss: cross-entropy. Returns (mean_loss, dlogits [SEQ*VOCAB]).
// dlogits = (softmax - one_hot) / SEQ, ready to back-prop.
// ---------------------------------------------------------------------------
fn cross_entropy(logits: &[f32], targets: &[usize]) -> (f32, Vec<f32>) {
    let mut dloss = vec![0.0f32; SEQ_LEN * VOCAB];
    let mut total = 0.0f32;
    for t in 0..SEQ_LEN {
        let row = &logits[t * VOCAB..(t + 1) * VOCAB];
        let max_l = row.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = row.iter().map(|&x| (x - max_l).exp()).collect();
        let sum_e: f32 = exps.iter().sum();
        total -= (exps[targets[t]] / sum_e).ln();
        let dr = &mut dloss[t * VOCAB..(t + 1) * VOCAB];
        for v in 0..VOCAB { dr[v] = exps[v] / (sum_e * SEQ_LEN as f32); }
        dr[targets[t]] -= 1.0 / SEQ_LEN as f32;
    }
    (total / SEQ_LEN as f32, dloss)
}

// ---------------------------------------------------------------------------
// Causal multi-head attention — forward and backward
// ---------------------------------------------------------------------------
fn attn_forward(
    q: &[f32], k: &[f32], v: &[f32], // each [SEQ * HIDDEN]
) -> (Vec<f32>, Vec<f32>) {           // (out [SEQ*HIDDEN], probs [H*S*S])
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut out   = vec![0.0f32; SEQ_LEN * HIDDEN];
    let mut probs = vec![0.0f32; N_HEADS * SEQ_LEN * SEQ_LEN];
    for h in 0..N_HEADS {
        let hd0 = h * HEAD_DIM;
        for t in 0..SEQ_LEN {
            let qth = &q[t * HIDDEN + hd0..t * HIDDEN + hd0 + HEAD_DIM];
            let mut max_s = f32::MIN;
            let mut scores = [0.0f32; SEQ_LEN];
            for s in 0..=t {
                let ksh = &k[s * HIDDEN + hd0..s * HIDDEN + hd0 + HEAD_DIM];
                scores[s] = qth.iter().zip(ksh).map(|(a, b)| a * b).sum::<f32>() * scale;
                if scores[s] > max_s { max_s = scores[s]; }
            }
            let sum_e: f32 = (0..=t).map(|s| (scores[s] - max_s).exp()).sum();
            for s in 0..=t {
                let p = (scores[s] - max_s).exp() / sum_e;
                probs[h * SEQ_LEN * SEQ_LEN + t * SEQ_LEN + s] = p;
                let vsh = &v[s * HIDDEN + hd0..s * HIDDEN + hd0 + HEAD_DIM];
                let oth = &mut out[t * HIDDEN + hd0..t * HIDDEN + hd0 + HEAD_DIM];
                for i in 0..HEAD_DIM { oth[i] += p * vsh[i]; }
            }
        }
    }
    (out, probs)
}

fn attn_backward(
    q: &[f32], k: &[f32], v: &[f32],
    probs: &[f32],
    d_out: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) { // (dq, dk, dv)
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut dq = vec![0.0f32; SEQ_LEN * HIDDEN];
    let mut dk = vec![0.0f32; SEQ_LEN * HIDDEN];
    let mut dv = vec![0.0f32; SEQ_LEN * HIDDEN];
    for h in 0..N_HEADS {
        let hd0 = h * HEAD_DIM;
        let p_head = &probs[h * SEQ_LEN * SEQ_LEN..(h + 1) * SEQ_LEN * SEQ_LEN];
        for t in 0..SEQ_LEN {
            let d_oth = &d_out[t * HIDDEN + hd0..t * HIDDEN + hd0 + HEAD_DIM];
            let p_row = &p_head[t * SEQ_LEN..t * SEQ_LEN + t + 1];
            // dV[s,h] += probs[t,s] * d_out[t,h]
            for s in 0..=t {
                let dvsh = &mut dv[s * HIDDEN + hd0..s * HIDDEN + hd0 + HEAD_DIM];
                for i in 0..HEAD_DIM { dvsh[i] += p_row[s] * d_oth[i]; }
            }
            // d_probs[s] = dot(d_out[t,h], v[s,h])
            let mut d_probs = [0.0f32; SEQ_LEN];
            for s in 0..=t {
                let vsh = &v[s * HIDDEN + hd0..s * HIDDEN + hd0 + HEAD_DIM];
                d_probs[s] = d_oth.iter().zip(vsh).map(|(a, b)| a * b).sum();
            }
            // softmax backward: d_scores = p * (dp - dot(p, dp))
            let dot_pdp: f32 = (0..=t).map(|s| p_row[s] * d_probs[s]).sum();
            // dQ[t,h] += sum_s d_scores[s] * K[s,h]
            let dqth = &mut dq[t * HIDDEN + hd0..t * HIDDEN + hd0 + HEAD_DIM];
            let qth  = &q[t * HIDDEN + hd0..t * HIDDEN + hd0 + HEAD_DIM];
            for s in 0..=t {
                let ds = p_row[s] * (d_probs[s] - dot_pdp) * scale;
                let ksh = &k[s * HIDDEN + hd0..s * HIDDEN + hd0 + HEAD_DIM];
                for i in 0..HEAD_DIM { dqth[i] += ds * ksh[i]; }
                // dK[s,h] += d_scores[s] * Q[t,h]
                let dksh = &mut dk[s * HIDDEN + hd0..s * HIDDEN + hd0 + HEAD_DIM];
                for i in 0..HEAD_DIM { dksh[i] += ds * qth[i]; }
            }
        }
    }
    (dq, dk, dv)
}

// ---------------------------------------------------------------------------
// Monarch helpers
// ---------------------------------------------------------------------------
fn zero_grads(p: &SharedMonarchMatmul) -> Grads {
    let b2 = p.m * p.m;
    Grads {
        dd1: vec![0.0f32; p.nd * b2], dd2: vec![0.0f32; p.nd * b2],
        da1: vec![0.0f32; p.p * p.q * p.m * p.nd],
        da2: vec![0.0f32; p.p * p.q * p.m * p.nd],
    }
}

fn acc_grads(acc: &mut Grads, g: Grads) {
    acc.dd1.iter_mut().zip(g.dd1).for_each(|(a, b)| *a += b);
    acc.dd2.iter_mut().zip(g.dd2).for_each(|(a, b)| *a += b);
    acc.da1.iter_mut().zip(g.da1).for_each(|(a, b)| *a += b);
    acc.da2.iter_mut().zip(g.da2).for_each(|(a, b)| *a += b);
}

fn monarch_new(in_dim: usize, out_dim: usize, seed: u64) -> SharedMonarchMatmul {
    let mut mm = SharedMonarchMatmul::new(out_dim / B, in_dim / B, M, ND, seed);
    // Depth-scale atoms so Var(output) = 1/(2·N_LAYERS). Var ∝ s_atom⁴, so
    // multiply atom values by (1/(2·N_LAYERS))^(1/4).
    let depth_factor = 1.0_f32 / (2.0 * N_LAYERS as f32).powf(0.25);
    for v in mm.d1.iter_mut().chain(mm.d2.iter_mut()) { *v *= depth_factor; }
    mm
}

// ---------------------------------------------------------------------------
// Adam optimiser
// ---------------------------------------------------------------------------
struct Adam { m: Vec<f32>, v: Vec<f32>, t: u32 }

impl Adam {
    fn new(n: usize) -> Self { Self { m: vec![0.0; n], v: vec![0.0; n], t: 0 } }

    fn step(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        const B1: f32 = 0.9; const B2: f32 = 0.999; const EPS: f32 = 1e-8;
        self.t += 1;
        let bc1 = 1.0 - B1.powi(self.t as i32);
        let bc2 = 1.0 - B2.powi(self.t as i32);
        for i in 0..params.len() {
            self.m[i] = B1 * self.m[i] + (1.0 - B1) * grads[i];
            self.v[i] = B2 * self.v[i] + (1.0 - B2) * grads[i] * grads[i];
            params[i] -= lr * (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + EPS);
        }
    }
}

struct ProjAdam { d1: Adam, d2: Adam, a1: Adam, a2: Adam }
impl ProjAdam {
    fn new(p: &SharedMonarchMatmul) -> Self {
        let b2 = p.m * p.m;
        Self {
            d1: Adam::new(p.nd * b2), d2: Adam::new(p.nd * b2),
            a1: Adam::new(p.p * p.q * p.m * p.nd),
            a2: Adam::new(p.p * p.q * p.m * p.nd),
        }
    }
    fn step(&mut self, proj: &mut SharedMonarchMatmul, g: &Grads, lr: f32) {
        self.d1.step(&mut proj.d1, &g.dd1, lr);
        self.d2.step(&mut proj.d2, &g.dd2, lr);
        self.a1.step(&mut proj.a1, &g.da1, lr);
        self.a2.step(&mut proj.a2, &g.da2, lr);
    }
}

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------
struct Layer {
    wq: SharedMonarchMatmul, wk: SharedMonarchMatmul,
    wv: SharedMonarchMatmul, wo: SharedMonarchMatmul,
    attn_gain: Vec<f32>,
    w_up: SharedMonarchMatmul, w_gate: SharedMonarchMatmul,
    w_down: SharedMonarchMatmul,
    ffn_gain: Vec<f32>,
}

struct LayerCache {
    x_in:     Vec<f32>,          // [S*H]
    h_attn:   Vec<f32>,          // [S*H] after attn norm
    attn_r:   Vec<f32>,          // [S] RMSNorm r
    q: Vec<f32>, k: Vec<f32>, v: Vec<f32>, // each [S*H]
    // batched caches: forward_batch's cache covers all S tokens in one
    // FwdCache; use `proj.zs_at(&cache, t)` to get a single token's slice.
    q_fc: FwdCache, k_fc: FwdCache, v_fc: FwdCache,
    probs:    Vec<f32>,           // [H*S*S]
    attn_out: Vec<f32>,           // [S*H]
    o_fc:     FwdCache,
    h_mid:    Vec<f32>,           // [S*H] after attn residual
    h_ffn:    Vec<f32>,           // [S*H] after ffn norm
    ffn_r:    Vec<f32>,           // [S]
    up: Vec<f32>, gate: Vec<f32>, // each [S*F]
    up_fc: FwdCache, gate_fc: FwdCache,
    act:      Vec<f32>,           // [S*F] SwiGLU output
    down_fc:  FwdCache,
}

struct LayerGrads {
    wq_g: Grads, wk_g: Grads, wv_g: Grads, wo_g: Grads,
    w_up_g: Grads, w_gate_g: Grads, w_down_g: Grads,
    d_attn_gain: Vec<f32>, d_ffn_gain: Vec<f32>,
}

struct LayerAdam {
    wq: ProjAdam, wk: ProjAdam, wv: ProjAdam, wo: ProjAdam,
    w_up: ProjAdam, w_gate: ProjAdam, w_down: ProjAdam,
    attn_gain: Adam, ffn_gain: Adam,
}

impl Layer {
    fn new(seed: u64) -> Self {
        Self {
            wq:    monarch_new(HIDDEN, HIDDEN, seed ^ 1),
            wk:    monarch_new(HIDDEN, HIDDEN, seed ^ 2),
            wv:    monarch_new(HIDDEN, HIDDEN, seed ^ 3),
            wo:    monarch_new(HIDDEN, HIDDEN, seed ^ 4),
            attn_gain: vec![1.0f32; HIDDEN],
            w_up:   monarch_new(HIDDEN, FFN_DIM, seed ^ 5),
            w_gate: monarch_new(HIDDEN, FFN_DIM, seed ^ 6),
            w_down: monarch_new(FFN_DIM, HIDDEN, seed ^ 7),
            ffn_gain: vec![1.0f32; HIDDEN],
        }
    }

    fn adam(l: &Layer) -> LayerAdam {
        LayerAdam {
            wq: ProjAdam::new(&l.wq), wk: ProjAdam::new(&l.wk),
            wv: ProjAdam::new(&l.wv), wo: ProjAdam::new(&l.wo),
            w_up: ProjAdam::new(&l.w_up), w_gate: ProjAdam::new(&l.w_gate),
            w_down: ProjAdam::new(&l.w_down),
            attn_gain: Adam::new(HIDDEN), ffn_gain: Adam::new(HIDDEN),
        }
    }

    fn forward(&self, x: &[f32]) -> (Vec<f32>, LayerCache) {
        let s = SEQ_LEN;
        let mut h_attn  = vec![0.0f32; s * HIDDEN];
        let mut attn_r  = vec![0.0f32; s];
        {
            let _s = span("norm_fwd");
            for t in 0..s {
                let xt = &x[t * HIDDEN..(t + 1) * HIDDEN];
                let hat = &mut h_attn[t * HIDDEN..(t + 1) * HIDDEN];
                attn_r[t] = norm::forward(xt, &self.attn_gain, 1e-5, hat);
            }
        }

        // One batched call per projection (all S tokens at once) instead of
        // S separate rayon dispatches — amortizes rayon's per-call dispatch
        // overhead across the whole sequence.
        let (q, q_fc);
        let (k, k_fc);
        let (v, v_fc);
        {
            let _s = span("qkv_proj_fwd");
            (q, q_fc) = { let _s = span("wq_fwd"); self.wq.forward_batch(&h_attn, s) };
            (k, k_fc) = { let _s = span("wk_fwd"); self.wk.forward_batch(&h_attn, s) };
            (v, v_fc) = { let _s = span("wv_fwd"); self.wv.forward_batch(&h_attn, s) };
        }

        let (attn_out, probs) = { let _s = span("attn_core_fwd"); attn_forward(&q, &k, &v) };

        let mut h_mid = vec![0.0f32; s * HIDDEN];
        let o_fc;
        {
            let _s = span("wo_proj_fwd");
            let (ot, oc) = self.wo.forward_batch(&attn_out, s);
            for i in 0..(s * HIDDEN) { h_mid[i] = x[i] + ot[i]; }
            o_fc = oc;
        }

        let mut h_ffn   = vec![0.0f32; s * HIDDEN];
        let mut ffn_r   = vec![0.0f32; s];
        {
            let _s = span("norm_fwd");
            for t in 0..s {
                let hm = &h_mid[t * HIDDEN..(t + 1) * HIDDEN];
                let hf = &mut h_ffn[t * HIDDEN..(t + 1) * HIDDEN];
                ffn_r[t] = norm::forward(hm, &self.ffn_gain, 1e-5, hf);
            }
        }

        let (up, up_fc);
        let (gate, gate_fc);
        let mut act = vec![0.0f32; s * FFN_DIM];
        let (down, down_fc);
        let mut out = vec![0.0f32; s * HIDDEN];
        {
            let _s = span("ffn_block_fwd");
            (up, up_fc)     = { let _s = span("up_proj_fwd");   self.w_up.forward_batch(&h_ffn, s) };
            (gate, gate_fc) = { let _s = span("gate_proj_fwd"); self.w_gate.forward_batch(&h_ffn, s) };
            { let _s = span("swiglu_fwd"); fastmath::swiglu_forward(&up, &gate, &mut act); }
            (down, down_fc) = { let _s = span("down_proj_fwd"); self.w_down.forward_batch(&act, s) };
            for i in 0..(s * HIDDEN) { out[i] = h_mid[i] + down[i]; }
        }

        (out, LayerCache {
            x_in: x.to_vec(), h_attn, attn_r, q, k, v, q_fc, k_fc, v_fc,
            probs, attn_out, o_fc, h_mid, h_ffn, ffn_r,
            up, gate, up_fc, gate_fc, act, down_fc,
        })
    }

    fn backward(&self, dout: &[f32], c: &LayerCache) -> (Vec<f32>, LayerGrads) {
        let s = SEQ_LEN;
        let mut dx       = vec![0.0f32; s * HIDDEN];
        let mut d_h_mid  = vec![0.0f32; s * HIDDEN];
        let mut d_ffn_gain = vec![0.0f32; HIDDEN];
        let mut w_down_g = zero_grads(&self.w_down);
        let mut w_up_g   = zero_grads(&self.w_up);
        let mut w_gate_g = zero_grads(&self.w_gate);

        // --- FFN backward ---
        // Each token's contribution is computed independently (in parallel,
        // via rayon) and collected into a Vec that preserves token order
        // (par_chunks_mut/into_par_iter are indexed iterators), then merged
        // sequentially in that fixed order below — same math, same
        // summation order as the old per-token loop, just the independent
        // per-token compute now runs on multiple cores. No profiling spans
        // inside the parallel closure: each rayon worker has its own
        // thread-local span stack, so per-op spans recorded there would be
        // invisible to print_profile_summary (which only reads the calling
        // thread's stack) — the outer span still gives an accurate total.
        {
            let _s = span("ffn_block_bwd");
            use rayon::prelude::*;
            struct FfnTokGrad { w_down_g: Grads, w_up_g: Grads, w_gate_g: Grads, d_ffn_gain: Vec<f32> }
            let per_token: Vec<FfnTokGrad> = dout.par_chunks(HIDDEN)
                .zip(d_h_mid.par_chunks_mut(HIDDEN))
                .enumerate()
                .map(|(t, (dout_t, d_h_mid_t))| {
                    for i in 0..HIDDEN { d_h_mid_t[i] += dout_t[i]; }

                    let mut d_act_t = vec![0.0f32; FFN_DIM];
                    let w_down_g = self.w_down.backward(
                        &c.act[t * FFN_DIM..(t + 1) * FFN_DIM],
                        self.w_down.zs_at(&c.down_fc, t), dout_t, &mut d_act_t);

                    let up_t   = &c.up[t * FFN_DIM..(t + 1) * FFN_DIM];
                    let gate_t = &c.gate[t * FFN_DIM..(t + 1) * FFN_DIM];
                    let mut d_up_t   = vec![0.0f32; FFN_DIM];
                    let mut d_gate_t = vec![0.0f32; FFN_DIM];
                    fastmath::swiglu_backward(up_t, gate_t, &d_act_t, &mut d_up_t, &mut d_gate_t);

                    let hf = &c.h_ffn[t * HIDDEN..(t + 1) * HIDDEN];
                    let mut dx_up   = vec![0.0f32; HIDDEN];
                    let mut dx_gate = vec![0.0f32; HIDDEN];
                    let w_up_g   = self.w_up.backward(hf, self.w_up.zs_at(&c.up_fc, t), &d_up_t, &mut dx_up);
                    let w_gate_g = self.w_gate.backward(hf, self.w_gate.zs_at(&c.gate_fc, t), &d_gate_t, &mut dx_gate);

                    let mut d_hf = vec![0.0f32; HIDDEN];
                    for i in 0..HIDDEN { d_hf[i] = dx_up[i] + dx_gate[i]; }
                    let mut d_ffn_gain_t = vec![0.0f32; HIDDEN];
                    norm::backward(&c.h_mid[t * HIDDEN..(t + 1) * HIDDEN],
                        &self.ffn_gain, &d_hf, c.ffn_r[t],
                        d_h_mid_t, &mut d_ffn_gain_t);

                    FfnTokGrad { w_down_g, w_up_g, w_gate_g, d_ffn_gain: d_ffn_gain_t }
                })
                .collect();

            for tg in per_token {
                acc_grads(&mut w_down_g, tg.w_down_g);
                acc_grads(&mut w_up_g, tg.w_up_g);
                acc_grads(&mut w_gate_g, tg.w_gate_g);
                for i in 0..HIDDEN { d_ffn_gain[i] += tg.d_ffn_gain[i]; }
            }
        }

        // --- Attention output projection backward ---
        let mut d_attn_out = vec![0.0f32; s * HIDDEN];
        let mut wo_g = zero_grads(&self.wo);
        {
            let _s = span("wo_proj_bwd");
            use rayon::prelude::*;
            for i in 0..(s * HIDDEN) { dx[i] += d_h_mid[i]; }
            let per_token: Vec<Grads> = c.attn_out.par_chunks(HIDDEN)
                .zip(d_h_mid.par_chunks(HIDDEN))
                .zip(d_attn_out.par_chunks_mut(HIDDEN))
                .enumerate()
                .map(|(t, ((ao, dh_mid), d_attn_out_t))| {
                    self.wo.backward(ao, self.wo.zs_at(&c.o_fc, t), dh_mid, d_attn_out_t)
                })
                .collect();
            for g in per_token { acc_grads(&mut wo_g, g); }
        }

        // --- Attention QKV backward ---
        let (dq, dk, dv) = { let _s = span("attn_core_bwd"); attn_backward(&c.q, &c.k, &c.v, &c.probs, &d_attn_out) };

        let mut d_h_attn    = vec![0.0f32; s * HIDDEN];
        let mut d_attn_gain = vec![0.0f32; HIDDEN];
        let mut wq_g = zero_grads(&self.wq);
        let mut wk_g = zero_grads(&self.wk);
        let mut wv_g = zero_grads(&self.wv);
        {
            let _s = span("qkv_proj_bwd");
            use rayon::prelude::*;
            struct QkvTokGrad { wq_g: Grads, wk_g: Grads, wv_g: Grads, d_attn_gain: Vec<f32> }
            let per_token: Vec<QkvTokGrad> = c.h_attn.par_chunks(HIDDEN)
                .zip(dq.par_chunks(HIDDEN)).zip(dk.par_chunks(HIDDEN)).zip(dv.par_chunks(HIDDEN))
                .zip(d_h_attn.par_chunks_mut(HIDDEN))
                .zip(dx.par_chunks_mut(HIDDEN))
                .enumerate()
                .map(|(t, (((((hat, dq_t), dk_t), dv_t), dh), dx_t))| {
                    let wq_g = self.wq.backward(hat, self.wq.zs_at(&c.q_fc, t), dq_t, dh);
                    let wk_g = self.wk.backward(hat, self.wk.zs_at(&c.k_fc, t), dk_t, dh);
                    let wv_g = self.wv.backward(hat, self.wv.zs_at(&c.v_fc, t), dv_t, dh);
                    let mut d_attn_gain_t = vec![0.0f32; HIDDEN];
                    norm::backward(&c.x_in[t * HIDDEN..(t + 1) * HIDDEN],
                        &self.attn_gain, dh, c.attn_r[t],
                        dx_t, &mut d_attn_gain_t);
                    QkvTokGrad { wq_g, wk_g, wv_g, d_attn_gain: d_attn_gain_t }
                })
                .collect();

            for tg in per_token {
                acc_grads(&mut wq_g, tg.wq_g);
                acc_grads(&mut wk_g, tg.wk_g);
                acc_grads(&mut wv_g, tg.wv_g);
                for i in 0..HIDDEN { d_attn_gain[i] += tg.d_attn_gain[i]; }
            }
        }

        (dx, LayerGrads {
            wq_g, wk_g, wv_g, wo_g,
            w_up_g, w_gate_g, w_down_g,
            d_attn_gain, d_ffn_gain,
        })
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------
struct Model {
    embed:      Vec<f32>,  // [VOCAB * HIDDEN]
    layers:     Vec<Layer>,
    out_gain:   Vec<f32>,  // [HIDDEN] final RMSNorm
}

struct ModelCache {
    tokens:     Vec<usize>,
    layer_caches: Vec<LayerCache>,
    h_final:    Vec<f32>, // [S*H] after last layer
    h_norm:     Vec<f32>, // [S*H] after final norm
    norm_r:     Vec<f32>, // [S]
}

struct ModelGrads {
    d_embed:    Vec<f32>,
    d_out_gain: Vec<f32>,
    layer_grads: Vec<LayerGrads>,
}

struct ModelAdam {
    embed:     Adam,
    out_gain:  Adam,
    layers:    Vec<LayerAdam>,
}

impl Model {
    fn new(seed: u64) -> Self {
        let scale = 0.02f32;
        let mut s = seed;
        let embed: Vec<f32> = (0..VOCAB * HIDDEN).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
        }).collect();
        let layers = (0..N_LAYERS).map(|i| Layer::new(seed ^ (i as u64 * 0x1000))).collect();
        Self { embed, layers, out_gain: vec![1.0f32; HIDDEN] }
    }

    fn param_count(&self) -> usize {
        VOCAB * HIDDEN + HIDDEN
            + N_LAYERS * (self.layers[0].wq.param_count() * 4
                + self.layers[0].w_up.param_count() * 2
                + self.layers[0].w_down.param_count()
                + HIDDEN * 2)
    }

    fn forward(&self, tokens: &[usize]) -> (Vec<f32>, ModelCache) {
        let s = SEQ_LEN;
        // Embed
        let mut h: Vec<f32> = tokens.iter().flat_map(|&t| {
            self.embed[t * HIDDEN..(t + 1) * HIDDEN].iter().copied()
        }).collect();
        // Layers
        let mut layer_caches = Vec::with_capacity(N_LAYERS);
        for layer in &self.layers {
            let (nh, lc) = layer.forward(&h);
            h = nh;
            layer_caches.push(lc);
        }
        let h_final = h.clone();
        // Final norm
        let mut h_norm = vec![0.0f32; s * HIDDEN];
        let mut norm_r = vec![0.0f32; s];
        for t in 0..s {
            norm_r[t] = norm::forward(
                &h[t * HIDDEN..(t + 1) * HIDDEN],
                &self.out_gain, 1e-5,
                &mut h_norm[t * HIDDEN..(t + 1) * HIDDEN]);
        }
        // Output head: logits[t,v] = dot(h_norm[t], embed[v])  (tied weights)
        let mut logits = vec![0.0f32; s * VOCAB];
        for t in 0..s {
            let hn = &h_norm[t * HIDDEN..(t + 1) * HIDDEN];
            for v in 0..VOCAB {
                let ev = &self.embed[v * HIDDEN..(v + 1) * HIDDEN];
                logits[t * VOCAB + v] = hn.iter().zip(ev).map(|(a, b)| a * b).sum();
            }
        }
        (logits, ModelCache { tokens: tokens.to_vec(), layer_caches, h_final, h_norm, norm_r })
    }

    fn backward(&self, c: &ModelCache, dlogits: &[f32]) -> ModelGrads {
        let s = SEQ_LEN;
        let mut d_embed    = vec![0.0f32; VOCAB * HIDDEN];
        let mut d_out_gain = vec![0.0f32; HIDDEN];

        // Output head backward (tied weights)
        let mut d_h_norm = vec![0.0f32; s * HIDDEN];
        for t in 0..s {
            let dlog = &dlogits[t * VOCAB..(t + 1) * VOCAB];
            let hn   = &c.h_norm[t * HIDDEN..(t + 1) * HIDDEN];
            let dhn  = &mut d_h_norm[t * HIDDEN..(t + 1) * HIDDEN];
            for v in 0..VOCAB {
                let ev = &self.embed[v * HIDDEN..(v + 1) * HIDDEN];
                let dl = dlog[v];
                for i in 0..HIDDEN { dhn[i] += dl * ev[i]; }
                let de = &mut d_embed[v * HIDDEN..(v + 1) * HIDDEN];
                for i in 0..HIDDEN { de[i] += dl * hn[i]; }
            }
        }

        // Final norm backward → d_h_final
        let mut d_h = vec![0.0f32; s * HIDDEN];
        for t in 0..s {
            norm::backward(
                &c.h_final[t * HIDDEN..(t + 1) * HIDDEN],
                &self.out_gain,
                &d_h_norm[t * HIDDEN..(t + 1) * HIDDEN],
                c.norm_r[t],
                &mut d_h[t * HIDDEN..(t + 1) * HIDDEN],
                &mut d_out_gain);
        }

        // Layers backward in reverse
        let mut layer_grads: Vec<LayerGrads> = Vec::with_capacity(N_LAYERS);
        for l in (0..N_LAYERS).rev() {
            let (new_dh, lg) = self.layers[l].backward(&d_h, &c.layer_caches[l]);
            d_h = new_dh;
            layer_grads.push(lg);
        }
        layer_grads.reverse();

        // Embedding backward: scatter d_h into d_embed by token id
        for t in 0..s {
            let v = c.tokens[t];
            let dh_t = &d_h[t * HIDDEN..(t + 1) * HIDDEN];
            let de   = &mut d_embed[v * HIDDEN..(v + 1) * HIDDEN];
            for i in 0..HIDDEN { de[i] += dh_t[i]; }
        }

        ModelGrads { d_embed, d_out_gain, layer_grads }
    }

    fn adam(m: &Model) -> ModelAdam {
        ModelAdam {
            embed:    Adam::new(VOCAB * HIDDEN),
            out_gain: Adam::new(HIDDEN),
            layers:   m.layers.iter().map(Layer::adam).collect(),
        }
    }
}

impl ModelAdam {
    fn step(&mut self, m: &mut Model, g: &ModelGrads, lr: f32) {
        self.embed.step(&mut m.embed, &g.d_embed, lr);
        self.out_gain.step(&mut m.out_gain, &g.d_out_gain, lr);
        for (i, (la, lg)) in self.layers.iter_mut().zip(&g.layer_grads).enumerate() {
            la.wq.step(&mut m.layers[i].wq, &lg.wq_g, lr);
            la.wk.step(&mut m.layers[i].wk, &lg.wk_g, lr);
            la.wv.step(&mut m.layers[i].wv, &lg.wv_g, lr);
            la.wo.step(&mut m.layers[i].wo, &lg.wo_g, lr);
            la.w_up.step(&mut m.layers[i].w_up, &lg.w_up_g, lr);
            la.w_gate.step(&mut m.layers[i].w_gate, &lg.w_gate_g, lr);
            la.w_down.step(&mut m.layers[i].w_down, &lg.w_down_g, lr);
            la.attn_gain.step(&mut m.layers[i].attn_gain, &lg.d_attn_gain, lr);
            la.ffn_gain.step(&mut m.layers[i].ffn_gain, &lg.d_ffn_gain, lr);
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient clipping (global L2 norm)
// ---------------------------------------------------------------------------
fn clip_grads(g: &mut ModelGrads, max_norm: f32) -> f32 {
    let mut sq = 0.0f32;
    let mut sum_sq = |v: &[f32]| { for x in v { sq += x * x; } };
    sum_sq(&g.d_embed);
    sum_sq(&g.d_out_gain);
    for lg in &g.layer_grads {
        for pg in [&lg.wq_g, &lg.wk_g, &lg.wv_g, &lg.wo_g,
                   &lg.w_up_g, &lg.w_gate_g, &lg.w_down_g] {
            sum_sq(&pg.dd1); sum_sq(&pg.dd2);
            sum_sq(&pg.da1); sum_sq(&pg.da2);
        }
        sum_sq(&lg.d_attn_gain);
        sum_sq(&lg.d_ffn_gain);
    }
    let norm = sq.sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        let scale_v = |v: &mut Vec<f32>| { for x in v { *x *= scale; } };
        scale_v(&mut g.d_embed);
        scale_v(&mut g.d_out_gain);
        for lg in &mut g.layer_grads {
            for pg in [&mut lg.wq_g, &mut lg.wk_g, &mut lg.wv_g, &mut lg.wo_g,
                       &mut lg.w_up_g, &mut lg.w_gate_g, &mut lg.w_down_g] {
                scale_v(&mut pg.dd1); scale_v(&mut pg.dd2);
                scale_v(&mut pg.da1); scale_v(&mut pg.da2);
            }
            scale_v(&mut lg.d_attn_gain);
            scale_v(&mut lg.d_ffn_gain);
        }
    }
    norm
}

// ---------------------------------------------------------------------------
// LR schedule: linear warmup → cosine decay
// ---------------------------------------------------------------------------
fn schedule_lr(opt_step: usize, n_opt_steps: usize, lr_max: f32) -> f32 {
    if opt_step < LR_WARMUP {
        lr_max * opt_step as f32 / LR_WARMUP as f32
    } else {
        let t = (opt_step - LR_WARMUP) as f32 / (n_opt_steps - LR_WARMUP).max(1) as f32;
        lr_max * 0.5 * (1.0 + (std::f32::consts::PI * t).cos())
    }
}

// ---------------------------------------------------------------------------
// Gradient accumulation helpers
// ---------------------------------------------------------------------------
fn zero_model_grads(model: &Model) -> ModelGrads {
    ModelGrads {
        d_embed:    vec![0.0f32; VOCAB * HIDDEN],
        d_out_gain: vec![0.0f32; HIDDEN],
        layer_grads: model.layers.iter().map(|l| LayerGrads {
            wq_g: zero_grads(&l.wq), wk_g: zero_grads(&l.wk),
            wv_g: zero_grads(&l.wv), wo_g: zero_grads(&l.wo),
            w_up_g: zero_grads(&l.w_up), w_gate_g: zero_grads(&l.w_gate),
            w_down_g: zero_grads(&l.w_down),
            d_attn_gain: vec![0.0f32; HIDDEN],
            d_ffn_gain:  vec![0.0f32; HIDDEN],
        }).collect(),
    }
}

fn reset_model_grads(g: &mut ModelGrads) {
    fn zv(v: &mut Vec<f32>) { v.fill(0.0); }
    fn zg(g: &mut Grads) { zv(&mut g.dd1); zv(&mut g.dd2); zv(&mut g.da1); zv(&mut g.da2); }
    zv(&mut g.d_embed); zv(&mut g.d_out_gain);
    for lg in &mut g.layer_grads {
        zg(&mut lg.wq_g); zg(&mut lg.wk_g); zg(&mut lg.wv_g); zg(&mut lg.wo_g);
        zg(&mut lg.w_up_g); zg(&mut lg.w_gate_g); zg(&mut lg.w_down_g);
        zv(&mut lg.d_attn_gain); zv(&mut lg.d_ffn_gain);
    }
}

fn add_model_grads(dst: &mut ModelGrads, src: &ModelGrads) {
    fn av(a: &mut Vec<f32>, b: &[f32]) { a.iter_mut().zip(b).for_each(|(x, y)| *x += y); }
    fn ag(a: &mut Grads, b: &Grads) {
        av(&mut a.dd1, &b.dd1); av(&mut a.dd2, &b.dd2);
        av(&mut a.da1, &b.da1); av(&mut a.da2, &b.da2);
    }
    av(&mut dst.d_embed, &src.d_embed);
    av(&mut dst.d_out_gain, &src.d_out_gain);
    for (dl, sl) in dst.layer_grads.iter_mut().zip(&src.layer_grads) {
        ag(&mut dl.wq_g, &sl.wq_g); ag(&mut dl.wk_g, &sl.wk_g);
        ag(&mut dl.wv_g, &sl.wv_g); ag(&mut dl.wo_g, &sl.wo_g);
        ag(&mut dl.w_up_g, &sl.w_up_g); ag(&mut dl.w_gate_g, &sl.w_gate_g);
        ag(&mut dl.w_down_g, &sl.w_down_g);
        av(&mut dl.d_attn_gain, &sl.d_attn_gain);
        av(&mut dl.d_ffn_gain, &sl.d_ffn_gain);
    }
}

fn scale_model_grads(g: &mut ModelGrads, s: f32) {
    fn sv(v: &mut Vec<f32>, s: f32) { for x in v { *x *= s; } }
    fn sg(g: &mut Grads, s: f32) {
        sv(&mut g.dd1, s); sv(&mut g.dd2, s); sv(&mut g.da1, s); sv(&mut g.da2, s);
    }
    sv(&mut g.d_embed, s); sv(&mut g.d_out_gain, s);
    for lg in &mut g.layer_grads {
        sg(&mut lg.wq_g, s); sg(&mut lg.wk_g, s); sg(&mut lg.wv_g, s); sg(&mut lg.wo_g, s);
        sg(&mut lg.w_up_g, s); sg(&mut lg.w_gate_g, s); sg(&mut lg.w_down_g, s);
        sv(&mut lg.d_attn_gain, s); sv(&mut lg.d_ffn_gain, s);
    }
}

// ---------------------------------------------------------------------------
// Checkpoint: write/read raw f32 slices
// ---------------------------------------------------------------------------
fn save_checkpoint(path: &str, m: &Model, step: usize) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();
    let step_bytes = (step as u64).to_le_bytes();
    f.write_all(&step_bytes).unwrap();
    let write_vec = |f: &mut std::fs::File, v: &Vec<f32>| {
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        f.write_all(bytes).unwrap();
    };
    write_vec(&mut f, &m.embed);
    write_vec(&mut f, &m.out_gain);
    for l in &m.layers {
        for v in [&l.wq.d1, &l.wq.d2, &l.wq.a1, &l.wq.a2,
                  &l.wk.d1, &l.wk.d2, &l.wk.a1, &l.wk.a2,
                  &l.wv.d1, &l.wv.d2, &l.wv.a1, &l.wv.a2,
                  &l.wo.d1, &l.wo.d2, &l.wo.a1, &l.wo.a2,
                  &l.w_up.d1, &l.w_up.d2, &l.w_up.a1, &l.w_up.a2,
                  &l.w_gate.d1, &l.w_gate.d2, &l.w_gate.a1, &l.w_gate.a2,
                  &l.w_down.d1, &l.w_down.d2, &l.w_down.a1, &l.w_down.a2,
                  &l.attn_gain, &l.ffn_gain] {
            write_vec(&mut f, v);
        }
    }
    eprintln!("  checkpoint saved → {path}  (step {step})");
}

fn load_checkpoint(path: &str, m: &mut Model) -> usize {
    use std::io::Read as IoRead;
    let mut f = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
    let mut step_bytes = [0u8; 8];
    f.read_exact(&mut step_bytes).unwrap();
    let step = u64::from_le_bytes(step_bytes) as usize;
    let read_vec = |f: &mut std::fs::File, v: &mut Vec<f32>| {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4)
        };
        f.read_exact(bytes).unwrap();
    };
    read_vec(&mut f, &mut m.embed);
    read_vec(&mut f, &mut m.out_gain);
    for l in &mut m.layers {
        for v in [&mut l.wq.d1, &mut l.wq.d2, &mut l.wq.a1, &mut l.wq.a2,
                  &mut l.wk.d1, &mut l.wk.d2, &mut l.wk.a1, &mut l.wk.a2,
                  &mut l.wv.d1, &mut l.wv.d2, &mut l.wv.a1, &mut l.wv.a2,
                  &mut l.wo.d1, &mut l.wo.d2, &mut l.wo.a1, &mut l.wo.a2,
                  &mut l.w_up.d1, &mut l.w_up.d2, &mut l.w_up.a1, &mut l.w_up.a2,
                  &mut l.w_gate.d1, &mut l.w_gate.d2, &mut l.w_gate.a1, &mut l.w_gate.a2,
                  &mut l.w_down.d1, &mut l.w_down.d2, &mut l.w_down.a1, &mut l.w_down.a2,
                  &mut l.attn_gain, &mut l.ffn_gain] {
            read_vec(&mut f, v);
        }
    }
    step
}

// Average cross-entropy over n_windows forward-only passes with a fixed seed.
// 500 windows = 64K tokens → σ ≈ 0.013 nats.
fn eval_loss(model: &Model, dataset: &Dataset, n_windows: usize) -> f32 {
    let mut rng = 0xEEEE_1234_5678_9ABCu64;
    let mut total = 0.0f32;
    for _ in 0..n_windows {
        let (inp, tgt) = dataset.sample(&mut rng);
        let (logits, _) = model.forward(&inp);
        let (loss, _) = cross_entropy(&logits, &tgt);
        total += loss;
    }
    total / n_windows as f32
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
fn main() {
    if std::env::var("PROFILE").ok().as_deref() == Some("1") {
        enable_profiling();
    }
    let data_path = std::env::args().nth(1)
        .unwrap_or_else(|| "data/input.txt".to_string());
    let ckpt_path = "checkpoint.bin";

    if std::env::args().nth(2).as_deref() == Some("--eval") {
        let dataset = Dataset::load(&data_path);
        let mut model = Model::new(0xFEED_BEEF_1234_5678);
        let step = load_checkpoint(ckpt_path, &mut model);
        eprintln!("checkpoint: step {step}");
        let n_eval = 500;
        eprintln!("running eval over {n_eval} windows ({} tokens) …",
            n_eval * SEQ_LEN);
        let t0 = Instant::now();
        let loss = eval_loss(&model, &dataset, n_eval);
        eprintln!("eval loss: {loss:.4}  ({:.1}s)", t0.elapsed().as_secs_f32());
        return;
    }

    // n_steps = number of optimizer steps (each does ACCUM_STEPS forward passes)
    let n_opt_steps: usize = std::env::args().nth(2)
        .and_then(|s| s.parse().ok()).unwrap_or(3000);
    let lr_max: f32 = 3e-4;
    let max_grad_norm: f32 = 1.0;

    let dataset = Dataset::load(&data_path);
    let mut model = Model::new(0xFEED_BEEF_1234_5678);
    let mut opt   = Model::adam(&model);
    let mut rng   = 0xDEAD_BEEF_u64;

    eprintln!("Fydel char-LM  hidden={HIDDEN}  ffn={FFN_DIM}  layers={N_LAYERS}  \
               heads={N_HEADS}  seq={SEQ_LEN}  vocab={VOCAB}");
    eprintln!("Params: {}K   data: {} tokens",
        model.param_count() / 1000, dataset.data.len());
    eprintln!("Opt-steps: {n_opt_steps}  accum={ACCUM_STEPS}  \
               lr_max={lr_max}  warmup={LR_WARMUP}  grad_clip={max_grad_norm}");
    eprintln!("─────────────────────────────────────────────────");

    let mut accum      = zero_model_grads(&model);
    let mut accum_loss = 0.0f32;
    let t0 = Instant::now();

    for opt_step in 0..n_opt_steps {
        // ACCUM_STEPS forward+backward passes, sum gradients
        for _ in 0..ACCUM_STEPS {
            let (inp, tgt) = dataset.sample(&mut rng);
            let (logits, cache) = model.forward(&inp);
            let (loss, dlogits) = cross_entropy(&logits, &tgt);
            let grads = model.backward(&cache, &dlogits);
            add_model_grads(&mut accum, &grads);
            accum_loss += loss;
        }

        // Average, clip, step
        scale_model_grads(&mut accum, 1.0 / ACCUM_STEPS as f32);
        let gnorm = clip_grads(&mut accum, max_grad_norm);
        let lr = schedule_lr(opt_step, n_opt_steps, lr_max);
        opt.step(&mut model, &accum, lr);

        let avg_loss = accum_loss / ACCUM_STEPS as f32;
        reset_model_grads(&mut accum);
        accum_loss = 0.0;

        if opt_step % 100 == 0 || opt_step < 10 {
            let elapsed = t0.elapsed().as_secs_f32();
            let fwd_done = (opt_step + 1) * ACCUM_STEPS;
            let ms_per_opt = if opt_step > 0 {
                elapsed / opt_step as f32 * 1000.0
            } else { 0.0 };
            eprintln!("step {:>5}  loss={:.4}  gnorm={:.3}  lr={:.2e}  {:.0}ms/step",
                opt_step, avg_loss, gnorm, lr, ms_per_opt);
            let _ = fwd_done;
        }
        if opt_step > 0 && opt_step % 500 == 0 {
            save_checkpoint(ckpt_path, &model, opt_step);
        }
    }

    let total = t0.elapsed().as_secs_f32();
    eprintln!("─────────────────────────────────────────────────");
    eprintln!("done  {n_opt_steps} opt-steps  {:.1}s  ({:.0}ms/opt-step)",
        total, total / n_opt_steps as f32 * 1000.0);
    save_checkpoint(ckpt_path, &model, n_opt_steps);
    if std::env::var("PROFILE").ok().as_deref() == Some("1") {
        print_profile_summary();
    }
}
