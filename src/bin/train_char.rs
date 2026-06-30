//! Character-level language model — step-1 proof of concept.
//! Validates that SharedMonarchMatmul can learn before scaling to 1B.
//!
//! Change the four CONFIG constants to run at full Fydel-1B scale.

use std::time::Instant;
use fydel::kernels::monarch::{SharedMonarchMatmul, FwdCache, Grads};
use fydel::kernels::norm;

// ---------------------------------------------------------------------------
// Config — the only lines you need to change for scale experiments
// ---------------------------------------------------------------------------
const HIDDEN:   usize = 256;   // 896 for full scale  (must be multiple of 64)
const FFN_DIM:  usize = 1024;  // 3072 for full scale (must be multiple of 64)
const N_HEADS:  usize = 4;     // HIDDEN / 64         (14 for full scale)
const N_LAYERS: usize = 2;     // 96 for full scale

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
    SharedMonarchMatmul::new(out_dim / B, in_dim / B, M, ND, seed)
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
    q_fc: Vec<FwdCache>, k_fc: Vec<FwdCache>, v_fc: Vec<FwdCache>,
    probs:    Vec<f32>,           // [H*S*S]
    attn_out: Vec<f32>,           // [S*H]
    o_fc:     Vec<FwdCache>,
    h_mid:    Vec<f32>,           // [S*H] after attn residual
    h_ffn:    Vec<f32>,           // [S*H] after ffn norm
    ffn_r:    Vec<f32>,           // [S]
    up: Vec<f32>, gate: Vec<f32>, // each [S*F]
    up_fc: Vec<FwdCache>, gate_fc: Vec<FwdCache>,
    act:      Vec<f32>,           // [S*F] SwiGLU output
    down_fc:  Vec<FwdCache>,
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
        let mut q = vec![0.0f32; s * HIDDEN];
        let mut k = vec![0.0f32; s * HIDDEN];
        let mut v = vec![0.0f32; s * HIDDEN];
        let mut q_fc = Vec::with_capacity(s);
        let mut k_fc = Vec::with_capacity(s);
        let mut v_fc = Vec::with_capacity(s);

        for t in 0..s {
            let xt = &x[t * HIDDEN..(t + 1) * HIDDEN];
            let hat = &mut h_attn[t * HIDDEN..(t + 1) * HIDDEN];
            attn_r[t] = norm::forward(xt, &self.attn_gain, 1e-5, hat);
            let (qt, qc) = self.wq.forward(hat);
            let (kt, kc) = self.wk.forward(hat);
            let (vt, vc) = self.wv.forward(hat);
            q[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&qt);
            k[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&kt);
            v[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&vt);
            q_fc.push(qc); k_fc.push(kc); v_fc.push(vc);
        }

        let (attn_out, probs) = attn_forward(&q, &k, &v);

        let mut h_mid = vec![0.0f32; s * HIDDEN];
        let mut o_fc  = Vec::with_capacity(s);
        for t in 0..s {
            let ao = &attn_out[t * HIDDEN..(t + 1) * HIDDEN];
            let (ot, oc) = self.wo.forward(ao);
            for i in 0..HIDDEN { h_mid[t * HIDDEN + i] = x[t * HIDDEN + i] + ot[i]; }
            o_fc.push(oc);
        }

        let mut h_ffn   = vec![0.0f32; s * HIDDEN];
        let mut ffn_r   = vec![0.0f32; s];
        let mut up      = vec![0.0f32; s * FFN_DIM];
        let mut gate    = vec![0.0f32; s * FFN_DIM];
        let mut act     = vec![0.0f32; s * FFN_DIM];
        let mut up_fc   = Vec::with_capacity(s);
        let mut gate_fc = Vec::with_capacity(s);
        let mut down_fc = Vec::with_capacity(s);
        let mut out = vec![0.0f32; s * HIDDEN];

        for t in 0..s {
            let hm = &h_mid[t * HIDDEN..(t + 1) * HIDDEN];
            let hf = &mut h_ffn[t * HIDDEN..(t + 1) * HIDDEN];
            ffn_r[t] = norm::forward(hm, &self.ffn_gain, 1e-5, hf);
            let (up_t, uc)   = self.w_up.forward(hf);
            let (gate_t, gc) = self.w_gate.forward(hf);
            let act_t = &mut act[t * FFN_DIM..(t + 1) * FFN_DIM];
            for i in 0..FFN_DIM {
                let g = gate_t[i];
                act_t[i] = up_t[i] * g / (1.0 + (-g).exp()); // SwiGLU
            }
            let (down_t, dc) = self.w_down.forward(act_t);
            for i in 0..HIDDEN { out[t * HIDDEN + i] = h_mid[t * HIDDEN + i] + down_t[i]; }
            up[t * FFN_DIM..(t + 1) * FFN_DIM].copy_from_slice(&up_t);
            gate[t * FFN_DIM..(t + 1) * FFN_DIM].copy_from_slice(&gate_t);
            up_fc.push(uc); gate_fc.push(gc); down_fc.push(dc);
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
        for t in 0..s {
            let dout_t  = &dout[t * HIDDEN..(t + 1) * HIDDEN];
            // residual into d_h_mid
            for i in 0..HIDDEN { d_h_mid[t * HIDDEN + i] += dout_t[i]; }

            // w_down backward → d_act[t]
            let mut d_act_t = vec![0.0f32; FFN_DIM];
            let g = self.w_down.backward(
                &c.act[t * FFN_DIM..(t + 1) * FFN_DIM],
                &c.down_fc[t], dout_t, &mut d_act_t);
            acc_grads(&mut w_down_g, g);

            // SwiGLU backward
            let up_t   = &c.up[t * FFN_DIM..(t + 1) * FFN_DIM];
            let gate_t = &c.gate[t * FFN_DIM..(t + 1) * FFN_DIM];
            let mut d_up_t   = vec![0.0f32; FFN_DIM];
            let mut d_gate_t = vec![0.0f32; FFN_DIM];
            for i in 0..FFN_DIM {
                let g = gate_t[i];
                let sig = 1.0 / (1.0 + (-g).exp());
                d_up_t[i]   = d_act_t[i] * g * sig;
                d_gate_t[i] = d_act_t[i] * up_t[i] * sig * (1.0 + g * (1.0 - sig));
            }

            // w_up / w_gate backward → accumulate into h_ffn gradient
            let hf = &c.h_ffn[t * HIDDEN..(t + 1) * HIDDEN];
            let mut dx_up   = vec![0.0f32; HIDDEN];
            let mut dx_gate = vec![0.0f32; HIDDEN];
            let g_up   = self.w_up.backward(hf, &c.up_fc[t],   &d_up_t,   &mut dx_up);
            let g_gate = self.w_gate.backward(hf, &c.gate_fc[t], &d_gate_t, &mut dx_gate);
            acc_grads(&mut w_up_g, g_up);
            acc_grads(&mut w_gate_g, g_gate);

            // RMSNorm backward (ffn) → d_h_mid
            let mut d_hf = vec![0.0f32; HIDDEN];
            for i in 0..HIDDEN { d_hf[i] = dx_up[i] + dx_gate[i]; }
            norm::backward(&c.h_mid[t * HIDDEN..(t + 1) * HIDDEN],
                &self.ffn_gain, &d_hf, c.ffn_r[t],
                &mut d_h_mid[t * HIDDEN..(t + 1) * HIDDEN],
                &mut d_ffn_gain);
        }

        // --- Attention output projection backward ---
        let mut d_attn_out = vec![0.0f32; s * HIDDEN];
        let mut wo_g = zero_grads(&self.wo);
        for t in 0..s {
            let dh_mid = &d_h_mid[t * HIDDEN..(t + 1) * HIDDEN];
            // residual: dx[t] += d_h_mid[t]
            for i in 0..HIDDEN { dx[t * HIDDEN + i] += dh_mid[i]; }
            let g = self.wo.backward(
                &c.attn_out[t * HIDDEN..(t + 1) * HIDDEN],
                &c.o_fc[t], dh_mid,
                &mut d_attn_out[t * HIDDEN..(t + 1) * HIDDEN]);
            acc_grads(&mut wo_g, g);
        }

        // --- Attention QKV backward ---
        let (dq, dk, dv) = attn_backward(&c.q, &c.k, &c.v, &c.probs, &d_attn_out);

        let mut d_h_attn    = vec![0.0f32; s * HIDDEN];
        let mut d_attn_gain = vec![0.0f32; HIDDEN];
        let mut wq_g = zero_grads(&self.wq);
        let mut wk_g = zero_grads(&self.wk);
        let mut wv_g = zero_grads(&self.wv);
        for t in 0..s {
            let hat = &c.h_attn[t * HIDDEN..(t + 1) * HIDDEN];
            let dh  = &mut d_h_attn[t * HIDDEN..(t + 1) * HIDDEN];
            let gq = self.wq.backward(hat, &c.q_fc[t], &dq[t*HIDDEN..(t+1)*HIDDEN], dh);
            let gk = self.wk.backward(hat, &c.k_fc[t], &dk[t*HIDDEN..(t+1)*HIDDEN], dh);
            let gv = self.wv.backward(hat, &c.v_fc[t], &dv[t*HIDDEN..(t+1)*HIDDEN], dh);
            acc_grads(&mut wq_g, gq);
            acc_grads(&mut wk_g, gk);
            acc_grads(&mut wv_g, gv);
            // RMSNorm backward (attn) → dx
            norm::backward(&c.x_in[t * HIDDEN..(t + 1) * HIDDEN],
                &self.attn_gain, dh, c.attn_r[t],
                &mut dx[t * HIDDEN..(t + 1) * HIDDEN],
                &mut d_attn_gain);
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
fn main() {
    let data_path = std::env::args().nth(1)
        .unwrap_or_else(|| "data/input.txt".to_string());
    let ckpt_path = "checkpoint.bin";
    let n_steps: usize = std::env::args().nth(2)
        .and_then(|s| s.parse().ok()).unwrap_or(3000);
    let lr: f32 = 3e-4;
    let max_grad_norm: f32 = 1.0;

    let dataset = Dataset::load(&data_path);
    let mut model = Model::new(0xFEED_BEEF_1234_5678);
    let mut opt   = Model::adam(&model);
    let mut rng   = 0xDEAD_BEEF_u64;

    eprintln!("Fydel char-LM  hidden={HIDDEN}  ffn={FFN_DIM}  layers={N_LAYERS}  \
               heads={N_HEADS}  seq={SEQ_LEN}  vocab={VOCAB}");
    eprintln!("Params: {}K   data: {} tokens",
        model.param_count() / 1000, dataset.data.len());
    eprintln!("Running {n_steps} steps  lr={lr}  grad_clip={max_grad_norm}");
    eprintln!("─────────────────────────────────────────────────");

    let t0 = Instant::now();
    for step in 0..n_steps {
        let (inp, tgt) = dataset.sample(&mut rng);
        let (logits, cache) = model.forward(&inp);
        let (loss, dlogits) = cross_entropy(&logits, &tgt);
        let mut grads = model.backward(&cache, &dlogits);
        let gnorm = clip_grads(&mut grads, max_grad_norm);
        opt.step(&mut model, &grads, lr);

        if step % 100 == 0 || step < 10 {
            let elapsed = t0.elapsed().as_secs_f32();
            let ms_per_step = if step > 0 { elapsed / step as f32 * 1000.0 } else { 0.0 };
            eprintln!("step {:>5}  loss={:.4}  gnorm={:.3}  {:.0}ms/step",
                step, loss, gnorm, ms_per_step);
        }
        if step > 0 && step % 500 == 0 {
            save_checkpoint(ckpt_path, &model, step);
        }
    }

    let total = t0.elapsed().as_secs_f32();
    eprintln!("─────────────────────────────────────────────────");
    eprintln!("done  {n_steps} steps  {:.1}s  ({:.0}ms/step)",
        total, total / n_steps as f32 * 1000.0);
    save_checkpoint(ckpt_path, &model, n_steps);
}
