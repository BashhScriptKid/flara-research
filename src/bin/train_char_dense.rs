//! Dense linear-layer baseline for comparison with SharedMonarchMatmul.
//!
//! Change the TWO marked lines to switch between comparison configs:
//!   Dense-256: HIDDEN=256, FFN_DIM=1024  → ~7M params  (expressiveness ceiling)
//!   Dense-64:  HIDDEN=64,  FFN_DIM=256   → ~139K params (param-matched to Monarch 113K)
//!
//! Usage:
//!   cargo run --bin train_char_dense --release -- data/input.txt [n_steps]
//!   cargo run --bin train_char_dense --release -- data/input.txt --eval

use std::time::Instant;
use fydel::kernels::norm;
use fydel::kernels::fastmath;

// ---------------------------------------------------------------------------
// Config — change THESE TWO LINES between runs
// ---------------------------------------------------------------------------
const HIDDEN:   usize = 64;    // ← change: 256 or 64
const FFN_DIM:  usize = 256;   // ← change: 1024 or 256

const N_HEADS:  usize = 4;
const N_LAYERS: usize = 2;
const ACCUM_STEPS: usize = 4;
const LR_WARMUP:   usize = 100;
const VOCAB:    usize = 128;
const SEQ_LEN:  usize = 128;
const HEAD_DIM: usize = HIDDEN / N_HEADS;

// ---------------------------------------------------------------------------
// Data (identical to train_char.rs)
// ---------------------------------------------------------------------------
fn encode(text: &str) -> Vec<usize> {
    text.bytes().map(|b| (b as usize).min(VOCAB - 1)).collect()
}

struct Dataset { data: Vec<usize> }
impl Dataset {
    fn load(path: &str) -> Self {
        let text = std::fs::read_to_string(path).unwrap_or_else(|_| {
            eprintln!("warn: {path} not found, using fallback text");
            "To be, or not to be, that is the question.".repeat(2000)
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
// Loss (identical to train_char.rs)
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
// Causal multi-head attention (identical to train_char.rs)
// ---------------------------------------------------------------------------
fn attn_forward(q: &[f32], k: &[f32], v: &[f32]) -> (Vec<f32>, Vec<f32>) {
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

fn attn_backward(q: &[f32], k: &[f32], v: &[f32], probs: &[f32], d_out: &[f32])
    -> (Vec<f32>, Vec<f32>, Vec<f32>)
{
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
            for s in 0..=t {
                let dvsh = &mut dv[s * HIDDEN + hd0..s * HIDDEN + hd0 + HEAD_DIM];
                for i in 0..HEAD_DIM { dvsh[i] += p_row[s] * d_oth[i]; }
            }
            let mut d_probs = [0.0f32; SEQ_LEN];
            for s in 0..=t {
                let vsh = &v[s * HIDDEN + hd0..s * HIDDEN + hd0 + HEAD_DIM];
                d_probs[s] = d_oth.iter().zip(vsh).map(|(a, b)| a * b).sum();
            }
            let dot_pdp: f32 = (0..=t).map(|s| p_row[s] * d_probs[s]).sum();
            let qth = &q[t * HIDDEN + hd0..t * HIDDEN + hd0 + HEAD_DIM];
            let dqth = &mut dq[t * HIDDEN + hd0..t * HIDDEN + hd0 + HEAD_DIM];
            for s in 0..=t {
                let ds = p_row[s] * (d_probs[s] - dot_pdp) * scale;
                let ksh = &k[s * HIDDEN + hd0..s * HIDDEN + hd0 + HEAD_DIM];
                for i in 0..HEAD_DIM { dqth[i] += ds * ksh[i]; }
                let dksh = &mut dk[s * HIDDEN + hd0..s * HIDDEN + hd0 + HEAD_DIM];
                for i in 0..HEAD_DIM { dksh[i] += ds * qth[i]; }
            }
        }
    }
    (dq, dk, dv)
}

// ---------------------------------------------------------------------------
// Dense projection helpers
// ---------------------------------------------------------------------------
fn randn_vec(n: usize, scale: f32, mut seed: u64) -> Vec<f32> {
    (0..n).map(|_| {
        seed = seed.wrapping_mul(6364136223846793005)
                   .wrapping_add(1442695040888963407);
        ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
    }).collect()
}

// Forward: y[o] = Σ_i W[o*in+i] * x[i]
fn proj_fwd(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        y[o] = w[o * in_dim..(o + 1) * in_dim].iter().zip(x).map(|(a, b)| a * b).sum();
    }
    y
}

// Backward: accumulate dW and dx in a single row pass (row-major, cache-friendly).
fn proj_bwd(w: &[f32], x: &[f32], dy: &[f32], dx: &mut [f32], dw: &mut [f32],
            out_dim: usize, in_dim: usize) {
    for o in 0..out_dim {
        let dy_o = dy[o];
        let w_row = &w[o * in_dim..(o + 1) * in_dim];
        let dw_row = &mut dw[o * in_dim..(o + 1) * in_dim];
        for i in 0..in_dim {
            dw_row[i] += dy_o * x[i];
            dx[i] += w_row[i] * dy_o;
        }
    }
}

// ---------------------------------------------------------------------------
// Adam
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

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------
struct Layer {
    wq: Vec<f32>, wk: Vec<f32>, wv: Vec<f32>, wo: Vec<f32>,
    attn_gain: Vec<f32>,
    w_up: Vec<f32>, w_gate: Vec<f32>, w_down: Vec<f32>,
    ffn_gain: Vec<f32>,
}

struct LayerCache {
    x_in:     Vec<f32>,   // [S*H] layer input
    h_attn:   Vec<f32>,   // [S*H] after attn RMSNorm
    attn_r:   Vec<f32>,   // [S] norm reciprocals
    q: Vec<f32>, k: Vec<f32>, v: Vec<f32>,  // [S*H]
    probs:    Vec<f32>,   // [N_HEADS*S*S]
    attn_out: Vec<f32>,   // [S*H]
    h_mid:    Vec<f32>,   // [S*H] after attn residual
    h_ffn:    Vec<f32>,   // [S*H] after FFN RMSNorm
    ffn_r:    Vec<f32>,   // [S]
    up:       Vec<f32>,   // [S*F]
    gate:     Vec<f32>,   // [S*F]
    act:      Vec<f32>,   // [S*F] SwiGLU output
}

struct LayerGrads {
    dw_q: Vec<f32>, dw_k: Vec<f32>, dw_v: Vec<f32>, dw_o: Vec<f32>,
    dw_up: Vec<f32>, dw_gate: Vec<f32>, dw_down: Vec<f32>,
    d_attn_gain: Vec<f32>, d_ffn_gain: Vec<f32>,
}

struct LayerAdam {
    wq: Adam, wk: Adam, wv: Adam, wo: Adam,
    w_up: Adam, w_gate: Adam, w_down: Adam,
    attn_gain: Adam, ffn_gain: Adam,
}

impl Layer {
    fn new(seed: u64) -> Self {
        // Xavier uniform: full range = 2/sqrt(in_dim)
        let hh_scale = 2.0 / (HIDDEN as f32).sqrt();
        let fh_scale = 2.0 / (HIDDEN as f32).sqrt();   // FFN up/gate: in_dim=HIDDEN
        let hf_scale = 2.0 / (FFN_DIM as f32).sqrt();  // FFN down: in_dim=FFN_DIM
        // Residual output projections (wo, w_down) scaled by 1/sqrt(N_LAYERS) per GPT-2
        let res_factor = 1.0 / (N_LAYERS as f32).sqrt();
        Self {
            wq:        randn_vec(HIDDEN * HIDDEN, hh_scale, seed ^ 1),
            wk:        randn_vec(HIDDEN * HIDDEN, hh_scale, seed ^ 2),
            wv:        randn_vec(HIDDEN * HIDDEN, hh_scale, seed ^ 3),
            wo:        randn_vec(HIDDEN * HIDDEN, hh_scale * res_factor, seed ^ 4),
            attn_gain: vec![1.0f32; HIDDEN],
            w_up:      randn_vec(FFN_DIM * HIDDEN, fh_scale, seed ^ 5),
            w_gate:    randn_vec(FFN_DIM * HIDDEN, fh_scale, seed ^ 6),
            w_down:    randn_vec(HIDDEN * FFN_DIM, hf_scale * res_factor, seed ^ 7),
            ffn_gain:  vec![1.0f32; HIDDEN],
        }
    }

    fn adam(_l: &Layer) -> LayerAdam {
        LayerAdam {
            wq:   Adam::new(HIDDEN * HIDDEN), wk: Adam::new(HIDDEN * HIDDEN),
            wv:   Adam::new(HIDDEN * HIDDEN), wo: Adam::new(HIDDEN * HIDDEN),
            w_up: Adam::new(FFN_DIM * HIDDEN), w_gate: Adam::new(FFN_DIM * HIDDEN),
            w_down: Adam::new(HIDDEN * FFN_DIM),
            attn_gain: Adam::new(HIDDEN), ffn_gain: Adam::new(HIDDEN),
        }
    }

    fn param_count(&self) -> usize {
        HIDDEN * HIDDEN * 4 + FFN_DIM * HIDDEN * 2 + HIDDEN * FFN_DIM + HIDDEN * 2
    }

    fn forward(&self, x: &[f32]) -> (Vec<f32>, LayerCache) {
        let s = SEQ_LEN;

        // ---- Attention block ----
        let mut h_attn = vec![0.0f32; s * HIDDEN];
        let mut attn_r = vec![0.0f32; s];
        for t in 0..s {
            attn_r[t] = norm::forward(
                &x[t * HIDDEN..(t + 1) * HIDDEN],
                &self.attn_gain, 1e-5,
                &mut h_attn[t * HIDDEN..(t + 1) * HIDDEN]);
        }

        let mut q = vec![0.0f32; s * HIDDEN];
        let mut k = vec![0.0f32; s * HIDDEN];
        let mut v = vec![0.0f32; s * HIDDEN];
        for t in 0..s {
            let hat = &h_attn[t * HIDDEN..(t + 1) * HIDDEN];
            let qt = proj_fwd(&self.wq, hat, HIDDEN, HIDDEN);
            let kt = proj_fwd(&self.wk, hat, HIDDEN, HIDDEN);
            let vt = proj_fwd(&self.wv, hat, HIDDEN, HIDDEN);
            q[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&qt);
            k[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&kt);
            v[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&vt);
        }

        let (attn_out, probs) = attn_forward(&q, &k, &v);

        let mut h_mid = x.to_vec();
        for t in 0..s {
            let ao = &attn_out[t * HIDDEN..(t + 1) * HIDDEN];
            let ot = proj_fwd(&self.wo, ao, HIDDEN, HIDDEN);
            for i in 0..HIDDEN { h_mid[t * HIDDEN + i] += ot[i]; }
        }

        // ---- FFN block ----
        let mut h_ffn = vec![0.0f32; s * HIDDEN];
        let mut ffn_r = vec![0.0f32; s];
        for t in 0..s {
            ffn_r[t] = norm::forward(
                &h_mid[t * HIDDEN..(t + 1) * HIDDEN],
                &self.ffn_gain, 1e-5,
                &mut h_ffn[t * HIDDEN..(t + 1) * HIDDEN]);
        }

        let mut up   = vec![0.0f32; s * FFN_DIM];
        let mut gate = vec![0.0f32; s * FFN_DIM];
        let mut act  = vec![0.0f32; s * FFN_DIM];
        let mut out  = h_mid.clone();
        for t in 0..s {
            let hft = &h_ffn[t * HIDDEN..(t + 1) * HIDDEN];
            let up_t   = proj_fwd(&self.w_up,   hft, FFN_DIM, HIDDEN);
            let gate_t = proj_fwd(&self.w_gate,  hft, FFN_DIM, HIDDEN);
            let mut act_t = vec![0.0f32; FFN_DIM];
            fastmath::glu_forward(&up_t, &gate_t, &mut act_t);
            let down_t = proj_fwd(&self.w_down, &act_t, HIDDEN, FFN_DIM);
            up[t * FFN_DIM..(t + 1) * FFN_DIM].copy_from_slice(&up_t);
            gate[t * FFN_DIM..(t + 1) * FFN_DIM].copy_from_slice(&gate_t);
            act[t * FFN_DIM..(t + 1) * FFN_DIM].copy_from_slice(&act_t);
            for i in 0..HIDDEN { out[t * HIDDEN + i] += down_t[i]; }
        }

        (out, LayerCache {
            x_in: x.to_vec(), h_attn, attn_r, q, k, v, probs, attn_out,
            h_mid, h_ffn, ffn_r, up, gate, act,
        })
    }

    fn backward(&self, d_out: &[f32], c: &LayerCache) -> (Vec<f32>, LayerGrads) {
        let s = SEQ_LEN;
        let mut dx = vec![0.0f32; s * HIDDEN];

        // ---- FFN backward ----
        let mut d_h_mid = vec![0.0f32; s * HIDDEN];
        let mut dw_down = vec![0.0f32; HIDDEN * FFN_DIM];
        let mut dw_up   = vec![0.0f32; FFN_DIM * HIDDEN];
        let mut dw_gate = vec![0.0f32; FFN_DIM * HIDDEN];
        let mut d_ffn_gain = vec![0.0f32; HIDDEN];

        for t in 0..s {
            let d_out_t = &d_out[t * HIDDEN..(t + 1) * HIDDEN];
            // FFN residual skip
            for i in 0..HIDDEN { d_h_mid[t * HIDDEN + i] += d_out_t[i]; }

            // w_down backward
            let act_t = &c.act[t * FFN_DIM..(t + 1) * FFN_DIM];
            let mut d_act = vec![0.0f32; FFN_DIM];
            proj_bwd(&self.w_down, act_t, d_out_t, &mut d_act, &mut dw_down,
                     HIDDEN, FFN_DIM);

            // SwiGLU backward: act = up * sigmoid(gate)
            let up_t   = &c.up[t * FFN_DIM..(t + 1) * FFN_DIM];
            let gate_t = &c.gate[t * FFN_DIM..(t + 1) * FFN_DIM];
            let mut d_up   = vec![0.0f32; FFN_DIM];
            let mut d_gate = vec![0.0f32; FFN_DIM];
            fastmath::glu_backward(up_t, gate_t, &d_act, &mut d_up, &mut d_gate);

            // w_up and w_gate backward (both read h_ffn)
            let h_ffn_t = &c.h_ffn[t * HIDDEN..(t + 1) * HIDDEN];
            let mut d_h_ffn = vec![0.0f32; HIDDEN];
            proj_bwd(&self.w_up,   h_ffn_t, &d_up,   &mut d_h_ffn, &mut dw_up,
                     FFN_DIM, HIDDEN);
            proj_bwd(&self.w_gate, h_ffn_t, &d_gate, &mut d_h_ffn, &mut dw_gate,
                     FFN_DIM, HIDDEN);

            // RMSNorm backward (FFN)
            let h_mid_t = &c.h_mid[t * HIDDEN..(t + 1) * HIDDEN];
            norm::backward(h_mid_t, &self.ffn_gain, &d_h_ffn, c.ffn_r[t],
                &mut d_h_mid[t * HIDDEN..(t + 1) * HIDDEN],
                &mut d_ffn_gain);
        }

        // ---- Attention backward ----
        let mut dw_q = vec![0.0f32; HIDDEN * HIDDEN];
        let mut dw_k = vec![0.0f32; HIDDEN * HIDDEN];
        let mut dw_v = vec![0.0f32; HIDDEN * HIDDEN];
        let mut dw_o = vec![0.0f32; HIDDEN * HIDDEN];
        let mut d_attn_gain = vec![0.0f32; HIDDEN];

        // wo backward → d_attn_out; attn residual skip → dx
        let mut d_attn_out = vec![0.0f32; s * HIDDEN];
        for t in 0..s {
            let d_hm = &d_h_mid[t * HIDDEN..(t + 1) * HIDDEN];
            for i in 0..HIDDEN { dx[t * HIDDEN + i] += d_hm[i]; }  // attn residual skip
            let ao = &c.attn_out[t * HIDDEN..(t + 1) * HIDDEN];
            proj_bwd(&self.wo, ao, d_hm, &mut d_attn_out[t * HIDDEN..(t + 1) * HIDDEN],
                     &mut dw_o, HIDDEN, HIDDEN);
        }

        let (dq, dk, dv) = attn_backward(&c.q, &c.k, &c.v, &c.probs, &d_attn_out);

        // wq/wk/wv backward → d_h_attn → RMSNorm backward → dx
        for t in 0..s {
            let hat = &c.h_attn[t * HIDDEN..(t + 1) * HIDDEN];
            let mut d_hat = vec![0.0f32; HIDDEN];
            proj_bwd(&self.wq, hat, &dq[t*HIDDEN..(t+1)*HIDDEN], &mut d_hat, &mut dw_q,
                     HIDDEN, HIDDEN);
            proj_bwd(&self.wk, hat, &dk[t*HIDDEN..(t+1)*HIDDEN], &mut d_hat, &mut dw_k,
                     HIDDEN, HIDDEN);
            proj_bwd(&self.wv, hat, &dv[t*HIDDEN..(t+1)*HIDDEN], &mut d_hat, &mut dw_v,
                     HIDDEN, HIDDEN);
            let x_t = &c.x_in[t * HIDDEN..(t + 1) * HIDDEN];
            norm::backward(x_t, &self.attn_gain, &d_hat, c.attn_r[t],
                &mut dx[t * HIDDEN..(t + 1) * HIDDEN],
                &mut d_attn_gain);
        }

        (dx, LayerGrads {
            dw_q, dw_k, dw_v, dw_o, dw_up, dw_gate, dw_down,
            d_attn_gain, d_ffn_gain,
        })
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------
struct Model { embed: Vec<f32>, layers: Vec<Layer>, out_gain: Vec<f32> }

struct ModelCache {
    tokens:       Vec<usize>,
    layer_caches: Vec<LayerCache>,
    h_final:      Vec<f32>,
    h_norm:       Vec<f32>,
    norm_r:       Vec<f32>,
}

struct ModelGrads {
    d_embed:     Vec<f32>,
    d_out_gain:  Vec<f32>,
    layer_grads: Vec<LayerGrads>,
}

struct ModelAdam {
    embed:    Adam,
    out_gain: Adam,
    layers:   Vec<LayerAdam>,
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
        VOCAB * HIDDEN + HIDDEN + N_LAYERS * self.layers[0].param_count()
    }

    fn forward(&self, tokens: &[usize]) -> (Vec<f32>, ModelCache) {
        let s = SEQ_LEN;
        let mut h: Vec<f32> = tokens.iter().flat_map(|&t| {
            self.embed[t * HIDDEN..(t + 1) * HIDDEN].iter().copied()
        }).collect();
        let mut layer_caches = Vec::with_capacity(N_LAYERS);
        for layer in &self.layers {
            let (nh, lc) = layer.forward(&h);
            h = nh;
            layer_caches.push(lc);
        }
        let h_final = h.clone();
        let mut h_norm = vec![0.0f32; s * HIDDEN];
        let mut norm_r = vec![0.0f32; s];
        for t in 0..s {
            norm_r[t] = norm::forward(
                &h[t * HIDDEN..(t + 1) * HIDDEN],
                &self.out_gain, 1e-5,
                &mut h_norm[t * HIDDEN..(t + 1) * HIDDEN]);
        }
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
        let mut d_h_norm   = vec![0.0f32; s * HIDDEN];
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
        let mut layer_grads: Vec<LayerGrads> = Vec::with_capacity(N_LAYERS);
        for l in (0..N_LAYERS).rev() {
            let (new_dh, lg) = self.layers[l].backward(&d_h, &c.layer_caches[l]);
            d_h = new_dh;
            layer_grads.push(lg);
        }
        layer_grads.reverse();
        for t in 0..s {
            let v = c.tokens[t];
            let dh_t = &d_h[t * HIDDEN..(t + 1) * HIDDEN];
            let de   = &mut d_embed[v * HIDDEN..(v + 1) * HIDDEN];
            for i in 0..HIDDEN { de[i] += dh_t[i]; }
        }
        ModelGrads { d_embed, d_out_gain, layer_grads }
    }
}

impl ModelAdam {
    fn new(m: &Model) -> Self {
        Self {
            embed:    Adam::new(VOCAB * HIDDEN),
            out_gain: Adam::new(HIDDEN),
            layers:   m.layers.iter().map(Layer::adam).collect(),
        }
    }

    fn step(&mut self, m: &mut Model, g: &ModelGrads, lr: f32) {
        self.embed.step(&mut m.embed, &g.d_embed, lr);
        self.out_gain.step(&mut m.out_gain, &g.d_out_gain, lr);
        for (i, (la, lg)) in self.layers.iter_mut().zip(&g.layer_grads).enumerate() {
            la.wq.step(&mut m.layers[i].wq, &lg.dw_q, lr);
            la.wk.step(&mut m.layers[i].wk, &lg.dw_k, lr);
            la.wv.step(&mut m.layers[i].wv, &lg.dw_v, lr);
            la.wo.step(&mut m.layers[i].wo, &lg.dw_o, lr);
            la.w_up.step(&mut m.layers[i].w_up, &lg.dw_up, lr);
            la.w_gate.step(&mut m.layers[i].w_gate, &lg.dw_gate, lr);
            la.w_down.step(&mut m.layers[i].w_down, &lg.dw_down, lr);
            la.attn_gain.step(&mut m.layers[i].attn_gain, &lg.d_attn_gain, lr);
            la.ffn_gain.step(&mut m.layers[i].ffn_gain, &lg.d_ffn_gain, lr);
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient helpers
// ---------------------------------------------------------------------------
fn zero_grads(m: &Model) -> ModelGrads {
    ModelGrads {
        d_embed:    vec![0.0f32; VOCAB * HIDDEN],
        d_out_gain: vec![0.0f32; HIDDEN],
        layer_grads: m.layers.iter().map(|_| LayerGrads {
            dw_q:  vec![0.0f32; HIDDEN * HIDDEN], dw_k: vec![0.0f32; HIDDEN * HIDDEN],
            dw_v:  vec![0.0f32; HIDDEN * HIDDEN], dw_o: vec![0.0f32; HIDDEN * HIDDEN],
            dw_up:   vec![0.0f32; FFN_DIM * HIDDEN],
            dw_gate: vec![0.0f32; FFN_DIM * HIDDEN],
            dw_down: vec![0.0f32; HIDDEN * FFN_DIM],
            d_attn_gain: vec![0.0f32; HIDDEN],
            d_ffn_gain:  vec![0.0f32; HIDDEN],
        }).collect(),
    }
}

fn reset_grads(g: &mut ModelGrads) {
    g.d_embed.fill(0.0); g.d_out_gain.fill(0.0);
    for lg in &mut g.layer_grads {
        lg.dw_q.fill(0.0); lg.dw_k.fill(0.0); lg.dw_v.fill(0.0); lg.dw_o.fill(0.0);
        lg.dw_up.fill(0.0); lg.dw_gate.fill(0.0); lg.dw_down.fill(0.0);
        lg.d_attn_gain.fill(0.0); lg.d_ffn_gain.fill(0.0);
    }
}

fn add_grads(dst: &mut ModelGrads, src: &ModelGrads) {
    let av = |a: &mut Vec<f32>, b: &[f32]| a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
    av(&mut dst.d_embed, &src.d_embed);
    av(&mut dst.d_out_gain, &src.d_out_gain);
    for (dl, sl) in dst.layer_grads.iter_mut().zip(&src.layer_grads) {
        av(&mut dl.dw_q, &sl.dw_q); av(&mut dl.dw_k, &sl.dw_k);
        av(&mut dl.dw_v, &sl.dw_v); av(&mut dl.dw_o, &sl.dw_o);
        av(&mut dl.dw_up, &sl.dw_up); av(&mut dl.dw_gate, &sl.dw_gate);
        av(&mut dl.dw_down, &sl.dw_down);
        av(&mut dl.d_attn_gain, &sl.d_attn_gain);
        av(&mut dl.d_ffn_gain, &sl.d_ffn_gain);
    }
}

fn scale_grads(g: &mut ModelGrads, s: f32) {
    let sv = |v: &mut Vec<f32>| v.iter_mut().for_each(|x| *x *= s);
    sv(&mut g.d_embed); sv(&mut g.d_out_gain);
    for lg in &mut g.layer_grads {
        sv(&mut lg.dw_q); sv(&mut lg.dw_k); sv(&mut lg.dw_v); sv(&mut lg.dw_o);
        sv(&mut lg.dw_up); sv(&mut lg.dw_gate); sv(&mut lg.dw_down);
        sv(&mut lg.d_attn_gain); sv(&mut lg.d_ffn_gain);
    }
}

fn clip_grads(g: &mut ModelGrads, max_norm: f32) -> f32 {
    let mut sq = 0.0f32;
    let ssq = |v: &[f32], sq: &mut f32| v.iter().for_each(|x| *sq += x * x);
    ssq(&g.d_embed, &mut sq); ssq(&g.d_out_gain, &mut sq);
    for lg in &g.layer_grads {
        for v in [&lg.dw_q, &lg.dw_k, &lg.dw_v, &lg.dw_o,
                  &lg.dw_up, &lg.dw_gate, &lg.dw_down,
                  &lg.d_attn_gain, &lg.d_ffn_gain] { ssq(v, &mut sq); }
    }
    let norm = sq.sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        let sv = |v: &mut Vec<f32>| v.iter_mut().for_each(|x| *x *= scale);
        sv(&mut g.d_embed); sv(&mut g.d_out_gain);
        for lg in &mut g.layer_grads {
            for v in [&mut lg.dw_q, &mut lg.dw_k, &mut lg.dw_v, &mut lg.dw_o,
                      &mut lg.dw_up, &mut lg.dw_gate, &mut lg.dw_down,
                      &mut lg.d_attn_gain, &mut lg.d_ffn_gain] { sv(v); }
        }
    }
    norm
}

// ---------------------------------------------------------------------------
// LR schedule (identical to train_char.rs)
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
// Checkpoint
// ---------------------------------------------------------------------------
fn save_checkpoint(path: &str, m: &Model, step: usize) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(step as u64).to_le_bytes()).unwrap();
    let wv = |f: &mut std::fs::File, v: &Vec<f32>| {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
        };
        f.write_all(bytes).unwrap();
    };
    wv(&mut f, &m.embed);
    wv(&mut f, &m.out_gain);
    for l in &m.layers {
        for v in [&l.wq, &l.wk, &l.wv, &l.wo,
                  &l.w_up, &l.w_gate, &l.w_down,
                  &l.attn_gain, &l.ffn_gain] { wv(&mut f, v); }
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
    let rv = |f: &mut std::fs::File, v: &mut Vec<f32>| {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4)
        };
        f.read_exact(bytes).unwrap();
    };
    rv(&mut f, &mut m.embed);
    rv(&mut f, &mut m.out_gain);
    for l in &mut m.layers {
        for v in [&mut l.wq, &mut l.wk, &mut l.wv, &mut l.wo,
                  &mut l.w_up, &mut l.w_gate, &mut l.w_down,
                  &mut l.attn_gain, &mut l.ffn_gain] { rv(&mut f, v); }
    }
    step
}

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
    let data_path = std::env::args().nth(1)
        .unwrap_or_else(|| "data/input.txt".to_string());
    let ckpt_path = "checkpoint_dense64.bin";

    if std::env::args().nth(2).as_deref() == Some("--eval") {
        let dataset = Dataset::load(&data_path);
        let mut model = Model::new(0xFEED_BEEF_1234_5678);
        let step = load_checkpoint(ckpt_path, &mut model);
        eprintln!("checkpoint: step {step}");
        let n_eval = 500;
        eprintln!("running eval over {n_eval} windows ({} tokens) …", n_eval * SEQ_LEN);
        let t0 = Instant::now();
        let loss = eval_loss(&model, &dataset, n_eval);
        eprintln!("eval loss: {loss:.4}  ({:.1}s)", t0.elapsed().as_secs_f32());
        return;
    }

    let n_opt_steps: usize = std::env::args().nth(2)
        .and_then(|s| s.parse().ok()).unwrap_or(3000);
    let lr_max: f32 = 3e-4;
    let max_grad_norm: f32 = 1.0;

    let dataset = Dataset::load(&data_path);
    let mut model = Model::new(0xFEED_BEEF_1234_5678);
    let mut opt   = ModelAdam::new(&model);
    let mut rng   = 0xDEAD_BEEF_u64;

    eprintln!("Dense baseline  hidden={HIDDEN}  ffn={FFN_DIM}  layers={N_LAYERS}  \
               heads={N_HEADS}  seq={SEQ_LEN}  vocab={VOCAB}");
    eprintln!("Params: {}K   data: {} tokens",
        model.param_count() / 1000, dataset.data.len());
    eprintln!("Opt-steps: {n_opt_steps}  accum={ACCUM_STEPS}  \
               lr_max={lr_max}  warmup={LR_WARMUP}  grad_clip={max_grad_norm}");
    eprintln!("─────────────────────────────────────────────────");

    let mut accum      = zero_grads(&model);
    let mut accum_loss = 0.0f32;
    let t0 = Instant::now();

    for opt_step in 0..n_opt_steps {
        for _ in 0..ACCUM_STEPS {
            let (inp, tgt) = dataset.sample(&mut rng);
            let (logits, cache) = model.forward(&inp);
            let (loss, dlogits) = cross_entropy(&logits, &tgt);
            let grads = model.backward(&cache, &dlogits);
            add_grads(&mut accum, &grads);
            accum_loss += loss;
        }

        scale_grads(&mut accum, 1.0 / ACCUM_STEPS as f32);
        let gnorm = clip_grads(&mut accum, max_grad_norm);
        let lr = schedule_lr(opt_step, n_opt_steps, lr_max);
        opt.step(&mut model, &accum, lr);

        let avg_loss = accum_loss / ACCUM_STEPS as f32;
        reset_grads(&mut accum);
        accum_loss = 0.0;

        if opt_step % 100 == 0 || opt_step < 10 {
            let elapsed = t0.elapsed().as_secs_f32();
            let ms_per_opt = if opt_step > 0 {
                elapsed / opt_step as f32 * 1000.0
            } else { 0.0 };
            eprintln!("step {:>5}  loss={:.4}  gnorm={:.3}  lr={:.2e}  {:.0}ms/step",
                opt_step, avg_loss, gnorm, lr, ms_per_opt);
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
}
