//! Single transformer layer throughput bench using SharedMonarchMatmul.
//!
//! Layer layout (Llama-style, no MoE routing):
//!   RMSNorm → Q,K,V projections → attention scores → O projection → residual
//!   RMSNorm → FFN up+gate (SwiGLU) → FFN down → residual
//!
//! Benchmarks:
//!   seq=1   — autoregressive decode, KV cache assumed (attn is a single vec × K/V)
//!   seq=256 — prefill / training forward pass (all tokens in parallel)

use std::time::Instant;
use fydel::kernels::monarch::SharedMonarchMatmul;
use fydel::kernels::norm;

// ---------------------------------------------------------------------------
// Config (Fydel 1B)
// ---------------------------------------------------------------------------

const HIDDEN:    usize = 896;
const FFN:       usize = 3072;
const N_HEADS:   usize = 14;
const HEAD_DIM:  usize = HIDDEN / N_HEADS;   // 64
const B:         usize = 64;                  // block size = m² = 8²
const ND:        usize = 8;                   // shared atom count

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

struct MonarchLayer {
    // Attention projections (HIDDEN×HIDDEN each)
    wq: SharedMonarchMatmul,
    wk: SharedMonarchMatmul,
    wv: SharedMonarchMatmul,
    wo: SharedMonarchMatmul,
    // FFN projections
    w_up:   SharedMonarchMatmul,  // HIDDEN → FFN
    w_gate: SharedMonarchMatmul,  // HIDDEN → FFN  (SwiGLU gate)
    w_down: SharedMonarchMatmul,  // FFN → HIDDEN
    // Norms
    attn_norm_gain: Vec<f32>,
    ffn_norm_gain:  Vec<f32>,
}

impl MonarchLayer {
    fn new(seed: u64) -> Self {
        let p_hh = HIDDEN / B;       // 14
        let q_hh = HIDDEN / B;       // 14
        let p_hf = FFN    / B;       // 48
        let q_hf = HIDDEN / B;       // 14
        let p_fh = HIDDEN / B;       // 14
        let q_fh = FFN    / B;       // 48
        let m = (B as f64).sqrt() as usize;
        Self {
            wq:    SharedMonarchMatmul::new(p_hh, q_hh, m, ND, seed ^ 0x01),
            wk:    SharedMonarchMatmul::new(p_hh, q_hh, m, ND, seed ^ 0x02),
            wv:    SharedMonarchMatmul::new(p_hh, q_hh, m, ND, seed ^ 0x03),
            wo:    SharedMonarchMatmul::new(p_hh, q_hh, m, ND, seed ^ 0x04),
            w_up:  SharedMonarchMatmul::new(p_hf, q_hf, m, ND, seed ^ 0x05),
            w_gate:SharedMonarchMatmul::new(p_hf, q_hf, m, ND, seed ^ 0x06),
            w_down:SharedMonarchMatmul::new(p_fh, q_fh, m, ND, seed ^ 0x07),
            attn_norm_gain: vec![1.0f32; HIDDEN],
            ffn_norm_gain:  vec![1.0f32; HIDDEN],
        }
    }

    fn param_count(&self) -> usize {
        self.wq.param_count() + self.wk.param_count() + self.wv.param_count()
            + self.wo.param_count() + self.w_up.param_count()
            + self.w_gate.param_count() + self.w_down.param_count()
            + 2 * HIDDEN
    }

    /// Forward for a single token (decode). Returns output hidden state.
    fn forward_decode(&self, h: &[f32]) -> Vec<f32> {
        // Attention pre-norm
        let mut h_norm = vec![0.0f32; HIDDEN];
        norm::forward(h, &self.attn_norm_gain, 1e-5, &mut h_norm);

        // Q, K, V projections
        let q = self.wq.forward_inference(&h_norm);
        let k = self.wk.forward_inference(&h_norm);
        let v = self.wv.forward_inference(&h_norm);

        // Single-token attention: for decode, KV history is 1 token (self-attend)
        // In practice KV cache is external; here we just do the same-token attend.
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut attn_out = vec![0.0f32; HIDDEN];
        for h_idx in 0..N_HEADS {
            let qs = &q[h_idx * HEAD_DIM..(h_idx + 1) * HEAD_DIM];
            let ks = &k[h_idx * HEAD_DIM..(h_idx + 1) * HEAD_DIM];
            let vs = &v[h_idx * HEAD_DIM..(h_idx + 1) * HEAD_DIM];
            // score = dot(q, k) * scale — scalar for seq=1
            let score: f32 = qs.iter().zip(ks).map(|(a, b)| a * b).sum::<f32>() * scale;
            let weight = score.exp() / score.exp(); // softmax over 1 token = 1.0
            let ao = &mut attn_out[h_idx * HEAD_DIM..(h_idx + 1) * HEAD_DIM];
            for i in 0..HEAD_DIM { ao[i] += weight * vs[i]; }
        }

        // Output projection + residual
        let o_proj = self.wo.forward_inference(&attn_out);
        let mut h2: Vec<f32> = h.iter().zip(&o_proj).map(|(a, b)| a + b).collect();

        // FFN pre-norm
        let mut h2_norm = vec![0.0f32; HIDDEN];
        norm::forward(&h2, &self.ffn_norm_gain, 1e-5, &mut h2_norm);

        // SwiGLU FFN
        let up   = self.w_up.forward_inference(&h2_norm);
        let gate = self.w_gate.forward_inference(&h2_norm);
        // SiLU(gate) * up
        let act: Vec<f32> = gate.iter().zip(&up)
            .map(|(g, u)| u * g / (1.0 + (-g).exp()))
            .collect();
        let down = self.w_down.forward_inference(&act);

        for i in 0..HIDDEN { h2[i] += down[i]; }
        h2
    }

    /// Forward for `seq` tokens sequentially. Returns output hidden states flat [seq * HIDDEN].
    fn forward_prefill(&self, tokens: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq = tokens.len();

        // Precompute all Q, K, V for all tokens first (parallel per-token via monarch's rayon)
        let qs: Vec<Vec<f32>> = tokens.iter().map(|h| {
            let mut n = vec![0.0f32; HIDDEN];
            norm::forward(h, &self.attn_norm_gain, 1e-5, &mut n);
            self.wq.forward_inference(&n)
        }).collect();
        let ks: Vec<Vec<f32>> = tokens.iter().map(|h| {
            let mut n = vec![0.0f32; HIDDEN];
            norm::forward(h, &self.attn_norm_gain, 1e-5, &mut n);
            self.wk.forward_inference(&n)
        }).collect();
        let vs: Vec<Vec<f32>> = tokens.iter().map(|h| {
            let mut n = vec![0.0f32; HIDDEN];
            norm::forward(h, &self.attn_norm_gain, 1e-5, &mut n);
            self.wv.forward_inference(&n)
        }).collect();

        // Attention: O(seq² × head_dim) — causal, parallel over t (disjoint output rows).
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut attn_outs: Vec<Vec<f32>> = vec![vec![0.0f32; HIDDEN]; seq];
        {
            use rayon::prelude::*;
            attn_outs.par_iter_mut().enumerate().for_each(|(t, ao_t)| {
                for h_idx in 0..N_HEADS {
                    let qt = &qs[t][h_idx * HEAD_DIM..(h_idx + 1) * HEAD_DIM];
                    let max_s = (0..=t).map(|s| {
                        let ks_h = &ks[s][h_idx * HEAD_DIM..(h_idx + 1) * HEAD_DIM];
                        qt.iter().zip(ks_h).map(|(a, b)| a * b).sum::<f32>() * scale
                    }).fold(f32::MIN, f32::max);
                    let exps: Vec<f32> = (0..=t).map(|s| {
                        let ks_h = &ks[s][h_idx * HEAD_DIM..(h_idx + 1) * HEAD_DIM];
                        let sc   = qt.iter().zip(ks_h).map(|(a, b)| a * b).sum::<f32>() * scale;
                        (sc - max_s).exp()
                    }).collect();
                    let sum_e: f32 = exps.iter().sum();
                    let ao = &mut ao_t[h_idx * HEAD_DIM..(h_idx + 1) * HEAD_DIM];
                    for (s, &w) in exps.iter().enumerate() {
                        let w = w / sum_e;
                        let vs_h = &vs[s][h_idx * HEAD_DIM..(h_idx + 1) * HEAD_DIM];
                        for i in 0..HEAD_DIM { ao[i] += w * vs_h[i]; }
                    }
                }
            });
        }

        // Output projection + FFN for each token
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(seq);
        for t in 0..seq {
            let o_proj = self.wo.forward_inference(&attn_outs[t]);
            let mut h2: Vec<f32> = tokens[t].iter().zip(&o_proj).map(|(a, b)| a + b).collect();

            let mut h2_norm = vec![0.0f32; HIDDEN];
            norm::forward(&h2, &self.ffn_norm_gain, 1e-5, &mut h2_norm);

            let up   = self.w_up.forward_inference(&h2_norm);
            let gate = self.w_gate.forward_inference(&h2_norm);
            let act: Vec<f32> = gate.iter().zip(&up)
                .map(|(g, u)| u * g / (1.0 + (-g).exp()))
                .collect();
            let down = self.w_down.forward_inference(&act);
            for i in 0..HIDDEN { h2[i] += down[i]; }
            out.push(h2);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let layer = MonarchLayer::new(42);
    let total_params: usize = layer.param_count();
    eprintln!("Layer params: {}K  ({} per projection avg)",
        total_params / 1000,
        total_params / 7);
    eprintln!("Full 1B est: {}M params across 96 layers",
        total_params * 96 / 1_000_000);

    // --- Decode bench (seq=1) ---
    let h0: Vec<f32> = (0..HIDDEN).map(|i| (i as f32 * 0.001).sin()).collect();
    let iters = 200;

    // Warmup
    for _ in 0..20 { let _ = std::hint::black_box(layer.forward_decode(&h0)); }
    let t = Instant::now();
    for _ in 0..iters { let _ = std::hint::black_box(layer.forward_decode(&h0)); }
    let decode_us = t.elapsed().as_secs_f64() / iters as f64 * 1e6;

    let full_model_decode_ms = decode_us * 96.0 / 1000.0;
    eprintln!("\n=== Decode (seq=1) ===");
    eprintln!("  1 layer:   {:.1}µs", decode_us);
    eprintln!("  96 layers: {:.1}ms/token  →  {:.0} tok/s",
        full_model_decode_ms,
        1000.0 / full_model_decode_ms);

    // --- Prefill bench (seq=256) ---
    let seq = 256usize;
    let tokens: Vec<Vec<f32>> = (0..seq)
        .map(|t| (0..HIDDEN).map(|i| ((i + t) as f32 * 0.001).sin()).collect())
        .collect();

    // Warmup
    for _ in 0..3 { let _ = std::hint::black_box(layer.forward_prefill(&tokens)); }
    let t = Instant::now();
    let prefill_iters = 10;
    for _ in 0..prefill_iters { let _ = std::hint::black_box(layer.forward_prefill(&tokens)); }
    let prefill_ms = t.elapsed().as_secs_f64() / prefill_iters as f64 * 1000.0;
    let prefill_us_per_tok = prefill_ms * 1000.0 / seq as f64;

    let full_model_prefill_ms = prefill_ms * 96.0;
    eprintln!("\n=== Prefill (seq={seq}) ===");
    eprintln!("  1 layer:   {:.1}ms  ({:.1}µs/tok)",
        prefill_ms, prefill_us_per_tok);
    eprintln!("  96 layers: {:.0}ms/step  →  {:.0} tok/s",
        full_model_prefill_ms,
        seq as f64 / (full_model_prefill_ms / 1000.0));
}
