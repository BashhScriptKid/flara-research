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
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use fydel::kernels::monarch::SharedMonarchMatmul;
use fydel::kernels::norm;
use fydel::kernels::fastmath;

// ---------------------------------------------------------------------------
// Coarse decode-phase profiling (investigation only — PROFILE=1 env var)
// ---------------------------------------------------------------------------
static PROFILE_ON: AtomicBool = AtomicBool::new(false);
static T_NORM1: AtomicU64 = AtomicU64::new(0);
static T_QKV_GROUP: AtomicU64 = AtomicU64::new(0);
static T_ATTN: AtomicU64 = AtomicU64::new(0);
static T_WO: AtomicU64 = AtomicU64::new(0);
static T_NORM2: AtomicU64 = AtomicU64::new(0);
static T_UPGATE_GROUP: AtomicU64 = AtomicU64::new(0);
static T_SWIGLU: AtomicU64 = AtomicU64::new(0);
static T_DOWN: AtomicU64 = AtomicU64::new(0);

#[inline]
fn add_ns(counter: &AtomicU64, t0: Instant) {
    if PROFILE_ON.load(Ordering::Relaxed) {
        counter.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

fn print_decode_profile(iters: usize) {
    let phases: [(&str, &AtomicU64); 8] = [
        ("norm1", &T_NORM1), ("qkv_group", &T_QKV_GROUP),
        ("attn", &T_ATTN), ("wo", &T_WO), ("norm2", &T_NORM2),
        ("upgate_group", &T_UPGATE_GROUP), ("swiglu", &T_SWIGLU), ("down", &T_DOWN),
    ];
    let total: u64 = phases.iter().map(|(_, c)| c.load(Ordering::Relaxed)).sum();
    eprintln!("\n─── decode phase profile (per layer, avg over {iters} iters) ───");
    for (name, c) in &phases {
        let ns = c.load(Ordering::Relaxed) / iters as u64;
        let us = ns as f64 / 1e3;
        let pct = if total > 0 { 100.0 * c.load(Ordering::Relaxed) as f64 / total as f64 } else { 0.0 };
        eprintln!("{name:<8} {us:>8.2} µs   {pct:>5.1}%");
    }
}

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
        let t0 = Instant::now();
        norm::forward(h, &self.attn_norm_gain, 1e-5, &mut h_norm);
        add_ns(&T_NORM1, t0);

        // Q, K, V projections — fused into one rayon dispatch since they
        // share the same input (h_norm); see forward_inference_grouped.
        let t0 = Instant::now();
        let qkv = SharedMonarchMatmul::forward_inference_grouped(&[&self.wq, &self.wk, &self.wv], &h_norm);
        let q = &qkv[0 * HIDDEN..1 * HIDDEN];
        let k = &qkv[1 * HIDDEN..2 * HIDDEN];
        let v = &qkv[2 * HIDDEN..3 * HIDDEN];
        add_ns(&T_QKV_GROUP, t0);

        // Single-token attention: for decode, KV history is 1 token (self-attend)
        // In practice KV cache is external; here we just do the same-token attend.
        let t0 = Instant::now();
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
        add_ns(&T_ATTN, t0);

        // Output projection + residual
        let t0 = Instant::now();
        let o_proj = self.wo.forward_inference(&attn_out);
        add_ns(&T_WO, t0);
        let mut h2: Vec<f32> = h.iter().zip(&o_proj).map(|(a, b)| a + b).collect();

        // FFN pre-norm
        let mut h2_norm = vec![0.0f32; HIDDEN];
        let t0 = Instant::now();
        norm::forward(&h2, &self.ffn_norm_gain, 1e-5, &mut h2_norm);
        add_ns(&T_NORM2, t0);

        // SwiGLU FFN — up/gate fused into one dispatch (same input h2_norm).
        let t0 = Instant::now();
        let upgate = SharedMonarchMatmul::forward_inference_grouped(&[&self.w_up, &self.w_gate], &h2_norm);
        let up   = &upgate[0 * FFN..1 * FFN];
        let gate = &upgate[1 * FFN..2 * FFN];
        add_ns(&T_UPGATE_GROUP, t0);
        // SiLU(gate) * up
        let t0 = Instant::now();
        let mut act = vec![0.0f32; FFN];
        fastmath::swiglu_forward(up, gate, &mut act);
        add_ns(&T_SWIGLU, t0);
        let t0 = Instant::now();
        let down = self.w_down.forward_inference(&act);
        add_ns(&T_DOWN, t0);

        for i in 0..HIDDEN { h2[i] += down[i]; }
        h2
    }

    /// Forward for a single decode token against a *real* KV cache — unlike
    /// `forward_decode` (which self-attends to one token, so softmax over 1
    /// token is always weight=1.0 and attention is essentially free), this
    /// does real causal attention over `ctx_len` cached positions, streamed
    /// from `k_cache`/`v_cache` (each `[max_ctx * HIDDEN]`, row-major by
    /// position). If `window` is `Some(w)`, only the last `w` cached
    /// positions are attended to (sliding-window attention, matching the
    /// production model's design for layers >= `full_attn_layers`).
    ///
    /// Returns `(output, new_k, new_v)` — caller appends `new_k`/`new_v`
    /// onto the cache for the next decode step.
    fn forward_decode_cached(
        &self, h: &[f32], k_cache: &[f32], v_cache: &[f32], ctx_len: usize, window: Option<usize>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut h_norm = vec![0.0f32; HIDDEN];
        norm::forward(h, &self.attn_norm_gain, 1e-5, &mut h_norm);

        let qkv = SharedMonarchMatmul::forward_inference_grouped(&[&self.wq, &self.wk, &self.wv], &h_norm);
        let q     = qkv[0 * HIDDEN..1 * HIDDEN].to_vec();
        let k_new = qkv[1 * HIDDEN..2 * HIDDEN].to_vec();
        let v_new = qkv[2 * HIDDEN..3 * HIDDEN].to_vec();

        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let start = match window {
            Some(w) => ctx_len.saturating_sub(w),
            None => 0,
        };
        let n_ctx = ctx_len - start;

        let mut attn_out = vec![0.0f32; HIDDEN];
        for h_idx in 0..N_HEADS {
            let qs = &q[h_idx * HEAD_DIM..(h_idx + 1) * HEAD_DIM];

            let mut scores = Vec::with_capacity(n_ctx + 1);
            let mut max_s = f32::MIN;
            for pos in start..ctx_len {
                let ks = &k_cache[pos * HIDDEN + h_idx * HEAD_DIM..pos * HIDDEN + (h_idx + 1) * HEAD_DIM];
                let s: f32 = qs.iter().zip(ks).map(|(a, b)| a * b).sum::<f32>() * scale;
                scores.push(s);
                if s > max_s { max_s = s; }
            }
            let ks_new = &k_new[h_idx * HEAD_DIM..(h_idx + 1) * HEAD_DIM];
            let s_self: f32 = qs.iter().zip(ks_new).map(|(a, b)| a * b).sum::<f32>() * scale;
            scores.push(s_self);
            if s_self > max_s { max_s = s_self; }

            let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum_e: f32 = exps.iter().sum();

            let ao = &mut attn_out[h_idx * HEAD_DIM..(h_idx + 1) * HEAD_DIM];
            for (i, pos) in (start..ctx_len).enumerate() {
                let w = exps[i] / sum_e;
                let vs = &v_cache[pos * HIDDEN + h_idx * HEAD_DIM..pos * HIDDEN + (h_idx + 1) * HEAD_DIM];
                for d in 0..HEAD_DIM { ao[d] += w * vs[d]; }
            }
            let w_self = exps[n_ctx] / sum_e;
            let vs_new = &v_new[h_idx * HEAD_DIM..(h_idx + 1) * HEAD_DIM];
            for d in 0..HEAD_DIM { ao[d] += w_self * vs_new[d]; }
        }

        let o_proj = self.wo.forward_inference(&attn_out);
        let mut h2: Vec<f32> = h.iter().zip(&o_proj).map(|(a, b)| a + b).collect();

        let mut h2_norm = vec![0.0f32; HIDDEN];
        norm::forward(&h2, &self.ffn_norm_gain, 1e-5, &mut h2_norm);

        let upgate = SharedMonarchMatmul::forward_inference_grouped(&[&self.w_up, &self.w_gate], &h2_norm);
        let up   = &upgate[0 * FFN..1 * FFN];
        let gate = &upgate[1 * FFN..2 * FFN];
        let mut act = vec![0.0f32; FFN];
        fastmath::swiglu_forward(up, gate, &mut act);
        let down = self.w_down.forward_inference(&act);
        for i in 0..HIDDEN { h2[i] += down[i]; }

        (h2, k_new, v_new)
    }

    /// Forward for `seq` tokens sequentially. Returns output hidden states flat [seq * HIDDEN].
    fn forward_prefill(&self, tokens: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq = tokens.len();

        // Precompute Q, K, V — one norm per token, three projections, parallel over tokens.
        use rayon::prelude::*;
        // NOTE: forward_inference_serial (no inner rayon) was tried here to
        // avoid nested-parallelism contention with the outer tokens.par_iter()
        // — benchmarked as a regression (RESEARCH_LOG.md), reverted to plain
        // forward_inference.
        let qkv: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = tokens.par_iter().map(|h| {
            let mut n = vec![0.0f32; HIDDEN];
            norm::forward(h, &self.attn_norm_gain, 1e-5, &mut n);
            let q = self.wq.forward_inference(&n);
            let k = self.wk.forward_inference(&n);
            let v = self.wv.forward_inference(&n);
            (q, k, v)
        }).collect();
        let (qs, ks, vs): (Vec<_>, Vec<_>, Vec<_>) = qkv.into_iter()
            .fold((Vec::with_capacity(seq), Vec::with_capacity(seq), Vec::with_capacity(seq)),
                |(mut qa, mut ka, mut va), (q, k, v)| { qa.push(q); ka.push(k); va.push(v); (qa, ka, va) });

        // Attention: O(seq² × head_dim) — causal, parallel over t (disjoint output rows).
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut attn_outs: Vec<Vec<f32>> = vec![vec![0.0f32; HIDDEN]; seq];
        {
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

        // Output projection + FFN — parallel over tokens (each token is independent).
        let mut out: Vec<Vec<f32>> = vec![vec![0.0f32; HIDDEN]; seq];
        out.par_iter_mut().enumerate().for_each(|(t, out_t)| {
            let o_proj = self.wo.forward_inference(&attn_outs[t]);
            let mut h2: Vec<f32> = tokens[t].iter().zip(&o_proj).map(|(a, b)| a + b).collect();

            let mut h2_norm = vec![0.0f32; HIDDEN];
            norm::forward(&h2, &self.ffn_norm_gain, 1e-5, &mut h2_norm);

            let up   = self.w_up.forward_inference(&h2_norm);
            let gate = self.w_gate.forward_inference(&h2_norm);
            let mut act = vec![0.0f32; FFN];
            fastmath::swiglu_forward(&up, &gate, &mut act);
            let down = self.w_down.forward_inference(&act);
            for i in 0..HIDDEN { h2[i] += down[i]; }
            *out_t = h2;
        });
        out
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    if std::env::var("PROFILE").ok().as_deref() == Some("1") {
        PROFILE_ON.store(true, Ordering::Relaxed);
    }
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
    for c in [&T_NORM1, &T_QKV_GROUP, &T_ATTN, &T_WO, &T_NORM2, &T_UPGATE_GROUP, &T_SWIGLU, &T_DOWN] {
        c.store(0, Ordering::Relaxed);
    }
    let t = Instant::now();
    for _ in 0..iters { let _ = std::hint::black_box(layer.forward_decode(&h0)); }
    let decode_us = t.elapsed().as_secs_f64() / iters as f64 * 1e6;

    let full_model_decode_ms = decode_us * 96.0 / 1000.0;
    eprintln!("\n=== Decode (seq=1) ===");
    eprintln!("  1 layer:   {:.1}µs", decode_us);
    eprintln!("  96 layers: {:.1}ms/token  →  {:.0} tok/s",
        full_model_decode_ms,
        1000.0 / full_model_decode_ms);
    if PROFILE_ON.load(Ordering::Relaxed) { print_decode_profile(iters); }

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

    // --- Decode WITH a real KV cache (streams K/V from memory, real causal
    // attention) — unlike the seq=1 bench above, where softmax over a
    // single self-attended token is always weight=1.0 and attention is
    // essentially free. Matches the production config: 24 full-attention
    // layers, 72 sliding-window layers (window=256), out of 96 total.
    const FULL_ATTN_LAYERS: usize = 24;
    const WINDOW: usize = 256;
    let n_windowed = 96 - FULL_ATTN_LAYERS;

    eprintln!("\n=== Decode with KV cache (real attention, {FULL_ATTN_LAYERS} full + {n_windowed} windowed[{WINDOW}] layers) ===");
    for &ctx_len in &[0usize, 512, 2048, 8192] {
        let max_ctx = ctx_len.max(1);
        let k_cache: Vec<f32> = (0..max_ctx * HIDDEN).map(|i| ((i as f32) * 0.0001).sin()).collect();
        let v_cache: Vec<f32> = (0..max_ctx * HIDDEN).map(|i| ((i as f32) * 0.0002).cos()).collect();

        let bench_one = |window: Option<usize>| -> f64 {
            for _ in 0..10 {
                let _ = std::hint::black_box(layer.forward_decode_cached(&h0, &k_cache, &v_cache, ctx_len, window));
            }
            let iters = 50;
            let t = Instant::now();
            for _ in 0..iters {
                let _ = std::hint::black_box(layer.forward_decode_cached(&h0, &k_cache, &v_cache, ctx_len, window));
            }
            t.elapsed().as_secs_f64() / iters as f64 * 1e6
        };

        let full_us = bench_one(None);
        let windowed_us = bench_one(Some(WINDOW));

        let full_model_ms = (FULL_ATTN_LAYERS as f64 * full_us + n_windowed as f64 * windowed_us) / 1000.0;
        eprintln!("  ctx={ctx_len:>5}: full-layer={:>7.1}µs  windowed-layer={:>6.1}µs  →  96-layer: {:.1}ms/token  →  {:.0} tok/s",
            full_us, windowed_us, full_model_ms, 1000.0 / full_model_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_decode_at_zero_ctx_matches_self_attend_decode() {
        // ctx_len=0 means attention only sees the new token itself — same
        // math as forward_decode's self-attend stub (softmax over 1 token
        // is always weight=1.0). This is a free correctness check: two
        // independently-written attention implementations should agree
        // exactly on this reduced case.
        let layer = MonarchLayer::new(7);
        let h0: Vec<f32> = (0..HIDDEN).map(|i| (i as f32 * 0.001).sin()).collect();

        let stub_out = layer.forward_decode(&h0);
        let (cached_out, _k, _v) = layer.forward_decode_cached(&h0, &[], &[], 0, None);

        assert_eq!(stub_out.len(), cached_out.len());
        for i in 0..stub_out.len() {
            assert!((stub_out[i] - cached_out[i]).abs() < 1e-5,
                "idx {i}: forward_decode={} forward_decode_cached(ctx=0)={}", stub_out[i], cached_out[i]);
        }
    }

    #[test]
    fn windowed_cache_only_reads_last_window_positions() {
        // With window=Some(w) and ctx_len > w, changing cache content
        // outside the window shouldn't change the output at all.
        let layer = MonarchLayer::new(11);
        let h0: Vec<f32> = (0..HIDDEN).map(|i| (i as f32 * 0.001).sin()).collect();
        let ctx_len = 300;
        let window = 100;

        let k_cache_a: Vec<f32> = (0..ctx_len * HIDDEN).map(|i| (i as f32 * 0.0001).sin()).collect();
        let v_cache_a: Vec<f32> = (0..ctx_len * HIDDEN).map(|i| (i as f32 * 0.0002).cos()).collect();
        // perturb only the out-of-window region (positions 0..ctx_len-window)
        let mut k_cache_b = k_cache_a.clone();
        let mut v_cache_b = v_cache_a.clone();
        for i in 0..(ctx_len - window) * HIDDEN {
            k_cache_b[i] += 100.0;
            v_cache_b[i] += 100.0;
        }

        let (out_a, _, _) = layer.forward_decode_cached(&h0, &k_cache_a, &v_cache_a, ctx_len, Some(window));
        let (out_b, _, _) = layer.forward_decode_cached(&h0, &k_cache_b, &v_cache_b, ctx_len, Some(window));

        for i in 0..out_a.len() {
            assert!((out_a[i] - out_b[i]).abs() < 1e-5,
                "idx {i}: windowed output changed despite perturbing only out-of-window cache entries");
        }
    }
}
