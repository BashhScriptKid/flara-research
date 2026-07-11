//! Wall-clock benchmarks (criterion) comparing Dense, CausalMonarch, Sliding,
//! and Meta MonarchAttention at the context lengths this project actually targets
//! (512/2048/8192, per RESEARCH_LOG.md's own full-attention-layer
//! benchmarks, cited in ROOFLINE_5500U.md). This is the wall-clock half
//! of the empirical check; `perf stat` against `src/bin/profile.rs`
//! covers the hardware-counter half (L2/L3 miss rate, instructions
//! retired) that criterion's own sampling/statistics make it a poor fit
//! for -- see that binary's doc comment.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use monarch_attn_kernel::causal_monarch::{causal_monarch_attention, CausalMonarchConfig};
use monarch_attn_kernel::dense::dense_causal_attention;
use monarch_attn_kernel::meta::{monarch_meta_threshold_fast_residual, MetaConfig, TauMode};
use monarch_attn_kernel::sliding::{sliding_monarch_causal, SlidingConfig};
use monarch_attn_kernel::{AttnConfig, HeadTensor};

fn randn_tensor(n_heads: usize, seq_len: usize, head_dim: usize, seed: u64) -> HeadTensor {
    let mut state = seed.max(1);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32 * 0.5
    };
    let data = (0..n_heads * seq_len * head_dim).map(|_| next()).collect();
    HeadTensor { data, n_heads, seq_len, head_dim }
}

const SEQ_LENS: [usize; 3] = [512, 2048, 8192];

fn bench_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense");
    let cfg = AttnConfig::production(); // head_dim=64, n_q_heads=14, n_kv_heads=2 (GQA)
    for &seq_len in &SEQ_LENS {
        let q = randn_tensor(cfg.n_q_heads, seq_len, cfg.head_dim, 1);
        let k = randn_tensor(cfg.n_kv_heads, seq_len, cfg.head_dim, 2);
        let v = randn_tensor(cfg.n_kv_heads, seq_len, cfg.head_dim, 3);
        group.bench_with_input(BenchmarkId::from_parameter(seq_len), &seq_len, |b, _| {
            b.iter(|| black_box(dense_causal_attention(black_box(&q), black_box(&k), black_box(&v), black_box(&cfg))));
        });
    }
    group.finish();
}

fn bench_causal_monarch(c: &mut Criterion) {
    let mut group = c.benchmark_group("causal_monarch");
    let cfg = CausalMonarchConfig { head_dim: 64, n_heads: 8, block: 64, t: 3 };
    for &seq_len in &SEQ_LENS {
        let q = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim, 1);
        let k = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim, 2);
        let v = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim, 3);
        group.bench_with_input(BenchmarkId::from_parameter(seq_len), &seq_len, |b, _| {
            b.iter(|| black_box(causal_monarch_attention(black_box(&q), black_box(&k), black_box(&v), black_box(&cfg))));
        });
    }
    group.finish();
}

fn bench_sliding(c: &mut Criterion) {
    let mut group = c.benchmark_group("sliding");
    // same head_dim/block as production, uniform 8 heads (reference algorithm predates GQA)
    let cfg = SlidingConfig { head_dim: 64, n_heads: 8, block: 64, w_blocks: 4, t: 3, w_refine: 4 };
    for &seq_len in &SEQ_LENS {
        let q = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim, 1);
        let k = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim, 2);
        let v = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim, 3);
        group.bench_with_input(BenchmarkId::from_parameter(seq_len), &seq_len, |b, _| {
            b.iter(|| black_box(sliding_monarch_causal(black_box(&q), black_box(&k), black_box(&v), black_box(&cfg))));
        });
    }
    group.finish();
}

fn bench_meta(c: &mut Criterion) {
    // PRODUCTION RECOMMENDATION: quickselect tau, fast-residual, no
    // T-iteration -- see profile.rs's "meta" doc comment and JOURNAL.md /
    // ROOFLINE_5500U.md for the full validation history.
    let mut group = c.benchmark_group("meta");
    let cfg = MetaConfig { head_dim: 64, head_dim_v: 64, n_heads: 8, block: 64, w_blocks: 1, quantile: 0.90, tau_mode: TauMode::Quickselect };
    for &seq_len in &SEQ_LENS {
        let q = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim, 1);
        let k = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim, 2);
        let v = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim_v, 3);
        group.bench_with_input(BenchmarkId::from_parameter(seq_len), &seq_len, |b, _| {
            b.iter(|| black_box(monarch_meta_threshold_fast_residual(black_box(&q), black_box(&k), black_box(&v), black_box(&cfg))));
        });
    }
    group.finish();
}

fn bench_meta_sortref(c: &mut Criterion) {
    // Reference-only: sort-based tau, same tau value/output as bench_meta,
    // kept for direct before/after cost comparison against the production
    // Quickselect path (~56% of cycles at N=8192 go to the sort -- see
    // JOURNAL.md's perf-record finding).
    let mut group = c.benchmark_group("meta_sortref");
    let cfg = MetaConfig { head_dim: 64, head_dim_v: 64, n_heads: 8, block: 64, w_blocks: 1, quantile: 0.90, tau_mode: TauMode::SortBased };
    for &seq_len in &SEQ_LENS {
        let q = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim, 1);
        let k = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim, 2);
        let v = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim_v, 3);
        group.bench_with_input(BenchmarkId::from_parameter(seq_len), &seq_len, |b, _| {
            b.iter(|| black_box(monarch_meta_threshold_fast_residual(black_box(&q), black_box(&k), black_box(&v), black_box(&cfg))));
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    // seq_len=8192 kernels can run into multi-second-per-iteration territory
    // (O(N^2)-ish cost); keep sample_size at criterion's minimum and cap
    // measurement/warm-up time so the full 3-kernel x 3-seq_len suite stays
    // tractable rather than running for tens of minutes.
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(15));
    targets = bench_dense, bench_causal_monarch, bench_sliding, bench_meta, bench_meta_sortref
}
criterion_main!(benches);
