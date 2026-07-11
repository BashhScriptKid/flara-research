//! Standalone profiling target for `perf stat` / wall-clock measurement.
//! Deliberately NOT criterion -- criterion's own harness overhead and
//! statistical resampling make it a poor target for attaching hardware
//! counters to; a plain binary running a fixed number of iterations in a
//! tight loop is the standard approach for perf-stat-based measurement.
//!
//! Prints a checksum of the output (sum of all output elements) so runs
//! are comparable/sane-checkable, and so the compiler cannot prove the
//! output is unused and dead-code-eliminate the whole computation.
//!
//! Usage: `profile <causal|sliding|meta> <seq_len> <iterations>`

use monarch_attn_kernel::causal::dense_causal_attention;
use monarch_attn_kernel::meta::{monarch_meta_threshold_fast_residual, MetaConfig, TauMode};
use monarch_attn_kernel::sliding::{sliding_monarch_causal, SlidingConfig};
use monarch_attn_kernel::{AttnConfig, HeadTensor};
use std::time::Instant;

fn randn_tensor(n_heads: usize, seq_len: usize, head_dim: usize, seed: u64) -> HeadTensor {
    // xorshift64 -- deterministic, dependency-free, good enough for benchmark inputs
    let mut state = seed.max(1);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // map to roughly N(0, 0.5^2)-ish range via a cheap Box-Muller-free trick:
        // uniform in [-1,1] scaled down, sufficient for a benchmark workload
        ((state as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32 * 0.5
    };
    let mut data = vec![0.0f32; n_heads * seq_len * head_dim];
    for x in data.iter_mut() {
        *x = next();
    }
    HeadTensor { data, n_heads, seq_len, head_dim }
}

fn checksum(t: &HeadTensor) -> f64 {
    t.data.iter().map(|&x| x as f64).sum()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: profile <causal|sliding|meta> <seq_len> <iterations>");
        std::process::exit(2);
    }
    let kernel = &args[1];
    let seq_len: usize = args[2].parse().expect("seq_len must be an integer");
    let iterations: usize = args[3].parse().expect("iterations must be an integer");

    let mut total_checksum = 0.0f64;
    let start = Instant::now();

    match kernel.as_str() {
        "causal" => {
            let cfg = AttnConfig::production(); // head_dim=64, n_q_heads=14, n_kv_heads=2
            let q = randn_tensor(cfg.n_q_heads, seq_len, cfg.head_dim, 1);
            let k = randn_tensor(cfg.n_kv_heads, seq_len, cfg.head_dim, 2);
            let v = randn_tensor(cfg.n_kv_heads, seq_len, cfg.head_dim, 3);
            for _ in 0..iterations {
                let out = dense_causal_attention(&q, &k, &v, &cfg);
                total_checksum += checksum(&out);
            }
        }
        "sliding" => {
            let cfg = SlidingConfig { head_dim: 64, n_heads: 8, block: 64, w_blocks: 4, t: 3, w_refine: 4 };
            let q = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim, 1);
            let k = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim, 2);
            let v = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim, 3);
            for _ in 0..iterations {
                let out = sliding_monarch_causal(&q, &k, &v, &cfg);
                total_checksum += checksum(&out);
            }
        }
        "meta" => {
            // PRODUCTION RECOMMENDATION: quickselect tau (O(n), same tau value
            // and output as sort-based -- see meta_reference_check.rs's
            // quickselect_tau_matches_sort_based_tau test), fast-residual,
            // no T-iteration. ~1.22x faster than dense at N=8192 on this
            // hardware, correctness parity with dense on every controlled
            // probe including the same-norm adversarial scenes that
            // disqualified Sliding. See JOURNAL.md / ROOFLINE_5500U.md.
            let cfg = MetaConfig { head_dim: 64, head_dim_v: 64, n_heads: 8, block: 64, w_blocks: 1, quantile: 0.90, tau_mode: TauMode::Quickselect };
            let q = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim, 1);
            let k = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim, 2);
            let v = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim_v, 3);
            for _ in 0..iterations {
                let out = monarch_meta_threshold_fast_residual(&q, &k, &v, &cfg);
                total_checksum += checksum(&out);
            }
        }
        "meta_sortref" => {
            // Reference-only: sort-based tau (O(n log n)), same tau value and
            // output as "meta" -- kept for direct before/after cost comparison
            // against the production Quickselect path, not itself a
            // recommendation (~56% of cycles at N=8192 go to the sort --
            // see JOURNAL.md's perf-record finding).
            let cfg = MetaConfig { head_dim: 64, head_dim_v: 64, n_heads: 8, block: 64, w_blocks: 1, quantile: 0.90, tau_mode: TauMode::SortBased };
            let q = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim, 1);
            let k = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim, 2);
            let v = randn_tensor(cfg.n_heads, seq_len, cfg.head_dim_v, 3);
            for _ in 0..iterations {
                let out = monarch_meta_threshold_fast_residual(&q, &k, &v, &cfg);
                total_checksum += checksum(&out);
            }
        }
        other => {
            eprintln!("unknown kernel: {other} (expected causal|sliding|meta|meta_sortref)");
            std::process::exit(2);
        }
    }

    let elapsed = start.elapsed();
    println!(
        "kernel={kernel} seq_len={seq_len} iterations={iterations} elapsed={elapsed:?} checksum={total_checksum:e}"
    );
}
