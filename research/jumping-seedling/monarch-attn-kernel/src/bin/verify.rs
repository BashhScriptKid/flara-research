//! Standalone correctness-verification binary. Runs the SAME checks as
//! `tests/*_reference_check.rs`, but as a binary so it can be built and
//! run in `--release` mode (LTO, opt-level=3, codegen-units=1) -- the
//! exact build profile used for benchmarking/profiling. This exists
//! because `cargo test` always uses the `test` profile, never `release`,
//! so it cannot by itself confirm release-mode codegen preserves
//! correctness. Rust does not reorder float ops by default (no fast-math
//! flags used here), so this is expected to match exactly -- verifying
//! that expectation rather than assuming it, per this whole project's
//! standing discipline.
//!
//! Exit code 0 = all checks passed, non-zero = at least one failed
//! (prints which, and by how much) -- suitable for CI/script use.

use monarch_attn_kernel::causal::dense_causal_attention;
use monarch_attn_kernel::meta::{monarch_meta_threshold_fast_residual, MetaConfig, TauMode};
use monarch_attn_kernel::sliding::{sliding_monarch_causal, SlidingConfig};
use monarch_attn_kernel::{AttnConfig, HeadTensor};
use std::path::{Path, PathBuf};

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("failed to read {path:?}: {e} -- run the export_*.py scripts first"));
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn testdata(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata").join(name)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut max_d = 0.0f32;
    let mut worst = 0;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs();
        if d > max_d {
            max_d = d;
            worst = i;
        }
    }
    (max_d, worst)
}

fn check(name: &str, max_d: f32, threshold: f32, worst_idx: usize, ok: &mut bool) {
    let pass = max_d < threshold && max_d.is_finite();
    println!(
        "[{}] {name}: max abs diff = {max_d:e} (threshold {threshold:e}, worst @ idx {worst_idx})",
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        *ok = false;
    }
}

fn main() {
    let mut all_ok = true;

    // --- Causal ---
    {
        let cfg = AttnConfig { head_dim: 64, n_q_heads: 14, n_kv_heads: 2 };
        let seq_len = 37;
        let q = HeadTensor { data: read_f32(&testdata("causal_q.bin")), n_heads: cfg.n_q_heads, seq_len, head_dim: cfg.head_dim };
        let k = HeadTensor { data: read_f32(&testdata("causal_k.bin")), n_heads: cfg.n_kv_heads, seq_len, head_dim: cfg.head_dim };
        let v = HeadTensor { data: read_f32(&testdata("causal_v.bin")), n_heads: cfg.n_kv_heads, seq_len, head_dim: cfg.head_dim };
        let expected = read_f32(&testdata("causal_out_ref.bin"));
        let out = dense_causal_attention(&q, &k, &v, &cfg);
        let (d, idx) = max_abs_diff(&out.data, &expected);
        check("causal (release build)", d, 1e-3, idx, &mut all_ok);
    }

    // --- Sliding ---
    {
        let cfg = SlidingConfig { head_dim: 16, n_heads: 2, block: 8, w_blocks: 2, t: 3, w_refine: 2 };
        let seq_len = 37;
        let q = HeadTensor { data: read_f32(&testdata("sliding_q.bin")), n_heads: cfg.n_heads, seq_len, head_dim: cfg.head_dim };
        let k = HeadTensor { data: read_f32(&testdata("sliding_k.bin")), n_heads: cfg.n_heads, seq_len, head_dim: cfg.head_dim };
        let v = HeadTensor { data: read_f32(&testdata("sliding_v.bin")), n_heads: cfg.n_heads, seq_len, head_dim: cfg.head_dim };
        let expected = read_f32(&testdata("sliding_out_ref.bin"));
        let out = sliding_monarch_causal(&q, &k, &v, &cfg);
        let (d, idx) = max_abs_diff(&out.data, &expected);
        check("sliding (release build)", d, 1e-3, idx, &mut all_ok);
    }

    // --- Meta ---
    {
        let cfg = MetaConfig { head_dim: 16, head_dim_v: 16, n_heads: 2, block: 8, w_blocks: 1, quantile: 0.90, tau_mode: TauMode::SortBased };
        let seq_len = 37;
        let q = HeadTensor { data: read_f32(&testdata("meta_q.bin")), n_heads: cfg.n_heads, seq_len, head_dim: cfg.head_dim };
        let k = HeadTensor { data: read_f32(&testdata("meta_k.bin")), n_heads: cfg.n_heads, seq_len, head_dim: cfg.head_dim };
        let v = HeadTensor { data: read_f32(&testdata("meta_v.bin")), n_heads: cfg.n_heads, seq_len, head_dim: cfg.head_dim_v };
        let expected = read_f32(&testdata("meta_out_ref.bin"));
        let out = monarch_meta_threshold_fast_residual(&q, &k, &v, &cfg);
        let (d, idx) = max_abs_diff(&out.data, &expected);
        check("meta (release build)", d, 1e-3, idx, &mut all_ok);
    }

    if all_ok {
        println!("\nAll kernels verified correct in this build profile.");
        std::process::exit(0);
    } else {
        eprintln!("\nAt least one kernel FAILED verification in this build profile.");
        std::process::exit(1);
    }
}
