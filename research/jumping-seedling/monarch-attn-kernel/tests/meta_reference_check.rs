//! Cross-validates the scalar Rust MetaMonarchAttention (fast-residual)
//! port against the real, validated PyTorch reference
//! (`ma_meta_threshold_fast_residual.py`). Regenerate with
//! `python export_meta_vectors.py` if this config changes.

use monarch_attn_kernel::meta::{monarch_meta_threshold_fast_residual, MetaConfig, TauMode};
use monarch_attn_kernel::HeadTensor;
use std::path::PathBuf;

fn read_f32(path: PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!("failed to read {path:?}: {e} -- run `python export_meta_vectors.py` first")
    });
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn testdata(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata").join(name)
}

#[test]
fn matches_pytorch_meta_threshold_reference() {
    let cfg = MetaConfig {
        head_dim: 16,
        head_dim_v: 16,
        n_heads: 2,
        block: 8,
        w_blocks: 1,
        quantile: 0.90,
        tau_mode: TauMode::SortBased,
    };
    let seq_len = 37;

    let q = HeadTensor {
        data: read_f32(testdata("meta_q.bin")),
        n_heads: cfg.n_heads,
        seq_len,
        head_dim: cfg.head_dim,
    };
    let k = HeadTensor {
        data: read_f32(testdata("meta_k.bin")),
        n_heads: cfg.n_heads,
        seq_len,
        head_dim: cfg.head_dim,
    };
    let v = HeadTensor {
        data: read_f32(testdata("meta_v.bin")),
        n_heads: cfg.n_heads,
        seq_len,
        head_dim: cfg.head_dim_v,
    };
    let expected = read_f32(testdata("meta_out_ref.bin"));

    let out = monarch_meta_threshold_fast_residual(&q, &k, &v, &cfg);

    assert_eq!(out.data.len(), expected.len());
    let mut max_abs_diff = 0.0f32;
    let mut worst_idx = 0;
    for (idx, (a, b)) in out.data.iter().zip(expected.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
            worst_idx = idx;
        }
    }
    println!(
        "max abs diff vs PyTorch MetaMonarchAttention (fast-residual) reference: {max_abs_diff:e} (at flat idx {worst_idx})"
    );
    assert!(
        max_abs_diff < 1e-3,
        "Rust MetaMonarchAttention diverges from PyTorch reference: max abs diff = {max_abs_diff:e}"
    );
}

/// Quickselect-based tau must produce the mathematically IDENTICAL result
/// to sort-based tau (same linear-interpolation quantile, different
/// algorithm) -- verifying this directly rather than assuming it, before
/// trusting any timing comparison between the two.
#[test]
fn quickselect_tau_matches_sort_based_tau() {
    let seq_len = 37;
    let base_cfg = MetaConfig {
        head_dim: 16,
        head_dim_v: 16,
        n_heads: 2,
        block: 8,
        w_blocks: 1,
        quantile: 0.90,
        tau_mode: TauMode::SortBased,
    };
    let q = HeadTensor { data: read_f32(testdata("meta_q.bin")), n_heads: base_cfg.n_heads, seq_len, head_dim: base_cfg.head_dim };
    let k = HeadTensor { data: read_f32(testdata("meta_k.bin")), n_heads: base_cfg.n_heads, seq_len, head_dim: base_cfg.head_dim };
    let v = HeadTensor { data: read_f32(testdata("meta_v.bin")), n_heads: base_cfg.n_heads, seq_len, head_dim: base_cfg.head_dim_v };

    let out_sort = monarch_meta_threshold_fast_residual(&q, &k, &v, &base_cfg);
    let quickselect_cfg = MetaConfig { tau_mode: TauMode::Quickselect, ..base_cfg };
    let out_qs = monarch_meta_threshold_fast_residual(&q, &k, &v, &quickselect_cfg);

    let mut max_abs_diff = 0.0f32;
    for (a, b) in out_sort.data.iter().zip(out_qs.data.iter()) {
        max_abs_diff = max_abs_diff.max((a - b).abs());
    }
    println!("max abs diff, quickselect tau vs sort-based tau: {max_abs_diff:e}");
    assert!(
        max_abs_diff < 1e-5,
        "Quickselect tau diverges from sort-based tau: max abs diff = {max_abs_diff:e}"
    );
}
