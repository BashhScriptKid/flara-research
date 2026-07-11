//! Cross-validates the scalar Rust dense-causal-GQA implementation against
//! a real PyTorch `scaled_dot_product_attention(is_causal=True)` reference,
//! at the production config (head_dim=64, n_q_heads=14, n_kv_heads=2 GQA),
//! seq_len=37 (deliberately not block-aligned). Regenerate the reference
//! vectors with `python export_reference_vectors.py` if this config changes.

use monarch_attn_kernel::causal::dense_causal_attention;
use monarch_attn_kernel::{AttnConfig, HeadTensor};
use std::path::PathBuf;

fn read_f32(path: PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "failed to read {path:?}: {e} -- run `python export_reference_vectors.py` first"
        )
    });
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn testdata(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata").join(name)
}

#[test]
fn matches_pytorch_dense_causal_gqa_reference() {
    let cfg = AttnConfig { head_dim: 64, n_q_heads: 14, n_kv_heads: 2 };
    let seq_len = 37;

    let q = HeadTensor {
        data: read_f32(testdata("causal_q.bin")),
        n_heads: cfg.n_q_heads,
        seq_len,
        head_dim: cfg.head_dim,
    };
    let k = HeadTensor {
        data: read_f32(testdata("causal_k.bin")),
        n_heads: cfg.n_kv_heads,
        seq_len,
        head_dim: cfg.head_dim,
    };
    let v = HeadTensor {
        data: read_f32(testdata("causal_v.bin")),
        n_heads: cfg.n_kv_heads,
        seq_len,
        head_dim: cfg.head_dim,
    };
    let expected = read_f32(testdata("causal_out_ref.bin"));

    let out = dense_causal_attention(&q, &k, &v, &cfg);

    assert_eq!(out.data.len(), expected.len());
    let mut max_abs_diff = 0.0f32;
    for (a, b) in out.data.iter().zip(expected.iter()) {
        max_abs_diff = max_abs_diff.max((a - b).abs());
    }
    println!("max abs diff vs PyTorch SDPA reference: {max_abs_diff:e}");
    assert!(
        max_abs_diff < 1e-3,
        "Rust dense causal attention diverges from PyTorch reference: max abs diff = {max_abs_diff:e}"
    );
}
