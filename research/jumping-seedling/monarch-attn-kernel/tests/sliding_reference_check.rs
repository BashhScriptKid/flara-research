//! Cross-validates the scalar Rust SlidingMonarchAttention port against
//! the real, validated PyTorch reference (`ma_sliding_monarch.py`), not a
//! reimplementation from memory. Regenerate with
//! `python export_sliding_vectors.py` if this config changes.

use monarch_attn_kernel::sliding::{sliding_monarch_causal, SlidingConfig};
use monarch_attn_kernel::HeadTensor;
use std::path::PathBuf;

fn read_f32(path: PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!("failed to read {path:?}: {e} -- run `python export_sliding_vectors.py` first")
    });
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn testdata(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata").join(name)
}

#[test]
fn matches_pytorch_sliding_monarch_reference() {
    let cfg = SlidingConfig { head_dim: 16, n_heads: 2, block: 8, w_blocks: 2, t: 3, w_refine: 2 };
    let seq_len = 37;

    let q = HeadTensor {
        data: read_f32(testdata("sliding_q.bin")),
        n_heads: cfg.n_heads,
        seq_len,
        head_dim: cfg.head_dim,
    };
    let k = HeadTensor {
        data: read_f32(testdata("sliding_k.bin")),
        n_heads: cfg.n_heads,
        seq_len,
        head_dim: cfg.head_dim,
    };
    let v = HeadTensor {
        data: read_f32(testdata("sliding_v.bin")),
        n_heads: cfg.n_heads,
        seq_len,
        head_dim: cfg.head_dim,
    };
    let expected = read_f32(testdata("sliding_out_ref.bin"));

    let out = sliding_monarch_causal(&q, &k, &v, &cfg);

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
        "max abs diff vs PyTorch SlidingMonarchAttention reference: {max_abs_diff:e} (at flat idx {worst_idx})"
    );
    assert!(
        max_abs_diff < 1e-3,
        "Rust SlidingMonarchAttention diverges from PyTorch reference: max abs diff = {max_abs_diff:e}"
    );
}
