//! Empirical Rust validation of Causal / Sliding / Meta MonarchAttention
//! against the analytical predictions in `../monarch-attn-causal/ROOFLINE_5500U.md`.
//!
//! Scalar-correct implementations first (validated against PyTorch reference
//! outputs), AVX2 versions follow once correctness is locked in. See that
//! document's own limitations section: every FLOP/byte/roofline number there
//! is an analytical estimate against public 5500U specs, never measured on
//! the actual chip -- this crate is the first real measurement.

pub mod causal;
pub mod meta;
pub mod simd;
pub mod sliding;

/// Head/dimension configuration. Defaults match Jumping Seedling's real
/// production config (`src/model/config.rs`) so benchmarks reflect the
/// actual target, not an arbitrary uniform-head approximation.
#[derive(Clone, Copy, Debug)]
pub struct AttnConfig {
    pub head_dim: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
}

impl AttnConfig {
    /// Jumping Seedling production defaults: head_dim=64, n_q_heads=14,
    /// n_kv_heads=2 (GQA, 7 query heads per KV head).
    pub fn production() -> Self {
        Self { head_dim: 64, n_q_heads: 14, n_kv_heads: 2 }
    }

    pub fn kv_group_size(&self) -> usize {
        self.n_q_heads / self.n_kv_heads
    }
}

/// Row-major tensor: `[n_heads, seq_len, head_dim]`.
#[derive(Clone, Debug)]
pub struct HeadTensor {
    pub data: Vec<f32>,
    pub n_heads: usize,
    pub seq_len: usize,
    pub head_dim: usize,
}

impl HeadTensor {
    pub fn zeros(n_heads: usize, seq_len: usize, head_dim: usize) -> Self {
        Self { data: vec![0.0; n_heads * seq_len * head_dim], n_heads, seq_len, head_dim }
    }

    #[inline]
    pub fn row(&self, head: usize, pos: usize) -> &[f32] {
        let base = (head * self.seq_len + pos) * self.head_dim;
        &self.data[base..base + self.head_dim]
    }

    #[inline]
    pub fn row_mut(&mut self, head: usize, pos: usize) -> &mut [f32] {
        let base = (head * self.seq_len + pos) * self.head_dim;
        &mut self.data[base..base + self.head_dim]
    }
}
