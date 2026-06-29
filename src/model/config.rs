//! Model configuration — the Fydel Jumping Seedling spec as a validated struct.
//!
//! Everything downstream (`layer`, `model`, `train`) reads its dimensions from
//! here. The defaults are the production 1B spec; tests build small instances by
//! overriding fields. All weight tensors except the tied embedding are
//! circular-basis compressed, so `dict_k` is the compression dial — the stored
//! footprint scales with it while the logical (decompressed-equivalent) param
//! count, the "~1B", does not.

use crate::kernels::ffn::FfnConfig;

/// Which attention kernel a layer runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum AttnKind {
    /// Full causal attention (FlashAttention-2) — global context.
    Full,
    /// Causal sliding-window attention — bandwidth-bounded local context.
    Sliding,
}

/// Full model configuration.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    pub n_layers: usize,
    pub hidden: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Tokenizer vocabulary size.
    pub vocab: usize,
    /// Tie the input embedding and the output (LM-head) projection.
    pub tied_embeddings: bool,
    /// FFN intermediate dimension.
    pub ffn_dim: usize,
    /// Circular-basis block size `b` (shared by all compressed matmuls).
    pub block: usize,
    /// Number of FFN micro-blocks routed active per token (top-k).
    pub n_active: usize,
    /// Dictionary atoms `K` (the compression dial).
    pub dict_k: usize,
    /// Attention key/value tile length.
    pub kv_block: usize,
    /// Sliding-window width for the `Sliding` layers.
    pub window: usize,
    /// Layers with index `< full_attn_layers` use `Full`, the rest `Sliding`.
    pub full_attn_layers: usize,
    pub rope_base: f32,
    pub max_seq: usize,
    pub norm_eps: f32,
}

impl Default for ModelConfig {
    /// The production Fydel-1B spec.
    fn default() -> Self {
        Self {
            n_layers: 96,
            hidden: 896,
            n_q_heads: 14,
            n_kv_heads: 2,
            head_dim: 64,
            vocab: 32768,
            tied_embeddings: true,
            ffn_dim: 3072,
            block: 64,
            n_active: 12,
            dict_k: 32,
            kv_block: 64,
            window: 256,
            full_attn_layers: 24,
            rope_base: 10000.0,
            max_seq: 1024,
            norm_eps: 1e-5,
        }
    }
}

impl ModelConfig {
    /// Total width of the query projection (`n_q_heads · head_dim`).
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.n_q_heads * self.head_dim
    }

    /// Total width of each of the key / value projections (`n_kv_heads · head_dim`).
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// GQA group size: query heads sharing one KV head.
    #[inline]
    pub fn group(&self) -> usize {
        self.n_q_heads / self.n_kv_heads
    }

    /// Attention kind for a given layer index.
    #[inline]
    pub fn attn_kind(&self, layer: usize) -> AttnKind {
        if layer < self.full_attn_layers { AttnKind::Full } else { AttnKind::Sliding }
    }

    /// Depth-scaled residual factor `1/√(2·n_layers)` — keeps the residual stream
    /// variance bounded across 96 layers (applied to each sub-layer's output).
    #[inline]
    pub fn residual_scale(&self) -> f32 {
        1.0 / ((2 * self.n_layers) as f32).sqrt()
    }

    /// The FFN sub-config derived from this model config.
    pub fn ffn_config(&self) -> FfnConfig {
        FfnConfig {
            hidden: self.hidden,
            ffn: self.ffn_dim,
            block: self.block,
            n_active: self.n_active,
            dict_k: self.dict_k,
        }
    }

    /// Validate the internal consistency the compressed kernels rely on. Panics
    /// with a specific message on the first violation.
    pub fn validate(&self) {
        let b = self.block;
        assert!(self.n_q_heads % self.n_kv_heads == 0, "n_q_heads must be a multiple of n_kv_heads");
        assert!(self.full_attn_layers <= self.n_layers, "full_attn_layers exceeds n_layers");
        for (name, dim) in
            [("hidden", self.hidden), ("q_dim", self.q_dim()), ("kv_dim", self.kv_dim()), ("ffn_dim", self.ffn_dim)]
        {
            assert!(dim % b == 0, "{name} ({dim}) must be divisible by block ({b})");
        }
        assert!(self.n_active <= self.ffn_dim / b, "n_active exceeds the number of FFN blocks");
        assert!(self.head_dim % 2 == 0, "head_dim must be even for RoPE");
        assert!(self.window > 0 && self.kv_block > 0, "window and kv_block must be positive");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_spec_is_valid() {
        ModelConfig::default().validate();
    }

    #[test]
    fn derived_dims_match_spec() {
        let c = ModelConfig::default();
        assert_eq!(c.q_dim(), 896);
        assert_eq!(c.kv_dim(), 128);
        assert_eq!(c.group(), 7);
        assert_eq!(c.attn_kind(0), AttnKind::Full);
        assert_eq!(c.attn_kind(23), AttnKind::Full);
        assert_eq!(c.attn_kind(24), AttnKind::Sliding);
        assert_eq!(c.attn_kind(95), AttnKind::Sliding);
        assert!((c.residual_scale() - 1.0 / (192.0f32).sqrt()).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "divisible by block")]
    fn rejects_indivisible_dims() {
        let mut c = ModelConfig::default();
        c.hidden = 900; // not a multiple of 64
        c.validate();
    }
}
