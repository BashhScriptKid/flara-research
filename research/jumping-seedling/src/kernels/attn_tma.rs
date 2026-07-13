//! Forward-only profiling shim for TauMonarchAttention (`meta.rs` in the
//! standalone `monarch-attn-kernel` crate), wired to run inside the real
//! model for wall-clock comparison against [`FlashAttention`](super::attn_flash::FlashAttention).
//!
//! `monarch-attn-kernel`'s validated kernel assumes MHA (one shared head
//! count for Q/K/V) and a `[n_heads, seq_len, head_dim]` `HeadTensor`
//! layout, while the model's GQA buffers are token-major `[T, n_heads*d]`
//! with fewer KV heads than Q heads. This shim transposes buffers into
//! `HeadTensor` layout and broadcasts each KV head across its query group
//! (correctness-preserving, not how a real GQA-aware kernel would do it --
//! fine for a forward-latency measurement, not for production).
//!
//! Inference/profiling only: there is no backward pass for TMA yet (the
//! tier/threshold selection is non-differentiable as implemented), so
//! [`TmaAttention::forward`] returns a zeroed LSE placeholder that must
//! never be fed to [`FlashAttention::backward`](super::attn_flash::FlashAttention::backward).

use monarch_attn_kernel::meta::{monarch_meta_threshold_fast_residual, MetaConfig, TauMode};
use monarch_attn_kernel::HeadTensor;

pub struct TmaAttention {
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    group: usize,
    cfg: MetaConfig,
}

impl TmaAttention {
    pub fn new(n_q_heads: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        assert!(n_q_heads % n_kv_heads == 0, "n_q_heads must be a multiple of n_kv_heads");
        Self {
            n_q_heads,
            n_kv_heads,
            head_dim,
            group: n_q_heads / n_kv_heads,
            cfg: MetaConfig {
                head_dim,
                head_dim_v: head_dim,
                n_heads: n_q_heads,
                block: 64,
                w_blocks: 1,
                quantile: 0.90,
                tau_mode: TauMode::Quickselect,
            },
        }
    }

    #[inline]
    pub fn seq_len(&self, q: &[f32]) -> usize {
        q.len() / (self.n_q_heads * self.head_dim)
    }

    /// Same signature shape as `FlashAttention::forward` for drop-in use in
    /// `AttnRunner`. The returned `Vec<f32>` is a zeroed LSE placeholder,
    /// not a real log-sum-exp -- forward-only profiling never reads it.
    pub fn forward(&self, q: &[f32], k: &[f32], v: &[f32], out: &mut [f32]) -> Vec<f32> {
        let (nq, nkv, d) = (self.n_q_heads, self.n_kv_heads, self.head_dim);
        let t = self.seq_len(q);
        assert_eq!(q.len(), t * nq * d, "q shape mismatch");
        assert_eq!(k.len(), t * nkv * d, "k shape mismatch");
        assert_eq!(v.len(), t * nkv * d, "v shape mismatch");
        assert_eq!(out.len(), q.len(), "out shape mismatch");

        // token-major [T, n_heads*d] -> HeadTensor [n_heads, T, d], broadcasting
        // KV heads across their query group so every one of the `nq` heads
        // gets its own (duplicated) K/V.
        let mut qh = HeadTensor::zeros(nq, t, d);
        let mut kh = HeadTensor::zeros(nq, t, d);
        let mut vh = HeadTensor::zeros(nq, t, d);
        for i in 0..t {
            for h_q in 0..nq {
                let h_kv = h_q / self.group;
                qh.row_mut(h_q, i).copy_from_slice(&q[(i * nq + h_q) * d..(i * nq + h_q) * d + d]);
                kh.row_mut(h_q, i).copy_from_slice(&k[(i * nkv + h_kv) * d..(i * nkv + h_kv) * d + d]);
                vh.row_mut(h_q, i).copy_from_slice(&v[(i * nkv + h_kv) * d..(i * nkv + h_kv) * d + d]);
            }
        }

        let oh = monarch_meta_threshold_fast_residual(&qh, &kh, &vh, &self.cfg);

        // HeadTensor [n_heads, T, d] -> token-major [T, n_heads*d]
        for i in 0..t {
            for h_q in 0..nq {
                out[(i * nq + h_q) * d..(i * nq + h_q) * d + d].copy_from_slice(oh.row(h_q, i));
            }
        }

        vec![0.0f32; t * nq]
    }
}
