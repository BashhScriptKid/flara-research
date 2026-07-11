//! Dense causal attention -- the baseline every other kernel in this crate
//! is measured against and correctness-checked relative to. Scalar,
//! portable, no SIMD yet. Supports GQA (n_kv_heads < n_q_heads).

use crate::{AttnConfig, HeadTensor};

/// q: `[n_q_heads, seq_len, head_dim]`, k/v: `[n_kv_heads, seq_len, head_dim]`.
/// Returns `[n_q_heads, seq_len, head_dim]`.
pub fn dense_causal_attention(
    q: &HeadTensor,
    k: &HeadTensor,
    v: &HeadTensor,
    cfg: &AttnConfig,
) -> HeadTensor {
    assert_eq!(q.n_heads, cfg.n_q_heads);
    assert_eq!(k.n_heads, cfg.n_kv_heads);
    assert_eq!(v.n_heads, cfg.n_kv_heads);
    assert_eq!(q.seq_len, k.seq_len);
    assert_eq!(q.seq_len, v.seq_len);
    assert_eq!(q.head_dim, cfg.head_dim);

    let seq_len = q.seq_len;
    let head_dim = cfg.head_dim;
    let group = cfg.kv_group_size();
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut out = HeadTensor::zeros(cfg.n_q_heads, seq_len, head_dim);

    for qh in 0..cfg.n_q_heads {
        let kvh = qh / group;
        for i in 0..seq_len {
            let q_row = q.row(qh, i);

            // scores over the causal prefix [0, i]
            let mut scores = vec![0.0f32; i + 1];
            let mut max_score = f32::NEG_INFINITY;
            for (j, s) in scores.iter_mut().enumerate() {
                let k_row = k.row(kvh, j);
                *s = crate::simd::dot(q_row, k_row) * scale;
                if *s > max_score {
                    max_score = *s;
                }
            }

            let mut sum_exp = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }

            let out_row = out.row_mut(qh, i);
            for j in 0..=i {
                let w = scores[j] / sum_exp;
                let v_row = v.row(kvh, j);
                crate::simd::axpy(out_row, w, v_row);
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity check independent of any external reference: a single-key
    /// sequence (seq_len=1) must return exactly that key's value, since
    /// softmax over one element is always weight 1.0.
    #[test]
    fn single_position_returns_its_own_value() {
        let cfg = AttnConfig { head_dim: 4, n_q_heads: 1, n_kv_heads: 1 };
        let q = HeadTensor { data: vec![1.0, 0.0, 0.0, 0.0], n_heads: 1, seq_len: 1, head_dim: 4 };
        let k = HeadTensor { data: vec![1.0, 0.0, 0.0, 0.0], n_heads: 1, seq_len: 1, head_dim: 4 };
        let v = HeadTensor { data: vec![1.0, 2.0, 3.0, 4.0], n_heads: 1, seq_len: 1, head_dim: 4 };
        let out = dense_causal_attention(&q, &k, &v, &cfg);
        assert_eq!(out.row(0, 0), &[1.0, 2.0, 3.0, 4.0]);
    }

    /// Causality: perturbing a future key must not change an earlier
    /// position's output at all (not approximately -- exactly, since
    /// dense causal attention never reads future positions).
    #[test]
    fn causal_no_leak_from_future_keys() {
        let cfg = AttnConfig { head_dim: 2, n_q_heads: 1, n_kv_heads: 1 };
        let seq_len = 5;
        let mut q = HeadTensor::zeros(1, seq_len, 2);
        let mut k = HeadTensor::zeros(1, seq_len, 2);
        let mut v = HeadTensor::zeros(1, seq_len, 2);
        for i in 0..seq_len {
            q.row_mut(0, i).copy_from_slice(&[1.0, i as f32 * 0.1]);
            k.row_mut(0, i).copy_from_slice(&[1.0, i as f32 * 0.1]);
            v.row_mut(0, i).copy_from_slice(&[i as f32, i as f32 * 2.0]);
        }
        let out_before = dense_causal_attention(&q, &k, &v, &cfg);

        k.row_mut(0, seq_len - 1).copy_from_slice(&[100.0, -100.0]);
        let out_after = dense_causal_attention(&q, &k, &v, &cfg);

        for i in 0..seq_len - 1 {
            assert_eq!(out_before.row(0, i), out_after.row(0, i), "position {i} leaked future key");
        }
    }
}
