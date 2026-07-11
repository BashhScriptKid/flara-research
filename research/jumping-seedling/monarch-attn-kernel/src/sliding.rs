//! SlidingMonarchAttention: exact local window + Monarch block
//! representative (refined over T-1 Sinkhorn-style cross-block
//! iterations) for the far/pre-window region. Faithful scalar port of
//! `../monarch-attn-causal/ma_sliding_monarch.py` -- see that file's
//! module docstring for the full design rationale. Single-head-group
//! (q/k/v share the same head count H), matching the validated reference;
//! GQA is not modeled here since the reference algorithm predates it.
//!
//! T-iteration was later found "structurally superseded" once threshold
//! selection (see `meta.rs`) reads real keys directly -- this kernel
//! exists purely as the empirical comparison baseline for that finding,
//! and for the decoy-pressure damping trade-off documented in
//! ROOFLINE_5500U.md's limitations section.

const NEG_INF: f32 = f32::NEG_INFINITY;

#[derive(Clone, Copy, Debug)]
pub struct SlidingConfig {
    pub head_dim: usize,
    pub n_heads: usize,
    pub block: usize,
    pub w_blocks: usize,
    pub t: usize,
    pub w_refine: usize,
}

/// Per-head block-structured buffer: `[m][b][d]` flattened as `m*b_size*dim + b*dim + d`.
struct Blocks {
    data: Vec<f32>,
    m: usize,
    b: usize,
    dim: usize,
}

impl Blocks {
    fn zeros(m: usize, b: usize, dim: usize) -> Self {
        Self { data: vec![0.0; m * b * dim], m, b, dim }
    }
    #[inline]
    fn row(&self, m: usize, b: usize) -> &[f32] {
        let base = (m * self.b + b) * self.dim;
        &self.data[base..base + self.dim]
    }
    #[inline]
    fn row_mut(&mut self, m: usize, b: usize) -> &mut [f32] {
        let base = (m * self.b + b) * self.dim;
        &mut self.data[base..base + self.dim]
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::dot(a, b)
}

/// `_local_pass`: within-block attention of `ar` over `k`, normalized by
/// `cr`, masked only by `valid_mb` (padding, never causal -- the window
/// owns all self-visibility). Returns `(al, cl)`; `r` is not retained
/// (only used transiently, matching the reference which discards it too
/// except in `_local_pass_with_v`).
fn local_pass(
    ar: &Blocks,
    k: &Blocks,
    cr: &[Vec<f32>],
    sm_scale: f32,
    valid_mb: &[Vec<bool>],
    eps: f32,
) -> (Blocks, Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>) {
    let (m_count, b_size, dim) = (ar.m, ar.b, ar.dim);
    let mut al = Blocks::zeros(m_count, b_size, dim);
    let mut cl = vec![vec![0.0f32; b_size]; m_count];
    let mut r_all = vec![vec![vec![0.0f32; b_size]; b_size]; m_count]; // r_all[m][i][j]

    for m in 0..m_count {
        for i in 0..b_size {
            let ar_row = ar.row(m, i);
            let mut scores = vec![0.0f32; b_size];
            let mut row_max = NEG_INF;
            for j in 0..b_size {
                let raw = if valid_mb[m][j] {
                    sm_scale * dot(ar_row, k.row(m, j)) / (cr[m][i] + eps)
                } else {
                    NEG_INF
                };
                scores[j] = raw;
                if raw > row_max {
                    row_max = raw;
                }
            }
            if !row_max.is_finite() {
                row_max = eps;
            }
            let mut sum_exp = 0.0f32;
            let mut exp_scores = vec![0.0f32; b_size];
            for j in 0..b_size {
                let e = if scores[j].is_finite() { (scores[j] - row_max).exp() } else { 0.0 };
                exp_scores[j] = e;
                sum_exp += e;
            }
            let tiny = f32::MIN_POSITIVE;
            let mut cl_i = 0.0f32;
            let al_row = al.row_mut(m, i);
            for j in 0..b_size {
                let r_ij = (exp_scores[j] / (sum_exp + eps)).max(tiny);
                r_all[m][i][j] = r_ij;
                cl_i += r_ij * r_ij.ln();
                let k_row = k.row(m, j);
                crate::simd::axpy(al_row, sm_scale * r_ij, k_row);
            }
            cl[m][i] = cl_i;
        }
    }
    (al, cl, r_all)
}

fn local_pass_with_v(
    ar: &Blocks,
    k: &Blocks,
    v: &Blocks,
    cr: &[Vec<f32>],
    sm_scale: f32,
    valid_mb: &[Vec<bool>],
    eps: f32,
) -> (Blocks, Blocks, Vec<Vec<f32>>) {
    let (al, cl, r_all) = local_pass(ar, k, cr, sm_scale, valid_mb, eps);
    let (m_count, b_size, dv) = (ar.m, ar.b, v.dim);
    let mut y = Blocks::zeros(m_count, b_size, dv);
    for m in 0..m_count {
        for i in 0..b_size {
            let y_row = y.row_mut(m, i);
            for j in 0..b_size {
                let r_ij = r_all[m][i][j];
                let v_row = v.row(m, j);
                crate::simd::axpy(y_row, r_ij, v_row);
            }
        }
    }
    (al, y, cl)
}

/// q/k/v: `[n_heads, seq_len, head_dim]` (v may have a different last dim,
/// handled via `crate::HeadTensor`). Returns `[n_heads, seq_len, head_dim_v]`.
pub fn sliding_monarch_causal(
    q: &crate::HeadTensor,
    k: &crate::HeadTensor,
    v: &crate::HeadTensor,
    cfg: &SlidingConfig,
) -> crate::HeadTensor {
    let eps = 1e-6f32;
    let seq_len = q.seq_len;
    let dim = cfg.head_dim;
    let dv = v.head_dim;
    let b_size = cfg.block;
    let m_count = seq_len.div_ceil(b_size);
    let n_padded = m_count * b_size;
    let sm_scale = 1.0f32 / (dim as f32).sqrt();

    let valid_mb: Vec<Vec<bool>> =
        (0..m_count).map(|m| (0..b_size).map(|b| m * b_size + b < seq_len).collect()).collect();

    let mut out = crate::HeadTensor::zeros(cfg.n_heads, seq_len, dv);

    for h in 0..cfg.n_heads {
        // pack this head's q/k/v into block-structured buffers (zero-padded)
        let mut qb = Blocks::zeros(m_count, b_size, dim);
        let mut kb = Blocks::zeros(m_count, b_size, dim);
        let mut vb = Blocks::zeros(m_count, b_size, dv);
        for m in 0..m_count {
            for b in 0..b_size {
                let pos = m * b_size + b;
                if pos < seq_len {
                    qb.row_mut(m, b).copy_from_slice(q.row(h, pos));
                    kb.row_mut(m, b).copy_from_slice(k.row(h, pos));
                    vb.row_mut(m, b).copy_from_slice(v.row(h, pos));
                }
            }
        }

        // ---- far/Monarch branch: T-1 cross-block refinement iterations ----
        let mut ar = Blocks::zeros(m_count, b_size, dim);
        ar.data.copy_from_slice(&qb.data);
        let mut cr: Vec<Vec<f32>> = vec![vec![1.0f32; b_size]; m_count];

        for _ in 0..cfg.t.saturating_sub(1) {
            let (al, cl, _r) = local_pass(&ar, &kb, &cr, sm_scale, &valid_mb, eps);

            // l_hat[i][mk][mq] = dot(al[mk][i], qb[mq][i]) - cl[mk][i], masked by mk <= mq - w_refine
            let mut l = vec![vec![vec![0.0f32; m_count]; m_count]; b_size]; // l[i][mk][mq]
            for i in 0..b_size {
                for mq in 0..m_count {
                    let mut col = vec![NEG_INF; m_count];
                    for mk in 0..m_count {
                        if mk as isize <= mq as isize - cfg.w_refine as isize {
                            col[mk] = dot(al.row(mk, i), qb.row(mq, i)) - cl[mk][i];
                        }
                    }
                    let mut row_max = col.iter().cloned().fold(NEG_INF, f32::max);
                    if !row_max.is_finite() {
                        row_max = 0.0;
                    }
                    let mut exp_col = vec![0.0f32; m_count];
                    let mut sum_exp = 0.0f32;
                    for mk in 0..m_count {
                        let e = if col[mk].is_finite() { (col[mk] - row_max).exp() } else { 0.0 };
                        exp_col[mk] = e;
                        sum_exp += e;
                    }
                    for mk in 0..m_count {
                        l[i][mk][mq] = exp_col[mk] / (sum_exp + eps);
                    }
                }
            }

            // cr_next[mk][i] = sum_mq l[i][mk][mq]; ar_next[mk][i] = sum_mq l[i][mk][mq]*qb[mq][i]
            let mut cr_next = vec![vec![0.0f32; b_size]; m_count];
            let mut ar_next = Blocks::zeros(m_count, b_size, dim);
            for i in 0..b_size {
                for mk in 0..m_count {
                    let mut s = 0.0f32;
                    let ar_row = ar_next.row_mut(mk, i);
                    for mq in 0..m_count {
                        let w = l[i][mk][mq];
                        s += w;
                        let q_row = qb.row(mq, i);
                        crate::simd::axpy(ar_row, w, q_row);
                    }
                    cr_next[mk][i] = s;
                }
            }
            ar = ar_next;
            cr = cr_next;
        }

        let (al_full, y_full, cl_full) = local_pass_with_v(&ar, &kb, &vb, &cr, sm_scale, &valid_mb, eps);

        // far logits for the final combination: l_hat_far[i][mq][mk], mask mk <= mq - w_blocks
        let mut l_hat_far = vec![vec![vec![NEG_INF; m_count]; m_count]; b_size]; // [i][mq][mk]
        for i in 0..b_size {
            for mq in 0..m_count {
                for mk in 0..m_count {
                    if mk as isize <= mq as isize - cfg.w_blocks as isize {
                        l_hat_far[i][mq][mk] = dot(qb.row(mq, i), al_full.row(mk, i)) - cl_full[mk][i];
                    }
                }
            }
        }

        // ---- local window branch + joint softmax combination ----
        for m_q in 0..m_count {
            let w_start = m_q.saturating_sub(cfg.w_blocks - 1);
            let n_win_blocks = m_q - w_start + 1;
            let win_len = n_win_blocks * b_size;

            for i in 0..b_size {
                let q_row = qb.row(m_q, i);

                let mut local_scores = vec![NEG_INF; win_len];
                for (wb, m_k) in (w_start..=m_q).enumerate() {
                    for b in 0..b_size {
                        let pos = m_k * b_size + b;
                        if pos >= seq_len {
                            continue;
                        }
                        let causal_ok = m_k < m_q || (m_k == m_q && b <= i);
                        if causal_ok {
                            local_scores[wb * b_size + b] = sm_scale * dot(q_row, kb.row(m_k, b));
                        }
                    }
                }

                let far_scores = &l_hat_far[i][m_q]; // len m_count

                let combined_max = local_scores
                    .iter()
                    .chain(far_scores.iter())
                    .cloned()
                    .fold(NEG_INF, f32::max);
                let row_max = if combined_max.is_finite() { combined_max } else { 0.0 };

                let mut sum_exp = 0.0f32;
                let mut local_w = vec![0.0f32; win_len];
                for j in 0..win_len {
                    let e = if local_scores[j].is_finite() { (local_scores[j] - row_max).exp() } else { 0.0 };
                    local_w[j] = e;
                    sum_exp += e;
                }
                let mut far_w = vec![0.0f32; m_count];
                for mk in 0..m_count {
                    let e = if far_scores[mk].is_finite() { (far_scores[mk] - row_max).exp() } else { 0.0 };
                    far_w[mk] = e;
                    sum_exp += e;
                }
                let denom = sum_exp + eps;

                let pos = m_q * b_size + i;
                if pos >= seq_len {
                    continue;
                }
                let out_row = out.row_mut(h, pos);
                for (wb, m_k) in (w_start..=m_q).enumerate() {
                    for b in 0..b_size {
                        let w = local_w[wb * b_size + b] / denom;
                        if w == 0.0 {
                            continue;
                        }
                        let v_row = vb.row(m_k, b);
                        crate::simd::axpy(out_row, w, v_row);
                    }
                }
                for mk in 0..m_count {
                    let w = far_w[mk] / denom;
                    if w == 0.0 {
                        continue;
                    }
                    let y_row = y_full.row(mk, i);
                    crate::simd::axpy(out_row, w, y_row);
                }
            }
        }
    }

    let _ = n_padded;
    out
}
