//! CausalMonarchAttention: the original Monarch-family causal-masking
//! building block (dual-representative trick), predating
//! SlidingMonarchAttention. Faithful scalar port of
//! `../monarch-attn-causal/ma_causal_dual_opt.py`.
//!
//! Structurally different from Sliding: there is NO exact local window
//! here at all. EVERY block, including the query's own/diagonal block,
//! is read via a T-iteration-refined Monarch representative -- the
//! diagonal block just gets a cheaper, causally-masked "own-block"
//! representative (`al_c`) used as a fast elementwise diagonal path,
//! while every block (including the diagonal one) also builds a
//! non-causal "full" representative (`al_f`) for reuse by strictly
//! later query blocks. Two representatives per block because a single
//! diagonal block has two conflicting visibility requirements: serve
//! its own causal self-attention, AND be reused unmasked by every later
//! block -- Sliding's docstring frames its own design as moving past
//! this specific trick once a real local window handles all
//! self-visibility, leaving only ONE far representative needed. This
//! kernel predates that move: it compresses everything, with no real
//! per-key window anywhere.

const NEG_INF: f32 = f32::NEG_INFINITY;

#[derive(Clone, Copy, Debug)]
pub struct CausalMonarchConfig {
    pub head_dim: usize,
    pub n_heads: usize,
    pub block: usize,
    pub t: usize,
}

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
/// `cr`. `causal_mask`: if true, key index j is only visible to
/// representative slot i when j <= i (own-block causal); if false,
/// every valid j is visible (the "full" reuse-by-later-blocks pass).
fn local_pass(
    ar: &Blocks,
    k: &Blocks,
    cr: &[Vec<f32>],
    sm_scale: f32,
    valid_mb: &[Vec<bool>],
    causal_mask: bool,
    eps: f32,
) -> (Blocks, Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>) {
    let (m_count, b_size, dim) = (ar.m, ar.b, ar.dim);
    let mut al = Blocks::zeros(m_count, b_size, dim);
    let mut cl = vec![vec![0.0f32; b_size]; m_count];
    let mut r_all = vec![vec![vec![0.0f32; b_size]; b_size]; m_count];

    for m in 0..m_count {
        for i in 0..b_size {
            let ar_row = ar.row(m, i);
            let mut scores = vec![0.0f32; b_size];
            let mut row_max = NEG_INF;
            for j in 0..b_size {
                let visible = valid_mb[m][j] && (!causal_mask || j <= i);
                let raw = if visible {
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
    causal_mask: bool,
    eps: f32,
) -> (Blocks, Blocks, Vec<Vec<f32>>) {
    let (al, cl, r_all) = local_pass(ar, k, cr, sm_scale, valid_mb, causal_mask, eps);
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

pub fn causal_monarch_attention(
    q: &crate::HeadTensor,
    k: &crate::HeadTensor,
    v: &crate::HeadTensor,
    cfg: &CausalMonarchConfig,
) -> crate::HeadTensor {
    let eps = 1e-6f32;
    let seq_len = q.seq_len;
    let dim = cfg.head_dim;
    let dv = v.head_dim;
    let b_size = cfg.block;
    let m_count = seq_len.div_ceil(b_size);
    let sm_scale = 1.0f32 / (dim as f32).sqrt();

    let valid_mb: Vec<Vec<bool>> =
        (0..m_count).map(|m| (0..b_size).map(|b| m * b_size + b < seq_len).collect()).collect();

    let mut out = crate::HeadTensor::zeros(cfg.n_heads, seq_len, dv);

    for h in 0..cfg.n_heads {
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

        let mut ar = Blocks::zeros(m_count, b_size, dim);
        ar.data.copy_from_slice(&qb.data);
        let mut cr: Vec<Vec<f32>> = vec![vec![1.0f32; b_size]; m_count];

        for _ in 0..cfg.t.saturating_sub(1) {
            let (al_c, cl_c, _) = local_pass(&ar, &kb, &cr, sm_scale, &valid_mb, true, eps);
            let (al_f, cl_f, _) = local_pass(&ar, &kb, &cr, sm_scale, &valid_mb, false, eps);

            // l[i][mk][mq]: diagonal (mk==mq) uses al_c/cl_c elementwise;
            // off-diagonal (mk<mq, strict) uses al_f/cl_f competitive scoring.
            let mut l = vec![vec![vec![0.0f32; m_count]; m_count]; b_size];
            for i in 0..b_size {
                for mq in 0..m_count {
                    let mut col = vec![NEG_INF; m_count];
                    for mk in 0..m_count {
                        if mk == mq {
                            col[mk] = dot(al_c.row(mk, i), qb.row(mq, i)) - cl_c[mk][i];
                        } else if mk < mq {
                            col[mk] = dot(al_f.row(mk, i), qb.row(mq, i)) - cl_f[mk][i];
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

        let (al_c, y_c, cl_c) = local_pass_with_v(&ar, &kb, &vb, &cr, sm_scale, &valid_mb, true, eps);
        let (al_f, y_f, cl_f) = local_pass_with_v(&ar, &kb, &vb, &cr, sm_scale, &valid_mb, false, eps);

        for mq in 0..m_count {
            for i in 0..b_size {
                let pos = mq * b_size + i;
                if pos >= seq_len {
                    continue;
                }
                let q_row = qb.row(mq, i);

                let mut col = vec![NEG_INF; m_count];
                for mk in 0..m_count {
                    if mk == mq {
                        col[mk] = dot(q_row, al_c.row(mq, i)) - cl_c[mq][i];
                    } else if mk < mq {
                        col[mk] = dot(q_row, al_f.row(mk, i)) - cl_f[mk][i];
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
                let denom = sum_exp + eps;

                let out_row = out.row_mut(h, pos);
                for mk in 0..m_count {
                    let w = exp_col[mk] / denom;
                    if w == 0.0 {
                        continue;
                    }
                    if mk == mq {
                        crate::simd::axpy(out_row, w, y_c.row(mq, i));
                    } else if mk < mq {
                        crate::simd::axpy(out_row, w, y_f.row(mk, i));
                    }
                }
            }
        }
    }

    out
}
