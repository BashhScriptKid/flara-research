//! MetaMonarchAttention (final confirmed design): Fenwick dyadic tier
//! selection + shared-tau threshold selection + exact-algebraic
//! fast-residual centroid for non-survivors + exact local window. No
//! T-iteration (structurally superseded), no bounding-ball pruning
//! (confirmed dead on this key geometry -- see ROOFLINE_5500U.md).
//!
//! Faithful scalar port of
//! `../monarch-attn-causal/ma_meta_threshold_fast_residual.py`. Tau uses
//! sort-based quantile (matching `torch.quantile`'s linear-interpolation
//! default) -- this is the CORRECTNESS reference; the reservoir-sampling
//! cost optimization documented in ROOFLINE_5500U.md is a follow-up, not
//! yet implemented here.

const NEG_INF: f32 = f32::NEG_INFINITY;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TauMode {
    /// Full sort (`torch.quantile`'s own approach) -- the correctness
    /// reference this whole module was validated against. O(n log n).
    SortBased,
    /// `select_nth_unstable_by` (quickselect), O(n) average. Produces the
    /// mathematically IDENTICAL tau value to SortBased (same linear-
    /// interpolation quantile, just computed via partial reordering instead
    /// of a full sort) -- a real fix, not a diagnostic stand-in, per
    /// Fable's recommended experiment to isolate the sort algorithm's cost
    /// from the (unavoidable, and correctness-relevant) tau computation
    /// itself.
    Quickselect,
}

#[derive(Clone, Copy, Debug)]
pub struct MetaConfig {
    pub head_dim: usize,
    pub head_dim_v: usize,
    pub n_heads: usize,
    pub block: usize,
    pub w_blocks: usize,
    pub quantile: f32,
    pub tau_mode: TauMode,
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::dot(a, b)
}

/// `torch.quantile`'s default (linear interpolation) behavior on an
/// already-sorted slice.
fn quantile_linear_sorted(sorted: &[f32], q: f32) -> f32 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n as f32 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = pos - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Same linear-interpolation quantile as `quantile_linear_sorted`, but via
/// `select_nth_unstable_by` (quickselect, O(n) average) instead of a full
/// sort (O(n log n)) -- mathematically identical result, cheaper algorithm.
/// Mutates `pooled` in place (partial reordering, not a full sort).
fn quantile_linear_quickselect(pooled: &mut [f32], q: f32) -> f32 {
    let n = pooled.len();
    if n == 1 {
        return pooled[0];
    }
    let pos = q * (n as f32 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        let (_, v, _) = pooled.select_nth_unstable_by(lo, |a, b| a.partial_cmp(b).unwrap());
        return *v;
    }
    let (_, v_lo, right) = pooled.select_nth_unstable_by(lo, |a, b| a.partial_cmp(b).unwrap());
    let v_lo = *v_lo;
    // after partitioning at `lo`, index `hi = lo+1`'s value is the minimum
    // of the remaining right partition -- no second full selection needed
    let v_hi = right.iter().cloned().fold(f32::INFINITY, f32::min);
    let frac = pos - lo as f32;
    v_lo * (1.0 - frac) + v_hi * frac
}

pub fn monarch_meta_threshold_fast_residual(
    q: &crate::HeadTensor,
    k: &crate::HeadTensor,
    v: &crate::HeadTensor,
    cfg: &MetaConfig,
) -> crate::HeadTensor {
    let eps = 1e-6f32;
    let seq_len = q.seq_len;
    let dim = cfg.head_dim;
    let dv = cfg.head_dim_v;
    let b = cfg.block;
    let sm_scale = 1.0f32 / (dim as f32).sqrt();

    let m_base_needed = seq_len.div_ceil(b).max(2);
    let l_tiers = ((m_base_needed as f32).log2().ceil() as usize).max(1);
    let n_padded = b * (1usize << l_tiers);

    let mut out = crate::HeadTensor::zeros(cfg.n_heads, seq_len, dv);

    for h in 0..cfg.n_heads {
        // flat, zero-padded K/V for this head: index j (0..n_padded) -> row j*dim
        let mut k_flat = vec![0.0f32; n_padded * dim];
        let mut v_flat = vec![0.0f32; n_padded * dv];
        for pos in 0..seq_len {
            k_flat[pos * dim..(pos + 1) * dim].copy_from_slice(k.row(h, pos));
            v_flat[pos * dv..(pos + 1) * dv].copy_from_slice(v.row(h, pos));
        }
        let mut q_flat = vec![0.0f32; n_padded * dim];
        for pos in 0..seq_len {
            q_flat[pos * dim..(pos + 1) * dim].copy_from_slice(q.row(h, pos));
        }

        // ---- precompute per-tier full-block K/V sums, ONCE, query-independent ----
        // sum_k[l][block_idx][d], sum_v[l][block_idx][dv]
        let mut sum_k: Vec<Vec<Vec<f32>>> = Vec::with_capacity(l_tiers);
        let mut sum_v: Vec<Vec<Vec<f32>>> = Vec::with_capacity(l_tiers);
        for l in 0..l_tiers {
            let bl = b * (1usize << l);
            let ml = n_padded / bl;
            let mut sk = vec![vec![0.0f32; dim]; ml];
            let mut sv = vec![vec![0.0f32; dv]; ml];
            for blk in 0..ml {
                for j in 0..bl {
                    let pos = blk * bl + j;
                    crate::simd::axpy(&mut sk[blk], 1.0, &k_flat[pos * dim..(pos + 1) * dim]);
                    crate::simd::axpy(&mut sv[blk], 1.0, &v_flat[pos * dv..(pos + 1) * dv]);
                }
            }
            sum_k.push(sk);
            sum_v.push(sv);
        }

        let m_count = n_padded / b;
        let valid = |pos: usize| pos < seq_len;

        for m0 in 0..m_count {
            let w_start = m0.saturating_sub(cfg.w_blocks - 1);
            let n_win_blocks = m0 - w_start + 1;
            let win_len = n_win_blocks * b;

            let n_signed = m0 as isize - cfg.w_blocks as isize + 1;
            let mut candidates: Vec<(usize, usize)> = Vec::new(); // (l, block_idx)
            if n_signed > 0 {
                let n = n_signed as usize;
                for l in 0..l_tiers {
                    if (n >> l) & 1 == 1 {
                        candidates.push((l, (n >> (l + 1)) << 1));
                    }
                }
            }
            // drop out-of-range candidates (block_idx >= Ml for this tier)
            candidates.retain(|&(l, block_idx)| {
                let bl = b * (1usize << l);
                let ml = n_padded / bl;
                block_idx < ml
            });

            for i in 0..b {
                let pos = m0 * b + i;
                if !valid(pos) {
                    continue;
                }
                let q_row = &q_flat[pos * dim..(pos + 1) * dim];

                // local window: exact, causal
                let mut local_scores = vec![NEG_INF; win_len];
                for (wb, m_k) in (w_start..=m0).enumerate() {
                    for bb in 0..b {
                        let kp = m_k * b + bb;
                        if !valid(kp) {
                            continue;
                        }
                        let causal_ok = m_k < m0 || (m_k == m0 && bb <= i);
                        if causal_ok {
                            let k_row = &k_flat[kp * dim..(kp + 1) * dim];
                            local_scores[wb * b + bb] = sm_scale * dot(q_row, k_row);
                        }
                    }
                }

                // pass 1: real scores for every active tier's candidate block
                let mut per_tier_scores: Vec<Vec<f32>> = Vec::with_capacity(candidates.len());
                for &(l, block_idx) in &candidates {
                    let bl = b * (1usize << l);
                    let base = block_idx * bl;
                    let mut scores = vec![0.0f32; bl];
                    for j in 0..bl {
                        let k_row = &k_flat[(base + j) * dim..(base + j + 1) * dim];
                        scores[j] = sm_scale * dot(q_row, k_row);
                    }
                    per_tier_scores.push(scores);
                }

                // shared tau: quantile over ALL pooled tier scores for this query row.
                // Same mathematical result either way -- SortBased and Quickselect
                // compute the IDENTICAL linear-interpolation quantile, just via a
                // different algorithm (O(n log n) sort vs O(n) average quickselect).
                let shared_tau = if !per_tier_scores.is_empty() {
                    let mut pooled: Vec<f32> = per_tier_scores.iter().flatten().cloned().collect();
                    match cfg.tau_mode {
                        TauMode::SortBased => {
                            pooled.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            quantile_linear_sorted(&pooled, cfg.quantile)
                        }
                        TauMode::Quickselect => quantile_linear_quickselect(&mut pooled, cfg.quantile),
                    }
                } else {
                    0.0
                };

                // pass 2: survivors + fast-residual (exact-subtraction) non-survivor centroid
                let mut tier_logits: Vec<f32> = Vec::new();
                let mut tier_values: Vec<Vec<f32>> = Vec::new(); // each entry is a dv-length value

                for (ci, &(l, block_idx)) in candidates.iter().enumerate() {
                    let bl = b * (1usize << l);
                    let base = block_idx * bl;
                    let scores = &per_tier_scores[ci];

                    let mut sum_surv_k = vec![0.0f32; dim];
                    let mut sum_surv_v = vec![0.0f32; dv];
                    let mut n_surv = 0usize;

                    for j in 0..bl {
                        let is_surv = scores[j] >= shared_tau;
                        if !is_surv {
                            continue;
                        }
                        // survivor: real logit + real value, uncollapsed candidate.
                        // Non-survivors contribute a -inf logit -> exactly 0 softmax
                        // weight, so skipping their storage entirely (rather than
                        // pushing a -inf placeholder + a value clone that will only
                        // ever be multiplied by zero) is mathematically identical to
                        // the reference, just without the wasted allocation/clone --
                        // this also shrinks the final combined-softmax array from
                        // O(Bl) down to O(num_survivors), matching the FLOP
                        // accounting's own "~10% survivors" cost model instead of
                        // silently paying O(Bl) for a masked-out majority.
                        n_surv += 1;
                        tier_logits.push(scores[j]);
                        let v_row = &v_flat[(base + j) * dv..(base + j + 1) * dv];
                        tier_values.push(v_row.to_vec());
                        let k_row = &k_flat[(base + j) * dim..(base + j + 1) * dim];
                        crate::simd::axpy(&mut sum_surv_k, 1.0, k_row);
                        crate::simd::axpy(&mut sum_surv_v, 1.0, v_row);
                    }

                    // exact-subtraction residual: sum(non-survivors) = sum(all) - sum(survivors)
                    let count_ns_raw = bl - n_surv;
                    let has_non_surv = count_ns_raw > 0;
                    let count_ns = count_ns_raw.max(1) as f32;

                    let mut mean_k = vec![0.0f32; dim];
                    let mut mean_v = vec![0.0f32; dv];
                    for d in 0..dim {
                        mean_k[d] = (sum_k[l][block_idx][d] - sum_surv_k[d]) / count_ns;
                    }
                    for d in 0..dv {
                        mean_v[d] = (sum_v[l][block_idx][d] - sum_surv_v[d]) / count_ns;
                    }

                    let residual_logit =
                        if has_non_surv { sm_scale * dot(&mean_k, q_row) } else { NEG_INF };
                    tier_logits.push(residual_logit);
                    tier_values.push(mean_v);
                }

                // combine local + tier logits, joint softmax
                let combined_max = local_scores
                    .iter()
                    .chain(tier_logits.iter())
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
                let mut tier_w = vec![0.0f32; tier_logits.len()];
                for (j, &s) in tier_logits.iter().enumerate() {
                    let e = if s.is_finite() { (s - row_max).exp() } else { 0.0 };
                    tier_w[j] = e;
                    sum_exp += e;
                }
                let denom = sum_exp + eps;

                let out_row = out.row_mut(h, pos);
                for (wb, m_k) in (w_start..=m0).enumerate() {
                    for bb in 0..b {
                        let w = local_w[wb * b + bb] / denom;
                        if w == 0.0 {
                            continue;
                        }
                        let kp = m_k * b + bb;
                        if !valid(kp) {
                            continue;
                        }
                        let v_row = &v_flat[kp * dv..(kp + 1) * dv];
                        crate::simd::axpy(out_row, w, v_row);
                    }
                }
                for (j, val) in tier_values.iter().enumerate() {
                    let w = tier_w[j] / denom;
                    if w == 0.0 {
                        continue;
                    }
                    crate::simd::axpy(out_row, w, val);
                }
            }
        }
    }

    out
}
