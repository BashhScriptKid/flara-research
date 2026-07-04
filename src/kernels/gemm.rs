//! Hand-written AVX2 kernel for the vocab projection `logits = normed · Eᵀ`,
//! shared by the tied LM head and CALM early-exit decoding — the single largest
//! cost at realistic vocab sizes.
//!
//! The scalar form is a `.sum()` dot product, which the compiler will not
//! auto-vectorize because float reduction is not associative; the AVX2 path uses
//! explicit FMA accumulators (two, to hide latency) and a horizontal sum.

/// Below this many token rows, rayon dispatch overhead isn't worth it — same
/// rationale/threshold as the attention and Monarch kernels.
const HEAD_PARALLEL_THRESHOLD: usize = 8;

/// `out[ti*v + vid] = Σ_j normed[ti*h + j] · embed[vid*h + j]`, i.e. `normed · Eᵀ`
/// where `normed` is `[t, h]` row-major and `embed` is `[v, h]` row-major.
/// Dispatches to AVX2+FMA when the CPU supports it, else a scalar fallback.
///
/// Parallelizes over token rows (`t`) when there are enough of them —
/// `embed` (`[v, h]`, typically far larger than L2/L3) is swept in full for
/// every row, so this is memory-bandwidth-bound; splitting the sweep across
/// cores lets the memory controller service concurrent reads from several
/// cores instead of one core alone bottlenecking on its own load latency
/// (measured: this was ~82% of a production-shaped training step, entirely
/// unparallelized — see RESEARCH_LOG.md 2026-07-04).
pub fn logits_from_embed(normed: &[f32], embed: &[f32], out: &mut [f32], t: usize, h: usize, v: usize) {
    if t < HEAD_PARALLEL_THRESHOLD {
        logits_from_embed_range(normed, embed, out, t, h, v);
    } else {
        use rayon::prelude::*;
        let n_chunks = rayon::current_num_threads().max(1).min(t);
        let chunk_len = t.div_ceil(n_chunks);
        normed.par_chunks(chunk_len * h).zip(out.par_chunks_mut(chunk_len * v)).for_each(|(normed_c, out_c)| {
            let t_c = normed_c.len() / h;
            logits_from_embed_range(normed_c, embed, out_c, t_c, h, v);
        });
    }
}

/// Same math as `logits_from_embed`, over a caller-chosen contiguous range of
/// token rows (`normed`/`out` already sliced to that range) — the sequential
/// per-chunk worker, and also used directly for `t < HEAD_PARALLEL_THRESHOLD`.
fn logits_from_embed_range(normed: &[f32], embed: &[f32], out: &mut [f32], t: usize, h: usize, v: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: the avx2 + fma features are confirmed present above.
            unsafe { logits_avx2(normed, embed, out, t, h, v) };
            return;
        }
    }
    logits_scalar(normed, embed, out, t, h, v);
}

fn logits_scalar(normed: &[f32], embed: &[f32], out: &mut [f32], t: usize, h: usize, v: usize) {
    for ti in 0..t {
        let row = &normed[ti * h..(ti + 1) * h];
        let dst = &mut out[ti * v..(ti + 1) * v];
        for (vid, lg) in dst.iter_mut().enumerate() {
            let e = &embed[vid * h..(vid + 1) * h];
            *lg = row.iter().zip(e).map(|(a, b)| a * b).sum();
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn logits_avx2(normed: &[f32], embed: &[f32], out: &mut [f32], t: usize, h: usize, v: usize) {
    use core::arch::x86_64::*;

    // Horizontal sum of a __m256 into a scalar.
    #[inline]
    unsafe fn hsum(x: __m256) -> f32 {
        unsafe {
            let hi = _mm256_extractf128_ps(x, 1);
            let lo = _mm256_castps256_ps128(x);
            let s = _mm_add_ps(hi, lo);
            let s = _mm_hadd_ps(s, s);
            let s = _mm_hadd_ps(s, s);
            _mm_cvtss_f32(s)
        }
    }

    unsafe {
        let h16 = h - (h % 16);
        for ti in 0..t {
            let row = normed.as_ptr().add(ti * h);
            for vid in 0..v {
                let e = embed.as_ptr().add(vid * h);
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut j = 0;
                while j < h16 {
                    acc0 = _mm256_fmadd_ps(
                        _mm256_loadu_ps(row.add(j)),
                        _mm256_loadu_ps(e.add(j)),
                        acc0,
                    );
                    acc1 = _mm256_fmadd_ps(
                        _mm256_loadu_ps(row.add(j + 8)),
                        _mm256_loadu_ps(e.add(j + 8)),
                        acc1,
                    );
                    j += 16;
                }
                while j + 8 <= h {
                    acc0 = _mm256_fmadd_ps(
                        _mm256_loadu_ps(row.add(j)),
                        _mm256_loadu_ps(e.add(j)),
                        acc0,
                    );
                    j += 8;
                }
                let mut sum = hsum(_mm256_add_ps(acc0, acc1));
                while j < h {
                    sum += *row.add(j) * *e.add(j);
                    j += 1;
                }
                *out.get_unchecked_mut(ti * v + vid) = sum;
            }
        }
    }
}

/// Backward of the tied LM head, fused: given `d_logits` (`[t, v]`), accumulate
/// `d_normed[ti] += Σ_vid d_logits[ti,vid]·embed[vid]` and
/// `d_embed[vid] += Σ_ti d_logits[ti,vid]·normed[ti]`. `d_normed` is overwritten
/// (caller zero-inits); `d_embed` is accumulated into (tied head + scatter share it).
/// Both passes sweep the full `embed`/`normed` table for every row they
/// process (same memory-bandwidth-bound shape as `logits_from_embed`), so
/// each is parallelized independently over its own natural disjoint-output
/// axis: pass A over token rows (`d_normed` is `[t, h]`), pass B over vocab
/// rows (`d_embed` is `[v, h]`) — never over the *other* axis, since that's
/// exactly where the accumulation happens (`dnf`/`de` sum over the full
/// inner range) and would need cross-thread merging instead of disjoint writes.
pub fn head_backward(
    d_logits: &[f32],
    embed: &[f32],
    normed: &[f32],
    d_normed: &mut [f32],
    d_embed: &mut [f32],
    t: usize,
    h: usize,
    v: usize,
) {
    if t < HEAD_PARALLEL_THRESHOLD {
        head_backward_pass_a_range(d_logits, embed, d_normed, t, h, v);
    } else {
        use rayon::prelude::*;
        let n_chunks = rayon::current_num_threads().max(1).min(t);
        let chunk_len = t.div_ceil(n_chunks);
        d_logits.par_chunks(chunk_len * v).zip(d_normed.par_chunks_mut(chunk_len * h)).for_each(|(dl_c, dn_c)| {
            let t_c = dn_c.len() / h;
            head_backward_pass_a_range(dl_c, embed, dn_c, t_c, h, v);
        });
    }

    if v < HEAD_PARALLEL_THRESHOLD {
        head_backward_pass_b_range(d_logits, normed, d_embed, t, h, v, 0);
    } else {
        use rayon::prelude::*;
        let n_chunks = rayon::current_num_threads().max(1).min(v);
        let chunk_len = v.div_ceil(n_chunks);
        d_embed.par_chunks_mut(chunk_len * h).enumerate().for_each(|(c, de_c)| {
            let vid0 = c * chunk_len;
            head_backward_pass_b_range(d_logits, normed, de_c, t, h, v, vid0);
        });
    }
}

/// Pass A over a caller-chosen contiguous range of token rows (`d_logits`/
/// `d_normed` already sliced to that range): `d_normed[ti] = Σ_vid
/// d_logits[ti,vid] · embed[vid]`; dnf stays L1-resident. `d_normed` is
/// accumulated into (caller zero-inits before the first call).
fn head_backward_pass_a_range(d_logits: &[f32], embed: &[f32], d_normed: &mut [f32], t: usize, h: usize, v: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: avx2 + fma confirmed present.
            unsafe { head_backward_pass_a_avx2(d_logits, embed, d_normed, t, h, v) };
            return;
        }
    }
    for ti in 0..t {
        let dl = &d_logits[ti * v..(ti + 1) * v];
        let dnf = &mut d_normed[ti * h..(ti + 1) * h];
        for vid in 0..v {
            let dlv = dl[vid];
            let e = &embed[vid * h..(vid + 1) * h];
            for j in 0..h {
                dnf[j] += dlv * e[j];
            }
        }
    }
}

/// Pass B over a caller-chosen contiguous range of vocab rows (`d_embed`
/// already sliced to that range, starting at absolute vocab index `vid0`):
/// `d_embed[vid] += Σ_ti d_logits[ti,vid] · normed[ti]`. With vid outermost,
/// each de row is written once instead of once per token — cutting d_embed
/// write traffic ~t× (the cache-residency win; this loop was bandwidth-bound
/// when fused).
fn head_backward_pass_b_range(
    d_logits: &[f32], normed: &[f32], d_embed: &mut [f32], t: usize, h: usize, v: usize, vid0: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: avx2 + fma confirmed present.
            unsafe { head_backward_pass_b_avx2(d_logits, normed, d_embed, t, h, v, vid0) };
            return;
        }
    }
    let v_c = d_embed.len() / h;
    for vc in 0..v_c {
        let vid = vid0 + vc;
        let de = &mut d_embed[vc * h..(vc + 1) * h];
        for ti in 0..t {
            let dlv = d_logits[ti * v + vid];
            let nf = &normed[ti * h..(ti + 1) * h];
            for j in 0..h {
                de[j] += dlv * nf[j];
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn head_backward_pass_a_avx2(
    d_logits: &[f32],
    embed: &[f32],
    d_normed: &mut [f32],
    t: usize,
    h: usize,
    v: usize,
) {
    use core::arch::x86_64::*;
    unsafe {
        let h8 = h - (h % 8);
        // d_normed[ti] = Σ_vid d_logits[ti,vid] · embed[vid].
        for ti in 0..t {
            let dl = d_logits.as_ptr().add(ti * v);
            let dnf = d_normed.as_mut_ptr().add(ti * h);
            for vid in 0..v {
                let dlv = *dl.add(vid);
                let dlb = _mm256_set1_ps(dlv);
                let e = embed.as_ptr().add(vid * h);
                let mut j = 0;
                while j < h8 {
                    let acc = _mm256_fmadd_ps(dlb, _mm256_loadu_ps(e.add(j)), _mm256_loadu_ps(dnf.add(j)));
                    _mm256_storeu_ps(dnf.add(j), acc);
                    j += 8;
                }
                while j < h {
                    *dnf.add(j) += dlv * *e.add(j);
                    j += 1;
                }
            }
        }
    }
}

/// `d_embed` is already sliced to this chunk's vocab rows (`v_c = d_embed.len()/h`
/// of them); `vid0` is that chunk's starting absolute vocab index, used to
/// index into the un-sliced `d_logits`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn head_backward_pass_b_avx2(
    d_logits: &[f32],
    normed: &[f32],
    d_embed: &mut [f32],
    t: usize,
    h: usize,
    v: usize,
    vid0: usize,
) {
    use core::arch::x86_64::*;
    unsafe {
        let h8 = h - (h % 8);
        let v_c = d_embed.len() / h;
        // d_embed[vid] += Σ_ti d_logits[ti,vid] · normed[ti]; de written once
        // per vid (cache-resident across the ti loop) instead of once per token.
        for vc in 0..v_c {
            let vid = vid0 + vc;
            let de = d_embed.as_mut_ptr().add(vc * h);
            for ti in 0..t {
                let dlv = *d_logits.as_ptr().add(ti * v + vid);
                let dlb = _mm256_set1_ps(dlv);
                let nf = normed.as_ptr().add(ti * h);
                let mut j = 0;
                while j < h8 {
                    let acc = _mm256_fmadd_ps(dlb, _mm256_loadu_ps(nf.add(j)), _mm256_loadu_ps(de.add(j)));
                    _mm256_storeu_ps(de.add(j), acc);
                    j += 8;
                }
                while j < h {
                    *de.add(j) += dlv * *nf.add(j);
                    j += 1;
                }
            }
        }
    }
}

/// Dot product `Σ_i a[i]·b[i]` over two equal-length slices, AVX2+FMA when present.
/// Used for the complex-dot reductions in `BasisMatmul::backward` (over interleaved
/// re/im), which are reductions and so do not auto-vectorize.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: avx2 + fma confirmed present; slices are equal length.
            return unsafe { dot_avx2(a.as_ptr(), b.as_ptr(), a.len()) };
        }
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: *const f32, b: *const f32, n: usize) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let n16 = n - (n % 16);
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut j = 0;
        while j < n16 {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(j)), _mm256_loadu_ps(b.add(j)), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(j + 8)), _mm256_loadu_ps(b.add(j + 8)), acc1);
            j += 16;
        }
        while j + 8 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(j)), _mm256_loadu_ps(b.add(j)), acc0);
            j += 8;
        }
        let acc = _mm256_add_ps(acc0, acc1);
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let s = _mm_add_ps(hi, lo);
        let s = _mm_hadd_ps(s, s);
        let s = _mm_hadd_ps(s, s);
        let mut sum = _mm_cvtss_f32(s);
        while j < n {
            sum += *a.add(j) * *b.add(j);
            j += 1;
        }
        sum
    }
}

/// Four dot products against a shared vector `a`: `[dot(a,b0), dot(a,b1),
/// dot(a,b2), dot(a,b3)]`. Register-blocked — `a`'s vector loads are shared
/// across all four accumulators instead of being reloaded once per `dot()`
/// call, amortizing the load when `a` is reused against several `b`s in a
/// row (e.g. one query row scored against several keys). AVX2+FMA when
/// present; scalar fallback otherwise. All four slices must share `a`'s length.
pub fn dot4(a: &[f32], b0: &[f32], b1: &[f32], b2: &[f32], b3: &[f32]) -> [f32; 4] {
    debug_assert_eq!(a.len(), b0.len());
    debug_assert_eq!(a.len(), b1.len());
    debug_assert_eq!(a.len(), b2.len());
    debug_assert_eq!(a.len(), b3.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: avx2 + fma confirmed present; slices share length `a.len()`.
            return unsafe { dot4_avx2(a.as_ptr(), b0.as_ptr(), b1.as_ptr(), b2.as_ptr(), b3.as_ptr(), a.len()) };
        }
    }
    [
        a.iter().zip(b0).map(|(x, y)| x * y).sum(),
        a.iter().zip(b1).map(|(x, y)| x * y).sum(),
        a.iter().zip(b2).map(|(x, y)| x * y).sum(),
        a.iter().zip(b3).map(|(x, y)| x * y).sum(),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot4_avx2(
    a: *const f32, b0: *const f32, b1: *const f32, b2: *const f32, b3: *const f32, n: usize,
) -> [f32; 4] {
    use core::arch::x86_64::*;
    unsafe {
        let n8 = n - (n % 8);
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut j = 0;
        while j < n8 {
            let av = _mm256_loadu_ps(a.add(j)); // loaded once, reused across all four
            acc0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b0.add(j)), acc0);
            acc1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b1.add(j)), acc1);
            acc2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b2.add(j)), acc2);
            acc3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b3.add(j)), acc3);
            j += 8;
        }
        let hsum = |acc: __m256| -> f32 {
            let hi = _mm256_extractf128_ps(acc, 1);
            let lo = _mm256_castps256_ps128(acc);
            let s = _mm_add_ps(hi, lo);
            let s = _mm_hadd_ps(s, s);
            let s = _mm_hadd_ps(s, s);
            _mm_cvtss_f32(s)
        };
        let mut out = [hsum(acc0), hsum(acc1), hsum(acc2), hsum(acc3)];
        while j < n {
            out[0] += *a.add(j) * *b0.add(j);
            out[1] += *a.add(j) * *b1.add(j);
            out[2] += *a.add(j) * *b2.add(j);
            out[3] += *a.add(j) * *b3.add(j);
            j += 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot4_matches_four_dot_calls() {
        let a: Vec<f32> = (0..37).map(|i| (i as f32) * 0.37 - 3.0).collect();
        let b0: Vec<f32> = (0..37).map(|i| (i as f32) * 0.11 + 1.0).collect();
        let b1: Vec<f32> = (0..37).map(|i| (i as f32) * -0.05 + 2.0).collect();
        let b2: Vec<f32> = (0..37).map(|i| (i as f32) * 0.23 - 1.5).collect();
        let b3: Vec<f32> = (0..37).map(|i| (i as f32) * 0.02 + 0.5).collect();
        let got = dot4(&a, &b0, &b1, &b2, &b3);
        let want = [dot(&a, &b0), dot(&a, &b1), dot(&a, &b2), dot(&a, &b3)];
        for i in 0..4 {
            assert!((got[i] - want[i]).abs() < 1e-3, "dot4[{i}] {} vs {}", got[i], want[i]);
        }
    }

    #[test]
    fn avx2_matches_scalar() {
        // h = 44 exercises the 16-wide body, the 8-wide tail, and the scalar
        // remainder all at once.
        let (t, h, v) = (3usize, 44usize, 7usize);
        let normed: Vec<f32> = (0..t * h).map(|i| (i as f32 * 0.017).sin()).collect();
        let embed: Vec<f32> = (0..v * h).map(|i| (i as f32 * 0.013).cos()).collect();

        let mut want = vec![0.0f32; t * v];
        logits_scalar(&normed, &embed, &mut want, t, h, v);

        let mut got = vec![0.0f32; t * v];
        logits_from_embed(&normed, &embed, &mut got, t, h, v);

        for (x, y) in want.iter().zip(&got) {
            assert!((x - y).abs() < 1e-4, "mismatch: scalar {x} vs dispatched {y}");
        }
    }

    /// Pure-scalar reference (no AVX2, no parallelism) for `head_backward_*_matches_scalar`.
    fn head_backward_naive(
        d_logits: &[f32], embed: &[f32], normed: &[f32],
        d_normed: &mut [f32], d_embed: &mut [f32], t: usize, h: usize, v: usize,
    ) {
        for ti in 0..t {
            let dl = &d_logits[ti * v..(ti + 1) * v];
            let dnf = &mut d_normed[ti * h..(ti + 1) * h];
            for vid in 0..v {
                let dlv = dl[vid];
                let e = &embed[vid * h..(vid + 1) * h];
                for j in 0..h {
                    dnf[j] += dlv * e[j];
                }
            }
        }
        for vid in 0..v {
            let de = &mut d_embed[vid * h..(vid + 1) * h];
            for ti in 0..t {
                let dlv = d_logits[ti * v + vid];
                let nf = &normed[ti * h..(ti + 1) * h];
                for j in 0..h {
                    de[j] += dlv * nf[j];
                }
            }
        }
    }

    #[test]
    fn head_backward_avx2_matches_scalar() {
        let (t, h, v) = (3usize, 44usize, 7usize);
        let d_logits: Vec<f32> = (0..t * v).map(|i| (i as f32 * 0.021).sin()).collect();
        let embed: Vec<f32> = (0..v * h).map(|i| (i as f32 * 0.013).cos()).collect();
        let normed: Vec<f32> = (0..t * h).map(|i| (i as f32 * 0.017).sin()).collect();

        // d_embed starts non-zero to exercise accumulation.
        let de0: Vec<f32> = (0..v * h).map(|i| (i as f32 * 0.005).cos()).collect();

        let mut dn_want = vec![0.0f32; t * h];
        let mut de_want = de0.clone();
        head_backward_naive(&d_logits, &embed, &normed, &mut dn_want, &mut de_want, t, h, v);

        let mut dn_got = vec![0.0f32; t * h];
        let mut de_got = de0.clone();
        head_backward(&d_logits, &embed, &normed, &mut dn_got, &mut de_got, t, h, v);

        for (x, y) in dn_want.iter().zip(&dn_got) {
            assert!((x - y).abs() < 1e-4, "d_normed: {x} vs {y}");
        }
        for (x, y) in de_want.iter().zip(&de_got) {
            assert!((x - y).abs() < 1e-4, "d_embed: {x} vs {y}");
        }
    }

    /// Same as `head_backward_avx2_matches_scalar` but with `t`/`v` large
    /// enough to exercise the rayon-parallel branches in both passes.
    #[test]
    fn head_backward_parallel_matches_scalar() {
        let (t, h, v) = (16usize, 44usize, 20usize);
        let d_logits: Vec<f32> = (0..t * v).map(|i| (i as f32 * 0.021).sin()).collect();
        let embed: Vec<f32> = (0..v * h).map(|i| (i as f32 * 0.013).cos()).collect();
        let normed: Vec<f32> = (0..t * h).map(|i| (i as f32 * 0.017).sin()).collect();
        let de0: Vec<f32> = (0..v * h).map(|i| (i as f32 * 0.005).cos()).collect();

        let mut dn_want = vec![0.0f32; t * h];
        let mut de_want = de0.clone();
        head_backward_naive(&d_logits, &embed, &normed, &mut dn_want, &mut de_want, t, h, v);

        let mut dn_got = vec![0.0f32; t * h];
        let mut de_got = de0.clone();
        head_backward(&d_logits, &embed, &normed, &mut dn_got, &mut de_got, t, h, v);

        for (x, y) in dn_want.iter().zip(&dn_got) {
            assert!((x - y).abs() < 1e-3, "d_normed: {x} vs {y}");
        }
        for (x, y) in de_want.iter().zip(&de_got) {
            assert!((x - y).abs() < 1e-3, "d_embed: {x} vs {y}");
        }
    }

    /// `logits_from_embed`'s rayon-parallel branch (t >= HEAD_PARALLEL_THRESHOLD).
    #[test]
    fn logits_from_embed_parallel_matches_scalar() {
        let (t, h, v) = (16usize, 44usize, 9usize);
        let normed: Vec<f32> = (0..t * h).map(|i| (i as f32 * 0.031).sin()).collect();
        let embed: Vec<f32> = (0..v * h).map(|i| (i as f32 * 0.019).cos()).collect();

        let mut want = vec![0.0f32; t * v];
        logits_scalar(&normed, &embed, &mut want, t, h, v);

        let mut got = vec![0.0f32; t * v];
        logits_from_embed(&normed, &embed, &mut got, t, h, v);

        for (x, y) in want.iter().zip(&got) {
            assert!((x - y).abs() < 1e-4, "mismatch: scalar {x} vs parallel {y}");
        }
    }

    #[test]
    fn dot_matches_scalar() {
        let n = 44; // 16-wide body + 8-wide tail + scalar remainder
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.031).sin()).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.019).cos()).collect();
        let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let got = dot(&a, &b);
        assert!((want - got).abs() < 1e-4, "dot: {want} vs {got}");
    }
}
