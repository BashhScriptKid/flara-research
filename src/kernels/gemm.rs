//! Hand-written AVX2 kernel for the vocab projection `logits = normed · Eᵀ`,
//! shared by the tied LM head and CALM early-exit decoding — the single largest
//! cost at realistic vocab sizes.
//!
//! The scalar form is a `.sum()` dot product, which the compiler will not
//! auto-vectorize because float reduction is not associative; the AVX2 path uses
//! explicit FMA accumulators (two, to hide latency) and a horizontal sum.

/// `out[ti*v + vid] = Σ_j normed[ti*h + j] · embed[vid*h + j]`, i.e. `normed · Eᵀ`
/// where `normed` is `[t, h]` row-major and `embed` is `[v, h]` row-major.
/// Dispatches to AVX2+FMA when the CPU supports it, else a scalar fallback.
pub fn logits_from_embed(normed: &[f32], embed: &[f32], out: &mut [f32], t: usize, h: usize, v: usize) {
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
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: avx2 + fma confirmed present.
            unsafe { head_backward_avx2(d_logits, embed, normed, d_normed, d_embed, t, h, v) };
            return;
        }
    }
    head_backward_scalar(d_logits, embed, normed, d_normed, d_embed, t, h, v);
}

fn head_backward_scalar(
    d_logits: &[f32],
    embed: &[f32],
    normed: &[f32],
    d_normed: &mut [f32],
    d_embed: &mut [f32],
    t: usize,
    h: usize,
    v: usize,
) {
    // Pass A: d_normed[ti] = Σ_vid d_logits[ti,vid] · embed[vid]; dnf stays L1-resident.
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
    // Pass B: d_embed[vid] += Σ_ti d_logits[ti,vid] · normed[ti]. With vid outermost,
    // each de row is written once instead of once per token — cutting d_embed write
    // traffic ~t× (the cache-residency win; this loop was bandwidth-bound when fused).
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn head_backward_avx2(
    d_logits: &[f32],
    embed: &[f32],
    normed: &[f32],
    d_normed: &mut [f32],
    d_embed: &mut [f32],
    t: usize,
    h: usize,
    v: usize,
) {
    use core::arch::x86_64::*;
    unsafe {
        let h8 = h - (h % 8);
        // Pass A: d_normed[ti] = Σ_vid d_logits[ti,vid] · embed[vid].
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
        // Pass B: d_embed[vid] += Σ_ti d_logits[ti,vid] · normed[ti]; de written once
        // per vid (cache-resident across the ti loop) instead of once per token.
        for vid in 0..v {
            let de = d_embed.as_mut_ptr().add(vid * h);
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

#[cfg(test)]
mod tests {
    use super::*;

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
        head_backward_scalar(&d_logits, &embed, &normed, &mut dn_want, &mut de_want, t, h, v);

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
