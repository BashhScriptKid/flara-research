//! AVX2/FMA-accelerated primitives, safe API with runtime feature
//! detection and a scalar fallback -- portable (works, just slower, on
//! non-AVX2 hardware), and the two operations (`dot`, `axpy`) are the
//! only hot inner loops in all three kernels, so vectorizing just these
//! two keeps the already-correctness-verified algorithm structure in
//! causal.rs/sliding.rs/meta.rs completely unchanged -- only the `for d
//! in 0..dim { ... }` loop bodies get swapped for calls into here.
//!
//! `unsafe` is confined to this module and gated behind
//! `is_x86_feature_detected!`, matching the standard pattern for safe
//! SIMD wrappers in Rust.

#[cfg(target_arch = "x86_64")]
mod avx2_impl {
    use std::arch::x86_64::*;

    /// Horizontal sum of an `__m256` (8 lanes) down to a scalar.
    #[target_feature(enable = "avx2")]
    unsafe fn hsum(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum4 = _mm_add_ps(hi, lo);
        let shuf = _mm_movehdup_ps(sum4);
        let sums = _mm_add_ps(sum4, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        let sums2 = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(sums2)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let mut acc = _mm256_setzero_ps();
        let chunks = n / 8;
        for i in 0..chunks {
            unsafe {
                let av = _mm256_loadu_ps(a.as_ptr().add(i * 8));
                let bv = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                acc = _mm256_fmadd_ps(av, bv, acc);
            }
        }
        let mut sum = unsafe { hsum(acc) };
        for i in chunks * 8..n {
            sum += a[i] * b[i];
        }
        sum
    }

    /// `y[i] += a * x[i]` for all i, vectorized.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn axpy_avx2(y: &mut [f32], a: f32, x: &[f32]) {
        let n = y.len();
        let av = _mm256_set1_ps(a);
        let chunks = n / 8;
        for i in 0..chunks {
            unsafe {
                let xv = _mm256_loadu_ps(x.as_ptr().add(i * 8));
                let yv = _mm256_loadu_ps(y.as_ptr().add(i * 8));
                let r = _mm256_fmadd_ps(av, xv, yv);
                _mm256_storeu_ps(y.as_mut_ptr().add(i * 8), r);
            }
        }
        for i in chunks * 8..n {
            y[i] += a * x[i];
        }
    }
}

#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
fn axpy_scalar(y: &mut [f32], a: f32, x: &[f32]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

/// Dot product, AVX2/FMA-accelerated when available, scalar fallback
/// otherwise. `a` and `b` must have equal length.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2_impl::dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

/// `y[i] += a * x[i]`, AVX2/FMA-accelerated when available.
#[inline]
pub fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    debug_assert_eq!(y.len(), x.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { avx2_impl::axpy_avx2(y, a, x) };
            return;
        }
    }
    axpy_scalar(y, a, x);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.max(1);
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32
            })
            .collect()
    }

    #[test]
    fn dot_matches_scalar_various_lengths() {
        // deliberately include lengths that are NOT multiples of 8, to
        // exercise the scalar remainder tail
        for &n in &[1usize, 7, 8, 9, 15, 16, 17, 64, 100] {
            let a = rand_vec(n, 42);
            let b = rand_vec(n, 43);
            let expected = dot_scalar(&a, &b);
            let got = dot(&a, &b);
            assert!(
                (expected - got).abs() < 1e-3,
                "dot mismatch at n={n}: scalar={expected}, simd={got}"
            );
        }
    }

    #[test]
    fn axpy_matches_scalar_various_lengths() {
        for &n in &[1usize, 7, 8, 9, 15, 16, 17, 64, 100] {
            let x = rand_vec(n, 42);
            let mut y1 = rand_vec(n, 44);
            let mut y2 = y1.clone();
            axpy_scalar(&mut y1, 0.37, &x);
            axpy(&mut y2, 0.37, &x);
            for (a, b) in y1.iter().zip(y2.iter()) {
                assert!((a - b).abs() < 1e-4, "axpy mismatch at n={n}: scalar={a}, simd={b}");
            }
        }
    }
}
