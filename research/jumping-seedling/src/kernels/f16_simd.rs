//! Fp16 storage <-> fp32 compute conversion primitives (F16C), the foundation
//! for the fp16-migration branch (RESEARCH_LOG.md 2026-07-05).
//!
//! This CPU (AVX2, no AVX-512FP16/BF16) has no native fp16 *arithmetic* units
//! — only F16C's conversion instructions (`VCVTPH2PS`/`VCVTPS2PH`). So fp16
//! here is a **storage format only**: every kernel converts to fp32
//! immediately before computing and back to fp16 before storing. The benefit
//! is a halved memory/cache footprint and halved bytes moved per load, not a
//! faster FMA — unlike GPU/TPU mixed-precision training, which gets its win
//! from native low-precision tensor-core hardware this CPU doesn't have.
//!
//! Values that accumulate across many contributions (gradient sums, AdaFactor
//! optimizer state) stay fp32 throughout this migration — precision loss in
//! an accumulator compounds across a whole training run in a way precision
//! loss in a single stored weight/activation value does not. Only learned
//! weights and forward activations are candidates for fp16 storage.

use half::f16;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Convert `src` (fp16) into `dst` (fp32), element-for-element. Panics (debug)
/// if the lengths differ.
pub fn f16_to_f32(src: &[f16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx2") {
            // SAFETY: f16c + avx2 confirmed present; lengths checked above.
            unsafe { f16_to_f32_avx2(src, dst) };
            return;
        }
    }
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = s.to_f32();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c,avx2")]
unsafe fn f16_to_f32_avx2(src: &[f16], dst: &mut [f32]) {
    unsafe {
        let n = src.len();
        let n8 = n - (n % 8);
        let mut j = 0;
        while j < n8 {
            let packed = _mm_loadu_si128(src.as_ptr().add(j) as *const __m128i);
            let widened = _mm256_cvtph_ps(packed);
            _mm256_storeu_ps(dst.as_mut_ptr().add(j), widened);
            j += 8;
        }
        while j < n {
            *dst.get_unchecked_mut(j) = src.get_unchecked(j).to_f32();
            j += 1;
        }
    }
}

/// Convert `src` (fp32) into `dst` (fp16), element-for-element (round to
/// nearest). Panics (debug) if the lengths differ.
pub fn f32_to_f16(src: &[f32], dst: &mut [f16]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx2") {
            // SAFETY: f16c + avx2 confirmed present; lengths checked above.
            unsafe { f32_to_f16_avx2(src, dst) };
            return;
        }
    }
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = f16::from_f32(*s);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c,avx2")]
unsafe fn f32_to_f16_avx2(src: &[f32], dst: &mut [f16]) {
    unsafe {
        let n = src.len();
        let n8 = n - (n % 8);
        let mut j = 0;
        while j < n8 {
            let v = _mm256_loadu_ps(src.as_ptr().add(j));
            let packed = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
            _mm_storeu_si128(dst.as_mut_ptr().add(j) as *mut __m128i, packed);
            j += 8;
        }
        while j < n {
            *dst.get_unchecked_mut(j) = f16::from_f32(*src.get_unchecked(j));
            j += 1;
        }
    }
}

/// Dot product of two fp16-stored vectors, computed in fp32 (narrow storage,
/// fp32 accumulate) — converts each 8-wide chunk via F16C immediately before
/// FMA, so no fp16-precision intermediate ever exists.
pub fn dot_f16(a: &[f16], b: &[f16]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: f16c + avx2 + fma confirmed present; lengths checked above.
            return unsafe { dot_f16_avx2(a, b) };
        }
    }
    a.iter().zip(b).map(|(x, y)| x.to_f32() * y.to_f32()).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c,avx2,fma")]
unsafe fn dot_f16_avx2(a: &[f16], b: &[f16]) -> f32 {
    unsafe {
        let n = a.len();
        let n8 = n - (n % 8);
        let mut acc = _mm256_setzero_ps();
        let mut j = 0;
        while j < n8 {
            let av = _mm256_cvtph_ps(_mm_loadu_si128(a.as_ptr().add(j) as *const __m128i));
            let bv = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(j) as *const __m128i));
            acc = _mm256_fmadd_ps(av, bv, acc);
            j += 8;
        }
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let s = _mm_add_ps(hi, lo);
        let s = _mm_hadd_ps(s, s);
        let s = _mm_hadd_ps(s, s);
        let mut sum = _mm_cvtss_f32(s);
        while j < n {
            sum += a.get_unchecked(j).to_f32() * b.get_unchecked(j).to_f32();
            j += 1;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn randvec_f32(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    #[test]
    fn f32_to_f16_to_f32_roundtrips_within_fp16_precision() {
        // n = 37 exercises the 8-wide AVX2 body plus a scalar remainder.
        let a = randvec_f32(37, 0xF16C_0001);
        let mut packed = vec![f16::from_f32(0.0); 37];
        f32_to_f16(&a, &mut packed);
        let mut back = vec![0.0f32; 37];
        f16_to_f32(&packed, &mut back);
        for (x, y) in a.iter().zip(&back) {
            // fp16 has ~3 decimal digits of precision; this is a roundtrip
            // sanity check (values are all in [-1,1]), not a tight bound.
            assert!((x - y).abs() < 5e-3, "roundtrip {x} -> {y}");
        }
    }

    #[test]
    fn f16_to_f32_avx2_matches_scalar_conversion() {
        let a = randvec_f32(37, 0xF16C_0002);
        let packed: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();

        let mut got = vec![0.0f32; 37];
        f16_to_f32(&packed, &mut got);

        let want: Vec<f32> = packed.iter().map(|p| p.to_f32()).collect();
        for (x, y) in want.iter().zip(&got) {
            assert_eq!(x.to_bits(), y.to_bits(), "AVX2 conversion must match half::f16::to_f32 exactly");
        }
    }

    #[test]
    fn dot_f16_matches_f32_reference_within_fp16_precision() {
        let a = randvec_f32(37, 0xF16C_0003);
        let b = randvec_f32(37, 0xF16C_0004);
        let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
        let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();

        // Reference: dot product computed directly on the fp32-rounded-through-fp16
        // values (i.e. what the fp16 storage actually represents), in f32.
        let want: f32 = a16.iter().zip(&b16).map(|(x, y)| x.to_f32() * y.to_f32()).sum();
        let got = dot_f16(&a16, &b16);
        assert!((got - want).abs() < 1e-3, "dot_f16 {got} vs reference {want}");
    }
}
