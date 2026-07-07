// Time-boxed spike: does packing TWO independent 8-element dot products into
// one 16-lane int16 AVX2 register (_mm256_madd_epi16) actually beat doing two
// separate 8-wide fp32 dot products (the current, already width-matched
// fp32 kernel)? This is the prerequisite question before any real int16 SIMD
// kernel rewrite of apply_block_avx2/bwd_block_avx2_hoisted — if this doesn't
// win at the primitive level, the larger rewrite isn't worth attempting.
//
// Not wired into production code. Standalone microbenchmark only.

use std::arch::x86_64::*;
use std::time::Instant;

const ITERS: usize = 20_000_000;

/// Two independent length-8 dot products, computed via two ordinary fp32
/// AVX2 8-wide multiplies + horizontal reduction (mirrors the reduction
/// style already used by matvec8 in monarch.rs).
#[target_feature(enable = "avx2,fma")]
unsafe fn dual_dot8_f32(mat0: &[f32; 8], mat1: &[f32; 8], vec0: &[f32; 8], vec1: &[f32; 8]) -> (f32, f32) {
    let p0 = _mm256_mul_ps(_mm256_loadu_ps(mat0.as_ptr()), _mm256_loadu_ps(vec0.as_ptr()));
    let p1 = _mm256_mul_ps(_mm256_loadu_ps(mat1.as_ptr()), _mm256_loadu_ps(vec1.as_ptr()));
    let h0 = _mm256_hadd_ps(p0, p1);
    let h1 = _mm256_hadd_ps(h0, h0);
    let s = _mm_add_ps(_mm256_castps256_ps128(h1), _mm256_extractf128_ps(h1, 1));
    (_mm_cvtss_f32(s), _mm_cvtss_f32(_mm_shuffle_ps(s, s, 0x01)))
}

/// Same two dot products, but both blocks packed into a single 16-lane int16
/// register (block0 in low 8 lanes, block1 in high 8 lanes) and computed with
/// ONE _mm256_madd_epi16 (16x i16 -> 8x i32, pairing adjacent lanes) instead
/// of two separate fp32 multiplies.
#[target_feature(enable = "avx2")]
unsafe fn dual_dot8_i16(mat0: &[i16; 8], mat1: &[i16; 8], vec0: &[i16; 8], vec1: &[i16; 8]) -> (i32, i32) {
    let mut mat_packed = [0i16; 16];
    mat_packed[0..8].copy_from_slice(mat0);
    mat_packed[8..16].copy_from_slice(mat1);
    let mut vec_packed = [0i16; 16];
    vec_packed[0..8].copy_from_slice(vec0);
    vec_packed[8..16].copy_from_slice(vec1);

    let m = _mm256_loadu_si256(mat_packed.as_ptr() as *const __m256i);
    let v = _mm256_loadu_si256(vec_packed.as_ptr() as *const __m256i);
    // madd pairs (0,1)(2,3)...(14,15) -> 8x i32: lanes 0-3 = block0's 4 partial
    // sums, lanes 4-7 = block1's 4 partial sums.
    let prod = _mm256_madd_epi16(m, v);
    let lo = _mm256_castsi256_si128(prod); // block0's 4 partials
    let hi = _mm256_extracti128_si256(prod, 1); // block1's 4 partials
    let sum_lo = _mm_hadd_epi32(lo, lo);
    let sum_lo = _mm_hadd_epi32(sum_lo, sum_lo);
    let sum_hi = _mm_hadd_epi32(hi, hi);
    let sum_hi = _mm_hadd_epi32(sum_hi, sum_hi);
    (_mm_extract_epi32(sum_lo, 0), _mm_extract_epi32(sum_hi, 0))
}

/// v2: same packing (explicit 128-bit loads + insert — what the compiler
/// already emitted for v1's array staging anyway), but the reduction mirrors
/// the fp32 matvec8 pattern: 2x full-width _mm256_hadd_epi32 on the 256-bit
/// madd result (block0 partials reduce in the low lane, block1 in the high
/// lane, in parallel), extracting to 128-bit only for the final scalar reads.
/// v1 used 4x 128-bit _mm_hadd_epi32 after an early extract.
#[target_feature(enable = "avx2")]
unsafe fn dual_dot8_i16_v2(mat0: &[i16; 8], mat1: &[i16; 8], vec0: &[i16; 8], vec1: &[i16; 8]) -> (i32, i32) {
    let m = _mm256_set_m128i(
        _mm_loadu_si128(mat1.as_ptr() as *const __m128i),
        _mm_loadu_si128(mat0.as_ptr() as *const __m128i),
    );
    let v = _mm256_set_m128i(
        _mm_loadu_si128(vec1.as_ptr() as *const __m128i),
        _mm_loadu_si128(vec0.as_ptr() as *const __m128i),
    );
    let prod = _mm256_madd_epi16(m, v); // lo lane: block0 partials, hi lane: block1 partials
    let h0 = _mm256_hadd_epi32(prod, prod); // per-lane pairwise sums
    let h1 = _mm256_hadd_epi32(h0, h0);     // lane element 0 = full sum per block
    let lo = _mm256_castsi256_si128(h1);
    let hi = _mm256_extracti128_si256(h1, 1);
    (_mm_cvtsi128_si32(lo), _mm_cvtsi128_si32(hi))
}

/// v3: no hadd at all. vphaddd is 2 uops on the shuffle ports; replace the
/// whole reduction with in-lane shuffle+add (each 1 uop, wider port choice).
#[target_feature(enable = "avx2")]
unsafe fn dual_dot8_i16_v3(mat0: &[i16; 8], mat1: &[i16; 8], vec0: &[i16; 8], vec1: &[i16; 8]) -> (i32, i32) {
    let m = _mm256_set_m128i(
        _mm_loadu_si128(mat1.as_ptr() as *const __m128i),
        _mm_loadu_si128(mat0.as_ptr() as *const __m128i),
    );
    let v = _mm256_set_m128i(
        _mm_loadu_si128(vec1.as_ptr() as *const __m128i),
        _mm_loadu_si128(vec0.as_ptr() as *const __m128i),
    );
    let prod = _mm256_madd_epi16(m, v);
    // In-lane reduce 4x i32 -> 1x i32 per 128-bit lane.
    let t1 = _mm256_add_epi32(prod, _mm256_shuffle_epi32(prod, 0b0100_1110)); // [0+2, 1+3, ..]
    let t2 = _mm256_add_epi32(t1, _mm256_shuffle_epi32(t1, 0b0000_0001));    // elem0 = total
    let lo = _mm256_castsi256_si128(t2);
    let hi = _mm256_extracti128_si256(t2, 1);
    (_mm_cvtsi128_si32(lo), _mm_cvtsi128_si32(hi))
}

fn main() {
    if !is_x86_feature_detected!("avx2") {
        println!("AVX2 not available on this machine — spike cannot run.");
        return;
    }

    let mat0 = [1.0f32, 2.0, -1.0, 0.5, 3.0, -2.0, 1.5, 0.25];
    let mat1 = [0.5f32, -1.0, 2.0, 1.0, -0.5, 1.5, -2.0, 0.75];
    let vec0 = [1.0f32, 0.5, -0.5, 2.0, 1.0, -1.0, 0.25, 1.5];
    let vec1 = [2.0f32, -1.0, 1.0, 0.5, -0.5, 1.5, 1.0, -2.0];

    let mat0_i16: [i16; 8] = std::array::from_fn(|i| (mat0[i] * 256.0) as i16);
    let mat1_i16: [i16; 8] = std::array::from_fn(|i| (mat1[i] * 256.0) as i16);
    let vec0_i16: [i16; 8] = std::array::from_fn(|i| (vec0[i] * 256.0) as i16);
    let vec1_i16: [i16; 8] = std::array::from_fn(|i| (vec1[i] * 256.0) as i16);

    // Correctness check first — no point benchmarking a wrong kernel.
    unsafe {
        let (f0, f1) = dual_dot8_f32(&mat0, &mat1, &vec0, &vec1);
        let (i0, i1) = dual_dot8_i16(&mat0_i16, &mat1_i16, &vec0_i16, &vec1_i16);
        // i16 path is scaled by 256*256 = 65536 relative to the fp32 result.
        let (i0_f, i1_f) = (i0 as f32 / 65536.0, i1 as f32 / 65536.0);
        println!("correctness: fp32=({f0:.4},{f1:.4})  i16-derived=({i0_f:.4},{i1_f:.4})");
        assert!((f0 - i0_f).abs() < 0.01 && (f1 - i1_f).abs() < 0.01, "i16 packed dot mismatch");

        let (j0, j1) = dual_dot8_i16_v2(&mat0_i16, &mat1_i16, &vec0_i16, &vec1_i16);
        assert!(j0 == i0 && j1 == i1, "v2 mismatch vs v1: ({j0},{j1}) vs ({i0},{i1})");
        let (k0, k1) = dual_dot8_i16_v3(&mat0_i16, &mat1_i16, &vec0_i16, &vec1_i16);
        assert!(k0 == i0 && k1 == i1, "v3 mismatch vs v1: ({k0},{k1}) vs ({i0},{i1})");
        println!("correctness: v2 and v3 match v1 exactly");
    }

    unsafe {
        // Warm up + time fp32 path.
        let mut acc = 0.0f32;
        let start = Instant::now();
        for _ in 0..ITERS {
            let (a, b) = dual_dot8_f32(
                std::hint::black_box(&mat0),
                std::hint::black_box(&mat1),
                std::hint::black_box(&vec0),
                std::hint::black_box(&vec1),
            );
            acc += a + b;
        }
        let fp32_time = start.elapsed();
        std::hint::black_box(acc);

        // Time int16 dual-packed path.
        let mut acc_i = 0i64;
        let start = Instant::now();
        for _ in 0..ITERS {
            let (a, b) = dual_dot8_i16(
                std::hint::black_box(&mat0_i16),
                std::hint::black_box(&mat1_i16),
                std::hint::black_box(&vec0_i16),
                std::hint::black_box(&vec1_i16),
            );
            acc_i += a as i64 + b as i64;
        }
        let i16_time = start.elapsed();
        std::hint::black_box(acc_i);

        // Time int16 v2 (full-width 256-bit hadd reduction).
        let mut acc_i2 = 0i64;
        let start = Instant::now();
        for _ in 0..ITERS {
            let (a, b) = dual_dot8_i16_v2(
                std::hint::black_box(&mat0_i16),
                std::hint::black_box(&mat1_i16),
                std::hint::black_box(&vec0_i16),
                std::hint::black_box(&vec1_i16),
            );
            acc_i2 += a as i64 + b as i64;
        }
        let i16_v2_time = start.elapsed();
        std::hint::black_box(acc_i2);

        // Time int16 v3 (shuffle+add reduction, no hadd).
        let mut acc_i3 = 0i64;
        let start = Instant::now();
        for _ in 0..ITERS {
            let (a, b) = dual_dot8_i16_v3(
                std::hint::black_box(&mat0_i16),
                std::hint::black_box(&mat1_i16),
                std::hint::black_box(&vec0_i16),
                std::hint::black_box(&vec1_i16),
            );
            acc_i3 += a as i64 + b as i64;
        }
        let i16_v3_time = start.elapsed();
        std::hint::black_box(acc_i3);

        println!("\n{ITERS} iterations, each computing 2 independent length-8 dot products:");
        println!("fp32 (2x separate 8-wide mul+hadd):        {:>8.2}ms  ({:>5.2}ns/pair)",
            fp32_time.as_secs_f64() * 1000.0, fp32_time.as_secs_f64() * 1e9 / ITERS as f64);
        println!("int16 v1 (madd + 4x 128-bit hadd):         {:>8.2}ms  ({:>5.2}ns/pair)",
            i16_time.as_secs_f64() * 1000.0, i16_time.as_secs_f64() * 1e9 / ITERS as f64);
        println!("int16 v2 (madd + 2x 256-bit hadd):         {:>8.2}ms  ({:>5.2}ns/pair)",
            i16_v2_time.as_secs_f64() * 1000.0, i16_v2_time.as_secs_f64() * 1e9 / ITERS as f64);
        println!("int16 v3 (madd + shuffle/add, no hadd):    {:>8.2}ms  ({:>5.2}ns/pair)",
            i16_v3_time.as_secs_f64() * 1000.0, i16_v3_time.as_secs_f64() * 1e9 / ITERS as f64);
        println!("v2 vs fp32: {:.2}x   v3 vs fp32: {:.2}x",
            fp32_time.as_secs_f64() / i16_v2_time.as_secs_f64(),
            fp32_time.as_secs_f64() / i16_v3_time.as_secs_f64());
        let speedup = fp32_time.as_secs_f64() / i16_time.as_secs_f64();
        println!("\nspeedup: {speedup:.2}x");
        if speedup > 1.15 {
            println!("VERDICT: dual-packing shows a real primitive-level win — worth scoping the full kernel rewrite.");
        } else if speedup > 0.9 {
            println!("VERDICT: roughly a wash at the primitive level — packing overhead (repacking into contiguous i16 arrays) eats the multiply-instruction savings. NOT worth the full kernel rewrite as currently structured.");
        } else {
            println!("VERDICT: dual-packing is slower — packing overhead dominates. NOT worth pursuing this approach.");
        }
    }
}
