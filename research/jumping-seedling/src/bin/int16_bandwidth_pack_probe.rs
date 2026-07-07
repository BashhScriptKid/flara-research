// Follow-up to int16_dualblock_spike.rs: that spike isolated the ALU-side
// question (does dual-packed int16 madd beat fp32 mul+hadd?) and found a
// real but modest ~15% win once the reduction was fixed to mirror matvec8's
// pattern (v3). Fable's report flagged two OTHER levers not yet tested,
// which this probe isolates:
//
//   (a) Memory bandwidth: does storing weights as int16 (half the bytes)
//       help once the working set is large enough to leave L1/L2, i.e. is
//       a real bandwidth-bound regime, unlike the tiny all-in-cache spike?
//   (b) Pack amortization: every probe so far repacks weights into the
//       interleaved int16 layout on EVERY call. Real weights are fixed for
//       a whole batch/step and reused across many tokens — pre-packing once
//       and reusing removes that cost from the hot per-token loop entirely.
//
// Goal: quantify both before deciding whether a full kernel rewrite is
// worth aggregating them into, rather than patching the ALU win in alone.

use std::arch::x86_64::*;
use std::time::Instant;

const NUM_PAIRS: usize = 65_536; // 65536 pairs x 2 blocks x 8 x4B(f32) = 4MiB fp32 weight data -> exceeds typical L2, forces real memory traffic
const TOKENS: usize = 64; // sweeps of the full weight set, mimicking reuse across a batch

#[target_feature(enable = "avx2,fma")]
unsafe fn dual_dot8_f32(mat0: &[f32], mat1: &[f32], vec: &[f32; 8]) -> (f32, f32) {
    let v = _mm256_loadu_ps(vec.as_ptr());
    let p0 = _mm256_mul_ps(_mm256_loadu_ps(mat0.as_ptr()), v);
    let p1 = _mm256_mul_ps(_mm256_loadu_ps(mat1.as_ptr()), v);
    let h0 = _mm256_hadd_ps(p0, p1);
    let h1 = _mm256_hadd_ps(h0, h0);
    let s = _mm_add_ps(_mm256_castps256_ps128(h1), _mm256_extractf128_ps(h1, 1));
    (_mm_cvtss_f32(s), _mm_cvtss_f32(_mm_shuffle_ps(s, s, 0x01)))
}

/// v3 reduction from int16_dualblock_spike.rs (the fastest verified variant).
/// `packed_mat` is already in the dual-block-interleaved i16 layout
/// (block0 in low 8 lanes, block1 in high 8 lanes) — packing cost NOT
/// included here, so this measures the steady-state bandwidth-bound cost
/// once packing is already amortized.
#[target_feature(enable = "avx2")]
unsafe fn dual_dot8_i16_prepacked(packed_mat: &[i16; 16], packed_vec: &[i16; 16]) -> (i32, i32) {
    let m = _mm256_loadu_si256(packed_mat.as_ptr() as *const __m256i);
    let v = _mm256_loadu_si256(packed_vec.as_ptr() as *const __m256i);
    let prod = _mm256_madd_epi16(m, v);
    let t1 = _mm256_add_epi32(prod, _mm256_shuffle_epi32(prod, 0b0100_1110));
    let t2 = _mm256_add_epi32(t1, _mm256_shuffle_epi32(t1, 0b0000_0001));
    let lo = _mm256_castsi256_si128(t2);
    let hi = _mm256_extracti128_si256(t2, 1);
    (_mm_cvtsi128_si32(lo), _mm_cvtsi128_si32(hi))
}

#[target_feature(enable = "avx2")]
unsafe fn pack_pair(mat0: &[i16], mat1: &[i16]) -> [i16; 16] {
    let mut out = [0i16; 16];
    out[0..8].copy_from_slice(mat0);
    out[8..16].copy_from_slice(mat1);
    out
}

fn make_weights(n_pairs: usize) -> (Vec<f32>, Vec<i16>) {
    let mut seed: u64 = 0x1234_5678_9ABC_DEF0;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let n = n_pairs * 2 * 8; // 2 blocks per pair, 8 elements per block
    let f32_data: Vec<f32> = (0..n).map(|_| (next() as f32 - 0.5) * 4.0).collect();
    let i16_data: Vec<i16> = f32_data.iter().map(|&v| (v * 4096.0).clamp(-32768.0, 32767.0) as i16).collect();
    (f32_data, i16_data)
}

fn main() {
    if !is_x86_feature_detected!("avx2") {
        println!("AVX2 not available — probe cannot run.");
        return;
    }
    let (weights_f32, weights_i16) = make_weights(NUM_PAIRS);
    let vec_f32 = [1.0f32, 0.5, -0.5, 2.0, 1.0, -1.0, 0.25, 1.5];
    let vec_i16: [i16; 8] = std::array::from_fn(|i| (vec_f32[i] * 4096.0) as i16);
    let packed_vec: [i16; 16] = { let mut v = [0i16; 16]; v[0..8].copy_from_slice(&vec_i16); v[8..16].copy_from_slice(&vec_i16); v };

    let weight_bytes_f32 = weights_f32.len() * 4;
    let weight_bytes_i16 = weights_i16.len() * 2;
    println!("weight set: {NUM_PAIRS} pairs, {weight_bytes_f32} bytes (fp32) / {weight_bytes_i16} bytes (i16), swept {TOKENS} times\n");

    unsafe {
        // --- Condition A: fp32 baseline, full sweep x TOKENS ---
        let mut acc = 0.0f32;
        let start = Instant::now();
        for _ in 0..TOKENS {
            for p in 0..NUM_PAIRS {
                let off = p * 16;
                let (a, b) = dual_dot8_f32(
                    std::hint::black_box(&weights_f32[off..off + 8]),
                    std::hint::black_box(&weights_f32[off + 8..off + 16]),
                    std::hint::black_box(&vec_f32),
                );
                acc += a + b;
            }
        }
        let t_f32 = start.elapsed();
        std::hint::black_box(acc);

        // --- Condition B: int16, PRE-PACKED ONCE before the token loop ---
        // (isolates bandwidth: packing cost paid once, not per token)
        let packed_weights: Vec<[i16; 16]> = (0..NUM_PAIRS)
            .map(|p| { let off = p * 16; pack_pair(&weights_i16[off..off + 8], &weights_i16[off + 8..off + 16]) })
            .collect();
        let mut acc_i: i64 = 0;
        let start = Instant::now();
        for _ in 0..TOKENS {
            for p in 0..NUM_PAIRS {
                let (a, b) = dual_dot8_i16_prepacked(std::hint::black_box(&packed_weights[p]), std::hint::black_box(&packed_vec));
                acc_i += a as i64 + b as i64;
            }
        }
        let t_i16_prepacked = start.elapsed();
        std::hint::black_box(acc_i);

        // --- Condition C: int16, REPACKED EVERY TOKEN (what every probe so far actually did) ---
        let mut acc_i2: i64 = 0;
        let start = Instant::now();
        for _ in 0..TOKENS {
            for p in 0..NUM_PAIRS {
                let off = p * 16;
                let packed = pack_pair(std::hint::black_box(&weights_i16[off..off + 8]), std::hint::black_box(&weights_i16[off + 8..off + 16]));
                let (a, b) = dual_dot8_i16_prepacked(&packed, std::hint::black_box(&packed_vec));
                acc_i2 += a as i64 + b as i64;
            }
        }
        let t_i16_repacked = start.elapsed();
        std::hint::black_box(acc_i2);

        let ops = (NUM_PAIRS * TOKENS) as f64;
        println!("A) fp32 baseline:                {:>8.2}ms  ({:>5.3}ns/pair)", t_f32.as_secs_f64() * 1000.0, t_f32.as_secs_f64() * 1e9 / ops);
        println!("B) int16, pre-packed once:        {:>8.2}ms  ({:>5.3}ns/pair)", t_i16_prepacked.as_secs_f64() * 1000.0, t_i16_prepacked.as_secs_f64() * 1e9 / ops);
        println!("C) int16, repacked every token:   {:>8.2}ms  ({:>5.3}ns/pair)", t_i16_repacked.as_secs_f64() * 1000.0, t_i16_repacked.as_secs_f64() * 1e9 / ops);

        let bw_speedup = t_f32.as_secs_f64() / t_i16_prepacked.as_secs_f64();
        let pack_cost_fraction = (t_i16_repacked.as_secs_f64() - t_i16_prepacked.as_secs_f64()) / t_i16_repacked.as_secs_f64() * 100.0;

        println!("\n(a) bandwidth effect (A vs B, packing cost excluded): {bw_speedup:.2}x");
        println!("(b) repacking-every-token overhead as % of int16's own time: {pack_cost_fraction:.1}%");

        if bw_speedup > 1.15 {
            println!("\nVERDICT (a): real bandwidth win once the working set exceeds cache — worth pursuing in a real kernel.");
        } else if bw_speedup > 0.95 {
            println!("\nVERDICT (a): roughly a wash even at this working-set size — bandwidth alone isn't the lever Fable's hypothesis needs.");
        } else {
            println!("\nVERDICT (a): int16 storage is SLOWER even bandwidth-side at this scale — unexpected, worth double-checking before trusting.");
        }
        if pack_cost_fraction > 15.0 {
            println!("VERDICT (b): repacking every token is a real, avoidable tax — pre-packing weights once per step is worth doing in a real kernel.");
        } else {
            println!("VERDICT (b): repacking cost is small relative to the compute itself — amortization matters less than expected.");
        }
    }
}
