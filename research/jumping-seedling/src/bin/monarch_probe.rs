//! Gate (a) for the Monarch/BTT-hybrid decision: does a Monarch-style block-GEMM
//! primitive hit a meaningfully higher fraction of the AVX2 roofline than the current
//! FFT-circulant `BasisMatmul` block, at the same projection size?
//!
//! All contractions go through the SAME `gemm::dot` AVX2 kernel the rest of the model
//! uses, so the comparison isolates the *structure* (real block-GEMM vs FFT-circulant),
//! not implementation sophistication. Per-token (matvec) is the apples-to-apples number
//! vs the existing `BasisMatmul` forward; the batched (L1-resident weights) number shows
//! the training-mode roofline.

use std::time::Instant;

use fydel::kernels::gemm;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f()).collect()
    }
}

/// Order-2 Monarch matvec for `n = m1*m2`: a block-diagonal GEMM (m1 blocks of m2xm2),
/// a transpose, then a second block-diagonal GEMM (m2 blocks of m1xm1). Output layout is
/// irrelevant to timing, so we leave the result in the post-step-2 order.
fn monarch_matvec(x: &[f32], a: &[f32], b: &[f32], m1: usize, m2: usize, scratch: &mut [f32], out: &mut [f32]) {
    // step 1: y[i] = A_i @ x[i], blocks of size m2xm2 over the m1 row-chunks of x.
    for i in 0..m1 {
        let xi = &x[i * m2..(i + 1) * m2];
        for r in 0..m2 {
            let row = &a[(i * m2 + r) * m2..(i * m2 + r) * m2 + m2];
            scratch[i * m2 + r] = gemm::dot(row, xi);
        }
    }
    // transpose [m1][m2] -> [m2][m1] into out as temporary holding.
    for i in 0..m1 {
        for j in 0..m2 {
            out[j * m1 + i] = scratch[i * m2 + j];
        }
    }
    // step 2: w[j] = B_j @ z[j], blocks of size m1xm1 over the m2 row-chunks of z.
    for j in 0..m2 {
        let zj = &out[j * m1..(j + 1) * m1];
        for r in 0..m1 {
            let row = &b[(j * m1 + r) * m1..(j * m1 + r) * m1 + m1];
            scratch[j * m1 + r] = gemm::dot(row, zj);
        }
    }
    out.copy_from_slice(scratch);
}

fn dense_matvec(x: &[f32], w: &[f32], n: usize, out: &mut [f32]) {
    for i in 0..n {
        out[i] = gemm::dot(&w[i * n..(i + 1) * n], x);
    }
}

fn gflops(flops: f64, secs: f64) -> f64 {
    flops / secs / 1e9
}

fn main() {
    let n = 512usize;
    let (m1, m2) = (16usize, 32usize);
    assert_eq!(m1 * m2, n);
    let mut rng = Lcg(0x1234_5678);

    let a = rng.vec(m1 * m2 * m2);
    let b = rng.vec(m2 * m1 * m1);
    let w = rng.vec(n * n);
    let x = rng.vec(n);
    let mut scratch = vec![0.0f32; n];
    let mut out = vec![0.0f32; n];

    let monarch_flops = 2.0 * (m1 * m2 * m2 + m2 * m1 * m1) as f64; // ~49 KFLOP
    let dense_flops = 2.0 * (n * n) as f64; // ~524 KFLOP
    let basis_fwd_us = 19.45; // measured: BasisMatmul 512x512 forward, us/token

    // warmup
    for _ in 0..2000 {
        monarch_matvec(&x, &a, &b, m1, m2, &mut scratch, &mut out);
        dense_matvec(&x, &w, n, &mut out);
    }

    let iters = 200_000u64;

    let t = Instant::now();
    for _ in 0..iters {
        monarch_matvec(&x, &a, &b, m1, m2, &mut scratch, &mut out);
        std::hint::black_box(&out);
    }
    let mon_s = t.elapsed().as_secs_f64() / iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        dense_matvec(&x, &w, n, &mut out);
        std::hint::black_box(&out);
    }
    let den_s = t.elapsed().as_secs_f64() / iters as f64;

    // batched (T tokens, weights stay L1/L2-resident) -> training-mode throughput.
    let t_tok = 256usize;
    let xb = rng.vec(n * t_tok);
    let t = Instant::now();
    for _ in 0..2000u64 {
        for tk in 0..t_tok {
            monarch_matvec(&xb[tk * n..(tk + 1) * n], &a, &b, m1, m2, &mut scratch, &mut out);
            std::hint::black_box(&out);
        }
    }
    let mon_batch_s = t.elapsed().as_secs_f64() / (2000 * t_tok) as f64;

    let t = Instant::now();
    for _ in 0..2000u64 {
        for tk in 0..t_tok {
            dense_matvec(&xb[tk * n..(tk + 1) * n], &w, n, &mut out);
            std::hint::black_box(&out);
        }
    }
    let den_batch_s = t.elapsed().as_secs_f64() / (2000 * t_tok) as f64;

    println!("n={n}, monarch blocking ({m1}x{m2}), gemm::dot AVX2 inner product\n");
    println!("{:<28} {:>10} {:>12} {:>10}", "primitive", "us/token", "MFLOP/op", "GFLOP/s");
    println!(
        "{:<28} {:>10.3} {:>12.3} {:>10.1}",
        "dense 512x512 (matvec)", den_s * 1e6, dense_flops / 1e6, gflops(dense_flops, den_s)
    );
    println!(
        "{:<28} {:>10.3} {:>12.3} {:>10.1}",
        "monarch 512 (matvec)", mon_s * 1e6, monarch_flops / 1e6, gflops(monarch_flops, mon_s)
    );
    println!(
        "{:<28} {:>10.3} {:>12.3} {:>10}",
        "BasisMatmul 512 (FFT, ref)", basis_fwd_us, "n/a", "—"
    );
    println!("\nbatched T=256 (weights L1/L2-resident, training mode):");
    println!(
        "{:<28} {:>10.3} {:>12} {:>10.1}",
        "dense 512x512 (matmul)", den_batch_s * 1e6, "", gflops(dense_flops, den_batch_s)
    );
    println!(
        "{:<28} {:>10.3} {:>12} {:>10.1}",
        "monarch 512 (matmul)", mon_batch_s * 1e6, "", gflops(monarch_flops, mon_batch_s)
    );
    println!("\n--- verdict ---");
    println!("monarch vs BasisMatmul (per-token fwd): {:.1}x", basis_fwd_us / (mon_s * 1e6));
    println!("monarch vs dense (per-token fwd):       {:.2}x  ({:.0}% fewer FLOPs)",
        den_s / mon_s, (1.0 - monarch_flops / dense_flops) * 100.0);
}
