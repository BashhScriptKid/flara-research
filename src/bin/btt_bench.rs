use std::time::Instant;
use fydel::kernels::btt::{BttDict, BttMatmul, init_btt_coeffs_random};
use fydel::kernels::fft::{Fft, BasisMatmul};
use rustfft::num_complex::Complex32;

fn bench_one(name: &str, out_dim: usize, in_dim: usize, m1: usize, m2: usize, k: usize, mf: usize, n_shared: usize, iters: usize) {
    let mut dict = BttDict::new(n_shared, m1, m2, mf);
    let vals: Vec<f32> = (0..dict.dict1.len()).map(|i| {
        let x = (i as f32 * 0.017 + 1.0).sin();
        x * 0.1
    }).collect();
    dict.dict1.copy_from_slice(&vals);

    let matmul = BttMatmul::new(out_dim, in_dim, m1, m2, k, mf);
    let n_coeffs = matmul.coeff_len();
    let coeffs = init_btt_coeffs_random(n_coeffs, 0xDEAD, 0.5);
    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.013).sin()).collect();
    let dy: Vec<f32> = (0..out_dim).map(|i| (i as f32 * 0.017).cos()).collect();

    for _ in 0..2 {
        let _y = matmul.forward(&dict, &coeffs, &x);
        let _g = matmul.backward(&dict, &coeffs, &x, &dy);
    }

    let mut t_fwd = 0.0f64;
    for _ in 0..iters {
        let a = Instant::now();
        let _y = matmul.forward(&dict, &coeffs, &x);
        t_fwd += a.elapsed().as_secs_f64();
    }
    let fwd_ms = t_fwd / iters as f64 * 1e3;

    let mut t_bwd = 0.0f64;
    for _ in 0..iters {
        let a = Instant::now();
        let _g = matmul.backward(&dict, &coeffs, &x, &dy);
        t_bwd += a.elapsed().as_secs_f64();
    }
    let bwd_ms = t_bwd / iters as f64 * 1e3;

    let atoms = matmul.p * matmul.q * k;
    let (p, q) = (matmul.p, matmul.q);
    let total_ms = fwd_ms + bwd_ms;
    let ratio = bwd_ms / fwd_ms;
    eprintln!("{name:40}  out={out_dim:>5} in={in_dim:>5} P={p:>2} Q={q:>2} K={k:>2} atoms={atoms:>7}  fwd={fwd_ms:>8.3}ms  bwd={bwd_ms:>8.3}ms  total={total_ms:>8.3}ms  bwd/fwd={ratio:.1}x");
}

/// Benchmark the FFT-circulant BasisMatmul forward.
fn bench_fft_circ(name: &str, out_dim: usize, in_dim: usize, b: usize, k: usize, iters: usize) {
    let basis = BasisMatmul::new(out_dim, in_dim, b, k);
    let n_coeffs = basis.coeff_len();
    let coeffs = init_btt_coeffs_random(n_coeffs, 0xBEEF, 0.5);

    // Complex dictionary: K × b
    let mut rng_state: u64 = 0xCAFE;
    let dict: Vec<Complex32> = (0..k * b).map(|_| {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let re = ((rng_state >> 40) as f32 / (1u64 << 23) as f32) - 0.5;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let im = ((rng_state >> 40) as f32 / (1u64 << 23) as f32) - 0.5;
        Complex32::new(re, im)
    }).collect();

    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.013).sin()).collect();

    for _ in 0..2 {
        let _y = basis.forward(&dict, &coeffs, &x);
    }

    let mut t_fwd = 0.0f64;
    for _ in 0..iters {
        let a = Instant::now();
        let _y = basis.forward(&dict, &coeffs, &x);
        t_fwd += a.elapsed().as_secs_f64();
    }
    let fwd_ms = t_fwd / iters as f64 * 1e3;

    let (p, q) = (basis.p, basis.q);
    eprintln!("{name:40}  out={out_dim:>5} in={in_dim:>5} P={p:>2} Q={q:>2} K={k:>2}  fwd={fwd_ms:>8.3}ms", name=name);
}

/// Benchmark the circulant Kronecker forward (FFT-based).
fn bench_circ_kron(name: &str, out_dim: usize, in_dim: usize, mf: usize, k: usize, iters: usize) {
    use fydel::kernels::btt::{precompute_circulant_dfts, circulant_kron_forward};

    let b = mf * mf;
    let p = out_dim / b;
    let q = in_dim / b;

    // Generate random circulant first-rows for K atoms
    let mut rng_state: u64 = 0xABCD;
    let atoms_a: Vec<f32> = (0..k * mf).map(|_| {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng_state >> 40) as f32 / (1u64 << 23) as f32) - 0.5
    }).collect();
    let atoms_b: Vec<f32> = (0..k * mf).map(|_| {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng_state >> 40) as f32 / (1u64 << 23) as f32) - 0.5
    }).collect();

    // Precompute DFTs
    let (dft_a, dft_b) = precompute_circulant_dfts(mf, k, &atoms_a, &atoms_b);

    // Coefficients
    let n_coeffs = p * q * k;
    let coeffs = init_btt_coeffs_random(n_coeffs, 0xDEAD, 0.5);

    // Input
    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.013).sin()).collect();

    // Scratch buffers
    let sz = mf * mf;
    let mut y = vec![0.0f32; out_dim];
    let mut freq_acc_re = vec![0.0f32; sz];
    let mut freq_acc_im = vec![0.0f32; sz];
    let mut fft_x_re = vec![0.0f32; sz];
    let mut fft_x_im = vec![0.0f32; sz];

    // Warmup
    for _ in 0..2 {
        circulant_kron_forward(mf, p, q, k, &dft_a, &dft_b, &coeffs, &x, &mut y,
            &mut freq_acc_re, &mut freq_acc_im, &mut fft_x_re, &mut fft_x_im);
    }

    // Forward
    let mut t_fwd = 0.0f64;
    for _ in 0..iters {
        let a = Instant::now();
        circulant_kron_forward(mf, p, q, k, &dft_a, &dft_b, &coeffs, &x, &mut y,
            &mut freq_acc_re, &mut freq_acc_im, &mut fft_x_re, &mut fft_x_im);
        t_fwd += a.elapsed().as_secs_f64();
    }
    let fwd_ms = t_fwd / iters as f64 * 1e3;

    eprintln!("{name:40}  out={out_dim:>5} in={in_dim:>5} P={p:>2} Q={q:>2} K={k:>2}  fwd={fwd_ms:>8.3}ms", name=name);
}

fn main() {
    let iters: usize = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);

    let mf = 8;
    let block = mf * mf; // 64
    let n_shared = 32;
    let k = 32;

    // === Existing dense Kronecker BTT benchmark ===
    eprintln!("=== Dense Kronecker BTT micro-benchmark (iters={iters}) ===\n");

    eprintln!("--- Production AttnProj (hidden=896) ---");
    bench_one("AttnProj 896x896", 896, 896, block, block, k, mf, n_shared, iters);

    eprintln!("\n--- Production FFN (896->3072) ---");
    bench_one("FFN 3072x896", 3072, 896, block, block, k, mf, n_shared, iters);

    // === FFT-circulant baseline ===
    eprintln!("\n=== FFT-circulant baseline (iters={iters}) ===\n");

    bench_fft_circ("FFT-circ AttnProj 896x896", 896, 896, block, k, iters);
    bench_fft_circ("FFT-circ FFN 3072x896", 3072, 896, block, k, iters);

    // === Circulant Kronecker (Monarch-style) ===
    eprintln!("\n=== Circulant Kronecker BTT (iters={iters}) ===\n");

    bench_circ_kron("CircKron AttnProj 896x896 K=32", 896, 896, mf, 32, iters);
    bench_circ_kron("CircKron FFN 3072x896 K=32", 3072, 896, mf, 32, iters);

    // === Scale test: vary K ===
    eprintln!("\n=== Scale test (FFN 3072x896) ===\n");
    for k_val in [1, 2, 4, 8, 16, 32] {
        bench_circ_kron(&format!("CircKron FFN 3072x896 K={k_val:<2}"), 3072, 896, mf, k_val, iters);
        bench_fft_circ(&format!("FFT-circ FFN 3072x896 K={k_val:<2}"), 3072, 896, block, k_val, iters);
    }
}
