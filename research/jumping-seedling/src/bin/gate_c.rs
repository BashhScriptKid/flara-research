//! Gate (c): SharedMonarchMatmul — P×Q tiling of b×b Monarch blocks with shared
//! atom dictionaries — wired as a full projection and benchmarked vs BasisMatmul.
//!
//! Three checks:
//!   1. Gradcheck (FD vs analytical, small dims)
//!   2. Timing vs BasisMatmul at FFN 3072×896 dims (P=14, Q=48, b=64)
//!   3. Same-family overfit with cosine-decayed Adam at nd=8

use std::time::Instant;

use fydel::kernels::fft::BasisMatmul;
use fydel::kernels::monarch::SharedMonarchMatmul;
use rustfft::num_complex::Complex32;

// ---------------------------------------------------------------------------
// LCG rng
// ---------------------------------------------------------------------------

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f()).collect()
    }
    fn vec_scaled(&mut self, n: usize, s: f32) -> Vec<f32> {
        (0..n).map(|_| self.f() * s).collect()
    }
    fn next_seed(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
}

// ---------------------------------------------------------------------------
// Gradcheck (finite differences vs analytical backward)
// ---------------------------------------------------------------------------

fn mse_loss(out: &[f32], target: &[f32]) -> f32 {
    out.iter().zip(target).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / out.len() as f32
}

fn mse_grad(out: &[f32], target: &[f32]) -> Vec<f32> {
    let n = out.len() as f32;
    out.iter().zip(target).map(|(a, b)| 2.0 * (a - b) / n).collect()
}

fn gradcheck(p: usize, q: usize, m: usize, nd: usize, rng: &mut Lcg) -> bool {
    let b = m * m;
    let in_dim = q * b;
    let out_dim = p * b;

    let mut mm = SharedMonarchMatmul::new(p, q, m, nd, rng.next_seed());
    let x: Vec<f32> = rng.vec(in_dim);
    let target: Vec<f32> = rng.vec(out_dim);

    let eps = 1e-3f32;
    let (out, cache) = mm.forward(&x);
    let dloss = mse_grad(&out, &target);
    let mut _dx = vec![0.0f32; in_dim];
    let grads = mm.backward(&x, &cache.zs, &dloss, &mut _dx);

    let mut max_err = 0.0f32;
    let mut checked = 0usize;

    for idx in (0..mm.a1.len()).step_by(mm.a1.len() / 16 + 1).take(16) {
        let orig = mm.a1[idx];
        mm.a1[idx] = orig + eps;
        let (out_p, _) = mm.forward(&x);
        mm.a1[idx] = orig - eps;
        let (out_m, _) = mm.forward(&x);
        mm.a1[idx] = orig;
        let fd = (mse_loss(&out_p, &target) - mse_loss(&out_m, &target)) / (2.0 * eps);
        max_err = max_err.max((fd - grads.da1[idx]).abs());
        checked += 1;
    }

    for idx in (0..mm.d1.len()).step_by(mm.d1.len() / 8 + 1).take(8) {
        let orig = mm.d1[idx];
        mm.d1[idx] = orig + eps;
        let (out_p, _) = mm.forward(&x);
        mm.d1[idx] = orig - eps;
        let (out_m, _) = mm.forward(&x);
        mm.d1[idx] = orig;
        let fd = (mse_loss(&out_p, &target) - mse_loss(&out_m, &target)) / (2.0 * eps);
        max_err = max_err.max((fd - grads.dd1[idx]).abs());
        checked += 1;
    }

    eprintln!("  gradcheck ({p}x{q} blocks, m={m}, nd={nd}): max_err={:.2e}  [{checked} params]  {}",
        max_err, if max_err < 0.05 { "PASS" } else { "FAIL" });
    max_err < 0.05
}

// ---------------------------------------------------------------------------
// Benchmark vs BasisMatmul
// ---------------------------------------------------------------------------

fn bench(out_dim: usize, in_dim: usize, b: usize, nd: usize, k: usize, iters: usize, rng: &mut Lcg) {
    let m = (b as f64).sqrt() as usize;
    assert_eq!(m * m, b, "b must be a perfect square");
    let p = out_dim / b;
    let q = in_dim / b;

    let mm = SharedMonarchMatmul::new(p, q, m, nd, rng.next_seed());
    let x: Vec<f32> = rng.vec(in_dim);

    // forward_inference (not forward) — BasisMatmul::forward below writes no
    // backward cache, so timing SharedMonarch's cache-writing `forward` would
    // unfairly tax it for work the comparison isn't asking for.
    for _ in 0..200 { let _ = std::hint::black_box(mm.forward_inference(&x)); }
    let t = Instant::now();
    for _ in 0..iters { let _ = std::hint::black_box(mm.forward_inference(&x)); }
    let mon_us = t.elapsed().as_secs_f64() / iters as f64 * 1e6;

    let dout: Vec<f32> = rng.vec(out_dim);
    let (_, cache) = mm.forward(&x);
    let mut dx_bench = vec![0.0f32; in_dim];
    for _ in 0..200 { dx_bench.fill(0.0); let _ = std::hint::black_box(mm.backward(&x, &cache.zs, &dout, &mut dx_bench)); }
    let t = Instant::now();
    for _ in 0..iters { dx_bench.fill(0.0); let _ = std::hint::black_box(mm.backward(&x, &cache.zs, &dout, &mut dx_bench)); }
    let mon_bwd_us = t.elapsed().as_secs_f64() / iters as f64 * 1e6;

    let basis = BasisMatmul::new(out_dim, in_dim, b, k);
    let n_dict = k * b;
    let dict: Vec<Complex32> = (0..n_dict).map(|_| Complex32::new(rng.f(), rng.f())).collect();
    let coeffs: Vec<f32> = (0..basis.coeff_len()).map(|_| rng.f() * 0.1).collect();

    for _ in 0..200 { let _ = std::hint::black_box(basis.forward(&dict, &coeffs, &x)); }
    let t = Instant::now();
    for _ in 0..iters { let _ = std::hint::black_box(basis.forward(&dict, &coeffs, &x)); }
    let basis_us = t.elapsed().as_secs_f64() / iters as f64 * 1e6;

    // BasisMatmul backward — previously never timed here, so every prior
    // "crossover" measurement from this function was forward-only. Training
    // cost is fwd+bwd, and Monarch's backward carries an O(nd) per-token
    // gradient-accumulation tax that forward-only hoisting can't remove (see
    // RESEARCH_LOG.md 2026-07-03, Opus review) — so the real crossover is
    // expected to sit higher than the forward-only one below.
    for _ in 0..200 { let _ = std::hint::black_box(basis.backward(&dict, &coeffs, &x, &dout)); }
    let t = Instant::now();
    for _ in 0..iters { let _ = std::hint::black_box(basis.backward(&dict, &coeffs, &x, &dout)); }
    let basis_bwd_us = t.elapsed().as_secs_f64() / iters as f64 * 1e6;

    let mon_params = nd * b * 2 + p * q * m * nd * 2;
    let basis_params = k * b * 2 + p * q * k;
    let mon_total = mon_us + mon_bwd_us;
    let basis_total = basis_us + basis_bwd_us;
    eprintln!(
        "  {out_dim}x{in_dim}  SharedMonarch(nd={nd}, params={mon_params}): fwd={:>7.2}µs  bwd={:>7.2}µs  BasisMatmul(K={k}, params={basis_params}): fwd={:>7.2}µs  bwd={:>7.2}µs  fwd-only speedup={:.2}×  fwd+bwd speedup={:.2}×",
        mon_us, mon_bwd_us, basis_us, basis_bwd_us, basis_us / mon_us, basis_total / mon_total
    );
}

// ---------------------------------------------------------------------------
// Cosine-decayed Adam
// ---------------------------------------------------------------------------

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
    b1: f32,
    b2: f32,
    eps: f32,
}

impl Adam {
    fn new(n: usize) -> Self {
        Self { m: vec![0.0; n], v: vec![0.0; n], t: 0, b1: 0.9, b2: 0.999, eps: 1e-8 }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        self.t += 1;
        let (b1, b2, eps) = (self.b1, self.b2, self.eps);
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        for i in 0..params.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * grads[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * grads[i] * grads[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

fn cosine_lr(step: usize, lr_max: f32, t_max: usize) -> f32 {
    lr_max * 0.5 * (1.0 + (std::f32::consts::PI * step as f32 / t_max as f32).cos())
}

// ---------------------------------------------------------------------------
// Training: same-family overfit
// ---------------------------------------------------------------------------

fn train(p: usize, q: usize, m: usize, nd: usize, rng: &mut Lcg) {
    let b = m * m;
    let in_dim = q * b;

    let teacher = SharedMonarchMatmul::new(p, q, m, nd, rng.next_seed());
    let x: Vec<f32> = rng.vec(in_dim);
    let (target, _) = teacher.forward(&x);

    let mut student = SharedMonarchMatmul::new(p, q, m, nd, rng.next_seed());

    let n_d1 = nd * b;
    let n_d2 = nd * b;
    let n_a1 = p * q * m * nd;
    let n_a2 = p * q * m * nd;

    let mut opt_d1 = Adam::new(n_d1);
    let mut opt_d2 = Adam::new(n_d2);
    let mut opt_a1 = Adam::new(n_a1);
    let mut opt_a2 = Adam::new(n_a2);

    let t_max = 2000usize;
    let lr_max = 1e-3f32;
    let print_at = [1, 100, 500, 1000, 2000];

    eprintln!("  training (p={p}, q={q}, m={m}, nd={nd}, steps={t_max}, lr_max={lr_max}):");
    for step in 1..=t_max {
        let (out, cache) = student.forward(&x);
        let loss = mse_loss(&out, &target);
        if print_at.contains(&step) {
            eprintln!("    step {:>5}  loss={:.6}", step, loss);
        }
        let dloss = mse_grad(&out, &target);
        let mut _dx = vec![0.0f32; in_dim];
        let g = student.backward(&x, &cache.zs, &dloss, &mut _dx);
        let lr = cosine_lr(step, lr_max, t_max);
        opt_d1.step(&mut student.d1, &g.dd1, lr);
        opt_d2.step(&mut student.d2, &g.dd2, lr);
        opt_a1.step(&mut student.a1, &g.da1, lr);
        opt_a2.step(&mut student.a2, &g.da2, lr);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let mut rng = Lcg(0xFEED_BEEF_1234_5678);

    eprintln!("=== Gradcheck ===");
    let pass1 = gradcheck(2, 2, 4, 4, &mut rng);
    let pass2 = gradcheck(3, 2, 4, 8, &mut rng);
    let pass3 = gradcheck(2, 3, 8, 8, &mut rng);
    let all_pass = pass1 && pass2 && pass3;
    eprintln!("  overall: {}", if all_pass { "PASS" } else { "FAIL" });

    if !all_pass {
        eprintln!("Gradcheck failed — aborting.");
        std::process::exit(1);
    }

    let iters = 5000;
    eprintln!("\n=== Timing (iters={iters}) ===");
    bench(896, 3072, 64, 8, 32, iters, &mut rng);
    bench(896, 896,  64, 8, 32, iters, &mut rng);

    // train_small_lod's actual attention-projection shapes (hidden=256, block=64,
    // dict_k=8) — SharedMonarch(nd=8) is what AttnProj now uses; BasisMatmul(k=8)
    // is what it used before the swap. Same nd/k so the comparison isolates the
    // kernel, not a compression-ratio difference.
    eprintln!("\n=== Timing @ train_small_lod attention-projection shapes (iters={iters}) ===");
    bench(256, 256, 64, 8, 8, iters, &mut rng); // wq / wo
    bench(64,  256, 64, 8, 8, iters, &mut rng); // wk / wv

    // Same shapes, but at true equal parameter count (K=2·m·nd=128 at m=8,
    // nd=8) instead of nd=k=8 — the row above hands Monarch 16x more
    // per-block params than BasisMatmul, so it isn't isolating the kernel,
    // it's mostly measuring a capacity difference. This is the fair version.
    eprintln!("\n=== Timing @ train_small_lod shapes, equal params (K=2*m*nd=128) (iters={iters}) ===");
    bench(256, 256, 64, 8, 128, iters, &mut rng); // wq / wo
    bench(64,  256, 64, 8, 128, iters, &mut rng); // wk / wv

    // Crossover search: square P×Q grids at nd=k=8, block=64, between the two
    // known points (256x256 loses at 0.2x, 896x896 wins at 1.3x).
    eprintln!("\n=== Crossover search: square grids, nd=k=8 (iters={iters}) ===");
    bench(384, 384, 64, 8, 8, iters, &mut rng); // P=Q=6
    bench(512, 512, 64, 8, 8, iters, &mut rng); // P=Q=8
    bench(640, 640, 64, 8, 8, iters, &mut rng); // P=Q=10
    bench(768, 768, 64, 8, 8, iters, &mut rng); // P=Q=12
    bench(832, 832, 64, 8, 8, iters, &mut rng); // P=Q=13
    bench(896, 896, 64, 8, 8, iters, &mut rng); // P=Q=14, k=8 (isolate grid size from the k=32 default)

    eprintln!("\n=== nd/k matched at production grid size (896x896, iters={iters}) ===");
    bench(896, 896, 64, 16, 16, iters, &mut rng);
    bench(896, 896, 64, 32, 32, iters, &mut rng); // matches the original "1.3x win" comparison's k, but with nd=k this time
    bench(256, 256, 64, 32, 32, iters, &mut rng); // does bumping nd=k=32 flip the toy-scale result too?

    // True equal-capacity comparison (Opus-derived): SharedMonarch has 2·m·nd
    // coeffs/block-pair vs BasisMatmul's K, so equal params needs K=2·m·nd=16·nd
    // at m=8. K=64 is the equal-*FLOP* point instead (BasisMatmul unpadded,
    // ratio m·nd/K=1) — both matter since equal-param can pad BasisMatmul past
    // its expressibility ceiling. nd anchored at 8 (monarch.rs: full-rank needs
    // nd≥8).
    eprintln!("\n=== Equal-capacity comparison, nd=8 (iters={iters}) ===");
    eprintln!("-- K=128 (equal param count) --");
    bench(896, 3072, 64, 8, 128, iters, &mut rng);
    bench(896, 896,  64, 8, 128, iters, &mut rng);
    eprintln!("-- K=64 (equal FLOPs, BasisMatmul unpadded) --");
    bench(896, 3072, 64, 8, 64, iters, &mut rng);
    bench(896, 896,  64, 8, 64, iters, &mut rng);

    eprintln!("\n=== Training (same-family overfit, nd=8) ===");
    train(4, 4, 8, 8, &mut rng);

    eprintln!("\n=== Training (scaled, nd=8) ===");
    train(6, 4, 8, 8, &mut rng);
}
