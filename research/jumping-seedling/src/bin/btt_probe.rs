//! Gate (b) for the shared-core BTT hybrid: does a Monarch-shaped block whose blocks
//! are linear combinations of a SHARED atom dictionary stay full-rank and trainable at
//! a real compression target? This is the load-bearing unknown after gate (a) proved
//! the GEMM realization is ~10x faster than the FFT one.
//!
//! Standalone and isolated from the FFT path. Real-valued. Order-2 Monarch:
//!   n = m1*m2; X[i][j] = x[i*m2+j].
//!   stage 1: block1_i (m2xm2) = sum_d a1[i][d]*D1[d];  y[i] = block1_i @ X[i]
//!   transpose: z[j][i] = y[i][j]
//!   stage 2: block2_j (m1xm1) = sum_d a2[j][d]*D2[d];  w[j] = block2_j @ z[j]
//!   out[j*m1+r] = w[j][r]
//!
//! Three checks: (1) gradcheck the backward, (2) numerical rank of the effective n*n
//! weight as nd sweeps, (3) overfit a random dense target.

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

#[derive(Clone)]
struct Block {
    m1: usize,
    m2: usize,
    nd: usize,
    d1: Vec<f32>, // nd * m2 * m2
    d2: Vec<f32>, // nd * m1 * m1
    a1: Vec<f32>, // m1 * nd   (per-weight coefficients, stage 1)
    a2: Vec<f32>, // m2 * nd   (per-weight coefficients, stage 2)
}

struct Fwd {
    y: Vec<f32>,  // m1*m2
    z: Vec<f32>,  // m2*m1
    out: Vec<f32>,
}

#[derive(Default)]
struct Grads {
    da1: Vec<f32>,
    da2: Vec<f32>,
    dd1: Vec<f32>,
    dd2: Vec<f32>,
}

impl Block {
    fn new(m1: usize, m2: usize, nd: usize, rng: &mut Lcg) -> Self {
        // Atoms scaled so block ~ sum of nd atoms is O(1); coeffs near 1/nd.
        let s1 = 1.0 / (m2 as f32).sqrt();
        let s2 = 1.0 / (m1 as f32).sqrt();
        let d1: Vec<f32> = (0..nd * m2 * m2).map(|_| rng.f() * 2.0 * s1).collect();
        let d2: Vec<f32> = (0..nd * m1 * m1).map(|_| rng.f() * 2.0 * s2).collect();
        let a1: Vec<f32> = (0..m1 * nd).map(|_| rng.f() * 2.0).collect();
        let a2: Vec<f32> = (0..m2 * nd).map(|_| rng.f() * 2.0).collect();
        Block { m1, m2, nd, d1, d2, a1, a2 }
    }
    fn n(&self) -> usize {
        self.m1 * self.m2
    }

    fn forward(&self, x: &[f32]) -> Fwd {
        let (m1, m2, nd) = (self.m1, self.m2, self.nd);
        let mut y = vec![0.0f32; m1 * m2];
        // stage 1: y[i][r] = sum_d a1[i,d] * (sum_c D1[d][r][c] * X[i][c])
        for i in 0..m1 {
            let xi = &x[i * m2..(i + 1) * m2];
            for r in 0..m2 {
                let mut acc = 0.0f32;
                for d in 0..nd {
                    let a = self.a1[i * nd + d];
                    let drow = &self.d1[(d * m2 + r) * m2..(d * m2 + r) * m2 + m2];
                    let mut u = 0.0f32;
                    for c in 0..m2 {
                        u += drow[c] * xi[c];
                    }
                    acc += a * u;
                }
                y[i * m2 + r] = acc;
            }
        }
        // transpose
        let mut z = vec![0.0f32; m2 * m1];
        for i in 0..m1 {
            for j in 0..m2 {
                z[j * m1 + i] = y[i * m2 + j];
            }
        }
        // stage 2: w[j][r] = sum_d a2[j,d] * (sum_c D2[d][r][c] * z[j][c])
        let mut out = vec![0.0f32; m2 * m1];
        for j in 0..m2 {
            let zj = &z[j * m1..(j + 1) * m1];
            for r in 0..m1 {
                let mut acc = 0.0f32;
                for d in 0..nd {
                    let a = self.a2[j * nd + d];
                    let drow = &self.d2[(d * m1 + r) * m1..(d * m1 + r) * m1 + m1];
                    let mut u = 0.0f32;
                    for c in 0..m1 {
                        u += drow[c] * zj[c];
                    }
                    acc += a * u;
                }
                out[j * m1 + r] = acc;
            }
        }
        Fwd { y, z, out }
    }

    fn backward(&self, x: &[f32], fwd: &Fwd, dout: &[f32]) -> Grads {
        let (m1, m2, nd) = (self.m1, self.m2, self.nd);
        let mut g = Grads {
            da1: vec![0.0; m1 * nd],
            da2: vec![0.0; m2 * nd],
            dd1: vec![0.0; nd * m2 * m2],
            dd2: vec![0.0; nd * m1 * m1],
        };
        // stage 2 backward -> da2, dd2, dz
        let mut dz = vec![0.0f32; m2 * m1];
        for j in 0..m2 {
            let zj = &fwd.z[j * m1..(j + 1) * m1];
            for r in 0..m1 {
                let dw = dout[j * m1 + r];
                if dw == 0.0 {
                    continue;
                }
                for d in 0..nd {
                    let a = self.a2[j * nd + d];
                    let drow = &self.d2[(d * m1 + r) * m1..(d * m1 + r) * m1 + m1];
                    let mut u = 0.0f32;
                    for c in 0..m1 {
                        u += drow[c] * zj[c];
                    }
                    g.da2[j * nd + d] += dw * u;
                    let dd2row = &mut g.dd2[(d * m1 + r) * m1..(d * m1 + r) * m1 + m1];
                    for c in 0..m1 {
                        dd2row[c] += dw * a * zj[c];
                        dz[j * m1 + c] += dw * a * drow[c];
                    }
                }
            }
        }
        // transpose back: dy[i][j] = dz[j][i]
        let mut dy = vec![0.0f32; m1 * m2];
        for j in 0..m2 {
            for i in 0..m1 {
                dy[i * m2 + j] = dz[j * m1 + i];
            }
        }
        // stage 1 backward -> da1, dd1
        for i in 0..m1 {
            let xi = &x[i * m2..(i + 1) * m2];
            for r in 0..m2 {
                let d_y = dy[i * m2 + r];
                if d_y == 0.0 {
                    continue;
                }
                for d in 0..nd {
                    let a = self.a1[i * nd + d];
                    let drow = &self.d1[(d * m2 + r) * m2..(d * m2 + r) * m2 + m2];
                    let mut u = 0.0f32;
                    for c in 0..m2 {
                        u += drow[c] * xi[c];
                    }
                    g.da1[i * nd + d] += d_y * u;
                    let dd1row = &mut g.dd1[(d * m2 + r) * m2..(d * m2 + r) * m2 + m2];
                    for c in 0..m2 {
                        dd1row[c] += d_y * a * xi[c];
                    }
                }
            }
        }
        g
    }

    /// Materialize the effective n*n linear map (column k = forward(e_k)).
    fn effective(&self) -> Vec<f32> {
        let n = self.n();
        let mut w = vec![0.0f32; n * n];
        let mut e = vec![0.0f32; n];
        for k in 0..n {
            e[k] = 1.0;
            let out = self.forward(&e).out;
            for r in 0..n {
                w[r * n + k] = out[r];
            }
            e[k] = 0.0;
        }
        w
    }

    fn params_per_weight(&self) -> usize {
        self.a1.len() + self.a2.len()
    }
}

/// Numerical rank + smallest pivot via Gaussian elimination with partial pivoting.
fn numerical_rank(mut a: Vec<f32>, n: usize, tol: f32) -> (usize, f32) {
    let mut rank = 0usize;
    let mut min_pivot = f32::INFINITY;
    let mut row = 0usize;
    for col in 0..n {
        // find pivot
        let mut piv = row;
        let mut best = a[row.min(n - 1) * n + col].abs();
        for r in (row + 1)..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if row >= n || best < tol {
            continue;
        }
        // swap
        if piv != row {
            for c in 0..n {
                a.swap(row * n + c, piv * n + c);
            }
        }
        let pv = a[row * n + col];
        min_pivot = min_pivot.min(pv.abs());
        for r in (row + 1)..n {
            let f = a[r * n + col] / pv;
            if f != 0.0 {
                for c in col..n {
                    a[r * n + c] -= f * a[row * n + c];
                }
            }
        }
        rank += 1;
        row += 1;
        if row >= n {
            break;
        }
    }
    (rank, if min_pivot.is_finite() { min_pivot } else { 0.0 })
}

fn gradcheck() {
    let mut rng = Lcg(0xBEEF);
    let blk = Block::new(8, 8, 4, &mut rng); // n=64
    let n = blk.n();
    let x = rng.vec(n);
    let dout_seed = rng.vec(n);
    // loss = sum(out * dout_seed)  => dL/dout = dout_seed
    let loss = |b: &Block| -> f64 {
        b.forward(&x).out.iter().zip(&dout_seed).map(|(o, s)| (*o as f64) * (*s as f64)).sum()
    };
    let fwd = blk.forward(&x);
    let g = blk.backward(&x, &fwd, &dout_seed);

    let eps = 1e-3f32;
    let mut worst = 0.0f32;
    let mut check = |name: &str, analytic: &[f32], get: &dyn Fn(&Block, usize) -> *const f32| {
        // sample up to 24 params per tensor
        let m = analytic.len().min(24);
        let mut max_rel = 0.0f32;
        for idx in 0..m {
            let pi = idx * (analytic.len() / m).max(1);
            let mut bp = blk.clone();
            let mut bm = blk.clone();
            unsafe {
                let pp = get(&bp, pi) as *mut f32;
                let pm = get(&bm, pi) as *mut f32;
                *pp += eps;
                *pm -= eps;
            }
            let num = ((loss(&bp) - loss(&bm)) / (2.0 * eps as f64)) as f32;
            let ana = analytic[pi];
            let rel = (num - ana).abs() / (ana.abs().max(num.abs()).max(1e-4));
            max_rel = max_rel.max(rel);
        }
        worst = worst.max(max_rel);
        println!("  {name:5} max rel err {max_rel:.2e}  ({m} sampled)");
    };
    check("a1", &g.da1, &|b, i| &b.a1[i]);
    check("a2", &g.da2, &|b, i| &b.a2[i]);
    check("D1", &g.dd1, &|b, i| &b.d1[i]);
    check("D2", &g.dd2, &|b, i| &b.d2[i]);
    println!(
        "  => gradcheck {} (worst {worst:.2e})\n",
        if worst < 2e-2 { "PASS" } else { "FAIL" }
    );
}

fn rank_sweep() {
    println!("rank sweep (n=64, m1=m2=8, full rank = 64):");
    println!("  {:>3}  {:>6}  {:>10}  {:>12}", "nd", "rank", "min_pivot", "params/wt");
    let mut rng = Lcg(0x515E);
    for nd in [1usize, 2, 4, 8, 16] {
        let blk = Block::new(8, 8, nd, &mut rng);
        let w = blk.effective();
        let (rank, mp) = numerical_rank(w, blk.n(), 1e-4);
        println!("  {nd:>3}  {rank:>6}  {mp:>10.2e}  {:>12}", blk.params_per_weight());
    }
    println!();
}

fn train_to_target(nd: usize, rng: &mut Lcg, lr_coeff: f32, lr_atom: f32, steps: usize, target_fn: &dyn Fn(&[f32]) -> Vec<f32>) -> f32 {
    train_to_target_m(8, nd, rng, lr_coeff, lr_atom, steps, target_fn)
}

/// Same as `train_to_target`, generalized to an arbitrary block size `m`
/// (m1=m2=m) instead of the fixed m=8 used throughout the original nd=4
/// investigation -- needed to test whether the nd=4 dead spot is tied to
/// this specific block size or appears at the analogous relative point
/// (teacher_nd = m/2) for other block sizes too.
#[allow(clippy::too_many_arguments)]
fn train_to_target_m(m: usize, nd: usize, rng: &mut Lcg, lr_coeff: f32, lr_atom: f32, steps: usize, target_fn: &dyn Fn(&[f32]) -> Vec<f32>) -> f32 {
    train_to_target_full(m, nd, rng, lr_coeff, lr_atom, steps, target_fn).1
}

/// Same as `train_to_target_m`, but also returns the trained `Block` --
/// needed to inspect the learned `a1`/`a2` coefficients directly (e.g. for
/// singular-value degeneracy checks) rather than just the final loss.
#[allow(clippy::too_many_arguments)]
fn train_to_target_full(m: usize, nd: usize, rng: &mut Lcg, lr_coeff: f32, lr_atom: f32, steps: usize, target_fn: &dyn Fn(&[f32]) -> Vec<f32>) -> (Block, f32) {
    let mut blk = Block::new(m, m, nd, rng);
    let n = blk.n();
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let mut adam = Adam::new(&blk);
    let mut rel = 0.0f32;
    let batch = 16;
    for step in 0..steps {
        let decay = 0.5 * (1.0 + (std::f32::consts::PI * step as f32 / steps as f32).cos());
        let mut grads = Grads {
            da1: vec![0.0; blk.a1.len()],
            da2: vec![0.0; blk.a2.len()],
            dd1: vec![0.0; blk.d1.len()],
            dd2: vec![0.0; blk.d2.len()],
        };
        let mut sse = 0.0f32;
        let mut energy = 0.0f32;
        for _ in 0..batch {
            let x = rng.vec(n);
            let fwd = blk.forward(&x);
            let to = target_fn(&x);
            let mut dout = vec![0.0f32; n];
            for r in 0..n {
                let e = fwd.out[r] - to[r];
                dout[r] = 2.0 * e / batch as f32;
                sse += e * e;
                energy += to[r] * to[r];
            }
            let g = blk.backward(&x, &fwd, &dout);
            add(&mut grads.da1, &g.da1);
            add(&mut grads.da2, &g.da2);
            add(&mut grads.dd1, &g.dd1);
            add(&mut grads.dd2, &g.dd2);
        }
        rel = (sse / energy.max(1e-9)).sqrt();
        adam.step(&mut blk, &grads, lr_coeff * decay, lr_atom * decay, b1, b2, eps, step + 1);
    }
    (blk, rel)
}

fn overfit() {
    println!("conditioning: cosine-8k (equal lr 5e-3), same-family target, 12 seeds/nd.");
    println!("  (decoupled atom-LR was tested and did NOT help under Adam -> dropped;");
    println!("   the cosine SCHEDULE is the lever, not the LR ratio.)");
    println!("  solved = rel_err < 1e-3. local minima show up as stuck instances.");
    println!("  {:>3}  {:>8}  {:>10}  {:>10}  {:>10}", "nd", "solved", "median", "best", "worst");
    for nd in [2usize, 4, 8, 16] {
        let mut errs = Vec::new();
        for s in 0..12u64 {
            let mut trng = Lcg(0xF17 ^ (nd as u64) ^ (s << 8));
            let teacher = Block::new(8, 8, nd, &mut trng);
            let tf = |x: &[f32]| teacher.forward(x).out;
            let e = train_to_target(nd, &mut Lcg(0xAAA ^ (nd as u64) ^ (s << 16)), 5e-3, 5e-3, 8000, &tf);
            errs.push(e);
        }
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let solved = errs.iter().filter(|&&e| e < 1e-3).count();
        let median = errs[errs.len() / 2];
        println!("  {nd:>3}  {solved:>6}/12  {median:>10.4}  {:>10.4}  {:>10.4}", errs[0], errs[errs.len() - 1]);
    }
    println!();
}

/// Decoupled sweep for the `nd=4` dead-spot anomaly (RESEARCH_LOG.md,
/// 2026-06-27 "Gate (b) follow-up"): the original 12-seed sweep varied
/// teacher rank and student capacity TOGETHER (both set by the same `nd`),
/// confounding "student capacity" with "teacher's true rank" -- so a dead
/// spot fixed at the literal value 4 is indistinguishable from a dead spot
/// that tracks wherever student capacity equals teacher rank. This fixes the
/// TEACHER at a specific rank and sweeps STUDENT capacity independently: if
/// the over-parameterization hypothesis holds, the dead spot should move to
/// track `student_nd == teacher_nd`, not stay pinned at 4.
fn decoupled_sweep(teacher_nd: usize, student_nds: &[usize], seeds: u64) {
    println!(
        "decoupled sweep: teacher FIXED at nd={teacher_nd}, student nd swept independently ({seeds} seeds/point)"
    );
    println!("  {:>11}  {:>8}  {:>10}  {:>10}  {:>10}", "student_nd", "solved", "median", "best", "worst");
    for &snd in student_nds {
        let mut errs = Vec::new();
        for s in 0..seeds {
            let mut trng = Lcg(0xF17 ^ (teacher_nd as u64) ^ (s << 8));
            let teacher = Block::new(8, 8, teacher_nd, &mut trng);
            let tf = |x: &[f32]| teacher.forward(x).out;
            let e = train_to_target(snd, &mut Lcg(0xAAA ^ (snd as u64) ^ (teacher_nd as u64) ^ (s << 16)), 5e-3, 5e-3, 8000, &tf);
            errs.push(e);
        }
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let solved = errs.iter().filter(|&&e| e < 1e-3).count();
        let median = errs[errs.len() / 2];
        let marker = if snd == teacher_nd { " <- student==teacher" } else { "" };
        println!(
            "  {snd:>11}  {solved:>6}/{seeds}  {median:>10.4}  {:>10.4}  {:>10.4}{marker}",
            errs[0], errs[errs.len() - 1]
        );
    }
    println!();
}

/// Steps sweep at the known-anomalous student_nd == teacher_nd == 4 point:
/// distinguishes "true local-minimum trap" (solved-rate stays low regardless
/// of training length) from "just slow convergence" (solved-rate climbs with
/// more steps).
fn steps_sweep(nd: usize, steps_list: &[usize], seeds: u64) {
    println!("steps sweep: same-family, student_nd=teacher_nd={nd} ({seeds} seeds/point)");
    println!("  {:>7}  {:>8}  {:>10}  {:>10}  {:>10}", "steps", "solved", "median", "best", "worst");
    for &steps in steps_list {
        let mut errs = Vec::new();
        for s in 0..seeds {
            let mut trng = Lcg(0xF17 ^ (nd as u64) ^ (s << 8));
            let teacher = Block::new(8, 8, nd, &mut trng);
            let tf = |x: &[f32]| teacher.forward(x).out;
            let e = train_to_target(nd, &mut Lcg(0xAAA ^ (nd as u64) ^ (steps as u64) ^ (s << 16)), 5e-3, 5e-3, steps, &tf);
            errs.push(e);
        }
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let solved = errs.iter().filter(|&&e| e < 1e-3).count();
        let median = errs[errs.len() / 2];
        println!("  {steps:>7}  {solved:>6}/{seeds}  {median:>10.4}  {:>10.4}  {:>10.4}", errs[0], errs[errs.len() - 1]);
    }
    println!();
}

/// Is the nd=4 dead spot tied to this specific block size (m=8), or does
/// the analogous relative point (teacher_nd = m/2) show the same
/// "matched capacity is stuck, even 2x overcapacity barely solves" pattern
/// at OTHER block sizes too? For each m, tests teacher_nd=m/2 at three
/// student points: matched (student_nd=teacher_nd), 2x overcapacity, and
/// well over (4x), mirroring what m=8/nd=4 showed.
fn block_size_sweep(ms: &[usize], seeds: u64) {
    println!("block-size sweep: teacher_nd = m/2, student at 1x/2x/4x that, for each block size m ({seeds} seeds/point)");
    println!(
        "  {:>4}  {:>11}  {:>14}  {:>14}  {:>14}",
        "m", "teacher_nd", "1x (matched)", "2x", "4x"
    );
    for &m in ms {
        let tnd = (m / 2).max(1);
        let points = [tnd, (2 * tnd).max(tnd + 1), (4 * tnd).max(tnd + 1)];
        let mut cell = |snd: usize| -> String {
            let mut errs = Vec::new();
            for s in 0..seeds {
                let mut trng = Lcg(0xF17 ^ (m as u64) ^ (tnd as u64) ^ (s << 8));
                let teacher = Block::new(m, m, tnd, &mut trng);
                let tf = |x: &[f32]| teacher.forward(x).out;
                let e = train_to_target_m(
                    m, snd, &mut Lcg(0xAAA ^ (m as u64) ^ (snd as u64) ^ (tnd as u64) ^ (s << 16)),
                    5e-3, 5e-3, 8000, &tf,
                );
                errs.push(e);
            }
            let solved = errs.iter().filter(|&&e| e < 1e-3).count();
            errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = errs[errs.len() / 2];
            format!("{solved}/{seeds} ({median:.3})")
        };
        println!(
            "  {m:>4}  {tnd:>11}  {:>14}  {:>14}  {:>14}",
            cell(points[0]), cell(points[1]), cell(points[2])
        );
    }
    println!();
}

/// Cyclic Jacobi eigenvalue solver for a small symmetric matrix (row-major,
/// `n x n`). Simple and numerically robust for the tiny matrices here
/// (n = nd, at most a few dozen) -- not intended for large-scale use.
fn jacobi_eigenvalues(mut a: Vec<f32>, n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    for _sweep in 0..100 {
        let mut off = 0.0f32;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[i * n + j] * a[i * n + j];
            }
        }
        if off < 1e-12 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-12 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let phi = 0.5 * (2.0 * apq).atan2(aqq - app);
                let (c, s) = (phi.cos(), phi.sin());
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut eig: Vec<f32> = (0..n).map(|i| a[i * n + i]).collect();
    eig.sort_by(|x, y| y.partial_cmp(x).unwrap()); // descending
    eig
}

/// Singular values of `a` (`rows x cols`, row-major), via eigenvalues of the
/// `cols x cols` Gram matrix `a^T a`. Descending order.
fn singular_values(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut gram = vec![0.0f32; cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            let mut s = 0.0f32;
            for r in 0..rows {
                s += a[r * cols + i] * a[r * cols + j];
            }
            gram[i * cols + j] = s;
        }
    }
    jacobi_eigenvalues(gram, cols).into_iter().map(|e| e.max(0.0).sqrt()).collect()
}

/// Mechanistic follow-up to the nd=4 dead-spot investigation: WHY are
/// matched-capacity instances stuck? Trains one deliberately-matched
/// (student==teacher, the known-hard point) and one deliberately-
/// overparameterized (student==2x teacher, known to solve reliably)
/// instance side by side, then compares the singular value spectrum of
/// their learned `a1`/`a2` coefficient matrices. A stuck instance whose
/// effective rank has collapsed (small trailing singular values, active
/// atoms redundant) would point at atom-collapse/redundancy as the
/// mechanism; a stuck instance with a healthy, full-rank spectrum despite
/// high loss would point elsewhere (e.g. a genuine rugged bilinear
/// landscape, not degenerate parameters).
fn condition_probe(m: usize, teacher_nd: usize, seed: u64) {
    println!("condition probe: m={m}, teacher_nd={teacher_nd}, seed={seed}");
    for (label, student_nd) in [("matched (stuck regime)", teacher_nd), ("2x overcapacity (solves)", 2 * teacher_nd)] {
        let mut trng = Lcg(0xF17 ^ (m as u64) ^ (teacher_nd as u64) ^ (seed << 8));
        let teacher = Block::new(m, m, teacher_nd, &mut trng);
        let tf = |x: &[f32]| teacher.forward(x).out;
        let (trained, rel) = train_to_target_full(
            m, student_nd,
            &mut Lcg(0xAAA ^ (m as u64) ^ (student_nd as u64) ^ (teacher_nd as u64) ^ (seed << 16)),
            5e-3, 5e-3, 8000, &tf,
        );
        let sv_a1 = singular_values(&trained.a1, m, student_nd);
        let sv_a2 = singular_values(&trained.a2, m, student_nd);
        let fmt = |v: &[f32]| v.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(", ");
        println!("  {label}: student_nd={student_nd}, rel_err={rel:.4}");
        println!("    a1 singular values: [{}]", fmt(&sv_a1));
        println!("    a2 singular values: [{}]", fmt(&sv_a2));
        let a1_ratio = sv_a1.last().copied().unwrap_or(0.0) / sv_a1.first().copied().unwrap_or(1.0).max(1e-9);
        let a2_ratio = sv_a2.last().copied().unwrap_or(0.0) / sv_a2.first().copied().unwrap_or(1.0).max(1e-9);
        println!("    smallest/largest ratio: a1={a1_ratio:.4}  a2={a2_ratio:.4}  (near-0 = collapsed/redundant atom usage)");
    }
    println!();
}

fn add(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

struct Adam {
    m: Grads,
    v: Grads,
}
impl Adam {
    fn new(b: &Block) -> Self {
        let z = |n| vec![0.0f32; n];
        Adam {
            m: Grads { da1: z(b.a1.len()), da2: z(b.a2.len()), dd1: z(b.d1.len()), dd2: z(b.d2.len()) },
            v: Grads { da1: z(b.a1.len()), da2: z(b.a2.len()), dd1: z(b.d1.len()), dd2: z(b.d2.len()) },
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn step(&mut self, blk: &mut Block, g: &Grads, lr_coeff: f32, lr_atom: f32, b1: f32, b2: f32, eps: f32, t: usize) {
        upd(&mut blk.a1, &g.da1, &mut self.m.da1, &mut self.v.da1, lr_coeff, b1, b2, eps, t);
        upd(&mut blk.a2, &g.da2, &mut self.m.da2, &mut self.v.da2, lr_coeff, b1, b2, eps, t);
        upd(&mut blk.d1, &g.dd1, &mut self.m.dd1, &mut self.v.dd1, lr_atom, b1, b2, eps, t);
        upd(&mut blk.d2, &g.dd2, &mut self.m.dd2, &mut self.v.dd2, lr_atom, b1, b2, eps, t);
    }
}
#[allow(clippy::too_many_arguments)]
fn upd(p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32], lr: f32, b1: f32, b2: f32, eps: f32, t: usize) {
    let bc1 = 1.0 - b1.powi(t as i32);
    let bc2 = 1.0 - b2.powi(t as i32);
    for i in 0..p.len() {
        m[i] = b1 * m[i] + (1.0 - b1) * g[i];
        v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
        let mh = m[i] / bc1;
        let vh = v[i] / bc2;
        p[i] -= lr * mh / (vh.sqrt() + eps);
    }
}

#[allow(dead_code)]
fn full_sweep_suite() {
    println!("=== gate (b): shared-core BTT block ===\n");
    println!("gradcheck (n=64, nd=4):");
    gradcheck();
    rank_sweep();
    overfit();

    println!("=== nd=4 dead-spot follow-up: over-parameterization hypothesis ===\n");
    // If the dead spot tracks student==teacher (not the literal value 4),
    // it should show up as a dip at student_nd=4 here (teacher fixed at 4,
    // matching the original anomaly) ...
    decoupled_sweep(4, &[2, 3, 4, 5, 6, 8], 12);
    // ... and shift to a dip at student_nd=8 here if the hypothesis holds
    // (teacher fixed at 8 -- a rank the original sweep never isolated,
    // since teacher and student nd were always the same number there).
    decoupled_sweep(8, &[4, 6, 7, 8, 9, 10, 16], 12);

    steps_sweep(4, &[8000, 16000, 32000], 8);

    println!("=== nd=4 dead-spot follow-up: is it tied to block size m=8? ===\n");
    block_size_sweep(&[4, 6, 8, 12, 16], 8);
}

fn main() {
    println!("=== nd=4 dead-spot follow-up: mechanism -- atom collapse or rugged landscape? ===\n");
    for seed in 0..4u64 {
        condition_probe(8, 4, seed);
    }
}
