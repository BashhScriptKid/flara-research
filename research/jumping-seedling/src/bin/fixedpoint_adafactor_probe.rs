// Corrected version of fixedpoint_adam_probe: the real codebase optimizer
// (src/kernels/optimizer.rs) is AdaFactor, not plain Adam — it factors the
// second moment into row-sums R and column-sums C (O(rows+cols) state)
// instead of a full per-parameter v (O(rows*cols)), and applies a global
// RMS-clip to the update. Both differences could plausibly change the
// int16-storage failure mode found in fixedpoint_adam_probe, so this rebuilds
// the same falsification test against AdaFactor's actual math instead.
//
// AdaFactorState's fields are private in the real struct, so this reimplements
// the same update rule (mirrored line-for-line from optimizer.rs::step) rather
// than importing it, to keep a fixed-point variant of R/C storage.

const ROWS: usize = 16;
const COLS: usize = 16;
const N: usize = ROWS * COLS;
const STEPS: usize = 1500;
const RESCALE_EVERY: usize = 8;

const LR: f32 = 0.05;
const DECAY: f32 = 0.8; // matches AdaFactor::default
const EPS1: f32 = 1e-30;
const CLIP: f32 = 1.0; // matches AdaFactor::default clip_threshold

#[derive(Clone)]
struct FixedVec {
    data: Vec<i16>,
    scale: i32,
}

impl FixedVec {
    fn zeros(n: usize) -> Self {
        FixedVec { data: vec![0i16; n], scale: 0 }
    }
    fn to_f32(&self) -> Vec<f32> {
        let mult = 2f32.powi(self.scale);
        self.data.iter().map(|&v| v as f32 * mult).collect()
    }
    fn rescale(&mut self, values: &[f32]) {
        let max_abs = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = pick_scale(max_abs);
        self.requantize_at(values, scale);
    }
    fn requantize_at(&mut self, values: &[f32], scale: i32) {
        let inv_mult = 2f32.powi(-scale);
        for (slot, &v) in self.data.iter_mut().zip(values.iter()) {
            let q = (v * inv_mult).round();
            *slot = q.clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        }
        self.scale = scale;
    }
    fn frac_zero(&self) -> f32 {
        self.data.iter().filter(|&&v| v == 0).count() as f32 / self.data.len() as f32
    }
}

fn pick_scale(max_abs: f32) -> i32 {
    if max_abs <= 0.0 || !max_abs.is_finite() {
        return 0;
    }
    (max_abs / 30000.0).log2().ceil() as i32
}

fn beta2_t(step: u64) -> f32 {
    1.0 - (step as f32).powf(-DECAY)
}

/// One AdaFactor step, mirroring optimizer.rs::step (factored branch, no
/// momentum, relative_step=false — matching the simplest real configuration).
/// `r_f`/`c_f` are the dequantized-to-fp32 row/col factors for this step;
/// caller re-quantizes them afterward (fixed) or just keeps them (reference).
fn adafactor_step(param: &mut [f32], grad: &[f32], r_f: &mut [f32], c_f: &mut [f32], step: u64) {
    let b2 = beta2_t(step);
    let mut u = vec![0.0f32; N];
    for i in 0..ROWS {
        let mut acc = 0.0f32;
        for j in 0..COLS {
            let g = grad[i * COLS + j];
            acc += g * g + EPS1;
        }
        r_f[i] = b2 * r_f[i] + (1.0 - b2) * acc;
    }
    for j in 0..COLS {
        let mut acc = 0.0f32;
        for i in 0..ROWS {
            let g = grad[i * COLS + j];
            acc += g * g + EPS1;
        }
        c_f[j] = b2 * c_f[j] + (1.0 - b2) * acc;
    }
    let total: f32 = r_f.iter().sum::<f32>().max(1e-30);
    for i in 0..ROWS {
        for j in 0..COLS {
            let vhat = r_f[i] * c_f[j] / total;
            u[i * COLS + j] = grad[i * COLS + j] / vhat.sqrt();
        }
    }
    let rms_u = (u.iter().map(|x| x * x).sum::<f32>() / N as f32).sqrt();
    let clip_factor = (rms_u / CLIP).max(1.0);
    if clip_factor > 1.0 {
        for x in u.iter_mut() {
            *x /= clip_factor;
        }
    }
    for idx in 0..N {
        param[idx] -= LR * u[idx];
    }
}

/// Matrix quadratic bowl with per-row AND per-column curvature multipliers,
/// so R[i] and C[j] each have real dynamic range across their own vector —
/// the thing that's actually relevant to quantizing R/C, as opposed to
/// per-element dynamic range which factoring already absorbs.
fn make_problem() -> (Vec<f32>, Vec<f32>) {
    let mut row_curv = vec![0.0f32; ROWS];
    let mut col_curv = vec![0.0f32; COLS];
    let mut target = vec![0.0f32; N];
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    let mut rand = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for i in 0..ROWS {
        row_curv[i] = 10f32.powf(((rand() * 4.0) - 2.0) as f32); // 1e-2 .. 1e2
    }
    for j in 0..COLS {
        col_curv[j] = 10f32.powf(((rand() * 4.0) - 2.0) as f32);
    }
    for t in target.iter_mut() {
        *t = (rand() as f32 - 0.5) * 4.0;
    }
    let mut a = vec![0.0f32; N];
    for i in 0..ROWS {
        for j in 0..COLS {
            a[i * COLS + j] = row_curv[i] * col_curv[j];
        }
    }
    (a, target)
}

fn loss(x: &[f32], a: &[f32], target: &[f32]) -> f32 {
    x.iter().zip(a).zip(target).map(|((xi, ai), ti)| ai * (xi - ti).powi(2)).sum::<f32>() / x.len() as f32
}
fn grad(x: &[f32], a: &[f32], target: &[f32]) -> Vec<f32> {
    x.iter().zip(a).zip(target).map(|((xi, ai), ti)| 2.0 * ai * (xi - ti)).collect()
}

fn main() {
    let (a, target) = make_problem();
    let mut w_fixed = vec![0.0f32; N];
    let mut w_ref = vec![0.0f32; N];

    let mut r_fixed = FixedVec::zeros(ROWS);
    let mut c_fixed = FixedVec::zeros(COLS);
    let mut r_ref = vec![0.0f32; ROWS];
    let mut c_ref = vec![0.0f32; COLS];

    println!("step,loss_fixed,loss_ref,ratio,r_scale,c_scale,r_zero%,c_zero%");
    for step in 1..=STEPS as u64 {
        // reference
        let g_ref = grad(&w_ref, &a, &target);
        adafactor_step(&mut w_ref, &g_ref, &mut r_ref, &mut c_ref, step);

        // fixed: dequantize R/C, do the same math in fp32, requantize
        let g_fixed = grad(&w_fixed, &a, &target);
        let mut r_f = r_fixed.to_f32();
        let mut c_f = c_fixed.to_f32();
        adafactor_step(&mut w_fixed, &g_fixed, &mut r_f, &mut c_f, step);
        if step as usize % RESCALE_EVERY == 0 {
            r_fixed.rescale(&r_f);
            c_fixed.rescale(&c_f);
        } else {
            r_fixed.requantize_at(&r_f, r_fixed.scale);
            c_fixed.requantize_at(&c_f, c_fixed.scale);
        }

        if step as usize % 100 == 0 || step as usize == STEPS {
            let l_fixed = loss(&w_fixed, &a, &target);
            let l_ref = loss(&w_ref, &a, &target);
            let ratio = if l_ref > 1e-12 { l_fixed / l_ref } else { f32::NAN };
            println!(
                "{step},{l_fixed:.6},{l_ref:.6},{ratio:.4},{},{},{:.1},{:.1}",
                r_fixed.scale, c_fixed.scale,
                r_fixed.frac_zero() * 100.0, c_fixed.frac_zero() * 100.0
            );
        }
    }

    let l_fixed = loss(&w_fixed, &a, &target);
    let l_ref = loss(&w_ref, &a, &target);
    println!();
    println!("final fixed-point loss: {l_fixed:.6}");
    println!("final fp32 reference loss:  {l_ref:.6}");
    println!("ratio (fixed/ref): {:.4}", l_fixed / l_ref);
    if l_fixed / l_ref > 2.0 {
        println!("VERDICT: int16 R/C factors meaningfully worse than fp32 AdaFactor.");
    } else if l_fixed / l_ref > 1.2 {
        println!("VERDICT: close but measurably behind — usable with a quality cost.");
    } else {
        println!("VERDICT: int16 R/C factors track fp32 AdaFactor closely — viable.");
    }
}
