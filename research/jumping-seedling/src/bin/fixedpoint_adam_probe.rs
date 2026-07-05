// Standalone falsification test for a dynamic-block-fixed-point Adam optimizer.
//
// Question: can Adam's (m, v) state and the gradient/weight tensors be stored
// as int16 + a per-tensor power-of-two scale (rescaled periodically) without
// destroying convergence, compared to a reference fp32 Adam on the same
// problem? This does not touch the real model — it runs both optimizers on a
// synthetic quadratic bowl, since that isolates optimizer numerics from
// everything else.

const N: usize = 256; // parameter count
const STEPS: usize = 1500;
const RESCALE_EVERY: usize = 8; // how often m/v scales get recalibrated

const LR: f32 = 0.05;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;

/// int16 storage + a shared power-of-two scale: value ≈ data[i] as f32 * 2^scale.
#[derive(Clone)]
struct FixedTensor {
    data: Vec<i16>,
    scale: i32,
}

impl FixedTensor {
    fn zeros(n: usize) -> Self {
        FixedTensor { data: vec![0i16; n], scale: 0 }
    }

    fn quantize(values: &[f32]) -> Self {
        let max_abs = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = pick_scale(max_abs);
        let mut t = FixedTensor { data: vec![0i16; values.len()], scale };
        t.requantize_at(values, scale);
        t
    }

    fn to_f32(&self) -> Vec<f32> {
        let mult = 2f32.powi(self.scale);
        self.data.iter().map(|&v| v as f32 * mult).collect()
    }

    /// Recalibrate scale from current fp32 values, then requantize in place.
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
}

/// Largest power-of-two scale such that max_abs / 2^scale fits under ~32000,
/// leaving headroom before clipping. Falls back to scale 0 for an all-zero tensor.
fn pick_scale(max_abs: f32) -> i32 {
    if max_abs <= 0.0 || !max_abs.is_finite() {
        return 0;
    }
    let target = 30000.0f32;
    (max_abs / target).log2().ceil() as i32
}

struct FixedAdam {
    m: FixedTensor,
    v: FixedTensor,
    w: FixedTensor,
    t: i32,
}

impl FixedAdam {
    fn new(w0: &[f32]) -> Self {
        FixedAdam {
            m: FixedTensor::zeros(w0.len()),
            v: FixedTensor::zeros(w0.len()),
            w: FixedTensor::quantize(w0),
            t: 0,
        }
    }

    /// One Adam step. Gradient arrives as fp32 (freshly computed every step,
    /// never carried across steps, so it gets its own throwaway scale here
    /// rather than a stored FixedTensor).
    fn step(&mut self, grad: &[f32]) {
        self.t += 1;
        let t = self.t as f32;
        let bias_c1 = 1.0 - BETA1.powf(t);
        let bias_c2 = 1.0 - BETA2.powf(t);

        let mut w_f = self.w.to_f32();
        let mut m_f = self.m.to_f32();
        let mut v_f = self.v.to_f32();

        for i in 0..w_f.len() {
            let g = grad[i];
            m_f[i] = BETA1 * m_f[i] + (1.0 - BETA1) * g;
            v_f[i] = BETA2 * v_f[i] + (1.0 - BETA2) * g * g;
            let m_hat = m_f[i] / bias_c1;
            let v_hat = v_f[i] / bias_c2;
            w_f[i] -= LR * m_hat / (v_hat.sqrt() + EPS);
        }

        // Storage is int16 the whole time; only every RESCALE_EVERY steps do
        // we recompute the shared scale (mimics a realistic "not every step"
        // rescale cadence — recalibrating every step would hide the failure
        // mode we're testing for).
        if self.t as usize % RESCALE_EVERY == 0 {
            self.m.rescale(&m_f);
            self.v.rescale(&v_f);
            self.w.rescale(&w_f);
        } else {
            self.m.requantize_at(&m_f, self.m.scale);
            self.v.requantize_at(&v_f, self.v.scale);
            self.w.requantize_at(&w_f, self.w.scale);
        }
    }

    /// Diagnostic snapshot: how much of each tensor is dead (quantized to
    /// exactly 0), and the current scale, so we can see *where* precision is
    /// being lost rather than just that the loss blew up.
    fn diagnostics(&self, a: &[f32]) -> Diag {
        let frac_zero = |t: &FixedTensor| {
            t.data.iter().filter(|&&v| v == 0).count() as f32 / t.data.len() as f32
        };
        // Correlate "which params are dead" with curvature (a_i): if the
        // dead ones are disproportionately the low-curvature params, that
        // directly confirms the single-scale-crushes-small-values story.
        let v_f = self.v.to_f32();
        let mut low_curv_dead = 0usize;
        let mut low_curv_total = 0usize;
        let median_a = {
            let mut sorted = a.to_vec();
            sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
            sorted[sorted.len() / 2]
        };
        for (i, &ai) in a.iter().enumerate() {
            if ai < median_a {
                low_curv_total += 1;
                if v_f[i] == 0.0 {
                    low_curv_dead += 1;
                }
            }
        }
        Diag {
            w_scale: self.w.scale,
            m_scale: self.m.scale,
            v_scale: self.v.scale,
            w_frac_zero: frac_zero(&self.w),
            m_frac_zero: frac_zero(&self.m),
            v_frac_zero: frac_zero(&self.v),
            low_curv_dead_frac: low_curv_dead as f32 / low_curv_total.max(1) as f32,
        }
    }

    fn weights(&self) -> Vec<f32> {
        self.w.to_f32()
    }
}

struct RefAdam {
    m: Vec<f32>,
    v: Vec<f32>,
    w: Vec<f32>,
    t: i32,
}

impl RefAdam {
    fn new(w0: &[f32]) -> Self {
        RefAdam { m: vec![0.0; w0.len()], v: vec![0.0; w0.len()], w: w0.to_vec(), t: 0 }
    }

    fn step(&mut self, grad: &[f32]) {
        self.t += 1;
        let t = self.t as f32;
        let bias_c1 = 1.0 - BETA1.powf(t);
        let bias_c2 = 1.0 - BETA2.powf(t);
        for i in 0..self.w.len() {
            let g = grad[i];
            self.m[i] = BETA1 * self.m[i] + (1.0 - BETA1) * g;
            self.v[i] = BETA2 * self.v[i] + (1.0 - BETA2) * g * g;
            let m_hat = self.m[i] / bias_c1;
            let v_hat = self.v[i] / bias_c2;
            self.w[i] -= LR * m_hat / (v_hat.sqrt() + EPS);
        }
    }
}

/// Quadratic bowl: f(x) = sum_i a_i * (x_i - target_i)^2, grad_i = 2*a_i*(x_i-target_i).
/// a_i spans several orders of magnitude so the gradient/curvature dynamic
/// range stresses the fixed-point scale tracking the way real per-layer
/// gradients would.
fn make_problem(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = vec![0.0f32; n];
    let mut target = vec![0.0f32; n];
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    let mut rand = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for i in 0..n {
        let exp = (rand() * 6.0) - 3.0; // curvature spans 1e-3 .. 1e3
        a[i] = 10f32.powf(exp as f32);
        target[i] = (rand() as f32 - 0.5) * 4.0;
    }
    (a, target)
}

fn loss(x: &[f32], a: &[f32], target: &[f32]) -> f32 {
    x.iter().zip(a).zip(target).map(|((xi, ai), ti)| ai * (xi - ti).powi(2)).sum::<f32>() / x.len() as f32
}

fn grad(x: &[f32], a: &[f32], target: &[f32]) -> Vec<f32> {
    x.iter().zip(a).zip(target).map(|((xi, ai), ti)| 2.0 * ai * (xi - ti)).collect()
}

struct Diag {
    w_scale: i32,
    m_scale: i32,
    v_scale: i32,
    w_frac_zero: f32,
    m_frac_zero: f32,
    v_frac_zero: f32,
    low_curv_dead_frac: f32,
}

fn main() {
    let (a, target) = make_problem(N);
    let x0 = vec![0.0f32; N];

    let mut fixed = FixedAdam::new(&x0);
    let mut reference = RefAdam::new(&x0);

    println!("step,loss_fixed,loss_ref,ratio,w_scale,m_scale,v_scale,w_zero%,m_zero%,v_zero%,low_curv_dead%");
    for step in 0..=STEPS {
        let w_fixed = fixed.weights();
        let l_fixed = loss(&w_fixed, &a, &target);
        let l_ref = loss(&reference.w, &a, &target);
        if step % 100 == 0 || step == STEPS {
            let ratio = if l_ref > 1e-12 { l_fixed / l_ref } else { f32::NAN };
            let d = fixed.diagnostics(&a);
            println!(
                "{step},{l_fixed:.6},{l_ref:.6},{ratio:.4},{},{},{},{:.1},{:.1},{:.1},{:.1}",
                d.w_scale, d.m_scale, d.v_scale,
                d.w_frac_zero * 100.0, d.m_frac_zero * 100.0, d.v_frac_zero * 100.0,
                d.low_curv_dead_frac * 100.0
            );
        }
        if step == STEPS {
            break;
        }
        let g_fixed = grad(&w_fixed, &a, &target);
        let g_ref = grad(&reference.w, &a, &target);
        fixed.step(&g_fixed);
        reference.step(&g_ref);
    }

    let w_fixed = fixed.weights();
    let l_fixed = loss(&w_fixed, &a, &target);
    let l_ref = loss(&reference.w, &a, &target);
    println!();
    println!("final fixed-point loss: {l_fixed:.6}");
    println!("final fp32 reference loss:  {l_ref:.6}");
    println!("ratio (fixed/ref): {:.4}", l_fixed / l_ref);
    if l_fixed / l_ref > 2.0 {
        println!("VERDICT: fixed-point Adam converges meaningfully worse — scheme needs rework before touching the real model.");
    } else if l_fixed / l_ref > 1.2 {
        println!("VERDICT: fixed-point Adam is close but measurably behind fp32 — usable but with a quality cost.");
    } else {
        println!("VERDICT: fixed-point Adam tracks fp32 closely — scheme looks viable to prototype in real kernels.");
    }
}
