//! AdaFactor optimizer — factored second moment.
//!
//! Adam stores a full per-parameter second-moment tensor `V` the same size as
//! the weights. At 1B params on an 8 MB-L3 laptop that doubling is exactly the
//! memory pressure this whole project is trying to avoid. AdaFactor
//! (Shazeer & Stern, 2018) keeps only the row-sums `R` and column-sums `C` of
//! the squared gradient and reconstructs a rank-1 estimate
//! `V̂[i,j] = R[i]·C[j] / ΣR` — `O(rows+cols)` state instead of `O(rows·cols)`.
//! The reconstruction is *exact* whenever `G²` is separable, and a good
//! approximation otherwise.
//!
//! Two more AdaFactor pieces are here: **update RMS-clipping** (rescale the whole
//! update so its RMS ≤ `clip`, the part that lets AdaFactor run without a tuned
//! per-tensor LR) and an optional **relative step size** (scale the LR by the
//! RMS of the parameter itself, so updates track parameter scale). First-moment
//! momentum is optional (`beta1 > 0`).
//!
//! 1-D tensors (biases, RMSNorm gains, the dictionary treated component-wise)
//! can't be factored, so they fall back to a full per-element second moment.
//!
//! **Research hooks (deferred, flagged — not yet implemented):**
//! - *Frequency-domain second moment.* The project thesis is that an FFT of the
//!   gradient compacts its energy into few coefficients, which should make `G²`
//!   *more* separable in the spectral domain and the rank-1 factorization
//!   tighter. That is a measurable experiment to run on top of this base, not a
//!   silent default — the factorization here is the standard spatial one.
//! - *INT8 momentum.* `mom` is f32 here for a correct reference; quantizing it to
//!   INT8 (via `QuantizedMomentum`) is the memory-optimization pass, alongside
//!   the AVX2 / sub-byte work deferred elsewhere.

/// AdaFactor hyperparameters (shared across all parameter tensors).
pub struct AdaFactor {
    /// First-moment decay. `0.0` disables momentum (pure AdaFactor).
    pub beta1: f32,
    /// Regularizer added to squared gradients before factoring.
    pub eps1: f32,
    /// Floor on parameter RMS for the relative step size.
    pub eps2: f32,
    /// Update RMS-clipping threshold.
    pub clip: f32,
    /// Exponent of the `β2_t = 1 − t^{−decay}` second-moment schedule.
    pub decay: f32,
    /// If true, the effective LR is `lr · max(eps2, RMS(param))`.
    pub relative_step: bool,
}

impl Default for AdaFactor {
    fn default() -> Self {
        Self {
            beta1: 0.0,
            eps1: 1e-30,
            eps2: 1e-3,
            clip: 1.0,
            decay: 0.8,
            relative_step: true,
        }
    }
}

/// Per-tensor optimizer state.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct AdaFactorState {
    /// Factored: row factor `[rows]`. Full (1-D): per-element second moment `[numel]`.
    r: Vec<f32>,
    /// Factored: column factor `[cols]`. Full: empty.
    c: Vec<f32>,
    /// First-moment buffer `[numel]` when `beta1 > 0`, else empty.
    mom: Vec<f32>,
    rows: usize,
    cols: usize,
    factored: bool,
    step: u64,
}

impl AdaFactorState {
    /// State for a `rows × cols` matrix parameter (factored second moment).
    pub fn matrix(rows: usize, cols: usize, momentum: bool) -> Self {
        assert!(rows > 0 && cols > 0);
        let factored = rows > 1 && cols > 1;
        Self {
            r: vec![0.0; if factored { rows } else { rows * cols }],
            c: vec![0.0; if factored { cols } else { 0 }],
            mom: if momentum { vec![0.0; rows * cols] } else { Vec::new() },
            rows,
            cols,
            factored,
            step: 0,
        }
    }

    /// State for a flat 1-D parameter (full per-element second moment).
    pub fn vector(numel: usize, momentum: bool) -> Self {
        Self::matrix(numel, 1, momentum)
    }
}

impl AdaFactor {
    #[inline]
    fn beta2_t(&self, step: u64) -> f32 {
        // step ≥ 1 ⇒ at step 1, β2 = 0 (first observation taken at full weight).
        1.0 - (step as f32).powf(-self.decay)
    }

    /// One in-place update step. `grad` is the gradient for `param`; both are
    /// `rows·cols` row-major. `lr` is the base learning rate (scaled by the
    /// parameter RMS if `relative_step`).
    pub fn step(&self, param: &mut [f32], grad: &[f32], st: &mut AdaFactorState, lr: f32) {
        let (rows, cols) = (st.rows, st.cols);
        let n = rows * cols;
        assert_eq!(param.len(), n, "param shape mismatch");
        assert_eq!(grad.len(), n, "grad shape mismatch");
        st.step += 1;
        let b2 = self.beta2_t(st.step);

        // --- build the (factored or full) second-moment estimate, fill update u ---
        let mut u = vec![0.0f32; n];
        if st.factored {
            // Row sums and column sums of (g² + eps1).
            for i in 0..rows {
                let mut acc = 0.0f32;
                for j in 0..cols {
                    let g = grad[i * cols + j];
                    acc += g * g + self.eps1;
                }
                st.r[i] = b2 * st.r[i] + (1.0 - b2) * acc;
            }
            for j in 0..cols {
                let mut acc = 0.0f32;
                for i in 0..rows {
                    let g = grad[i * cols + j];
                    acc += g * g + self.eps1;
                }
                st.c[j] = b2 * st.c[j] + (1.0 - b2) * acc;
            }
            let total: f32 = st.r.iter().sum::<f32>().max(1e-30);
            for i in 0..rows {
                for j in 0..cols {
                    let vhat = st.r[i] * st.c[j] / total;
                    u[i * cols + j] = grad[i * cols + j] / vhat.sqrt();
                }
            }
        } else {
            for idx in 0..n {
                let g = grad[idx];
                let v = b2 * st.r[idx] + (1.0 - b2) * (g * g + self.eps1);
                st.r[idx] = v;
                u[idx] = g / v.sqrt();
            }
        }

        // --- RMS-clip the whole update ---
        let rms_u = (u.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
        let clip_factor = (rms_u / self.clip).max(1.0);
        if clip_factor > 1.0 {
            for x in u.iter_mut() {
                *x /= clip_factor;
            }
        }

        // --- relative step size ---
        let eff_lr = if self.relative_step {
            let rms_p = (param.iter().map(|p| p * p).sum::<f32>() / n as f32).sqrt();
            lr * rms_p.max(self.eps2)
        } else {
            lr
        };

        // --- optional momentum, then apply ---
        if self.beta1 > 0.0 {
            for idx in 0..n {
                st.mom[idx] = self.beta1 * st.mom[idx] + (1.0 - self.beta1) * u[idx];
                param[idx] -= eff_lr * st.mom[idx];
            }
        } else {
            for idx in 0..n {
                param[idx] -= eff_lr * u[idx];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factored_reconstructs_separable_second_moment() {
        // G²[i,j] = a_i·b_j is exactly rank-1, so V̂ should equal G² and the
        // normalized update u = G/sqrt(V̂) = 1 everywhere. With lr=1, no clip,
        // no momentum, no relative step, param (init 0) moves to exactly −1.
        let (rows, cols) = (3, 4);
        let a = [0.5f32, 2.0, 1.3];
        let b = [0.7f32, 0.2, 1.1, 0.9];
        let mut grad = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                grad[i * cols + j] = (a[i] * b[j]).sqrt(); // g = sqrt(a_i b_j) > 0
            }
        }
        let opt = AdaFactor {
            beta1: 0.0,
            eps1: 0.0,
            eps2: 0.0,
            clip: 1e9,
            decay: 0.8,
            relative_step: false,
        };
        let mut st = AdaFactorState::matrix(rows, cols, false);
        let mut param = vec![0.0f32; rows * cols];
        opt.step(&mut param, &grad, &mut st, 1.0);
        for (idx, &p) in param.iter().enumerate() {
            assert!((p + 1.0).abs() < 1e-5, "param[{idx}] = {p}, expected −1");
        }
    }

    #[test]
    fn converges_on_diagonal_quadratic() {
        // f(x) = Σ s_i x_i² , ill-conditioned scales. AdaFactor with relative
        // step should drive the loss down by orders of magnitude.
        let (rows, cols) = (8, 6);
        let n = rows * cols;
        let scales: Vec<f32> = (0..n).map(|i| 0.01 + 3.0 * ((i * 7 % 11) as f32)).collect();
        let mut x: Vec<f32> = (0..n).map(|i| 1.0 - 0.5 * ((i % 5) as f32)).collect();
        let loss = |x: &[f32]| -> f32 { x.iter().zip(&scales).map(|(xi, s)| s * xi * xi).sum() };

        let l0 = loss(&x);
        let opt = AdaFactor::default();
        let mut st = AdaFactorState::matrix(rows, cols, false);
        let mut grad = vec![0.0f32; n];
        for _ in 0..400 {
            for idx in 0..n {
                grad[idx] = 2.0 * scales[idx] * x[idx];
            }
            opt.step(&mut x, &grad, &mut st, 0.3);
        }
        let l1 = loss(&x);
        // Sanity bound: relative step floors descent at the eps2·lr scale, so we
        // assert orders-of-magnitude descent (≥100×) rather than to-zero.
        assert!(l1 < l0 * 1e-2, "loss {l0} -> {l1} (insufficient descent)");
    }

    #[test]
    fn momentum_path_also_descends() {
        let (rows, cols) = (5, 5);
        let n = rows * cols;
        let mut x: Vec<f32> = (0..n).map(|i| 0.8 - 0.1 * (i as f32 % 3.0)).collect();
        let loss = |x: &[f32]| -> f32 { x.iter().map(|v| v * v).sum() };
        let l0 = loss(&x);
        let opt = AdaFactor { beta1: 0.9, ..AdaFactor::default() };
        let mut st = AdaFactorState::matrix(rows, cols, true);
        let mut grad = vec![0.0f32; n];
        for _ in 0..400 {
            for idx in 0..n {
                grad[idx] = 2.0 * x[idx];
            }
            opt.step(&mut x, &grad, &mut st, 0.3);
        }
        assert!(loss(&x) < l0 * 1e-3, "momentum descent insufficient: {l0} -> {}", loss(&x));
    }

    #[test]
    fn vector_fallback_uses_full_second_moment() {
        let st = AdaFactorState::vector(10, false);
        assert!(!st.factored, "1-D must not factor");
        assert_eq!(st.r.len(), 10);
        assert!(st.c.is_empty());
        // and it should still optimize a 1-D quadratic
        let opt = AdaFactor::default();
        let mut st = AdaFactorState::vector(10, false);
        let mut x = vec![1.0f32; 10];
        let mut g = vec![0.0f32; 10];
        let l0: f32 = x.iter().map(|v| v * v).sum();
        for _ in 0..300 {
            for i in 0..10 {
                g[i] = 2.0 * x[i];
            }
            opt.step(&mut x, &g, &mut st, 0.3);
        }
        let l1: f32 = x.iter().map(|v| v * v).sum();
        assert!(l1 < l0 * 1e-4, "1-D descent insufficient: {l0} -> {l1}");
    }

    #[test]
    fn update_rms_clipping_bounds_step() {
        // A huge gradient must not produce an unbounded step: with relative_step
        // off, lr=1, clip=1, the update RMS is ≤ 1 so |Δparam| stays O(1).
        let opt = AdaFactor {
            beta1: 0.0,
            relative_step: false,
            clip: 1.0,
            ..AdaFactor::default()
        };
        let mut st = AdaFactorState::matrix(4, 4, false);
        let mut param = vec![0.0f32; 16];
        let grad = vec![1e6f32; 16];
        opt.step(&mut param, &grad, &mut st, 1.0);
        let rms_step = (param.iter().map(|p| p * p).sum::<f32>() / 16.0).sqrt();
        assert!(rms_step <= 1.0 + 1e-4, "step RMS {rms_step} exceeded clip");
    }
}
