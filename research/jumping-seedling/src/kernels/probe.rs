//! Early-exit confidence probe (CALM-style) — one per layer.
//!
//! A single linear `H → 1` head sitting on the pre-normed hidden state at the top
//! of each layer. It predicts, per token, whether the running prediction has
//! already converged so decoding can halt before the remaining layers run. On
//! CPU this is the lever that buys variable depth for free: easy tokens exit
//! shallow, hard tokens run deep, and — unlike on a GPU where a warp stalls on
//! its laggard lane — a single-token CPU decode loop simply stops, so the saved
//! layers are real wall-clock savings.
//!
//! **Gradient-stopped input.** The probe is trained as an auxiliary head: its
//! gradient updates the probe's own `w`/`bias`, but does NOT flow back into the
//! hidden state. If it did, the probe could lower its own loss by degrading the
//! representation (making tokens "look easy"), corrupting the backbone. So
//! [`backward`](ExitProbe::backward) returns grads for the probe parameters only
//! — there is deliberately no `d_h`. Supervision comes from the layer
//! (CALM: BCE/KL of the probe logit against whether the shallow prediction
//! matched the full-depth one); this kernel only owns the linear head.

/// Per-layer linear early-exit probe.
pub struct ExitProbe {
    /// Weight over the hidden dimension, length `hidden`.
    pub w: Vec<f32>,
    pub bias: f32,
}

/// Probe gradients (w.r.t. the probe parameters only — input is gradient-stopped).
pub struct ProbeGrads {
    pub d_w: Vec<f32>,
    pub d_bias: f32,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl ExitProbe {
    /// Zero-initialized probe (bias 0 ⇒ p=0.5 ⇒ neutral halting at step 0, the
    /// CALM default before the auxiliary loss is annealed in).
    pub fn new(hidden: usize) -> Self {
        Self { w: vec![0.0; hidden], bias: 0.0 }
    }

    /// Deterministic small-random init for testing/standalone use.
    pub fn with_init(hidden: usize, seed: u64) -> Self {
        let mut s = seed | 1;
        let mut w = vec![0.0f32; hidden];
        for wi in w.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *wi = (((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0) * 0.1;
        }
        Self { w, bias: 0.1 }
    }

    #[inline]
    pub fn hidden(&self) -> usize {
        self.w.len()
    }

    /// Raw exit logit `w·h + bias` for one token.
    pub fn logit(&self, h: &[f32]) -> f32 {
        debug_assert_eq!(h.len(), self.w.len(), "hidden mismatch");
        let mut acc = self.bias;
        for (wi, hi) in self.w.iter().zip(h) {
            acc += wi * hi;
        }
        acc
    }

    /// Exit probability `σ(w·h + bias)` for one token.
    #[inline]
    pub fn forward(&self, h: &[f32]) -> f32 {
        sigmoid(self.logit(h))
    }

    /// Backward through the probe. `p` is the forward output (the sigmoid), `d_p`
    /// the upstream gradient of the probe loss w.r.t. `p`. Returns grads for the
    /// probe's own parameters; the input gradient is intentionally NOT produced
    /// (gradient-stopped backbone).
    pub fn backward(&self, h: &[f32], p: f32, d_p: f32) -> ProbeGrads {
        debug_assert_eq!(h.len(), self.w.len(), "hidden mismatch");
        // d_logit = d_p · σ'(logit) = d_p · p(1−p)
        let d_logit = d_p * p * (1.0 - p);
        let d_w = h.iter().map(|&hi| d_logit * hi).collect();
        ProbeGrads { d_w, d_bias: d_logit }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_is_sigmoid_of_affine() {
        let probe = ExitProbe { w: vec![1.0, -2.0, 0.5], bias: 0.25 };
        let h = [0.4, 1.0, -0.6];
        let z = 1.0 * 0.4 - 2.0 * 1.0 + 0.5 * -0.6 + 0.25f32;
        let expect = 1.0 / (1.0 + (-z).exp());
        assert!((probe.forward(&h) - expect).abs() < 1e-6);
    }

    #[test]
    fn neutral_at_zero_init() {
        let probe = ExitProbe::new(16);
        let h = vec![0.7f32; 16];
        assert!((probe.forward(&h) - 0.5).abs() < 1e-7, "zero-init probe must be neutral");
    }

    #[test]
    fn backward_gradcheck() {
        // Loss = r · p ; check d_w, d_bias against central differences.
        let probe = ExitProbe::with_init(12, 0xBEEF);
        let h: Vec<f32> = (0..12).map(|i| 0.3 * (i as f32) - 1.0).collect();
        let r = 0.7f32;

        let p = probe.forward(&h);
        let g = probe.backward(&h, p, r); // d_p = r

        const EPS: f32 = 1e-4;
        // d_bias
        let mut up = ExitProbe { w: probe.w.clone(), bias: probe.bias + EPS };
        let mut dn = ExitProbe { w: probe.w.clone(), bias: probe.bias - EPS };
        let fd_b = (r * up.forward(&h) - r * dn.forward(&h)) / (2.0 * EPS);
        assert!((fd_b - g.d_bias).abs() < 2e-3, "d_bias fd {fd_b} an {}", g.d_bias);
        // d_w
        for i in 0..probe.w.len() {
            up = ExitProbe { w: probe.w.clone(), bias: probe.bias };
            dn = ExitProbe { w: probe.w.clone(), bias: probe.bias };
            up.w[i] += EPS;
            dn.w[i] -= EPS;
            let fd = (r * up.forward(&h) - r * dn.forward(&h)) / (2.0 * EPS);
            assert!((fd - g.d_w[i]).abs() < 2e-3, "d_w[{i}] fd {fd} an {}", g.d_w[i]);
        }
    }

    #[test]
    fn gradient_stopped_no_input_grad() {
        // Structural guarantee: ProbeGrads carries no input gradient field.
        let probe = ExitProbe::with_init(4, 7);
        let h = [0.1, 0.2, 0.3, 0.4];
        let p = probe.forward(&h);
        let g = probe.backward(&h, p, 1.0);
        assert_eq!(g.d_w.len(), probe.hidden());
    }
}
