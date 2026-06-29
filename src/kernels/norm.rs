//! RMSNorm — pre-norm normalization used before attention and FFN.
//!
//! `y_i = x_i · r · g_i` where `r = 1/sqrt(mean(x²) + eps)` and `g` is a
//! per-feature learned gain. Cheaper than LayerNorm (no mean subtraction, no
//! bias) and numerically stable; the gain `g` is the only parameter.
//!
//! Forward returns the per-row `r` so the backward pass can reuse it as a
//! training cache instead of recomputing the reduction.

/// RMSNorm forward for a single length-`d` vector.
///
/// Returns `r = 1/sqrt(mean(x²) + eps)`, writing the normalized result to `y`.
pub fn forward(x: &[f32], gain: &[f32], eps: f32, y: &mut [f32]) -> f32 {
    let d = x.len();
    debug_assert_eq!(gain.len(), d);
    debug_assert_eq!(y.len(), d);
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let r = 1.0 / (ss / d as f32 + eps).sqrt();
    for i in 0..d {
        y[i] = x[i] * r * gain[i];
    }
    r
}

/// RMSNorm backward for a single vector.
///
/// Given upstream `dy`, the cached `r` from [`forward`], and the original `x`,
/// accumulates input gradient into `dx` and gain gradient into `dg`.
///
/// Derivation: with `a_i = dy_i·g_i` and `dot = Σ a_i x_i`,
/// - `dg_i = dy_i · x_i · r`
/// - `dx_j = r·(a_j - x_j·r²·dot/d)`
///
/// `dx` and `dg` are accumulated (`+=`) so callers can sum over tokens; zero
/// them first if a fresh gradient is wanted.
pub fn backward(x: &[f32], gain: &[f32], dy: &[f32], r: f32, dx: &mut [f32], dg: &mut [f32]) {
    let d = x.len();
    debug_assert_eq!(gain.len(), d);
    debug_assert_eq!(dy.len(), d);
    debug_assert_eq!(dx.len(), d);
    debug_assert_eq!(dg.len(), d);

    let mut dot = 0.0f32;
    for i in 0..d {
        dot += dy[i] * gain[i] * x[i];
    }
    let coef = r * r * dot / d as f32;
    for j in 0..d {
        let a = dy[j] * gain[j];
        dx[j] += r * (a - x[j] * coef);
        dg[j] += dy[j] * x[j] * r;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
    }

    #[test]
    fn unit_gain_normalizes_rms_to_one() {
        let x = [3.0f32, -4.0, 1.0, 2.0, -2.0, 0.5, -1.5, 0.0];
        let g = [1.0f32; 8];
        let mut y = [0.0f32; 8];
        forward(&x, &g, 0.0, &mut y);
        let ms: f32 = y.iter().map(|v| v * v).sum::<f32>() / y.len() as f32;
        assert!((ms - 1.0).abs() < 1e-5, "rms not unit: {ms}");
    }

    #[test]
    fn backward_gradcheck() {
        let d = 8;
        let mut rng = Lcg(0x1234_5678);
        let x: Vec<f32> = (0..d).map(|_| rng.f()).collect();
        let g: Vec<f32> = (0..d).map(|_| 0.5 + rng.f().abs()).collect();
        let r_out: Vec<f32> = (0..d).map(|_| rng.f()).collect(); // dL/dy
        let eps = 1e-5f32;

        let mut y = vec![0.0f32; d];
        let r = forward(&x, &g, eps, &mut y);
        let mut dx = vec![0.0f32; d];
        let mut dg = vec![0.0f32; d];
        backward(&x, &g, &r_out, r, &mut dx, &mut dg);

        let loss = |xx: &[f32], gg: &[f32]| -> f32 {
            let mut yy = vec![0.0f32; d];
            forward(xx, gg, eps, &mut yy);
            yy.iter().zip(&r_out).map(|(a, b)| a * b).sum()
        };
        let h = 1e-3f32;
        for i in 0..d {
            let mut xp = x.clone();
            xp[i] += h;
            let lp = loss(&xp, &g);
            xp[i] -= 2.0 * h;
            let lm = loss(&xp, &g);
            assert!(((lp - lm) / (2.0 * h) - dx[i]).abs() < 1e-2, "dx[{i}]");

            let mut gp = g.clone();
            gp[i] += h;
            let lp = loss(&x, &gp);
            gp[i] -= 2.0 * h;
            let lm = loss(&x, &gp);
            assert!(((lp - lm) / (2.0 * h) - dg[i]).abs() < 1e-2, "dg[{i}]");
        }
    }
}
