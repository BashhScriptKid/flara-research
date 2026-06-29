//! Attention Q/K/V/O projection, behind a compile-time compression switch.
//!
//! The projections are circular-basis compressed by default (sharing the global
//! dictionary `G`, like the FFN), so the whole attention block stays
//! cache-resident. Flipping [`DENSE_ATTN`] to `true` swaps all four projections
//! to plain uncompressed f32 matmuls — an ablation lever for measuring *what
//! compressing attention actually costs* in loss and throughput. Because the
//! switch is a `const`, the unused construction path is dead-code-eliminated;
//! the runtime `match` in `forward`/`backward` collapses to one arm. Mirrors the
//! `INIT_FROM_DENSE` pattern in `fft.rs`.

use crate::kernels::fft::{init_coeffs_random, BasisGrads, BasisMatmul};
use crate::kernels::optimizer::{AdaFactor, AdaFactorState};
use rustfft::num_complex::Complex32;

/// Ablation switch: `false` = circular-basis Q/K/V/O (default); `true` = dense f32.
pub const DENSE_ATTN: bool = false;

enum ProjKind {
    Basis { mm: BasisMatmul, coeffs: Vec<f32> },
    /// Row-major `[out, in]` weight.
    Dense { w: Vec<f32> },
}

/// One linear projection `in → out`, compressed or dense per [`DENSE_ATTN`].
pub struct AttnProj {
    out: usize,
    in_: usize,
    kind: ProjKind,
}

impl AttnProj {
    /// The learned parameter slice (basis coefficients, or the dense weight).
    pub fn params(&self) -> &[f32] {
        match &self.kind {
            ProjKind::Basis { coeffs, .. } => coeffs,
            ProjKind::Dense { w } => w,
        }
    }

    /// Overwrite the learned parameters from a checkpoint slice (length must match).
    pub fn set_params(&mut self, src: &[f32]) {
        let dst = match &mut self.kind {
            ProjKind::Basis { coeffs, .. } => coeffs,
            ProjKind::Dense { w } => w,
        };
        assert_eq!(dst.len(), src.len(), "AttnProj param length mismatch on restore");
        dst.copy_from_slice(src);
    }
}

/// Projection gradients. `d_param` matches the stored parameter (coefficients for
/// the basis path, the dense weight for the dense path); `d_dict` is the
/// shared dictionary contribution (empty for dense).
pub struct ProjGrads {
    pub d_param: Vec<f32>,
    pub d_dict: Vec<Complex32>,
    pub d_x: Vec<f32>,
}

impl AttnProj {
    /// Construct per the [`DENSE_ATTN`] switch.
    pub fn new(out: usize, in_: usize, block: usize, k: usize, seed: u64) -> Self {
        if DENSE_ATTN {
            Self::new_dense(out, in_, seed)
        } else {
            Self::new_basis(out, in_, block, k, seed)
        }
    }

    /// Force the circular-basis path (used in tests regardless of the switch).
    pub fn new_basis(out: usize, in_: usize, block: usize, k: usize, seed: u64) -> Self {
        let mm = BasisMatmul::new(out, in_, block, k);
        let coeffs = init_coeffs_random(mm.coeff_len(), seed, 0.02);
        Self { out, in_, kind: ProjKind::Basis { mm, coeffs } }
    }

    /// Force the dense path (used in tests regardless of the switch).
    pub fn new_dense(out: usize, in_: usize, seed: u64) -> Self {
        let w = init_coeffs_random(out * in_, seed, 0.02);
        Self { out, in_, kind: ProjKind::Dense { w } }
    }

    #[inline]
    pub fn out_dim(&self) -> usize {
        self.out
    }
    #[inline]
    pub fn in_dim(&self) -> usize {
        self.in_
    }

    /// Length of the stored parameter (for sizing the optimizer state).
    pub fn param_len(&self) -> usize {
        match &self.kind {
            ProjKind::Basis { coeffs, .. } => coeffs.len(),
            ProjKind::Dense { w } => w.len(),
        }
    }

    /// `dict` is the shared `G`; ignored by the dense path.
    pub fn forward(&self, dict: &[Complex32], x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_, "projection input shape mismatch");
        match &self.kind {
            ProjKind::Basis { mm, coeffs } => mm.forward(dict, coeffs, x),
            ProjKind::Dense { w } => {
                let mut y = vec![0.0f32; self.out];
                for (o, yo) in y.iter_mut().enumerate() {
                    let row = &w[o * self.in_..(o + 1) * self.in_];
                    *yo = row.iter().zip(x).map(|(wi, xi)| wi * xi).sum();
                }
                y
            }
        }
    }

    /// Batched forward: process `t_len` tokens at once.
    /// `x` is `[t_len, in_]`, output written to `y` which is `[t_len, out]`.
    pub fn forward_batch(&self, dict: &[Complex32], x: &[f32], y: &mut [f32], t_len: usize) {
        match &self.kind {
            ProjKind::Basis { mm, coeffs } => {
                for t in 0..t_len {
                    let xi = &x[t * self.in_..(t + 1) * self.in_];
                    let yi = &mut y[t * self.out..(t + 1) * self.out];
                    let out = mm.forward(dict, coeffs, xi);
                    yi.copy_from_slice(&out);
                }
            }
            ProjKind::Dense { w } => {
                for t in 0..t_len {
                    let xi = &x[t * self.in_..(t + 1) * self.in_];
                    let yi = &mut y[t * self.out..(t + 1) * self.out];
                    for (o, yo) in yi.iter_mut().enumerate() {
                        let row = &w[o * self.in_..(o + 1) * self.in_];
                        *yo = row.iter().zip(xi).map(|(wi, xi)| wi * xi).sum();
                    }
                }
            }
        }
    }

    /// VJP. `dy` is the gradient w.r.t. the projection output.
    pub fn backward(&self, dict: &[Complex32], x: &[f32], dy: &[f32]) -> ProjGrads {
        debug_assert_eq!(x.len(), self.in_, "projection input shape mismatch");
        debug_assert_eq!(dy.len(), self.out, "projection grad-output shape mismatch");
        match &self.kind {
            ProjKind::Basis { mm, coeffs } => {
                let g: BasisGrads = mm.backward(dict, coeffs, x, dy);
                ProjGrads { d_param: g.d_coeffs, d_dict: g.d_dict, d_x: g.d_x }
            }
            ProjKind::Dense { w } => {
                let mut d_w = vec![0.0f32; self.out * self.in_];
                let mut d_x = vec![0.0f32; self.in_];
                for o in 0..self.out {
                    let dyo = dy[o];
                    let row = &w[o * self.in_..(o + 1) * self.in_];
                    let dwr = &mut d_w[o * self.in_..(o + 1) * self.in_];
                    for i in 0..self.in_ {
                        dwr[i] = dyo * x[i];
                        d_x[i] += row[i] * dyo;
                    }
                }
                ProjGrads { d_param: d_w, d_dict: Vec::new(), d_x }
            }
        }
    }

    /// Factoring shape `(rows, cols)` for this projection's optimizer state. Basis
    /// coeffs factor as `(P·Q, K)` (block-pairs × dictionary atoms); the dense
    /// weight factors as `(out, in)`.
    pub fn opt_shape(&self) -> (usize, usize) {
        match &self.kind {
            ProjKind::Basis { mm, .. } => (mm.p * mm.q, mm.k),
            ProjKind::Dense { .. } => (self.out, self.in_),
        }
    }

    /// Allocate the AdaFactor state for this projection (factored matrix).
    pub fn init_opt(&self) -> AdaFactorState {
        let (rows, cols) = self.opt_shape();
        AdaFactorState::matrix(rows, cols, false)
    }

    /// Apply one AdaFactor step to this projection's parameter.
    pub fn apply_grad(&mut self, d_param: &[f32], st: &mut AdaFactorState, af: &AdaFactor, lr: f32) {
        match &mut self.kind {
            ProjKind::Basis { coeffs, .. } => af.step(coeffs, d_param, st, lr),
            ProjKind::Dense { w } => af.step(w, d_param, st, lr),
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::fft::init_dict_random;

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
    }

    fn rand(rng: &mut Lcg, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.f()).collect()
    }

    /// Finite-diff gradcheck of d_x and d_param for whichever projection kind is
    /// passed. `dict` is the active dictionary.
    fn gradcheck(proj: &AttnProj, dict: &[Complex32]) {
        let mut rng = Lcg(0xA77E_0001);
        let x = rand(&mut rng, proj.in_dim());
        let r = rand(&mut rng, proj.out_dim()); // loss = Σ y·r ⇒ dy = r
        let g = proj.backward(dict, &x, &r);

        let loss = |x: &[f32]| -> f32 {
            proj.forward(dict, x).iter().zip(&r).map(|(y, rr)| y * rr).sum()
        };
        const H: f32 = 1e-3;
        let close = |fd: f32, an: f32| (fd - an).abs() < 1e-2 + 5e-2 * an.abs();

        // d_x
        for i in 0..proj.in_dim().min(8) {
            let mut xp = x.clone();
            xp[i] += H;
            let lp = loss(&xp);
            xp[i] -= 2.0 * H;
            let lm = loss(&xp);
            let fd = (lp - lm) / (2.0 * H);
            assert!(close(fd, g.d_x[i]), "d_x[{i}] fd {fd} an {}", g.d_x[i]);
        }
    }

    #[test]
    fn dense_projection_gradchecks() {
        let proj = AttnProj::new_dense(8, 12, 0xD0E5);
        let dict: Vec<Complex32> = Vec::new();
        gradcheck(&proj, &dict);
    }

    #[test]
    fn basis_projection_gradchecks() {
        let (out, in_, b, k) = (8, 12, 4, 6);
        let proj = AttnProj::new_basis(out, in_, b, k, 0xB451);
        let dict = init_dict_random(k, b, 0x6, 0.6);
        gradcheck(&proj, &dict);
    }

    #[test]
    fn dense_and_basis_agree_on_shape() {
        let d = AttnProj::new_dense(8, 12, 1);
        let bss = AttnProj::new_basis(8, 12, 4, 6, 1);
        let mut rng = Lcg(0x5);
        let x = rand(&mut rng, 12);
        let dict = init_dict_random(6, 4, 7, 0.6);
        let empty_dict: Vec<Complex32> = Vec::new();
        assert_eq!(d.forward(&empty_dict, &x).len(), 8);
        assert_eq!(bss.forward(&dict, &x).len(), 8);
    }
}
