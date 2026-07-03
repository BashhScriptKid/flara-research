//! Attention Q/K/V/O projection, behind a compile-time compression switch.
//!
//! The projections are Monarch-compressed by default (sharing a global real
//! atom dictionary `d1`/`d2` across every projection in the model — the
//! attention analogue of the FFN's complex dictionary), so the whole attention
//! block stays cache-resident. Flipping [`DENSE_ATTN`] to `true` swaps all four
//! projections to plain uncompressed f32 matmuls — an ablation lever for
//! measuring *what compressing attention actually costs* in loss and
//! throughput. Because the switch is a `const`, the unused construction path
//! is dead-code-eliminated; the runtime `match` in `forward`/`backward`
//! collapses to one arm. Mirrors the `INIT_FROM_DENSE` pattern in `fft.rs`.

use crate::kernels::monarch::{FwdCache, Grads as MonarchGrads, SharedMonarchProj};
use crate::kernels::optimizer::{AdaFactor, AdaFactorState};

/// Ablation switch: `false` = Monarch-compressed Q/K/V/O (default); `true` = dense f32.
pub const DENSE_ATTN: bool = false;

enum ProjKind {
    Monarch { proj: SharedMonarchProj },
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
    /// The learned parameter slice: for the Monarch path, `a1` followed by `a2`
    /// concatenated (matches [`SharedMonarchProj`]'s natural `(p·q·m, nd)` row
    /// layout for both halves, so it factors directly for the optimizer); the
    /// dense weight for the dense path.
    pub fn params(&self) -> Vec<f32> {
        match &self.kind {
            ProjKind::Monarch { proj } => {
                let mut v = Vec::with_capacity(proj.a1.len() + proj.a2.len());
                v.extend_from_slice(&proj.a1);
                v.extend_from_slice(&proj.a2);
                v
            }
            ProjKind::Dense { w } => w.clone(),
        }
    }

    /// Overwrite the learned parameters from a checkpoint slice (length must match).
    pub fn set_params(&mut self, src: &[f32]) {
        match &mut self.kind {
            ProjKind::Monarch { proj } => {
                assert_eq!(proj.a1.len() + proj.a2.len(), src.len(), "AttnProj param length mismatch on restore");
                let (a1, a2) = src.split_at(proj.a1.len());
                proj.a1.copy_from_slice(a1);
                proj.a2.copy_from_slice(a2);
            }
            ProjKind::Dense { w } => {
                assert_eq!(w.len(), src.len(), "AttnProj param length mismatch on restore");
                w.copy_from_slice(src);
            }
        }
    }
}

/// Projection gradients. `d_param` matches the stored parameter (`a1`+`a2`
/// concatenated for the Monarch path, or the dense weight); `d_d1`/`d_d2` are
/// the shared-dictionary contribution (empty for dense); `d_x` is the input
/// gradient.
pub struct ProjGrads {
    pub d_param: Vec<f32>,
    pub d_d1: Vec<f32>,
    pub d_d2: Vec<f32>,
    pub d_x: Vec<f32>,
}

/// Same as [`ProjGrads`], batched over `t_len` tokens: `d_x` is
/// `[t_len, in_]`; `d_param`/`d_d1`/`d_d2` are already summed across every
/// token in the batch (they're weight gradients, one value per parameter
/// regardless of batch size).
pub struct ProjGradsBatch {
    pub d_param: Vec<f32>,
    pub d_d1: Vec<f32>,
    pub d_d2: Vec<f32>,
    pub d_x: Vec<f32>,
}

impl AttnProj {
    /// Construct per the [`DENSE_ATTN`] switch. `block` is the Monarch block
    /// size `b = m²` (so `m = √block`); `k` is the atom count `nd`.
    pub fn new(out: usize, in_: usize, block: usize, k: usize, seed: u64) -> Self {
        if DENSE_ATTN {
            Self::new_dense(out, in_, seed)
        } else {
            Self::new_monarch(out, in_, block, k, seed)
        }
    }

    /// Force the Monarch path (used in tests regardless of the switch).
    pub fn new_monarch(out: usize, in_: usize, block: usize, k: usize, seed: u64) -> Self {
        let m = (block as f64).sqrt() as usize;
        assert_eq!(m * m, block, "block must be a perfect square");
        assert_eq!(out % block, 0, "out ({out}) must be divisible by block ({block})");
        assert_eq!(in_ % block, 0, "in_ ({in_}) must be divisible by block ({block})");
        let proj = SharedMonarchProj::new(out / block, in_ / block, m, k, seed);
        Self { out, in_, kind: ProjKind::Monarch { proj } }
    }

    /// Force the dense path (used in tests regardless of the switch).
    pub fn new_dense(out: usize, in_: usize, seed: u64) -> Self {
        let w = crate::kernels::fft::init_coeffs_random(out * in_, seed, 0.02);
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
            ProjKind::Monarch { proj } => proj.a1.len() + proj.a2.len(),
            ProjKind::Dense { w } => w.len(),
        }
    }

    /// `d1`/`d2` are the shared Monarch dictionary; ignored by the dense path.
    pub fn forward(&self, d1: &[f32], d2: &[f32], x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_, "projection input shape mismatch");
        match &self.kind {
            ProjKind::Monarch { proj } => proj.forward(d1, d2, x).0,
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

    /// Batched forward: process `t_len` tokens at once in a single dispatch
    /// (Monarch path) — `x` is `[t_len, in_]`, output written to `y` which is
    /// `[t_len, out]`. Returns the cache `backward` needs for the Monarch path
    /// (empty for dense, which doesn't need one).
    pub fn forward_batch(&self, d1: &[f32], d2: &[f32], x: &[f32], y: &mut [f32], t_len: usize) -> FwdCache {
        match &self.kind {
            ProjKind::Monarch { proj } => {
                let (out, cache) = proj.forward_batch(d1, d2, x, t_len);
                y.copy_from_slice(&out);
                cache
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
                FwdCache { zs: Vec::new() }
            }
        }
    }

    /// Slice a single token's cache slice out of a batched `forward_batch`
    /// cache (no-op / empty for the dense path).
    pub fn zs_at<'a>(&self, cache: &'a FwdCache, token: usize) -> &'a [f32] {
        match &self.kind {
            ProjKind::Monarch { proj } => proj.zs_at(cache, token),
            ProjKind::Dense { .. } => &[],
        }
    }

    /// VJP. `dy` is the gradient w.r.t. the projection output; `zs` is this
    /// token's cache slice from `forward_batch` (ignored by the dense path).
    pub fn backward(&self, d1: &[f32], d2: &[f32], x: &[f32], zs: &[f32], dy: &[f32]) -> ProjGrads {
        debug_assert_eq!(x.len(), self.in_, "projection input shape mismatch");
        debug_assert_eq!(dy.len(), self.out, "projection grad-output shape mismatch");
        match &self.kind {
            ProjKind::Monarch { proj } => {
                let mut dx = vec![0.0f32; self.in_];
                let g: MonarchGrads = proj.backward(d1, d2, x, zs, dy, &mut dx);
                let mut d_param = Vec::with_capacity(g.da1.len() + g.da2.len());
                d_param.extend_from_slice(&g.da1);
                d_param.extend_from_slice(&g.da2);
                ProjGrads { d_param, d_d1: g.dd1, d_d2: g.dd2, d_x: dx }
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
                ProjGrads { d_param: d_w, d_d1: Vec::new(), d_d2: Vec::new(), d_x }
            }
        }
    }

    /// Batched VJP: `x` is `[t_len, in_]`, `cache` is the [`FwdCache`] from a
    /// matching `forward_batch` call, `dy` is `[t_len, out]`. For the Monarch
    /// path this reconstructs each weight block once and reuses it across
    /// every token (see [`SharedMonarchProj::backward_batch`]) instead of
    /// once per token, which is what `backward` called in a loop would do.
    pub fn backward_batch(&self, d1: &[f32], d2: &[f32], x: &[f32], cache: &FwdCache, dy: &[f32], t_len: usize) -> ProjGradsBatch {
        match &self.kind {
            ProjKind::Monarch { proj } => {
                let mut dx = vec![0.0f32; t_len * self.in_];
                let g: MonarchGrads = proj.backward_batch(d1, d2, x, &cache.zs, dy, &mut dx, t_len);
                let mut d_param = Vec::with_capacity(g.da1.len() + g.da2.len());
                d_param.extend_from_slice(&g.da1);
                d_param.extend_from_slice(&g.da2);
                ProjGradsBatch { d_param, d_d1: g.dd1, d_d2: g.dd2, d_x: dx }
            }
            ProjKind::Dense { w } => {
                let mut d_w = vec![0.0f32; self.out * self.in_];
                let mut d_x = vec![0.0f32; t_len * self.in_];
                for t in 0..t_len {
                    let xi = &x[t * self.in_..(t + 1) * self.in_];
                    let dyi = &dy[t * self.out..(t + 1) * self.out];
                    let dxi = &mut d_x[t * self.in_..(t + 1) * self.in_];
                    for o in 0..self.out {
                        let dyo = dyi[o];
                        let row = &w[o * self.in_..(o + 1) * self.in_];
                        let dwr = &mut d_w[o * self.in_..(o + 1) * self.in_];
                        for i in 0..self.in_ {
                            dwr[i] += dyo * xi[i];
                            dxi[i] += row[i] * dyo;
                        }
                    }
                }
                ProjGradsBatch { d_param: d_w, d_d1: Vec::new(), d_d2: Vec::new(), d_x }
            }
        }
    }

    /// Factoring shape `(rows, cols)` for this projection's optimizer state.
    /// Monarch's concatenated `a1`+`a2` factors as `(2·p·q·m, nd)` (each half
    /// is already `(p·q·m, nd)` row-major); the dense weight as `(out, in)`.
    pub fn opt_shape(&self) -> (usize, usize) {
        match &self.kind {
            ProjKind::Monarch { proj } => (2 * proj.p * proj.q * proj.m, proj.nd),
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
            ProjKind::Monarch { proj } => {
                let n1 = proj.a1.len();
                let mut p = Vec::with_capacity(n1 + proj.a2.len());
                p.extend_from_slice(&proj.a1);
                p.extend_from_slice(&proj.a2);
                af.step(&mut p, d_param, st, lr);
                proj.a1.copy_from_slice(&p[..n1]);
                proj.a2.copy_from_slice(&p[n1..]);
            }
            ProjKind::Dense { w } => af.step(w, d_param, st, lr),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn randvec(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = seed;
        (0..n).map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        }).collect()
    }
}
