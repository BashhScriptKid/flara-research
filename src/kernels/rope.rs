//! RoPE — rotary position embedding applied to Q and K per head.
//!
//! Adjacent-pair convention: dim `2i` and `2i+1` form a 2D vector rotated by
//! angle `m·θ_i` at position `m`, with `θ_i = base^{-2i/d}`. The rotation per
//! pair is orthogonal:
//!
//! ```text
//! x'_{2i}   = x_{2i}·cos − x_{2i+1}·sin
//! x'_{2i+1} = x_{2i}·sin + x_{2i+1}·cos
//! ```
//!
//! RoPE has no learnable parameters. Because each pair rotation `R` is
//! orthogonal, the backward pass is `Rᵀ` (rotation by `−angle`), and applying
//! backward after forward recovers the input — used as a correctness test.
//!
//! `cos`/`sin` tables are precomputed up to `max_seq` so the hot path is a
//! table lookup plus four multiplies per pair.

/// Precomputed rotary tables for a fixed head dimension.
pub struct Rope {
    dim: usize,
    half: usize,
    /// `cos[pos*half + i]`, `sin[...]` for position `pos`, pair `i`.
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    pub fn new(dim: usize, max_seq: usize, base: f32) -> Self {
        assert_eq!(dim % 2, 0, "RoPE head dim must be even");
        let half = dim / 2;
        let mut cos = vec![0.0f32; max_seq * half];
        let mut sin = vec![0.0f32; max_seq * half];
        for pos in 0..max_seq {
            for i in 0..half {
                let theta = (base).powf(-2.0 * i as f32 / dim as f32);
                let angle = pos as f32 * theta;
                cos[pos * half + i] = angle.cos();
                sin[pos * half + i] = angle.sin();
            }
        }
        Self { dim, half, cos, sin }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Apply RoPE in place to a single head vector at position `pos`.
    pub fn apply(&self, x: &mut [f32], pos: usize) {
        debug_assert_eq!(x.len(), self.dim);
        let base = pos * self.half;
        for i in 0..self.half {
            let c = self.cos[base + i];
            let s = self.sin[base + i];
            let (a, b) = (x[2 * i], x[2 * i + 1]);
            x[2 * i] = a * c - b * s;
            x[2 * i + 1] = a * s + b * c;
        }
    }

    /// Backward pass: propagate gradient through the rotation (apply `Rᵀ`).
    pub fn apply_backward(&self, dx: &mut [f32], pos: usize) {
        debug_assert_eq!(dx.len(), self.dim);
        let base = pos * self.half;
        for i in 0..self.half {
            let c = self.cos[base + i];
            let s = self.sin[base + i];
            let (a, b) = (dx[2 * i], dx[2 * i + 1]);
            dx[2 * i] = a * c + b * s;
            dx[2 * i + 1] = -a * s + b * c;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_zero_is_identity() {
        let rope = Rope::new(8, 16, 10000.0);
        let orig = [1.0f32, -2.0, 3.0, 0.5, -1.0, 2.5, 0.0, -0.25];
        let mut x = orig;
        rope.apply(&mut x, 0);
        for (a, b) in orig.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn rotation_preserves_norm() {
        let rope = Rope::new(8, 64, 10000.0);
        let orig = [1.0f32, -2.0, 3.0, 0.5, -1.0, 2.5, 0.7, -0.25];
        let n0: f32 = orig.iter().map(|v| v * v).sum();
        let mut x = orig;
        rope.apply(&mut x, 7);
        let n1: f32 = x.iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() < 1e-4, "norm changed: {n0} vs {n1}");
    }

    #[test]
    fn backward_inverts_forward() {
        // R is orthogonal so Rᵀ = R⁻¹: backward after forward recovers input.
        let rope = Rope::new(8, 64, 10000.0);
        let orig = [0.3f32, -1.2, 2.0, 0.9, -0.5, 1.1, 0.2, -2.0];
        let mut x = orig;
        rope.apply(&mut x, 13);
        rope.apply_backward(&mut x, 13);
        for (a, b) in orig.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-5, "not recovered: {a} vs {b}");
        }
    }
}
