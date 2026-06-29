//! Block-Tensor-Train (BTT) matmul kernel with Kronecker-structured atoms.
//!
//! Each weight `W[out, in]` is tiled into `m2×m1` blocks. Each block is a real
//! linear combination of shared atom matrices from `BttDict`. The atoms have
//! Kronecker structure: `atom_k = kron(A_k, B_k)` where `A_k` and `B_k` are
//! `mf×mf` factors (`mf = √b`). This enables a fast two-stage matvec:
//!
//! ```text
//! kron(A, B) @ x  =  vec(A @ X @ B^T)
//! ```
//!
//! where `X` is the `mf×mf` row-major reshape of `x`. Each stage costs `O(mf³)`
//! instead of `O(b²)`, giving a `b/mf = mf` × improvement per atom.
//!
//! For b=64, mf=8: 1024 ops per atom (vs 4096 naive), K=32 atoms per block pair.

use crate::kernels::gemm;

// ---------------------------------------------------------------------------
// Fused 8×8 matmul kernel — the inner workhorse of the Kronecker BTT.
// ---------------------------------------------------------------------------

/// `c[..mf*mf] += alpha * a[..mf*mf] @ b[..mf*mf]` where all three are mf×mf row-major.
///
/// The current `gemm::dot`-based approach calls the dot-product kernel mf² times
/// per matmul (each with AVX2 feature detection + horizontal reduction). This
/// fused version does mf² FMA ops with zero horizontal reductions — all work is
/// row-oriented and maps directly to AVX2 `_mm256_fmadd_ps`.
#[inline(always)]
fn matmul8x8(c: &mut [f32], a: &[f32], b: &[f32], alpha: f32, mf: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if mf == 8 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { matmul8x8_avx2(c.as_mut_ptr(), a.as_ptr(), b.as_ptr(), alpha) };
            return;
        }
    }
    matmul8x8_scalar(c, a, b, alpha, mf);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul8x8_avx2(c: *mut f32, a: *const f32, b: *const f32, alpha: f32) {
    use core::arch::x86_64::*;
    let alpha_v = _mm256_set1_ps(alpha);
    for i in 0..8 {
        let mut acc = _mm256_setzero_ps();
        let a_row = a.add(i * 8);
        let mut k = 0;
        while k < 8 {
            let a_ik = _mm256_set1_ps(*a_row.add(k));
            let b_row = _mm256_loadu_ps(b.add(k * 8));
            acc = _mm256_fmadd_ps(a_ik, b_row, acc);
            k += 1;
        }
        let c_row = _mm256_loadu_ps(c.add(i * 8));
        _mm256_storeu_ps(c.add(i * 8), _mm256_fmadd_ps(alpha_v, acc, c_row));
    }
}

/// Like `matmul8x8` but stores `alpha * A @ B` directly (no accumulation into c).
/// Eliminates the need for the caller to zero c first.
#[inline(always)]
fn matmul8x8_init(c: &mut [f32], a: &[f32], b: &[f32], alpha: f32, mf: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if mf == 8 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { matmul8x8_init_avx2(c.as_mut_ptr(), a.as_ptr(), b.as_ptr(), alpha) };
            return;
        }
    }
    matmul8x8_init_scalar(c, a, b, alpha, mf);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul8x8_init_avx2(c: *mut f32, a: *const f32, b: *const f32, alpha: f32) {
    use core::arch::x86_64::*;
    let alpha_v = _mm256_set1_ps(alpha);
    for i in 0..8 {
        let mut acc = _mm256_setzero_ps();
        let a_row = a.add(i * 8);
        let mut k = 0;
        while k < 8 {
            let a_ik = _mm256_set1_ps(*a_row.add(k));
            let b_row = _mm256_loadu_ps(b.add(k * 8));
            acc = _mm256_fmadd_ps(a_ik, b_row, acc);
            k += 1;
        }
        _mm256_storeu_ps(c.add(i * 8), _mm256_mul_ps(alpha_v, acc));
    }
}

fn matmul8x8_init_scalar(c: &mut [f32], a: &[f32], b: &[f32], alpha: f32, mf: usize) {
    for i in 0..mf {
        for j in 0..mf {
            let mut s = 0.0f32;
            for k in 0..mf {
                s += a[i * mf + k] * b[k * mf + j];
            }
            c[i * mf + j] = alpha * s;
        }
    }
}

fn matmul8x8_scalar(c: &mut [f32], a: &[f32], b: &[f32], alpha: f32, mf: usize) {
    for i in 0..mf {
        for j in 0..mf {
            let mut s = 0.0f32;
            for k in 0..mf {
                s += a[i * mf + k] * b[k * mf + j];
            }
            c[i * mf + j] += alpha * s;
        }
    }
}

/// `out[..mf*mf] = inp[..mf*mf]^T` where both are mf×mf row-major.
#[inline(always)]
fn transpose8x8(out: &mut [f32], inp: &[f32], mf: usize) {
    for i in 0..mf {
        for j in 0..mf {
            out[j * mf + i] = inp[i * mf + j];
        }
    }
}

/// Shared Kronecker-factor dictionaries for the BTT decomposition.
///
/// `dict1` stores stage-1 factors: for each of `n_shared` atoms, two `mf×mf`
/// matrices A_k and B_k laid out as `[n_shared × 2 × mf × mf]`.
/// Atom k: `A_k = dict1[k*2*mf² .. (k*2+1)*mf²]`, `B_k = dict1[(k*2+1)*mf² .. (k+1)*2*mf²]`.
pub struct BttDict {
    pub dict1: Vec<f32>,
    pub dict2: Vec<f32>,
    pub n_shared: usize,
    pub m1: usize,
    pub m2: usize,
    pub mf: usize,
}

impl BttDict {
    pub fn new(n_shared: usize, m1: usize, m2: usize, mf: usize) -> Self {
        Self {
            dict1: vec![0.0; n_shared * 2 * mf * mf],
            dict2: vec![0.0; n_shared * 2 * mf * mf],
            n_shared, m1, m2, mf,
        }
    }

    #[inline]
    pub fn dict1_len(&self) -> usize {
        self.n_shared * 2 * self.mf * self.mf
    }

    #[inline]
    pub fn dict2_len(&self) -> usize {
        self.n_shared * 2 * self.mf * self.mf
    }
}

/// Apply `kron(A, B) @ x` where A, B are `mf×mf` and x is length `mf²`.
///
/// Computes `y = vec(A @ X @ B^T)` where X is the `mf×mf` row-major reshape of x.
/// `buf` must have length ≥ `2 * mf * mf` (scratch space).
#[inline]
pub fn kron_apply(mf: usize, a: &[f32], b: &[f32], x: &[f32], y: &mut [f32], buf: &mut [f32]) {
    debug_assert!(buf.len() >= 2 * mf * mf);
    let sz = mf * mf;

    let (b0, b1) = buf.split_at_mut(sz);

    // b1 = A @ X (X is already row-major mf×mf) — init (no fill needed)
    matmul8x8_init(b1, a, x, 1.0, mf);

    // b0 = B^T
    transpose8x8(b0, b, mf);

    // y = (A @ X) @ B^T — init (no fill needed)
    matmul8x8_init(y, b1, b0, 1.0, mf);
}

/// Apply `kron(A^T, B^T) @ x` where A, B are `mf×mf` and x is length `mf²`.
///
/// Computes `y = vec(A^T @ X @ B)` where X is the `mf×mf` row-major reshape of x.
/// Used for the input gradient in the backward pass.
/// `buf` must have length ≥ `2 * mf * mf` (scratch space).
#[inline]
pub fn kron_transpose_apply(mf: usize, a: &[f32], b: &[f32], x: &[f32], y: &mut [f32], buf: &mut [f32]) {
    debug_assert!(buf.len() >= 2 * mf * mf);
    let sz = mf * mf;

    let (b0, b1) = buf.split_at_mut(sz);

    // kron(A^T, B^T) @ vec(X) = vec(A^T @ X @ B)

    // b0 = A^T
    transpose8x8(b0, a, mf);

    // b1 = A^T @ X — init (no fill needed)
    matmul8x8_init(b1, b0, x, 1.0, mf);

    // Y = (A^T @ X) @ B — init (no fill needed)
    matmul8x8_init(y, b1, b, 1.0, mf);
}

/// Fused factor-gradient accumulation for one atom in the square case (m1 = m2 = mf²).
///
/// Given Kronecker atom `kron(A, B)` with factor matrices `a` and `b` (each mf²),
/// input block `x` (mf²) and output gradient block `dy` (mf²), accumulates:
///
/// ```text
/// dA += coeff * dY @ B @ X^T
/// dB += coeff * (A @ X)^T @ dY
/// ```
///
/// `buf` must have length ≥ `2 * mf * mf`. `d_dict1` is the full factor gradient
/// buffer; `dbase` is the offset for this atom's factors.
#[inline]
pub fn factor_grad8x8(
    mf: usize,
    a: &[f32],
    b: &[f32],
    x: &[f32],
    dy: &[f32],
    d_dict1: &mut [f32],
    dbase: usize,
    coeff: f32,
    buf: &mut [f32],
) {
    debug_assert!(buf.len() >= 2 * mf * mf);
    let sz = mf * mf;

    let (b0, b1) = buf.split_at_mut(sz);

    // b0 = X^T
    transpose8x8(b0, x, mf);

    // b1 = B @ X^T
    b1.fill(0.0);
    matmul8x8(b1, b, b0, 1.0, mf);

    // dA += coeff * dY @ (B @ X^T)
    {
        let (da, rest) = d_dict1[dbase..].split_at_mut(mf * mf);
        let (db, _) = rest.split_at_mut(mf * mf);
        matmul8x8(da, dy, b1, coeff, mf);

        // b0 = A @ X
        b0.fill(0.0);
        matmul8x8(b0, a, x, 1.0, mf);

        // b1 = (A @ X)^T
        transpose8x8(b1, b0, mf);

        // dB += coeff * (A @ X)^T @ dY
        matmul8x8(db, b1, dy, coeff, mf);
    }
}

/// Geometry + coefficients for one logical weight matrix `W[out, in]`
/// expressed over shared BTT dictionaries with Kronecker-structured atoms.
pub struct BttMatmul {
    pub m2: usize,
    pub m1: usize,
    pub p: usize,
    pub q: usize,
    pub k: usize,
    pub mf: usize,
}

impl BttMatmul {
    pub fn new(out: usize, in_: usize, m1: usize, m2: usize, k: usize, mf: usize) -> Self {
        assert!(m1 > 0 && m2 > 0 && k > 0 && mf > 0);
        assert_eq!(m1 % mf, 0, "m1 ({m1}) must be divisible by mf ({mf})");
        assert_eq!(m2 % mf, 0, "m2 ({m2}) must be divisible by mf ({mf})");
        assert_eq!(out % m2, 0, "out ({out}) must be divisible by m2 ({m2})");
        assert_eq!(in_ % m1, 0, "in_ ({in_}) must be divisible by m1 ({m1})");
        Self { m2, m1, p: out / m2, q: in_ / m1, k, mf }
    }

    #[inline]
    pub fn out_dim(&self) -> usize {
        self.p * self.m2
    }

    #[inline]
    pub fn in_dim(&self) -> usize {
        self.q * self.m1
    }

    #[inline]
    pub fn coeff_len(&self) -> usize {
        self.p * self.q * self.k
    }

    #[inline]
    fn coeff1<'a>(&self, coeffs: &'a [f32], pp: usize, qq: usize) -> &'a [f32] {
        let base = (pp * self.q + qq) * self.k;
        &coeffs[base..base + self.k]
    }

    /// Forward pass `y = W x`. For each output block `pp`, accumulate
    /// `Σ_qq Σ_k coeff[k] * kron(A_k, B_k) @ x_qq` using Kronecker matvec.
    pub fn forward(&self, dict: &BttDict, coeffs: &[f32], x: &[f32]) -> Vec<f32> {
        assert_eq!(coeffs.len(), self.coeff_len(), "coefficient shape mismatch");
        assert_eq!(x.len(), self.in_dim(), "input shape mismatch");

        let (m1, m2, mf) = (self.m1, self.m2, self.mf);
        let mut y = vec![0.0f32; self.out_dim()];
        let mut atom_out = vec![0.0f32; m2];
        let mut buf = vec![0.0f32; 2 * mf * mf];

        for pp in 0..self.p {
            let mut acc = vec![0.0f32; m2];
            for qq in 0..self.q {
                let c1 = self.coeff1(coeffs, pp, qq);
                let x_qq = &x[qq * m1..(qq + 1) * m1];

                // Accumulate: acc += Σ_k coeff[k] * kron(A_k, B_k) @ x_qq
                for kk in 0..self.k {
                    let a = c1[kk];
                    if a == 0.0 { continue; }
                    let fbase = kk * 2 * mf * mf;
                    let ak = &dict.dict1[fbase..fbase + mf * mf];
                    let bk = &dict.dict1[fbase + mf * mf..fbase + 2 * mf * mf];
                    kron_apply(mf, ak, bk, x_qq, &mut atom_out, &mut buf);
                    for r in 0..m2 {
                        acc[r] += a * atom_out[r];
                    }
                }
            }
            y[pp * m2..(pp + 1) * m2].copy_from_slice(&acc);
        }
        y
    }

    /// Batched forward: process `t_len` tokens at once.
    /// `x` is `[t_len, in_dim]`, output written to `y` which is `[t_len, out_dim]`.
    /// Reuses scratch buffers across all tokens — eliminates per-token allocation.
    pub fn forward_batch(&self, dict: &BttDict, coeffs: &[f32], x: &[f32], y: &mut [f32], t_len: usize) {
        assert_eq!(coeffs.len(), self.coeff_len(), "coefficient shape mismatch");
        assert_eq!(x.len(), t_len * self.in_dim(), "input shape mismatch");
        assert_eq!(y.len(), t_len * self.out_dim(), "output shape mismatch");

        let (m1, m2, mf) = (self.m1, self.m2, self.mf);
        y.fill(0.0);
        let mut atom_out = vec![0.0f32; m2];
        let mut buf = vec![0.0f32; 2 * mf * mf];
        let mut acc = vec![0.0f32; m2];

        for pp in 0..self.p {
            for qq in 0..self.q {
                let c1 = self.coeff1(coeffs, pp, qq);

                for kk in 0..self.k {
                    let a = c1[kk];
                    if a == 0.0 { continue; }
                    let fbase = kk * 2 * mf * mf;
                    let ak = &dict.dict1[fbase..fbase + mf * mf];
                    let bk = &dict.dict1[fbase + mf * mf..fbase + 2 * mf * mf];

                    for t in 0..t_len {
                        let x_qq = &x[t * self.in_dim() + qq * m1..t * self.in_dim() + (qq + 1) * m1];
                        kron_apply(mf, ak, bk, x_qq, &mut atom_out, &mut buf);
                        let y_pp = &mut y[t * self.out_dim() + pp * m2..t * self.out_dim() + (pp + 1) * m2];
                        for r in 0..m2 {
                            y_pp[r] += a * atom_out[r];
                        }
                    }
                }
            }
        }
    }

    /// Backward pass. Returns gradients w.r.t. coefficients, dictionary factors, and input `x`.
    pub fn backward(
        &self,
        dict: &BttDict,
        coeffs: &[f32],
        x: &[f32],
        dy: &[f32],
    ) -> BttGrads {
        assert_eq!(coeffs.len(), self.coeff_len());
        assert_eq!(x.len(), self.in_dim());
        assert_eq!(dy.len(), self.out_dim());

        let (m1, m2, mf) = (self.m1, self.m2, self.mf);
        let mut d_coeffs = vec![0.0f32; self.coeff_len()];
        let mut d_dict1 = vec![0.0f32; dict.dict1_len()];
        let mut d_x = vec![0.0f32; self.in_dim()];
        let mut atom_out = vec![0.0f32; m2];
        let mut buf = vec![0.0f32; 2 * mf * mf];

        for pp in 0..self.p {
            let dy_pp = &dy[pp * m2..(pp + 1) * m2];
            for qq in 0..self.q {
                let x_qq = &x[qq * m1..(qq + 1) * m1];
                let c1 = self.coeff1(coeffs, pp, qq);
                let base_c = (pp * self.q + qq) * self.k;

                for kk in 0..self.k {
                    let a = c1[kk];
                    let fbase = kk * 2 * mf * mf;
                    let ak = &dict.dict1[fbase..fbase + mf * mf];
                    let bk = &dict.dict1[fbase + mf * mf..fbase + 2 * mf * mf];

                    // Coefficient gradient: d_coeff[k] = dy · kron(A_k, B_k) @ x_qq
                    kron_apply(mf, ak, bk, x_qq, &mut atom_out, &mut buf);
                    d_coeffs[base_c + kk] = gemm::dot(dy_pp, &atom_out);

                    // Input gradient: d_x_qq += coeff[k] * kron(A_k^T, B_k^T) @ dy_pp
                    kron_transpose_apply(mf, ak, bk, dy_pp, &mut atom_out, &mut buf);
                    for r in 0..m1 {
                        d_x[qq * m1 + r] += a * atom_out[r];
                    }

                    // Factor gradients (square case)
                    if a != 0.0 && m1 == mf * mf && m2 == mf * mf {
                        factor_grad8x8(mf, ak, bk, x_qq, dy_pp, &mut d_dict1, fbase, a, &mut buf);
                    }
                }
            }
        }

        BttGrads { d_coeffs, d_dict1, d_dict2: vec![0.0; dict.dict2_len()], d_x }
    }

    /// Forward computing only the selected output row-blocks.
    pub fn forward_rows(
        &self,
        dict: &BttDict,
        coeffs: &[f32],
        x: &[f32],
        active_pp: &[usize],
    ) -> Vec<f32> {
        assert_eq!(coeffs.len(), self.coeff_len());
        assert_eq!(x.len(), self.in_dim());

        let (m1, m2, mf) = (self.m1, self.m2, self.mf);
        let mut y = vec![0.0f32; self.out_dim()];
        let mut atom_out = vec![0.0f32; m2];
        let mut buf = vec![0.0f32; 2 * mf * mf];

        for &pp in active_pp {
            debug_assert!(pp < self.p);
            let mut acc = vec![0.0f32; m2];
            for qq in 0..self.q {
                let c1 = self.coeff1(coeffs, pp, qq);
                let x_qq = &x[qq * m1..(qq + 1) * m1];
                for kk in 0..self.k {
                    let a = c1[kk];
                    if a == 0.0 { continue; }
                    let fbase = kk * 2 * mf * mf;
                    let ak = &dict.dict1[fbase..fbase + mf * mf];
                    let bk = &dict.dict1[fbase + mf * mf..fbase + 2 * mf * mf];
                    kron_apply(mf, ak, bk, x_qq, &mut atom_out, &mut buf);
                    for r in 0..m2 {
                        acc[r] += a * atom_out[r];
                    }
                }
            }
            y[pp * m2..(pp + 1) * m2].copy_from_slice(&acc);
        }
        y
    }

    /// Backward for [`forward_rows`](Self::forward_rows).
    pub fn backward_rows(
        &self,
        dict: &BttDict,
        coeffs: &[f32],
        x: &[f32],
        dy: &[f32],
        active_pp: &[usize],
    ) -> BttGrads {
        assert_eq!(coeffs.len(), self.coeff_len());
        assert_eq!(x.len(), self.in_dim());
        assert_eq!(dy.len(), self.out_dim());

        let (m1, m2, mf) = (self.m1, self.m2, self.mf);
        let mut d_coeffs = vec![0.0f32; self.coeff_len()];
        let mut d_dict1 = vec![0.0f32; dict.dict1_len()];
        let mut d_x = vec![0.0f32; self.in_dim()];
        let mut atom_out = vec![0.0f32; m2];
        let mut buf = vec![0.0f32; 2 * mf * mf];

        for &pp in active_pp {
            debug_assert!(pp < self.p);
            let dy_pp = &dy[pp * m2..(pp + 1) * m2];
            for qq in 0..self.q {
                let x_qq = &x[qq * m1..(qq + 1) * m1];
                let c1 = self.coeff1(coeffs, pp, qq);
                let base_c = (pp * self.q + qq) * self.k;

                for kk in 0..self.k {
                    let a = c1[kk];
                    let fbase = kk * 2 * mf * mf;
                    let ak = &dict.dict1[fbase..fbase + mf * mf];
                    let bk = &dict.dict1[fbase + mf * mf..fbase + 2 * mf * mf];

                    kron_apply(mf, ak, bk, x_qq, &mut atom_out, &mut buf);
                    d_coeffs[base_c + kk] = gemm::dot(dy_pp, &atom_out);

                    // Input gradient
                    kron_transpose_apply(mf, ak, bk, dy_pp, &mut atom_out, &mut buf);
                    for r in 0..m1 {
                        d_x[qq * m1 + r] += a * atom_out[r];
                    }

                    // Factor gradients (square case)
                    if a != 0.0 && m1 == mf * mf && m2 == mf * mf {
                        factor_grad8x8(mf, ak, bk, x_qq, dy_pp, &mut d_dict1, fbase, a, &mut buf);
                    }
                }
            }
        }

        BttGrads { d_coeffs, d_dict1, d_dict2: vec![0.0; dict.dict2_len()], d_x }
    }

    /// Forward summing only over the selected input col-blocks.
    pub fn forward_cols(
        &self,
        dict: &BttDict,
        coeffs: &[f32],
        x: &[f32],
        active_q: &[usize],
    ) -> Vec<f32> {
        assert_eq!(coeffs.len(), self.coeff_len());
        assert_eq!(x.len(), self.in_dim());

        let (m1, m2, mf) = (self.m1, self.m2, self.mf);
        let mut y = vec![0.0f32; self.out_dim()];
        let mut atom_out = vec![0.0f32; m2];
        let mut buf = vec![0.0f32; 2 * mf * mf];

        for pp in 0..self.p {
            let mut acc = vec![0.0f32; m2];
            for &qq in active_q {
                debug_assert!(qq < self.q);
                let c1 = self.coeff1(coeffs, pp, qq);
                let x_qq = &x[qq * m1..(qq + 1) * m1];
                for kk in 0..self.k {
                    let a = c1[kk];
                    if a == 0.0 { continue; }
                    let fbase = kk * 2 * mf * mf;
                    let ak = &dict.dict1[fbase..fbase + mf * mf];
                    let bk = &dict.dict1[fbase + mf * mf..fbase + 2 * mf * mf];
                    kron_apply(mf, ak, bk, x_qq, &mut atom_out, &mut buf);
                    for r in 0..m2 {
                        acc[r] += a * atom_out[r];
                    }
                }
            }
            y[pp * m2..(pp + 1) * m2].copy_from_slice(&acc);
        }
        y
    }

    /// Backward for [`forward_cols`](Self::forward_cols).
    pub fn backward_cols(
        &self,
        dict: &BttDict,
        coeffs: &[f32],
        x: &[f32],
        dy: &[f32],
        active_q: &[usize],
    ) -> BttGrads {
        assert_eq!(coeffs.len(), self.coeff_len());
        assert_eq!(x.len(), self.in_dim());
        assert_eq!(dy.len(), self.out_dim());

        let (m1, m2, mf) = (self.m1, self.m2, self.mf);
        let mut d_coeffs = vec![0.0f32; self.coeff_len()];
        let mut d_dict1 = vec![0.0f32; dict.dict1_len()];
        let mut d_x = vec![0.0f32; self.in_dim()];
        let mut atom_out = vec![0.0f32; m2];
        let mut buf = vec![0.0f32; 2 * mf * mf];

        for pp in 0..self.p {
            let dy_pp = &dy[pp * m2..(pp + 1) * m2];
            for &qq in active_q {
                debug_assert!(qq < self.q);
                let x_qq = &x[qq * m1..(qq + 1) * m1];
                let c1 = self.coeff1(coeffs, pp, qq);
                let base_c = (pp * self.q + qq) * self.k;

                for kk in 0..self.k {
                    let a = c1[kk];
                    let fbase = kk * 2 * mf * mf;
                    let ak = &dict.dict1[fbase..fbase + mf * mf];
                    let bk = &dict.dict1[fbase + mf * mf..fbase + 2 * mf * mf];

                    kron_apply(mf, ak, bk, x_qq, &mut atom_out, &mut buf);
                    d_coeffs[base_c + kk] = gemm::dot(dy_pp, &atom_out);

                    kron_transpose_apply(mf, ak, bk, dy_pp, &mut atom_out, &mut buf);
                    for r in 0..m1 {
                        d_x[qq * m1 + r] += a * atom_out[r];
                    }

                    if a != 0.0 && m1 == mf * mf && m2 == mf * mf {
                        factor_grad8x8(mf, ak, bk, x_qq, dy_pp, &mut d_dict1, fbase, a, &mut buf);
                    }
                }
            }
        }

        BttGrads { d_coeffs, d_dict1, d_dict2: vec![0.0; dict.dict2_len()], d_x }
    }

    /// Compact variant: `x` has length `active_q.len() * m1`.
    pub fn forward_cols_compact(
        &self,
        dict: &BttDict,
        coeffs: &[f32],
        x: &[f32],
        active_q: &[usize],
    ) -> Vec<f32> {
        assert_eq!(coeffs.len(), self.coeff_len());
        assert_eq!(x.len(), active_q.len() * self.m1);

        let (m1, m2, mf) = (self.m1, self.m2, self.mf);
        let mut y = vec![0.0f32; self.out_dim()];
        let mut atom_out = vec![0.0f32; m2];
        let mut buf = vec![0.0f32; 2 * mf * mf];

        for pp in 0..self.p {
            let mut acc = vec![0.0f32; m2];
            for (si, &qq) in active_q.iter().enumerate() {
                let c1 = self.coeff1(coeffs, pp, qq);
                let xb = &x[si * m1..(si + 1) * m1];
                for kk in 0..self.k {
                    let a = c1[kk];
                    if a == 0.0 { continue; }
                    let fbase = kk * 2 * mf * mf;
                    let ak = &dict.dict1[fbase..fbase + mf * mf];
                    let bk = &dict.dict1[fbase + mf * mf..fbase + 2 * mf * mf];
                    kron_apply(mf, ak, bk, xb, &mut atom_out, &mut buf);
                    for r in 0..m2 {
                        acc[r] += a * atom_out[r];
                    }
                }
            }
            y[pp * m2..(pp + 1) * m2].copy_from_slice(&acc);
        }
        y
    }

    /// Compact variant of [`backward_cols`](Self::backward_cols).
    pub fn backward_cols_compact(
        &self,
        dict: &BttDict,
        coeffs: &[f32],
        x: &[f32],
        dy: &[f32],
        active_q: &[usize],
    ) -> BttGrads {
        assert_eq!(coeffs.len(), self.coeff_len());
        assert_eq!(x.len(), active_q.len() * self.m1);
        assert_eq!(dy.len(), self.out_dim());

        let (m1, m2, mf) = (self.m1, self.m2, self.mf);
        let mut d_coeffs = vec![0.0f32; self.coeff_len()];
        let mut d_dict1 = vec![0.0f32; dict.dict1_len()];
        let mut d_x = vec![0.0f32; active_q.len() * m1];
        let mut atom_out = vec![0.0f32; m2];
        let mut buf = vec![0.0f32; 2 * mf * mf];

        for pp in 0..self.p {
            let dy_pp = &dy[pp * m2..(pp + 1) * m2];
            for (si, &qq) in active_q.iter().enumerate() {
                let xb = &x[si * m1..(si + 1) * m1];
                let c1 = self.coeff1(coeffs, pp, qq);
                let base_c = (pp * self.q + qq) * self.k;

                for kk in 0..self.k {
                    let a = c1[kk];
                    let fbase = kk * 2 * mf * mf;
                    let ak = &dict.dict1[fbase..fbase + mf * mf];
                    let bk = &dict.dict1[fbase + mf * mf..fbase + 2 * mf * mf];

                    kron_apply(mf, ak, bk, xb, &mut atom_out, &mut buf);
                    d_coeffs[base_c + kk] = gemm::dot(dy_pp, &atom_out);

                    kron_transpose_apply(mf, ak, bk, dy_pp, &mut atom_out, &mut buf);
                    for r in 0..m1 {
                        d_x[si * m1 + r] += a * atom_out[r];
                    }

                    if a != 0.0 && m1 == mf * mf && m2 == mf * mf {
                        factor_grad8x8(mf, ak, bk, xb, dy_pp, &mut d_dict1, fbase, a, &mut buf);
                    }
                }
            }
        }

        BttGrads { d_coeffs, d_dict1, d_dict2: vec![0.0; dict.dict2_len()], d_x }
    }

    /// Fused forward for two projections sharing input `x` and dictionary.
    pub fn forward_rows_pair(
        &self,
        dict: &BttDict,
        coeffs_a: &[f32],
        coeffs_b: &[f32],
        x: &[f32],
        active_pp: &[usize],
    ) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(coeffs_a.len(), self.coeff_len());
        assert_eq!(coeffs_b.len(), self.coeff_len());
        assert_eq!(x.len(), self.in_dim());

        let (m1, m2, mf) = (self.m1, self.m2, self.mf);
        let mut ya = vec![0.0f32; self.out_dim()];
        let mut yb = vec![0.0f32; self.out_dim()];
        let mut atom_a = vec![0.0f32; m2];
        let mut atom_b = vec![0.0f32; m2];
        let mut buf = vec![0.0f32; 2 * mf * mf];

        for &pp in active_pp {
            debug_assert!(pp < self.p);
            let mut acc_a = vec![0.0f32; m2];
            let mut acc_b = vec![0.0f32; m2];
            for qq in 0..self.q {
                let ca = self.coeff1(coeffs_a, pp, qq);
                let cb = self.coeff1(coeffs_b, pp, qq);
                let x_qq = &x[qq * m1..(qq + 1) * m1];
                for kk in 0..self.k {
                    let aa = ca[kk];
                    let ab = cb[kk];
                    let fbase = kk * 2 * mf * mf;
                    let ak = &dict.dict1[fbase..fbase + mf * mf];
                    let bk = &dict.dict1[fbase + mf * mf..fbase + 2 * mf * mf];
                    kron_apply(mf, ak, bk, x_qq, &mut atom_a, &mut buf);
                    for r in 0..m2 {
                        acc_a[r] += aa * atom_a[r];
                        acc_b[r] += ab * atom_a[r];
                    }
                }
            }
            ya[pp * m2..(pp + 1) * m2].copy_from_slice(&acc_a);
            yb[pp * m2..(pp + 1) * m2].copy_from_slice(&acc_b);
        }
        (ya, yb)
    }

    /// Batched fused forward for two projections sharing input `x` and dictionary.
    /// Processes `t_len` tokens at once, reusing scratch buffers.
    /// `x` is `[t_len, in_dim]`, `active_pp_per_token` is per-token active rows.
    /// `ya` and `yb` are pre-allocated `[t_len, out_dim]` (zeroed by caller).
    pub fn forward_rows_pair_batch(
        &self,
        dict: &BttDict,
        coeffs_a: &[f32],
        coeffs_b: &[f32],
        x: &[f32],
        active_pp_per_token: &[Vec<usize>],
        ya: &mut [f32],
        yb: &mut [f32],
        t_len: usize,
    ) {
        assert_eq!(coeffs_a.len(), self.coeff_len());
        assert_eq!(coeffs_b.len(), self.coeff_len());
        assert_eq!(x.len(), t_len * self.in_dim());
        assert_eq!(ya.len(), t_len * self.out_dim());
        assert_eq!(yb.len(), t_len * self.out_dim());

        let (m1, m2, mf) = (self.m1, self.m2, self.mf);
        ya.fill(0.0);
        yb.fill(0.0);
        let mut atom_a = vec![0.0f32; m2];
        let mut buf = vec![0.0f32; 2 * mf * mf];

        // Pre-compute which tokens are active for each pp.
        let mut tokens_for_pp: Vec<Vec<usize>> = vec![Vec::new(); self.p];
        for t in 0..t_len {
            for &pp in &active_pp_per_token[t] {
                tokens_for_pp[pp].push(t);
            }
        }

        for pp in 0..self.p {
            let toks = &tokens_for_pp[pp];
            if toks.is_empty() { continue; }
            for qq in 0..self.q {
                let ca = self.coeff1(coeffs_a, pp, qq);
                let cb = self.coeff1(coeffs_b, pp, qq);
                for kk in 0..self.k {
                    let aa = ca[kk];
                    let ab = cb[kk];
                    let fbase = kk * 2 * mf * mf;
                    let ak = &dict.dict1[fbase..fbase + mf * mf];
                    let bk = &dict.dict1[fbase + mf * mf..fbase + 2 * mf * mf];

                    for &t in toks {
                        let x_qq = &x[t * self.in_dim() + qq * m1..t * self.in_dim() + (qq + 1) * m1];
                        kron_apply(mf, ak, bk, x_qq, &mut atom_a, &mut buf);
                        let ya_tp = &mut ya[t * self.out_dim() + pp * m2..t * self.out_dim() + (pp + 1) * m2];
                        let yb_tp = &mut yb[t * self.out_dim() + pp * m2..t * self.out_dim() + (pp + 1) * m2];
                        for r in 0..m2 {
                            ya_tp[r] += aa * atom_a[r];
                            yb_tp[r] += ab * atom_a[r];
                        }
                    }
                }
            }
        }
    }

    /// Batched forward_cols: process `t_len` tokens, each with its own `active_q`.
    /// `x` is `[t_len, in_dim]`, `y` is `[t_len, out_dim]` (zeroed by caller).
    pub fn forward_cols_batch(
        &self,
        dict: &BttDict,
        coeffs: &[f32],
        x: &[f32],
        active_q_per_token: &[Vec<usize>],
        y: &mut [f32],
        t_len: usize,
    ) {
        assert_eq!(coeffs.len(), self.coeff_len());
        assert_eq!(x.len(), t_len * self.in_dim());
        assert_eq!(y.len(), t_len * self.out_dim());

        let (m1, m2, mf) = (self.m1, self.m2, self.mf);
        y.fill(0.0);
        let mut atom_out = vec![0.0f32; m2];
        let mut buf = vec![0.0f32; 2 * mf * mf];

        // Pre-compute which tokens are active for each qq.
        let mut tokens_for_qq: Vec<Vec<usize>> = vec![Vec::new(); self.q];
        for t in 0..t_len {
            for &qq in &active_q_per_token[t] {
                tokens_for_qq[qq].push(t);
            }
        }

        for pp in 0..self.p {
            for qq in 0..self.q {
                let toks = &tokens_for_qq[qq];
                if toks.is_empty() { continue; }
                let c1 = self.coeff1(coeffs, pp, qq);
                for kk in 0..self.k {
                    let a = c1[kk];
                    if a == 0.0 { continue; }
                    let fbase = kk * 2 * mf * mf;
                    let ak = &dict.dict1[fbase..fbase + mf * mf];
                    let bk = &dict.dict1[fbase + mf * mf..fbase + 2 * mf * mf];

                    for &t in toks {
                        let x_qq = &x[t * self.in_dim() + qq * m1..t * self.in_dim() + (qq + 1) * m1];
                        kron_apply(mf, ak, bk, x_qq, &mut atom_out, &mut buf);
                        let y_tp = &mut y[t * self.out_dim() + pp * m2..t * self.out_dim() + (pp + 1) * m2];
                        for r in 0..m2 {
                            y_tp[r] += a * atom_out[r];
                        }
                    }
                }
            }
        }
    }

    /// Fused backward for two projections sharing input `x` and dictionary.
    pub fn backward_rows_pair(
        &self,
        dict: &BttDict,
        coeffs_a: &[f32],
        coeffs_b: &[f32],
        x: &[f32],
        dy_a: &[f32],
        dy_b: &[f32],
        active_pp: &[usize],
    ) -> PairBttGrads {
        assert_eq!(coeffs_a.len(), self.coeff_len());
        assert_eq!(coeffs_b.len(), self.coeff_len());
        assert_eq!(x.len(), self.in_dim());
        assert_eq!(dy_a.len(), self.out_dim());
        assert_eq!(dy_b.len(), self.out_dim());

        let (m1, m2, mf) = (self.m1, self.m2, self.mf);
        let mut dca = vec![0.0f32; self.coeff_len()];
        let mut dcb = vec![0.0f32; self.coeff_len()];
        let mut d_dict1 = vec![0.0f32; dict.dict1_len()];
        let mut d_x = vec![0.0f32; self.in_dim()];
        let mut atom_out = vec![0.0f32; m2];
        let mut buf = vec![0.0f32; 2 * mf * mf];

        for &pp in active_pp {
            debug_assert!(pp < self.p);
            let dy_a_pp = &dy_a[pp * m2..(pp + 1) * m2];
            let dy_b_pp = &dy_b[pp * m2..(pp + 1) * m2];
            for qq in 0..self.q {
                let x_qq = &x[qq * m1..(qq + 1) * m1];
                let ca = self.coeff1(coeffs_a, pp, qq);
                let cb = self.coeff1(coeffs_b, pp, qq);
                let base_c = (pp * self.q + qq) * self.k;

                for kk in 0..self.k {
                    let aa = ca[kk];
                    let ab = cb[kk];
                    let fbase = kk * 2 * mf * mf;
                    let ak = &dict.dict1[fbase..fbase + mf * mf];
                    let bk = &dict.dict1[fbase + mf * mf..fbase + 2 * mf * mf];

                    // Coefficient gradients
                    kron_apply(mf, ak, bk, x_qq, &mut atom_out, &mut buf);
                    dca[base_c + kk] = gemm::dot(dy_a_pp, &atom_out);
                    dcb[base_c + kk] = gemm::dot(dy_b_pp, &atom_out);

                    // Input gradients (combined)
                    kron_transpose_apply(mf, ak, bk, dy_a_pp, &mut atom_out, &mut buf);
                    for r in 0..m1 {
                        d_x[qq * m1 + r] += aa * atom_out[r];
                    }
                    kron_transpose_apply(mf, ak, bk, dy_b_pp, &mut atom_out, &mut buf);
                    for r in 0..m1 {
                        d_x[qq * m1 + r] += ab * atom_out[r];
                    }

                    // Factor gradients (combined from both projections)
                    if (aa != 0.0 || ab != 0.0) && m1 == mf * mf && m2 == mf * mf {
                        for r in 0..m2 {
                            atom_out[r] = aa * dy_a_pp[r] + ab * dy_b_pp[r];
                        }
                        factor_grad8x8(mf, ak, bk, x_qq, &atom_out[..m2], &mut d_dict1, fbase, 1.0, &mut buf);
                    }
                }
            }
        }

        PairBttGrads { d_coeffs_a: dca, d_coeffs_b: dcb, d_dict1, d_dict2: vec![0.0; dict.dict2_len()], d_x }
    }

    /// Software-prefetch coefficient tiles for `forward_rows`/`backward_rows`.
    pub fn prefetch_rows(&self, coeffs: &[f32], active_pp: &[usize]) {
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let span = self.q * self.k;
            for &pp in active_pp {
                let base = pp * span;
                let mut off = base;
                while off < (base + span).min(coeffs.len()) {
                    unsafe { _mm_prefetch(coeffs.as_ptr().add(off) as *const i8, _MM_HINT_T0) };
                    off += 16;
                }
            }
        }
        let _ = (coeffs, active_pp);
    }

    /// Prefetch coefficient tiles for `forward_cols`/`backward_cols`.
    pub fn prefetch_cols(&self, coeffs: &[f32], active_q: &[usize]) {
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            for pp in 0..self.p {
                let row_base = pp * self.q * self.k;
                for &qq in active_q {
                    let base = row_base + qq * self.k;
                    let mut off = base;
                    while off < (base + self.k).min(coeffs.len()) {
                        unsafe { _mm_prefetch(coeffs.as_ptr().add(off) as *const i8, _MM_HINT_T0) };
                        off += 16;
                    }
                }
            }
        }
        let _ = (coeffs, active_q);
    }
}

/// Gradients produced by [`BttMatmul::backward`].
pub struct BttGrads {
    pub d_coeffs: Vec<f32>,
    pub d_dict1: Vec<f32>,
    pub d_dict2: Vec<f32>,
    pub d_x: Vec<f32>,
}

/// Gradients from fused backward of two projections sharing one input.
pub struct PairBttGrads {
    pub d_coeffs_a: Vec<f32>,
    pub d_coeffs_b: Vec<f32>,
    pub d_dict1: Vec<f32>,
    pub d_dict2: Vec<f32>,
    pub d_x: Vec<f32>,
}

/// Random coefficient vector for BTT.
pub fn init_btt_coeffs_random(len: usize, seed: u64, std: f32) -> Vec<f32> {
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_sym(state: &mut u64) -> f32 {
        ((splitmix64(state) >> 40) as f32 / (1u64 << 23) as f32) * 2.0 - 1.0
    }
    let mut s = seed ^ 0xC0EF_F1C1_E175_EED5;
    (0..len).map(|_| next_sym(&mut s) * std).collect()
}

/// Apply `sum_k coeff[k] * kron(circ(A_k), circ(B_k)) @ x` using 2D FFT.
///
/// Each atom k is the Kronecker product of two circulant mf×mf matrices.
/// The first row of circ(A_k) is `atoms_a[k*mf .. (k+1)*mf]`, and similarly
/// for B_k. The circulant's eigenvalues are `DFT(first_row)`.
///
/// The Kronecker product of two circulant matrices is diagonalized by the 2D DFT:
/// `kron(circ(a), circ(b)) @ x = IFFT2( DFT(a) ⊗ DFT(b) ⊙ FFT2(x) )`
///
/// where `⊗` is the outer product (rank-1 frequency pattern).
///
/// Algorithm:
/// 1. FFT2(x) — O(b log b)
/// 2. Accumulate sum_k coeff[k] * outer(DFT(a_k), DFT(b_k)) — O(K × b)
/// 3. Pointwise multiply — O(b)
/// 4. IFFT2 — O(b log b)
///
/// Total: O(K × b + b log b) per block pair, vs O(K × b^{3/2}) for dense kron.
///
/// `mf_dft` must be precomputed DFTs of atoms_a, `nf_dft` of atoms_b, both
/// `[K × mf]` complex. `buf_re`/`buf_im` are scratch `[mf × mf]` each.
#[inline]
pub fn circulant_kron_apply(
    mf: usize,
    k: usize,
    atoms_a_dft: &[f32],  // [K × 2*mf] interleaved complex: [re0,im0, re1,im1, ...]
    atoms_b_dft: &[f32],
    coeffs: &[f32],       // [K]
    x: &[f32],            // [mf*mf]
    y: &mut [f32],        // [mf*mf]
    freq_acc_re: &mut [f32],  // [mf*mf]
    freq_acc_im: &mut [f32],  // [mf*mf]
    fft_x_re: &mut [f32],     // [mf*mf]
    fft_x_im: &mut [f32],     // [mf*mf]
) {
    let sz = mf * mf;
    debug_assert!(freq_acc_re.len() >= sz);
    debug_assert!(freq_acc_im.len() >= sz);
    debug_assert!(fft_x_re.len() >= sz);
    debug_assert!(fft_x_im.len() >= sz);

    // Step 1: 2D FFT of input x (real-valued)
    fft2d_real(mf, x, fft_x_re, fft_x_im);

    // Step 2: Accumulate frequency response: sum_k coeff[k] * outer(dft_a_k, dft_b_k)
    freq_acc_re.fill(0.0);
    freq_acc_im.fill(0.0);

    for ki in 0..k {
        let c = coeffs[ki];
        if c == 0.0 { continue; }
        let a_base = ki * 2 * mf;
        let b_base = ki * 2 * mf;

        for i in 0..mf {
            let a_re = atoms_a_dft[a_base + 2 * i];
            let a_im = atoms_a_dft[a_base + 2 * i + 1];
            for j in 0..mf {
                let b_re = atoms_b_dft[b_base + 2 * j];
                let b_im = atoms_b_dft[b_base + 2 * j + 1];
                // outer product: (a_re + i*a_im)(b_re + i*b_im)
                let prod_re = a_re * b_re - a_im * b_im;
                let prod_im = a_re * b_im + a_im * b_re;
                let idx = i * mf + j;
                freq_acc_re[idx] += c * prod_re;
                freq_acc_im[idx] += c * prod_im;
            }
        }
    }

    // Step 3: Pointwise multiply freq_acc ⊙ fft_x
    for i in 0..sz {
        let f_re = freq_acc_re[i];
        let f_im = freq_acc_im[i];
        let x_re = fft_x_re[i];
        let x_im = fft_x_im[i];
        fft_x_re[i] = f_re * x_re - f_im * x_im;
        fft_x_im[i] = f_re * x_im + f_im * x_re;
    }

    // Step 4: 2D IFFT to get output
    ifft2d(mf, fft_x_re, fft_x_im, y);
}

/// 2D FFT of a real-valued mf×mf matrix stored row-major.
fn fft2d_real(mf: usize, x: &[f32], out_re: &mut [f32], out_im: &mut [f32]) {
    // Copy input to output (real → complex)
    for i in 0..mf * mf {
        out_re[i] = x[i];
        out_im[i] = 0.0;
    }
    // FFT each row
    for i in 0..mf {
        fft1d(mf, &mut out_re[i * mf..(i + 1) * mf], &mut out_im[i * mf..(i + 1) * mf]);
    }
    // FFT each column (transpose → FFT → transpose for cache friendliness)
    // For mf=8, just do it directly
    let mut col_re = vec![0.0f32; mf];
    let mut col_im = vec![0.0f32; mf];
    for j in 0..mf {
        for i in 0..mf {
            col_re[i] = out_re[i * mf + j];
            col_im[i] = out_im[i * mf + j];
        }
        fft1d(mf, &mut col_re, &mut col_im);
        for i in 0..mf {
            out_re[i * mf + j] = col_re[i];
            out_im[i * mf + j] = col_im[i];
        }
    }
}

/// 2D IFFT producing a real-valued mf×mf matrix.
fn ifft2d(mf: usize, freq_re: &mut [f32], freq_im: &mut [f32], out: &mut [f32]) {
    // IFFT each column
    let mut col_re = vec![0.0f32; mf];
    let mut col_im = vec![0.0f32; mf];
    for j in 0..mf {
        for i in 0..mf {
            col_re[i] = freq_re[i * mf + j];
            col_im[i] = freq_im[i * mf + j];
        }
        ifft1d(mf, &mut col_re, &mut col_im);
        for i in 0..mf {
            freq_re[i * mf + j] = col_re[i];
            freq_im[i * mf + j] = col_im[i];
        }
    }
    // IFFT each row
    for i in 0..mf {
        ifft1d(mf, &mut freq_re[i * mf..(i + 1) * mf], &mut freq_im[i * mf..(i + 1) * mf]);
    }
    // Extract real part (output should be real-valued)
    for i in 0..mf * mf {
        out[i] = freq_re[i];
    }
}

/// Radix-2 DIT FFT for power-of-2 lengths. mf must be a power of 2.
/// In-place on separate real/imag arrays.
fn fft1d(mf: usize, re: &mut [f32], im: &mut [f32]) {
    debug_assert_eq!(re.len(), mf);
    debug_assert_eq!(im.len(), mf);
    debug_assert!(mf.is_power_of_two(), "fft1d requires power-of-2 length");

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 0..mf {
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
        let mut m = mf >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Cooley-Tukey butterfly
    let mut len = 2;
    while len <= mf {
        let half = len / 2;
        let angle_step = -std::f32::consts::TAU / len as f32;
        for start in (0..mf).step_by(len) {
            for k in 0..half {
                let angle = angle_step * k as f32;
                let wr = angle.cos();
                let wi = angle.sin();
                let t_re = wr * re[start + k + half] - wi * im[start + k + half];
                let t_im = wr * im[start + k + half] + wi * re[start + k + half];
                let u_re = re[start + k];
                let u_im = im[start + k];
                re[start + k] = u_re + t_re;
                im[start + k] = u_im + t_im;
                re[start + k + half] = u_re - t_re;
                im[start + k + half] = u_im - t_im;
            }
        }
        len <<= 1;
    }
}

/// In-place inverse DFT, normalized by 1/mf.
fn ifft1d(mf: usize, re: &mut [f32], im: &mut [f32]) {
    // Conjugate → FFT → conjugate → scale
    for v in im.iter_mut() {
        *v = -*v;
    }
    fft1d(mf, re, im);
    for v in im.iter_mut() {
        *v = -*v;
    }
    let scale = 1.0 / mf as f32;
    for r in re.iter_mut() {
        *r *= scale;
    }
    for i in im.iter_mut() {
        *i *= scale;
    }
}

/// Precompute DFT of circulant first-rows for all atoms.
/// `atoms_a` is `[K × mf]` (first rows of A_k), `atoms_b` similarly.
/// Output is `[K × 2*mf]` interleaved complex.
pub fn precompute_circulant_dfts(mf: usize, k: usize, atoms_a: &[f32], atoms_b: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut dft_a = vec![0.0f32; k * 2 * mf];
    let mut dft_b = vec![0.0f32; k * 2 * mf];
    let mut re = vec![0.0f32; mf];
    let mut im = vec![0.0f32; mf];

    for ki in 0..k {
        // DFT of A_k's first row
        re.copy_from_slice(&atoms_a[ki * mf..(ki + 1) * mf]);
        im.fill(0.0);
        fft1d(mf, &mut re, &mut im);
        for i in 0..mf {
            dft_a[ki * 2 * mf + 2 * i] = re[i];
            dft_a[ki * 2 * mf + 2 * i + 1] = im[i];
        }
        // DFT of B_k's first row
        re.copy_from_slice(&atoms_b[ki * mf..(ki + 1) * mf]);
        im.fill(0.0);
        fft1d(mf, &mut re, &mut im);
        for i in 0..mf {
            dft_b[ki * 2 * mf + 2 * i] = re[i];
            dft_b[ki * 2 * mf + 2 * i + 1] = im[i];
        }
    }
    (dft_a, dft_b)
}

/// Full-model-style circulant Kronecker forward: accumulate over all P×Q block pairs.
pub fn circulant_kron_forward(
    mf: usize, p: usize, q: usize, k: usize,
    atoms_a_dft: &[f32], atoms_b_dft: &[f32],
    coeffs: &[f32],  // [P × Q × K]
    x: &[f32],       // [Q × mf × mf]
    y: &mut [f32],   // [P × mf × mf]
    freq_acc_re: &mut [f32],
    freq_acc_im: &mut [f32],
    fft_x_re: &mut [f32],
    fft_x_im: &mut [f32],
) {
    let sz = mf * mf;
    let m1 = mf * mf; // block input dim
    let m2 = mf * mf; // block output dim

    for pp in 0..p {
        // Zero accumulator for this output block
        freq_acc_re.fill(0.0);
        freq_acc_im.fill(0.0);

        for qq in 0..q {
            let c_base = (pp * q + qq) * k;
            let x_block = &x[qq * m1..(qq + 1) * m1];

            // FFT2 of this input block
            fft2d_real(mf, x_block, fft_x_re, fft_x_im);

            // Accumulate: sum_k coeff[k] * outer(dft_a_k, dft_b_k)
            for ki in 0..k {
                let c = coeffs[c_base + ki];
                if c == 0.0 { continue; }
                let a_base = ki * 2 * mf;
                let b_base = ki * 2 * mf;
                for i in 0..mf {
                    let a_re = atoms_a_dft[a_base + 2 * i];
                    let a_im = atoms_a_dft[a_base + 2 * i + 1];
                    for j in 0..mf {
                        let b_re = atoms_b_dft[b_base + 2 * j];
                        let b_im = atoms_b_dft[b_base + 2 * j + 1];
                        let prod_re = a_re * b_re - a_im * b_im;
                        let prod_im = a_re * b_im + a_im * b_re;
                        let idx = i * mf + j;
                        freq_acc_re[idx] += c * prod_re;
                        freq_acc_im[idx] += c * prod_im;
                    }
                }
            }
        }

        // Now freq_acc has the accumulated frequency response.
        // We need to apply it to ALL input blocks and sum the results.
        // But wait — the freq_acc is the sum over qq of the frequency responses.
        // The correct formula is: for each qq, y_pp += IFFT2(freq_resp_qq ⊙ FFT2(x_qq))
        // We can't just sum freq_acc and apply once — that's wrong.
        // We need to accumulate in the time domain.

        // Reset and redo per-qq
        y[pp * m2..(pp + 1) * m2].fill(0.0);
        let mut y_block = vec![0.0f32; sz];

        for qq in 0..q {
            let c_base = (pp * q + qq) * k;
            let x_block = &x[qq * m1..(qq + 1) * m1];

            // FFT2 of input block
            fft2d_real(mf, x_block, fft_x_re, fft_x_im);

            // Build frequency response for this (pp,qq)
            freq_acc_re.fill(0.0);
            freq_acc_im.fill(0.0);
            for ki in 0..k {
                let c = coeffs[c_base + ki];
                if c == 0.0 { continue; }
                let a_base = ki * 2 * mf;
                let b_base = ki * 2 * mf;
                for i in 0..mf {
                    let a_re = atoms_a_dft[a_base + 2 * i];
                    let a_im = atoms_a_dft[a_base + 2 * i + 1];
                    for j in 0..mf {
                        let b_re = atoms_b_dft[b_base + 2 * j];
                        let b_im = atoms_b_dft[b_base + 2 * j + 1];
                        let prod_re = a_re * b_re - a_im * b_im;
                        let prod_im = a_re * b_im + a_im * b_re;
                        let idx = i * mf + j;
                        freq_acc_re[idx] += c * prod_re;
                        freq_acc_im[idx] += c * prod_im;
                    }
                }
            }

            // Pointwise multiply
            for i in 0..sz {
                let f_re = freq_acc_re[i];
                let f_im = freq_acc_im[i];
                let x_re = fft_x_re[i];
                let x_im = fft_x_im[i];
                fft_x_re[i] = f_re * x_re - f_im * x_im;
                fft_x_im[i] = f_re * x_im + f_im * x_re;
            }

            // IFFT2
            ifft2d(mf, fft_x_re, fft_x_im, &mut y_block);

            // Accumulate into output
            for i in 0..sz {
                y[pp * m2 + i] += y_block[i];
            }
        }
    }
}

/// Random BTT dictionary with Kronecker-structured atoms.
/// Each atom k is kron(A_k, B_k) where A_k and B_k are mf×mf random matrices.
pub fn init_btt_dict_random(n_shared: usize, m1: usize, m2: usize, seed: u64, std: f32) -> BttDict {
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_sym(state: &mut u64) -> f32 {
        ((splitmix64(state) >> 40) as f32 / (1u64 << 23) as f32) * 2.0 - 1.0
    }
    let mut s = seed ^ 0xD1C7_DBA5_512D_1C75;

    let mf = (m1 as f64).sqrt() as usize;
    assert!(mf * mf == m1, "m1 ({m1}) must be a perfect square for Kronecker atoms");
    let mf2 = (m2 as f64).sqrt() as usize;
    assert!(mf2 * mf2 == m2, "m2 ({m2}) must be a perfect square for Kronecker atoms");
    assert_eq!(mf, mf2, "m1 and m2 must have the same sqrt for Kronecker atoms");

    // Store factors as [n_shared × 2 × mf × mf]: for each atom, A_k then B_k.
    let mut factors = vec![0.0f32; n_shared * 2 * mf * mf];
    for k in 0..n_shared {
        let base = k * 2 * mf * mf;
        // A_k
        for v in factors[base..base + mf * mf].iter_mut() {
            *v = next_sym(&mut s) * std;
        }
        // B_k
        for v in factors[base + mf * mf..base + 2 * mf * mf].iter_mut() {
            *v = next_sym(&mut s) * std;
        }
    }

    BttDict { dict1: factors, dict2: vec![0.0; n_shared * 2 * mf * mf], n_shared, m1, m2, mf }
}
