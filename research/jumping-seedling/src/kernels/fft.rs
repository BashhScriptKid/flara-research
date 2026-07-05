//! Circular-basis matmul kernel (scheme B: shared frequency dictionary).
//!
//! A weight matrix `W` [out, in] used as `y = W x` is tiled into `b×b`
//! circulant blocks. A circulant is diagonalized by the DFT, so a block's
//! action is `IFFT(λ ⊙ FFT(x_block))` where `λ ∈ ℂ^b` are its eigenvalues.
//!
//! The compression: every block's eigenvalues are a real linear combination of
//! a *shared* complex dictionary `G ∈ ℂ^{K×b}`:
//!
//! ```text
//! λ_{p,q} = Σ_k α_{p,q}[k] · G_k
//! ```
//!
//! `G` is shared across all blocks of all matrices and all layers — it is the
//! cache-resident "basis". Only the real coefficients `α_{p,q} ∈ ℝ^K` vary per
//! block, so per-matrix storage is `P·Q·K` reals instead of `out·in`.
//!
//! This Stage-1 module implements the forward path and its correctness tests.
//! Backward (VJP w.r.t. `α` and `G`) and the load-time compress path that fits
//! `(G, α)` to a dense target under the loss-tolerance dial live in later work.
//!
//! Implementation note: this is a clear, correct reference using `rustfft` and
//! scalar complex arithmetic. AVX2 vectorization of the `λ`-build and the
//! pointwise spectral products is a later optimization; the block structure is
//! deliberately the unit that will map onto cache tiles.

use crate::types::{AlignedVec, CompressedWeight};
use rustfft::num_complex::Complex32;
use rustfft::{Fft as RustFft, FftPlanner};
use std::sync::Arc;

/// Cached forward/inverse FFT planners for a fixed transform length.
pub struct Fft {
    n: usize,
    fwd: Arc<dyn RustFft<f32>>,
    inv: Arc<dyn RustFft<f32>>,
}

thread_local! {
    /// Count of `fft` + `ifft` calls on this thread (a profiling aid; the increment
    /// is a single thread-local store, negligible on the hot path).
    static FFT_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Reset the thread-local FFT call counter.
pub fn fft_calls_reset() {
    FFT_CALLS.with(|c| c.set(0));
}

/// Read the thread-local FFT call counter.
pub fn fft_calls() -> u64 {
    FFT_CALLS.with(|c| c.get())
}

impl Fft {
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n);
        Self { n, fwd, inv }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// In-place forward DFT (unnormalized, matching the convolution theorem).
    #[inline]
    pub fn fft(&self, buf: &mut [Complex32]) {
        debug_assert_eq!(buf.len(), self.n);
        FFT_CALLS.with(|c| c.set(c.get() + 1));
        self.fwd.process(buf);
    }

    /// In-place inverse DFT, normalized by `1/n` so `ifft(fft(x)) == x`.
    #[inline]
    pub fn ifft(&self, buf: &mut [Complex32]) {
        debug_assert_eq!(buf.len(), self.n);
        FFT_CALLS.with(|c| c.set(c.get() + 1));
        self.inv.process(buf);
        let scale = 1.0 / self.n as f32;
        for v in buf.iter_mut() {
            *v *= scale;
        }
    }
}

/// Circulant matrix-vector product via FFT: `y = circ(c) · x`.
///
/// `circ(c)` is the circulant matrix with first column `c`, i.e.
/// `y_i = Σ_j c[(i - j) mod n] · x[j]` (circular convolution of `c` and `x`).
/// Computed as `IFFT(FFT(c) ⊙ FFT(x))`. Reference path used to validate the
/// block kernel and as the spec for the eigenvalue convention.
pub fn circulant_matvec(fft: &Fft, c: &[f32], x: &[f32]) -> Vec<f32> {
    let n = fft.len();
    assert_eq!(c.len(), n, "first column length must match transform length");
    assert_eq!(x.len(), n, "input length must match transform length");

    let mut cf: Vec<Complex32> = c.iter().map(|&r| Complex32::new(r, 0.0)).collect();
    let mut xf: Vec<Complex32> = x.iter().map(|&r| Complex32::new(r, 0.0)).collect();
    fft.fft(&mut cf);
    fft.fft(&mut xf);
    for i in 0..n {
        cf[i] *= xf[i];
    }
    fft.ifft(&mut cf);
    cf.iter().map(|z| z.re).collect()
}

/// Shared-dictionary block-circulant matmul (scheme B forward).
///
/// Holds the geometry of one logical weight matrix `W` [out, in] expressed over
/// a shared dictionary. The dictionary `G` and coefficients `α` are passed in at
/// call time rather than owned, because `G` is shared across many `BasisMatmul`
/// instances (one dictionary, many matrices).
pub struct BasisMatmul {
    /// Block size `b` (transform length of each circulant block).
    pub b: usize,
    /// Dictionary atom count `K`.
    pub k: usize,
    /// Row-blocks `P = out / b`.
    pub p: usize,
    /// Col-blocks `Q = in / b`.
    pub q: usize,
    fft: Fft,
}

impl BasisMatmul {
    pub fn new(out: usize, in_: usize, b: usize, k: usize) -> Self {
        assert!(b > 0 && k > 0, "block size and dictionary size must be positive");
        assert_eq!(out % b, 0, "out ({out}) must be divisible by block size {b}");
        assert_eq!(in_ % b, 0, "in ({in_}) must be divisible by block size {b}");
        Self { b, k, p: out / b, q: in_ / b, fft: Fft::new(b) }
    }

    #[inline]
    pub fn out_dim(&self) -> usize {
        self.p * self.b
    }

    #[inline]
    pub fn in_dim(&self) -> usize {
        self.q * self.b
    }

    /// Number of real coefficients for one matrix over this geometry.
    #[inline]
    pub fn coeff_len(&self) -> usize {
        self.p * self.q * self.k
    }

    /// Expected dictionary length (`K · b` complex atoms, row-major `K×b`).
    #[inline]
    pub fn dict_len(&self) -> usize {
        self.k * self.b
    }

    /// Reconstruct one block's eigenvalues `λ = Σ_k α[k] · G_k` into `out`.
    ///
    /// `dict` is `K×b` row-major; `coeffs` is the length-`K` slice for this block.
    #[inline]
    fn block_eigs(&self, dict: &[Complex32], coeffs: &[f32], out: &mut [Complex32]) {
        // This is an axpy (independent stores, no reduction), so with AVX2 codegen
        // enabled the compiler already auto-vectorizes it — a hand-written intrinsic
        // version measured no improvement, so it is deliberately left scalar.
        for v in out.iter_mut() {
            *v = Complex32::new(0.0, 0.0);
        }
        for kk in 0..self.k {
            let a = coeffs[kk];
            let atom = &dict[kk * self.b..(kk + 1) * self.b];
            for f in 0..self.b {
                out[f] += atom[f] * a;
            }
        }
    }

    /// Coefficient + dictionary gradients for one `(p,q)` block.
    /// `d_coeffs[base+kk] = Σ_f Re(pbuf[f]·dict_kk[f])` is a complex-dot reduction
    /// (won't auto-vectorize) done via `gemm::dot` over the `[re,-im]` sign-flip of
    /// `pbuf`; `d_dict[kk] += pbuf·coeffs[base+kk]` is a separable axpy.
    #[inline]
    fn accum_block_grads(
        &self,
        pbuf: &[Complex32],
        dict: &[Complex32],
        coeffs: &[f32],
        base: usize,
        d_coeffs: &mut [f32],
        d_dict: &mut [Complex32],
    ) {
        let b = self.b;
        let mut psign = vec![0.0f32; 2 * b];
        for f in 0..b {
            psign[2 * f] = pbuf[f].re;
            psign[2 * f + 1] = -pbuf[f].im;
        }
        for kk in 0..self.k {
            let atom = &dict[kk * b..(kk + 1) * b];
            // SAFETY: Complex32 is repr(C){re,im}; a [b] complex slice aliases a
            // [2b] f32 slice of the same lifetime.
            let atom_f = unsafe { std::slice::from_raw_parts(atom.as_ptr() as *const f32, 2 * b) };
            d_coeffs[base + kk] = crate::kernels::gemm::dot(&psign, atom_f);
        }
        for kk in 0..self.k {
            let alpha = coeffs[base + kk];
            let dd = &mut d_dict[kk * b..(kk + 1) * b];
            for f in 0..b {
                dd[f] += pbuf[f] * alpha;
            }
        }
    }

    /// Fused-pair variant: two coefficient sets sharing one input block and the
    /// shared dictionary (the up+gate FFN backward).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn accum_pair_block_grads(
        &self,
        pba: &[Complex32],
        pbb: &[Complex32],
        dict: &[Complex32],
        coeffs_a: &[f32],
        coeffs_b: &[f32],
        base: usize,
        dca: &mut [f32],
        dcb: &mut [f32],
        d_dict: &mut [Complex32],
    ) {
        let b = self.b;
        let mut psa = vec![0.0f32; 2 * b];
        let mut psb = vec![0.0f32; 2 * b];
        for f in 0..b {
            psa[2 * f] = pba[f].re;
            psa[2 * f + 1] = -pba[f].im;
            psb[2 * f] = pbb[f].re;
            psb[2 * f + 1] = -pbb[f].im;
        }
        for kk in 0..self.k {
            let atom = &dict[kk * b..(kk + 1) * b];
            let atom_f = unsafe { std::slice::from_raw_parts(atom.as_ptr() as *const f32, 2 * b) };
            dca[base + kk] = crate::kernels::gemm::dot(&psa, atom_f);
            dcb[base + kk] = crate::kernels::gemm::dot(&psb, atom_f);
        }
        for kk in 0..self.k {
            let alpha_a = coeffs_a[base + kk];
            let alpha_b = coeffs_b[base + kk];
            let dd = &mut d_dict[kk * b..(kk + 1) * b];
            for f in 0..b {
                dd[f] += pba[f] * alpha_a + pbb[f] * alpha_b;
            }
        }
    }

    /// Forward pass `y = W x` over the shared dictionary.
    ///
    /// - `dict`: shared dictionary `G`, `K×b` complex, row-major.
    /// - `coeffs`: `α`, length [`Self::coeff_len`], laid out `[p][q][k]`.
    /// - `x`: input, length [`Self::in_dim`].
    ///
    /// Returns `y`, length [`Self::out_dim`].
    pub fn forward(&self, dict: &[Complex32], coeffs: &[f32], x: &[f32]) -> Vec<f32> {
        assert_eq!(dict.len(), self.dict_len(), "dictionary shape mismatch");
        assert_eq!(coeffs.len(), self.coeff_len(), "coefficient shape mismatch");
        assert_eq!(x.len(), self.in_dim(), "input shape mismatch");

        let b = self.b;

        // Precompute FFT of each input block once (reused across all row-blocks).
        let mut xq: Vec<Vec<Complex32>> = Vec::with_capacity(self.q);
        for qq in 0..self.q {
            let mut blk: Vec<Complex32> =
                x[qq * b..(qq + 1) * b].iter().map(|&r| Complex32::new(r, 0.0)).collect();
            self.fft.fft(&mut blk);
            xq.push(blk);
        }

        let mut y = vec![0.0f32; self.out_dim()];
        let mut lambda = vec![Complex32::new(0.0, 0.0); b];
        let mut acc = vec![Complex32::new(0.0, 0.0); b];

        for pp in 0..self.p {
            for v in acc.iter_mut() {
                *v = Complex32::new(0.0, 0.0);
            }
            for qq in 0..self.q {
                let base = (pp * self.q + qq) * self.k;
                let coeffs_blk = &coeffs[base..base + self.k];
                self.block_eigs(dict, coeffs_blk, &mut lambda);
                let xblk = &xq[qq];
                // acc += λ ⊙ FFT(x_q): accumulate this block's spectral product.
                for f in 0..b {
                    acc[f] += lambda[f] * xblk[f];
                }
            }
            // y_p = real(IFFT(acc))
            self.fft.ifft(&mut acc);
            let yblk = &mut y[pp * b..(pp + 1) * b];
            for f in 0..b {
                yblk[f] = acc[f].re;
            }
        }
        y
    }

    /// Materialize the dense `out×in` matrix this `(dict, coeffs)` represents,
    /// by forwarding canonical basis vectors. Test/debug aid only — defeats the
    /// whole point of the compression, so never on a hot path.
    pub fn to_dense(&self, dict: &[Complex32], coeffs: &[f32]) -> Vec<f32> {
        let (out, in_) = (self.out_dim(), self.in_dim());
        let mut w = vec![0.0f32; out * in_];
        let mut e = vec![0.0f32; in_];
        for j in 0..in_ {
            e[j] = 1.0;
            let col = self.forward(dict, coeffs, &e);
            for i in 0..out {
                w[i * in_ + j] = col[i];
            }
            e[j] = 0.0;
        }
        w
    }
}

/// Compile-time switch selecting how a layer's coefficients are initialized.
///
/// `false` (default, the primary research path): from-scratch random init —
/// `α` and the dictionary `G` are trained directly under the circular
/// constraint. `true`: warm-start — fit `α` to provided dense weights via the
/// compress path ([`BasisMatmul::fit_from_dense`]).
pub const INIT_FROM_DENSE: bool = false;

/// Gradients produced by [`BasisMatmul::backward`].
pub struct BasisGrads {
    /// dL/dα, length [`BasisMatmul::coeff_len`].
    pub d_coeffs: Vec<f32>,
    /// dL/dG for the shared dictionary, length [`BasisMatmul::dict_len`].
    /// Stored so `.re = ∂L/∂Re(G)` and `.im = ∂L/∂Im(G)`.
    pub d_dict: Vec<Complex32>,
    /// dL/dx, length [`BasisMatmul::in_dim`].
    pub d_x: Vec<f32>,
}

/// Gradients from the fused [`BasisMatmul::backward_rows_pair`] of two
/// projections that share one input and one dictionary (e.g. FFN up + gate).
/// `d_coeffs_a`/`d_coeffs_b` are the two per-projection coefficient gradients;
/// `d_dict` and `d_x` are the *combined* contributions of both projections to
/// the shared dictionary and shared input (their sums).
pub struct PairGrads {
    pub d_coeffs_a: Vec<f32>,
    pub d_coeffs_b: Vec<f32>,
    pub d_dict: Vec<Complex32>,
    pub d_x: Vec<f32>,
}

/// Deterministic splitmix64 — reproducible parameter init without a dependency.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform sample in `[-1, 1)` from a splitmix64 stream (24-bit mantissa).
fn next_sym(state: &mut u64) -> f32 {
    ((splitmix64(state) >> 40) as f32 / (1u64 << 23) as f32) * 2.0 - 1.0
}

/// Random shared dictionary `G` (`K×b` complex, row-major), each component
/// `~ U[-std, std)`. Shared across all matrices/layers that use it.
pub fn init_dict_random(k: usize, b: usize, seed: u64, std: f32) -> Vec<Complex32> {
    let mut s = seed ^ 0xD1C7_DBA5_512D_1C75;
    (0..k * b).map(|_| Complex32::new(next_sym(&mut s) * std, next_sym(&mut s) * std)).collect()
}

/// Random coefficient vector `α ~ U[-std, std)` for one matrix.
pub fn init_coeffs_random(len: usize, seed: u64, std: f32) -> Vec<f32> {
    let mut s = seed ^ 0xC0EF_F1C1_E175_EED5;
    (0..len).map(|_| next_sym(&mut s) * std).collect()
}

/// Solve `A x = b` for small dense `A` (`n×n`, row-major) via Gauss-Jordan with
/// partial pivoting. Used for the per-block least-squares coefficient fit.
fn solve_dense(mut a: Vec<f32>, mut b: Vec<f32>, n: usize) -> Vec<f32> {
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if piv != col {
            for c in 0..n {
                a.swap(col * n + c, piv * n + c);
            }
            b.swap(col, piv);
        }
        let d = a[col * n + col];
        if d.abs() < 1e-12 {
            continue; // near-singular column; leave as 0 downstream
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r * n + col] / d;
            if f != 0.0 {
                for c in col..n {
                    a[r * n + c] -= f * a[col * n + c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    let mut x = vec![0.0f32; n];
    for i in 0..n {
        let d = a[i * n + i];
        x[i] = if d.abs() < 1e-12 { 0.0 } else { b[i] / d };
    }
    x
}

/// Quantize coefficients into per-group adaptive precision under `dial` (max
/// absolute dequantization error). Returns `(bit_widths, packed, scales)`.
/// Picks 8-bit where it stays within tolerance, else 16-bit. Sub-byte widths
/// are a later optimization; 8/16 keep packing byte-aligned and correct.
fn quantize_coeffs(coeffs: &[f32], group_size: usize, dial: f32) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    let n = coeffs.len();
    let ng = n.div_ceil(group_size);
    let mut bit_widths = Vec::with_capacity(ng);
    let mut scales = Vec::with_capacity(ng);
    let mut packed = Vec::new();
    for g in 0..ng {
        let s = g * group_size;
        let e = ((g + 1) * group_size).min(n);
        let grp = &coeffs[s..e];
        let maxabs = grp.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let (bits, scale) = if maxabs == 0.0 {
            (8u8, 0.0f32)
        } else {
            let s8 = maxabs / 127.0;
            let err8 = grp.iter().fold(0.0f32, |m, &v| {
                let q = (v / s8).round().clamp(-127.0, 127.0);
                m.max((v - q * s8).abs())
            });
            if err8 <= dial { (8u8, s8) } else { (16u8, maxabs / 32767.0) }
        };
        bit_widths.push(bits);
        scales.push(scale);
        for &v in grp {
            if bits == 8 {
                let code = if scale == 0.0 { 0i8 } else { (v / scale).round().clamp(-127.0, 127.0) as i8 };
                packed.push(code as u8);
            } else {
                let code =
                    if scale == 0.0 { 0i16 } else { (v / scale).round().clamp(-32767.0, 32767.0) as i16 };
                packed.extend_from_slice(&code.to_le_bytes());
            }
        }
    }
    (bit_widths, packed, scales)
}

/// Inverse of [`quantize_coeffs`]; `n` is the total coefficient count.
fn decompress_coeffs(cw: &CompressedWeight, n: usize) -> Vec<f32> {
    let gs = cw.group_size;
    let bw = cw.bit_widths.as_slice();
    let sc = cw.scales.as_slice();
    let pk = cw.packed.as_slice();
    let mut out = vec![0.0f32; n];
    let mut off = 0usize;
    for (g, &bits) in bw.iter().enumerate() {
        let s = g * gs;
        let e = ((g + 1) * gs).min(n);
        let scale = sc[g];
        if bits == 8 {
            for slot in out[s..e].iter_mut() {
                *slot = pk[off] as i8 as f32 * scale;
                off += 1;
            }
        } else {
            for slot in out[s..e].iter_mut() {
                let code = i16::from_le_bytes([pk[off], pk[off + 1]]);
                *slot = code as f32 * scale;
                off += 2;
            }
        }
    }
    out
}

impl BasisMatmul {
    /// Backward pass. Given upstream `dy = dL/dy`, returns gradients w.r.t. the
    /// coefficients `α`, the shared dictionary `G`, and the input `x`.
    ///
    /// Derivation (real loss, all real DOFs): with `A_p = (1/b)·conj(FFT(dy_p))`
    /// and `X_q = FFT(x_q)`,
    /// - `dL/dα_{p,q,k} = Σ_f Re(A_p[f]·G_k[f]·X_q[f])`
    /// - `dL/dG_k       = conj(Σ_{p,q} α_{p,q,k}·A_p·X_q)`
    /// - `dL/dx_q       = Re(FFT(C_q))`, `C_q = Σ_p A_p·λ_{p,q}`
    pub fn backward(
        &self,
        dict: &[Complex32],
        coeffs: &[f32],
        x: &[f32],
        dy: &[f32],
    ) -> BasisGrads {
        assert_eq!(dict.len(), self.dict_len(), "dictionary shape mismatch");
        assert_eq!(coeffs.len(), self.coeff_len(), "coefficient shape mismatch");
        assert_eq!(x.len(), self.in_dim(), "input shape mismatch");
        assert_eq!(dy.len(), self.out_dim(), "grad-output shape mismatch");

        let b = self.b;
        let inv_b = 1.0 / b as f32;

        // X_q = FFT(x_q)
        let mut xq: Vec<Vec<Complex32>> = Vec::with_capacity(self.q);
        for qq in 0..self.q {
            let mut blk: Vec<Complex32> =
                x[qq * b..(qq + 1) * b].iter().map(|&r| Complex32::new(r, 0.0)).collect();
            self.fft.fft(&mut blk);
            xq.push(blk);
        }
        // A_p = (1/b) conj(FFT(dy_p))
        let mut ap: Vec<Vec<Complex32>> = Vec::with_capacity(self.p);
        for pp in 0..self.p {
            let mut blk: Vec<Complex32> =
                dy[pp * b..(pp + 1) * b].iter().map(|&r| Complex32::new(r, 0.0)).collect();
            self.fft.fft(&mut blk);
            for v in blk.iter_mut() {
                *v = v.conj() * inv_b;
            }
            ap.push(blk);
        }

        let mut d_coeffs = vec![0.0f32; self.coeff_len()];
        let mut d_dict = vec![Complex32::new(0.0, 0.0); self.dict_len()]; // accumulates B_k
        let mut cq = vec![vec![Complex32::new(0.0, 0.0); b]; self.q];
        let mut lambda = vec![Complex32::new(0.0, 0.0); b];
        let mut pbuf = vec![Complex32::new(0.0, 0.0); b];
        // pbuf with imaginary parts negated, interleaved as 2b reals, so that
        // Re(pbuf·atom) becomes a plain real dot against the atom's re/im layout.
        let mut psign = vec![0.0f32; 2 * b];

        for pp in 0..self.p {
            let a = &ap[pp];
            for qq in 0..self.q {
                let xb = &xq[qq];
                let base = (pp * self.q + qq) * self.k;
                self.block_eigs(dict, &coeffs[base..base + self.k], &mut lambda);
                let cqq = &mut cq[qq];
                for f in 0..b {
                    pbuf[f] = a[f] * xb[f];
                    cqq[f] += a[f] * lambda[f];
                }
                // d_coeffs[k] = Σ_f Re(pbuf[f]·atom_k[f]); with psign = [re, -im] of
                // pbuf this is a real dot against atom's re/im layout — a reduction,
                // so hand-vectorized via gemm::dot.
                for f in 0..b {
                    psign[2 * f] = pbuf[f].re;
                    psign[2 * f + 1] = -pbuf[f].im;
                }
                for kk in 0..self.k {
                    let atom = &dict[kk * b..(kk + 1) * b];
                    // SAFETY: Complex32 is repr(C){re,im}, so a [b] complex slice
                    // aliases a [2b] f32 slice of the same lifetime.
                    let atom_f = unsafe { std::slice::from_raw_parts(atom.as_ptr() as *const f32, 2 * b) };
                    d_coeffs[base + kk] = crate::kernels::gemm::dot(&psign, atom_f);
                }
                // d_dict[k] += pbuf · alpha[k]: a complex axpy that auto-vectorizes
                // once unfused from the reduction above.
                for kk in 0..self.k {
                    let alpha = coeffs[base + kk];
                    let dd = &mut d_dict[kk * b..(kk + 1) * b];
                    for f in 0..b {
                        dd[f] += pbuf[f] * alpha;
                    }
                }
            }
        }
        // dL/dG = conj(B)
        for v in d_dict.iter_mut() {
            *v = v.conj();
        }
        // dL/dx_q = Re(FFT(C_q))
        let mut d_x = vec![0.0f32; self.in_dim()];
        for qq in 0..self.q {
            let mut buf = std::mem::take(&mut cq[qq]);
            self.fft.fft(&mut buf);
            for n in 0..b {
                d_x[qq * b + n] = buf[n].re;
            }
        }
        BasisGrads { d_coeffs, d_dict, d_x }
    }

    /// Masked backward for [`forward_rows`]. Only the output blocks in
    /// `active_p` carried signal in the forward, so gradient flows only from
    /// them: coefficients of inactive output blocks get zero gradient (they did
    /// not participate), and `dy` is read only on the active blocks. `d_x` still
    /// sums over all input blocks `q`, since every input fed the active outputs.
    pub fn backward_rows(
        &self,
        dict: &[Complex32],
        coeffs: &[f32],
        x: &[f32],
        dy: &[f32],
        active_p: &[usize],
    ) -> BasisGrads {
        assert_eq!(dict.len(), self.dict_len(), "dictionary shape mismatch");
        assert_eq!(coeffs.len(), self.coeff_len(), "coefficient shape mismatch");
        assert_eq!(x.len(), self.in_dim(), "input shape mismatch");
        assert_eq!(dy.len(), self.out_dim(), "grad-output shape mismatch");

        let b = self.b;
        let inv_b = 1.0 / b as f32;

        // X_q = FFT(x_q) for every input block.
        let mut xq: Vec<Vec<Complex32>> = Vec::with_capacity(self.q);
        for qq in 0..self.q {
            let mut blk: Vec<Complex32> =
                x[qq * b..(qq + 1) * b].iter().map(|&r| Complex32::new(r, 0.0)).collect();
            self.fft.fft(&mut blk);
            xq.push(blk);
        }

        let mut d_coeffs = vec![0.0f32; self.coeff_len()];
        let mut d_dict = vec![Complex32::new(0.0, 0.0); self.dict_len()];
        let mut cq = vec![vec![Complex32::new(0.0, 0.0); b]; self.q];
        let mut lambda = vec![Complex32::new(0.0, 0.0); b];
        let mut pbuf = vec![Complex32::new(0.0, 0.0); b];

        for &pp in active_p {
            debug_assert!(pp < self.p, "active row-block {pp} out of range");
            // A_p = (1/b) conj(FFT(dy_p))
            let mut a: Vec<Complex32> =
                dy[pp * b..(pp + 1) * b].iter().map(|&r| Complex32::new(r, 0.0)).collect();
            self.fft.fft(&mut a);
            for v in a.iter_mut() {
                *v = v.conj() * inv_b;
            }
            for qq in 0..self.q {
                let xb = &xq[qq];
                let base = (pp * self.q + qq) * self.k;
                self.block_eigs(dict, &coeffs[base..base + self.k], &mut lambda);
                let cqq = &mut cq[qq];
                for f in 0..b {
                    pbuf[f] = a[f] * xb[f];
                    cqq[f] += a[f] * lambda[f];
                }
                self.accum_block_grads(&pbuf, dict, coeffs, base, &mut d_coeffs, &mut d_dict);
            }
        }
        for v in d_dict.iter_mut() {
            *v = v.conj();
        }
        let mut d_x = vec![0.0f32; self.in_dim()];
        for qq in 0..self.q {
            let mut buf = std::mem::take(&mut cq[qq]);
            self.fft.fft(&mut buf);
            for n in 0..b {
                d_x[qq * b + n] = buf[n].re;
            }
        }
        BasisGrads { d_coeffs, d_dict, d_x }
    }

    /// Masked backward for [`forward_cols`]. Only the input blocks in `active_q`
    /// fed the forward (the rest were exactly zero), so coefficients of inactive
    /// input blocks get zero gradient and `d_x` is nonzero only on `active_q`.
    /// Every output block `p` still receives gradient from `dy`.
    pub fn backward_cols(
        &self,
        dict: &[Complex32],
        coeffs: &[f32],
        x: &[f32],
        dy: &[f32],
        active_q: &[usize],
    ) -> BasisGrads {
        assert_eq!(dict.len(), self.dict_len(), "dictionary shape mismatch");
        assert_eq!(coeffs.len(), self.coeff_len(), "coefficient shape mismatch");
        assert_eq!(x.len(), self.in_dim(), "input shape mismatch");
        assert_eq!(dy.len(), self.out_dim(), "grad-output shape mismatch");

        let b = self.b;
        let inv_b = 1.0 / b as f32;

        // X_q = FFT(x_q) for active input blocks only (rest are zero anyway).
        let mut xq: Vec<Vec<Complex32>> = vec![Vec::new(); self.q];
        for &qq in active_q {
            debug_assert!(qq < self.q, "active col-block {qq} out of range");
            let mut blk: Vec<Complex32> =
                x[qq * b..(qq + 1) * b].iter().map(|&r| Complex32::new(r, 0.0)).collect();
            self.fft.fft(&mut blk);
            xq[qq] = blk;
        }
        // A_p = (1/b) conj(FFT(dy_p)) for every output block.
        let mut ap: Vec<Vec<Complex32>> = Vec::with_capacity(self.p);
        for pp in 0..self.p {
            let mut blk: Vec<Complex32> =
                dy[pp * b..(pp + 1) * b].iter().map(|&r| Complex32::new(r, 0.0)).collect();
            self.fft.fft(&mut blk);
            for v in blk.iter_mut() {
                *v = v.conj() * inv_b;
            }
            ap.push(blk);
        }

        let mut d_coeffs = vec![0.0f32; self.coeff_len()];
        let mut d_dict = vec![Complex32::new(0.0, 0.0); self.dict_len()];
        let mut cq = vec![vec![Complex32::new(0.0, 0.0); b]; self.q];
        let mut lambda = vec![Complex32::new(0.0, 0.0); b];
        let mut pbuf = vec![Complex32::new(0.0, 0.0); b];

        for pp in 0..self.p {
            let a = &ap[pp];
            for &qq in active_q {
                let xb = &xq[qq];
                let base = (pp * self.q + qq) * self.k;
                self.block_eigs(dict, &coeffs[base..base + self.k], &mut lambda);
                let cqq = &mut cq[qq];
                for f in 0..b {
                    pbuf[f] = a[f] * xb[f];
                    cqq[f] += a[f] * lambda[f];
                }
                self.accum_block_grads(&pbuf, dict, coeffs, base, &mut d_coeffs, &mut d_dict);
            }
        }
        for v in d_dict.iter_mut() {
            *v = v.conj();
        }
        let mut d_x = vec![0.0f32; self.in_dim()];
        for &qq in active_q {
            let mut buf = std::mem::take(&mut cq[qq]);
            self.fft.fft(&mut buf);
            for n in 0..b {
                d_x[qq * b + n] = buf[n].re;
            }
        }
        BasisGrads { d_coeffs, d_dict, d_x }
    }

    /// Fused row-skipped forward for two projections (`a`, `b`) that share this
    /// matmul's shape, the input `x`, and the dictionary `G` — the FFN up + gate
    /// case. `X_q = FFT(x)` is computed **once** and fed to both; the two
    /// spectral-product/IFFT pipelines are interleaved so the out-of-order core
    /// has two independent dependency chains per block to schedule. The result
    /// is bit-for-bit equal to calling [`forward_rows`](Self::forward_rows)
    /// twice — only the FFT and the input loads are shared.
    pub fn forward_rows_pair(
        &self,
        dict: &[Complex32],
        coeffs_a: &[f32],
        coeffs_b: &[f32],
        x: &[f32],
        active_p: &[usize],
    ) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(dict.len(), self.dict_len(), "dictionary shape mismatch");
        assert_eq!(coeffs_a.len(), self.coeff_len(), "coeff_a shape mismatch");
        assert_eq!(coeffs_b.len(), self.coeff_len(), "coeff_b shape mismatch");
        assert_eq!(x.len(), self.in_dim(), "input shape mismatch");

        let b = self.b;
        let mut xq: Vec<Vec<Complex32>> = Vec::with_capacity(self.q);
        for qq in 0..self.q {
            let mut blk: Vec<Complex32> =
                x[qq * b..(qq + 1) * b].iter().map(|&r| Complex32::new(r, 0.0)).collect();
            self.fft.fft(&mut blk);
            xq.push(blk);
        }

        let mut ya = vec![0.0f32; self.out_dim()];
        let mut yb = vec![0.0f32; self.out_dim()];
        let mut la = vec![Complex32::new(0.0, 0.0); b];
        let mut lb = vec![Complex32::new(0.0, 0.0); b];
        let mut aca = vec![Complex32::new(0.0, 0.0); b];
        let mut acb = vec![Complex32::new(0.0, 0.0); b];
        for &pp in active_p {
            debug_assert!(pp < self.p, "active row-block {pp} out of range");
            for f in 0..b {
                aca[f] = Complex32::new(0.0, 0.0);
                acb[f] = Complex32::new(0.0, 0.0);
            }
            for qq in 0..self.q {
                let base = (pp * self.q + qq) * self.k;
                self.block_eigs(dict, &coeffs_a[base..base + self.k], &mut la);
                self.block_eigs(dict, &coeffs_b[base..base + self.k], &mut lb);
                let xb = &xq[qq];
                for f in 0..b {
                    aca[f] += la[f] * xb[f];
                    acb[f] += lb[f] * xb[f];
                }
            }
            self.fft.ifft(&mut aca);
            self.fft.ifft(&mut acb);
            let (yablk, ybblk) =
                (&mut ya[pp * b..(pp + 1) * b], &mut yb[pp * b..(pp + 1) * b]);
            for f in 0..b {
                yablk[f] = aca[f].re;
            }
            for f in 0..b {
                ybblk[f] = acb[f].re;
            }
        }
        (ya, yb)
    }

    /// Fused row-skipped backward for two projections sharing input `x` and
    /// dictionary `G` (FFN up + gate). Shares `X_q = FFT(x)` across both, and
    /// returns the two coefficient gradients plus the *combined* dictionary and
    /// input gradients (the sums over both projections — exactly what summing
    /// two [`backward_rows`](Self::backward_rows) calls would give).
    pub fn backward_rows_pair(
        &self,
        dict: &[Complex32],
        coeffs_a: &[f32],
        coeffs_b: &[f32],
        x: &[f32],
        dy_a: &[f32],
        dy_b: &[f32],
        active_p: &[usize],
    ) -> PairGrads {
        assert_eq!(dict.len(), self.dict_len(), "dictionary shape mismatch");
        assert_eq!(coeffs_a.len(), self.coeff_len(), "coeff_a shape mismatch");
        assert_eq!(coeffs_b.len(), self.coeff_len(), "coeff_b shape mismatch");
        assert_eq!(x.len(), self.in_dim(), "input shape mismatch");
        assert_eq!(dy_a.len(), self.out_dim(), "grad-output a shape mismatch");
        assert_eq!(dy_b.len(), self.out_dim(), "grad-output b shape mismatch");

        let b = self.b;
        let inv_b = 1.0 / b as f32;

        let mut xq: Vec<Vec<Complex32>> = Vec::with_capacity(self.q);
        for qq in 0..self.q {
            let mut blk: Vec<Complex32> =
                x[qq * b..(qq + 1) * b].iter().map(|&r| Complex32::new(r, 0.0)).collect();
            self.fft.fft(&mut blk);
            xq.push(blk);
        }

        let mut dca = vec![0.0f32; self.coeff_len()];
        let mut dcb = vec![0.0f32; self.coeff_len()];
        let mut d_dict = vec![Complex32::new(0.0, 0.0); self.dict_len()];
        let mut cqa = vec![vec![Complex32::new(0.0, 0.0); b]; self.q];
        let mut cqb = vec![vec![Complex32::new(0.0, 0.0); b]; self.q];
        let mut la = vec![Complex32::new(0.0, 0.0); b];
        let mut lb = vec![Complex32::new(0.0, 0.0); b];
        let mut pba = vec![Complex32::new(0.0, 0.0); b];
        let mut pbb = vec![Complex32::new(0.0, 0.0); b];

        for &pp in active_p {
            debug_assert!(pp < self.p, "active row-block {pp} out of range");
            let mut aa: Vec<Complex32> =
                dy_a[pp * b..(pp + 1) * b].iter().map(|&r| Complex32::new(r, 0.0)).collect();
            self.fft.fft(&mut aa);
            for v in aa.iter_mut() {
                *v = v.conj() * inv_b;
            }
            let mut ab: Vec<Complex32> =
                dy_b[pp * b..(pp + 1) * b].iter().map(|&r| Complex32::new(r, 0.0)).collect();
            self.fft.fft(&mut ab);
            for v in ab.iter_mut() {
                *v = v.conj() * inv_b;
            }
            for qq in 0..self.q {
                let xb = &xq[qq];
                let base = (pp * self.q + qq) * self.k;
                self.block_eigs(dict, &coeffs_a[base..base + self.k], &mut la);
                self.block_eigs(dict, &coeffs_b[base..base + self.k], &mut lb);
                let (cqaq, cqbq) = (&mut cqa[qq], &mut cqb[qq]);
                for f in 0..b {
                    pba[f] = aa[f] * xb[f];
                    pbb[f] = ab[f] * xb[f];
                    cqaq[f] += aa[f] * la[f];
                    cqbq[f] += ab[f] * lb[f];
                }
                self.accum_pair_block_grads(
                    &pba, &pbb, dict, coeffs_a, coeffs_b, base, &mut dca, &mut dcb, &mut d_dict,
                );
            }
        }
        for v in d_dict.iter_mut() {
            *v = v.conj();
        }
        let mut d_x = vec![0.0f32; self.in_dim()];
        for qq in 0..self.q {
            let mut bufa = std::mem::take(&mut cqa[qq]);
            let mut bufb = std::mem::take(&mut cqb[qq]);
            self.fft.fft(&mut bufa);
            self.fft.fft(&mut bufb);
            for n in 0..b {
                d_x[qq * b + n] = bufa[n].re + bufb[n].re;
            }
        }
        PairGrads { d_coeffs_a: dca, d_coeffs_b: dcb, d_dict, d_x }
    }

    /// Initialize coefficients per the [`INIT_FROM_DENSE`] switch. `dense` is
    /// only consulted on the warm-start path.
    pub fn init_coeffs(&self, dict: &[Complex32], dense: Option<&[f32]>, seed: u64) -> Vec<f32> {
        if INIT_FROM_DENSE {
            let w = dense.expect("INIT_FROM_DENSE: dense weights required");
            self.fit_from_dense(dict, w)
        } else {
            init_coeffs_random(self.coeff_len(), seed, 0.02)
        }
    }

    /// Best least-squares circulant approximation of one `b×b` block: average
    /// each circular diagonal to get the first column, then FFT to eigenvalues.
    pub fn best_circulant_eigs(&self, block: &[f32]) -> Vec<Complex32> {
        let b = self.b;
        assert_eq!(block.len(), b * b, "block must be b×b");
        let mut c = vec![0.0f32; b];
        for (d, slot) in c.iter_mut().enumerate() {
            let mut s = 0.0f32;
            for i in 0..b {
                s += block[i * b + ((i + b - d) % b)];
            }
            *slot = s / b as f32;
        }
        let mut cc: Vec<Complex32> = c.iter().map(|&r| Complex32::new(r, 0.0)).collect();
        self.fft.fft(&mut cc);
        cc
    }

    /// Least-squares fit of real coefficients `α` so that `Σ_k α_k G_k ≈ target`
    /// (target eigenvalues), via the `K×K` normal equations.
    pub fn fit_coeffs(&self, dict: &[Complex32], target: &[Complex32]) -> Vec<f32> {
        let (k, b) = (self.k, self.b);
        let mut nmat = vec![0.0f32; k * k];
        let mut rhs = vec![0.0f32; k];
        for kk in 0..k {
            let ak = &dict[kk * b..(kk + 1) * b];
            for jj in 0..k {
                let aj = &dict[jj * b..(jj + 1) * b];
                let mut s = 0.0f32;
                for f in 0..b {
                    s += ak[f].re * aj[f].re + ak[f].im * aj[f].im;
                }
                nmat[kk * k + jj] = s;
            }
            let mut r = 0.0f32;
            for f in 0..b {
                r += ak[f].re * target[f].re + ak[f].im * target[f].im;
            }
            rhs[kk] = r;
        }
        solve_dense(nmat, rhs, k)
    }

    /// Fit coefficients for a full dense matrix `W` (`out×in`, row-major):
    /// per block, best-circulant approx then least-squares fit against `dict`.
    pub fn fit_from_dense(&self, dict: &[Complex32], w: &[f32]) -> Vec<f32> {
        let (in_, b) = (self.in_dim(), self.b);
        assert_eq!(w.len(), self.out_dim() * in_, "dense weight shape mismatch");
        let mut coeffs = vec![0.0f32; self.coeff_len()];
        let mut block = vec![0.0f32; b * b];
        for pp in 0..self.p {
            for qq in 0..self.q {
                for i in 0..b {
                    for j in 0..b {
                        block[i * b + j] = w[(pp * b + i) * in_ + qq * b + j];
                    }
                }
                let eigs = self.best_circulant_eigs(&block);
                let a = self.fit_coeffs(dict, &eigs);
                let base = (pp * self.q + qq) * self.k;
                coeffs[base..base + self.k].copy_from_slice(&a);
            }
        }
        coeffs
    }

    /// Compress dense weights `W` into a [`CompressedWeight`] under `dial`.
    pub fn compress_dense(
        &self,
        dict: &[Complex32],
        w: &[f32],
        dial: f32,
        group_size: usize,
    ) -> CompressedWeight {
        let coeffs = self.fit_from_dense(dict, w);
        let (bw, packed, scales) = quantize_coeffs(&coeffs, group_size, dial);
        CompressedWeight {
            bit_widths: AlignedVec::from_slice(&bw),
            scales: AlignedVec::from_slice(&scales),
            packed: AlignedVec::from_slice(&packed),
            shape: [self.out_dim(), self.in_dim()],
            group_size,
        }
    }

    /// Forward from a compressed weight: dequantize coefficients, then [`forward`].
    ///
    /// [`forward`]: Self::forward
    pub fn forward_compressed(
        &self,
        dict: &[Complex32],
        cw: &CompressedWeight,
        x: &[f32],
    ) -> Vec<f32> {
        let coeffs = decompress_coeffs(cw, self.coeff_len());
        self.forward(dict, &coeffs, x)
    }
}

impl BasisMatmul {
    /// Forward computing only the selected output row-blocks (`active_p`); every
    /// other output block is left zero. This is the P-side skip for the routed
    /// `W_up` / `W_gate` projections — only the routed output blocks are needed.
    ///
    /// Cost scales with `|active_p|` instead of `P`.
    pub fn forward_rows(
        &self,
        dict: &[Complex32],
        coeffs: &[f32],
        x: &[f32],
        active_p: &[usize],
    ) -> Vec<f32> {
        assert_eq!(dict.len(), self.dict_len(), "dictionary shape mismatch");
        assert_eq!(coeffs.len(), self.coeff_len(), "coefficient shape mismatch");
        assert_eq!(x.len(), self.in_dim(), "input shape mismatch");

        let b = self.b;
        let mut xq: Vec<Vec<Complex32>> = Vec::with_capacity(self.q);
        for qq in 0..self.q {
            let mut blk: Vec<Complex32> =
                x[qq * b..(qq + 1) * b].iter().map(|&r| Complex32::new(r, 0.0)).collect();
            self.fft.fft(&mut blk);
            xq.push(blk);
        }

        let mut y = vec![0.0f32; self.out_dim()];
        let mut lambda = vec![Complex32::new(0.0, 0.0); b];
        let mut acc = vec![Complex32::new(0.0, 0.0); b];
        for &pp in active_p {
            debug_assert!(pp < self.p, "active row-block {pp} out of range");
            for v in acc.iter_mut() {
                *v = Complex32::new(0.0, 0.0);
            }
            for qq in 0..self.q {
                let base = (pp * self.q + qq) * self.k;
                self.block_eigs(dict, &coeffs[base..base + self.k], &mut lambda);
                let xb = &xq[qq];
                for f in 0..b {
                    acc[f] += lambda[f] * xb[f];
                }
            }
            self.fft.ifft(&mut acc);
            let yblk = &mut y[pp * b..(pp + 1) * b];
            for f in 0..b {
                yblk[f] = acc[f].re;
            }
        }
        y
    }

    /// Forward summing only over the selected input col-blocks (`active_q`); all
    /// output blocks are produced. This is the Q-side skip for the routed
    /// `W_down` projection, whose input (the FFN activation) is zero outside the
    /// routed blocks — so skipping them is exact, not approximate.
    ///
    /// Cost scales with `|active_q|` instead of `Q`.
    pub fn forward_cols(
        &self,
        dict: &[Complex32],
        coeffs: &[f32],
        x: &[f32],
        active_q: &[usize],
    ) -> Vec<f32> {
        assert_eq!(dict.len(), self.dict_len(), "dictionary shape mismatch");
        assert_eq!(coeffs.len(), self.coeff_len(), "coefficient shape mismatch");
        assert_eq!(x.len(), self.in_dim(), "input shape mismatch");

        let b = self.b;
        // FFT only the active input blocks, aligned with `active_q`.
        let mut xq_active: Vec<Vec<Complex32>> = Vec::with_capacity(active_q.len());
        for &qq in active_q {
            debug_assert!(qq < self.q, "active col-block {qq} out of range");
            let mut blk: Vec<Complex32> =
                x[qq * b..(qq + 1) * b].iter().map(|&r| Complex32::new(r, 0.0)).collect();
            self.fft.fft(&mut blk);
            xq_active.push(blk);
        }

        let mut y = vec![0.0f32; self.out_dim()];
        let mut lambda = vec![Complex32::new(0.0, 0.0); b];
        let mut acc = vec![Complex32::new(0.0, 0.0); b];
        for pp in 0..self.p {
            for v in acc.iter_mut() {
                *v = Complex32::new(0.0, 0.0);
            }
            for (idx, &qq) in active_q.iter().enumerate() {
                let base = (pp * self.q + qq) * self.k;
                self.block_eigs(dict, &coeffs[base..base + self.k], &mut lambda);
                let xb = &xq_active[idx];
                for f in 0..b {
                    acc[f] += lambda[f] * xb[f];
                }
            }
            self.fft.ifft(&mut acc);
            let yblk = &mut y[pp * b..(pp + 1) * b];
            for f in 0..b {
                yblk[f] = acc[f].re;
            }
        }
        y
    }

    /// Compact variant of [`forward_cols`](Self::forward_cols) where `x` has length
    /// `active_q.len() * b` — block `active_q[si]` maps to `x[si*b..(si+1)*b]`.
    /// Avoids allocating and zero-filling a full `in_dim()`-sized input buffer.
    pub fn forward_cols_compact(
        &self,
        dict: &[Complex32],
        coeffs: &[f32],
        x: &[f32],
        active_q: &[usize],
    ) -> Vec<f32> {
        assert_eq!(dict.len(), self.dict_len(), "dictionary shape mismatch");
        assert_eq!(coeffs.len(), self.coeff_len(), "coefficient shape mismatch");
        assert_eq!(x.len(), active_q.len() * self.b, "compact input length mismatch");

        let b = self.b;
        let mut xq_active: Vec<Vec<Complex32>> = Vec::with_capacity(active_q.len());
        for si in 0..active_q.len() {
            let qq = active_q[si];
            debug_assert!(qq < self.q, "active col-block {qq} out of range");
            let mut blk: Vec<Complex32> =
                x[si * b..(si + 1) * b].iter().map(|&r| Complex32::new(r, 0.0)).collect();
            self.fft.fft(&mut blk);
            xq_active.push(blk);
        }

        let mut y = vec![0.0f32; self.out_dim()];
        let mut lambda = vec![Complex32::new(0.0, 0.0); b];
        let mut acc = vec![Complex32::new(0.0, 0.0); b];
        for pp in 0..self.p {
            for v in acc.iter_mut() {
                *v = Complex32::new(0.0, 0.0);
            }
            for (idx, &qq) in active_q.iter().enumerate() {
                let base = (pp * self.q + qq) * self.k;
                self.block_eigs(dict, &coeffs[base..base + self.k], &mut lambda);
                let xb = &xq_active[idx];
                for f in 0..b {
                    acc[f] += lambda[f] * xb[f];
                }
            }
            self.fft.ifft(&mut acc);
            let yblk = &mut y[pp * b..(pp + 1) * b];
            for f in 0..b {
                yblk[f] = acc[f].re;
            }
        }
        y
    }

    /// Compact variant of [`backward_cols`](Self::backward_cols) where `x` has
    /// length `active_q.len() * b` — block `active_q[si]` maps to
    /// `x[si*b..(si+1)*b]`. Returns `d_x` in the same compact layout.
    pub fn backward_cols_compact(
        &self,
        dict: &[Complex32],
        coeffs: &[f32],
        x: &[f32],
        dy: &[f32],
        active_q: &[usize],
    ) -> BasisGrads {
        assert_eq!(dict.len(), self.dict_len(), "dictionary shape mismatch");
        assert_eq!(coeffs.len(), self.coeff_len(), "coefficient shape mismatch");
        assert_eq!(x.len(), active_q.len() * self.b, "compact input length mismatch");
        assert_eq!(dy.len(), self.out_dim(), "grad-output shape mismatch");

        let b = self.b;
        let inv_b = 1.0 / b as f32;

        let mut xq: Vec<Vec<Complex32>> = vec![Vec::new(); self.q];
        for si in 0..active_q.len() {
            let qq = active_q[si];
            debug_assert!(qq < self.q, "active col-block {qq} out of range");
            let mut blk: Vec<Complex32> =
                x[si * b..(si + 1) * b].iter().map(|&r| Complex32::new(r, 0.0)).collect();
            self.fft.fft(&mut blk);
            xq[qq] = blk;
        }
        let mut ap: Vec<Vec<Complex32>> = Vec::with_capacity(self.p);
        for pp in 0..self.p {
            let mut blk: Vec<Complex32> =
                dy[pp * b..(pp + 1) * b].iter().map(|&r| Complex32::new(r, 0.0)).collect();
            self.fft.fft(&mut blk);
            for v in blk.iter_mut() {
                *v = v.conj() * inv_b;
            }
            ap.push(blk);
        }

        let mut d_coeffs = vec![0.0f32; self.coeff_len()];
        let mut d_dict = vec![Complex32::new(0.0, 0.0); self.dict_len()];
        let mut cq = vec![vec![Complex32::new(0.0, 0.0); b]; self.q];
        let mut lambda = vec![Complex32::new(0.0, 0.0); b];
        let mut pbuf = vec![Complex32::new(0.0, 0.0); b];

        for pp in 0..self.p {
            let a = &ap[pp];
            for &qq in active_q {
                let xb = &xq[qq];
                let base = (pp * self.q + qq) * self.k;
                self.block_eigs(dict, &coeffs[base..base + self.k], &mut lambda);
                let cqq = &mut cq[qq];
                for f in 0..b {
                    pbuf[f] = a[f] * xb[f];
                    cqq[f] += a[f] * lambda[f];
                }
                self.accum_block_grads(&pbuf, dict, coeffs, base, &mut d_coeffs, &mut d_dict);
            }
        }
        for v in d_dict.iter_mut() {
            *v = v.conj();
        }
        let mut d_x = vec![0.0f32; active_q.len() * b];
        for (si, &qq) in active_q.iter().enumerate() {
            let mut buf = std::mem::take(&mut cq[qq]);
            self.fft.fft(&mut buf);
            for n in 0..b {
                d_x[si * b + n] = buf[n].re;
            }
        }
        BasisGrads { d_coeffs, d_dict, d_x }
    }

    /// Software-prefetch the coefficient tiles that [`forward_rows`](Self::forward_rows)
    /// / [`backward_rows`](Self::backward_rows) will read for the given output
    /// blocks, pulling them toward L1 ahead of the spectral compute. The coeff
    /// layout is `[P][Q][K]`, so each output block's tile is the **contiguous**
    /// span `[pp·Q·K, (pp+1)·Q·K)`. Purely a hint — correctness-neutral, and a
    /// no-op off `x86_64`.
    pub fn prefetch_rows(&self, coeffs: &[f32], active_p: &[usize]) {
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let span = self.q * self.k;
            for &pp in active_p {
                debug_assert!(pp < self.p);
                let base = pp * span;
                let mut off = base;
                while off < base + span {
                    // SAFETY: off < (pp+1)·Q·K ≤ P·Q·K = coeffs.len(); prefetch is a hint.
                    unsafe { _mm_prefetch(coeffs.as_ptr().add(off) as *const i8, _MM_HINT_T0) };
                    off += 16; // one 64-byte cache line = 16 f32
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (coeffs, active_p);
    }

    /// Prefetch the coefficient tiles [`forward_cols`](Self::forward_cols) /
    /// [`backward_cols`](Self::backward_cols) will read for the given input
    /// blocks. Here the access is **strided**: each input block `qq` is touched
    /// once per output block `pp`, at `[(pp·Q + qq)·K, +K)`. No-op off `x86_64`.
    pub fn prefetch_cols(&self, coeffs: &[f32], active_q: &[usize]) {
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            for pp in 0..self.p {
                let row = pp * self.q * self.k;
                for &qq in active_q {
                    debug_assert!(qq < self.q);
                    let base = row + qq * self.k;
                    let mut off = base;
                    while off < base + self.k {
                        // SAFETY: base+K ≤ (pp·Q + (qq+1))·K ≤ P·Q·K = coeffs.len().
                        unsafe { _mm_prefetch(coeffs.as_ptr().add(off) as *const i8, _MM_HINT_T0) };
                        off += 16;
                    }
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (coeffs, active_q);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_circular_conv(c: &[f32], x: &[f32]) -> Vec<f32> {
        let n = c.len();
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            let mut s = 0.0f32;
            for j in 0..n {
                s += c[(i + n - j) % n] * x[j];
            }
            y[i] = s;
        }
        y
    }

    // Tiny deterministic PRNG so tests don't pull a dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
    }

    #[test]
    fn fft_roundtrip() {
        let fft = Fft::new(8);
        let orig: Vec<Complex32> = (0..8).map(|i| Complex32::new(i as f32, -(i as f32))).collect();
        let mut buf = orig.clone();
        fft.fft(&mut buf);
        fft.ifft(&mut buf);
        for (a, b) in orig.iter().zip(buf.iter()) {
            assert!((a.re - b.re).abs() < 1e-4, "re {} vs {}", a.re, b.re);
            assert!((a.im - b.im).abs() < 1e-4, "im {} vs {}", a.im, b.im);
        }
    }

    #[test]
    fn circulant_matches_naive() {
        let fft = Fft::new(6);
        let c = [1.0f32, -2.0, 0.5, 3.0, -1.0, 0.25];
        let x = [0.5f32, 1.0, -1.0, 2.0, 0.0, -0.5];
        let got = circulant_matvec(&fft, &c, &x);
        let want = naive_circular_conv(&c, &x);
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-4, "got {g} want {w}");
        }
    }

    #[test]
    fn forward_is_linear_in_x() {
        // The forward must equal multiplication by the dense matrix it encodes.
        let (out, in_, b, k) = (8usize, 12usize, 4usize, 5usize);
        let bm = BasisMatmul::new(out, in_, b, k);

        let mut rng = Lcg(0x1234_5678_9abc_def0);
        let dict: Vec<Complex32> =
            (0..bm.dict_len()).map(|_| Complex32::new(rng.next_f32(), rng.next_f32())).collect();
        let coeffs: Vec<f32> = (0..bm.coeff_len()).map(|_| rng.next_f32()).collect();

        let w = bm.to_dense(&dict, &coeffs);

        let x: Vec<f32> = (0..in_).map(|_| rng.next_f32()).collect();
        let got = bm.forward(&dict, &coeffs, &x);

        // Reference: dense W @ x.
        let mut want = vec![0.0f32; out];
        for i in 0..out {
            let mut s = 0.0f32;
            for j in 0..in_ {
                s += w[i * in_ + j] * x[j];
            }
            want[i] = s;
        }
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-3, "got {g} want {w}");
        }
    }

    #[test]
    fn forward_superposition() {
        // Linearity check independent of to_dense: f(x1+x2) == f(x1)+f(x2).
        let bm = BasisMatmul::new(8, 8, 4, 3);
        let mut rng = Lcg(0xdead_beef_cafe_1234);
        let dict: Vec<Complex32> =
            (0..bm.dict_len()).map(|_| Complex32::new(rng.next_f32(), rng.next_f32())).collect();
        let coeffs: Vec<f32> = (0..bm.coeff_len()).map(|_| rng.next_f32()).collect();
        let x1: Vec<f32> = (0..bm.in_dim()).map(|_| rng.next_f32()).collect();
        let x2: Vec<f32> = (0..bm.in_dim()).map(|_| rng.next_f32()).collect();
        let xs: Vec<f32> = x1.iter().zip(&x2).map(|(a, b)| a + b).collect();

        let y1 = bm.forward(&dict, &coeffs, &x1);
        let y2 = bm.forward(&dict, &coeffs, &x2);
        let ys = bm.forward(&dict, &coeffs, &xs);
        for i in 0..bm.out_dim() {
            assert!((ys[i] - (y1[i] + y2[i])).abs() < 1e-3, "superposition broke at {i}");
        }
    }

    #[test]
    fn backward_gradcheck() {
        // Finite-difference check of dα, dG (re/im), dx against analytic VJP.
        let (out, in_, b, k) = (8usize, 8usize, 4usize, 5usize);
        let bm = BasisMatmul::new(out, in_, b, k);
        let mut rng = Lcg(0x00AB_CDEF_1234_5678);
        let dict: Vec<Complex32> =
            (0..bm.dict_len()).map(|_| Complex32::new(rng.next_f32(), rng.next_f32())).collect();
        let coeffs: Vec<f32> = (0..bm.coeff_len()).map(|_| rng.next_f32()).collect();
        let x: Vec<f32> = (0..in_).map(|_| rng.next_f32()).collect();
        let r: Vec<f32> = (0..out).map(|_| rng.next_f32()).collect(); // dL/dy

        let grads = bm.backward(&dict, &coeffs, &x, &r);
        let loss = |c: &[f32], d: &[Complex32], xx: &[f32]| -> f32 {
            bm.forward(d, c, xx).iter().zip(&r).map(|(y, rr)| y * rr).sum()
        };
        let eps = 1e-3f32;
        let tol = 1e-2f32;

        for i in 0..coeffs.len() {
            let mut cp = coeffs.clone();
            cp[i] += eps;
            let lp = loss(&cp, &dict, &x);
            cp[i] -= 2.0 * eps;
            let lm = loss(&cp, &dict, &x);
            let fd = (lp - lm) / (2.0 * eps);
            assert!((fd - grads.d_coeffs[i]).abs() < tol, "dα[{i}] fd {fd} an {}", grads.d_coeffs[i]);
        }
        for i in 0..dict.len() {
            let mut dp = dict.clone();
            dp[i].re += eps;
            let lp = loss(&coeffs, &dp, &x);
            dp[i].re -= 2.0 * eps;
            let lm = loss(&coeffs, &dp, &x);
            let fd = (lp - lm) / (2.0 * eps);
            assert!((fd - grads.d_dict[i].re).abs() < tol, "dG.re[{i}] fd {fd} an {}", grads.d_dict[i].re);

            let mut dq = dict.clone();
            dq[i].im += eps;
            let lp = loss(&coeffs, &dq, &x);
            dq[i].im -= 2.0 * eps;
            let lm = loss(&coeffs, &dq, &x);
            let fd = (lp - lm) / (2.0 * eps);
            assert!((fd - grads.d_dict[i].im).abs() < tol, "dG.im[{i}] fd {fd} an {}", grads.d_dict[i].im);
        }
        for i in 0..x.len() {
            let mut xp = x.clone();
            xp[i] += eps;
            let lp = loss(&coeffs, &dict, &xp);
            xp[i] -= 2.0 * eps;
            let lm = loss(&coeffs, &dict, &xp);
            let fd = (lp - lm) / (2.0 * eps);
            assert!((fd - grads.d_x[i]).abs() < tol, "dx[{i}] fd {fd} an {}", grads.d_x[i]);
        }
    }

    #[test]
    fn compress_roundtrip() {
        // K = 2b makes the per-block fit exact, so compressed forward should
        // reproduce the dense matvec up to quantization error.
        let (out, in_, b) = (8usize, 8usize, 4usize);
        let k = 2 * b;
        let bm = BasisMatmul::new(out, in_, b, k);
        let mut rng = Lcg(0x0055_0055_0055_0055);
        let dict: Vec<Complex32> =
            (0..bm.dict_len()).map(|_| Complex32::new(rng.next_f32(), rng.next_f32())).collect();
        let alpha: Vec<f32> = (0..bm.coeff_len()).map(|_| rng.next_f32()).collect();

        let w = bm.to_dense(&dict, &alpha);
        let cw = bm.compress_dense(&dict, &w, 1e-3, k);
        let x: Vec<f32> = (0..in_).map(|_| rng.next_f32()).collect();
        let yc = bm.forward_compressed(&dict, &cw, &x);

        let mut yd = vec![0.0f32; out];
        for i in 0..out {
            let mut s = 0.0f32;
            for j in 0..in_ {
                s += w[i * in_ + j] * x[j];
            }
            yd[i] = s;
        }
        for i in 0..out {
            assert!((yc[i] - yd[i]).abs() < 5e-2, "compressed[{i}] {} vs dense {}", yc[i], yd[i]);
        }
    }

    #[test]
    fn forward_rows_matches_full_on_active_blocks() {
        // Selected output blocks equal the full forward; others are zero.
        let (out, in_, b, k) = (24usize, 8usize, 4usize, 5usize); // P=6, Q=2
        let bm = BasisMatmul::new(out, in_, b, k);
        let mut rng = Lcg(0x0F0F_1234);
        let dict: Vec<Complex32> =
            (0..bm.dict_len()).map(|_| Complex32::new(rng.next_f32(), rng.next_f32())).collect();
        let coeffs: Vec<f32> = (0..bm.coeff_len()).map(|_| rng.next_f32()).collect();
        let x: Vec<f32> = (0..in_).map(|_| rng.next_f32()).collect();

        let full = bm.forward(&dict, &coeffs, &x);
        let active = [1usize, 3, 4];
        let rows = bm.forward_rows(&dict, &coeffs, &x, &active);

        for pp in 0..bm.p {
            let blk = pp * b..(pp + 1) * b;
            if active.contains(&pp) {
                for f in blk.clone() {
                    assert!((rows[f] - full[f]).abs() < 1e-4, "active block {pp} differs at {f}");
                }
            } else {
                for f in blk {
                    assert!(rows[f].abs() < 1e-6, "inactive block {pp} not zero at {f}");
                }
            }
        }
    }

    #[test]
    fn forward_cols_matches_zeroed_input() {
        // Summing only active input blocks == full forward on an input whose
        // inactive blocks are zeroed (the routed-down-projection invariant).
        let (out, in_, b, k) = (8usize, 24usize, 4usize, 5usize); // P=2, Q=6
        let bm = BasisMatmul::new(out, in_, b, k);
        let mut rng = Lcg(0x7777_2222);
        let dict: Vec<Complex32> =
            (0..bm.dict_len()).map(|_| Complex32::new(rng.next_f32(), rng.next_f32())).collect();
        let coeffs: Vec<f32> = (0..bm.coeff_len()).map(|_| rng.next_f32()).collect();

        let active = [0usize, 2, 5];
        let mut x = vec![0.0f32; in_];
        for &qq in &active {
            for f in qq * b..(qq + 1) * b {
                x[f] = rng.next_f32();
            }
        }

        let full = bm.forward(&dict, &coeffs, &x);
        let cols = bm.forward_cols(&dict, &coeffs, &x, &active);
        for i in 0..out {
            assert!((cols[i] - full[i]).abs() < 1e-4, "cols[{i}] {} vs full {}", cols[i], full[i]);
        }
    }

    #[test]
    fn masked_backwards_reduce_to_full_when_all_active() {
        // With every block active, both masked backwards must equal `backward`.
        let (out, in_, b, k) = (12usize, 8usize, 4usize, 5usize); // P=3, Q=2
        let bm = BasisMatmul::new(out, in_, b, k);
        let mut rng = Lcg(0xABCD_0001);
        let dict: Vec<Complex32> =
            (0..bm.dict_len()).map(|_| Complex32::new(rng.next_f32(), rng.next_f32())).collect();
        let coeffs: Vec<f32> = (0..bm.coeff_len()).map(|_| rng.next_f32()).collect();
        let x: Vec<f32> = (0..in_).map(|_| rng.next_f32()).collect();
        let dy: Vec<f32> = (0..out).map(|_| rng.next_f32()).collect();

        let full = bm.backward(&dict, &coeffs, &x, &dy);
        let all_p: Vec<usize> = (0..bm.p).collect();
        let all_q: Vec<usize> = (0..bm.q).collect();
        let rows = bm.backward_rows(&dict, &coeffs, &x, &dy, &all_p);
        let cols = bm.backward_cols(&dict, &coeffs, &x, &dy, &all_q);

        for i in 0..full.d_coeffs.len() {
            assert!((rows.d_coeffs[i] - full.d_coeffs[i]).abs() < 1e-4, "rows d_coeffs[{i}]");
            assert!((cols.d_coeffs[i] - full.d_coeffs[i]).abs() < 1e-4, "cols d_coeffs[{i}]");
        }
        for i in 0..full.d_dict.len() {
            assert!((rows.d_dict[i] - full.d_dict[i]).norm() < 1e-4, "rows d_dict[{i}]");
            assert!((cols.d_dict[i] - full.d_dict[i]).norm() < 1e-4, "cols d_dict[{i}]");
        }
        for i in 0..full.d_x.len() {
            assert!((rows.d_x[i] - full.d_x[i]).abs() < 1e-4, "rows d_x[{i}]");
            assert!((cols.d_x[i] - full.d_x[i]).abs() < 1e-4, "cols d_x[{i}]");
        }
    }

    #[test]
    fn fused_pair_matches_separate_forward_and_backward() {
        // up+gate fusion must be numerically identical to two separate calls:
        // forward returns the same outputs; backward returns the same per-
        // projection coeff grads and the *summed* dict/input grads.
        let (out, in_, b, k) = (12usize, 8usize, 4usize, 5usize); // P=3, Q=2
        let bm = BasisMatmul::new(out, in_, b, k);
        let mut rng = Lcg(0x5151_AAAA);
        let dict: Vec<Complex32> =
            (0..bm.dict_len()).map(|_| Complex32::new(rng.next_f32(), rng.next_f32())).collect();
        let ca: Vec<f32> = (0..bm.coeff_len()).map(|_| rng.next_f32()).collect();
        let cb: Vec<f32> = (0..bm.coeff_len()).map(|_| rng.next_f32()).collect();
        let x: Vec<f32> = (0..in_).map(|_| rng.next_f32()).collect();
        let dya: Vec<f32> = (0..out).map(|_| rng.next_f32()).collect();
        let dyb: Vec<f32> = (0..out).map(|_| rng.next_f32()).collect();
        let active = [0usize, 2];

        let (ya, yb) = bm.forward_rows_pair(&dict, &ca, &cb, &x, &active);
        let ra = bm.forward_rows(&dict, &ca, &x, &active);
        let rb = bm.forward_rows(&dict, &cb, &x, &active);
        for i in 0..out {
            assert!((ya[i] - ra[i]).abs() < 1e-5, "pair fwd a[{i}]");
            assert!((yb[i] - rb[i]).abs() < 1e-5, "pair fwd b[{i}]");
        }

        let pg = bm.backward_rows_pair(&dict, &ca, &cb, &x, &dya, &dyb, &active);
        let ga = bm.backward_rows(&dict, &ca, &x, &dya, &active);
        let gb = bm.backward_rows(&dict, &cb, &x, &dyb, &active);
        for i in 0..bm.coeff_len() {
            assert!((pg.d_coeffs_a[i] - ga.d_coeffs[i]).abs() < 1e-5, "pair d_coeffs_a[{i}]");
            assert!((pg.d_coeffs_b[i] - gb.d_coeffs[i]).abs() < 1e-5, "pair d_coeffs_b[{i}]");
        }
        for i in 0..bm.dict_len() {
            assert!((pg.d_dict[i] - (ga.d_dict[i] + gb.d_dict[i])).norm() < 1e-5, "pair d_dict[{i}]");
        }
        for i in 0..in_ {
            assert!((pg.d_x[i] - (ga.d_x[i] + gb.d_x[i])).abs() < 1e-5, "pair d_x[{i}]");
        }
    }
}
