//! Gate (c): SharedMonarchMatmul — P×Q tiling of b×b Monarch blocks with shared
//! atom dictionaries — wired as a full projection and benchmarked vs BasisMatmul.
//!
//! Three checks:
//!   1. Gradcheck (FD vs analytical, small dims)
//!   2. Timing vs BasisMatmul at FFN 3072×896 dims (P=14, Q=48, b=64)
//!   3. Same-family overfit with cosine-decayed Adam at nd=8

use std::time::Instant;

use fydel::kernels::fft::BasisMatmul;
use fydel::kernels::gemm;
use rustfft::num_complex::Complex32;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// ---------------------------------------------------------------------------
// LCG rng
// ---------------------------------------------------------------------------

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f()).collect()
    }
    fn vec_scaled(&mut self, n: usize, s: f32) -> Vec<f32> {
        (0..n).map(|_| self.f() * s).collect()
    }
}

// ---------------------------------------------------------------------------
// AVX2 kernels for forward_block (m=8 specialisation)
// ---------------------------------------------------------------------------

/// dst[0..64] = alpha * src[0..64], 8 YMM multiply lanes (no accumulate — replaces fill+axpy).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn axpy64_init_avx2(dst: *mut f32, src: *const f32, alpha: f32) {
    let av = _mm256_set1_ps(alpha);
    for i in 0..8 {
        let s = _mm256_loadu_ps(src.add(i * 8));
        _mm256_storeu_ps(dst.add(i * 8), _mm256_mul_ps(av, s));
    }
}

/// dst[0..64] += alpha * src[0..64], 8 YMM FMA lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn axpy64_avx2(dst: *mut f32, src: *const f32, alpha: f32) {
    let av = _mm256_set1_ps(alpha);
    for i in 0..8 {
        let s = _mm256_loadu_ps(src.add(i * 8));
        let d = _mm256_loadu_ps(dst.add(i * 8));
        _mm256_storeu_ps(dst.add(i * 8), _mm256_fmadd_ps(av, s, d));
    }
}

/// out[0..8] = mat[0..64] @ vec[0..8], row-major mat, 2-rows-per-hadd-sequence.
/// h0 = vhaddps(p0,p1): lo=[a01,a23,b01,b23], hi=[a45,a67,b45,b67]
/// h1 = vhaddps(h0,h0): lo=[a0123,b0123,...], hi=[a4567,b4567,...]
/// add(lo128, hi128) → [a_sum, b_sum, ...] in lanes 0, 1.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn matvec8_avx2(out: *mut f32, mat: *const f32, vec: *const f32) {
    let v = _mm256_loadu_ps(vec);
    for pair in 0..4 {
        let r0 = pair * 2;
        let p0 = _mm256_mul_ps(_mm256_loadu_ps(mat.add(r0 * 8)), v);
        let p1 = _mm256_mul_ps(_mm256_loadu_ps(mat.add((r0 + 1) * 8)), v);
        let h0 = _mm256_hadd_ps(p0, p1);
        let h1 = _mm256_hadd_ps(h0, h0);
        let lo = _mm256_castps256_ps128(h1);
        let hi = _mm256_extractf128_ps(h1, 1);
        let s  = _mm_add_ps(lo, hi);
        *out.add(r0)     = _mm_cvtss_f32(s);
        *out.add(r0 + 1) = _mm_cvtss_f32(_mm_shuffle_ps(s, s, 0x01));
    }
}

/// out[0..8] += mat[0..64] @ vec[0..8] — accumulates rather than overwrites.
/// Used for the stage-2 output which sums contributions from all qq blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn matvec8_accum_avx2(out: *mut f32, mat: *const f32, vec: *const f32) {
    let v = _mm256_loadu_ps(vec);
    for pair in 0..4 {
        let r0 = pair * 2;
        let p0 = _mm256_mul_ps(_mm256_loadu_ps(mat.add(r0 * 8)), v);
        let p1 = _mm256_mul_ps(_mm256_loadu_ps(mat.add((r0 + 1) * 8)), v);
        let h0 = _mm256_hadd_ps(p0, p1);
        let h1 = _mm256_hadd_ps(h0, h0);
        let lo = _mm256_castps256_ps128(h1);
        let hi = _mm256_extractf128_ps(h1, 1);
        let s  = _mm_add_ps(lo, hi);
        *out.add(r0)     += _mm_cvtss_f32(s);
        *out.add(r0 + 1) += _mm_cvtss_f32(_mm_shuffle_ps(s, s, 0x01));
    }
}

/// forward_block for m=8, any nd. All inner loops have compile-time trip counts
/// Horizontal dot of two 8-element f32 vectors.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot8_avx2(a: *const f32, b: *const f32) -> f32 {
    let va   = _mm256_loadu_ps(a);
    let vb   = _mm256_loadu_ps(b);
    let prod = _mm256_mul_ps(va, vb);
    let lo   = _mm256_castps256_ps128(prod);
    let hi   = _mm256_extractf128_ps(prod, 1);
    let s4   = _mm_add_ps(lo, hi);
    let s2   = _mm_hadd_ps(s4, s4);
    _mm_cvtss_f32(_mm_hadd_ps(s2, s2))
}

/// dst[0..8] += alpha * src[0..8], single YMM FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn axpy8_avx2(dst: *mut f32, src: *const f32, alpha: f32) {
    let av = _mm256_set1_ps(alpha);
    _mm256_storeu_ps(dst, _mm256_fmadd_ps(av, _mm256_loadu_ps(src), _mm256_loadu_ps(dst)));
}

/// (B=64 for AXPY, M=8 for transpose) so LLVM can fully unroll alongside the
/// explicit AVX2 intrinsics.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn forward_block_avx2(
    d1: &[f32], d2: &[f32], a1_blk: &[f32], a2_blk: &[f32],
    x_blk: &[f32], nd: usize,
    y1: &mut [f32], z: &mut [f32], out: &mut [f32],
    eff: &mut [f32],
) {
    const M: usize = 8;
    const B: usize = 64;

    // Stage 1: init from first atom (avoids fill(0.0) + first axpy), then accumulate rest.
    for i in 0..M {
        axpy64_init_avx2(eff.as_mut_ptr(), d1.as_ptr().add(0), *a1_blk.get_unchecked(i * nd));
        for d in 1..nd {
            axpy64_avx2(eff.as_mut_ptr(), d1.as_ptr().add(d * B), *a1_blk.get_unchecked(i * nd + d));
        }
        matvec8_avx2(y1.as_mut_ptr().add(i * M), eff.as_ptr(), x_blk.as_ptr().add(i * M));
    }

    // Transpose y1 → z
    for i in 0..M {
        for j in 0..M {
            *z.get_unchecked_mut(j * M + i) = *y1.get_unchecked(i * M + j);
        }
    }

    // Stage 2: same init trick; output accumulates into out (= ypp, summed across qq).
    for j in 0..M {
        axpy64_init_avx2(eff.as_mut_ptr(), d2.as_ptr().add(0), *a2_blk.get_unchecked(j * nd));
        for d in 1..nd {
            axpy64_avx2(eff.as_mut_ptr(), d2.as_ptr().add(d * B), *a2_blk.get_unchecked(j * nd + d));
        }
        matvec8_accum_avx2(out.as_mut_ptr().add(j * M), eff.as_ptr(), z.as_ptr().add(j * M));
    }
}

/// Backward pass for m=8, any nd. Computes da1, da2, dd1, dd2.
/// dx (input grad) is omitted — add when wiring into full transformer.
/// dd1/dd2 are global accumulators; caller keeps backward sequential over blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn backward_block_avx2(
    d1: *const f32, d2: *const f32,
    a1_blk: *const f32, a2_blk: *const f32,
    x_blk: *const f32, z: *const f32,
    dout_pp: *const f32, // [M*M] grad into this pp row
    nd: usize,
    da1: *mut f32, da2: *mut f32, // [M*nd] each, offset to this block
    dd1: *mut f32, dd2: *mut f32, // [nd*B] each, global accumulators
) {
    const M: usize = 8;
    const B: usize = 64;
    let mut eff = [0.0f32; B];
    let mut dz  = [0.0f32; B];
    let mut dy1 = [0.0f32; B];

    // Stage 2 backward
    for j in 0..M {
        let zj     = z.add(j * M);
        let dout_j = dout_pp.add(j * M);

        // eff_j = Σ_d a2[j,d]*D2[d] (same as forward)
        axpy64_init_avx2(eff.as_mut_ptr(), d2.add(0), *a2_blk.add(j * nd));
        for d in 1..nd {
            axpy64_avx2(eff.as_mut_ptr(), d2.add(d * B), *a2_blk.add(j * nd + d));
        }

        // dz[j,:] = eff_j^T @ dout_j  via row-AXPY: dz_j += dout_j[r] * eff_j[r,:]
        let dz_j = dz.as_mut_ptr().add(j * M);
        for r in 0..M {
            axpy8_avx2(dz_j, eff.as_ptr().add(r * M), *dout_j.add(r));
        }

        // da2[j,d] += Σ_r dout_j[r] * dot8(D2[d,r,:], z_j)
        // dd2[d,r,:] += dout_j[r] * a2[j,d] * z_j
        for r in 0..M {
            let dy = *dout_j.add(r);
            for d in 0..nd {
                let a     = *a2_blk.add(j * nd + d);
                let d2row = d2.add((d * M + r) * M);
                *da2.add(j * nd + d) += dy * dot8_avx2(d2row, zj);
                axpy8_avx2(dd2.add((d * M + r) * M), zj, dy * a);
            }
        }
    }

    // Transpose dz → dy1
    for i in 0..M {
        for j in 0..M {
            dy1[i * M + j] = dz[j * M + i];
        }
    }

    // Stage 1 backward
    for i in 0..M {
        let xi    = x_blk.add(i * M);
        let dy1_i = dy1.as_ptr().add(i * M);

        axpy64_init_avx2(eff.as_mut_ptr(), d1.add(0), *a1_blk.add(i * nd));
        for d in 1..nd {
            axpy64_avx2(eff.as_mut_ptr(), d1.add(d * B), *a1_blk.add(i * nd + d));
        }

        for r in 0..M {
            let d_y = *dy1_i.add(r);
            for d in 0..nd {
                let a     = *a1_blk.add(i * nd + d);
                let d1row = d1.add((d * M + r) * M);
                *da1.add(i * nd + d) += d_y * dot8_avx2(d1row, xi);
                axpy8_avx2(dd1.add((d * M + r) * M), xi, d_y * a);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SharedMonarchMatmul
// ---------------------------------------------------------------------------
//
// Projects x ∈ R^{Q*b} → y ∈ R^{P*b} via a P×Q tiling of b×b Monarch blocks.
// Each b×b block uses a 2-stage block-diagonal GEMM (m = sqrt(b)):
//
//   stage-1 (m blocks of m×m): y1[i,r] = Σ_d a1[pp,qq,i,d] · D1[d,r,:] · x_i
//   transpose: z[j][i] = y1[i][j]
//   stage-2 (m blocks of m×m): out[j,r] = Σ_d a2[pp,qq,j,d] · D2[d,r,:] · z_j
//
// Atoms D1, D2 are shared across ALL (pp,qq) block pairs.
// Per-block coefficients a1[pp,qq,i,d] and a2[pp,qq,j,d] are learned.

struct SharedMonarchMatmul {
    p: usize,   // out_dim / b
    q: usize,   // in_dim  / b
    m: usize,   // sqrt(b)
    nd: usize,  // atoms in each shared dictionary

    d1: Vec<f32>, // [nd, m, m]  shared stage-1 atoms
    d2: Vec<f32>, // [nd, m, m]  shared stage-2 atoms
    a1: Vec<f32>, // [P, Q, m, nd]  per-block stage-1 coefficients
    a2: Vec<f32>, // [P, Q, m, nd]  per-block stage-2 coefficients
}

// Cache for backward: per (pp,qq) block, store y1 and z intermediates.
struct FwdCache {
    // [P*Q, b] each
    y1s: Vec<f32>,
    zs: Vec<f32>,
}

struct Grads {
    dd1: Vec<f32>, // [nd, m, m]  accumulated across all blocks
    dd2: Vec<f32>,
    da1: Vec<f32>, // [P, Q, m, nd]
    da2: Vec<f32>,
}

impl SharedMonarchMatmul {
    fn new(p: usize, q: usize, m: usize, nd: usize, rng: &mut Lcg) -> Self {
        let b = m * m;
        let s_atom = 1.0 / (m as f32).sqrt();
        // Atoms: O(1/sqrt(m)) so that a single atom application is O(1).
        // Coefficients: uniform in [-1, 1], small initial mix.
        let d1 = rng.vec_scaled(nd * b, 2.0 * s_atom);
        let d2 = rng.vec_scaled(nd * b, 2.0 * s_atom);
        let a1 = rng.vec_scaled(p * q * m * nd, 2.0 / (nd as f32).sqrt());
        let a2 = rng.vec_scaled(p * q * m * nd, 2.0 / (nd as f32).sqrt());
        Self { p, q, m, nd, d1, d2, a1, a2 }
    }

    #[inline]
    fn a1_blk(&self, pp: usize, qq: usize) -> &[f32] {
        let base = (pp * self.q + qq) * self.m * self.nd;
        &self.a1[base..base + self.m * self.nd]
    }

    #[inline]
    fn a2_blk(&self, pp: usize, qq: usize) -> &[f32] {
        let base = (pp * self.q + qq) * self.m * self.nd;
        &self.a2[base..base + self.m * self.nd]
    }

    fn forward_block(
        d1: &[f32], d2: &[f32], a1_blk: &[f32], a2_blk: &[f32],
        x_blk: &[f32], m: usize, nd: usize,
        y1: &mut [f32], z: &mut [f32], out: &mut [f32],
        eff: &mut [f32], // scratch m*m — avoids per-call allocation
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if m == 8 {
            unsafe {
                forward_block_avx2(d1, d2, a1_blk, a2_blk, x_blk, nd, y1, z, out, eff);
            }
            return;
        }
        let b = m * m;
        // Stage 1: build effective m×m matrix per block-row via AXPY, then apply.
        // All loops are scalar with fixed trip counts — LLVM auto-vectorizes both
        // the AXPY accumulation and the m×m mat-vec without function call boundaries.
        for i in 0..m {
            let xi = &x_blk[i * m..(i + 1) * m];
            eff[..b].fill(0.0);
            for d in 0..nd {
                let a = a1_blk[i * nd + d];
                let atom = &d1[d * b..d * b + b];
                for e in 0..b { eff[e] += a * atom[e]; }
            }
            for r in 0..m {
                let mut acc = 0.0f32;
                for c in 0..m { acc += eff[r * m + c] * xi[c]; }
                y1[i * m + r] = acc;
            }
        }
        // Transpose y1 → z
        for i in 0..m {
            for j in 0..m {
                z[j * m + i] = y1[i * m + j];
            }
        }
        // Stage 2: same pattern over z
        for j in 0..m {
            let zj = &z[j * m..(j + 1) * m];
            eff[..b].fill(0.0);
            for d in 0..nd {
                let a = a2_blk[j * nd + d];
                let atom = &d2[d * b..d * b + b];
                for e in 0..b { eff[e] += a * atom[e]; }
            }
            for r in 0..m {
                let mut acc = 0.0f32;
                for c in 0..m { acc += eff[r * m + c] * zj[c]; }
                out[j * m + r] += acc;  // accumulate: caller passes ypp (pre-zeroed)
            }
        }
    }

    fn forward(&self, x: &[f32]) -> (Vec<f32>, FwdCache) {
        use rayon::prelude::*;
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;

        let mut y   = vec![0.0f32; p * b];
        let mut y1s = vec![0.0f32; p * q * b];
        let mut zs  = vec![0.0f32; p * q * b];

        // Collect disjoint per-pp mutable slice triples, then parallelize.
        // Each pp writes to y[pp*b..(pp+1)*b] and cache rows [pp*q*b..(pp+1)*q*b] only.
        let pp_data: Vec<(&mut [f32], &mut [f32], &mut [f32])> = y.chunks_mut(b)
            .zip(y1s.chunks_mut(q * b))
            .zip(zs.chunks_mut(q * b))
            .map(|((ypp, y1pp), zpp)| (ypp, y1pp, zpp))
            .collect();

        pp_data.into_par_iter().enumerate().for_each(|(pp, (ypp, y1s_pp, zs_pp))| {
            let mut eff = vec![0.0f32; b];
            for qq in 0..q {
                let y1 = &mut y1s_pp[qq * b..(qq + 1) * b];
                let z  = &mut zs_pp[qq * b..(qq + 1) * b];
                Self::forward_block(
                    &self.d1, &self.d2,
                    self.a1_blk(pp, qq), self.a2_blk(pp, qq),
                    &x[qq * b..(qq + 1) * b],
                    m, nd, y1, z, ypp, &mut eff,
                );
            }
        });

        (y, FwdCache { y1s, zs })
    }

    // Returns (dd1, dd2, da1, da2). Input grad skipped (not needed for isolated proj test).
    fn backward(&self, x: &[f32], cache: &FwdCache, dout: &[f32]) -> Grads {
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;

        let mut g = Grads {
            dd1: vec![0.0f32; nd * b],
            dd2: vec![0.0f32; nd * b],
            da1: vec![0.0f32; p * q * m * nd],
            da2: vec![0.0f32; p * q * m * nd],
        };

        let mut dz    = vec![0.0f32; b];
        let mut dy1   = vec![0.0f32; b];
        let mut eff_j = vec![0.0f32; b];
        let mut eff_i = vec![0.0f32; b];

        for pp in 0..p {
            let dout_pp = &dout[pp * b..(pp + 1) * b];
            for qq in 0..q {
                let bk = pp * q + qq;
                let z  = &cache.zs[bk * b..(bk + 1) * b];
                let y1 = &cache.y1s[bk * b..(bk + 1) * b];
                let x_blk = &x[qq * b..(qq + 1) * b];

                let da2_blk_base = bk * m * nd;
                let da1_blk_base = bk * m * nd;

                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                if m == 8 {
                    unsafe {
                        let dd1_ptr = g.dd1.as_mut_ptr();
                        let dd2_ptr = g.dd2.as_mut_ptr();
                        let da1_ptr = g.da1.as_mut_ptr().add(da1_blk_base);
                        let da2_ptr = g.da2.as_mut_ptr().add(da2_blk_base);
                        backward_block_avx2(
                            self.d1.as_ptr(), self.d2.as_ptr(),
                            self.a1_blk(pp, qq).as_ptr(), self.a2_blk(pp, qq).as_ptr(),
                            x_blk.as_ptr(), z.as_ptr(),
                            dout_pp.as_ptr(), nd,
                            da1_ptr, da2_ptr, dd1_ptr, dd2_ptr,
                        );
                    }
                    continue;
                }

                // Stage 2 backward → da2, dd2, dz.
                // For each block-row j: effective eff_j = Σ_d a2[j,d]*D2[d] (same as fwd).
                // dz[j,:] += eff_j^T @ dout[j,:]
                // da2[j,d] += dout[j,:] · (D2[d] @ z_j) = dot(dout_j, D2[d] @ z_j)
                //           = dot(D2[d]^T @ dout_j, z_j)   (rearranged for efficiency)
                // dd2[d,r,:] += dout[j,r] * a2[j,d] * z_j
                dz.fill(0.0);
                for j in 0..m {
                    let zj = &z[j * m..(j + 1) * m];
                    let dout_j = &dout_pp[j * m..(j + 1) * m];
                    // Build effective matrix for this block-row
                    eff_j.fill(0.0);
                    for d in 0..nd {
                        let a = self.a2_blk(pp, qq)[j * nd + d];
                        let atom = &self.d2[d * b..d * b + b];
                        for e in 0..b { eff_j[e] += a * atom[e]; }
                    }
                    // dz[j,:] += eff_j^T @ dout_j  (inline, no call boundary)
                    for r in 0..m {
                        let dy = dout_j[r];
                        for c in 0..m { dz[j * m + c] += eff_j[r * m + c] * dy; }
                    }
                    // da2 and dd2 via outer product dout_j ⊗ z_j, weighted by atom rows
                    for r in 0..m {
                        let dy = dout_j[r];
                        if dy == 0.0 { continue; }
                        for d in 0..nd {
                            let a = self.a2_blk(pp, qq)[j * nd + d];
                            let drow = &self.d2[(d * m + r) * m..(d * m + r) * m + m];
                            let u = gemm::dot(drow, zj);
                            g.da2[da2_blk_base + j * nd + d] += dy * u;
                            let dd2row = &mut g.dd2[(d * m + r) * m..(d * m + r) * m + m];
                            for c in 0..m { dd2row[c] += dy * a * zj[c]; }
                        }
                    }
                }

                // Transpose dz → dy1
                dy1.fill(0.0);
                for j in 0..m {
                    for i in 0..m {
                        dy1[i * m + j] = dz[j * m + i];
                    }
                }

                // Stage 1 backward → da1, dd1 (same pattern as stage 2)
                for i in 0..m {
                    let xi = &x_blk[i * m..(i + 1) * m];
                    let dy1_i = &dy1[i * m..(i + 1) * m];
                    eff_i.fill(0.0);
                    for d in 0..nd {
                        let a = self.a1_blk(pp, qq)[i * nd + d];
                        let atom = &self.d1[d * b..d * b + b];
                        for e in 0..b { eff_i[e] += a * atom[e]; }
                    }
                    // No dx needed (isolated proj test), but da1 and dd1:
                    for r in 0..m {
                        let d_y = dy1_i[r];
                        if d_y == 0.0 { continue; }
                        for d in 0..nd {
                            let a = self.a1_blk(pp, qq)[i * nd + d];
                            let drow = &self.d1[(d * m + r) * m..(d * m + r) * m + m];
                            let u = gemm::dot(drow, xi);
                            g.da1[da1_blk_base + i * nd + d] += d_y * u;
                            let dd1row = &mut g.dd1[(d * m + r) * m..(d * m + r) * m + m];
                            for c in 0..m { dd1row[c] += d_y * a * xi[c]; }
                        }
                    }
                }
            }
        }
        g
    }
}

// ---------------------------------------------------------------------------
// Gradcheck (f64 finite differences, small dims)
// ---------------------------------------------------------------------------

fn mse_loss(out: &[f32], target: &[f32]) -> f32 {
    out.iter().zip(target).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / out.len() as f32
}

fn mse_grad(out: &[f32], target: &[f32]) -> Vec<f32> {
    let n = out.len() as f32;
    out.iter().zip(target).map(|(a, b)| 2.0 * (a - b) / n).collect()
}

fn gradcheck(p: usize, q: usize, m: usize, nd: usize, rng: &mut Lcg) -> bool {
    let b = m * m;
    let in_dim = q * b;
    let out_dim = p * b;

    let mut mm = SharedMonarchMatmul::new(p, q, m, nd, rng);
    let x: Vec<f32> = rng.vec(in_dim);
    let target: Vec<f32> = rng.vec(out_dim);

    let eps = 1e-3f32;
    let (out, cache) = mm.forward(&x);
    let dloss = mse_grad(&out, &target);
    let grads = mm.backward(&x, &cache, &dloss);

    let mut max_err = 0.0f32;
    let mut checked = 0usize;

    // Check da1 (a sample)
    for idx in (0..mm.a1.len()).step_by(mm.a1.len() / 16 + 1).take(16) {
        let orig = mm.a1[idx];
        mm.a1[idx] = orig + eps;
        let (out_p, _) = mm.forward(&x);
        mm.a1[idx] = orig - eps;
        let (out_m, _) = mm.forward(&x);
        mm.a1[idx] = orig;
        let fd = (mse_loss(&out_p, &target) - mse_loss(&out_m, &target)) / (2.0 * eps);
        max_err = max_err.max((fd - grads.da1[idx]).abs());
        checked += 1;
    }

    // Check dd1
    for idx in (0..mm.d1.len()).step_by(mm.d1.len() / 8 + 1).take(8) {
        let orig = mm.d1[idx];
        mm.d1[idx] = orig + eps;
        let (out_p, _) = mm.forward(&x);
        mm.d1[idx] = orig - eps;
        let (out_m, _) = mm.forward(&x);
        mm.d1[idx] = orig;
        let fd = (mse_loss(&out_p, &target) - mse_loss(&out_m, &target)) / (2.0 * eps);
        max_err = max_err.max((fd - grads.dd1[idx]).abs());
        checked += 1;
    }

    eprintln!("  gradcheck ({p}x{q} blocks, m={m}, nd={nd}): max_err={:.2e}  [{checked} params]  {}",
        max_err, if max_err < 0.05 { "PASS" } else { "FAIL" });
    max_err < 0.05
}

// ---------------------------------------------------------------------------
// Benchmark vs BasisMatmul
// ---------------------------------------------------------------------------

fn bench(out_dim: usize, in_dim: usize, b: usize, nd: usize, k: usize, iters: usize, rng: &mut Lcg) {
    let m = (b as f64).sqrt() as usize;
    assert_eq!(m * m, b, "b must be a perfect square");
    let p = out_dim / b;
    let q = in_dim / b;

    // SharedMonarchMatmul
    let mm = SharedMonarchMatmul::new(p, q, m, nd, rng);
    let x: Vec<f32> = rng.vec(in_dim);

    for _ in 0..200 {
        let _ = std::hint::black_box(mm.forward(&x));
    }
    let t = Instant::now();
    for _ in 0..iters {
        let _ = std::hint::black_box(mm.forward(&x));
    }
    let mon_us = t.elapsed().as_secs_f64() / iters as f64 * 1e6;

    // backward timing
    let dout: Vec<f32> = rng.vec(out_dim);
    let (_, cache) = mm.forward(&x);
    for _ in 0..200 { let _ = std::hint::black_box(mm.backward(&x, &cache, &dout)); }
    let t = Instant::now();
    for _ in 0..iters { let _ = std::hint::black_box(mm.backward(&x, &cache, &dout)); }
    let mon_bwd_us = t.elapsed().as_secs_f64() / iters as f64 * 1e6;

    // BasisMatmul baseline
    let basis = BasisMatmul::new(out_dim, in_dim, b, k);
    let n_dict = k * b;
    let dict: Vec<Complex32> = (0..n_dict).map(|_| {
        Complex32::new(rng.f(), rng.f())
    }).collect();
    let coeffs: Vec<f32> = (0..basis.coeff_len()).map(|_| rng.f() * 0.1).collect();

    for _ in 0..200 {
        let _ = std::hint::black_box(basis.forward(&dict, &coeffs, &x));
    }
    let t = Instant::now();
    for _ in 0..iters {
        let _ = std::hint::black_box(basis.forward(&dict, &coeffs, &x));
    }
    let basis_us = t.elapsed().as_secs_f64() / iters as f64 * 1e6;

    let mon_params = nd * b * 2 + p * q * m * nd * 2;
    let basis_params = k * b * 2 + p * q * k; // complex dict counts as 2 reals
    eprintln!(
        "  {out_dim}x{in_dim}  SharedMonarch(nd={nd}): fwd={:>7.2}µs  bwd={:>7.2}µs  BasisMatmul(K={k}): {:>7.2}µs  speedup={:.1}×",
        mon_us, mon_bwd_us, basis_us, basis_us / mon_us
    );
}

// ---------------------------------------------------------------------------
// Cosine-decayed Adam
// ---------------------------------------------------------------------------

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
    b1: f32,
    b2: f32,
    eps: f32,
}

impl Adam {
    fn new(n: usize) -> Self {
        Self { m: vec![0.0; n], v: vec![0.0; n], t: 0, b1: 0.9, b2: 0.999, eps: 1e-8 }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        self.t += 1;
        let (b1, b2, eps) = (self.b1, self.b2, self.eps);
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        for i in 0..params.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * grads[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * grads[i] * grads[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

fn cosine_lr(step: usize, lr_max: f32, t_max: usize) -> f32 {
    lr_max * 0.5 * (1.0 + (std::f32::consts::PI * step as f32 / t_max as f32).cos())
}

// ---------------------------------------------------------------------------
// Training: same-family overfit
// ---------------------------------------------------------------------------

fn train(p: usize, q: usize, m: usize, nd: usize, rng: &mut Lcg) {
    let b = m * m;
    let in_dim = q * b;

    // Teacher: fixed random SharedMonarchMatmul
    let teacher = SharedMonarchMatmul::new(p, q, m, nd, rng);
    let x: Vec<f32> = rng.vec(in_dim);
    let (target, _) = teacher.forward(&x);

    // Student: fresh init
    let mut student = SharedMonarchMatmul::new(p, q, m, nd, rng);

    let n_d1 = nd * b;
    let n_d2 = nd * b;
    let n_a1 = p * q * m * nd;
    let n_a2 = p * q * m * nd;

    let mut opt_d1 = Adam::new(n_d1);
    let mut opt_d2 = Adam::new(n_d2);
    let mut opt_a1 = Adam::new(n_a1);
    let mut opt_a2 = Adam::new(n_a2);

    let t_max = 2000usize;
    let lr_max = 1e-3f32;
    let print_at = [1, 100, 500, 1000, 2000];

    eprintln!("  training (p={p}, q={q}, m={m}, nd={nd}, steps={t_max}, lr_max={lr_max}):");
    for step in 1..=t_max {
        let (out, cache) = student.forward(&x);
        let loss = mse_loss(&out, &target);
        if print_at.contains(&step) {
            eprintln!("    step {:>5}  loss={:.6}", step, loss);
        }
        let dloss = mse_grad(&out, &target);
        let g = student.backward(&x, &cache, &dloss);
        let lr = cosine_lr(step, lr_max, t_max);
        opt_d1.step(&mut student.d1, &g.dd1, lr);
        opt_d2.step(&mut student.d2, &g.dd2, lr);
        opt_a1.step(&mut student.a1, &g.da1, lr);
        opt_a2.step(&mut student.a2, &g.da2, lr);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let mut rng = Lcg(0xFEED_BEEF_1234_5678);

    // --- Gradcheck (small dims) ---
    eprintln!("=== Gradcheck ===");
    let pass1 = gradcheck(2, 2, 4, 4, &mut rng);
    let pass2 = gradcheck(3, 2, 4, 8, &mut rng);
    let pass3 = gradcheck(2, 3, 8, 8, &mut rng);
    let all_pass = pass1 && pass2 && pass3;
    eprintln!("  overall: {}", if all_pass { "PASS" } else { "FAIL" });

    if !all_pass {
        eprintln!("Gradcheck failed — aborting.");
        std::process::exit(1);
    }

    // --- Timing vs BasisMatmul ---
    // FFN down: 896×3072 (P=14, Q=48, b=64)
    // nd=8 for training reliability; K=32 for BasisMatmul to match production.
    let iters = 5000;
    eprintln!("\n=== Timing (iters={iters}) ===");
    bench(896, 3072, 64, 8, 32, iters, &mut rng);
    bench(896, 896,  64, 8, 32, iters, &mut rng);

    // --- Training ---
    // Small dims to keep training fast; proves loss descent.
    eprintln!("\n=== Training (same-family overfit, nd=8) ===");
    train(4, 4, 8, 8, &mut rng);

    // Larger dims to show it scales.
    eprintln!("\n=== Training (scaled, nd=8) ===");
    train(6, 4, 8, 8, &mut rng);
}
