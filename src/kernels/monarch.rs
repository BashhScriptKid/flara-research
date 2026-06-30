//! SharedMonarchMatmul — P×Q tiling of b×b Monarch blocks with shared atom dictionaries.
#![allow(unsafe_op_in_unsafe_fn)]
//!
//! Each b×b block uses a 2-stage block-diagonal GEMM (m = sqrt(b)):
//!   stage-1 (m blocks of m×m): y1[i,r] = Σ_d a1[pp,qq,i,d] · D1[d,r,:] · x_i
//!   transpose: z[j][i] = y1[i][j]
//!   stage-2 (m blocks of m×m): out[j,r] = Σ_d a2[pp,qq,j,d] · D2[d,r,:] · z_j
//!
//! Atoms D1, D2 shared across all (pp,qq) block pairs.
//! Per-block coefficients a1/a2 are learned; shared atoms reduce parameter count
//! while maintaining full-rank expressibility at nd ≥ 8.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::kernels::gemm;

// ---------------------------------------------------------------------------
// AVX2 kernels (m=8 specialisation)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn axpy64_init(dst: *mut f32, src: *const f32, alpha: f32) {
    let av = _mm256_set1_ps(alpha);
    for i in 0..8 {
        _mm256_storeu_ps(dst.add(i * 8), _mm256_mul_ps(av, _mm256_loadu_ps(src.add(i * 8))));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn axpy64(dst: *mut f32, src: *const f32, alpha: f32) {
    let av = _mm256_set1_ps(alpha);
    for i in 0..8 {
        _mm256_storeu_ps(dst.add(i * 8),
            _mm256_fmadd_ps(av, _mm256_loadu_ps(src.add(i * 8)), _mm256_loadu_ps(dst.add(i * 8))));
    }
}

/// out[0..8] = mat[0..64] @ vec[0..8]. Pair-wise hadd: 2 rows per hadd sequence.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn matvec8(out: *mut f32, mat: *const f32, vec: *const f32) {
    let v = _mm256_loadu_ps(vec);
    for pair in 0..4 {
        let r0 = pair * 2;
        let p0 = _mm256_mul_ps(_mm256_loadu_ps(mat.add(r0 * 8)), v);
        let p1 = _mm256_mul_ps(_mm256_loadu_ps(mat.add((r0 + 1) * 8)), v);
        let h0 = _mm256_hadd_ps(p0, p1);
        let h1 = _mm256_hadd_ps(h0, h0);
        let s  = _mm_add_ps(_mm256_castps256_ps128(h1), _mm256_extractf128_ps(h1, 1));
        *out.add(r0)     = _mm_cvtss_f32(s);
        *out.add(r0 + 1) = _mm_cvtss_f32(_mm_shuffle_ps(s, s, 0x01));
    }
}

/// out[0..8] += mat[0..64] @ vec[0..8]. Accumulates into out.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn matvec8_accum(out: *mut f32, mat: *const f32, vec: *const f32) {
    let v = _mm256_loadu_ps(vec);
    for pair in 0..4 {
        let r0 = pair * 2;
        let p0 = _mm256_mul_ps(_mm256_loadu_ps(mat.add(r0 * 8)), v);
        let p1 = _mm256_mul_ps(_mm256_loadu_ps(mat.add((r0 + 1) * 8)), v);
        let h0 = _mm256_hadd_ps(p0, p1);
        let h1 = _mm256_hadd_ps(h0, h0);
        let s  = _mm_add_ps(_mm256_castps256_ps128(h1), _mm256_extractf128_ps(h1, 1));
        *out.add(r0)     += _mm_cvtss_f32(s);
        *out.add(r0 + 1) += _mm_cvtss_f32(_mm_shuffle_ps(s, s, 0x01));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot8(a: *const f32, b: *const f32) -> f32 {
    let prod = _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
    let s4   = _mm_add_ps(_mm256_castps256_ps128(prod), _mm256_extractf128_ps(prod, 1));
    let s2   = _mm_hadd_ps(s4, s4);
    _mm_cvtss_f32(_mm_hadd_ps(s2, s2))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn axpy8(dst: *mut f32, src: *const f32, alpha: f32) {
    let av = _mm256_set1_ps(alpha);
    _mm256_storeu_ps(dst, _mm256_fmadd_ps(av, _mm256_loadu_ps(src), _mm256_loadu_ps(dst)));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fwd_block_avx2(
    d1: &[f32], d2: &[f32], a1_blk: &[f32], a2_blk: &[f32],
    x_blk: &[f32], nd: usize,
    y1: &mut [f32], z: &mut [f32], out: &mut [f32], eff: &mut [f32],
) {
    const M: usize = 8;
    const B: usize = 64;
    for i in 0..M {
        axpy64_init(eff.as_mut_ptr(), d1.as_ptr(),            *a1_blk.get_unchecked(i * nd));
        for d in 1..nd {
            axpy64(eff.as_mut_ptr(), d1.as_ptr().add(d * B), *a1_blk.get_unchecked(i * nd + d));
        }
        matvec8(y1.as_mut_ptr().add(i * M), eff.as_ptr(), x_blk.as_ptr().add(i * M));
    }
    for i in 0..M { for j in 0..M { *z.get_unchecked_mut(j*M+i) = *y1.get_unchecked(i*M+j); } }
    for j in 0..M {
        axpy64_init(eff.as_mut_ptr(), d2.as_ptr(),            *a2_blk.get_unchecked(j * nd));
        for d in 1..nd {
            axpy64(eff.as_mut_ptr(), d2.as_ptr().add(d * B), *a2_blk.get_unchecked(j * nd + d));
        }
        matvec8_accum(out.as_mut_ptr().add(j * M), eff.as_ptr(), z.as_ptr().add(j * M));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bwd_block_avx2(
    d1: *const f32, d2: *const f32,
    a1_blk: *const f32, a2_blk: *const f32,
    x_blk: *const f32, z: *const f32,
    dout_pp: *const f32, nd: usize,
    da1: *mut f32, da2: *mut f32,
    dd1: *mut f32, dd2: *mut f32,
) {
    const M: usize = 8;
    const B: usize = 64;
    let mut eff = [0.0f32; B];
    let mut dz  = [0.0f32; B];
    let mut dy1 = [0.0f32; B];

    for j in 0..M {
        let zj     = z.add(j * M);
        let dout_j = dout_pp.add(j * M);
        axpy64_init(eff.as_mut_ptr(), d2, *a2_blk.add(j * nd));
        for d in 1..nd { axpy64(eff.as_mut_ptr(), d2.add(d * B), *a2_blk.add(j * nd + d)); }
        let dz_j = dz.as_mut_ptr().add(j * M);
        for r in 0..M { axpy8(dz_j, eff.as_ptr().add(r * M), *dout_j.add(r)); }
        for r in 0..M {
            let dy = *dout_j.add(r);
            for d in 0..nd {
                let a = *a2_blk.add(j * nd + d);
                *da2.add(j * nd + d) += dy * dot8(d2.add((d * M + r) * M), zj);
                axpy8(dd2.add((d * M + r) * M), zj, dy * a);
            }
        }
    }
    for i in 0..M { for j in 0..M { dy1[i*M+j] = dz[j*M+i]; } }
    for i in 0..M {
        let xi    = x_blk.add(i * M);
        let dy1_i = dy1.as_ptr().add(i * M);
        axpy64_init(eff.as_mut_ptr(), d1, *a1_blk.add(i * nd));
        for d in 1..nd { axpy64(eff.as_mut_ptr(), d1.add(d * B), *a1_blk.add(i * nd + d)); }
        for r in 0..M {
            let d_y = *dy1_i.add(r);
            for d in 0..nd {
                let a = *a1_blk.add(i * nd + d);
                *da1.add(i * nd + d) += d_y * dot8(d1.add((d * M + r) * M), xi);
                axpy8(dd1.add((d * M + r) * M), xi, d_y * a);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

pub struct SharedMonarchMatmul {
    pub p: usize,
    pub q: usize,
    pub m: usize,
    pub nd: usize,
    pub d1: Vec<f32>,
    pub d2: Vec<f32>,
    pub a1: Vec<f32>,
    pub a2: Vec<f32>,
}

pub struct FwdCache {
    pub y1s: Vec<f32>,
    pub zs:  Vec<f32>,
}

pub struct Grads {
    pub dd1: Vec<f32>,
    pub dd2: Vec<f32>,
    pub da1: Vec<f32>,
    pub da2: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Constructor helpers
// ---------------------------------------------------------------------------

fn lcg_next(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

fn randn_vec(n: usize, scale: f32, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| lcg_next(&mut s) * scale).collect()
}

// ---------------------------------------------------------------------------
// impl
// ---------------------------------------------------------------------------

impl SharedMonarchMatmul {
    /// Construct a new projection with LCG-initialised weights.
    /// Atoms ~ U[-1/√m, 1/√m], coefficients ~ U[-1/√nd, 1/√nd].
    pub fn new(p: usize, q: usize, m: usize, nd: usize, seed: u64) -> Self {
        let b = m * m;
        let s_atom  = 1.0 / (m  as f32).sqrt();
        let s_coeff = 1.0 / (nd as f32).sqrt();
        Self {
            p, q, m, nd,
            d1: randn_vec(nd * b, 2.0 * s_atom,  seed ^ 0x1111),
            d2: randn_vec(nd * b, 2.0 * s_atom,  seed ^ 0x2222),
            a1: randn_vec(p * q * m * nd, 2.0 * s_coeff, seed ^ 0x3333),
            a2: randn_vec(p * q * m * nd, 2.0 * s_coeff, seed ^ 0x4444),
        }
    }

    /// Number of parameters (atoms + coefficients).
    pub fn param_count(&self) -> usize {
        let b = self.m * self.m;
        2 * self.nd * b + 2 * self.p * self.q * self.m * self.nd
    }

    #[inline]
    pub fn a1_blk(&self, pp: usize, qq: usize) -> &[f32] {
        let base = (pp * self.q + qq) * self.m * self.nd;
        &self.a1[base..base + self.m * self.nd]
    }

    #[inline]
    pub fn a2_blk(&self, pp: usize, qq: usize) -> &[f32] {
        let base = (pp * self.q + qq) * self.m * self.nd;
        &self.a2[base..base + self.m * self.nd]
    }

    fn fwd_block(
        d1: &[f32], d2: &[f32], a1_blk: &[f32], a2_blk: &[f32],
        x_blk: &[f32], m: usize, nd: usize,
        y1: &mut [f32], z: &mut [f32], out: &mut [f32], eff: &mut [f32],
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if m == 8 {
            unsafe { fwd_block_avx2(d1, d2, a1_blk, a2_blk, x_blk, nd, y1, z, out, eff); }
            return;
        }
        let b = m * m;
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
        for i in 0..m { for j in 0..m { z[j*m+i] = y1[i*m+j]; } }
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
                out[j * m + r] += acc;
            }
        }
    }

    pub fn forward(&self, x: &[f32]) -> (Vec<f32>, FwdCache) {
        use rayon::prelude::*;
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        let mut y   = vec![0.0f32; p * b];
        let mut y1s = vec![0.0f32; p * q * b];
        let mut zs  = vec![0.0f32; p * q * b];

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
                Self::fwd_block(
                    &self.d1, &self.d2,
                    self.a1_blk(pp, qq), self.a2_blk(pp, qq),
                    &x[qq * b..(qq + 1) * b],
                    m, nd, y1, z, ypp, &mut eff,
                );
            }
        });
        (y, FwdCache { y1s, zs })
    }

    /// Forward without storing a cache — for inference where backward is not needed.
    /// Allocates only the output vec; y1/z scratch are reused per block, not kept.
    pub fn forward_inference(&self, x: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        let mut y = vec![0.0f32; p * b];

        let ypp_chunks: Vec<&mut [f32]> = y.chunks_mut(b).collect();
        ypp_chunks.into_par_iter().enumerate().for_each(|(pp, ypp)| {
            let mut eff = vec![0.0f32; b];
            let mut y1  = vec![0.0f32; b]; // per-block scratch, reused each qq
            let mut z   = vec![0.0f32; b];
            for qq in 0..q {
                Self::fwd_block(
                    &self.d1, &self.d2,
                    self.a1_blk(pp, qq), self.a2_blk(pp, qq),
                    &x[qq * b..(qq + 1) * b],
                    m, nd, &mut y1, &mut z, ypp, &mut eff,
                );
            }
        });
        y
    }

    pub fn backward(&self, x: &[f32], cache: &FwdCache, dout: &[f32]) -> Grads {
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
                let bk     = pp * q + qq;
                let z      = &cache.zs[bk * b..(bk + 1) * b];
                let x_blk  = &x[qq * b..(qq + 1) * b];
                let da_base = bk * m * nd;

                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                if m == 8 {
                    unsafe {
                        let dd1_ptr = g.dd1.as_mut_ptr();
                        let dd2_ptr = g.dd2.as_mut_ptr();
                        let da1_ptr = g.da1.as_mut_ptr().add(da_base);
                        let da2_ptr = g.da2.as_mut_ptr().add(da_base);
                        bwd_block_avx2(
                            self.d1.as_ptr(), self.d2.as_ptr(),
                            self.a1_blk(pp, qq).as_ptr(), self.a2_blk(pp, qq).as_ptr(),
                            x_blk.as_ptr(), z.as_ptr(),
                            dout_pp.as_ptr(), nd,
                            da1_ptr, da2_ptr, dd1_ptr, dd2_ptr,
                        );
                    }
                    continue;
                }

                // Scalar fallback (m≠8 or non-x86)
                dz.fill(0.0);
                for j in 0..m {
                    let zj     = &z[j * m..(j + 1) * m];
                    let dout_j = &dout_pp[j * m..(j + 1) * m];
                    eff_j.fill(0.0);
                    for d in 0..nd {
                        let a = self.a2_blk(pp, qq)[j * nd + d];
                        let atom = &self.d2[d * b..d * b + b];
                        for e in 0..b { eff_j[e] += a * atom[e]; }
                    }
                    for r in 0..m {
                        let dy = dout_j[r];
                        for c in 0..m { dz[j * m + c] += eff_j[r * m + c] * dy; }
                    }
                    for r in 0..m {
                        let dy = dout_j[r];
                        if dy == 0.0 { continue; }
                        for d in 0..nd {
                            let a    = self.a2_blk(pp, qq)[j * nd + d];
                            let drow = &self.d2[(d * m + r) * m..(d * m + r) * m + m];
                            let u    = gemm::dot(drow, zj);
                            g.da2[da_base + j * nd + d] += dy * u;
                            let dd2row = &mut g.dd2[(d * m + r) * m..(d * m + r) * m + m];
                            for c in 0..m { dd2row[c] += dy * a * zj[c]; }
                        }
                    }
                }
                dy1.fill(0.0);
                for j in 0..m { for i in 0..m { dy1[i*m+j] = dz[j*m+i]; } }
                for i in 0..m {
                    let xi    = &x_blk[i * m..(i + 1) * m];
                    let dy1_i = &dy1[i * m..(i + 1) * m];
                    eff_i.fill(0.0);
                    for d in 0..nd {
                        let a = self.a1_blk(pp, qq)[i * nd + d];
                        let atom = &self.d1[d * b..d * b + b];
                        for e in 0..b { eff_i[e] += a * atom[e]; }
                    }
                    for r in 0..m {
                        let d_y = dy1_i[r];
                        if d_y == 0.0 { continue; }
                        for d in 0..nd {
                            let a    = self.a1_blk(pp, qq)[i * nd + d];
                            let drow = &self.d1[(d * m + r) * m..(d * m + r) * m + m];
                            let u    = gemm::dot(drow, xi);
                            g.da1[da_base + i * nd + d] += d_y * u;
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
