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
    dx: *mut f32,
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
            // dx[i,:] += dy1[i,r] * eff1[i][r,:]  — transpose matmul, one row per axpy8
            axpy8(dx.add(i * M), eff.as_ptr().add(r * M), d_y);
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
    /// Stored fp16 (fp16-migration branch, RESEARCH_LOG.md 2026-07-05): this
    /// is the single largest buffer `BufPool` manages, written once per
    /// token per block in forward and read back once per token per block in
    /// backward's phase1 -- a real, contained bandwidth/memory win, unlike
    /// the tied embedding table (no accumulator/gradient counterpart needs a
    /// persistent fp32 "master copy" here). The hot `apply_block`/
    /// `apply_block_avx2` kernel that PRODUCES this value is deliberately
    /// left untouched (fp32 throughout) -- conversion happens once at the
    /// caller boundary, right after that kernel returns, not inside it.
    pub zs: Vec<half::f16>,
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
        let s_atom  = 3.0 / (m as f32).sqrt() / (q as f32).powf(0.25);
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

    /// Below this many parallel units (`p`), rayon's fork-join wake cost
    /// outweighs the actual per-unit work — see `SharedMonarchProj`'s
    /// identical constant and the gate_c.rs toy-shape benchmarks for the
    /// measured motivation.
    const PARALLEL_THRESHOLD: usize = 8;

    pub fn forward(&self, x: &[f32]) -> (Vec<f32>, FwdCache) {
        if self.p < Self::PARALLEL_THRESHOLD {
            return self.forward_serial(x);
        }
        use rayon::prelude::*;
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        let mut y   = vec![0.0f32; p * b];
        let mut zs  = vec![0.0f32; p * q * b];

        let pp_data: Vec<(&mut [f32], &mut [f32])> = y.chunks_mut(b)
            .zip(zs.chunks_mut(q * b))
            .collect();

        pp_data.into_par_iter().enumerate().for_each(|(pp, (ypp, zs_pp))| {
            let mut eff = vec![0.0f32; b];
            let mut y1  = vec![0.0f32; b]; // scratch only — not cached, backward never reads it
            for qq in 0..q {
                let z = &mut zs_pp[qq * b..(qq + 1) * b];
                Self::fwd_block(
                    &self.d1, &self.d2,
                    self.a1_blk(pp, qq), self.a2_blk(pp, qq),
                    &x[qq * b..(qq + 1) * b],
                    m, nd, &mut y1, z, ypp, &mut eff,
                );
            }
        });
        let zs = { let mut z16 = vec![half::f16::from_f32(0.0); zs.len()]; crate::kernels::f16_simd::f32_to_f16(&zs, &mut z16); z16 };
        (y, FwdCache { zs })
    }

    /// Same math as `forward`, no rayon — for `p < PARALLEL_THRESHOLD`.
    fn forward_serial(&self, x: &[f32]) -> (Vec<f32>, FwdCache) {
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        let mut y   = vec![0.0f32; p * b];
        let mut zs  = vec![0.0f32; p * q * b];
        let mut eff = vec![0.0f32; b];
        let mut y1  = vec![0.0f32; b];
        for pp in 0..p {
            let ypp = &mut y[pp * b..(pp + 1) * b];
            for qq in 0..q {
                let z = &mut zs[(pp * q + qq) * b..(pp * q + qq + 1) * b];
                Self::fwd_block(
                    &self.d1, &self.d2,
                    self.a1_blk(pp, qq), self.a2_blk(pp, qq),
                    &x[qq * b..(qq + 1) * b],
                    m, nd, &mut y1, z, ypp, &mut eff,
                );
            }
        }
        let zs = { let mut z16 = vec![half::f16::from_f32(0.0); zs.len()]; crate::kernels::f16_simd::f32_to_f16(&zs, &mut z16); z16 };
        (y, FwdCache { zs })
    }

    /// Forward without storing a cache — for inference where backward is not needed.
    /// Allocates only the output vec; y1/z scratch are reused per block, not kept.
    pub fn forward_inference(&self, x: &[f32]) -> Vec<f32> {
        if self.p < Self::PARALLEL_THRESHOLD {
            return self.forward_inference_serial(x);
        }
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

    /// Same math as `forward_inference`, parallelizing over `qq` instead of
    /// `pp` — intended for when `q > p` (e.g. `w_down`'s transpose-of-
    /// `w_up`/`w_gate` shape), where `pp`-parallelism would only have `p`
    /// units to spread across the thread pool. Needs a reduction (multiple
    /// `qq`-workers contribute to every `pp`'s output), done via `collect()`
    /// into per-worker partials + a sequential merge for determinism.
    ///
    /// Benchmarked and found to be a **regression** vs. plain
    /// `forward_inference` for `w_down` at 1B scale (see RESEARCH_LOG.md) —
    /// the per-worker heap allocation for each partial buffer (one Vec per
    /// `qq`, `q` allocations total) plus the serial merge apparently costs
    /// more than the extra parallelism recovers at this problem size. Left
    /// here, not wired into any default path, in case it's worth revisiting
    /// with allocation reuse (e.g. scratch pool) rather than fresh Vecs.
    pub fn forward_inference_wide(&self, x: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        let partials: Vec<Vec<f32>> = (0..q).into_par_iter().map(|qq| {
            let mut y_partial = vec![0.0f32; p * b];
            let mut eff = vec![0.0f32; b];
            let mut y1  = vec![0.0f32; b];
            let mut z   = vec![0.0f32; b];
            let x_qq = &x[qq * b..(qq + 1) * b];
            for pp in 0..p {
                let ypp = &mut y_partial[pp * b..(pp + 1) * b];
                Self::fwd_block(
                    &self.d1, &self.d2,
                    self.a1_blk(pp, qq), self.a2_blk(pp, qq),
                    x_qq, m, nd, &mut y1, &mut z, ypp, &mut eff,
                );
            }
            y_partial
        }).collect();
        let mut y = vec![0.0f32; p * b];
        for part in partials {
            for i in 0..(p * b) { y[i] += part[i]; }
        }
        y
    }

    /// Same math as `forward_inference`, no rayon at all — for use inside a
    /// context that's *already* parallel (e.g. prefill's outer
    /// `tokens.par_iter()`), where an inner rayon dispatch per projection
    /// per token would create nested-parallelism contention instead of
    /// useful work. See the prefill dispatch-fusion entry in
    /// RESEARCH_LOG.md for why fusion made prefill *slower*: the
    /// hypothesis is that the outer parallelism already hides per-call
    /// dispatch overhead, so adding more (or wider) inner rayon dispatches
    /// only adds scheduling contention.
    pub fn forward_inference_serial(&self, x: &[f32]) -> Vec<f32> {
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        let mut y = vec![0.0f32; p * b];
        let mut eff = vec![0.0f32; b];
        let mut y1  = vec![0.0f32; b];
        let mut z   = vec![0.0f32; b];
        for pp in 0..p {
            let ypp = &mut y[pp * b..(pp + 1) * b];
            for qq in 0..q {
                Self::fwd_block(
                    &self.d1, &self.d2,
                    self.a1_blk(pp, qq), self.a2_blk(pp, qq),
                    &x[qq * b..(qq + 1) * b],
                    m, nd, &mut y1, &mut z, ypp, &mut eff,
                );
            }
        }
        y
    }

    /// Run `forward_inference` for several weight-disjoint projections that
    /// all read the *same* input `x` (e.g. wq/wk/wv, or up/gate), in one
    /// rayon dispatch instead of one per projection. This exists for
    /// autoregressive decode: unlike training, there's no batch of tokens
    /// to amortize dispatch overhead over (only one token exists per decode
    /// step), so the only available fusion axis is across sibling
    /// projections that share an input. All `projs` must have identical
    /// `p`/`q`/`m`/`nd` (same shape) — panics otherwise.
    ///
    /// Returns one `[projs.len() * p*b]` buffer; projection `g`'s output is
    /// `result[g*p*b .. (g+1)*p*b]`.
    pub fn forward_inference_grouped(projs: &[&SharedMonarchMatmul], x: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        let n = projs.len();
        let (p, q, m, nd) = (projs[0].p, projs[0].q, projs[0].m, projs[0].nd);
        for proj in projs {
            assert_eq!((proj.p, proj.q, proj.m, proj.nd), (p, q, m, nd),
                "forward_inference_grouped: all projections must share shape");
        }
        let b = m * m;
        let mut y = vec![0.0f32; n * p * b];

        let y_chunks: Vec<&mut [f32]> = y.chunks_mut(b).collect();
        y_chunks.into_par_iter().enumerate().for_each(|(idx, ypp)| {
            let g  = idx / p;
            let pp = idx % p;
            let proj = projs[g];
            let mut eff = vec![0.0f32; b];
            let mut y1  = vec![0.0f32; b];
            let mut z   = vec![0.0f32; b];
            for qq in 0..q {
                Self::fwd_block(
                    &proj.d1, &proj.d2,
                    proj.a1_blk(pp, qq), proj.a2_blk(pp, qq),
                    &x[qq * b..(qq + 1) * b],
                    m, nd, &mut y1, &mut z, ypp, &mut eff,
                );
            }
        });
        y
    }

    /// `zs` is the per-token `[p*q*b]` cache slice from `FwdCache` (or a
    /// token-offset sub-slice of a batched `forward_batch` cache — see
    /// `zs_at`).
    /// Forward for `n_tokens` tokens in one call, parallelizing over the
    /// flattened `(token, pp)` space instead of just `pp`. Weights (d1/d2/
    /// a1/a2) are shared across tokens — only the input `x` varies — so each
    /// `(token, pp)` unit is fully independent and safe to dispatch in one
    /// rayon pass. This exists because calling `forward` once per token (as
    /// the naive per-token loop does) pays rayon's dispatch/wake overhead
    /// per token; batching amortizes that overhead across the whole
    /// sequence in a single dispatch.
    ///
    /// `x` is `[n_tokens * q*b]`; returns `y: [n_tokens * p*b]` and a cache
    /// whose `zs` is `[n_tokens * p*q*b]` — use `zs_at` to get a single
    /// token's slice back out for `backward`.
    pub fn forward_batch(&self, x: &[f32], n_tokens: usize) -> (Vec<f32>, FwdCache) {
        use rayon::prelude::*;
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        let in_dim = q * b;
        let mut y   = vec![0.0f32; n_tokens * p * b];
        let mut zs  = vec![0.0f32; n_tokens * p * q * b];

        let units: Vec<(&mut [f32], &mut [f32])> = y.chunks_mut(b)
            .zip(zs.chunks_mut(q * b))
            .collect();

        units.into_par_iter().enumerate().for_each(|(idx, (ypp, zs_pp))| {
            let t  = idx / p;
            let pp = idx % p;
            let x_t = &x[t * in_dim..(t + 1) * in_dim];
            let mut eff = vec![0.0f32; b];
            let mut y1  = vec![0.0f32; b]; // scratch only — not cached, backward never reads it
            for qq in 0..q {
                let z = &mut zs_pp[qq * b..(qq + 1) * b];
                Self::fwd_block(
                    &self.d1, &self.d2,
                    self.a1_blk(pp, qq), self.a2_blk(pp, qq),
                    &x_t[qq * b..(qq + 1) * b],
                    m, nd, &mut y1, z, ypp, &mut eff,
                );
            }
        });
        let zs = { let mut z16 = vec![half::f16::from_f32(0.0); zs.len()]; crate::kernels::f16_simd::f32_to_f16(&zs, &mut z16); z16 };
        (y, FwdCache { zs })
    }

    /// Slice a single token's `zs` out of a batched `forward_batch` cache
    /// (or pass `token = 0` for a single-token `forward` cache).
    #[inline]
    pub fn zs_at<'a>(&self, cache: &'a FwdCache, token: usize) -> &'a [half::f16] {
        let per_token = self.p * self.q * self.m * self.m;
        &cache.zs[token * per_token..(token + 1) * per_token]
    }

    pub fn backward(&self, x: &[f32], zs: &[half::f16], dout: &[f32], dx: &mut [f32]) -> Grads {
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        // Not the hot path (used by tests/research bins only -- production
        // goes through backward_batch/rows/cols_batch's per-block-boundary
        // conversion); one bulk convert here is simplest and correctness-
        // equivalent.
        let mut zs_f32 = vec![0.0f32; zs.len()];
        crate::kernels::f16_simd::f16_to_f32(zs, &mut zs_f32);
        let zs = &zs_f32[..];
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
                let z      = &zs[bk * b..(bk + 1) * b];
                let x_blk  = &x[qq * b..(qq + 1) * b];
                let da_base = bk * m * nd;

                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                if m == 8 {
                    unsafe {
                        let dd1_ptr = g.dd1.as_mut_ptr();
                        let dd2_ptr = g.dd2.as_mut_ptr();
                        let da1_ptr = g.da1.as_mut_ptr().add(da_base);
                        let da2_ptr = g.da2.as_mut_ptr().add(da_base);
                        let dx_ptr  = dx.as_mut_ptr().add(qq * b);
                        bwd_block_avx2(
                            self.d1.as_ptr(), self.d2.as_ptr(),
                            self.a1_blk(pp, qq).as_ptr(), self.a2_blk(pp, qq).as_ptr(),
                            x_blk.as_ptr(), z.as_ptr(),
                            dout_pp.as_ptr(), nd,
                            da1_ptr, da2_ptr, dd1_ptr, dd2_ptr,
                            dx_ptr,
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
                        let dx_i = &mut dx[qq * b + i * m..qq * b + (i + 1) * m];
                        for c in 0..m { dx_i[c] += d_y * eff_i[r * m + c]; }
                    }
                }
            }
        }
        g
    }
}

// ---------------------------------------------------------------------------
// SharedMonarchProj — same block math as SharedMonarchMatmul, but the atom
// dictionary (d1/d2) is owned externally instead of per-instance. Exists so a
// single dictionary can be shared model-wide (like BasisMatmul's `dict`
// parameter), while each projection keeps its own private coefficients
// (a1/a2). All the actual block math is reused unchanged from
// SharedMonarchMatmul's private helpers (`fwd_block`, `bwd_block_avx2`) —
// those already take d1/d2 as plain slice parameters, so nothing numeric is
// duplicated here, only the dispatch/ownership wrapper.
// ---------------------------------------------------------------------------

pub struct SharedMonarchProj {
    pub p: usize,
    pub q: usize,
    pub m: usize,
    pub nd: usize,
    pub a1: Vec<f32>,
    pub a2: Vec<f32>,
}

/// Initialise a shared atom dictionary (`d1`, `d2`) for use across many
/// [`SharedMonarchProj`] instances — the model-wide analogue of
/// [`SharedMonarchMatmul::new`]'s per-instance atom init.
pub fn init_shared_atoms(nd: usize, m: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let b = m * m;
    let s_atom = 3.0 / (m as f32).sqrt();
    (
        randn_vec(nd * b, 2.0 * s_atom, seed ^ 0x1111),
        randn_vec(nd * b, 2.0 * s_atom, seed ^ 0x2222),
    )
}

impl SharedMonarchProj {
    /// Construct a new projection's coefficients only — the atom dictionary is
    /// owned externally and passed into `forward`/`backward`. Use
    /// [`init_shared_atoms`] to build one.
    ///
    /// The `q^{-1/4}` term from the two-stage variance derivation (see
    /// RESEARCH_LOG.md, 2026-06-30 — originally applied to the atoms in
    /// `SharedMonarchMatmul::new`) lives here instead: the dictionary is now
    /// shared across projections that may have different `q`, so the
    /// per-projection compensation has to live on the per-projection
    /// coefficients, not the shared atoms (`init_shared_atoms` has no `q` to
    /// scale by). Atom and coefficient scale enter the composed variance as a
    /// product, so relocating the exponent preserves the same solved target.
    pub fn new(p: usize, q: usize, m: usize, nd: usize, seed: u64) -> Self {
        let s_coeff = 1.0 / (nd as f32).sqrt() / (q as f32).powf(0.25);
        Self {
            p, q, m, nd,
            a1: randn_vec(p * q * m * nd, 2.0 * s_coeff, seed ^ 0x3333),
            a2: randn_vec(p * q * m * nd, 2.0 * s_coeff, seed ^ 0x4444),
        }
    }

    /// Number of parameters owned by this instance (coefficients only — the
    /// shared dictionary is not counted here, since it's amortized model-wide).
    pub fn param_count(&self) -> usize {
        2 * self.p * self.q * self.m * self.nd
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

    /// Below this many parallel units (`p`), rayon's fork-join wake cost
    /// (~5-15µs on this machine) outweighs the actual per-unit work — a
    /// handful of microseconds each at toy scale — so we run inline instead.
    /// See `forward_inference_serial`'s doc comment and the gate_c.rs
    /// `train_small_lod`-shape benchmarks (256x256, P=4: 5x slower than
    /// BasisMatmul) for the measured motivation.
    const PARALLEL_THRESHOLD: usize = 8;

    pub fn forward(&self, d1: &[f32], d2: &[f32], x: &[f32]) -> (Vec<f32>, FwdCache) {
        if self.p < Self::PARALLEL_THRESHOLD {
            return self.forward_serial(d1, d2, x);
        }
        use rayon::prelude::*;
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        let mut y   = vec![0.0f32; p * b];
        let mut zs  = vec![0.0f32; p * q * b];

        let pp_data: Vec<(&mut [f32], &mut [f32])> = y.chunks_mut(b)
            .zip(zs.chunks_mut(q * b))
            .collect();

        pp_data.into_par_iter().enumerate().for_each(|(pp, (ypp, zs_pp))| {
            let mut eff = vec![0.0f32; b];
            let mut y1  = vec![0.0f32; b];
            for qq in 0..q {
                let z = &mut zs_pp[qq * b..(qq + 1) * b];
                SharedMonarchMatmul::fwd_block(
                    d1, d2,
                    self.a1_blk(pp, qq), self.a2_blk(pp, qq),
                    &x[qq * b..(qq + 1) * b],
                    m, nd, &mut y1, z, ypp, &mut eff,
                );
            }
        });
        let zs = { let mut z16 = vec![half::f16::from_f32(0.0); zs.len()]; crate::kernels::f16_simd::f32_to_f16(&zs, &mut z16); z16 };
        (y, FwdCache { zs })
    }

    /// Same math as `forward`, no rayon — for `p < PARALLEL_THRESHOLD`.
    fn forward_serial(&self, d1: &[f32], d2: &[f32], x: &[f32]) -> (Vec<f32>, FwdCache) {
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        let mut y   = vec![0.0f32; p * b];
        let mut zs  = vec![0.0f32; p * q * b];
        let mut eff = vec![0.0f32; b];
        let mut y1  = vec![0.0f32; b];
        for pp in 0..p {
            let ypp = &mut y[pp * b..(pp + 1) * b];
            for qq in 0..q {
                let z = &mut zs[(pp * q + qq) * b..(pp * q + qq + 1) * b];
                SharedMonarchMatmul::fwd_block(
                    d1, d2,
                    self.a1_blk(pp, qq), self.a2_blk(pp, qq),
                    &x[qq * b..(qq + 1) * b],
                    m, nd, &mut y1, z, ypp, &mut eff,
                );
            }
        }
        let zs = { let mut z16 = vec![half::f16::from_f32(0.0); zs.len()]; crate::kernels::f16_simd::f32_to_f16(&zs, &mut z16); z16 };
        (y, FwdCache { zs })
    }

    /// Reconstruct every `(pp,qq)` block's effective weight once — weight-only,
    /// independent of any token — into flat `[p*q, m*b]`-shaped buffers.
    pub fn expand_all_blocks(d1: &[f32], d2: &[f32], a1: &[f32], a2: &[f32], p: usize, q: usize, m: usize, nd: usize) -> (Vec<f32>, Vec<f32>) {
        let b = m * m;
        let mut eff1_all = vec![0.0f32; p * q * m * b];
        let mut eff2_all = vec![0.0f32; p * q * m * b];
        for pp in 0..p {
            for qq in 0..q {
                let idx = pp * q + qq;
                let base = idx * m * nd;
                SharedMonarchMatmul::expand_block(
                    d1, d2, &a1[base..base + m * nd], &a2[base..base + m * nd], m, nd,
                    &mut eff1_all[idx * m * b..(idx + 1) * m * b],
                    &mut eff2_all[idx * m * b..(idx + 1) * m * b],
                );
            }
        }
        (eff1_all, eff2_all)
    }

    /// Forward for `n_tokens` tokens in one call. Every `(pp,qq)` block's
    /// effective weight is reconstructed from the shared dictionary **once**
    /// (weight-only, independent of any token — see `expand_all_blocks`),
    /// then applied per token, parallelizing over tokens (the axis with
    /// abundant work — `P`/`Q` are typically small, single digits to tens,
    /// while `n_tokens` is the sequence length). See RESEARCH_LOG.md
    /// (2026-07-03, Fable review + follow-up) for why an earlier version
    /// that parallelized over `P` instead of tokens was a regression at
    /// small `P`: it lost real multi-core parallelism without a
    /// compensating win, since reconstruction wasn't forward's bottleneck
    /// per token to begin with.
    pub fn forward_batch(
        &self, d1: &[f32], d2: &[f32], x: &[f32], n_tokens: usize, pool: &mut crate::kernels::scratch::BufPool,
    ) -> (Vec<f32>, FwdCache) {
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        let in_dim = q * b;
        let (eff1_all, eff2_all) = Self::expand_all_blocks(d1, d2, &self.a1, &self.a2, p, q, m, nd);

        // `y` is accumulated into (`out[..] += ..` in `apply_block`), so it
        // must start zeroed; `zs` is pure write-before-read (every element
        // assigned exactly once across the pp/qq loop), so `take_uninit`
        // skips a wasted zero-fill pass on top of skipping the allocation.
        let mut y   = pool.take_zeroed(n_tokens * p * b);
        let mut zs  = pool.take_f16_uninit(n_tokens * p * q * b);

        let apply_token = |t: usize, y_t: &mut [f32], zs_t: &mut [half::f16]| {
            let x_t = &x[t * in_dim..(t + 1) * in_dim];
            let mut y1 = vec![0.0f32; b];
            let mut z_f32 = vec![0.0f32; b];
            for pp in 0..p {
                let ypp = &mut y_t[pp * b..(pp + 1) * b];
                for qq in 0..q {
                    let idx = pp * q + qq;
                    SharedMonarchMatmul::apply_block(
                        &eff1_all[idx * m * b..(idx + 1) * m * b],
                        &eff2_all[idx * m * b..(idx + 1) * m * b],
                        &x_t[qq * b..(qq + 1) * b], m, &mut y1, &mut z_f32, ypp,
                    );
                    let z_dst = &mut zs_t[(pp * q + qq) * b..(pp * q + qq + 1) * b];
                    crate::kernels::f16_simd::f32_to_f16(&z_f32, z_dst);
                }
            }
        };

        if n_tokens < Self::PARALLEL_THRESHOLD {
            for (t, (y_t, zs_t)) in y.chunks_mut(p * b).zip(zs.chunks_mut(p * q * b)).enumerate() {
                apply_token(t, y_t, zs_t);
            }
        } else {
            use rayon::prelude::*;
            y.par_chunks_mut(p * b).zip(zs.par_chunks_mut(p * q * b)).enumerate()
                .for_each(|(t, (y_t, zs_t))| apply_token(t, y_t, zs_t));
        }
        (y, FwdCache { zs })
    }

    /// VJP for `forward_batch`. Same hoist: every block's effective weight is
    /// reconstructed once, then every token's gradient contribution is
    /// computed against the shared, precomputed blocks. Parallelizes over
    /// **chunks of tokens** (not over `P`, for the same reason as
    /// `forward_batch`) — each chunk accumulates its own local
    /// `dd1`/`dd2`/`da1`/`da2` partials (since every token touches every
    /// block, a per-token partial would mean `n_tokens` full-sized copies;
    /// chunking to `num_threads` keeps that bounded). `dx` is
    /// `[n_tokens, in_dim]` and is zeroed here.
    /// Two-phase batched backward (see RESEARCH_LOG.md 2026-07-03, Fable
    /// review + implementation): phase 1 accumulates each block's `s1`/`s2`
    /// outer-product sums across every token in the batch (cheap — no `nd`
    /// dependence); phase 2 contracts each block with the dictionary exactly
    /// once (was: once per token). Same math as the old per-token
    /// `backward_block_hoisted` loop, reassociated — see
    /// `phase1_plus_contract_matches_backward_block_hoisted`.
    pub fn backward_batch(
        &self, d1: &[f32], d2: &[f32], x: &[f32], cache: FwdCache, dout: &[f32], dx: &mut [f32], n_tokens: usize,
        pool: &mut crate::kernels::scratch::BufPool,
    ) -> Grads {
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        let in_dim = q * b;
        let out_dim = p * b;
        dx.iter_mut().for_each(|v| *v = 0.0);
        let (eff1_all, eff2_all) = Self::expand_all_blocks(d1, d2, &self.a1, &self.a2, p, q, m, nd);
        // `cache.zs` is only read here, from this point until it's returned
        // to the pool below -- taking it by value (rather than the old
        // `zs: &[f32]`) lets us recycle its allocation for next call.
        let FwdCache { zs } = cache;

        // Phase 1: accumulate s1/s2 (p*q*m*b each) and dx for tokens t0..t1
        // — shared by both the sequential and rayon-chunked callers below.
        let run_range = |t0: usize, t1: usize, s1: &mut [f32], s2: &mut [f32], dx_out: &mut [f32]| {
            let mut z_f32 = vec![0.0f32; b];
            for t in t0..t1 {
                for pp in 0..p {
                    let dout_pp = &dout[t * out_dim + pp * b..t * out_dim + (pp + 1) * b];
                    for qq in 0..q {
                        let idx = pp * q + qq;
                        let eff1 = &eff1_all[idx * m * b..(idx + 1) * m * b];
                        let eff2 = &eff2_all[idx * m * b..(idx + 1) * m * b];
                        let x_blk = &x[t * in_dim + qq * b..t * in_dim + (qq + 1) * b];
                        let z16 = &zs[((t * p + pp) * q + qq) * b..((t * p + pp) * q + qq + 1) * b];
                        crate::kernels::f16_simd::f16_to_f32(z16, &mut z_f32);
                        let dx_blk = &mut dx_out[(t - t0) * in_dim + qq * b..(t - t0) * in_dim + (qq + 1) * b];
                        let s1_blk = &mut s1[idx * m * b..(idx + 1) * m * b];
                        let s2_blk = &mut s2[idx * m * b..(idx + 1) * m * b];
                        SharedMonarchMatmul::backward_block_phase1(eff1, eff2, x_blk, &z_f32, dout_pp, dx_blk, m, s1_blk, s2_blk);
                    }
                }
            }
        };

        if n_tokens < Self::PARALLEL_THRESHOLD {
            let mut s1 = vec![0.0f32; p * q * m * b];
            let mut s2 = vec![0.0f32; p * q * m * b];
            let mut dx_full = vec![0.0f32; n_tokens * in_dim];
            run_range(0, n_tokens, &mut s1, &mut s2, &mut dx_full);
            dx.copy_from_slice(&dx_full);
            pool.give_f16(zs);
            return SharedMonarchMatmul::contract_all_blocks(d1, d2, &self.a1, &self.a2, &s1, &s2, p, q, m, nd);
        }

        use rayon::prelude::*;
        let n_chunks = rayon::current_num_threads().max(1).min(n_tokens);
        let chunk_len = n_tokens.div_ceil(n_chunks);
        let results: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> =
            (0..n_chunks).into_par_iter().map(|c| {
                let t0 = c * chunk_len;
                let t1 = ((c + 1) * chunk_len).min(n_tokens);
                let mut s1 = vec![0.0f32; p * q * m * b];
                let mut s2 = vec![0.0f32; p * q * m * b];
                let mut dx_chunk = vec![0.0f32; t1.saturating_sub(t0) * in_dim];
                run_range(t0, t1, &mut s1, &mut s2, &mut dx_chunk);
                (s1, s2, dx_chunk, t0, t1)
            }).collect();

        let mut s1_all = vec![0.0f32; p * q * m * b];
        let mut s2_all = vec![0.0f32; p * q * m * b];
        for (s1, s2, dx_chunk, t0, t1) in results {
            for i in 0..s1_all.len() { s1_all[i] += s1[i]; s2_all[i] += s2[i]; }
            dx[t0 * in_dim..t1 * in_dim].copy_from_slice(&dx_chunk);
        }
        pool.give_f16(zs);
        SharedMonarchMatmul::contract_all_blocks(d1, d2, &self.a1, &self.a2, &s1_all, &s2_all, p, q, m, nd)
    }

    /// Slice a single token's `zs` out of a batched `forward_batch` cache (or
    /// pass `token = 0` for a single-token `forward` cache).
    #[inline]
    pub fn zs_at<'a>(&self, cache: &'a FwdCache, token: usize) -> &'a [half::f16] {
        let per_token = self.p * self.q * self.m * self.m;
        &cache.zs[token * per_token..(token + 1) * per_token]
    }

    pub fn backward(
        &self, d1: &[f32], d2: &[f32], x: &[f32], zs: &[half::f16], dout: &[f32], dx: &mut [f32],
    ) -> Grads {
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        let mut zs_f32 = vec![0.0f32; zs.len()];
        crate::kernels::f16_simd::f16_to_f32(zs, &mut zs_f32);
        let zs = &zs_f32[..];
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
                let z      = &zs[bk * b..(bk + 1) * b];
                let x_blk  = &x[qq * b..(qq + 1) * b];
                let da_base = bk * m * nd;

                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                if m == 8 {
                    unsafe {
                        let dd1_ptr = g.dd1.as_mut_ptr();
                        let dd2_ptr = g.dd2.as_mut_ptr();
                        let da1_ptr = g.da1.as_mut_ptr().add(da_base);
                        let da2_ptr = g.da2.as_mut_ptr().add(da_base);
                        let dx_ptr  = dx.as_mut_ptr().add(qq * b);
                        bwd_block_avx2(
                            d1.as_ptr(), d2.as_ptr(),
                            self.a1_blk(pp, qq).as_ptr(), self.a2_blk(pp, qq).as_ptr(),
                            x_blk.as_ptr(), z.as_ptr(),
                            dout_pp.as_ptr(), nd,
                            da1_ptr, da2_ptr, dd1_ptr, dd2_ptr,
                            dx_ptr,
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
                        let atom = &d2[d * b..d * b + b];
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
                            let drow = &d2[(d * m + r) * m..(d * m + r) * m + m];
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
                        let atom = &d1[d * b..d * b + b];
                        for e in 0..b { eff_i[e] += a * atom[e]; }
                    }
                    for r in 0..m {
                        let d_y = dy1_i[r];
                        if d_y == 0.0 { continue; }
                        for d in 0..nd {
                            let a    = self.a1_blk(pp, qq)[i * nd + d];
                            let drow = &d1[(d * m + r) * m..(d * m + r) * m + m];
                            let u    = gemm::dot(drow, xi);
                            g.da1[da_base + i * nd + d] += d_y * u;
                            let dd1row = &mut g.dd1[(d * m + r) * m..(d * m + r) * m + m];
                            for c in 0..m { dd1row[c] += d_y * a * xi[c]; }
                        }
                        let dx_i = &mut dx[qq * b + i * m..qq * b + (i + 1) * m];
                        for c in 0..m { dx_i[c] += d_y * eff_i[r * m + c]; }
                    }
                }
            }
        }
        g
    }

    /// Forward restricted to a subset of output row-blocks (mirrors
    /// `BasisMatmul::forward_rows`). Used for MoE-style block routing where
    /// only some output blocks are needed per token — an exact skip, not an
    /// approximation, since the caller guarantees the unselected rows are
    /// never read.
    pub fn forward_rows(&self, d1: &[f32], d2: &[f32], x: &[f32], active_p: &[usize]) -> (Vec<f32>, FwdCache) {
        let (q, m, nd) = (self.q, self.m, self.nd);
        let b = m * m;
        let mut y = vec![0.0f32; self.p * b];
        let mut zs = vec![0.0f32; self.p * q * b];
        for &pp in active_p {
            debug_assert!(pp < self.p, "active row-block {pp} out of range");
            let ypp = &mut y[pp * b..(pp + 1) * b];
            let mut eff = vec![0.0f32; b];
            let mut y1  = vec![0.0f32; b];
            for qq in 0..q {
                let z = &mut zs[(pp * q + qq) * b..(pp * q + qq + 1) * b];
                SharedMonarchMatmul::fwd_block(
                    d1, d2,
                    self.a1_blk(pp, qq), self.a2_blk(pp, qq),
                    &x[qq * b..(qq + 1) * b],
                    m, nd, &mut y1, z, ypp, &mut eff,
                );
            }
        }
        let zs = { let mut z16 = vec![half::f16::from_f32(0.0); zs.len()]; crate::kernels::f16_simd::f32_to_f16(&zs, &mut z16); z16 };
        (y, FwdCache { zs })
    }

    /// Forward summing only over the selected input col-blocks (mirrors
    /// `BasisMatmul::forward_cols`). Used for the routed `W_down` projection,
    /// whose input (the FFN activation) is exactly zero outside the routed
    /// blocks — skipping them is exact. All output blocks are produced.
    pub fn forward_cols(&self, d1: &[f32], d2: &[f32], x: &[f32], active_q: &[usize]) -> (Vec<f32>, FwdCache) {
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        let mut y = vec![0.0f32; p * b];
        let mut zs = vec![0.0f32; p * q * b];
        for pp in 0..p {
            let ypp = &mut y[pp * b..(pp + 1) * b];
            let mut eff = vec![0.0f32; b];
            let mut y1  = vec![0.0f32; b];
            for &qq in active_q {
                debug_assert!(qq < q, "active col-block {qq} out of range");
                let z = &mut zs[(pp * q + qq) * b..(pp * q + qq + 1) * b];
                SharedMonarchMatmul::fwd_block(
                    d1, d2,
                    self.a1_blk(pp, qq), self.a2_blk(pp, qq),
                    &x[qq * b..(qq + 1) * b],
                    m, nd, &mut y1, z, ypp, &mut eff,
                );
            }
        }
        let zs = { let mut z16 = vec![half::f16::from_f32(0.0); zs.len()]; crate::kernels::f16_simd::f32_to_f16(&zs, &mut z16); z16 };
        (y, FwdCache { zs })
    }

    /// Backward for [`forward_rows`](Self::forward_rows). Only `active_p`
    /// output blocks were produced, so only they contribute gradient; `dx`
    /// still spans the full input (every `qq` fed every active `pp`).
    pub fn backward_rows(
        &self, d1: &[f32], d2: &[f32], x: &[f32], zs: &[half::f16], dout: &[f32], dx: &mut [f32],
        active_p: &[usize],
    ) -> Grads {
        let (q, m, nd) = (self.q, self.m, self.nd);
        let b = m * m;
        let mut zs_f32 = vec![0.0f32; zs.len()];
        crate::kernels::f16_simd::f16_to_f32(zs, &mut zs_f32);
        let zs = &zs_f32[..];
        let mut g = Grads {
            dd1: vec![0.0f32; nd * b],
            dd2: vec![0.0f32; nd * b],
            da1: vec![0.0f32; self.p * q * m * nd],
            da2: vec![0.0f32; self.p * q * m * nd],
        };
        let mut dz    = vec![0.0f32; b];
        let mut dy1   = vec![0.0f32; b];
        let mut eff_j = vec![0.0f32; b];
        let mut eff_i = vec![0.0f32; b];

        for &pp in active_p {
            debug_assert!(pp < self.p, "active row-block {pp} out of range");
            let dout_pp = &dout[pp * b..(pp + 1) * b];
            for qq in 0..q {
                Self::backward_block(
                    self, d1, d2, x, zs, dout_pp, dx, pp, qq,
                    &mut g, &mut dz, &mut dy1, &mut eff_j, &mut eff_i,
                );
            }
        }
        g
    }

    /// Backward for [`forward_cols`](Self::forward_cols). Every output block
    /// receives gradient, but only `active_q` input blocks fed the forward, so
    /// only they get nonzero coefficient/dict/`dx` gradient.
    pub fn backward_cols(
        &self, d1: &[f32], d2: &[f32], x: &[f32], zs: &[half::f16], dout: &[f32], dx: &mut [f32],
        active_q: &[usize],
    ) -> Grads {
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        let mut zs_f32 = vec![0.0f32; zs.len()];
        crate::kernels::f16_simd::f16_to_f32(zs, &mut zs_f32);
        let zs = &zs_f32[..];
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
            for &qq in active_q {
                debug_assert!(qq < q, "active col-block {qq} out of range");
                Self::backward_block(
                    self, d1, d2, x, zs, dout_pp, dx, pp, qq,
                    &mut g, &mut dz, &mut dy1, &mut eff_j, &mut eff_i,
                );
            }
        }
        g
    }

    /// Batched, hoisted `forward_rows`: every `(pp,qq)` block is reconstructed
    /// once (dense — see `expand_all_blocks`; with per-token routing, the
    /// union of blocks touched across a real-sized batch typically covers
    /// most/all of `P` anyway, so reconstructing everything up front is
    /// simpler than tracking a per-batch active set and is not meaningfully
    /// more work), then applied per token restricted to that token's own
    /// `active_p[t]` — the routing skip is preserved exactly, only the
    /// reconstruction is shared. `x` is `[n_tokens, in_dim]`; `active_p` has
    /// one entry per token.
    pub fn forward_rows_batch(
        &self, d1: &[f32], d2: &[f32], x: &[f32], active_p: &[Vec<usize>], n_tokens: usize,
        pool: &mut crate::kernels::scratch::BufPool,
    ) -> (Vec<f32>, FwdCache) {
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        let in_dim = q * b;
        let (eff1_all, eff2_all) = Self::expand_all_blocks(d1, d2, &self.a1, &self.a2, p, q, m, nd);

        let mut y  = pool.take_zeroed(n_tokens * p * b);
        let mut zs = pool.take_f16_uninit(n_tokens * p * q * b);

        let apply_token = |t: usize, y_t: &mut [f32], zs_t: &mut [half::f16]| {
            let x_t = &x[t * in_dim..(t + 1) * in_dim];
            let mut y1 = vec![0.0f32; b];
            let mut z_f32 = vec![0.0f32; b];
            for &pp in &active_p[t] {
                debug_assert!(pp < p, "active row-block {pp} out of range");
                let ypp = &mut y_t[pp * b..(pp + 1) * b];
                for qq in 0..q {
                    let idx = pp * q + qq;
                    SharedMonarchMatmul::apply_block(
                        &eff1_all[idx * m * b..(idx + 1) * m * b], &eff2_all[idx * m * b..(idx + 1) * m * b],
                        &x_t[qq * b..(qq + 1) * b], m, &mut y1, &mut z_f32, ypp,
                    );
                    let z_dst = &mut zs_t[(pp * q + qq) * b..(pp * q + qq + 1) * b];
                    crate::kernels::f16_simd::f32_to_f16(&z_f32, z_dst);
                }
            }
        };

        if n_tokens < Self::PARALLEL_THRESHOLD {
            for (t, (y_t, zs_t)) in y.chunks_mut(p * b).zip(zs.chunks_mut(p * q * b)).enumerate() {
                apply_token(t, y_t, zs_t);
            }
        } else {
            use rayon::prelude::*;
            y.par_chunks_mut(p * b).zip(zs.par_chunks_mut(p * q * b)).enumerate()
                .for_each(|(t, (y_t, zs_t))| apply_token(t, y_t, zs_t));
        }
        (y, FwdCache { zs })
    }

    /// Batched, hoisted `forward_cols` — same hoist as `forward_rows_batch`,
    /// restricting the *input* col-blocks per token instead of the output
    /// row-blocks (for the routed `W_down` projection, whose input is exactly
    /// zero outside the routed blocks).
    pub fn forward_cols_batch(
        &self, d1: &[f32], d2: &[f32], x: &[f32], active_q: &[Vec<usize>], n_tokens: usize,
        pool: &mut crate::kernels::scratch::BufPool,
    ) -> (Vec<f32>, FwdCache) {
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        let in_dim = q * b;
        let (eff1_all, eff2_all) = Self::expand_all_blocks(d1, d2, &self.a1, &self.a2, p, q, m, nd);

        let mut y  = pool.take_zeroed(n_tokens * p * b);
        let mut zs = pool.take_f16_uninit(n_tokens * p * q * b);

        let apply_token = |t: usize, y_t: &mut [f32], zs_t: &mut [half::f16]| {
            let x_t = &x[t * in_dim..(t + 1) * in_dim];
            let mut y1 = vec![0.0f32; b];
            let mut z_f32 = vec![0.0f32; b];
            for pp in 0..p {
                let ypp = &mut y_t[pp * b..(pp + 1) * b];
                for &qq in &active_q[t] {
                    debug_assert!(qq < q, "active col-block {qq} out of range");
                    let idx = pp * q + qq;
                    SharedMonarchMatmul::apply_block(
                        &eff1_all[idx * m * b..(idx + 1) * m * b], &eff2_all[idx * m * b..(idx + 1) * m * b],
                        &x_t[qq * b..(qq + 1) * b], m, &mut y1, &mut z_f32, ypp,
                    );
                    let z_dst = &mut zs_t[(pp * q + qq) * b..(pp * q + qq + 1) * b];
                    crate::kernels::f16_simd::f32_to_f16(&z_f32, z_dst);
                }
            }
        };

        if n_tokens < Self::PARALLEL_THRESHOLD {
            for (t, (y_t, zs_t)) in y.chunks_mut(p * b).zip(zs.chunks_mut(p * q * b)).enumerate() {
                apply_token(t, y_t, zs_t);
            }
        } else {
            use rayon::prelude::*;
            y.par_chunks_mut(p * b).zip(zs.par_chunks_mut(p * q * b)).enumerate()
                .for_each(|(t, (y_t, zs_t))| apply_token(t, y_t, zs_t));
        }
        (y, FwdCache { zs })
    }

    /// Batched, hoisted `backward_rows` — token-chunked parallelism (see
    /// `backward_batch`), restricted per token to `active_p[t]`. `dx` is
    /// `[n_tokens, in_dim]`, zeroed here.
    /// Two-phase batched `backward_rows` — same restructuring as
    /// `backward_batch` (accumulate `s1`/`s2` across the batch, contract once
    /// per block afterward), restricted per token to `active_p[t]`.
    pub fn backward_rows_batch(
        &self, d1: &[f32], d2: &[f32], x: &[f32], cache: FwdCache, dout: &[f32], dx: &mut [f32],
        active_p: &[Vec<usize>], n_tokens: usize,
        pool: &mut crate::kernels::scratch::BufPool,
    ) -> Grads {
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        let in_dim = q * b;
        let out_dim = p * b;
        dx.iter_mut().for_each(|v| *v = 0.0);
        let (eff1_all, eff2_all) = Self::expand_all_blocks(d1, d2, &self.a1, &self.a2, p, q, m, nd);
        let FwdCache { zs } = cache;

        let run_range = |t0: usize, t1: usize, s1: &mut [f32], s2: &mut [f32], dx_out: &mut [f32]| {
            let mut z_f32 = vec![0.0f32; b];
            for t in t0..t1 {
                let dout_t = &dout[t * out_dim..(t + 1) * out_dim];
                for &pp in &active_p[t] {
                    debug_assert!(pp < p, "active row-block {pp} out of range");
                    let dout_pp = &dout_t[pp * b..(pp + 1) * b];
                    for qq in 0..q {
                        let idx = pp * q + qq;
                        let eff1 = &eff1_all[idx * m * b..(idx + 1) * m * b];
                        let eff2 = &eff2_all[idx * m * b..(idx + 1) * m * b];
                        let x_blk = &x[t * in_dim + qq * b..t * in_dim + (qq + 1) * b];
                        let z16 = &zs[((t * p + pp) * q + qq) * b..((t * p + pp) * q + qq + 1) * b];
                        crate::kernels::f16_simd::f16_to_f32(z16, &mut z_f32);
                        let dx_blk = &mut dx_out[(t - t0) * in_dim + qq * b..(t - t0) * in_dim + (qq + 1) * b];
                        let s1_blk = &mut s1[idx * m * b..(idx + 1) * m * b];
                        let s2_blk = &mut s2[idx * m * b..(idx + 1) * m * b];
                        SharedMonarchMatmul::backward_block_phase1(eff1, eff2, x_blk, &z_f32, dout_pp, dx_blk, m, s1_blk, s2_blk);
                    }
                }
            }
        };

        if n_tokens < Self::PARALLEL_THRESHOLD {
            let mut s1 = vec![0.0f32; p * q * m * b];
            let mut s2 = vec![0.0f32; p * q * m * b];
            let mut dx_full = vec![0.0f32; n_tokens * in_dim];
            run_range(0, n_tokens, &mut s1, &mut s2, &mut dx_full);
            dx.copy_from_slice(&dx_full);
            pool.give_f16(zs);
            return SharedMonarchMatmul::contract_all_blocks(d1, d2, &self.a1, &self.a2, &s1, &s2, p, q, m, nd);
        }

        use rayon::prelude::*;
        let n_chunks = rayon::current_num_threads().max(1).min(n_tokens);
        let chunk_len = n_tokens.div_ceil(n_chunks);
        let results: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> =
            (0..n_chunks).into_par_iter().map(|c| {
                let t0 = c * chunk_len;
                let t1 = ((c + 1) * chunk_len).min(n_tokens);
                let mut s1 = vec![0.0f32; p * q * m * b];
                let mut s2 = vec![0.0f32; p * q * m * b];
                let mut dx_chunk = vec![0.0f32; t1.saturating_sub(t0) * in_dim];
                run_range(t0, t1, &mut s1, &mut s2, &mut dx_chunk);
                (s1, s2, dx_chunk, t0, t1)
            }).collect();

        let mut s1_all = vec![0.0f32; p * q * m * b];
        let mut s2_all = vec![0.0f32; p * q * m * b];
        for (s1, s2, dx_chunk, t0, t1) in results {
            for i in 0..s1_all.len() { s1_all[i] += s1[i]; s2_all[i] += s2[i]; }
            dx[t0 * in_dim..t1 * in_dim].copy_from_slice(&dx_chunk);
        }
        pool.give_f16(zs);
        SharedMonarchMatmul::contract_all_blocks(d1, d2, &self.a1, &self.a2, &s1_all, &s2_all, p, q, m, nd)
    }

    /// Two-phase batched `backward_cols` — mirrors `backward_rows_batch`,
    /// restricting the *input* col-blocks per token instead of the output
    /// row-blocks. Every output block contributes gradient (not restricted).
    pub fn backward_cols_batch(
        &self, d1: &[f32], d2: &[f32], x: &[f32], cache: FwdCache, dout: &[f32], dx: &mut [f32],
        active_q: &[Vec<usize>], n_tokens: usize,
        pool: &mut crate::kernels::scratch::BufPool,
    ) -> Grads {
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        let in_dim = q * b;
        let out_dim = p * b;
        dx.iter_mut().for_each(|v| *v = 0.0);
        let (eff1_all, eff2_all) = Self::expand_all_blocks(d1, d2, &self.a1, &self.a2, p, q, m, nd);
        let FwdCache { zs } = cache;

        let run_range = |t0: usize, t1: usize, s1: &mut [f32], s2: &mut [f32], dx_out: &mut [f32]| {
            let mut z_f32 = vec![0.0f32; b];
            for t in t0..t1 {
                let dout_t = &dout[t * out_dim..(t + 1) * out_dim];
                for pp in 0..p {
                    let dout_pp = &dout_t[pp * b..(pp + 1) * b];
                    for &qq in &active_q[t] {
                        debug_assert!(qq < q, "active col-block {qq} out of range");
                        let idx = pp * q + qq;
                        let eff1 = &eff1_all[idx * m * b..(idx + 1) * m * b];
                        let eff2 = &eff2_all[idx * m * b..(idx + 1) * m * b];
                        let x_blk = &x[t * in_dim + qq * b..t * in_dim + (qq + 1) * b];
                        let z16 = &zs[((t * p + pp) * q + qq) * b..((t * p + pp) * q + qq + 1) * b];
                        crate::kernels::f16_simd::f16_to_f32(z16, &mut z_f32);
                        let dx_blk = &mut dx_out[(t - t0) * in_dim + qq * b..(t - t0) * in_dim + (qq + 1) * b];
                        let s1_blk = &mut s1[idx * m * b..(idx + 1) * m * b];
                        let s2_blk = &mut s2[idx * m * b..(idx + 1) * m * b];
                        SharedMonarchMatmul::backward_block_phase1(eff1, eff2, x_blk, &z_f32, dout_pp, dx_blk, m, s1_blk, s2_blk);
                    }
                }
            }
        };

        if n_tokens < Self::PARALLEL_THRESHOLD {
            let mut s1 = vec![0.0f32; p * q * m * b];
            let mut s2 = vec![0.0f32; p * q * m * b];
            let mut dx_full = vec![0.0f32; n_tokens * in_dim];
            run_range(0, n_tokens, &mut s1, &mut s2, &mut dx_full);
            dx.copy_from_slice(&dx_full);
            pool.give_f16(zs);
            return SharedMonarchMatmul::contract_all_blocks(d1, d2, &self.a1, &self.a2, &s1, &s2, p, q, m, nd);
        }

        use rayon::prelude::*;
        let n_chunks = rayon::current_num_threads().max(1).min(n_tokens);
        let chunk_len = n_tokens.div_ceil(n_chunks);
        let results: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> =
            (0..n_chunks).into_par_iter().map(|c| {
                let t0 = c * chunk_len;
                let t1 = ((c + 1) * chunk_len).min(n_tokens);
                let mut s1 = vec![0.0f32; p * q * m * b];
                let mut s2 = vec![0.0f32; p * q * m * b];
                let mut dx_chunk = vec![0.0f32; t1.saturating_sub(t0) * in_dim];
                run_range(t0, t1, &mut s1, &mut s2, &mut dx_chunk);
                (s1, s2, dx_chunk, t0, t1)
            }).collect();

        let mut s1_all = vec![0.0f32; p * q * m * b];
        let mut s2_all = vec![0.0f32; p * q * m * b];
        for (s1, s2, dx_chunk, t0, t1) in results {
            for i in 0..s1_all.len() { s1_all[i] += s1[i]; s2_all[i] += s2[i]; }
            dx[t0 * in_dim..t1 * in_dim].copy_from_slice(&dx_chunk);
        }
        pool.give_f16(zs);
        SharedMonarchMatmul::contract_all_blocks(d1, d2, &self.a1, &self.a2, &s1_all, &s2_all, p, q, m, nd)
    }

    /// Shared per-(pp,qq)-block backward step, factored out of `backward` so
    /// `backward_rows`/`backward_cols` can reuse it over a restricted loop
    /// without duplicating the AVX2/scalar dispatch.
    #[allow(clippy::too_many_arguments)]
    fn backward_block(
        &self, d1: &[f32], d2: &[f32], x: &[f32], zs: &[f32], dout_pp: &[f32], dx: &mut [f32],
        pp: usize, qq: usize, g: &mut Grads,
        dz: &mut [f32], dy1: &mut [f32], eff_j: &mut [f32], eff_i: &mut [f32],
    ) {
        let (q, m, nd) = (self.q, self.m, self.nd);
        let b = m * m;
        let bk      = pp * q + qq;
        let z       = &zs[bk * b..(bk + 1) * b];
        let x_blk   = &x[qq * b..(qq + 1) * b];
        let da_base = bk * m * nd;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if m == 8 {
            unsafe {
                let dd1_ptr = g.dd1.as_mut_ptr();
                let dd2_ptr = g.dd2.as_mut_ptr();
                let da1_ptr = g.da1.as_mut_ptr().add(da_base);
                let da2_ptr = g.da2.as_mut_ptr().add(da_base);
                let dx_ptr  = dx.as_mut_ptr().add(qq * b);
                bwd_block_avx2(
                    d1.as_ptr(), d2.as_ptr(),
                    self.a1_blk(pp, qq).as_ptr(), self.a2_blk(pp, qq).as_ptr(),
                    x_blk.as_ptr(), z.as_ptr(),
                    dout_pp.as_ptr(), nd,
                    da1_ptr, da2_ptr, dd1_ptr, dd2_ptr,
                    dx_ptr,
                );
            }
            return;
        }

        // Scalar fallback (m≠8 or non-x86)
        dz.fill(0.0);
        for j in 0..m {
            let zj     = &z[j * m..(j + 1) * m];
            let dout_j = &dout_pp[j * m..(j + 1) * m];
            eff_j.fill(0.0);
            for d in 0..nd {
                let a = self.a2_blk(pp, qq)[j * nd + d];
                let atom = &d2[d * b..d * b + b];
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
                    let drow = &d2[(d * m + r) * m..(d * m + r) * m + m];
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
                let atom = &d1[d * b..d * b + b];
                for e in 0..b { eff_i[e] += a * atom[e]; }
            }
            for r in 0..m {
                let d_y = dy1_i[r];
                if d_y == 0.0 { continue; }
                for d in 0..nd {
                    let a    = self.a1_blk(pp, qq)[i * nd + d];
                    let drow = &d1[(d * m + r) * m..(d * m + r) * m + m];
                    let u    = gemm::dot(drow, xi);
                    g.da1[da_base + i * nd + d] += d_y * u;
                    let dd1row = &mut g.dd1[(d * m + r) * m..(d * m + r) * m + m];
                    for c in 0..m { dd1row[c] += d_y * a * xi[c]; }
                }
                let dx_i = &mut dx[qq * b + i * m..qq * b + (i + 1) * m];
                for c in 0..m { dx_i[c] += d_y * eff_i[r * m + c]; }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Hoisted-reconstruction primitives: split "rebuild the effective block from
// the shared dictionary" (weight-only, independent of any token) out of the
// per-token math, so a batched caller can reconstruct once per (pp,qq) block
// and reuse it across every token in the batch. See RESEARCH_LOG.md
// (2026-07-03, Fable review) — the un-hoisted per-token reconstruction was
// the dominant cost of the real training-throughput regression.
// ---------------------------------------------------------------------------

impl SharedMonarchMatmul {
    /// Rebuild the `m` stage-1 and `m` stage-2 effective row/col matrices for
    /// one `(pp,qq)` block from the shared dictionary. `eff1`/`eff2` must
    /// each be `m*b` long. Weight-only — safe to call once per block and
    /// reuse across an entire token batch.
    pub fn expand_block(
        d1: &[f32], d2: &[f32], a1_blk: &[f32], a2_blk: &[f32],
        m: usize, nd: usize, eff1: &mut [f32], eff2: &mut [f32],
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if m == 8 {
            unsafe { expand_block_avx2(d1, d2, a1_blk, a2_blk, nd, eff1, eff2); }
            return;
        }
        let b = m * m;
        for i in 0..m {
            let e = &mut eff1[i * b..(i + 1) * b];
            e.fill(0.0);
            for d in 0..nd {
                let a = a1_blk[i * nd + d];
                let atom = &d1[d * b..d * b + b];
                for c in 0..b { e[c] += a * atom[c]; }
            }
        }
        for j in 0..m {
            let e = &mut eff2[j * b..(j + 1) * b];
            e.fill(0.0);
            for d in 0..nd {
                let a = a2_blk[j * nd + d];
                let atom = &d2[d * b..d * b + b];
                for c in 0..b { e[c] += a * atom[c]; }
            }
        }
    }

    /// Apply an already-expanded block to one token's input, accumulating
    /// into `out` (matches `fwd_block`'s `+=` convention on `out`).
    pub fn apply_block(
        eff1: &[f32], eff2: &[f32], x_blk: &[f32], m: usize,
        y1: &mut [f32], z: &mut [f32], out: &mut [f32],
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if m == 8 {
            unsafe { apply_block_avx2(eff1, eff2, x_blk, y1, z, out); }
            return;
        }
        let b = m * m;
        for i in 0..m {
            let xi = &x_blk[i * m..(i + 1) * m];
            let e = &eff1[i * b..(i + 1) * b];
            for r in 0..m {
                let mut acc = 0.0f32;
                for c in 0..m { acc += e[r * m + c] * xi[c]; }
                y1[i * m + r] = acc;
            }
        }
        for i in 0..m { for j in 0..m { z[j * m + i] = y1[i * m + j]; } }
        for j in 0..m {
            let zj = &z[j * m..(j + 1) * m];
            let e = &eff2[j * b..(j + 1) * b];
            for r in 0..m {
                let mut acc = 0.0f32;
                for c in 0..m { acc += e[r * m + c] * zj[c]; }
                out[j * m + r] += acc;
            }
        }
    }

    /// VJP for one `(pp,qq,token)` triple against an already-expanded block.
    /// `da_base`/`g_da1`/`g_da2` follow `backward_block`'s convention (full
    /// model-wide `da1`/`da2` slices, indexed at this block's offset);
    /// `g_dd1`/`g_dd2` are the shared-dictionary gradient accumulators
    /// (summed across every block and token that reads them). `dz`/`dy1`
    /// are scratch, caller-owned to avoid a per-call allocation.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_block_hoisted(
        d1: &[f32], d2: &[f32], a1_blk: &[f32], a2_blk: &[f32],
        eff1: &[f32], eff2: &[f32],
        x_blk: &[f32], z: &[f32], dout_pp: &[f32], dx_blk: &mut [f32],
        m: usize, nd: usize, da_base: usize,
        g_da1: &mut [f32], g_da2: &mut [f32], g_dd1: &mut [f32], g_dd2: &mut [f32],
        dz: &mut [f32], dy1: &mut [f32],
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if m == 8 {
            unsafe {
                bwd_block_avx2_hoisted(
                    d1.as_ptr(), d2.as_ptr(),
                    a1_blk.as_ptr(), a2_blk.as_ptr(),
                    eff1.as_ptr(), eff2.as_ptr(),
                    x_blk.as_ptr(), z.as_ptr(), dout_pp.as_ptr(), nd,
                    g_da1.as_mut_ptr().add(da_base), g_da2.as_mut_ptr().add(da_base),
                    g_dd1.as_mut_ptr(), g_dd2.as_mut_ptr(),
                    dx_blk.as_mut_ptr(),
                );
            }
            return;
        }

        let b = m * m;
        dz.fill(0.0);
        for j in 0..m {
            let zj = &z[j * m..(j + 1) * m];
            let dout_j = &dout_pp[j * m..(j + 1) * m];
            let eff_j = &eff2[j * b..(j + 1) * b];
            for r in 0..m {
                let dy = dout_j[r];
                for c in 0..m { dz[j * m + c] += eff_j[r * m + c] * dy; }
            }
            for r in 0..m {
                let dy = dout_j[r];
                if dy == 0.0 { continue; }
                for d in 0..nd {
                    let a = a2_blk[j * nd + d];
                    let drow = &d2[(d * m + r) * m..(d * m + r) * m + m];
                    let u = gemm::dot(drow, zj);
                    g_da2[da_base + j * nd + d] += dy * u;
                    let dd2row = &mut g_dd2[(d * m + r) * m..(d * m + r) * m + m];
                    for c in 0..m { dd2row[c] += dy * a * zj[c]; }
                }
            }
        }
        dy1.fill(0.0);
        for j in 0..m { for i in 0..m { dy1[i * m + j] = dz[j * m + i]; } }
        for i in 0..m {
            let xi = &x_blk[i * m..(i + 1) * m];
            let dy1_i = &dy1[i * m..(i + 1) * m];
            let eff_i = &eff1[i * b..(i + 1) * b];
            for r in 0..m {
                let d_y = dy1_i[r];
                if d_y == 0.0 { continue; }
                for d in 0..nd {
                    let a = a1_blk[i * nd + d];
                    let drow = &d1[(d * m + r) * m..(d * m + r) * m + m];
                    let u = gemm::dot(drow, xi);
                    g_da1[da_base + i * nd + d] += d_y * u;
                    let dd1row = &mut g_dd1[(d * m + r) * m..(d * m + r) * m + m];
                    for c in 0..m { dd1row[c] += d_y * a * xi[c]; }
                }
                let dx_i = &mut dx_blk[i * m..(i + 1) * m];
                for c in 0..m { dx_i[c] += d_y * eff_i[r * m + c]; }
            }
        }
    }

    /// Phase 1 of the two-phase batched backward (see RESEARCH_LOG.md
    /// 2026-07-03, Fable review — "attack backward, not forward"): computes
    /// `dz`/`dx` exactly as `backward_block_hoisted` does (cheap, uses the
    /// precomputed `eff`), but instead of contracting with the dictionary
    /// per token (the `nd`-scaled cost that dominated backward), accumulates
    /// the per-block outer products `s1 += Σ dy1_i ⊗ x_i`, `s2 += Σ dout_j ⊗
    /// z_j` into caller-owned `s1`/`s2` (`m*b` each, zero-initialized by the
    /// caller, `+=` here — summed across every token in the batch). The
    /// dictionary contraction happens exactly once per block, after the
    /// whole batch, via `contract_block`/`contract_all_blocks` — algebraic
    /// reassociation of the same math, not an approximation (see the
    /// `phase1_plus_contract_matches_backward_block_hoisted` test).
    #[allow(clippy::too_many_arguments)]
    fn backward_block_phase1(
        eff1: &[f32], eff2: &[f32], x_blk: &[f32], z: &[f32], dout_pp: &[f32], dx_blk: &mut [f32],
        m: usize, s1: &mut [f32], s2: &mut [f32],
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if m == 8 {
            unsafe {
                backward_block_phase1_avx2(
                    eff1.as_ptr(), eff2.as_ptr(), x_blk.as_ptr(), z.as_ptr(), dout_pp.as_ptr(),
                    dx_blk.as_mut_ptr(), s1.as_mut_ptr(), s2.as_mut_ptr(),
                );
            }
            return;
        }

        let b = m * m;
        let mut dz = vec![0.0f32; b];
        for j in 0..m {
            let zj = &z[j * m..(j + 1) * m];
            let dout_j = &dout_pp[j * m..(j + 1) * m];
            let eff_j = &eff2[j * b..(j + 1) * b];
            let s2_j = &mut s2[j * b..(j + 1) * b];
            for r in 0..m {
                let dy = dout_j[r];
                for c in 0..m {
                    dz[j * m + c] += eff_j[r * m + c] * dy;
                    s2_j[r * m + c] += dy * zj[c];
                }
            }
        }
        let mut dy1 = vec![0.0f32; b];
        for j in 0..m { for i in 0..m { dy1[i * m + j] = dz[j * m + i]; } }
        for i in 0..m {
            let xi = &x_blk[i * m..(i + 1) * m];
            let dy1_i = &dy1[i * m..(i + 1) * m];
            let eff_i = &eff1[i * b..(i + 1) * b];
            let s1_i = &mut s1[i * b..(i + 1) * b];
            for r in 0..m {
                let d_y = dy1_i[r];
                for c in 0..m {
                    dx_blk[i * m + c] += d_y * eff_i[r * m + c];
                    s1_i[r * m + c] += d_y * xi[c];
                }
            }
        }
    }

    /// Phase 2: contract one block's accumulated outer products (`s1`/`s2`,
    /// `m*b` each, from `backward_block_phase1` summed over the whole batch)
    /// with the dictionary — the `nd`-scaled work, now done once per block
    /// instead of once per token.
    #[allow(clippy::too_many_arguments)]
    fn contract_block(
        d1: &[f32], d2: &[f32], a1_blk: &[f32], a2_blk: &[f32], s1: &[f32], s2: &[f32],
        m: usize, nd: usize, da1: &mut [f32], da2: &mut [f32], dd1: &mut [f32], dd2: &mut [f32],
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if m == 8 {
            unsafe {
                contract_block_avx2(
                    d1.as_ptr(), d2.as_ptr(), a1_blk.as_ptr(), a2_blk.as_ptr(),
                    s1.as_ptr(), s2.as_ptr(), nd,
                    da1.as_mut_ptr(), da2.as_mut_ptr(), dd1.as_mut_ptr(), dd2.as_mut_ptr(),
                );
            }
            return;
        }

        let b = m * m;
        for j in 0..m {
            let s2_j = &s2[j * b..(j + 1) * b];
            for r in 0..m {
                let s_r = &s2_j[r * m..(r + 1) * m];
                for d in 0..nd {
                    let a = a2_blk[j * nd + d];
                    let drow = &d2[(d * m + r) * m..(d * m + r) * m + m];
                    da2[j * nd + d] += gemm::dot(drow, s_r);
                    let dd2row = &mut dd2[(d * m + r) * m..(d * m + r) * m + m];
                    for c in 0..m { dd2row[c] += a * s_r[c]; }
                }
            }
        }
        for i in 0..m {
            let s1_i = &s1[i * b..(i + 1) * b];
            for r in 0..m {
                let s_r = &s1_i[r * m..(r + 1) * m];
                for d in 0..nd {
                    let a = a1_blk[i * nd + d];
                    let drow = &d1[(d * m + r) * m..(d * m + r) * m + m];
                    da1[i * nd + d] += gemm::dot(drow, s_r);
                    let dd1row = &mut dd1[(d * m + r) * m..(d * m + r) * m + m];
                    for c in 0..m { dd1row[c] += a * s_r[c]; }
                }
            }
        }
    }

    /// Below this many blocks, rayon dispatch overhead isn't worth it — same
    /// rationale/threshold as the other kernels.
    const CONTRACT_PARALLEL_THRESHOLD: usize = 8;

    /// Contract every `(pp,qq)` block's accumulated `s1_all`/`s2_all`
    /// (`p*q*m*b` each) into a fresh `Grads`. Cost is `P·Q·m·nd`, done once
    /// per step — negligible at toy `nd` (dict_k), but scales linearly with
    /// `nd` and was measured at production width (`dict_k=32`, RESEARCH_LOG
    /// 2026-07-05, Fable review) to be ~6.6% of a whole training step, all
    /// of it serial. Parallelized over `(pp,qq)` blocks: `da1`/`da2` are
    /// written disjointly per block (direct `par_chunks_mut`); `dd1`/`dd2`
    /// are shared accumulators (every block contributes to the same
    /// dictionary), so each chunk accumulates its own local copy, merged
    /// (summed) afterward — same pattern as the two-phase backward's
    /// `s1`/`s2` token-chunk accumulation.
    pub fn contract_all_blocks(
        d1: &[f32], d2: &[f32], a1: &[f32], a2: &[f32], s1_all: &[f32], s2_all: &[f32],
        p: usize, q: usize, m: usize, nd: usize,
    ) -> Grads {
        let _t = crate::kernels::profiling::Timer::start(&crate::kernels::profiling::MONARCH_CONTRACT);
        let b = m * m;
        let n_blocks = p * q;
        let mut da1 = vec![0.0f32; n_blocks * m * nd];
        let mut da2 = vec![0.0f32; n_blocks * m * nd];

        let run_range = |idx0: usize, idx1: usize, da1_c: &mut [f32], da2_c: &mut [f32], dd1: &mut [f32], dd2: &mut [f32]| {
            for (local_i, idx) in (idx0..idx1).enumerate() {
                let base = idx * m * nd;
                let s1 = &s1_all[idx * m * b..(idx + 1) * m * b];
                let s2 = &s2_all[idx * m * b..(idx + 1) * m * b];
                let da1_blk = &mut da1_c[local_i * m * nd..(local_i + 1) * m * nd];
                let da2_blk = &mut da2_c[local_i * m * nd..(local_i + 1) * m * nd];
                Self::contract_block(
                    d1, d2, &a1[base..base + m * nd], &a2[base..base + m * nd], s1, s2, m, nd,
                    da1_blk, da2_blk, dd1, dd2,
                );
            }
        };

        if n_blocks < Self::CONTRACT_PARALLEL_THRESHOLD {
            let mut dd1 = vec![0.0f32; nd * b];
            let mut dd2 = vec![0.0f32; nd * b];
            run_range(0, n_blocks, &mut da1, &mut da2, &mut dd1, &mut dd2);
            return Grads { dd1, dd2, da1, da2 };
        }

        use rayon::prelude::*;
        let n_chunks = rayon::current_num_threads().max(1).min(n_blocks);
        let chunk_len = n_blocks.div_ceil(n_chunks);
        let results: Vec<(Vec<f32>, Vec<f32>)> = da1
            .par_chunks_mut(chunk_len * m * nd)
            .zip(da2.par_chunks_mut(chunk_len * m * nd))
            .enumerate()
            .map(|(c, (da1_c, da2_c))| {
                let idx0 = c * chunk_len;
                let idx1 = ((c + 1) * chunk_len).min(n_blocks);
                let mut dd1_local = vec![0.0f32; nd * b];
                let mut dd2_local = vec![0.0f32; nd * b];
                run_range(idx0, idx1, da1_c, da2_c, &mut dd1_local, &mut dd2_local);
                (dd1_local, dd2_local)
            })
            .collect();

        let mut dd1 = vec![0.0f32; nd * b];
        let mut dd2 = vec![0.0f32; nd * b];
        for (l1, l2) in results {
            for i in 0..dd1.len() { dd1[i] += l1[i]; }
            for i in 0..dd2.len() { dd2[i] += l2[i]; }
        }
        Grads { dd1, dd2, da1, da2 }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn backward_block_phase1_avx2(
    eff1: *const f32, eff2: *const f32, x_blk: *const f32, z: *const f32, dout_pp: *const f32,
    dx: *mut f32, s1: *mut f32, s2: *mut f32,
) {
    const M: usize = 8;
    const B: usize = 64;
    let mut dz  = [0.0f32; B];
    let mut dy1 = [0.0f32; B];

    for j in 0..M {
        let zj     = z.add(j * M);
        let dout_j = dout_pp.add(j * M);
        let eff    = eff2.add(j * B);
        let dz_j = dz.as_mut_ptr().add(j * M);
        let s2_j = s2.add(j * B);
        for r in 0..M {
            let dy = *dout_j.add(r);
            axpy8(dz_j, eff.add(r * M), dy);
            axpy8(s2_j.add(r * M), zj, dy);
        }
    }
    for i in 0..M { for j in 0..M { dy1[i*M+j] = dz[j*M+i]; } }
    for i in 0..M {
        let xi    = x_blk.add(i * M);
        let dy1_i = dy1.as_ptr().add(i * M);
        let eff   = eff1.add(i * B);
        let s1_i  = s1.add(i * B);
        for r in 0..M {
            let d_y = *dy1_i.add(r);
            axpy8(dx.add(i * M), eff.add(r * M), d_y);
            axpy8(s1_i.add(r * M), xi, d_y);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn contract_block_avx2(
    d1: *const f32, d2: *const f32, a1_blk: *const f32, a2_blk: *const f32,
    s1: *const f32, s2: *const f32, nd: usize,
    da1: *mut f32, da2: *mut f32, dd1: *mut f32, dd2: *mut f32,
) {
    const M: usize = 8;
    const B: usize = 64;
    for j in 0..M {
        let s2_j = s2.add(j * B);
        for r in 0..M {
            let s_r = s2_j.add(r * M);
            for d in 0..nd {
                let a = *a2_blk.add(j * nd + d);
                *da2.add(j * nd + d) += dot8(d2.add((d * M + r) * M), s_r);
                axpy8(dd2.add((d * M + r) * M), s_r, a);
            }
        }
    }
    for i in 0..M {
        let s1_i = s1.add(i * B);
        for r in 0..M {
            let s_r = s1_i.add(r * M);
            for d in 0..nd {
                let a = *a1_blk.add(i * nd + d);
                *da1.add(i * nd + d) += dot8(d1.add((d * M + r) * M), s_r);
                axpy8(dd1.add((d * M + r) * M), s_r, a);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn expand_block_avx2(
    d1: &[f32], d2: &[f32], a1_blk: &[f32], a2_blk: &[f32], nd: usize,
    eff1: &mut [f32], eff2: &mut [f32],
) {
    const M: usize = 8;
    const B: usize = 64;
    for i in 0..M {
        let e = eff1.as_mut_ptr().add(i * B);
        axpy64_init(e, d1.as_ptr(), *a1_blk.get_unchecked(i * nd));
        for d in 1..nd { axpy64(e, d1.as_ptr().add(d * B), *a1_blk.get_unchecked(i * nd + d)); }
    }
    for j in 0..M {
        let e = eff2.as_mut_ptr().add(j * B);
        axpy64_init(e, d2.as_ptr(), *a2_blk.get_unchecked(j * nd));
        for d in 1..nd { axpy64(e, d2.as_ptr().add(d * B), *a2_blk.get_unchecked(j * nd + d)); }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_block_avx2(
    eff1: &[f32], eff2: &[f32], x_blk: &[f32], y1: &mut [f32], z: &mut [f32], out: &mut [f32],
) {
    const M: usize = 8;
    for i in 0..M {
        matvec8(y1.as_mut_ptr().add(i * M), eff1.as_ptr().add(i * 64), x_blk.as_ptr().add(i * M));
    }
    for i in 0..M { for j in 0..M { *z.get_unchecked_mut(j * M + i) = *y1.get_unchecked(i * M + j); } }
    for j in 0..M {
        matvec8_accum(out.as_mut_ptr().add(j * M), eff2.as_ptr().add(j * 64), z.as_ptr().add(j * M));
    }
}

/// Same math as `bwd_block_avx2`, but `eff1`/`eff2` are precomputed
/// (via `expand_block_avx2`) rather than rebuilt from the dictionary on
/// every call — the da/dd accumulation still touches `d1`/`d2` directly
/// since those gradients are inherently per-token (depend on this token's
/// `x`/`z`), but the `dz`/`dx` propagation reuses the hoisted `eff`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bwd_block_avx2_hoisted(
    d1: *const f32, d2: *const f32,
    a1_blk: *const f32, a2_blk: *const f32,
    eff1: *const f32, eff2: *const f32,
    x_blk: *const f32, z: *const f32,
    dout_pp: *const f32, nd: usize,
    da1: *mut f32, da2: *mut f32,
    dd1: *mut f32, dd2: *mut f32,
    dx: *mut f32,
) {
    const M: usize = 8;
    const B: usize = 64;
    let mut dz  = [0.0f32; B];
    let mut dy1 = [0.0f32; B];

    for j in 0..M {
        let zj     = z.add(j * M);
        let dout_j = dout_pp.add(j * M);
        let eff    = eff2.add(j * B);
        let dz_j = dz.as_mut_ptr().add(j * M);
        for r in 0..M { axpy8(dz_j, eff.add(r * M), *dout_j.add(r)); }
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
        let eff   = eff1.add(i * B);
        for r in 0..M {
            let d_y = *dy1_i.add(r);
            for d in 0..nd {
                let a = *a1_blk.add(i * nd + d);
                *da1.add(i * nd + d) += d_y * dot8(d1.add((d * M + r) * M), xi);
                axpy8(dd1.add((d * M + r) * M), xi, d_y * a);
            }
            axpy8(dx.add(i * M), eff.add(r * M), d_y);
        }
    }
}

/// Research-only variant of `bwd_block_avx2_hoisted` that skips the
/// `dd1`/`dd2` (shared-dictionary gradient) accumulation entirely — i.e. the
/// "frozen dictionary, learn coefficients only" experiment (Opus review,
/// RESEARCH_LOG.md 2026-07-03): `axpy8(dd*, ...)` is exactly half of the
/// da/dd inner-loop work per `(r,d)` pair. Not wired into the live model —
/// freezing the dictionary is a training/capacity decision, not just a perf
/// one, and needs its own evaluation. Used by `bench_frozen_dict` only.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bwd_block_avx2_hoisted_frozen(
    d1: *const f32, d2: *const f32,
    a1_blk: *const f32, a2_blk: *const f32,
    eff1: *const f32, eff2: *const f32,
    x_blk: *const f32, z: *const f32,
    dout_pp: *const f32, nd: usize,
    da1: *mut f32, da2: *mut f32,
    dx: *mut f32,
) {
    const M: usize = 8;
    const B: usize = 64;
    let mut dz  = [0.0f32; B];
    let mut dy1 = [0.0f32; B];

    for j in 0..M {
        let zj     = z.add(j * M);
        let dout_j = dout_pp.add(j * M);
        let eff    = eff2.add(j * B);
        let dz_j = dz.as_mut_ptr().add(j * M);
        for r in 0..M { axpy8(dz_j, eff.add(r * M), *dout_j.add(r)); }
        for r in 0..M {
            let dy = *dout_j.add(r);
            for d in 0..nd {
                *da2.add(j * nd + d) += dy * dot8(d2.add((d * M + r) * M), zj);
            }
        }
    }
    for i in 0..M { for j in 0..M { dy1[i*M+j] = dz[j*M+i]; } }
    for i in 0..M {
        let xi    = x_blk.add(i * M);
        let dy1_i = dy1.as_ptr().add(i * M);
        let eff   = eff1.add(i * B);
        for r in 0..M {
            let d_y = *dy1_i.add(r);
            for d in 0..nd {
                *da1.add(i * nd + d) += d_y * dot8(d1.add((d * M + r) * M), xi);
            }
            axpy8(dx.add(i * M), eff.add(r * M), d_y);
        }
    }
    let _ = (a1_blk, a2_blk); // kept for signature symmetry with the non-frozen variant
}

/// Research-only: `backward_block_hoisted` with the dictionary gradient
/// (`dd1`/`dd2`) skipped — see `bwd_block_avx2_hoisted_frozen`. `m==8` only
/// (no scalar fallback — this is a benchmark probe, not a production path).
#[allow(clippy::too_many_arguments)]
pub fn backward_block_hoisted_frozen(
    d1: &[f32], d2: &[f32], a1_blk: &[f32], a2_blk: &[f32],
    eff1: &[f32], eff2: &[f32],
    x_blk: &[f32], z: &[f32], dout_pp: &[f32], dx_blk: &mut [f32],
    nd: usize, da_base: usize,
    g_da1: &mut [f32], g_da2: &mut [f32],
) {
    unsafe {
        bwd_block_avx2_hoisted_frozen(
            d1.as_ptr(), d2.as_ptr(),
            a1_blk.as_ptr(), a2_blk.as_ptr(),
            eff1.as_ptr(), eff2.as_ptr(),
            x_blk.as_ptr(), z.as_ptr(), dout_pp.as_ptr(), nd,
            g_da1.as_mut_ptr().add(da_base), g_da2.as_mut_ptr().add(da_base),
            dx_blk.as_mut_ptr(),
        );
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

    #[test]
    fn forward_inference_wide_axis_matches_forward() {
        // q > p (like w_down at 1B scale: P=14, Q=48) — exercises the
        // qq-parallel + reduction path.
        let mm = SharedMonarchMatmul::new(2, 5, 8, 4, 0xD0D0);
        let (p, q, m) = (mm.p, mm.q, mm.m);
        assert!(q > p, "test setup should exercise the q>p case");
        let b = m * m;
        let x = randvec(q * b, 0xABCD);

        let wide_out = mm.forward_inference_wide(&x);
        let (forward_out, _) = mm.forward(&x);
        assert_eq!(wide_out.len(), forward_out.len());
        for i in 0..wide_out.len() {
            assert!((wide_out[i] - forward_out[i]).abs() < 1e-5,
                "idx {i}: forward_inference_wide={} forward={}", wide_out[i], forward_out[i]);
        }
    }

    #[test]
    fn forward_inference_serial_matches_parallel() {
        for (p, q) in [(3, 2), (2, 5)] {
            let mm = SharedMonarchMatmul::new(p, q, 8, 4, 0x5E71A1);
            let b = mm.m * mm.m;
            let x = randvec(q * b, 0x1234 + p as u64);
            let par = mm.forward_inference(&x);
            let ser = mm.forward_inference_serial(&x);
            for i in 0..par.len() {
                assert!((par[i] - ser[i]).abs() < 1e-6,
                    "p={p} q={q} idx {i}: parallel={} serial={}", par[i], ser[i]);
            }
        }
    }

    #[test]
    fn forward_inference_grouped_matches_individual_calls() {
        let mm_a = SharedMonarchMatmul::new(3, 2, 8, 4, 0x1111);
        let mm_b = SharedMonarchMatmul::new(3, 2, 8, 4, 0x2222);
        let mm_c = SharedMonarchMatmul::new(3, 2, 8, 4, 0x3333);
        let (p, q, m) = (mm_a.p, mm_a.q, mm_a.m);
        let b = m * m;
        let in_dim = q * b;
        let out_dim = p * b;

        let mut rng = 0xFEEDu64;
        let x: Vec<f32> = (0..in_dim)
            .map(|_| { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); (rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5 })
            .collect();

        let grouped = SharedMonarchMatmul::forward_inference_grouped(&[&mm_a, &mm_b, &mm_c], &x);
        assert_eq!(grouped.len(), 3 * out_dim);

        for (g, mm) in [&mm_a, &mm_b, &mm_c].iter().enumerate() {
            let individual = mm.forward_inference(&x);
            let grouped_g = &grouped[g * out_dim..(g + 1) * out_dim];
            for i in 0..out_dim {
                assert!((grouped_g[i] - individual[i]).abs() < 1e-6,
                    "proj {g} idx {i}: grouped={} individual={}", grouped_g[i], individual[i]);
            }
        }
    }

    #[test]
    fn forward_batch_matches_looped_forward() {
        let mm = SharedMonarchMatmul::new(3, 2, 8, 4, 0xC0FFEE);
        let (p, q, m) = (mm.p, mm.q, mm.m);
        let b = m * m;
        let in_dim = q * b;
        let out_dim = p * b;
        let n_tokens = 5;

        let mut rng = 0xABCDu64;
        let x: Vec<f32> = (0..n_tokens * in_dim)
            .map(|_| { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); (rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5 })
            .collect();

        let (y_batch, cache_batch) = mm.forward_batch(&x, n_tokens);
        assert_eq!(y_batch.len(), n_tokens * out_dim);

        for t in 0..n_tokens {
            let x_t = &x[t * in_dim..(t + 1) * in_dim];
            let (y_t, cache_t) = mm.forward(x_t);
            let y_batch_t = &y_batch[t * out_dim..(t + 1) * out_dim];
            for i in 0..out_dim {
                assert!((y_batch_t[i] - y_t[i]).abs() < 1e-6,
                    "token {t} idx {i}: batch={} looped={}", y_batch_t[i], y_t[i]);
            }
            let zs_batch_t = mm.zs_at(&cache_batch, t);
            assert_eq!(zs_batch_t.len(), cache_t.zs.len());
            for i in 0..zs_batch_t.len() {
                let (a, b) = (zs_batch_t[i].to_f32(), cache_t.zs[i].to_f32());
                assert!((a - b).abs() < 1e-6, "token {t} zs idx {i}: batch={a} looped={b}");
            }
        }
    }

    #[test]
    fn backward_from_batched_cache_matches_backward_from_single_cache() {
        let mm = SharedMonarchMatmul::new(2, 3, 8, 4, 0xFEEDBEEF);
        let (p, q, m) = (mm.p, mm.q, mm.m);
        let b = m * m;
        let in_dim = q * b;
        let out_dim = p * b;
        let n_tokens = 4;

        let mut rng = 0x1234u64;
        let mut next = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); (rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5 };
        let x: Vec<f32> = (0..n_tokens * in_dim).map(|_| next()).collect();
        let dout: Vec<f32> = (0..n_tokens * out_dim).map(|_| next()).collect();

        let (_, cache_batch) = mm.forward_batch(&x, n_tokens);

        for t in 0..n_tokens {
            let x_t = &x[t * in_dim..(t + 1) * in_dim];
            let dout_t = &dout[t * out_dim..(t + 1) * out_dim];
            let (_, cache_t) = mm.forward(x_t);

            let mut dx_looped = vec![0.0f32; in_dim];
            let g_looped = mm.backward(x_t, &cache_t.zs, dout_t, &mut dx_looped);

            let mut dx_batched = vec![0.0f32; in_dim];
            let zs_t = mm.zs_at(&cache_batch, t);
            let g_batched = mm.backward(x_t, zs_t, dout_t, &mut dx_batched);

            for i in 0..dx_looped.len() {
                assert!((dx_looped[i] - dx_batched[i]).abs() < 1e-6, "token {t} dx idx {i}");
            }
            for i in 0..g_looped.da1.len() {
                assert!((g_looped.da1[i] - g_batched.da1[i]).abs() < 1e-6, "token {t} da1 idx {i}");
            }
        }
    }

    #[test]
    fn shared_monarch_proj_matches_shared_monarch_matmul() {
        // Same seeds should produce numerically identical forward output whether
        // the atoms are owned per-instance (SharedMonarchMatmul) or passed in
        // externally (SharedMonarchProj) — proves the dispatch/ownership split
        // didn't change the math.
        let (p, q, m, nd) = (3, 2, 8, 4);
        let mm = SharedMonarchMatmul::new(p, q, m, nd, 0xAAAA);
        let proj = SharedMonarchProj {
            p, q, m, nd,
            a1: mm.a1.clone(),
            a2: mm.a2.clone(),
        };
        let b = m * m;
        let x = randvec(q * b, 0xBEEF);

        let (out_mm, cache_mm) = mm.forward(&x);
        let (out_proj, cache_proj) = proj.forward(&mm.d1, &mm.d2, &x);
        assert_eq!(out_mm, out_proj);
        assert_eq!(cache_mm.zs, cache_proj.zs);

        let dout = randvec(p * b, 0xC0DE);
        let mut dx_mm = vec![0.0f32; q * b];
        let mut dx_proj = vec![0.0f32; q * b];
        let g_mm = mm.backward(&x, &cache_mm.zs, &dout, &mut dx_mm);
        let g_proj = proj.backward(&mm.d1, &mm.d2, &x, &cache_proj.zs, &dout, &mut dx_proj);
        assert_eq!(dx_mm, dx_proj);
        assert_eq!(g_mm.da1, g_proj.da1);
        assert_eq!(g_mm.da2, g_proj.da2);
        assert_eq!(g_mm.dd1, g_proj.dd1);
        assert_eq!(g_mm.dd2, g_proj.dd2);
    }

    #[test]
    fn shared_monarch_proj_gradcheck() {
        // Finite-difference check against SharedMonarchProj::backward, covering
        // gradients w.r.t. the externally-owned dictionary (dd1/dd2) as well as
        // the instance's own coefficients (da1) — the dict gradient path is new
        // code (not exercised by SharedMonarchMatmul's own tests).
        let (p, q, m, nd) = (2, 3, 8, 4);
        let (mut d1, mut d2) = init_shared_atoms(nd, m, 0x5EED);
        let proj = SharedMonarchProj::new(p, q, m, nd, 0x51DE);
        let b = m * m;
        let in_dim = q * b;
        let out_dim = p * b;

        let x = randvec(in_dim, 0x1010);
        let target = randvec(out_dim, 0x2020);
        let eps = 1e-3f32;

        let loss = |out: &[f32]| -> f32 {
            out.iter().zip(&target).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / out.len() as f32
        };
        let dloss = |out: &[f32]| -> Vec<f32> {
            let n = out.len() as f32;
            out.iter().zip(&target).map(|(a, b)| 2.0 * (a - b) / n).collect()
        };

        let (out, cache) = proj.forward(&d1, &d2, &x);
        let dl = dloss(&out);
        let mut dx = vec![0.0f32; in_dim];
        let g = proj.backward(&d1, &d2, &x, &cache.zs, &dl, &mut dx);

        let mut max_err = 0.0f32;
        let mut checked = 0usize;

        for idx in (0..proj.a1.len()).step_by(proj.a1.len() / 8 + 1).take(8) {
            let mut proj_p = SharedMonarchProj { p, q, m, nd, a1: proj.a1.clone(), a2: proj.a2.clone() };
            proj_p.a1[idx] += eps;
            let (out_p, _) = proj_p.forward(&d1, &d2, &x);
            proj_p.a1[idx] -= 2.0 * eps;
            let (out_m, _) = proj_p.forward(&d1, &d2, &x);
            let fd = (loss(&out_p) - loss(&out_m)) / (2.0 * eps);
            max_err = max_err.max((fd - g.da1[idx]).abs());
            checked += 1;
        }

        for idx in (0..d1.len()).step_by(d1.len() / 8 + 1).take(8) {
            let orig = d1[idx];
            d1[idx] = orig + eps;
            let (out_p, _) = proj.forward(&d1, &d2, &x);
            d1[idx] = orig - eps;
            let (out_m, _) = proj.forward(&d1, &d2, &x);
            d1[idx] = orig;
            let fd = (loss(&out_p) - loss(&out_m)) / (2.0 * eps);
            max_err = max_err.max((fd - g.dd1[idx]).abs());
            checked += 1;
        }

        for idx in (0..d2.len()).step_by(d2.len() / 8 + 1).take(8) {
            let orig = d2[idx];
            d2[idx] = orig + eps;
            let (out_p, _) = proj.forward(&d1, &d2, &x);
            d2[idx] = orig - eps;
            let (out_m, _) = proj.forward(&d1, &d2, &x);
            d2[idx] = orig;
            let fd = (loss(&out_p) - loss(&out_m)) / (2.0 * eps);
            max_err = max_err.max((fd - g.dd2[idx]).abs());
            checked += 1;
        }

        assert!(max_err < 0.05, "gradcheck max_err={max_err:.2e} over {checked} params");
    }

    #[test]
    fn forward_rows_matches_full_on_active_blocks() {
        let (p, q, m, nd) = (5, 3, 8, 4);
        let proj = SharedMonarchProj::new(p, q, m, nd, 0x2020);
        let (d1, d2) = init_shared_atoms(nd, m, 0x3030);
        let b = m * m;
        let x = randvec(q * b, 0x4040);
        let active = [1usize, 3, 4];

        let (full, _) = proj.forward(&d1, &d2, &x);
        let (rows, _) = proj.forward_rows(&d1, &d2, &x, &active);

        for &pp in &active {
            for i in 0..b {
                assert!((full[pp * b + i] - rows[pp * b + i]).abs() < 1e-6,
                    "pp={pp} i={i}: full={} rows={}", full[pp * b + i], rows[pp * b + i]);
            }
        }
        // Inactive blocks are exactly zero, not garbage.
        for pp in 0..p {
            if active.contains(&pp) { continue; }
            for i in 0..b {
                assert_eq!(rows[pp * b + i], 0.0, "inactive pp={pp} i={i} should be zero");
            }
        }
    }

    #[test]
    fn forward_cols_matches_zeroed_input() {
        let (p, q, m, nd) = (3, 5, 8, 4);
        let proj = SharedMonarchProj::new(p, q, m, nd, 0x5050);
        let (d1, d2) = init_shared_atoms(nd, m, 0x6060);
        let b = m * m;
        let mut x = randvec(q * b, 0x7070);
        let active = [0usize, 2, 4];

        // Reference: zero out the inactive input blocks, run the full forward.
        let mut x_masked = x.clone();
        for qq in 0..q {
            if !active.contains(&qq) {
                x_masked[qq * b..(qq + 1) * b].fill(0.0);
            }
        }
        let (full_masked, _) = proj.forward(&d1, &d2, &x_masked);
        let (cols, _) = proj.forward_cols(&d1, &d2, &x, &active);

        for i in 0..full_masked.len() {
            assert!((full_masked[i] - cols[i]).abs() < 1e-6,
                "i={i}: masked-full={} forward_cols={}", full_masked[i], cols[i]);
        }
        // forward_cols never reads the inactive blocks of x at all — perturbing
        // them must not change the output.
        for qq in 0..q {
            if active.contains(&qq) { continue; }
            x[qq * b..(qq + 1) * b].fill(999.0);
        }
        let (cols2, _) = proj.forward_cols(&d1, &d2, &x, &active);
        for i in 0..cols.len() {
            assert_eq!(cols[i], cols2[i], "forward_cols read an inactive input block at i={i}");
        }
    }

    #[test]
    fn backward_rows_matches_full_backward_on_active_blocks() {
        let (p, q, m, nd) = (5, 3, 8, 4);
        let proj = SharedMonarchProj::new(p, q, m, nd, 0x8080);
        let (d1, d2) = init_shared_atoms(nd, m, 0x9090);
        let b = m * m;
        let x = randvec(q * b, 0xA0A0);
        let active = [1usize, 3, 4];

        let (_, cache_full) = proj.forward(&d1, &d2, &x);
        let (_, cache_rows) = proj.forward_rows(&d1, &d2, &x, &active);

        // dout is nonzero only on active blocks (matches how the caller would
        // use this — gradient only flows from blocks that were computed).
        let mut dout = vec![0.0f32; p * b];
        let dout_active = randvec(active.len() * b, 0xB0B0);
        for (si, &pp) in active.iter().enumerate() {
            dout[pp * b..(pp + 1) * b].copy_from_slice(&dout_active[si * b..(si + 1) * b]);
        }

        let mut dx_full = vec![0.0f32; q * b];
        let g_full = proj.backward(&d1, &d2, &x, &cache_full.zs, &dout, &mut dx_full);

        let mut dx_rows = vec![0.0f32; q * b];
        let g_rows = proj.backward_rows(&d1, &d2, &x, &cache_rows.zs, &dout, &mut dx_rows, &active);

        for i in 0..dx_full.len() {
            assert!((dx_full[i] - dx_rows[i]).abs() < 1e-6, "dx[{i}]: full={} rows={}", dx_full[i], dx_rows[i]);
        }
        for i in 0..g_full.da1.len() {
            assert!((g_full.da1[i] - g_rows.da1[i]).abs() < 1e-6, "da1[{i}]");
            assert!((g_full.da2[i] - g_rows.da2[i]).abs() < 1e-6, "da2[{i}]");
        }
        for i in 0..g_full.dd1.len() {
            assert!((g_full.dd1[i] - g_rows.dd1[i]).abs() < 1e-5, "dd1[{i}]");
            assert!((g_full.dd2[i] - g_rows.dd2[i]).abs() < 1e-5, "dd2[{i}]");
        }
    }

    #[test]
    fn backward_cols_matches_masked_full_backward() {
        let (p, q, m, nd) = (3, 5, 8, 4);
        let proj = SharedMonarchProj::new(p, q, m, nd, 0xC0C0);
        let (d1, d2) = init_shared_atoms(nd, m, 0xD0D0);
        let b = m * m;
        let mut x = randvec(q * b, 0xE0E0);
        let active = [0usize, 2, 4];
        for qq in 0..q {
            if !active.contains(&qq) { x[qq * b..(qq + 1) * b].fill(0.0); }
        }

        let (_, cache_full) = proj.forward(&d1, &d2, &x);
        let (_, cache_cols) = proj.forward_cols(&d1, &d2, &x, &active);
        let dout = randvec(p * b, 0xF0F0);

        let mut dx_full = vec![0.0f32; q * b];
        let g_full = proj.backward(&d1, &d2, &x, &cache_full.zs, &dout, &mut dx_full);

        let mut dx_cols = vec![0.0f32; q * b];
        let g_cols = proj.backward_cols(&d1, &d2, &x, &cache_cols.zs, &dout, &mut dx_cols, &active);

        // By design (matches BasisMatmul::backward_cols), dx is only populated on
        // active_q — inactive blocks stay exactly zero even though the true
        // unrestricted gradient (dx_full) need not vanish there (x being zero
        // doesn't make dL/dx zero by chain rule; backward_cols deliberately
        // doesn't compute it since the caller's routing decision means it's unused).
        for &qq in &active {
            for i in 0..b {
                let idx = qq * b + i;
                assert!((dx_full[idx] - dx_cols[idx]).abs() < 1e-6,
                    "dx[{idx}] (active qq={qq}): full={} cols={}", dx_full[idx], dx_cols[idx]);
            }
        }
        for qq in 0..q {
            if active.contains(&qq) { continue; }
            for i in 0..b {
                assert_eq!(dx_cols[qq * b + i], 0.0, "inactive qq={qq} dx should stay zero");
            }
        }
        for i in 0..g_full.da1.len() {
            assert!((g_full.da1[i] - g_cols.da1[i]).abs() < 1e-6, "da1[{i}]");
            assert!((g_full.da2[i] - g_cols.da2[i]).abs() < 1e-6, "da2[{i}]");
        }
    }

    fn backward_batch_matches_summed_looped_backward_at(p: usize, q: usize) {
        // forward_batch/backward_batch reconstruct each (pp,qq) block's
        // effective weight once and reuse it across every token, instead of
        // once per (token,pp,qq) — this proves the hoist didn't change the
        // math: backward_batch's grads must equal the sum, over tokens, of
        // independent single-token backward() calls, and dx per token must
        // match exactly (not just in aggregate).
        let (m, nd) = (8, 4);
        let (d1, d2) = init_shared_atoms(nd, m, 0x51DE);
        let proj = SharedMonarchProj::new(p, q, m, nd, 0xF00D);
        let b = m * m;
        let in_dim = q * b;
        let out_dim = p * b;
        let n_tokens = 5;

        let x: Vec<f32> = randvec(n_tokens * in_dim, 0x1010);
        let dout: Vec<f32> = randvec(n_tokens * out_dim, 0x2020);

        let mut pool = crate::kernels::scratch::BufPool::new();
        let (_, cache_batch) = proj.forward_batch(&d1, &d2, &x, n_tokens, &mut pool);
        let mut dx_batch = vec![0.0f32; n_tokens * in_dim];
        let g_batch = proj.backward_batch(&d1, &d2, &x, cache_batch, &dout, &mut dx_batch, n_tokens, &mut pool);

        let mut g_looped = Grads {
            dd1: vec![0.0f32; nd * b],
            dd2: vec![0.0f32; nd * b],
            da1: vec![0.0f32; p * q * m * nd],
            da2: vec![0.0f32; p * q * m * nd],
        };
        for t in 0..n_tokens {
            let x_t = &x[t * in_dim..(t + 1) * in_dim];
            let dout_t = &dout[t * out_dim..(t + 1) * out_dim];
            let (_, cache_t) = proj.forward(&d1, &d2, x_t);
            let mut dx_t = vec![0.0f32; in_dim];
            let g_t = proj.backward(&d1, &d2, x_t, &cache_t.zs, dout_t, &mut dx_t);

            for i in 0..in_dim {
                let got = dx_batch[t * in_dim + i];
                assert!((got - dx_t[i]).abs() < 1e-5, "p={p} q={q} token {t} dx[{i}]: batch={got} looped={}", dx_t[i]);
            }
            for i in 0..g_t.da1.len() { g_looped.da1[i] += g_t.da1[i]; }
            for i in 0..g_t.da2.len() { g_looped.da2[i] += g_t.da2[i]; }
            for i in 0..g_t.dd1.len() { g_looped.dd1[i] += g_t.dd1[i]; }
            for i in 0..g_t.dd2.len() { g_looped.dd2[i] += g_t.dd2[i]; }
        }

        for i in 0..g_batch.da1.len() {
            assert!((g_batch.da1[i] - g_looped.da1[i]).abs() < 1e-4, "p={p} q={q} da1[{i}]: batch={} looped={}", g_batch.da1[i], g_looped.da1[i]);
            assert!((g_batch.da2[i] - g_looped.da2[i]).abs() < 1e-4, "p={p} q={q} da2[{i}]: batch={} looped={}", g_batch.da2[i], g_looped.da2[i]);
        }
        for i in 0..g_batch.dd1.len() {
            assert!((g_batch.dd1[i] - g_looped.dd1[i]).abs() < 1e-4, "p={p} q={q} dd1[{i}]: batch={} looped={}", g_batch.dd1[i], g_looped.dd1[i]);
            assert!((g_batch.dd2[i] - g_looped.dd2[i]).abs() < 1e-4, "p={p} q={q} dd2[{i}]: batch={} looped={}", g_batch.dd2[i], g_looped.dd2[i]);
        }
    }

    #[test]
    fn backward_batch_matches_summed_looped_backward() {
        // p=3 exercises the sequential branch (p < PARALLEL_THRESHOLD=8);
        // p=10 exercises the rayon-parallel branch.
        backward_batch_matches_summed_looped_backward_at(3, 2);
        backward_batch_matches_summed_looped_backward_at(10, 2);
    }

    #[test]
    fn phase1_plus_contract_matches_backward_block_hoisted() {
        // The two-phase restructuring (accumulate per-block outer products
        // across tokens, contract with the dictionary once per block
        // afterward) is an algebraic reassociation of backward_block_hoisted's
        // math, not an approximation — this proves phase1+contract over N
        // tokens equals N calls to backward_block_hoisted summed, at m=8
        // (AVX2 path) and a non-8 m (scalar path).
        for &(m, nd) in &[(8usize, 8usize), (4, 4)] {
            let b = m * m;
            let (p, q) = (2, 3);
            let (d1, d2) = init_shared_atoms(nd, m, 0xC0DE);
            let a1 = randvec(p * q * m * nd, 0x1111);
            let a2 = randvec(p * q * m * nd, 0x2222);
            let n_tokens = 5;

            let (eff1_all, eff2_all) = SharedMonarchProj::expand_all_blocks(&d1, &d2, &a1, &a2, p, q, m, nd);

            let x: Vec<f32> = randvec(n_tokens * q * b, 0x3333);
            let dout: Vec<f32> = randvec(n_tokens * p * b, 0x4444);
            // Fabricate a plausible zs cache: apply each block per token.
            let mut zs = vec![0.0f32; n_tokens * p * q * b];
            for t in 0..n_tokens {
                let mut y1 = vec![0.0f32; b];
                let mut y_dummy = vec![0.0f32; b];
                for pp in 0..p {
                    for qq in 0..q {
                        let idx = pp * q + qq;
                        let z = &mut zs[((t * p + pp) * q + qq) * b..((t * p + pp) * q + qq + 1) * b];
                        SharedMonarchMatmul::apply_block(
                            &eff1_all[idx * m * b..(idx + 1) * m * b], &eff2_all[idx * m * b..(idx + 1) * m * b],
                            &x[t * q * b + qq * b..t * q * b + (qq + 1) * b], m, &mut y1, z, &mut y_dummy,
                        );
                    }
                }
            }

            // Old: N calls to backward_block_hoisted, summed.
            let mut g_old = Grads {
                dd1: vec![0.0f32; nd * b], dd2: vec![0.0f32; nd * b],
                da1: vec![0.0f32; p * q * m * nd], da2: vec![0.0f32; p * q * m * nd],
            };
            let mut dz = vec![0.0f32; b];
            let mut dy1 = vec![0.0f32; b];
            let mut dx_old = vec![0.0f32; n_tokens * q * b];
            for t in 0..n_tokens {
                for pp in 0..p {
                    let dout_pp = &dout[t * p * b + pp * b..t * p * b + (pp + 1) * b];
                    for qq in 0..q {
                        let idx = pp * q + qq;
                        let eff1 = &eff1_all[idx * m * b..(idx + 1) * m * b];
                        let eff2 = &eff2_all[idx * m * b..(idx + 1) * m * b];
                        let x_blk = &x[t * q * b + qq * b..t * q * b + (qq + 1) * b];
                        let z = &zs[((t * p + pp) * q + qq) * b..((t * p + pp) * q + qq + 1) * b];
                        let dx_blk = &mut dx_old[t * q * b + qq * b..t * q * b + (qq + 1) * b];
                        let base = idx * m * nd;
                        SharedMonarchMatmul::backward_block_hoisted(
                            &d1, &d2, &a1[base..base + m * nd], &a2[base..base + m * nd],
                            eff1, eff2, x_blk, z, dout_pp, dx_blk, m, nd, base,
                            &mut g_old.da1, &mut g_old.da2, &mut g_old.dd1, &mut g_old.dd2, &mut dz, &mut dy1,
                        );
                    }
                }
            }

            // New: phase1 (accumulate s1/s2 across all tokens) + one contract_all_blocks.
            let mut s1_all = vec![0.0f32; p * q * m * b];
            let mut s2_all = vec![0.0f32; p * q * m * b];
            let mut dx_new = vec![0.0f32; n_tokens * q * b];
            for t in 0..n_tokens {
                for pp in 0..p {
                    let dout_pp = &dout[t * p * b + pp * b..t * p * b + (pp + 1) * b];
                    for qq in 0..q {
                        let idx = pp * q + qq;
                        let eff1 = &eff1_all[idx * m * b..(idx + 1) * m * b];
                        let eff2 = &eff2_all[idx * m * b..(idx + 1) * m * b];
                        let x_blk = &x[t * q * b + qq * b..t * q * b + (qq + 1) * b];
                        let z = &zs[((t * p + pp) * q + qq) * b..((t * p + pp) * q + qq + 1) * b];
                        let dx_blk = &mut dx_new[t * q * b + qq * b..t * q * b + (qq + 1) * b];
                        let s1 = &mut s1_all[idx * m * b..(idx + 1) * m * b];
                        let s2 = &mut s2_all[idx * m * b..(idx + 1) * m * b];
                        SharedMonarchMatmul::backward_block_phase1(eff1, eff2, x_blk, z, dout_pp, dx_blk, m, s1, s2);
                    }
                }
            }
            let g_new = SharedMonarchMatmul::contract_all_blocks(&d1, &d2, &a1, &a2, &s1_all, &s2_all, p, q, m, nd);

            for i in 0..dx_old.len() {
                assert!((dx_old[i] - dx_new[i]).abs() < 1e-4, "m={m} dx[{i}]: old={} new={}", dx_old[i], dx_new[i]);
            }
            for i in 0..g_old.da1.len() {
                assert!((g_old.da1[i] - g_new.da1[i]).abs() < 1e-3, "m={m} da1[{i}]: old={} new={}", g_old.da1[i], g_new.da1[i]);
                assert!((g_old.da2[i] - g_new.da2[i]).abs() < 1e-3, "m={m} da2[{i}]: old={} new={}", g_old.da2[i], g_new.da2[i]);
            }
            for i in 0..g_old.dd1.len() {
                assert!((g_old.dd1[i] - g_new.dd1[i]).abs() < 1e-3, "m={m} dd1[{i}]: old={} new={}", g_old.dd1[i], g_new.dd1[i]);
                assert!((g_old.dd2[i] - g_new.dd2[i]).abs() < 1e-3, "m={m} dd2[{i}]: old={} new={}", g_old.dd2[i], g_new.dd2[i]);
            }
        }
    }

    #[test]
    fn routed_batch_matches_looped_forward_rows_and_cols() {
        // Ffn's up/gate projections use forward_rows/backward_rows (output
        // routed); down uses forward_cols/backward_cols (input routed). Each
        // token gets a genuinely different active set (mimicking real
        // per-token routing) — proves the *_batch variants' shared dense
        // reconstruction doesn't corrupt the per-token restriction.
        let (p, q, m, nd) = (6, 3, 8, 4);
        let (d1, d2) = init_shared_atoms(nd, m, 0xD1CE);
        let proj = SharedMonarchProj::new(p, q, m, nd, 0xFACE);
        let b = m * m;
        let in_dim = q * b;
        let out_dim = p * b;
        let n_tokens = 7;
        let n_active = 2;

        let x: Vec<f32> = randvec(n_tokens * in_dim, 0x1111);
        let dout: Vec<f32> = randvec(n_tokens * out_dim, 0x2222);
        // Deterministic but token-varying "routing".
        let active_p: Vec<Vec<usize>> = (0..n_tokens).map(|t| {
            (0..n_active).map(|k| (t * 3 + k * 5) % p).collect::<std::collections::BTreeSet<_>>().into_iter().collect()
        }).collect();

        let mut pool = crate::kernels::scratch::BufPool::new();
        let (y_batch, cache_batch) = proj.forward_rows_batch(&d1, &d2, &x, &active_p, n_tokens, &mut pool);
        let mut dx_batch = vec![0.0f32; n_tokens * in_dim];
        let g_batch = proj.backward_rows_batch(&d1, &d2, &x, cache_batch, &dout, &mut dx_batch, &active_p, n_tokens, &mut pool);

        let mut g_looped_da1 = vec![0.0f32; p * q * m * nd];
        let mut g_looped_da2 = vec![0.0f32; p * q * m * nd];
        for t in 0..n_tokens {
            let x_t = &x[t * in_dim..(t + 1) * in_dim];
            let dout_t = &dout[t * out_dim..(t + 1) * out_dim];
            let (y_t, cache_t) = proj.forward_rows(&d1, &d2, x_t, &active_p[t]);
            for i in 0..out_dim {
                let got = y_batch[t * out_dim + i];
                assert!((got - y_t[i]).abs() < 1e-5, "token {t} y[{i}]: batch={got} looped={}", y_t[i]);
            }
            let mut dx_t = vec![0.0f32; in_dim];
            let g_t = proj.backward_rows(&d1, &d2, x_t, &cache_t.zs, dout_t, &mut dx_t, &active_p[t]);
            for i in 0..in_dim {
                let got = dx_batch[t * in_dim + i];
                assert!((got - dx_t[i]).abs() < 1e-5, "token {t} dx[{i}]: batch={got} looped={}", dx_t[i]);
            }
            for i in 0..g_t.da1.len() { g_looped_da1[i] += g_t.da1[i]; g_looped_da2[i] += g_t.da2[i]; }
        }
        for i in 0..g_batch.da1.len() {
            assert!((g_batch.da1[i] - g_looped_da1[i]).abs() < 1e-4, "da1[{i}]: batch={} looped={}", g_batch.da1[i], g_looped_da1[i]);
            assert!((g_batch.da2[i] - g_looped_da2[i]).abs() < 1e-4, "da2[{i}]: batch={} looped={}", g_batch.da2[i], g_looped_da2[i]);
        }

        // Same, for the cols (input-routed, down-proj) variant.
        let active_q: Vec<Vec<usize>> = (0..n_tokens).map(|t| {
            (0..n_active).map(|k| (t * 2 + k * 7) % q).collect::<std::collections::BTreeSet<_>>().into_iter().collect()
        }).collect();
        let (y_batch, cache_batch) = proj.forward_cols_batch(&d1, &d2, &x, &active_q, n_tokens, &mut pool);
        let mut dx_batch = vec![0.0f32; n_tokens * in_dim];
        let g_batch = proj.backward_cols_batch(&d1, &d2, &x, cache_batch, &dout, &mut dx_batch, &active_q, n_tokens, &mut pool);
        for t in 0..n_tokens {
            let x_t = &x[t * in_dim..(t + 1) * in_dim];
            let dout_t = &dout[t * out_dim..(t + 1) * out_dim];
            let (y_t, cache_t) = proj.forward_cols(&d1, &d2, x_t, &active_q[t]);
            for i in 0..out_dim {
                let got = y_batch[t * out_dim + i];
                assert!((got - y_t[i]).abs() < 1e-5, "cols token {t} y[{i}]: batch={got} looped={}", y_t[i]);
            }
            let mut dx_t = vec![0.0f32; in_dim];
            let g_t = proj.backward_cols(&d1, &d2, x_t, &cache_t.zs, dout_t, &mut dx_t, &active_q[t]);
            for i in 0..in_dim {
                let got = dx_batch[t * in_dim + i];
                assert!((got - dx_t[i]).abs() < 1e-5, "cols token {t} dx[{i}]: batch={got} looped={}", dx_t[i]);
            }
        }
    }
}
