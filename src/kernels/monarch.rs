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
    pub zs: Vec<f32>,
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
        (y, FwdCache { zs })
    }

    /// Slice a single token's `zs` out of a batched `forward_batch` cache
    /// (or pass `token = 0` for a single-token `forward` cache).
    #[inline]
    pub fn zs_at<'a>(&self, cache: &'a FwdCache, token: usize) -> &'a [f32] {
        let per_token = self.p * self.q * self.m * self.m;
        &cache.zs[token * per_token..(token + 1) * per_token]
    }

    pub fn backward(&self, x: &[f32], zs: &[f32], dout: &[f32], dx: &mut [f32]) -> Grads {
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
        (y, FwdCache { zs })
    }

    /// Forward for `n_tokens` tokens in one call — see
    /// [`SharedMonarchMatmul::forward_batch`] for why this exists (amortizing
    /// rayon dispatch overhead across a whole sequence instead of paying it
    /// once per token).
    pub fn forward_batch(&self, d1: &[f32], d2: &[f32], x: &[f32], n_tokens: usize) -> (Vec<f32>, FwdCache) {
        let (p, q, m, nd) = (self.p, self.q, self.m, self.nd);
        let b = m * m;
        let in_dim = q * b;
        let mut y   = vec![0.0f32; n_tokens * p * b];
        let mut zs  = vec![0.0f32; n_tokens * p * q * b];

        if n_tokens * p < Self::PARALLEL_THRESHOLD {
            let mut eff = vec![0.0f32; b];
            let mut y1  = vec![0.0f32; b];
            for t in 0..n_tokens {
                let x_t = &x[t * in_dim..(t + 1) * in_dim];
                for pp in 0..p {
                    let ypp = &mut y[(t * p + pp) * b..(t * p + pp + 1) * b];
                    for qq in 0..q {
                        let z = &mut zs[((t * p + pp) * q + qq) * b..((t * p + pp) * q + qq + 1) * b];
                        SharedMonarchMatmul::fwd_block(
                            d1, d2,
                            self.a1_blk(pp, qq), self.a2_blk(pp, qq),
                            &x_t[qq * b..(qq + 1) * b],
                            m, nd, &mut y1, z, ypp, &mut eff,
                        );
                    }
                }
            }
            return (y, FwdCache { zs });
        }

        use rayon::prelude::*;
        let units: Vec<(&mut [f32], &mut [f32])> = y.chunks_mut(b)
            .zip(zs.chunks_mut(q * b))
            .collect();

        units.into_par_iter().enumerate().for_each(|(idx, (ypp, zs_pp))| {
            let t  = idx / p;
            let pp = idx % p;
            let x_t = &x[t * in_dim..(t + 1) * in_dim];
            let mut eff = vec![0.0f32; b];
            let mut y1  = vec![0.0f32; b];
            for qq in 0..q {
                let z = &mut zs_pp[qq * b..(qq + 1) * b];
                SharedMonarchMatmul::fwd_block(
                    d1, d2,
                    self.a1_blk(pp, qq), self.a2_blk(pp, qq),
                    &x_t[qq * b..(qq + 1) * b],
                    m, nd, &mut y1, z, ypp, &mut eff,
                );
            }
        });
        (y, FwdCache { zs })
    }

    /// Slice a single token's `zs` out of a batched `forward_batch` cache (or
    /// pass `token = 0` for a single-token `forward` cache).
    #[inline]
    pub fn zs_at<'a>(&self, cache: &'a FwdCache, token: usize) -> &'a [f32] {
        let per_token = self.p * self.q * self.m * self.m;
        &cache.zs[token * per_token..(token + 1) * per_token]
    }

    pub fn backward(
        &self, d1: &[f32], d2: &[f32], x: &[f32], zs: &[f32], dout: &[f32], dx: &mut [f32],
    ) -> Grads {
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
        (y, FwdCache { zs })
    }

    /// Backward for [`forward_rows`](Self::forward_rows). Only `active_p`
    /// output blocks were produced, so only they contribute gradient; `dx`
    /// still spans the full input (every `qq` fed every active `pp`).
    pub fn backward_rows(
        &self, d1: &[f32], d2: &[f32], x: &[f32], zs: &[f32], dout: &[f32], dx: &mut [f32],
        active_p: &[usize],
    ) -> Grads {
        let (q, m, nd) = (self.q, self.m, self.nd);
        let b = m * m;
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
        &self, d1: &[f32], d2: &[f32], x: &[f32], zs: &[f32], dout: &[f32], dx: &mut [f32],
        active_q: &[usize],
    ) -> Grads {
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
                assert!((zs_batch_t[i] - cache_t.zs[i]).abs() < 1e-6,
                    "token {t} zs idx {i}: batch={} looped={}", zs_batch_t[i], cache_t.zs[i]);
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
}
