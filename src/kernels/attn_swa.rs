//! Sliding-window attention (SWA) with GQA — layers 25-96.
//!
//! Identical online-softmax machinery to [`crate::kernels::attn_flash`], except
//! each query `i` attends only to the causal window of the last `window` keys,
//! `j ∈ [i−window+1, i]` (clamped at 0). Bounding the key span to a constant
//! `window` turns attention from `O(T²)` into `O(T·window)` and — the part that
//! matters on CPU — caps the KV bytes streamed per query at `window·head_dim`,
//! keeping the working set cache-resident regardless of sequence length. Most
//! layers are windowed; a few full-attention layers (1-24) carry global context.
//!
//! Layout and GQA mapping match flash exactly. RoPE is applied externally by the
//! layer before this kernel. [`forward`](SlidingWindowAttention::forward)
//! returns the per-row log-sum-exp for the backward pass.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    crate::kernels::gemm::dot(a, b)
}

#[inline]
fn axpy(acc: &mut [f32], alpha: f32, x: &[f32]) {
    debug_assert_eq!(acc.len(), x.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                let n8 = acc.len() - (acc.len() % 8);
                let mut j = 0;
                while j < n8 {
                    let a_v = _mm256_loadu_ps(acc.as_ptr().add(j));
                    let x_v = _mm256_loadu_ps(x.as_ptr().add(j));
                    let alpha_v = _mm256_set1_ps(alpha);
                    _mm256_storeu_ps(acc.as_mut_ptr().add(j), _mm256_fmadd_ps(alpha_v, x_v, a_v));
                    j += 8;
                }
                for jj in j..acc.len() {
                    acc[jj] += alpha * x[jj];
                }
                return;
            }
        }
    }
    for i in 0..acc.len() {
        acc[i] += alpha * x[i];
    }
}

#[inline]
fn scale_acc(acc: &mut [f32], c: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                let cv = _mm256_set1_ps(c);
                let n8 = acc.len() - (acc.len() % 8);
                let mut j = 0;
                while j < n8 {
                    let a = _mm256_loadu_ps(acc.as_ptr().add(j));
                    _mm256_storeu_ps(acc.as_mut_ptr().add(j), _mm256_mul_ps(a, cv));
                    j += 8;
                }
                for jj in j..acc.len() {
                    acc[jj] *= c;
                }
                return;
            }
        }
    }
    for a in acc.iter_mut() {
        *a *= c;
    }
}

/// Sliding-window causal attention over one sequence.
pub struct SlidingWindowAttention {
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Number of keys in the causal window (including the current position).
    pub window: usize,
    /// Key/value tile length (the cache-resident block).
    pub kv_block: usize,
    scale: f32,
    group: usize,
}



impl SlidingWindowAttention {
    /// Below this many query rows, rayon dispatch overhead isn't worth it —
    /// same threshold/rationale as `SharedMonarchProj` (see monarch.rs) and
    /// `FlashAttention` (see attn_flash.rs).
    const PARALLEL_THRESHOLD: usize = 8;

    pub fn new(
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        window: usize,
        kv_block: usize,
    ) -> Self {
        assert!(n_q_heads % n_kv_heads == 0, "n_q_heads must be a multiple of n_kv_heads");
        assert!(window > 0, "window must be positive");
        assert!(kv_block > 0, "kv_block must be positive");
        Self {
            n_q_heads,
            n_kv_heads,
            head_dim,
            window,
            kv_block,
            scale: 1.0 / (head_dim as f32).sqrt(),
            group: n_q_heads / n_kv_heads,
        }
    }

    #[inline]
    pub fn seq_len(&self, q: &[f32]) -> usize {
        q.len() / (self.n_q_heads * self.head_dim)
    }

    #[inline]
    fn kv_head_of(&self, h_q: usize) -> usize {
        h_q / self.group
    }

    /// First key index in the causal window of query `i`.
    #[inline]
    fn window_start(&self, i: usize) -> usize {
        (i + 1).saturating_sub(self.window)
    }

    /// SWA forward. Writes attention output into `out` (shape of `q`) and returns
    /// the per-row log-sum-exp, layout `[T, n_q_heads]`.
    ///
    /// Parallelizes over query rows (independent per row), same rationale and
    /// row-chunking scheme as `FlashAttention::forward` (see attn_flash.rs).
    pub fn forward(&self, q: &[f32], k: &[f32], v: &[f32], out: &mut [f32]) -> Vec<f32> {
        let (nq, nkv, d) = (self.n_q_heads, self.n_kv_heads, self.head_dim);
        let t = self.seq_len(q);
        assert_eq!(q.len(), t * nq * d, "q shape mismatch");
        assert_eq!(k.len(), t * nkv * d, "k shape mismatch");
        assert_eq!(v.len(), t * nkv * d, "v shape mismatch");
        assert_eq!(out.len(), q.len(), "out shape mismatch");

        let mut lse = vec![0.0f32; t * nq];

        let run_range = |t0: usize, t1: usize, out_c: &mut [f32], lse_c: &mut [f32]| {
            let mut acc = vec![0.0f32; d];
            let mut scores = vec![0.0f32; self.kv_block];
            for i in t0..t1 {
                let ri = i - t0;
                for h_q in 0..nq {
                    let h_kv = self.kv_head_of(h_q);
                    let qoff = (i * nq + h_q) * d;
                    let qi = &q[qoff..qoff + d];

                    let mut m = f32::NEG_INFINITY;
                    let mut l = 0.0f32;
                    for a in acc.iter_mut() {
                        *a = 0.0;
                    }

                    let j_end = i + 1;
                    let mut j0 = self.window_start(i);
                    while j0 < j_end {
                        let j1 = (j0 + self.kv_block).min(j_end);

                        // Register-blocked in groups of 4 keys — see
                        // attn_flash.rs's counterpart for the rationale.
                        let mut block_max = f32::NEG_INFINITY;
                        let mut jj = j0;
                        while jj + 4 <= j1 {
                            let k0 = &k[(jj * nkv + h_kv) * d..(jj * nkv + h_kv) * d + d];
                            let k1 = &k[((jj + 1) * nkv + h_kv) * d..((jj + 1) * nkv + h_kv) * d + d];
                            let k2 = &k[((jj + 2) * nkv + h_kv) * d..((jj + 2) * nkv + h_kv) * d + d];
                            let k3 = &k[((jj + 3) * nkv + h_kv) * d..((jj + 3) * nkv + h_kv) * d + d];
                            let s4 = crate::kernels::gemm::dot4(qi, k0, k1, k2, k3);
                            for lane in 0..4 {
                                let s = self.scale * s4[lane];
                                scores[jj + lane - j0] = s;
                                block_max = block_max.max(s);
                            }
                            jj += 4;
                        }
                        while jj < j1 {
                            let kvoff = (jj * nkv + h_kv) * d;
                            let s = self.scale * dot(qi, &k[kvoff..kvoff + d]);
                            scores[jj - j0] = s;
                            block_max = block_max.max(s);
                            jj += 1;
                        }

                        let m_new = m.max(block_max);
                        let correction = (m - m_new).exp();
                        l *= correction;
                        scale_acc(&mut acc, correction);
                        for j in j0..j1 {
                            let p = (scores[j - j0] - m_new).exp();
                            l += p;
                            let kvoff = (j * nkv + h_kv) * d;
                            let vj = &v[kvoff..kvoff + d];
                            axpy(&mut acc, p, vj);
                        }
                        m = m_new;
                        j0 = j1;
                    }

                    let inv_l = 1.0 / l;
                    let o = &mut out_c[(ri * nq + h_q) * d..(ri * nq + h_q) * d + d];
                    for dd in 0..d {
                        o[dd] = acc[dd] * inv_l;
                    }
                    lse_c[ri * nq + h_q] = m + l.ln();
                }
            }
        };

        if t < Self::PARALLEL_THRESHOLD {
            run_range(0, t, out, &mut lse);
        } else {
            use rayon::prelude::*;
            let n_chunks = rayon::current_num_threads().max(1).min(t);
            let chunk_len = t.div_ceil(n_chunks);
            out.par_chunks_mut(chunk_len * nq * d)
                .zip(lse.par_chunks_mut(chunk_len * nq))
                .enumerate()
                .for_each(|(c, (out_c, lse_c))| {
                    let t0 = c * chunk_len;
                    let t1 = (t0 + chunk_len).min(t);
                    run_range(t0, t1, out_c, lse_c);
                });
        }
        lse
    }

    /// SWA backward. Same delta-trick VJP as flash, restricted to the causal
    /// window. `out`/`lse` come from [`forward`](Self::forward); `dq`/`dk`/`dv`
    /// are written (zeroed first), dK/dV accumulating over GQA-shared heads.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &[f32],
        lse: &[f32],
        d_out: &[f32],
        dq: &mut [f32],
        dk: &mut [f32],
        dv: &mut [f32],
    ) {
        let (nq, nkv, d) = (self.n_q_heads, self.n_kv_heads, self.head_dim);
        let t = self.seq_len(q);
        assert_eq!(k.len(), t * nkv * d, "k shape mismatch");
        assert_eq!(v.len(), t * nkv * d, "v shape mismatch");
        assert_eq!(out.len(), q.len(), "out shape mismatch");
        assert_eq!(lse.len(), t * nq, "lse shape mismatch");
        assert_eq!(d_out.len(), q.len(), "d_out shape mismatch");
        assert_eq!(dq.len(), q.len(), "dq shape mismatch");
        assert_eq!(dk.len(), k.len(), "dk shape mismatch");
        assert_eq!(dv.len(), v.len(), "dv shape mismatch");

        // Same rationale as FlashAttention::backward: dq[i] is independent per
        // row; dk/dv accumulate over every row i whose window covers key j, so
        // a row-chunked parallel split needs per-chunk local dk/dv accumulators
        // merged afterward.
        let run_range = |t0: usize, t1: usize, dq_c: &mut [f32], dk_acc: &mut [f32], dv_acc: &mut [f32]| {
            for i in t0..t1 {
                let ri = i - t0;
                for h_q in 0..nq {
                    let h_kv = self.kv_head_of(h_q);
                    let qoff = (i * nq + h_q) * d;
                    let qi = &q[qoff..qoff + d];
                    let oi = &out[qoff..qoff + d];
                    let doi = &d_out[qoff..qoff + d];
                    let lse_i = lse[i * nq + h_q];

                    let delta = dot(doi, oi);

                    let j_start = self.window_start(i);
                    let j_end = i + 1;
                    let apply_j = |j: usize, s: f32, dp: f32, dk_acc: &mut [f32], dv_acc: &mut [f32], dq_c: &mut [f32]| {
                        let kvoff = (j * nkv + h_kv) * d;
                        let kj = &k[kvoff..kvoff + d];
                        let p = (s - lse_i).exp();
                        let ds = p * (dp - delta);
                        axpy(&mut dk_acc[kvoff..kvoff + d], self.scale * ds, qi);
                        axpy(&mut dv_acc[kvoff..kvoff + d], p, doi);
                        axpy(&mut dq_c[(ri * nq + h_q) * d..(ri * nq + h_q) * d + d], self.scale * ds, kj);
                    };

                    // Register-blocked in groups of 4 keys — see
                    // attn_flash.rs's counterpart for the rationale.
                    let mut jj = j_start;
                    while jj + 4 <= j_end {
                        let kvoff = |j: usize| (j * nkv + h_kv) * d;
                        let k0 = &k[kvoff(jj)..kvoff(jj) + d];
                        let k1 = &k[kvoff(jj + 1)..kvoff(jj + 1) + d];
                        let k2 = &k[kvoff(jj + 2)..kvoff(jj + 2) + d];
                        let k3 = &k[kvoff(jj + 3)..kvoff(jj + 3) + d];
                        let s4 = crate::kernels::gemm::dot4(qi, k0, k1, k2, k3);
                        let v0 = &v[kvoff(jj)..kvoff(jj) + d];
                        let v1 = &v[kvoff(jj + 1)..kvoff(jj + 1) + d];
                        let v2 = &v[kvoff(jj + 2)..kvoff(jj + 2) + d];
                        let v3 = &v[kvoff(jj + 3)..kvoff(jj + 3) + d];
                        let dp4 = crate::kernels::gemm::dot4(doi, v0, v1, v2, v3);
                        for lane in 0..4 {
                            apply_j(jj + lane, self.scale * s4[lane], dp4[lane], dk_acc, dv_acc, dq_c);
                        }
                        jj += 4;
                    }
                    while jj < j_end {
                        let kvoff = (jj * nkv + h_kv) * d;
                        let s = self.scale * dot(qi, &k[kvoff..kvoff + d]);
                        let dp = dot(doi, &v[kvoff..kvoff + d]);
                        apply_j(jj, s, dp, dk_acc, dv_acc, dq_c);
                        jj += 1;
                    }
                }
            }
        };

        for x in dq.iter_mut() {
            *x = 0.0;
        }
        for x in dk.iter_mut() {
            *x = 0.0;
        }
        for x in dv.iter_mut() {
            *x = 0.0;
        }

        if t < Self::PARALLEL_THRESHOLD {
            run_range(0, t, dq, dk, dv);
        } else {
            use rayon::prelude::*;
            let n_chunks = rayon::current_num_threads().max(1).min(t);
            let chunk_len = t.div_ceil(n_chunks);
            let results: Vec<(Vec<f32>, Vec<f32>)> = dq
                .par_chunks_mut(chunk_len * nq * d)
                .enumerate()
                .map(|(c, dq_c)| {
                    let t0 = c * chunk_len;
                    let t1 = (t0 + chunk_len).min(t);
                    let mut dk_local = vec![0.0f32; dk.len()];
                    let mut dv_local = vec![0.0f32; dv.len()];
                    run_range(t0, t1, dq_c, &mut dk_local, &mut dv_local);
                    (dk_local, dv_local)
                })
                .collect();
            for (dk_local, dv_local) in results {
                for i in 0..dk.len() {
                    dk[i] += dk_local[i];
                }
                for i in 0..dv.len() {
                    dv[i] += dv_local[i];
                }
            }
        }
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

    fn rand_buf(rng: &mut Lcg, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.f()).collect()
    }

    /// Naive O(T·window) causal windowed GQA softmax attention.
    fn naive(sa: &SlidingWindowAttention, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        let (nq, nkv, d) = (sa.n_q_heads, sa.n_kv_heads, sa.head_dim);
        let t = sa.seq_len(q);
        let mut out = vec![0.0f32; q.len()];
        for h_q in 0..nq {
            let h_kv = h_q / sa.group;
            for i in 0..t {
                let qi = &q[(i * nq + h_q) * d..(i * nq + h_q) * d + d];
                let j0 = sa.window_start(i);
                let mut s = vec![0.0f32; i + 1 - j0];
                let mut mx = f32::NEG_INFINITY;
                for (idx, sj) in s.iter_mut().enumerate() {
                    let j = j0 + idx;
                    let kj = &k[(j * nkv + h_kv) * d..(j * nkv + h_kv) * d + d];
                    *sj = sa.scale * dot(qi, kj);
                    mx = mx.max(*sj);
                }
                let mut den = 0.0f32;
                for sj in &mut s {
                    *sj = (*sj - mx).exp();
                    den += *sj;
                }
                let o = &mut out[(i * nq + h_q) * d..(i * nq + h_q) * d + d];
                for (idx, &p) in s.iter().enumerate() {
                    let j = j0 + idx;
                    let vj = &v[(j * nkv + h_kv) * d..(j * nkv + h_kv) * d + d];
                    let w = p / den;
                    for dd in 0..d {
                        o[dd] += w * vj[dd];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn swa_matches_naive_windowed() {
        let sa = SlidingWindowAttention::new(4, 2, 4, 3, 2); // window=3, kv_block=2
        let (t, nq, nkv, d) = (8, 4, 2, 4);
        let mut rng = Lcg(0x7A1E_0001);
        let q = rand_buf(&mut rng, t * nq * d);
        let k = rand_buf(&mut rng, t * nkv * d);
        let v = rand_buf(&mut rng, t * nkv * d);

        let mut out = vec![0.0f32; q.len()];
        sa.forward(&q, &k, &v, &mut out);
        let ref_out = naive(&sa, &q, &k, &v);
        for i in 0..out.len() {
            assert!((out[i] - ref_out[i]).abs() < 1e-5, "out[{i}] {} vs {}", out[i], ref_out[i]);
        }
    }

    #[test]
    fn window_excludes_old_keys() {
        // With window W, key 0 is only inside the windows of rows 0..W. Perturbing
        // it must leave every row >= W unchanged.
        let w = 2;
        let sa = SlidingWindowAttention::new(4, 2, 4, w, 2);
        let (t, nq, nkv, d) = (6, 4, 2, 4);
        let mut rng = Lcg(0x01D_CAFE);
        let q = rand_buf(&mut rng, t * nq * d);
        let k = rand_buf(&mut rng, t * nkv * d);
        let mut v = rand_buf(&mut rng, t * nkv * d);

        let mut base = vec![0.0f32; q.len()];
        sa.forward(&q, &k, &v, &mut base);
        for dd in 0..nkv * d {
            v[dd] += 5.0; // clobber key position 0
        }
        let mut perturbed = vec![0.0f32; q.len()];
        sa.forward(&q, &k, &v, &mut perturbed);

        for i in w..t {
            for x in 0..nq * d {
                let idx = i * nq * d + x;
                assert!((base[idx] - perturbed[idx]).abs() < 1e-7, "old key leaked into row {i}");
            }
        }
    }

    #[test]
    fn causality_future_keys_dont_leak() {
        let sa = SlidingWindowAttention::new(4, 2, 4, 3, 2);
        let (t, nq, nkv, d) = (6, 4, 2, 4);
        let mut rng = Lcg(0xC0DE_5A7A);
        let q = rand_buf(&mut rng, t * nq * d);
        let k = rand_buf(&mut rng, t * nkv * d);
        let mut v = rand_buf(&mut rng, t * nkv * d);

        let mut base = vec![0.0f32; q.len()];
        sa.forward(&q, &k, &v, &mut base);
        for dd in 0..nkv * d {
            v[(t - 1) * nkv * d + dd] += 3.0;
        }
        let mut perturbed = vec![0.0f32; q.len()];
        sa.forward(&q, &k, &v, &mut perturbed);
        for i in 0..t - 1 {
            for x in 0..nq * d {
                let idx = i * nq * d + x;
                assert!((base[idx] - perturbed[idx]).abs() < 1e-7, "future key leaked into row {i}");
            }
        }
    }

    #[test]
    fn backward_gradcheck() {
        let sa = SlidingWindowAttention::new(4, 2, 4, 3, 2);
        let (t, nq, nkv, d) = (7, 4, 2, 4);
        let mut rng = Lcg(0xBACC_0009);
        let q = rand_buf(&mut rng, t * nq * d);
        let k = rand_buf(&mut rng, t * nkv * d);
        let v = rand_buf(&mut rng, t * nkv * d);
        let r = rand_buf(&mut rng, t * nq * d);

        let mut out = vec![0.0f32; q.len()];
        let lse = sa.forward(&q, &k, &v, &mut out);
        let mut dq = vec![0.0f32; q.len()];
        let mut dk = vec![0.0f32; k.len()];
        let mut dv = vec![0.0f32; v.len()];
        sa.backward(&q, &k, &v, &out, &lse, &r, &mut dq, &mut dk, &mut dv);

        let loss = |qq: &[f32], kk: &[f32], vv: &[f32]| -> f32 {
            let mut o = vec![0.0f32; qq.len()];
            sa.forward(qq, kk, vv, &mut o);
            o.iter().zip(&r).map(|(a, b)| a * b).sum()
        };
        const H: f32 = 1e-3;
        let close = |fd: f32, an: f32| (fd - an).abs() < 1e-2 + 5e-2 * an.abs();
        let central = |base: &[f32], idx: usize, which: u8| -> f32 {
            let mut bp = base.to_vec();
            bp[idx] += H;
            let lp = match which {
                0 => loss(&bp, &k, &v),
                1 => loss(&q, &bp, &v),
                _ => loss(&q, &k, &bp),
            };
            bp[idx] -= 2.0 * H;
            let lm = match which {
                0 => loss(&bp, &k, &v),
                1 => loss(&q, &bp, &v),
                _ => loss(&q, &k, &bp),
            };
            (lp - lm) / (2.0 * H)
        };

        for &i in &[0usize, 9, 21, 47, 90] {
            assert!(close(central(&q, i, 0), dq[i]), "dq[{i}]: an {}", dq[i]);
        }
        for &i in &[0usize, 7, 15, 27, 41] {
            assert!(close(central(&k, i, 1), dk[i]), "dk[{i}]: an {}", dk[i]);
            assert!(close(central(&v, i, 2), dv[i]), "dv[{i}]: an {}", dv[i]);
        }
    }

    /// `t=16 >= PARALLEL_THRESHOLD` exercises the row-chunked rayon path in
    /// `forward` — checked against the same untouched `naive()` reference the
    /// sequential path is checked against (see attn_flash.rs's counterpart).
    #[test]
    fn forward_parallel_matches_naive_softmax() {
        let sa = SlidingWindowAttention::new(4, 2, 4, 3, 2);
        let (t, nq, nkv, d) = (16, 4, 2, 4);
        let mut rng = Lcg(0x7A11_9101);
        let q = rand_buf(&mut rng, t * nq * d);
        let k = rand_buf(&mut rng, t * nkv * d);
        let v = rand_buf(&mut rng, t * nkv * d);

        let mut out = vec![0.0f32; q.len()];
        sa.forward(&q, &k, &v, &mut out);
        let ref_out = naive(&sa, &q, &k, &v);
        for i in 0..out.len() {
            assert!((out[i] - ref_out[i]).abs() < 1e-5, "out[{i}] {} vs {}", out[i], ref_out[i]);
        }
    }

    /// Same rationale as attn_flash.rs's `backward_parallel_chunking_matches_single_chunk`:
    /// compares a forced single-chunk run against a forced many-chunk run to
    /// isolate the dk/dv chunk-merge logic, sidestepping FD-precision limits at
    /// this scale (see attn_flash.rs for why an FD-based test was dropped).
    #[test]
    fn backward_parallel_chunking_matches_single_chunk() {
        let sa = SlidingWindowAttention::new(4, 2, 4, 3, 2);
        let (t, nq, nkv, d) = (16, 4, 2, 4);
        let mut rng = Lcg(0x7A11_9102);
        let q = rand_buf(&mut rng, t * nq * d);
        let k = rand_buf(&mut rng, t * nkv * d);
        let v = rand_buf(&mut rng, t * nkv * d);
        let r = rand_buf(&mut rng, t * nq * d);

        let mut out = vec![0.0f32; q.len()];
        let lse = sa.forward(&q, &k, &v, &mut out);

        let run = |n_threads: usize| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(n_threads).build().unwrap();
            pool.install(|| {
                let mut dq = vec![0.0f32; q.len()];
                let mut dk = vec![0.0f32; k.len()];
                let mut dv = vec![0.0f32; v.len()];
                sa.backward(&q, &k, &v, &out, &lse, &r, &mut dq, &mut dk, &mut dv);
                (dq, dk, dv)
            })
        };

        let (dq1, dk1, dv1) = run(1);
        let (dq8, dk8, dv8) = run(8);

        for i in 0..dq1.len() {
            assert!((dq1[i] - dq8[i]).abs() < 1e-4, "dq[{i}]: 1-chunk {} vs 8-chunk {}", dq1[i], dq8[i]);
        }
        for i in 0..dk1.len() {
            assert!((dk1[i] - dk8[i]).abs() < 1e-4, "dk[{i}]: 1-chunk {} vs 8-chunk {}", dk1[i], dk8[i]);
            assert!((dv1[i] - dv8[i]).abs() < 1e-4, "dv[{i}]: 1-chunk {} vs 8-chunk {}", dv1[i], dv8[i]);
        }
    }
}
