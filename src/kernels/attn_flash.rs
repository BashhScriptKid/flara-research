//! Flash attention (full / global) with GQA and causal masking — layers 1-24.
//!
//! Online-softmax, key-tiled attention. For each query row the keys are walked
//! in tiles of `kv_block`; a running `(m, l, acc)` triple carries the softmax
//! max, denominator, and weighted value sum, rescaled when a new block raises
//! the max. This is the FlashAttention-2 forward: never materialize the `T×T`
//! score matrix, keep the working set (one query row + one key/value tile) in
//! L1/L2. On CPU that tiling is the whole point — the algorithm is memory-bound
//! and the tile is sized to stay cache-resident.
//!
//! Layout: `q` is `[T, n_q_heads, head_dim]` row-major; `k`/`v` are
//! `[T, n_kv_heads, head_dim]`. GQA maps query head `h` to KV head
//! `h / (n_q_heads / n_kv_heads)`, so a group of query heads shares one KV head
//! (fewer KV bytes to stream — the CPU-friendly win). Causal: query `i` attends
//! to keys `0..=i`.
//!
//! RoPE is applied by the layer to `q`/`k` *before* this kernel (see
//! [`crate::kernels::rope`]); flash attention sees already-rotated vectors.
//!
//! [`forward`](FlashAttention::forward) returns the per-row log-sum-exp
//! `lse = m + ln(l)` (layout `[T, n_q_heads]`), the statistic the backward pass
//! reloads to recompute softmax probabilities without storing them.

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

/// Configuration + scratch-free flash attention over one sequence.
pub struct FlashAttention {
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Key/value tile length (the cache-resident block).
    pub kv_block: usize,
    scale: f32,
    /// Query heads per KV head (`n_q_heads / n_kv_heads`).
    group: usize,
}



impl FlashAttention {
    pub fn new(n_q_heads: usize, n_kv_heads: usize, head_dim: usize, kv_block: usize) -> Self {
        assert!(n_q_heads % n_kv_heads == 0, "n_q_heads must be a multiple of n_kv_heads");
        assert!(kv_block > 0, "kv_block must be positive");
        Self {
            n_q_heads,
            n_kv_heads,
            head_dim,
            kv_block,
            scale: 1.0 / (head_dim as f32).sqrt(),
            group: n_q_heads / n_kv_heads,
        }
    }

    /// Sequence length implied by a Q buffer.
    #[inline]
    pub fn seq_len(&self, q: &[f32]) -> usize {
        q.len() / (self.n_q_heads * self.head_dim)
    }

    #[inline]
    fn kv_head_of(&self, h_q: usize) -> usize {
        h_q / self.group
    }

    /// Flash attention forward. Writes attention output into `out` (same shape
    /// as `q`) and returns the per-row log-sum-exp, layout `[T, n_q_heads]`.
    pub fn forward(&self, q: &[f32], k: &[f32], v: &[f32], out: &mut [f32]) -> Vec<f32> {
        let (nq, nkv, d) = (self.n_q_heads, self.n_kv_heads, self.head_dim);
        let t = self.seq_len(q);
        assert_eq!(q.len(), t * nq * d, "q shape mismatch");
        assert_eq!(k.len(), t * nkv * d, "k shape mismatch");
        assert_eq!(v.len(), t * nkv * d, "v shape mismatch");
        assert_eq!(out.len(), q.len(), "out shape mismatch");

        let mut lse = vec![0.0f32; t * nq];
        let mut acc = vec![0.0f32; d];
        let mut scores = vec![0.0f32; self.kv_block];

        for h_q in 0..nq {
            let h_kv = self.kv_head_of(h_q);
            for i in 0..t {
                let qi = &q[(i * nq + h_q) * d..(i * nq + h_q) * d + d];

                let mut m = f32::NEG_INFINITY;
                let mut l = 0.0f32;
                for a in acc.iter_mut() {
                    *a = 0.0;
                }

                // Causal: keys 0..=i, walked in tiles.
                let j_end = i + 1;
                let mut j0 = 0;
                while j0 < j_end {
                    let j1 = (j0 + self.kv_block).min(j_end);

                    // Block scores + block max.
                    let mut block_max = f32::NEG_INFINITY;
                    for j in j0..j1 {
                        let kj = &k[(j * nkv + h_kv) * d..(j * nkv + h_kv) * d + d];
                        let s = self.scale * dot(qi, kj);
                        scores[j - j0] = s;
                        block_max = block_max.max(s);
                    }

                    let m_new = m.max(block_max);
                    // exp(-inf) = 0 on the first block, so acc/l start cleanly.
                    let correction = (m - m_new).exp();
                    l *= correction;
                    scale_acc(&mut acc, correction);
                    for j in j0..j1 {
                        let p = (scores[j - j0] - m_new).exp();
                        l += p;
                        let vj = &v[(j * nkv + h_kv) * d..(j * nkv + h_kv) * d + d];
                        axpy(&mut acc, p, vj);
                    }
                    m = m_new;
                    j0 = j1;
                }

                let inv_l = 1.0 / l;
                let o = &mut out[(i * nq + h_q) * d..(i * nq + h_q) * d + d];
                for dd in 0..d {
                    o[dd] = acc[dd] * inv_l;
                }
                lse[i * nq + h_q] = m + l.ln();
            }
        }
        lse
    }

    /// Flash attention backward. Recomputes scores from `q`/`k` and the saved
    /// `lse` — the `T×T` probability matrix is never materialized — and uses the
    /// per-row delta `D_i = Σ_d dO_i[d]·O_i[d]` (= `dot(dO_i, O_i)`) for the
    /// softmax VJP `dS_j = p_j·(dP_j − D_i)`. `out` and `lse` come from
    /// [`forward`](Self::forward). `dq`/`dk`/`dv` are written (zeroed first);
    /// dK/dV accumulate over the query heads sharing each KV head (GQA).
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

        for x in dq.iter_mut() {
            *x = 0.0;
        }
        for x in dk.iter_mut() {
            *x = 0.0;
        }
        for x in dv.iter_mut() {
            *x = 0.0;
        }

        for h_q in 0..nq {
            let h_kv = self.kv_head_of(h_q);
            for i in 0..t {
                let qoff = (i * nq + h_q) * d;
                let qi = &q[qoff..qoff + d];
                let oi = &out[qoff..qoff + d];
                let doi = &d_out[qoff..qoff + d];
                let lse_i = lse[i * nq + h_q];

                // delta D_i = dot(dO_i, O_i)
                let delta = dot(doi, oi);

                for j in 0..=i {
                    let kvoff = (j * nkv + h_kv) * d;
                    let kj = &k[kvoff..kvoff + d];
                    let vj = &v[kvoff..kvoff + d];

                    let s = self.scale * dot(qi, kj);
                    let p = (s - lse_i).exp();
                    let dp = dot(doi, vj);
                    let ds = p * (dp - delta);

                    let dkj = &mut dk[kvoff..kvoff + d];
                    axpy(dkj, self.scale * ds, qi);
                    let dvj = &mut dv[kvoff..kvoff + d];
                    axpy(dvj, p, doi);
                    let dqi = &mut dq[qoff..qoff + d];
                    axpy(dqi, self.scale * ds, kj);
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

    /// Straightforward O(T²) causal GQA softmax attention, for cross-checking.
    fn naive(fa: &FlashAttention, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        let (nq, nkv, d) = (fa.n_q_heads, fa.n_kv_heads, fa.head_dim);
        let t = fa.seq_len(q);
        let mut out = vec![0.0f32; q.len()];
        for h_q in 0..nq {
            let h_kv = h_q / fa.group;
            for i in 0..t {
                let qi = &q[(i * nq + h_q) * d..(i * nq + h_q) * d + d];
                let mut s = vec![0.0f32; i + 1];
                let mut mx = f32::NEG_INFINITY;
                for (j, sj) in s.iter_mut().enumerate() {
                    let kj = &k[(j * nkv + h_kv) * d..(j * nkv + h_kv) * d + d];
                    *sj = fa.scale * dot(qi, kj);
                    mx = mx.max(*sj);
                }
                let mut den = 0.0f32;
                for sj in &mut s {
                    *sj = (*sj - mx).exp();
                    den += *sj;
                }
                let o = &mut out[(i * nq + h_q) * d..(i * nq + h_q) * d + d];
                for (j, &p) in s.iter().enumerate() {
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

    fn rand_buf(rng: &mut Lcg, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.f()).collect()
    }

    #[test]
    fn flash_matches_naive_softmax() {
        let fa = FlashAttention::new(4, 2, 4, 2); // group=2, head_dim=4, kv_block=2
        let (t, nq, nkv, d) = (5, 4, 2, 4);
        let mut rng = Lcg(0x9E37_1234);
        let q = rand_buf(&mut rng, t * nq * d);
        let k = rand_buf(&mut rng, t * nkv * d);
        let v = rand_buf(&mut rng, t * nkv * d);

        let mut out = vec![0.0f32; q.len()];
        fa.forward(&q, &k, &v, &mut out);
        let ref_out = naive(&fa, &q, &k, &v);
        for i in 0..out.len() {
            assert!((out[i] - ref_out[i]).abs() < 1e-5, "out[{i}] {} vs {}", out[i], ref_out[i]);
        }
    }

    #[test]
    fn tiling_block_size_invariant() {
        // The result must not depend on kv_block (tile size is perf-only).
        let (t, nq, nkv, d) = (7, 4, 2, 4);
        let mut rng = Lcg(0x5151_7777);
        let q = rand_buf(&mut rng, t * nq * d);
        let k = rand_buf(&mut rng, t * nkv * d);
        let v = rand_buf(&mut rng, t * nkv * d);

        let mut o1 = vec![0.0f32; q.len()];
        let mut o3 = vec![0.0f32; q.len()];
        FlashAttention::new(nq, nkv, d, 1).forward(&q, &k, &v, &mut o1);
        FlashAttention::new(nq, nkv, d, 3).forward(&q, &k, &v, &mut o3);
        for i in 0..o1.len() {
            assert!((o1[i] - o3[i]).abs() < 1e-5, "block-size dependence at {i}");
        }
    }

    #[test]
    fn causality_future_keys_dont_leak() {
        // Perturbing K/V at a future position must not change earlier outputs.
        let fa = FlashAttention::new(4, 2, 4, 2);
        let (t, nq, nkv, d) = (6, 4, 2, 4);
        let mut rng = Lcg(0xC0DE_0001);
        let q = rand_buf(&mut rng, t * nq * d);
        let k = rand_buf(&mut rng, t * nkv * d);
        let mut v = rand_buf(&mut rng, t * nkv * d);

        let mut base = vec![0.0f32; q.len()];
        fa.forward(&q, &k, &v, &mut base);

        // Clobber position t-1 (last) in V.
        for dd in 0..nkv * d {
            v[(t - 1) * nkv * d + dd] += 3.0;
        }
        let mut perturbed = vec![0.0f32; q.len()];
        fa.forward(&q, &k, &v, &mut perturbed);

        // Rows 0..t-2 must be untouched; row t-1 attends to it and may change.
        for i in 0..t - 1 {
            for h in 0..nq {
                for dd in 0..d {
                    let idx = (i * nq + h) * d + dd;
                    assert!((base[idx] - perturbed[idx]).abs() < 1e-7, "leak at row {i}");
                }
            }
        }
    }

    #[test]
    fn lse_matches_reference() {
        // Returned log-sum-exp must equal log Σ_j exp(scale·q·k_j) over causal j.
        let fa = FlashAttention::new(2, 1, 4, 2);
        let (t, nq, nkv, d) = (4, 2, 1, 4);
        let mut rng = Lcg(0x1A2B_3C4D);
        let q = rand_buf(&mut rng, t * nq * d);
        let k = rand_buf(&mut rng, t * nkv * d);
        let v = rand_buf(&mut rng, t * nkv * d);

        let mut out = vec![0.0f32; q.len()];
        let lse = fa.forward(&q, &k, &v, &mut out);
        for h_q in 0..nq {
            for i in 0..t {
                let qi = &q[(i * nq + h_q) * d..(i * nq + h_q) * d + d];
                let mut sum = 0.0f32;
                let mut mx = f32::NEG_INFINITY;
                let mut ss = vec![0.0f32; i + 1];
                for (j, sj) in ss.iter_mut().enumerate() {
                    let kj = &k[j * nkv * d..j * nkv * d + d];
                    *sj = fa.scale * dot(qi, kj);
                    mx = mx.max(*sj);
                }
                for sj in &ss {
                    sum += (sj - mx).exp();
                }
                let want = mx + sum.ln();
                let got = lse[i * nq + h_q];
                assert!((got - want).abs() < 1e-5, "lse[{i},{h_q}] {got} vs {want}");
            }
        }
    }

    #[test]
    fn backward_gradcheck() {
        let fa = FlashAttention::new(4, 2, 4, 2);
        let (t, nq, nkv, d) = (5, 4, 2, 4);
        let mut rng = Lcg(0xBEEF_0007);
        let q = rand_buf(&mut rng, t * nq * d);
        let k = rand_buf(&mut rng, t * nkv * d);
        let v = rand_buf(&mut rng, t * nkv * d);
        let r = rand_buf(&mut rng, t * nq * d);

        let mut out = vec![0.0f32; q.len()];
        let lse = fa.forward(&q, &k, &v, &mut out);
        let mut dq = vec![0.0f32; q.len()];
        let mut dk = vec![0.0f32; k.len()];
        let mut dv = vec![0.0f32; v.len()];
        fa.backward(&q, &k, &v, &out, &lse, &r, &mut dq, &mut dk, &mut dv);

        let loss = |qq: &[f32], kk: &[f32], vv: &[f32]| -> f32 {
            let mut o = vec![0.0f32; qq.len()];
            fa.forward(qq, kk, vv, &mut o);
            o.iter().zip(&r).map(|(a, b)| a * b).sum()
        };
        const H: f32 = 1e-3;
        let close = |fd: f32, an: f32| (fd - an).abs() < 1e-2 + 5e-2 * an.abs();
        // 0 = q, 1 = k, 2 = v
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

        for &i in &[0usize, 7, 15, 33, 60] {
            assert!(close(central(&q, i, 0), dq[i]), "dq[{i}]: fd vs an {}", dq[i]);
        }
        for &i in &[0usize, 5, 11, 19, 31] {
            assert!(close(central(&k, i, 1), dk[i]), "dk[{i}]: fd vs an {}", dk[i]);
            assert!(close(central(&v, i, 2), dv[i]), "dv[{i}]: fd vs an {}", dv[i]);
        }
    }
}
