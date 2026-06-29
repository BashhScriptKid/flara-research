//! Block-routed ReGLU FFN.
//!
//! The FFN intermediate (`F`-dim) is split into `M = F/b` `b`-wide blocks
//! ("micro-experts"). A per-token router picks the top-`n_active` blocks; only
//! those are computed. This is *structured activation sparsity*, not classic
//! MoE — there is no expert replication, so total parameters equal one dense
//! FFN and RAM footprint does not grow. The routing granularity matches the
//! block granularity, which is what makes the skip exact.
//!
//! ```text
//! logits = W_router · h                     (M logits)
//! S      = top-n_active(logits)             (selected block indices)
//! w      = softmax(logits[S])               (gate weights)
//! up     = W_up · h    on blocks S          (forward_rows)
//! gate   = W_gate · h  on blocks S          (forward_rows)
//! act_S  = (ReLU(gate) ⊙ up) · w            (per selected block)
//! out    = W_down · act on blocks S         (forward_cols)
//! ```
//!
//! `W_up`/`W_gate`/`W_down` are BasisMatmul-compressed projections over the
//! shared dictionary (passed in at call time, shared across all layers).
//! Only the router weights and the per-projection coefficients are owned here.

use crate::kernels::fft::{init_coeffs_random, BasisMatmul, PairGrads};
use crate::kernels::optimizer::{AdaFactor, AdaFactorState};
use rustfft::num_complex::Complex32;

/// Static configuration of a block-routed FFN.
#[derive(Clone, Copy)]
pub struct FfnConfig {
    /// Model hidden size `H`.
    pub hidden: usize,
    /// FFN intermediate size `F`.
    pub ffn: usize,
    /// Block size `b` (= routing granularity).
    pub block: usize,
    /// Number of blocks kept active per token (`top-k`).
    pub n_active: usize,
    /// Shared dictionary atom count `K`.
    pub dict_k: usize,
}

impl FfnConfig {
    /// Number of FFN blocks `M = F / b`.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.ffn / self.block
    }
}

/// Forward result, retaining the intermediates the backward pass will need.
pub struct FfnForward {
    /// FFN output, length `H`.
    pub out: Vec<f32>,
    /// Selected block indices (length `n_active`).
    pub selected: Vec<usize>,
    /// Gate weights aligned with `selected` (softmax over selected logits).
    pub gates: Vec<f32>,
    /// Router logits (length `M`) — kept for the router backward.
    pub logits: Vec<f32>,
    /// `W_up · h` on selected blocks (zero elsewhere), length `F`.
    pub up: Vec<f32>,
    /// `W_gate · h` on selected blocks (zero elsewhere), length `F`.
    pub gate: Vec<f32>,
    /// Routed activation fed to `W_down`, length `F`.
    pub act: Vec<f32>,
}

/// Routing decision for one token, produced by [`Ffn::select`] *before* the heavy
/// compute. Exposing this seam lets the caller resolve the router early
/// and prefetch the selected coefficient tiles while their cache lines are
/// still in flight.
pub struct FfnSelection {
    pub selected: Vec<usize>,
    pub gates: Vec<f32>,
    pub logits: Vec<f32>,
}

/// Gradients produced by [`Ffn::backward`]. Coefficient/router gradients are
/// per-layer; `d_dict` accumulates the shared-dictionary contribution
/// from all three projections (the caller sums it across layers). `d_h` is the
/// gradient w.r.t. this layer's input, to flow into the layer below.
pub struct FfnGrads {
    pub d_up_coeffs: Vec<f32>,
    pub d_gate_coeffs: Vec<f32>,
    pub d_down_coeffs: Vec<f32>,
    pub d_dict: Vec<Complex32>,
    /// `M×H` row-major, matching `router_w`.
    pub d_router_w: Vec<f32>,
    /// Gradient w.r.t. the FFN input `h`, length `H`.
    pub d_h: Vec<f32>,
}

/// A block-routed FFN layer. Owns its coefficients and router; the dictionary
/// is shared and supplied to [`Ffn::forward`].
pub struct Ffn {
    pub cfg: FfnConfig,
    /// Shared shape for up and gate (both `F×H`); one matmul serves both so the
    /// fused forward/backward can share the block reconstruction.
    up_gate: BasisMatmul,
    down: BasisMatmul,
    pub up_coeffs: Vec<f32>,
    pub gate_coeffs: Vec<f32>,
    pub down_coeffs: Vec<f32>,
    /// Router weights, `M×H` row-major.
    pub router_w: Vec<f32>,
    m: usize,
}

impl Ffn {
    pub fn new(cfg: FfnConfig, seed: u64) -> Self {
        let (h, f, b, k) = (cfg.hidden, cfg.ffn, cfg.block, cfg.dict_k);
        let up_gate = BasisMatmul::new(f, h, b, k);
        let down = BasisMatmul::new(h, f, b, k);
        let up_coeffs = init_coeffs_random(up_gate.coeff_len(), seed ^ 0x01, 0.02);
        let gate_coeffs = init_coeffs_random(up_gate.coeff_len(), seed ^ 0x02, 0.02);
        let down_coeffs = init_coeffs_random(down.coeff_len(), seed ^ 0x03, 0.02);
        let m = cfg.num_blocks();
        let router_w = init_coeffs_random(m * h, seed ^ 0x04, 0.02);
        Self { cfg, up_gate, down, up_coeffs, gate_coeffs, down_coeffs, router_w, m }
    }

    /// Route the token: returns selected block indices (descending logit) and
    /// the softmax gate weights over them.
    fn route(&self, logits: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let mut idx: Vec<usize> = (0..self.m).collect();
        idx.sort_by(|&a, &c| logits[c].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
        idx.truncate(self.cfg.n_active);

        let maxl = idx.iter().map(|&s| logits[s]).fold(f32::MIN, f32::max);
        let mut gates: Vec<f32> = idx.iter().map(|&s| (logits[s] - maxl).exp()).collect();
        let sum: f32 = gates.iter().sum();
        for g in gates.iter_mut() {
            *g /= sum;
        }
        (idx, gates)
    }

    /// Forward for a single token `h` (length `H`). `dict` is the shared dictionary.
    pub fn select(&self, h: &[f32]) -> FfnSelection {
        debug_assert_eq!(h.len(), self.cfg.hidden);
        let mut logits = vec![0.0f32; self.m];
        for (blk, slot) in logits.iter_mut().enumerate() {
            let row = &self.router_w[blk * self.cfg.hidden..(blk + 1) * self.cfg.hidden];
            *slot = row.iter().zip(h).map(|(w, x)| w * x).sum();
        }
        let (selected, gates) = self.route(&logits);
        FfnSelection { selected, gates, logits }
    }

    /// Software-prefetch the up/gate (row-contiguous) and down (col-strided)
    /// coefficient tiles for the selected blocks into cache.
    pub fn prefetch_coeffs(&self, sel: &FfnSelection) {
        self.up_gate.prefetch_rows(&self.up_coeffs, &sel.selected);
        self.up_gate.prefetch_rows(&self.gate_coeffs, &sel.selected);
        self.down.prefetch_cols(&self.down_coeffs, &sel.selected);
    }

    /// Phase 2 — the compute given a [`FfnSelection`].
    pub fn compute(&self, dict: &[Complex32], h: &[f32], sel: FfnSelection) -> FfnForward {
        let (b, f) = (self.cfg.block, self.cfg.ffn);
        let FfnSelection { selected, gates, logits } = sel;

        let (up, gate) =
            self.up_gate.forward_rows_pair(dict, &self.up_coeffs, &self.gate_coeffs, h, &selected);

        let mut act = vec![0.0f32; f];
        for (si, &blk) in selected.iter().enumerate() {
            let w = gates[si];
            for j in blk * b..(blk + 1) * b {
                let g = gate[j].max(0.0);
                act[j] = g * up[j] * w;
            }
        }

        let out = self.down.forward_cols(dict, &self.down_coeffs, &act, &selected);

        FfnForward { out, selected, gates, logits, up, gate, act }
    }

    /// Convenience forward = [`select`](Self::select) then [`compute`](Self::compute).
    pub fn forward(&self, dict: &[Complex32], h: &[f32]) -> FfnForward {
        let sel = self.select(h);
        self.compute(dict, h, sel)
    }

    /// Batch select: route all `t_len` tokens at once.
    /// `h` is `[t_len, hidden]`. Returns per-token selections.
    pub fn select_batch(&self, h: &[f32], t_len: usize) -> Vec<FfnSelection> {
        let hh = self.cfg.hidden;
        let mut sels = Vec::with_capacity(t_len);
        for t in 0..t_len {
            sels.push(self.select(&h[t * hh..(t + 1) * hh]));
        }
        sels
    }

    /// Batch compute: process all `t_len` tokens through the FFN at once.
    /// Loops over tokens calling per-token compute (BasisMatmul has no batch methods).
    /// `h` is `[t_len, hidden]`, `sels` is per-token selections.
    /// Returns per-token `FfnForward` results.
    pub fn compute_batch(&self, dict: &[Complex32], h: &[f32], sels: &[FfnSelection], t_len: usize) -> Vec<FfnForward> {
        let hh = self.cfg.hidden;
        let mut results = Vec::with_capacity(t_len);
        for t in 0..t_len {
            let fwd = self.compute(dict, &h[t * hh..(t + 1) * hh], FfnSelection {
                selected: sels[t].selected.clone(),
                gates: sels[t].gates.clone(),
                logits: sels[t].logits.clone(),
            });
            results.push(fwd);
        }
        results
    }

    /// Backward for a single token.
    pub fn backward(
        &self,
        dict: &[Complex32],
        h: &[f32],
        fwd: &FfnForward,
        d_out: &[f32],
    ) -> FfnGrads {
        let (b, f, hh) = (self.cfg.block, self.cfg.ffn, self.cfg.hidden);
        debug_assert_eq!(d_out.len(), hh);

        let g_down =
            self.down.backward_cols(dict, &self.down_coeffs, &fwd.act, d_out, &fwd.selected);
        let d_act = &g_down.d_x;

        let mut d_up = vec![0.0f32; f];
        let mut d_gate = vec![0.0f32; f];
        let mut d_w = vec![0.0f32; fwd.selected.len()];
        for (si, &blk) in fwd.selected.iter().enumerate() {
            let w = fwd.gates[si];
            for j in blk * b..(blk + 1) * b {
                let gate_pre = fwd.gate[j];
                let g = gate_pre.max(0.0);
                let upj = fwd.up[j];
                let da = d_act[j];
                d_up[j] = da * g * w;
                if gate_pre > 0.0 {
                    d_gate[j] = da * upj * w;
                }
                d_w[si] += da * g * upj;
            }
        }

        let g_pair: PairGrads = self.up_gate.backward_rows_pair(
            dict,
            &self.up_coeffs,
            &self.gate_coeffs,
            h,
            &d_up,
            &d_gate,
            &fwd.selected,
        );

        let dotp: f32 = d_w.iter().zip(&fwd.gates).map(|(dw, w)| dw * w).sum();
        let mut d_logit = vec![0.0f32; self.m];
        for (si, &blk) in fwd.selected.iter().enumerate() {
            d_logit[blk] = fwd.gates[si] * (d_w[si] - dotp);
        }

        let mut d_router_w = vec![0.0f32; self.m * hh];
        let mut d_h = vec![0.0f32; hh];
        for blk in 0..self.m {
            let dl = d_logit[blk];
            if dl != 0.0 {
                let row = &self.router_w[blk * hh..(blk + 1) * hh];
                let drow = &mut d_router_w[blk * hh..(blk + 1) * hh];
                for i in 0..hh {
                    drow[i] = dl * h[i];
                    d_h[i] += dl * row[i];
                }
            }
        }

        for i in 0..hh {
            d_h[i] += g_pair.d_x[i];
        }

        let mut d_dict = g_down.d_dict;
        for i in 0..d_dict.len() {
            d_dict[i] += g_pair.d_dict[i];
        }

        FfnGrads {
            d_up_coeffs: g_pair.d_coeffs_a,
            d_gate_coeffs: g_pair.d_coeffs_b,
            d_down_coeffs: g_down.d_coeffs,
            d_dict,
            d_router_w,
            d_h,
        }
    }

    /// Allocate this FFN's optimizer state (factored coeffs + router).
    pub fn init_opt(&self) -> FfnOptState {
        let k = self.cfg.dict_k;
        let h = self.cfg.hidden;
        FfnOptState {
            up: AdaFactorState::matrix(self.up_coeffs.len() / k, k, false),
            gate: AdaFactorState::matrix(self.gate_coeffs.len() / k, k, false),
            down: AdaFactorState::matrix(self.down_coeffs.len() / k, k, false),
            router: AdaFactorState::matrix(self.router_w.len() / h, h, false),
        }
    }

    /// Apply one AdaFactor step to the up/gate/down coeffs and the router.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_grad(
        &mut self,
        d_up: &[f32],
        d_gate: &[f32],
        d_down: &[f32],
        d_router: &[f32],
        st: &mut FfnOptState,
        af: &AdaFactor,
        lr: f32,
    ) {
        af.step(&mut self.up_coeffs, d_up, &mut st.up, lr);
        af.step(&mut self.gate_coeffs, d_gate, &mut st.gate, lr);
        af.step(&mut self.down_coeffs, d_down, &mut st.down, lr);
        af.step(&mut self.router_w, d_router, &mut st.router, lr);
    }
}

/// Optimizer state for a block-routed FFN: factored second moments for the three
/// coefficient tensors and the router weight.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct FfnOptState {
    pub up: AdaFactorState,
    pub gate: AdaFactorState,
    pub down: AdaFactorState,
    pub router: AdaFactorState,
}

/// Switch-Transformer load-balancing auxiliary loss over a batch of `T` tokens
/// with `m` blocks. Discourages the router from collapsing onto a few blocks.
///
/// `f_j` is piecewise-constant in the logits (it depends on the discrete
/// top-k), so it is held fixed (straight-through); gradient flows only through
/// `P_j`. Returns `(aux, d_logits)` with `d_logits` shaped `T×M` row-major.
pub fn load_balance_aux(
    logits: &[Vec<f32>],
    selected: &[Vec<usize>],
    m: usize,
) -> (f32, Vec<Vec<f32>>) {
    let t = logits.len();
    assert_eq!(selected.len(), t, "logits/selected token count mismatch");
    if t == 0 {
        return (0.0, Vec::new());
    }
    let inv_t = 1.0 / t as f32;

    let mut probs: Vec<Vec<f32>> = Vec::with_capacity(t);
    let mut big_p = vec![0.0f32; m];
    for lg in logits {
        debug_assert_eq!(lg.len(), m);
        let maxl = lg.iter().cloned().fold(f32::MIN, f32::max);
        let mut p: Vec<f32> = lg.iter().map(|&l| (l - maxl).exp()).collect();
        let s: f32 = p.iter().sum();
        for (pj, big) in p.iter_mut().zip(big_p.iter_mut()) {
            *pj /= s;
            *big += *pj * inv_t;
        }
        probs.push(p);
    }

    let mut f = vec![0.0f32; m];
    for sel in selected {
        for &j in sel {
            f[j] += inv_t;
        }
    }

    let aux = m as f32 * f.iter().zip(&big_p).map(|(fj, pj)| fj * pj).sum::<f32>();

    let scale = m as f32 * inv_t;
    let mut d_logits: Vec<Vec<f32>> = Vec::with_capacity(t);
    for p in &probs {
        let fdot: f32 = f.iter().zip(p).map(|(fk, pk)| fk * pk).sum();
        let row: Vec<f32> = p.iter().zip(&f).map(|(&pj, &fj)| scale * pj * (fj - fdot)).collect();
        d_logits.push(row);
    }
    (aux, d_logits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::fft::init_dict_random;

    fn small_ffn() -> (Ffn, Vec<Complex32>) {
        let cfg = FfnConfig { hidden: 8, ffn: 12, block: 4, n_active: 2, dict_k: 6 };
        let ffn = Ffn::new(cfg, 0xFEED);
        let dict = init_dict_random(cfg.dict_k, cfg.block, 0xD1C7, 0.3);
        (ffn, dict)
    }

    #[test]
    fn routing_invariants() {
        let (ffn, dict) = small_ffn();
        let h = [0.5f32, -1.0, 2.0, 0.3, -0.7, 1.1, 0.0, -0.4];
        let fwd = ffn.forward(&dict, &h);
        assert_eq!(fwd.selected.len(), 2);
        assert_ne!(fwd.selected[0], fwd.selected[1]);
        let gsum: f32 = fwd.gates.iter().sum();
        assert!((gsum - 1.0).abs() < 1e-5, "gates sum {gsum}");
        assert!(fwd.gates.iter().all(|&g| g >= 0.0));
        assert_eq!(fwd.out.len(), 8);
    }

    #[test]
    fn routed_forward_matches_dense_masked_reference() {
        let (ffn, dict) = small_ffn();
        let h = [1.0f32, 0.2, -0.5, 1.3, -1.1, 0.6, 0.9, -0.2];
        let fwd = ffn.forward(&dict, &h);

        let (b, f) = (ffn.cfg.block, ffn.cfg.ffn);
        let up_full = ffn.up_gate.forward(&dict, &ffn.up_coeffs, &h);
        let gate_full = ffn.up_gate.forward(&dict, &ffn.gate_coeffs, &h);
        let mut act_ref = vec![0.0f32; f];
        for (si, &blk) in fwd.selected.iter().enumerate() {
            let w = fwd.gates[si];
            for j in blk * b..(blk + 1) * b {
                act_ref[j] = gate_full[j].max(0.0) * up_full[j] * w;
            }
        }
        let out_ref = ffn.down.forward(&dict, &ffn.down_coeffs, &act_ref);

        for i in 0..ffn.cfg.hidden {
            assert!((fwd.out[i] - out_ref[i]).abs() < 1e-4, "out[{i}] {} vs {}", fwd.out[i], out_ref[i]);
        }
    }

    #[test]
    fn output_responds_to_input() {
        let (ffn, dict) = small_ffn();
        let h0 = [0.1f32; 8];
        let h1 = [0.9f32, -0.9, 0.5, -0.5, 0.3, -0.3, 0.7, -0.7];
        let y0 = ffn.forward(&dict, &h0).out;
        let y1 = ffn.forward(&dict, &h1).out;
        let diff: f32 = y0.iter().zip(&y1).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "output did not respond to input change (diff {diff})");
    }

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
    }

    fn dense_ffn() -> (Ffn, Vec<Complex32>) {
        let cfg = FfnConfig { hidden: 8, ffn: 12, block: 4, n_active: 2, dict_k: 6 };
        let mut ffn = Ffn::new(cfg, 0x51A1);
        let mut rng = Lcg(0x9911_7733);
        for c in ffn.up_coeffs.iter_mut() {
            *c = rng.f() * 0.6;
        }
        for c in ffn.gate_coeffs.iter_mut() {
            *c = rng.f() * 0.6;
        }
        for c in ffn.down_coeffs.iter_mut() {
            *c = rng.f() * 0.6;
        }
        for c in ffn.router_w.iter_mut() {
            *c = rng.f() * 0.6;
        }
        let dict = init_dict_random(cfg.dict_k, cfg.block, 0xD1C7, 0.6);
        (ffn, dict)
    }

    #[test]
    fn split_select_compute_matches_forward_and_prefetch_is_safe() {
        let (ffn, dict) = dense_ffn();
        let mut rng = Lcg(0x5D11_7000);
        let h: Vec<f32> = (0..ffn.cfg.hidden).map(|_| rng.f()).collect();

        let mono = ffn.forward(&dict, &h);

        let sel = ffn.select(&h);
        ffn.prefetch_coeffs(&sel);
        let split = ffn.compute(&dict, &h, sel);

        assert_eq!(mono.selected, split.selected, "selection diverged");
        for (a, b) in mono.out.iter().zip(&split.out) {
            assert!((a - b).abs() < 1e-12, "split output not bit-exact: {a} vs {b}");
        }
    }

    fn loss_and_sel(ffn: &Ffn, dict: &[Complex32], h: &[f32], r: &[f32]) -> (f32, Vec<usize>) {
        let fwd = ffn.forward(dict, h);
        let l: f32 = fwd.out.iter().zip(r).map(|(o, r)| o * r).sum();
        (l, fwd.selected)
    }

    enum P {
        Up,
        Gate,
        Down,
        Router,
    }

    fn p_get(f: &Ffn, w: &P, i: usize) -> f32 {
        match w {
            P::Up => f.up_coeffs[i],
            P::Gate => f.gate_coeffs[i],
            P::Down => f.down_coeffs[i],
            P::Router => f.router_w[i],
        }
    }
    fn p_set(f: &mut Ffn, w: &P, i: usize, v: f32) {
        match w {
            P::Up => f.up_coeffs[i] = v,
            P::Gate => f.gate_coeffs[i] = v,
            P::Down => f.down_coeffs[i] = v,
            P::Router => f.router_w[i] = v,
        }
    }

    fn fd_param(
        ffn: &mut Ffn,
        dict: &[Complex32],
        h: &[f32],
        r: &[f32],
        base_sel: &[usize],
        w: &P,
        i: usize,
        dh: f32,
    ) -> Option<f32> {
        let save = p_get(ffn, w, i);
        p_set(ffn, w, i, save + dh);
        let (lp, sp) = loss_and_sel(ffn, dict, h, r);
        p_set(ffn, w, i, save - dh);
        let (lm, sm) = loss_and_sel(ffn, dict, h, r);
        p_set(ffn, w, i, save);
        if sp.as_slice() == base_sel && sm.as_slice() == base_sel {
            Some((lp - lm) / (2.0 * dh))
        } else {
            None
        }
    }

    #[test]
    fn backward_gradcheck() {
        const DH: f32 = 1e-3;
        let close = |fd: f32, an: f32| (fd - an).abs() < 1e-2 + 5e-2 * an.abs();

        let (mut ffn, dict) = dense_ffn();
        let h = vec![0.7f32, -1.1, 0.4, 0.9, -0.6, 1.2, -0.3, 0.5];
        let r = [0.5f32, -0.8, 1.1, 0.2, -0.4, 0.9, -1.0, 0.3];
        let hh = ffn.cfg.hidden;

        let fwd = ffn.forward(&dict, &h);
        let base_sel = fwd.selected.clone();
        let grads = ffn.backward(&dict, &h, &fwd, &r);

        for &i in &[0usize, 5, 11, 17] {
            for (w, an) in [
                (P::Up, &grads.d_up_coeffs),
                (P::Gate, &grads.d_gate_coeffs),
                (P::Down, &grads.d_down_coeffs),
                (P::Router, &grads.d_router_w),
            ] {
                if i >= an.len() {
                    continue;
                }
                if let Some(fd) = fd_param(&mut ffn, &dict, &h, &r, &base_sel, &w, i, DH) {
                    assert!(close(fd, an[i]), "coeff d[{i}]: fd {fd} vs an {}", an[i]);
                }
            }
        }

        let mut hp = h.clone();
        for i in 0..hh {
            let save = hp[i];
            hp[i] = save + DH;
            let (lp, sp) = loss_and_sel(&ffn, &dict, &hp, &r);
            hp[i] = save - DH;
            let (lm, sm) = loss_and_sel(&ffn, &dict, &hp, &r);
            hp[i] = save;
            if sp == base_sel && sm == base_sel {
                let fd = (lp - lm) / (2.0 * DH);
                assert!(close(fd, grads.d_h[i]), "d_h[{i}]: fd {fd} vs an {}", grads.d_h[i]);
            }
        }

        // Verify d_dict is finite
        for v in &grads.d_dict {
            assert!(v.norm().is_finite(), "d_dict contains non-finite value");
        }
    }

    #[test]
    fn aux_loss_minimized_at_uniform_load() {
        let m = 4;
        let bal_logits = vec![vec![0.0f32; m]; 4];
        let bal_sel = vec![vec![0usize, 1], vec![2, 3], vec![0, 2], vec![1, 3]];
        let (bal, _) = load_balance_aux(&bal_logits, &bal_sel, m);

        let col_logits = vec![vec![5.0f32, 5.0, -5.0, -5.0]; 4];
        let col_sel = vec![vec![0usize, 1]; 4];
        let (col, _) = load_balance_aux(&col_logits, &col_sel, m);

        assert!(bal < col, "balanced aux {bal} should be < collapsed aux {col}");
        assert!((bal - 2.0).abs() < 1e-4, "balanced aux {bal} != uniform minimum 2.0");
    }

    #[test]
    fn aux_loss_gradcheck() {
        const DH: f32 = 1e-3;
        let m = 4;
        let mut rng = Lcg(0x4A11_7C0D);
        let logits: Vec<Vec<f32>> =
            (0..5).map(|_| (0..m).map(|_| rng.f()).collect()).collect();
        let selected =
            vec![vec![0usize, 2], vec![1, 3], vec![0, 1], vec![2, 3], vec![1, 2]];

        let (_, d_logits) = load_balance_aux(&logits, &selected, m);

        for t in 0..logits.len() {
            for j in 0..m {
                let mut lp = logits.clone();
                lp[t][j] += DH;
                let (ap, _) = load_balance_aux(&lp, &selected, m);
                lp[t][j] -= 2.0 * DH;
                let (am, _) = load_balance_aux(&lp, &selected, m);
                let fd = (ap - am) / (2.0 * DH);
                let an = d_logits[t][j];
                assert!((fd - an).abs() < 1e-3, "d_logit[{t}][{j}]: fd {fd} vs an {an}");
            }
        }
    }
}
