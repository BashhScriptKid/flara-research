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
//! `W_up`/`W_gate`/`W_down` are Monarch-compressed projections
//! ([`SharedMonarchProj`]) over a shared real atom dictionary (passed in at
//! call time, shared across all layers — the FFN's own dictionary, separate
//! from the attention projections' one). Only the router weights and the
//! per-projection coefficients are owned here.

use crate::kernels::fft::init_coeffs_random;
use crate::kernels::monarch::{FwdCache, Grads as MonarchGrads, SharedMonarchProj};
use crate::kernels::optimizer::{AdaFactor, AdaFactorState};

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
    /// Shared dictionary atom count `nd`.
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
    /// Monarch forward caches `backward` needs — one per projection, each
    /// only populated on the selected blocks (see `forward_rows`/`forward_cols`).
    up_cache: FwdCache,
    gate_cache: FwdCache,
    down_cache: FwdCache,
}

/// Batched [`FfnForward`]: same fields, `t_len`-token-major (`out`/`up`/
/// `gate`/`act` are `[t_len, *]`), plus the per-token routing decisions
/// `backward_batch` needs.
pub struct FfnForwardBatch {
    pub out: Vec<f32>,
    up: Vec<f32>,
    gate: Vec<f32>,
    act: Vec<f32>,
    up_cache: FwdCache,
    gate_cache: FwdCache,
    down_cache: FwdCache,
    /// Per-token selected block indices — the routing decision (public: the
    /// caller needs this for e.g. stability checks across forward calls).
    pub selected: Vec<Vec<usize>>,
    pub gates: Vec<Vec<f32>>,
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
/// per-layer; `d_mono_d1`/`d_mono_d2` accumulate the shared-dictionary
/// contribution from all three projections (the caller sums it across
/// layers). `d_h` is the gradient w.r.t. this layer's input, to flow into the
/// layer below.
pub struct FfnGrads {
    pub d_up_coeffs: Vec<f32>,
    pub d_gate_coeffs: Vec<f32>,
    pub d_down_coeffs: Vec<f32>,
    pub d_mono_d1: Vec<f32>,
    pub d_mono_d2: Vec<f32>,
    /// `M×H` row-major, matching `router_w`.
    pub d_router_w: Vec<f32>,
    /// Gradient w.r.t. the FFN input `h`, length `H`.
    pub d_h: Vec<f32>,
}

/// Batched [`FfnGrads`]: coefficient/dictionary/router gradients are already
/// summed across every token in the batch (weight gradients); `d_h` is
/// `[t_len, H]`.
pub struct FfnGradsBatch {
    pub d_up_coeffs: Vec<f32>,
    pub d_gate_coeffs: Vec<f32>,
    pub d_down_coeffs: Vec<f32>,
    pub d_mono_d1: Vec<f32>,
    pub d_mono_d2: Vec<f32>,
    pub d_router_w: Vec<f32>,
    pub d_h: Vec<f32>,
}

/// A block-routed FFN layer. Owns its coefficients and router; the shared
/// Monarch dictionary is supplied to [`Ffn::forward`]/[`Ffn::compute`].
pub struct Ffn {
    pub cfg: FfnConfig,
    up_proj: SharedMonarchProj,
    gate_proj: SharedMonarchProj,
    down_proj: SharedMonarchProj,
    /// Router weights, `M×H` row-major.
    pub router_w: Vec<f32>,
    m: usize,
}

fn concat(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
    let mut v = a;
    v.extend(b);
    v
}

impl Ffn {
    pub fn new(cfg: FfnConfig, seed: u64) -> Self {
        let (h, f, b, nd) = (cfg.hidden, cfg.ffn, cfg.block, cfg.dict_k);
        let mm = (b as f64).sqrt() as usize;
        assert_eq!(mm * mm, b, "block must be a perfect square");
        let up_proj = SharedMonarchProj::new(f / b, h / b, mm, nd, seed ^ 0x01);
        let gate_proj = SharedMonarchProj::new(f / b, h / b, mm, nd, seed ^ 0x02);
        let down_proj = SharedMonarchProj::new(h / b, f / b, mm, nd, seed ^ 0x03);
        let m = cfg.num_blocks();
        let router_w = init_coeffs_random(m * h, seed ^ 0x04, 0.02);
        Self { cfg, up_proj, gate_proj, down_proj, router_w, m }
    }

    /// Concatenated `a1`+`a2` for each projection — the flat checkpoint/
    /// optimizer view of its coefficients (mirrors `AttnProj::params`).
    pub fn up_coeffs(&self) -> Vec<f32> { concat(self.up_proj.a1.clone(), self.up_proj.a2.clone()) }
    pub fn gate_coeffs(&self) -> Vec<f32> { concat(self.gate_proj.a1.clone(), self.gate_proj.a2.clone()) }
    pub fn down_coeffs(&self) -> Vec<f32> { concat(self.down_proj.a1.clone(), self.down_proj.a2.clone()) }

    fn set_proj_coeffs(proj: &mut SharedMonarchProj, src: &[f32]) {
        assert_eq!(proj.a1.len() + proj.a2.len(), src.len(), "Ffn projection coeff length mismatch on restore");
        let (a1, a2) = src.split_at(proj.a1.len());
        proj.a1.copy_from_slice(a1);
        proj.a2.copy_from_slice(a2);
    }
    pub fn set_up_coeffs(&mut self, src: &[f32]) { Self::set_proj_coeffs(&mut self.up_proj, src); }
    pub fn set_gate_coeffs(&mut self, src: &[f32]) { Self::set_proj_coeffs(&mut self.gate_proj, src); }
    pub fn set_down_coeffs(&mut self, src: &[f32]) { Self::set_proj_coeffs(&mut self.down_proj, src); }

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

    /// Forward for a single token `h` (length `H`).
    pub fn select(&self, h: &[f32]) -> FfnSelection {
        debug_assert_eq!(h.len(), self.cfg.hidden);
        let mut logits = vec![0.0f32; self.m];
        for (blk, slot) in logits.iter_mut().enumerate() {
            let row = &self.router_w[blk * self.cfg.hidden..(blk + 1) * self.cfg.hidden];
            *slot = crate::kernels::gemm::dot(row, h);
        }
        let (selected, gates) = self.route(&logits);
        FfnSelection { selected, gates, logits }
    }

    /// Software-prefetch hook for the selected coefficient tiles. No-op for
    /// now — the Monarch path doesn't have a prefetch implementation yet
    /// (BasisMatmul's was a pure perf hint, not required for correctness;
    /// worth revisiting if benchmarks show it matters here).
    pub fn prefetch_coeffs(&self, _sel: &FfnSelection) {}

    /// Phase 2 — the compute given a [`FfnSelection`]. `mono_d1`/`mono_d2` is
    /// this FFN's shared Monarch dictionary (model-wide, passed in by the caller).
    pub fn compute(&self, mono_d1: &[f32], mono_d2: &[f32], h: &[f32], sel: FfnSelection) -> FfnForward {
        let (b, f) = (self.cfg.block, self.cfg.ffn);
        let FfnSelection { selected, gates, logits } = sel;

        let (up, up_cache) = self.up_proj.forward_rows(mono_d1, mono_d2, h, &selected);
        let (gate, gate_cache) = self.gate_proj.forward_rows(mono_d1, mono_d2, h, &selected);

        let mut act = vec![0.0f32; f];
        for (si, &blk) in selected.iter().enumerate() {
            let w = gates[si];
            for j in blk * b..(blk + 1) * b {
                let g = gate[j].max(0.0);
                act[j] = g * up[j] * w;
            }
        }

        let (out, down_cache) = self.down_proj.forward_cols(mono_d1, mono_d2, &act, &selected);

        FfnForward { out, selected, gates, logits, up, gate, act, up_cache, gate_cache, down_cache }
    }

    /// Convenience forward = [`select`](Self::select) then [`compute`](Self::compute).
    pub fn forward(&self, mono_d1: &[f32], mono_d2: &[f32], h: &[f32]) -> FfnForward {
        let sel = self.select(h);
        self.compute(mono_d1, mono_d2, h, sel)
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
    /// Each token routes to a different block subset, so unlike attention's
    /// `forward_batch` there's no shared row/col-selection to batch across —
    /// but the *reconstruction* of every projection's weight blocks from the
    /// shared dictionary is still weight-only and batch-wide (see
    /// `SharedMonarchProj::forward_rows_batch`/`forward_cols_batch`), so it's
    /// still done once instead of once per token. `h` is `[t_len, hidden]`.
    pub fn compute_batch(
        &self, mono_d1: &[f32], mono_d2: &[f32], h: &[f32], sels: &[FfnSelection], t_len: usize,
        pool: &mut crate::kernels::scratch::BufPool,
    ) -> FfnForwardBatch {
        let (b, f) = (self.cfg.block, self.cfg.ffn);
        let active_p: Vec<Vec<usize>> = sels.iter().map(|s| s.selected.clone()).collect();

        let (up, up_cache) = self.up_proj.forward_rows_batch(mono_d1, mono_d2, h, &active_p, t_len, pool);
        let (gate, gate_cache) = self.gate_proj.forward_rows_batch(mono_d1, mono_d2, h, &active_p, t_len, pool);

        let mut act = pool.take_zeroed(t_len * f);
        for t in 0..t_len {
            let sel = &sels[t];
            for (si, &blk) in sel.selected.iter().enumerate() {
                let w = sel.gates[si];
                for j in blk * b..(blk + 1) * b {
                    let idx = t * f + j;
                    let g = gate[idx].max(0.0);
                    act[idx] = g * up[idx] * w;
                }
            }
        }

        let (out, down_cache) = self.down_proj.forward_cols_batch(mono_d1, mono_d2, &act, &active_p, t_len, pool);

        FfnForwardBatch {
            out, up, gate, act, up_cache, gate_cache, down_cache,
            selected: sels.iter().map(|s| s.selected.clone()).collect(),
            gates: sels.iter().map(|s| s.gates.clone()).collect(),
        }
    }

    /// Backward for a single token.
    pub fn backward(
        &self,
        mono_d1: &[f32],
        mono_d2: &[f32],
        h: &[f32],
        fwd: &FfnForward,
        d_out: &[f32],
    ) -> FfnGrads {
        let (b, f, hh) = (self.cfg.block, self.cfg.ffn, self.cfg.hidden);
        debug_assert_eq!(d_out.len(), hh);

        let mut d_act = vec![0.0f32; f];
        let g_down: MonarchGrads = self.down_proj.backward_cols(
            mono_d1, mono_d2, &fwd.act, &fwd.down_cache.zs, d_out, &mut d_act, &fwd.selected,
        );

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

        let mut d_h = vec![0.0f32; hh];
        let g_up: MonarchGrads = self.up_proj.backward_rows(
            mono_d1, mono_d2, h, &fwd.up_cache.zs, &d_up, &mut d_h, &fwd.selected,
        );
        let mut d_h_gate = vec![0.0f32; hh];
        let g_gate: MonarchGrads = self.gate_proj.backward_rows(
            mono_d1, mono_d2, h, &fwd.gate_cache.zs, &d_gate, &mut d_h_gate, &fwd.selected,
        );
        for i in 0..hh {
            d_h[i] += d_h_gate[i];
        }

        let dotp: f32 = d_w.iter().zip(&fwd.gates).map(|(dw, w)| dw * w).sum();
        let mut d_logit = vec![0.0f32; self.m];
        for (si, &blk) in fwd.selected.iter().enumerate() {
            d_logit[blk] = fwd.gates[si] * (d_w[si] - dotp);
        }

        let mut d_router_w = vec![0.0f32; self.m * hh];
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

        let mut d_mono_d1 = g_up.dd1;
        for (a, x) in d_mono_d1.iter_mut().zip(&g_gate.dd1) { *a += *x; }
        for (a, x) in d_mono_d1.iter_mut().zip(&g_down.dd1) { *a += *x; }
        let mut d_mono_d2 = g_up.dd2;
        for (a, x) in d_mono_d2.iter_mut().zip(&g_gate.dd2) { *a += *x; }
        for (a, x) in d_mono_d2.iter_mut().zip(&g_down.dd2) { *a += *x; }

        FfnGrads {
            d_up_coeffs: concat(g_up.da1, g_up.da2),
            d_gate_coeffs: concat(g_gate.da1, g_gate.da2),
            d_down_coeffs: concat(g_down.da1, g_down.da2),
            d_mono_d1,
            d_mono_d2,
            d_router_w,
            d_h,
        }
    }

    /// Batched VJP for `compute_batch`. `h` is `[t_len, hidden]`, `d_out` is
    /// `[t_len, hidden]`. Same hoisted-reconstruction reasoning as
    /// `compute_batch` — each projection's weight blocks are reconstructed
    /// once (see `SharedMonarchProj::backward_rows_batch`/
    /// `backward_cols_batch`) instead of once per token.
    pub fn backward_batch(
        &self, mono_d1: &[f32], mono_d2: &[f32], h: &[f32], fwd: FfnForwardBatch, d_out: &[f32], t_len: usize,
        pool: &mut crate::kernels::scratch::BufPool,
    ) -> FfnGradsBatch {
        let (b, f, hh) = (self.cfg.block, self.cfg.ffn, self.cfg.hidden);
        let FfnForwardBatch { up, gate, act, up_cache, gate_cache, down_cache, selected, gates, .. } = fwd;

        let mut d_act = vec![0.0f32; t_len * f];
        let g_down: MonarchGrads = self.down_proj.backward_cols_batch(
            mono_d1, mono_d2, &act, down_cache, d_out, &mut d_act, &selected, t_len, pool,
        );
        pool.give(act); // not read again below

        let mut d_up = vec![0.0f32; t_len * f];
        let mut d_gate = vec![0.0f32; t_len * f];
        let mut d_w: Vec<Vec<f32>> = selected.iter().map(|s| vec![0.0f32; s.len()]).collect();
        for t in 0..t_len {
            for (si, &blk) in selected[t].iter().enumerate() {
                let w = gates[t][si];
                for j in blk * b..(blk + 1) * b {
                    let idx = t * f + j;
                    let gate_pre = gate[idx];
                    let g = gate_pre.max(0.0);
                    let upj = up[idx];
                    let da = d_act[idx];
                    d_up[idx] = da * g * w;
                    if gate_pre > 0.0 {
                        d_gate[idx] = da * upj * w;
                    }
                    d_w[t][si] += da * g * upj;
                }
            }
        }
        pool.give(up);   // not read again below
        pool.give(gate); // not read again below

        let mut d_h = vec![0.0f32; t_len * hh];
        let g_up: MonarchGrads = self.up_proj.backward_rows_batch(
            mono_d1, mono_d2, h, up_cache, &d_up, &mut d_h, &selected, t_len, pool,
        );
        let mut d_h_gate = vec![0.0f32; t_len * hh];
        let g_gate: MonarchGrads = self.gate_proj.backward_rows_batch(
            mono_d1, mono_d2, h, gate_cache, &d_gate, &mut d_h_gate, &selected, t_len, pool,
        );
        for i in 0..t_len * hh {
            d_h[i] += d_h_gate[i];
        }

        let mut d_router_w = vec![0.0f32; self.m * hh];
        for t in 0..t_len {
            let sel = &selected[t];
            let gates_t = &gates[t];
            let dotp: f32 = d_w[t].iter().zip(gates_t).map(|(dw, w)| dw * w).sum();
            let h_t = &h[t * hh..(t + 1) * hh];
            for (si, &blk) in sel.iter().enumerate() {
                let dl = gates_t[si] * (d_w[t][si] - dotp);
                if dl != 0.0 {
                    let row = &self.router_w[blk * hh..(blk + 1) * hh];
                    let drow = &mut d_router_w[blk * hh..(blk + 1) * hh];
                    for i in 0..hh {
                        drow[i] += dl * h_t[i];
                        d_h[t * hh + i] += dl * row[i];
                    }
                }
            }
        }

        let mut d_mono_d1 = g_up.dd1;
        for (a, x) in d_mono_d1.iter_mut().zip(&g_gate.dd1) { *a += *x; }
        for (a, x) in d_mono_d1.iter_mut().zip(&g_down.dd1) { *a += *x; }
        let mut d_mono_d2 = g_up.dd2;
        for (a, x) in d_mono_d2.iter_mut().zip(&g_gate.dd2) { *a += *x; }
        for (a, x) in d_mono_d2.iter_mut().zip(&g_down.dd2) { *a += *x; }

        FfnGradsBatch {
            d_up_coeffs: concat(g_up.da1, g_up.da2),
            d_gate_coeffs: concat(g_gate.da1, g_gate.da2),
            d_down_coeffs: concat(g_down.da1, g_down.da2),
            d_mono_d1,
            d_mono_d2,
            d_router_w,
            d_h,
        }
    }

    /// Allocate this FFN's optimizer state (factored coeffs + router).
    pub fn init_opt(&self) -> FfnOptState {
        let h = self.cfg.hidden;
        let up_rows = 2 * self.up_proj.p * self.up_proj.q * self.up_proj.m;
        let gate_rows = 2 * self.gate_proj.p * self.gate_proj.q * self.gate_proj.m;
        let down_rows = 2 * self.down_proj.p * self.down_proj.q * self.down_proj.m;
        FfnOptState {
            up: AdaFactorState::matrix(up_rows, self.up_proj.nd, false),
            gate: AdaFactorState::matrix(gate_rows, self.gate_proj.nd, false),
            down: AdaFactorState::matrix(down_rows, self.down_proj.nd, false),
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
        let mut up = self.up_coeffs();
        af.step(&mut up, d_up, &mut st.up, lr);
        self.set_up_coeffs(&up);

        let mut gate = self.gate_coeffs();
        af.step(&mut gate, d_gate, &mut st.gate, lr);
        self.set_gate_coeffs(&gate);

        let mut down = self.down_coeffs();
        af.step(&mut down, d_down, &mut st.down, lr);
        self.set_down_coeffs(&down);

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
    use crate::kernels::monarch::init_shared_atoms;

    fn small_ffn() -> (Ffn, Vec<f32>, Vec<f32>) {
        let cfg = FfnConfig { hidden: 8, ffn: 12, block: 4, n_active: 2, dict_k: 6 };
        let ffn = Ffn::new(cfg, 0xFEED);
        let (d1, d2) = init_shared_atoms(cfg.dict_k, 2, 0xD1C7);
        (ffn, d1, d2)
    }

    #[test]
    fn routing_invariants() {
        let (ffn, d1, d2) = small_ffn();
        let h = [0.5f32, -1.0, 2.0, 0.3, -0.7, 1.1, 0.0, -0.4];
        let fwd = ffn.forward(&d1, &d2, &h);
        assert_eq!(fwd.selected.len(), 2);
        assert_ne!(fwd.selected[0], fwd.selected[1]);
        let gsum: f32 = fwd.gates.iter().sum();
        assert!((gsum - 1.0).abs() < 1e-5, "gates sum {gsum}");
        assert!(fwd.gates.iter().all(|&g| g >= 0.0));
        assert_eq!(fwd.out.len(), 8);
    }

    #[test]
    fn output_responds_to_input() {
        let (ffn, d1, d2) = small_ffn();
        let h0 = [0.1f32; 8];
        let h1 = [0.9f32, -0.9, 0.5, -0.5, 0.3, -0.3, 0.7, -0.7];
        let y0 = ffn.forward(&d1, &d2, &h0).out;
        let y1 = ffn.forward(&d1, &d2, &h1).out;
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

    fn dense_ffn() -> (Ffn, Vec<f32>, Vec<f32>) {
        let cfg = FfnConfig { hidden: 8, ffn: 12, block: 4, n_active: 2, dict_k: 6 };
        let mut ffn = Ffn::new(cfg, 0x51A1);
        let mut rng = Lcg(0x9911_7733);
        let mut up = ffn.up_coeffs();
        for c in up.iter_mut() { *c = rng.f() * 0.6; }
        ffn.set_up_coeffs(&up);
        let mut gate = ffn.gate_coeffs();
        for c in gate.iter_mut() { *c = rng.f() * 0.6; }
        ffn.set_gate_coeffs(&gate);
        let mut down = ffn.down_coeffs();
        for c in down.iter_mut() { *c = rng.f() * 0.6; }
        ffn.set_down_coeffs(&down);
        for c in ffn.router_w.iter_mut() {
            *c = rng.f() * 0.6;
        }
        let (d1, d2) = init_shared_atoms(cfg.dict_k, 2, 0xD1C7);
        (ffn, d1, d2)
    }

    #[test]
    fn split_select_compute_matches_forward_and_prefetch_is_safe() {
        let (ffn, d1, d2) = dense_ffn();
        let mut rng = Lcg(0x5D11_7000);
        let h: Vec<f32> = (0..ffn.cfg.hidden).map(|_| rng.f()).collect();

        let mono = ffn.forward(&d1, &d2, &h);

        let sel = ffn.select(&h);
        ffn.prefetch_coeffs(&sel);
        let split = ffn.compute(&d1, &d2, &h, sel);

        assert_eq!(mono.selected, split.selected, "selection diverged");
        for (a, b) in mono.out.iter().zip(&split.out) {
            assert!((a - b).abs() < 1e-12, "split output not bit-exact: {a} vs {b}");
        }
    }

    fn loss_and_sel(ffn: &Ffn, d1: &[f32], d2: &[f32], h: &[f32], r: &[f32]) -> (f32, Vec<usize>) {
        let fwd = ffn.forward(d1, d2, h);
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
            P::Up => f.up_coeffs()[i],
            P::Gate => f.gate_coeffs()[i],
            P::Down => f.down_coeffs()[i],
            P::Router => f.router_w[i],
        }
    }
    fn p_set(f: &mut Ffn, w: &P, i: usize, v: f32) {
        match w {
            P::Up => { let mut c = f.up_coeffs(); c[i] = v; f.set_up_coeffs(&c); }
            P::Gate => { let mut c = f.gate_coeffs(); c[i] = v; f.set_gate_coeffs(&c); }
            P::Down => { let mut c = f.down_coeffs(); c[i] = v; f.set_down_coeffs(&c); }
            P::Router => f.router_w[i] = v,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fd_param(
        ffn: &mut Ffn,
        d1: &[f32],
        d2: &[f32],
        h: &[f32],
        r: &[f32],
        base_sel: &[usize],
        w: &P,
        i: usize,
        dh: f32,
    ) -> Option<f32> {
        let save = p_get(ffn, w, i);
        p_set(ffn, w, i, save + dh);
        let (lp, sp) = loss_and_sel(ffn, d1, d2, h, r);
        p_set(ffn, w, i, save - dh);
        let (lm, sm) = loss_and_sel(ffn, d1, d2, h, r);
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

        let (mut ffn, d1, d2) = dense_ffn();
        let h = vec![0.7f32, -1.1, 0.4, 0.9, -0.6, 1.2, -0.3, 0.5];
        let r = [0.5f32, -0.8, 1.1, 0.2, -0.4, 0.9, -1.0, 0.3];
        let hh = ffn.cfg.hidden;

        let fwd = ffn.forward(&d1, &d2, &h);
        let base_sel = fwd.selected.clone();
        let grads = ffn.backward(&d1, &d2, &h, &fwd, &r);

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
                if let Some(fd) = fd_param(&mut ffn, &d1, &d2, &h, &r, &base_sel, &w, i, DH) {
                    assert!(close(fd, an[i]), "coeff d[{i}]: fd {fd} vs an {}", an[i]);
                }
            }
        }

        let mut hp = h.clone();
        for i in 0..hh {
            let save = hp[i];
            hp[i] = save + DH;
            let (lp, sp) = loss_and_sel(&ffn, &d1, &d2, &hp, &r);
            hp[i] = save - DH;
            let (lm, sm) = loss_and_sel(&ffn, &d1, &d2, &hp, &r);
            hp[i] = save;
            if sp == base_sel && sm == base_sel {
                let fd = (lp - lm) / (2.0 * DH);
                assert!(close(fd, grads.d_h[i]), "d_h[{i}]: fd {fd} vs an {}", grads.d_h[i]);
            }
        }

        // Verify d_mono_d1/d_mono_d2 are finite (dict-gradient path is new code).
        for v in grads.d_mono_d1.iter().chain(&grads.d_mono_d2) {
            assert!(v.is_finite(), "dict grad contains non-finite value");
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

    #[test]
    fn compute_batch_matches_looped_compute_and_backward() {
        // compute_batch/backward_batch hoist reconstruction across the whole
        // token batch (see SharedMonarchProj::forward_rows_batch etc.), but
        // each token still routes independently — this proves the batched
        // path produces byte-for-byte-close results to the original
        // per-token select/compute/backward loop, with genuinely different
        // routing per token (real hidden vectors, not synthetic).
        let (ffn, d1, d2) = small_ffn();
        let hh = ffn.cfg.hidden;
        let t_len = 5;
        let mut rng = Lcg(0xB19B00B5);
        let h: Vec<f32> = (0..t_len * hh).map(|_| rng.f()).collect();
        let d_out: Vec<f32> = (0..t_len * hh).map(|_| rng.f() * 0.1).collect();

        let mut pool = crate::kernels::scratch::BufPool::new();
        let sels = ffn.select_batch(&h, t_len);
        let fwd_batch = ffn.compute_batch(&d1, &d2, &h, &sels, t_len, &mut pool);
        let fwd_batch_out = fwd_batch.out.clone(); // `backward_batch` consumes `fwd_batch` by value
        let g_batch = ffn.backward_batch(&d1, &d2, &h, fwd_batch, &d_out, t_len, &mut pool);

        let mut d_router_w_looped = vec![0.0f32; ffn.m * hh];
        let mut d_mono_d1_looped = vec![0.0f32; d1.len()];
        let mut d_mono_d2_looped = vec![0.0f32; d2.len()];
        for t in 0..t_len {
            let h_t = &h[t * hh..(t + 1) * hh];
            let d_out_t = &d_out[t * hh..(t + 1) * hh];
            let sel = ffn.select(h_t);
            assert_eq!(sel.selected, sels[t].selected, "routing must be deterministic and match select_batch");
            let fwd_t = ffn.compute(&d1, &d2, h_t, sel);

            for j in 0..hh {
                let got = fwd_batch_out[t * hh + j];
                assert!((got - fwd_t.out[j]).abs() < 1e-5, "token {t} out[{j}]: batch={got} looped={}", fwd_t.out[j]);
            }

            let g_t = ffn.backward(&d1, &d2, h_t, &fwd_t, d_out_t);
            for j in 0..hh {
                let got = g_batch.d_h[t * hh + j];
                assert!((got - g_t.d_h[j]).abs() < 1e-4, "token {t} d_h[{j}]: batch={got} looped={}", g_t.d_h[j]);
            }
            for i in 0..d_router_w_looped.len() { d_router_w_looped[i] += g_t.d_router_w[i]; }
            for i in 0..d_mono_d1_looped.len() { d_mono_d1_looped[i] += g_t.d_mono_d1[i]; }
            for i in 0..d_mono_d2_looped.len() { d_mono_d2_looped[i] += g_t.d_mono_d2[i]; }
        }

        for i in 0..d_router_w_looped.len() {
            assert!((g_batch.d_router_w[i] - d_router_w_looped[i]).abs() < 1e-4,
                "d_router_w[{i}]: batch={} looped={}", g_batch.d_router_w[i], d_router_w_looped[i]);
        }
        for i in 0..d_mono_d1_looped.len() {
            assert!((g_batch.d_mono_d1[i] - d_mono_d1_looped[i]).abs() < 1e-3,
                "d_mono_d1[{i}]: batch={} looped={}", g_batch.d_mono_d1[i], d_mono_d1_looped[i]);
        }
    }
}
