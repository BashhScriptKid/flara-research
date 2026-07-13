//! One transformer block: pre-norm attention + pre-norm block-routed FFN, with a
//! gradient-stopped early-exit probe on the attention input.
//!
//! Dataflow (pre-norm, depth-scaled residuals):
//! ```text
//!   h ─RMSNorm─► Q/K/V proj ─RoPE(q,k)─► attn(Full|Sliding) ─O proj─► +scale·· ─┐
//!     └─ ExitProbe(normed h)  [gradient-stopped]                                 │
//!   h' ─RMSNorm─► FFN(select → prefetch → compute) ─► +scale··  ───────────────► out
//! ```
//! Attention is sequence-level; the FFN and probe run per token. The FFN uses the
//! `select → prefetch_coeffs → compute` seam so coefficient fetch overlaps
//! routing. RoPE is applied here (the kernels see already-rotated Q/K). The shared
//! dictionary `G` is owned by the model and threaded in.

use crate::kernels::attn_flash::FlashAttention;
use crate::kernels::attn_swa::SlidingWindowAttention;
use crate::kernels::attn_tma::TmaAttention;
use crate::kernels::ffn::{Ffn, FfnForward, FfnOptState};
use crate::kernels::norm;
use crate::kernels::optimizer::{AdaFactor, AdaFactorState};
use crate::kernels::probe::ExitProbe;
use crate::kernels::rope::Rope;
use crate::model::attn_proj::AttnProj;
use crate::model::config::{AttnKind, ModelConfig};
use rustfft::num_complex::Complex32;

enum AttnRunner {
    Full(FlashAttention),
    Sliding(SlidingWindowAttention),
    /// Forward-only profiling variant: TauMonarchAttention in place of Full
    /// FlashAttention. Selected when the `TMA_ATTN=1` env var is set at
    /// layer construction time (see `TransformerLayer::new`). No backward
    /// pass exists for this variant -- `backward` panics if called.
    Tma(TmaAttention),
}

impl AttnRunner {
    fn forward(&self, q: &[f32], k: &[f32], v: &[f32], out: &mut [f32]) -> Vec<f32> {
        match self {
            AttnRunner::Full(a) => a.forward(q, k, v, out),
            AttnRunner::Sliding(a) => a.forward(q, k, v, out),
            AttnRunner::Tma(a) => a.forward(q, k, v, out),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn backward(
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
        match self {
            AttnRunner::Full(a) => a.backward(q, k, v, out, lse, d_out, dq, dk, dv),
            AttnRunner::Sliding(a) => a.backward(q, k, v, out, lse, d_out, dq, dk, dv),
            AttnRunner::Tma(_) => panic!(
                "AttnRunner::Tma has no backward pass -- TMA_ATTN=1 is forward-only profiling, not for training"
            ),
        }
    }
}

/// Cached forward intermediates the backward pass will consume.
pub struct LayerForward {
    /// Block output, length `T·H`.
    pub out: Vec<f32>,
    /// Attention-norm output (also the probe input), `T·H`.
    pub normed: Vec<f32>,
    /// Attention-norm reciprocal-RMS per token, length `T`.
    pub rinv: Vec<f32>,
    /// Post-RoPE projected Q/K/V (what attention saw).
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// Attention output before the O projection, `T·q_dim`.
    pub attn_out: Vec<f32>,
    /// Per-row log-sum-exp from attention.
    pub lse: Vec<f32>,
    /// Residual stream after the attention sub-block, `T·H`.
    pub h_mid: Vec<f32>,
    /// FFN-norm output, `T·H`.
    pub normed2: Vec<f32>,
    /// FFN-norm reciprocal-RMS per token, `T`.
    pub rinv2: Vec<f32>,
    /// Per-token FFN forward caches.
    pub ffn_fwds: Vec<FfnForward>,
    /// Per-token early-exit probability, length `T`.
    pub probe_p: Vec<f32>,
}

/// Gradients for one layer. `d_hidden` flows to the layer below; the rest are
/// parameter grads. `d_dict` is the shared-`G` contribution summed across all
/// four projections and the FFN (empty arms — the dense projection path — add
/// nothing). Probe grads are populated only when an upstream probe gradient is
/// supplied; the probe is gradient-stopped, so it never touches `d_hidden`.
pub struct LayerGrads {
    pub d_hidden: Vec<f32>,
    pub d_dict: Vec<Complex32>,
    pub d_attn_norm_gain: Vec<f32>,
    pub d_ffn_norm_gain: Vec<f32>,
    pub d_wq: Vec<f32>,
    pub d_wk: Vec<f32>,
    pub d_wv: Vec<f32>,
    pub d_wo: Vec<f32>,
    pub d_up_coeffs: Vec<f32>,
    pub d_gate_coeffs: Vec<f32>,
    pub d_down_coeffs: Vec<f32>,
    pub d_router_w: Vec<f32>,
    pub d_probe_w: Vec<f32>,
    pub d_probe_bias: f32,
}

impl LayerGrads {
    /// Accumulate another micro-batch's *parameter* gradients into `self`.
    /// `d_hidden` (a per-token activation gradient whose length tracks the seq
    /// curriculum) and the per-layer `d_dict` (subsumed by the model-level sum)
    /// are deliberately not accumulated.
    pub fn add(&mut self, other: &LayerGrads) {
        add_f(&mut self.d_attn_norm_gain, &other.d_attn_norm_gain);
        add_f(&mut self.d_ffn_norm_gain, &other.d_ffn_norm_gain);
        add_f(&mut self.d_wq, &other.d_wq);
        add_f(&mut self.d_wk, &other.d_wk);
        add_f(&mut self.d_wv, &other.d_wv);
        add_f(&mut self.d_wo, &other.d_wo);
        add_f(&mut self.d_up_coeffs, &other.d_up_coeffs);
        add_f(&mut self.d_gate_coeffs, &other.d_gate_coeffs);
        add_f(&mut self.d_down_coeffs, &other.d_down_coeffs);
        add_f(&mut self.d_router_w, &other.d_router_w);
        add_f(&mut self.d_probe_w, &other.d_probe_w);
        self.d_probe_bias += other.d_probe_bias;
    }

    /// Scale every accumulated parameter gradient in place (e.g. by `1/n` to take
    /// the mean over `n` accumulated micro-batches).
    pub fn scale(&mut self, f: f32) {
        for v in [
            &mut self.d_attn_norm_gain,
            &mut self.d_ffn_norm_gain,
            &mut self.d_wq,
            &mut self.d_wk,
            &mut self.d_wv,
            &mut self.d_wo,
            &mut self.d_up_coeffs,
            &mut self.d_gate_coeffs,
            &mut self.d_down_coeffs,
            &mut self.d_router_w,
            &mut self.d_probe_w,
        ] {
            for x in v.iter_mut() {
                *x *= f;
            }
        }
        self.d_probe_bias *= f;
    }
}

/// Optimizer state for one transformer layer — one AdaFactor state per parameter
/// tensor, mirroring [`LayerGrads`]. The FFN's states are bundled in [`FfnOptState`].
#[derive(serde::Serialize, serde::Deserialize)]
pub struct LayerOptState {
    pub attn_norm: AdaFactorState,
    pub ffn_norm: AdaFactorState,
    pub wq: AdaFactorState,
    pub wk: AdaFactorState,
    pub wv: AdaFactorState,
    pub wo: AdaFactorState,
    pub ffn: FfnOptState,
    pub probe_w: AdaFactorState,
    pub probe_bias: AdaFactorState,
}

/// Accumulate `src` into `acc`, seeding `acc` on first use. Lets us sum
/// per-token parameter grads without pre-computing every sub-kernel's length.
fn add_f(acc: &mut Vec<f32>, src: &[f32]) {
    if acc.is_empty() {
        acc.extend_from_slice(src);
    } else {
        for (a, b) in acc.iter_mut().zip(src) {
            *a += *b;
        }
    }
}

fn add_c(acc: &mut Vec<Complex32>, src: &[Complex32]) {
    if acc.is_empty() {
        acc.extend_from_slice(src);
    } else {
        for (a, b) in acc.iter_mut().zip(src) {
            *a += *b;
        }
    }
}

/// A single transformer layer. Owns its projections, FFN, norms and probe; the
/// dictionary `G` is shared and supplied to [`forward`](TransformerLayer::forward).
pub struct TransformerLayer {
    cfg: ModelConfig,
    kind: AttnKind,
    attn_norm_gain: Vec<f32>,
    ffn_norm_gain: Vec<f32>,
    wq: AttnProj,
    wk: AttnProj,
    wv: AttnProj,
    wo: AttnProj,
    ffn: Ffn,
    probe: ExitProbe,
    attn: AttnRunner,
}

/// Serializable snapshot of one layer's learned parameters. FFT plans, attention
/// runners, and config are *not* stored — they are rebuilt from [`ModelConfig`].
#[derive(serde::Serialize, serde::Deserialize)]
pub struct LayerCheckpoint {
    pub attn_norm_gain: Vec<f32>,
    pub ffn_norm_gain: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub up_coeffs: Vec<f32>,
    pub gate_coeffs: Vec<f32>,
    pub down_coeffs: Vec<f32>,
    pub router_w: Vec<f32>,
    pub probe_w: Vec<f32>,
    pub probe_bias: f32,
}

impl TransformerLayer {
    /// Capture this layer's learned parameters into a serializable checkpoint.
    pub fn to_checkpoint(&self) -> LayerCheckpoint {
        LayerCheckpoint {
            attn_norm_gain: self.attn_norm_gain.clone(),
            ffn_norm_gain: self.ffn_norm_gain.clone(),
            wq: self.wq.params().to_vec(),
            wk: self.wk.params().to_vec(),
            wv: self.wv.params().to_vec(),
            wo: self.wo.params().to_vec(),
            up_coeffs: self.ffn.up_coeffs.clone(),
            gate_coeffs: self.ffn.gate_coeffs.clone(),
            down_coeffs: self.ffn.down_coeffs.clone(),
            router_w: self.ffn.router_w.clone(),
            probe_w: self.probe.w.clone(),
            probe_bias: self.probe.bias,
        }
    }

    /// Restore this layer's learned parameters from a checkpoint, in place.
    pub fn load_checkpoint(&mut self, c: &LayerCheckpoint) {
        self.attn_norm_gain.copy_from_slice(&c.attn_norm_gain);
        self.ffn_norm_gain.copy_from_slice(&c.ffn_norm_gain);
        self.wq.set_params(&c.wq);
        self.wk.set_params(&c.wk);
        self.wv.set_params(&c.wv);
        self.wo.set_params(&c.wo);
        self.ffn.up_coeffs.copy_from_slice(&c.up_coeffs);
        self.ffn.gate_coeffs.copy_from_slice(&c.gate_coeffs);
        self.ffn.down_coeffs.copy_from_slice(&c.down_coeffs);
        self.ffn.router_w.copy_from_slice(&c.router_w);
        self.probe.w.copy_from_slice(&c.probe_w);
        self.probe.bias = c.probe_bias;
    }
}

impl TransformerLayer {
    pub fn new(cfg: &ModelConfig, layer_idx: usize, seed: u64) -> Self {
        let (h, b, k) = (cfg.hidden, cfg.block, cfg.dict_k);
        let (qd, kvd) = (cfg.q_dim(), cfg.kv_dim());
        let kind = cfg.attn_kind(layer_idx);
        let tma_profiling = std::env::var("TMA_ATTN").as_deref() == Ok("1");
        let attn = match kind {
            AttnKind::Full if tma_profiling => AttnRunner::Tma(TmaAttention::new(
                cfg.n_q_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
            )),
            AttnKind::Full => AttnRunner::Full(FlashAttention::new(
                cfg.n_q_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
                cfg.kv_block,
            )),
            AttnKind::Sliding => AttnRunner::Sliding(SlidingWindowAttention::new(
                cfg.n_q_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
                cfg.window,
                cfg.kv_block,
            )),
        };
        let wq_p = AttnProj::new(qd, h, b, k, seed ^ 0x11);
        let wk_p = AttnProj::new(kvd, h, b, k, seed ^ 0x12);
        let wv_p = AttnProj::new(kvd, h, b, k, seed ^ 0x13);
        let wo_p = AttnProj::new(h, qd, b, k, seed ^ 0x14);
        let ffn_o = Ffn::new(cfg.ffn_config(), seed ^ 0x15);
        Self {
            cfg: cfg.clone(),
            kind,
            attn_norm_gain: vec![1.0; h],
            ffn_norm_gain: vec![1.0; h],
            wq: wq_p,
            wk: wk_p,
            wv: wv_p,
            wo: wo_p,
            ffn: ffn_o,
            probe: ExitProbe::new(h),
            attn,
        }
    }

    #[inline]
    pub fn kind(&self) -> AttnKind {
        self.kind
    }

    /// Forward over a sequence of `t_len` tokens. `hidden` is `[T, H]` row-major;
    /// `dict` is the shared `G`; `rope` is the shared rotary table.
    pub fn forward(
        &self,
        dict: &[Complex32],
        rope: &Rope,
        hidden: &[f32],
        t_len: usize,
    ) -> LayerForward {
        let cfg = &self.cfg;
        let (h, qd, kvd, hd) = (cfg.hidden, cfg.q_dim(), cfg.kv_dim(), cfg.head_dim);
        let scale = cfg.residual_scale();
        debug_assert_eq!(hidden.len(), t_len * h);

        // --- attention sub-block ---
        let mut normed = vec![0.0f32; t_len * h];
        let mut rinv = vec![0.0f32; t_len];
        let mut q = vec![0.0f32; t_len * qd];
        let mut k = vec![0.0f32; t_len * kvd];
        let mut v = vec![0.0f32; t_len * kvd];
        let mut probe_p = vec![0.0f32; t_len];

        for ti in 0..t_len {
            let hin = &hidden[ti * h..(ti + 1) * h];
            let nrm = &mut normed[ti * h..(ti + 1) * h];
            rinv[ti] = norm::forward(hin, &self.attn_norm_gain, cfg.norm_eps, nrm);
            probe_p[ti] = self.probe.forward(nrm);
        }

        self.wq.forward_batch(dict, &normed, &mut q, t_len);
        self.wk.forward_batch(dict, &normed, &mut k, t_len);
        self.wv.forward_batch(dict, &normed, &mut v, t_len);

        for ti in 0..t_len {
            for head in 0..cfg.n_q_heads {
                rope.apply(&mut q[ti * qd + head * hd..ti * qd + (head + 1) * hd], ti);
            }
            for head in 0..cfg.n_kv_heads {
                rope.apply(&mut k[ti * kvd + head * hd..ti * kvd + (head + 1) * hd], ti);
            }
        }

        let mut attn_out = vec![0.0f32; t_len * qd];
        let lse = self.attn.forward(&q, &k, &v, &mut attn_out);

        let mut h_mid = hidden.to_vec();
        let mut o_proj = vec![0.0f32; t_len * qd];
        self.wo.forward_batch(dict, &attn_out, &mut o_proj, t_len);
        for ti in 0..t_len {
            let oi = &o_proj[ti * qd..(ti + 1) * qd];
            let dst = &mut h_mid[ti * h..(ti + 1) * h];
            for j in 0..h {
                dst[j] += scale * oi[j];
            }
        }

        // --- FFN sub-block ---
        let mut normed2 = vec![0.0f32; t_len * h];
        let mut rinv2 = vec![0.0f32; t_len];
        let mut out = h_mid.clone();
        let mut ffn_fwds = Vec::with_capacity(t_len);
        for ti in 0..t_len {
            let hin = &h_mid[ti * h..(ti + 1) * h];
            let nrm = &mut normed2[ti * h..(ti + 1) * h];
            rinv2[ti] = norm::forward(hin, &self.ffn_norm_gain, cfg.norm_eps, nrm);
            let sel = self.ffn.select(nrm);
            self.ffn.prefetch_coeffs(&sel);
            let fwd = self.ffn.compute(dict, nrm, sel);
            let dst = &mut out[ti * h..(ti + 1) * h];
            for j in 0..h {
                dst[j] += scale * fwd.out[j];
            }
            ffn_fwds.push(fwd);
        }

        LayerForward {
            out,
            normed,
            rinv,
            q,
            k,
            v,
            attn_out,
            lse,
            h_mid,
            normed2,
            rinv2,
            ffn_fwds,
            probe_p,
        }
    }

    /// Reverse of [`forward`](TransformerLayer::forward). `d_out` is the gradient
    /// w.r.t. the layer output (`T·H`). `d_probe_p`, if given, is the upstream
    /// gradient of the early-exit loss w.r.t. each token's probe probability; it
    /// produces probe param grads only — the probe is gradient-stopped, so it does
    /// not contribute to `d_hidden`. Returns the full gradient bundle.
    pub fn backward(
        &self,
        dict: &[Complex32],
        rope: &Rope,
        hidden: &[f32],
        fwd: &LayerForward,
        d_out: &[f32],
        d_probe_p: Option<&[f32]>,
        t_len: usize,
    ) -> LayerGrads {
        let cfg = &self.cfg;
        let (h, qd, kvd, hd) = (cfg.hidden, cfg.q_dim(), cfg.kv_dim(), cfg.head_dim);
        let scale = cfg.residual_scale();
        debug_assert_eq!(d_out.len(), t_len * h);

        let mut g = LayerGrads {
            d_hidden: vec![0.0; t_len * h],
            d_dict: Vec::new(),
            d_attn_norm_gain: Vec::new(),
            d_ffn_norm_gain: Vec::new(),
            d_wq: Vec::new(),
            d_wk: Vec::new(),
            d_wv: Vec::new(),
            d_wo: Vec::new(),
            d_up_coeffs: Vec::new(),
            d_gate_coeffs: Vec::new(),
            d_down_coeffs: Vec::new(),
            d_router_w: Vec::new(),
            d_probe_w: Vec::new(),
            d_probe_bias: 0.0,
        };

        // ---- FFN sub-block (last in forward ⇒ first in backward) ----
        // out = h_mid + scale·ffn(normed2(h_mid)).
        let mut d_h_mid = vec![0.0f32; t_len * h];
        for ti in 0..t_len {
            let mut d_ffn_out = vec![0.0f32; h];
            for j in 0..h {
                d_h_mid[ti * h + j] = d_out[ti * h + j]; // identity residual
                d_ffn_out[j] = scale * d_out[ti * h + j];
            }
            let fg = self.ffn.backward(
                dict,
                &fwd.normed2[ti * h..(ti + 1) * h],
                &fwd.ffn_fwds[ti],
                &d_ffn_out,
            );
            add_f(&mut g.d_up_coeffs, &fg.d_up_coeffs);
            add_f(&mut g.d_gate_coeffs, &fg.d_gate_coeffs);
            add_f(&mut g.d_down_coeffs, &fg.d_down_coeffs);
            add_f(&mut g.d_router_w, &fg.d_router_w);
            add_c(&mut g.d_dict, &fg.d_dict);

            // FFN-norm backward folds the FFN-input grad back into the residual.
            let mut dx = vec![0.0f32; h];
            let mut dg = vec![0.0f32; h];
            norm::backward(
                &fwd.h_mid[ti * h..(ti + 1) * h],
                &self.ffn_norm_gain,
                &fg.d_h,
                fwd.rinv2[ti],
                &mut dx,
                &mut dg,
            );
            for j in 0..h {
                d_h_mid[ti * h + j] += dx[j];
            }
            add_f(&mut g.d_ffn_norm_gain, &dg);
        }

        // ---- attention sub-block ----
        // O projection: h_mid = hidden + scale·O(attn_out).
        let mut d_attn_out = vec![0.0f32; t_len * qd];
        for ti in 0..t_len {
            let mut d_oi = vec![0.0f32; h];
            for j in 0..h {
                d_oi[j] = scale * d_h_mid[ti * h + j];
            }
            let pg = self.wo.backward(dict, &fwd.attn_out[ti * qd..(ti + 1) * qd], &d_oi);
            add_f(&mut g.d_wo, &pg.d_param);
            add_c(&mut g.d_dict, &pg.d_dict);
            d_attn_out[ti * qd..(ti + 1) * qd].copy_from_slice(&pg.d_x);
        }
        // Identity residual of the attention sub-block.
        for i in 0..t_len * h {
            g.d_hidden[i] += d_h_mid[i];
        }

        // Attention backward (sequence-level), then undo RoPE on q/k grads.
        let mut dq = vec![0.0f32; t_len * qd];
        let mut dk = vec![0.0f32; t_len * kvd];
        let mut dv = vec![0.0f32; t_len * kvd];
        self.attn.backward(
            &fwd.q, &fwd.k, &fwd.v, &fwd.attn_out, &fwd.lse, &d_attn_out, &mut dq, &mut dk, &mut dv,
        );
        for ti in 0..t_len {
            for head in 0..cfg.n_q_heads {
                rope.apply_backward(&mut dq[ti * qd + head * hd..ti * qd + (head + 1) * hd], ti);
            }
            for head in 0..cfg.n_kv_heads {
                rope.apply_backward(&mut dk[ti * kvd + head * hd..ti * kvd + (head + 1) * hd], ti);
            }
        }

        // Q/K/V projections → d_normed, then attention-norm backward → d_hidden.
        for ti in 0..t_len {
            let nrm = &fwd.normed[ti * h..(ti + 1) * h];
            let gq = self.wq.backward(dict, nrm, &dq[ti * qd..(ti + 1) * qd]);
            let gk = self.wk.backward(dict, nrm, &dk[ti * kvd..(ti + 1) * kvd]);
            let gv = self.wv.backward(dict, nrm, &dv[ti * kvd..(ti + 1) * kvd]);
            add_f(&mut g.d_wq, &gq.d_param);
            add_c(&mut g.d_dict, &gq.d_dict);
            add_f(&mut g.d_wk, &gk.d_param);
            add_c(&mut g.d_dict, &gk.d_dict);
            add_f(&mut g.d_wv, &gv.d_param);
            add_c(&mut g.d_dict, &gv.d_dict);

            let mut d_normed = vec![0.0f32; h];
            for j in 0..h {
                d_normed[j] = gq.d_x[j] + gk.d_x[j] + gv.d_x[j];
            }
            let mut dx = vec![0.0f32; h];
            let mut dg = vec![0.0f32; h];
            norm::backward(
                &hidden[ti * h..(ti + 1) * h],
                &self.attn_norm_gain,
                &d_normed,
                fwd.rinv[ti],
                &mut dx,
                &mut dg,
            );
            for j in 0..h {
                g.d_hidden[ti * h + j] += dx[j];
            }
            add_f(&mut g.d_attn_norm_gain, &dg);
        }

        // Probe head: gradient-stopped, so params only — no path into d_hidden.
        if let Some(dp) = d_probe_p {
            let mut dw = vec![0.0f32; h];
            let mut db = 0.0f32;
            for ti in 0..t_len {
                let pg = self.probe.backward(&fwd.normed[ti * h..(ti + 1) * h], fwd.probe_p[ti], dp[ti]);
                for j in 0..h {
                    dw[j] += pg.d_w[j];
                }
                db += pg.d_bias;
            }
            g.d_probe_w = dw;
            g.d_probe_bias = db;
        }

        g
    }

    /// Allocate this layer's optimizer state.
    pub fn init_opt(&self) -> LayerOptState {
        let h = self.cfg.hidden;
        LayerOptState {
            attn_norm: AdaFactorState::vector(h, false),
            ffn_norm: AdaFactorState::vector(h, false),
            wq: self.wq.init_opt(),
            wk: self.wk.init_opt(),
            wv: self.wv.init_opt(),
            wo: self.wo.init_opt(),
            ffn: self.ffn.init_opt(),
            probe_w: AdaFactorState::vector(h, false),
            probe_bias: AdaFactorState::vector(1, false),
        }
    }

    /// Apply one AdaFactor step to every parameter in this layer. Probe params are
    /// stepped only when probe gradients are present (the early-exit head is
    /// trained by a separate, annealed loss wired in the training loop).
    pub fn apply_grad(&mut self, g: &LayerGrads, st: &mut LayerOptState, af: &AdaFactor, lr: f32) {
        af.step(&mut self.attn_norm_gain, &g.d_attn_norm_gain, &mut st.attn_norm, lr);
        af.step(&mut self.ffn_norm_gain, &g.d_ffn_norm_gain, &mut st.ffn_norm, lr);
        self.wq.apply_grad(&g.d_wq, &mut st.wq, af, lr);
        self.wk.apply_grad(&g.d_wk, &mut st.wk, af, lr);
        self.wv.apply_grad(&g.d_wv, &mut st.wv, af, lr);
        self.wo.apply_grad(&g.d_wo, &mut st.wo, af, lr);
        self.ffn.apply_grad(
            &g.d_up_coeffs, &g.d_gate_coeffs, &g.d_down_coeffs, &g.d_router_w, &mut st.ffn, af, lr,
        );
        if !g.d_probe_w.is_empty() {
            af.step(&mut self.probe.w, &g.d_probe_w, &mut st.probe_w, lr);
            af.step(
                core::slice::from_mut(&mut self.probe.bias),
                core::slice::from_ref(&g.d_probe_bias),
                &mut st.probe_bias,
                lr,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::fft::init_dict_random;

    /// Small but spec-valid config for fast tests.
    fn tiny_cfg() -> ModelConfig {
        let mut c = ModelConfig::default();
        c.n_layers = 2;
        c.hidden = 8;
        c.n_q_heads = 2;
        c.n_kv_heads = 1;
        c.head_dim = 4;
        c.ffn_dim = 12;
        c.block = 4;
        c.n_active = 2;
        c.dict_k = 6;
        c.kv_block = 2;
        c.window = 3;
        c.full_attn_layers = 1;
        c.vocab = 32;
        c.max_seq = 16;
        c.validate();
        c
    }

    fn tiny_dict(c: &ModelConfig) -> Vec<Complex32> {
        init_dict_random(c.dict_k, c.block, 0x6, 0.6)
    }

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
    }

    #[test]
    fn forward_shapes_and_probe_range() {
        let c = tiny_cfg();
        let dict = tiny_dict(&c);
        let rope = Rope::new(c.head_dim, c.max_seq, c.rope_base);
        let mut layer = TransformerLayer::new(&c, 0, 0xAB);
        let t = 5;
        let mut rng = Lcg(0x1234);
        let hidden: Vec<f32> = (0..t * c.hidden).map(|_| rng.f()).collect();

        let fwd = layer.forward(&dict, &rope, &hidden, t);
        assert_eq!(fwd.out.len(), t * c.hidden);
        assert_eq!(fwd.probe_p.len(), t);
        assert_eq!(fwd.ffn_fwds.len(), t);
        for &p in &fwd.probe_p {
            assert!(p > 0.0 && p < 1.0, "probe prob out of range: {p}");
        }
    }

    #[test]
    fn layer_attention_is_causal() {
        // Perturbing the last token's input must not change token 0's output:
        // attention is causal and the FFN/probe are per-token.
        let c = tiny_cfg();
        let dict = tiny_dict(&c);
        let rope = Rope::new(c.head_dim, c.max_seq, c.rope_base);
        let mut layer = TransformerLayer::new(&c, 0, 0xCD); // layer 0 = Full attention
        let t = 4;
        let mut rng = Lcg(0x77);
        let mut hidden: Vec<f32> = (0..t * c.hidden).map(|_| rng.f()).collect();

        let base = layer.forward(&dict, &rope, &hidden, t).out;
        for j in (t - 1) * c.hidden..t * c.hidden {
            hidden[j] += 1.5;
        }
        let perturbed = layer.forward(&dict, &rope, &hidden, t).out;
        for j in 0..c.hidden {
            assert!((base[j] - perturbed[j]).abs() < 1e-7, "future token leaked into token 0");
        }
    }

    #[test]
    fn sliding_layer_runs_and_is_causal() {
        let c = tiny_cfg();
        let dict = tiny_dict(&c);
        let rope = Rope::new(c.head_dim, c.max_seq, c.rope_base);
        let mut layer = TransformerLayer::new(&c, 1, 0xEF); // layer 1 = Sliding
        assert_eq!(layer.kind(), AttnKind::Sliding);
        let t = 5;
        let mut rng = Lcg(0x99);
        let mut hidden: Vec<f32> = (0..t * c.hidden).map(|_| rng.f()).collect();
        let base = layer.forward(&dict, &rope, &hidden, t).out;
        for j in (t - 1) * c.hidden..t * c.hidden {
            hidden[j] += 2.0;
        }
        let perturbed = layer.forward(&dict, &rope, &hidden, t).out;
        for j in 0..c.hidden {
            assert!((base[j] - perturbed[j]).abs() < 1e-7, "future leaked (sliding)");
        }
    }

    /// End-to-end composition check: finite-diff `d_hidden` exercises the whole
    /// reverse chain (FFN → norm → O → attention → RoPE → QKV → norm). The
    /// per-kernel param grads are already gradchecked individually; this proves
    /// the layer wires their `d_x` outputs together correctly.
    #[test]
    fn layer_backward_d_hidden_gradchecks() {
        let c = tiny_cfg();
        let dict = tiny_dict(&c);
        let rope = Rope::new(c.head_dim, c.max_seq, c.rope_base);
        let mut layer = TransformerLayer::new(&c, 0, 0x2024); // Full attention
        let t = 4;
        let mut rng = Lcg(0x5151);
        let hidden: Vec<f32> = (0..t * c.hidden).map(|_| rng.f()).collect();
        let r: Vec<f32> = (0..t * c.hidden).map(|_| rng.f()).collect(); // loss=Σ out·r ⇒ d_out=r

        let base = layer.forward(&dict, &rope, &hidden, t);
        let grads = layer.backward(&dict, &rope, &hidden, &base, &r, None, t);
        let base_sel: Vec<Vec<usize>> = base.ffn_fwds.iter().map(|f| f.selected.clone()).collect();

        let loss = |fwd: &LayerForward| -> f32 { fwd.out.iter().zip(&r).map(|(o, rr)| o * rr).sum() };
        let sel_stable =
            |fwd: &LayerForward| fwd.ffn_fwds.iter().zip(&base_sel).all(|(f, s)| f.selected == *s);

        const H: f32 = 1e-3;
        let mut checked = 0;
        for i in 0..t * c.hidden {
            let mut hp = hidden.clone();
            hp[i] += H;
            let fp = layer.forward(&dict, &rope, &hp, t);
            hp[i] -= 2.0 * H;
            let fm = layer.forward(&dict, &rope, &hp, t);
            // Skip coords where the FFN top-k routing flips (non-smooth kink).
            if !sel_stable(&fp) || !sel_stable(&fm) {
                continue;
            }
            let fd = (loss(&fp) - loss(&fm)) / (2.0 * H);
            let an = grads.d_hidden[i];
            assert!((fd - an).abs() < 1e-2 + 5e-2 * an.abs(), "d_hidden[{i}] fd {fd} an {an}");
            checked += 1;
        }
        assert!(checked >= t * c.hidden / 2, "too many coords skipped: {checked}");
    }

    /// The probe is gradient-stopped: supplying an upstream probe gradient must
    /// populate the probe param grads but leave the backbone `d_hidden` untouched.
    #[test]
    fn probe_is_gradient_stopped() {
        let c = tiny_cfg();
        let dict = tiny_dict(&c);
        let rope = Rope::new(c.head_dim, c.max_seq, c.rope_base);
        let mut layer = TransformerLayer::new(&c, 0, 0x7);
        let t = 3;
        let mut rng = Lcg(0xC0DE);
        let hidden: Vec<f32> = (0..t * c.hidden).map(|_| rng.f()).collect();
        let d_out = vec![0.1f32; t * c.hidden];
        let fwd = layer.forward(&dict, &rope, &hidden, t);

        let g_none = layer.backward(&dict, &rope, &hidden, &fwd, &d_out, None, t);
        let dp = vec![0.5f32; t];
        let g_probe = layer.backward(&dict, &rope, &hidden, &fwd, &d_out, Some(&dp), t);

        for j in 0..t * c.hidden {
            assert!((g_none.d_hidden[j] - g_probe.d_hidden[j]).abs() < 1e-12, "probe leaked into backbone");
        }
        assert!(g_none.d_probe_w.is_empty());
        assert_eq!(g_probe.d_probe_w.len(), c.hidden);
        assert!(
            g_probe.d_probe_w.iter().any(|&x| x.abs() > 0.0) || g_probe.d_probe_bias.abs() > 0.0,
            "probe grads should be nonzero"
        );
    }
}
