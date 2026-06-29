//! The full model: tied embedding → transformer stack → final norm → tied LM head.
//!
//! The model owns the three things shared across every layer — the dictionary `G`
//! (all compressed matmuls decode against it), the rotary table `Rope`, and the
//! token embedding `E` — plus the layer stack and the final-norm gain. The
//! embedding is the one tensor that is *not* circular-basis compressed (a plain
//! `[vocab, hidden]` table) and it is **tied**: the LM head reuses `E` as its
//! output projection.
//!
//! ```text
//!   ids ─lookup E─► x ─[layer × N]─► x ─RMSNorm─► h ─h·Eᵀ─► logits[T, vocab]
//! ```
//!
//! The forward stores every layer's [`LayerForward`] activation cache so the (next
//! increment) backward can run without recomputation. Activation checkpointing —
//! trading that storage for recompute — is a deferred memory optimization.

use crate::kernels::fft::{init_coeffs_random, init_dict_random};
use crate::kernels::norm;
use crate::kernels::rope::Rope;
use crate::model::config::ModelConfig;
use crate::kernels::optimizer::{AdaFactor, AdaFactorState};
use crate::model::layer::{LayerCheckpoint, LayerForward, LayerGrads, LayerOptState, TransformerLayer};
use rustfft::num_complex::Complex32;

/// Cached forward state for the whole model, consumed by the backward pass.
pub struct ModelForward {
    /// Next-token logits, `[T, vocab]` row-major.
    pub logits: Vec<f32>,
    /// The input token ids (needed to scatter embedding gradients).
    pub token_ids: Vec<usize>,
    /// Per-layer activation caches, `n_layers` of them, in forward order.
    pub layer_fwds: Vec<LayerForward>,
    /// Output of the last layer / input to the final norm, `[T, H]`.
    pub final_x: Vec<f32>,
    /// Final-norm output (the LM-head input), `[T, H]`.
    pub normed_final: Vec<f32>,
    /// Final-norm reciprocal-RMS per token, length `T`.
    pub rinv_final: Vec<f32>,
}

/// Per-parameter gradients for the whole model. `d_embed` and `d_dict` are the
/// tied/shared tensors (the embedding-cum-LM-head and the dictionary); `layers`
/// holds the per-layer grads in forward order. Each layer's own `d_dict` is
/// subsumed by the model-level `d_dict` sum and should be ignored by the optimizer.
pub struct ModelGrads {
    pub d_embed: Vec<f32>,
    pub d_dict: Vec<Complex32>,
    pub d_final_norm_gain: Vec<f32>,
    pub layers: Vec<LayerGrads>,
}

impl ModelGrads {
    /// Accumulate another micro-batch's gradients into `self`. Model-level tensors
    /// (embed, dict, final norm) share fixed parameter shapes across micro-batches,
    /// so they sum elementwise; per-layer accumulation defers to [`LayerGrads::add`].
    pub fn add(&mut self, other: &ModelGrads) {
        for (a, b) in self.d_embed.iter_mut().zip(&other.d_embed) {
            *a += *b;
        }
        for (a, b) in self.d_dict.iter_mut().zip(&other.d_dict) {
            *a += *b;
        }
        for (a, b) in self.d_final_norm_gain.iter_mut().zip(&other.d_final_norm_gain) {
            *a += *b;
        }
        for (la, lb) in self.layers.iter_mut().zip(&other.layers) {
            la.add(lb);
        }
    }

    /// Scale every parameter gradient in place (e.g. by `1/n` to mean over the
    /// `n` micro-batches accumulated toward the effective batch).
    pub fn scale(&mut self, f: f32) {
        for x in self.d_embed.iter_mut() {
            *x *= f;
        }
        for x in self.d_dict.iter_mut() {
            *x = Complex32::new(x.re * f, x.im * f);
        }
        for x in self.d_final_norm_gain.iter_mut() {
            *x *= f;
        }
        for l in self.layers.iter_mut() {
            l.scale(f);
        }
    }
}

/// Optimizer state for the whole model, mirroring [`ModelGrads`]. `embed` factors
/// as `[vocab, hidden]`; `dict` is a full second moment over the `2·K·b` real
/// components of the complex dictionary; per-layer state lives in `layers`.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ModelOptState {
    pub embed: AdaFactorState,
    pub dict: AdaFactorState,
    pub final_norm: AdaFactorState,
    pub layers: Vec<LayerOptState>,
}

/// Serializable snapshot of a model's learned parameters. FFT plans, RoPE tables,
/// and attention runners are *not* stored — they are rebuilt from `cfg` on load.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Checkpoint {
    pub cfg: ModelConfig,
    /// The complex dictionary `G`, `K×b` complex (= `2·K·b` reals for the optimizer).
    pub dict: Vec<Complex32>,
    pub embed: Vec<f32>,
    pub final_norm_gain: Vec<f32>,
    pub layers: Vec<LayerCheckpoint>,
}

impl Model {
    /// Capture all learned parameters into a serializable checkpoint.
    pub fn to_checkpoint(&self) -> Checkpoint {
        Checkpoint {
            cfg: self.cfg.clone(),
            dict: self.dict.clone(),
            embed: self.embed.clone(),
            final_norm_gain: self.final_norm_gain.clone(),
            layers: self.layers.iter().map(|l| l.to_checkpoint()).collect(),
        }
    }

    /// Rebuild a model from a checkpoint: construct the skeleton (and its FFT plans)
    /// from `cfg`, then overwrite every learned tensor.
    pub fn from_checkpoint(c: &Checkpoint) -> Model {
        let mut m = Model::new(c.cfg.clone(), 0);
        m.dict = c.dict.clone();
        m.embed = c.embed.clone();
        m.final_norm_gain = c.final_norm_gain.clone();
        for (layer, lc) in m.layers.iter_mut().zip(&c.layers) {
            layer.load_checkpoint(lc);
        }
        m
    }
}

/// Mean token-level cross-entropy and its gradient w.r.t. the logits.
///
/// `logits` is `[T, vocab]`; `targets[ti]` is the gold next-token id at position
/// `ti`. Returns `(mean_nll, d_logits)` where `d_logits = (softmax − onehot)/T`,
/// already averaged so it composes directly with [`Model::backward`].
pub fn cross_entropy(logits: &[f32], vocab: usize, targets: &[usize]) -> (f32, Vec<f32>) {
    let t = targets.len();
    debug_assert_eq!(logits.len(), t * vocab);
    let mut d = vec![0.0f32; logits.len()];
    let mut loss = 0.0f32;
    let inv_t = 1.0 / t as f32;
    for ti in 0..t {
        let row = &logits[ti * vocab..(ti + 1) * vocab];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|&x| (x - m).exp()).sum();
        let lse = m + sum.ln();
        let tgt = targets[ti];
        debug_assert!(tgt < vocab, "target {tgt} out of vocab");
        loss += lse - row[tgt];
        let dr = &mut d[ti * vocab..(ti + 1) * vocab];
        for (v, dv) in dr.iter_mut().enumerate() {
            *dv = (row[v] - lse).exp() * inv_t;
        }
        dr[tgt] -= inv_t;
    }
    (loss * inv_t, d)
}

/// Fydel Jumping Seedling — the assembled model.
pub struct Model {
    cfg: ModelConfig,
    /// Shared dictionary `G`, `K×b` complex, row-major.
    dict: Vec<Complex32>,
    /// Tied token embedding `E`, `[vocab, hidden]` row-major.
    embed: Vec<f32>,
    final_norm_gain: Vec<f32>,
    rope: Rope,
    layers: Vec<TransformerLayer>,
}

impl Model {
    /// Build a model from a (validated) config with reproducible random init.
    pub fn new(cfg: ModelConfig, seed: u64) -> Self {
        cfg.validate();
        let dict = init_dict_random(cfg.dict_k, cfg.block, seed ^ 0xD1C7, 1.0);
        let embed = init_coeffs_random(cfg.vocab * cfg.hidden, seed ^ 0xE3BE, 0.02);
        let layers = (0..cfg.n_layers)
            .map(|i| TransformerLayer::new(&cfg, i, seed.wrapping_add(i as u64 * 0x9E37)))
            .collect();
        let rope = Rope::new(cfg.head_dim, cfg.max_seq, cfg.rope_base);
        let final_norm_gain = vec![1.0; cfg.hidden];
        Self { cfg, dict, embed, final_norm_gain, rope, layers }
    }

    #[inline]
    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    /// Embed `ids` into the residual stream, `[T, H]` row-major.
    fn embed_lookup(&self, ids: &[usize]) -> Vec<f32> {
        let h = self.cfg.hidden;
        let mut x = vec![0.0f32; ids.len() * h];
        for (ti, &id) in ids.iter().enumerate() {
            debug_assert!(id < self.cfg.vocab, "token id {id} out of vocab");
            x[ti * h..(ti + 1) * h].copy_from_slice(&self.embed[id * h..(id + 1) * h]);
        }
        x
    }

    /// Forward a single sequence of token ids → logits, caching activations.
    pub fn forward(&self, ids: &[usize]) -> ModelForward {
        let (h, v) = (self.cfg.hidden, self.cfg.vocab);
        let t = ids.len();

        let mut x = self.embed_lookup(ids);

        let mut layer_fwds = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let lf = layer.forward(&self.dict, &self.rope, &x, t);
            x = lf.out.clone();
            layer_fwds.push(lf);
        }
        let final_x = x;

        // Final RMSNorm.
        let mut normed_final = vec![0.0f32; t * h];
        let mut rinv_final = vec![0.0f32; t];
        for ti in 0..t {
            let nrm = &mut normed_final[ti * h..(ti + 1) * h];
            rinv_final[ti] =
                norm::forward(&final_x[ti * h..(ti + 1) * h], &self.final_norm_gain, self.cfg.norm_eps, nrm);
        }

        // Tied LM head: logits = normed_final · Eᵀ.
        let mut logits = vec![0.0f32; t * v];
        crate::kernels::gemm::logits_from_embed(&normed_final, &self.embed, &mut logits, t, h, v);

        ModelForward { logits, token_ids: ids.to_vec(), layer_fwds, final_x, normed_final, rinv_final }
    }

    /// Reverse of [`forward`](Model::forward). `d_logits` is the gradient w.r.t.
    /// the logits (typically from [`cross_entropy`]). Returns the full gradient
    /// bundle. The probe heads are gradient-stopped side outputs; their gradients
    /// are wired separately in the training loop (the early-exit KL term), so this
    /// pass runs the layer backward with no probe gradient.
    /// Decode a layer-input residual stream `x` (`t·hidden`) through the shared
    /// final norm + tied LM head into early-exit logits (`t·vocab`). Carries no
    /// gradient; used only to build the CALM probe's consistency target.
    pub fn early_logits(&self, x: &[f32], t: usize) -> Vec<f32> {
        let h = self.cfg.hidden;
        let v = self.cfg.vocab;
        let mut normed = vec![0.0f32; t * h];
        for ti in 0..t {
            norm::forward(
                &x[ti * h..(ti + 1) * h],
                &self.final_norm_gain,
                self.cfg.norm_eps,
                &mut normed[ti * h..(ti + 1) * h],
            );
        }
        let mut logits = vec![0.0f32; t * v];
        crate::kernels::gemm::logits_from_embed(&normed, &self.embed, &mut logits, t, h, v);
        logits
    }

    /// CALM probe supervision. For each layer the early-exit prediction is the
    /// decode of that layer's *input* residual stream; the binary target is whether
    /// its argmax matches the full model's argmax. Returns the mean (unweighted) BCE
    /// loss for logging and the per-layer `d_probe_p` (length `t` each), already
    /// scaled by `weight` (the annealed probe coefficient) and averaged over layers
    /// and tokens so it can be handed straight to [`Model::backward`].
    pub fn probe_consistency(&self, fwd: &ModelForward, weight: f32, stride: usize) -> (f32, Vec<Vec<f32>>) {
        let v = self.cfg.vocab;
        let t = fwd.token_ids.len();
        let n = self.layers.len();
        let argmax = |row: &[f32]| -> usize {
            let mut bi = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &x) in row.iter().enumerate() {
                if x > bv {
                    bv = x;
                    bi = i;
                }
            }
            bi
        };
        let final_arg: Vec<usize> =
            (0..t).map(|ti| argmax(&fwd.logits[ti * v..(ti + 1) * v])).collect();
        let embed_x = self.embed_lookup(&fwd.token_ids);
        // Supervise only every `stride`-th layer's exit probe (cost control): the
        // expensive part is decoding each layer's input through the vocab head, so
        // skipping most layers cuts the cost ~`stride`×. Unsupervised layers get an
        // empty grad vector, which the backward pass treats as "no probe gradient".
        let stride = stride.max(1);
        let n_sup = (0..n).filter(|l| l % stride == 0).count();
        let denom = (n_sup * t).max(1) as f32;
        let mut total = 0.0f32;
        let mut grads = Vec::with_capacity(n);
        for l in 0..n {
            if l % stride != 0 {
                grads.push(Vec::new());
                continue;
            }
            let input: &[f32] = if l == 0 { &embed_x } else { &fwd.layer_fwds[l - 1].out };
            let early = self.early_logits(input, t);
            let pp = &fwd.layer_fwds[l].probe_p;
            let mut dpl = vec![0.0f32; t];
            for ti in 0..t {
                let tgt = if argmax(&early[ti * v..(ti + 1) * v]) == final_arg[ti] {
                    1.0
                } else {
                    0.0
                };
                let p = 1.0 / (1.0 + (-pp[ti]).exp());
                let eps = 1e-7;
                total += -(tgt * (p + eps).ln() + (1.0 - tgt) * (1.0 - p + eps).ln());
                dpl[ti] = weight * (p - tgt) / denom;
            }
            grads.push(dpl);
        }
        (total / denom, grads)
    }

    pub fn backward(
        &self,
        fwd: &ModelForward,
        d_logits: &[f32],
        d_probe_p: Option<&[Vec<f32>]>,
    ) -> ModelGrads {
        let (h, v) = (self.cfg.hidden, self.cfg.vocab);
        let t = fwd.token_ids.len();
        debug_assert_eq!(d_logits.len(), t * v);

        let mut d_embed = vec![0.0f32; v * h];

        // --- tied LM head: logits = normed_final · Eᵀ ---
        // d_normed_final = d_logits · E ; d_embed += d_logitsᵀ · normed_final.
        let mut d_normed_final = vec![0.0f32; t * h];
        crate::kernels::gemm::head_backward(
            d_logits,
            &self.embed,
            &fwd.normed_final,
            &mut d_normed_final,
            &mut d_embed,
            t,
            h,
            v,
        );

        // --- final RMSNorm ---
        let mut d_final_x = vec![0.0f32; t * h];
        let mut d_final_norm_gain = vec![0.0f32; h];
        for ti in 0..t {
            let mut dx = vec![0.0f32; h];
            let mut dg = vec![0.0f32; h];
            norm::backward(
                &fwd.final_x[ti * h..(ti + 1) * h],
                &self.final_norm_gain,
                &d_normed_final[ti * h..(ti + 1) * h],
                fwd.rinv_final[ti],
                &mut dx,
                &mut dg,
            );
            d_final_x[ti * h..(ti + 1) * h].copy_from_slice(&dx);
            for j in 0..h {
                d_final_norm_gain[j] += dg[j];
            }
        }

        // --- layer stack, in reverse; the shared dictionary grad sums across layers ---
        let embed_x = self.embed_lookup(&fwd.token_ids);
        let mut d_x = d_final_x;
        let mut d_dict: Vec<Complex32> = Vec::new();
        let mut layer_grads_rev = Vec::with_capacity(self.layers.len());
        for l in (0..self.layers.len()).rev() {
            // Layer l's forward input is the previous layer's output (or the embedding).
            let input = if l == 0 { &embed_x } else { &fwd.layer_fwds[l - 1].out };
            let lpp = d_probe_p.and_then(|p| {
                let s = p[l].as_slice();
                (!s.is_empty()).then_some(s)
            });
            let lg = self.layers[l].backward(&self.dict, &self.rope, input, &fwd.layer_fwds[l], &d_x, lpp, t);
            if d_dict.is_empty() {
                d_dict = lg.d_dict.clone();
            } else {
                for (a, b) in d_dict.iter_mut().zip(&lg.d_dict) { *a += *b; }
            }
            d_x = lg.d_hidden.clone();
            layer_grads_rev.push(lg);
        }
        layer_grads_rev.reverse();

        // --- embedding lookup: scatter-add the residual-stream grad by token id ---
        // (tied: this lands on the same E that the LM head already accumulated into.)
        for (ti, &id) in fwd.token_ids.iter().enumerate() {
            let de = &mut d_embed[id * h..(id + 1) * h];
            let dx = &d_x[ti * h..(ti + 1) * h];
            for j in 0..h {
                de[j] += dx[j];
            }
        }

        ModelGrads { d_embed, d_dict, d_final_norm_gain, layers: layer_grads_rev }
    }

    /// Allocate optimizer state for every parameter in the model.
    pub fn init_opt(&self) -> ModelOptState {
        let dict_len = self.dict.len() * 2; // complex → real pairs for optimizer state
        ModelOptState {
            embed: AdaFactorState::matrix(self.cfg.vocab, self.cfg.hidden, false),
            dict: AdaFactorState::vector(dict_len, false),
            final_norm: AdaFactorState::vector(self.cfg.hidden, false),
            layers: self.layers.iter().map(|l| l.init_opt()).collect(),
        }
    }

    /// Apply one AdaFactor step to every parameter from a (possibly accumulated)
    /// gradient bundle. The shared dictionary steps once from the model-level
    /// summed `d_dict`; the per-layer `d_dict` copies are ignored.
    pub fn apply_grad(&mut self, g: &ModelGrads, st: &mut ModelOptState, af: &AdaFactor, lr: f32) {
        af.step(&mut self.embed, &g.d_embed, &mut st.embed, lr);

        // Interleave complex dict as [re0, im0, re1, im1, ...] for the optimizer,
        // then deinterleave back.
        let n = self.dict.len();
        let mut p = vec![0.0f32; 2 * n];
        for i in 0..n {
            p[2 * i] = self.dict[i].re;
            p[2 * i + 1] = self.dict[i].im;
        }
        let mut dg = vec![0.0f32; 2 * n];
        for i in 0..g.d_dict.len().min(n) {
            dg[2 * i] = g.d_dict[i].re;
            dg[2 * i + 1] = g.d_dict[i].im;
        }
        af.step(&mut p, &dg, &mut st.dict, lr);
        for i in 0..n {
            self.dict[i] = Complex32::new(p[2 * i], p[2 * i + 1]);
        }

        af.step(&mut self.final_norm_gain, &g.d_final_norm_gain, &mut st.final_norm, lr);
        for (l, layer) in self.layers.iter_mut().enumerate() {
            layer.apply_grad(&g.layers[l], &mut st.layers[l], af, lr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small but spec-valid model config for fast tests.
    fn tiny_cfg() -> ModelConfig {
        let mut c = ModelConfig::default();
        c.n_layers = 3;
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
        c.full_attn_layers = 1; // mix of Full + Sliding layers
        c.vocab = 16;
        c.max_seq = 16;
        c.validate();
        c
    }

    #[test]
    fn forward_shapes() {
        let c = tiny_cfg();
        let (v, n) = (c.vocab, c.n_layers);
        let mut m = Model::new(c, 0xABCD);
        let ids = [1usize, 5, 0, 9, 3];
        let fwd = m.forward(&ids);
        assert_eq!(fwd.logits.len(), ids.len() * v);
        assert_eq!(fwd.layer_fwds.len(), n);
        assert!(fwd.logits.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn forward_is_deterministic() {
        let mut m = Model::new(tiny_cfg(), 0x11);
        let ids = [2usize, 2, 7, 1];
        let a = m.forward(&ids).logits;
        let b = m.forward(&ids).logits;
        assert_eq!(a, b);
    }

    #[test]
    fn model_is_causal_end_to_end() {
        // Changing a later token's id must not move token 0's logits.
        let c = tiny_cfg();
        let v = c.vocab;
        let mut m = Model::new(c, 0x77);
        let base = m.forward(&[3usize, 8, 1, 4]).logits;
        let perturbed = m.forward(&[3usize, 8, 1, 12]).logits; // only last id changed
        for vid in 0..v {
            assert!((base[vid] - perturbed[vid]).abs() < 1e-6, "future token leaked into logits[0]");
        }
    }

    #[test]
    fn cross_entropy_gradchecks() {
        let (t, vocab) = (4usize, 7usize);
        let mut rng = Lcg(0x4E11);
        let logits: Vec<f32> = (0..t * vocab).map(|_| rng.f()).collect();
        let targets = [2usize, 0, 6, 3];
        let (_, d) = cross_entropy(&logits, vocab, &targets);

        const H: f32 = 1e-3;
        for i in 0..t * vocab {
            let mut lp = logits.clone();
            lp[i] += H;
            let a = cross_entropy(&lp, vocab, &targets).0;
            lp[i] -= 2.0 * H;
            let b = cross_entropy(&lp, vocab, &targets).0;
            let fd = (a - b) / (2.0 * H);
            assert!((fd - d[i]).abs() < 1e-3, "d_logits[{i}] fd {fd} an {}", d[i]);
        }
    }

    /// End-to-end gradcheck of the new model-backward code (LM head + final norm +
    /// embedding scatter), via finite-diff over the tied embedding. `tests` is a
    /// child module, so it may perturb the private `embed` directly. Validates the
    /// weight tie: every perturbed `E` entry feeds both the LM head and (if that
    /// token id appears) the embedding lookup, and `d_embed` must capture both.
    #[test]
    fn model_backward_embed_gradchecks() {
        let c = tiny_cfg();
        let vocab = c.vocab;
        let mut m = Model::new(c, 0x9001);
        let ids = [4usize, 1, 6, 2, 4]; // token 4 repeats ⇒ exercises scatter accumulation
        let targets = [1usize, 6, 2, 4, 0];

        let fwd = m.forward(&ids);
        let (_, d_logits) = cross_entropy(&fwd.logits, vocab, &targets);
        let grads = m.backward(&fwd, &d_logits, None);

        // Base FFN routing per (layer, token), to detect top-k kinks under perturbation.
        let sel_of = |f: &ModelForward| -> Vec<Vec<usize>> {
            f.layer_fwds.iter().flat_map(|lf| lf.ffn_fwds.iter().map(|x| x.selected.clone())).collect()
        };
        let base_sel = sel_of(&fwd);

        const H: f32 = 1e-3;
        let mut rng = Lcg(0xBEEF);
        let mut checked = 0;
        // Sample a spread of embedding entries (full V·H is large).
        for _ in 0..24 {
            let i = (rng.0 as usize) % (vocab * m.config().hidden);
            rng.f(); // advance
            let save = m.embed[i];
            m.embed[i] = save + H;
            let fp = m.forward(&ids);
            let lp = cross_entropy(&fp.logits, vocab, &targets).0;
            m.embed[i] = save - H;
            let fm = m.forward(&ids);
            let lm = cross_entropy(&fm.logits, vocab, &targets).0;
            m.embed[i] = save;
            if sel_of(&fp) != base_sel || sel_of(&fm) != base_sel {
                continue; // routing flipped: non-smooth, skip
            }
            let fd = (lp - lm) / (2.0 * H);
            let an = grads.d_embed[i];
            assert!((fd - an).abs() < 2e-2 + 6e-2 * an.abs(), "d_embed[{i}] fd {fd} an {an}");
            checked += 1;
        }
        assert!(checked >= 8, "too many coords skipped ({checked}) — test ineffective");
    }

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
    }
}
