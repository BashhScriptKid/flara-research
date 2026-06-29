//! Optimizer driver: holds the AdaFactor hyperparameters plus the whole-model
//! optimizer-state tree, and applies a step to a [`Model`] from a gradient bundle.
//!
//! Gradient accumulation across micro-batches (to the ~512K-token effective batch)
//! is the training loop's concern and lands alongside it; this type performs the
//! single AdaFactor update once a gradient bundle is ready.

use crate::kernels::optimizer::AdaFactor;
use crate::model::model::{Model, ModelGrads, ModelOptState};

/// A configured AdaFactor optimizer bound to one model's parameter shapes.
pub struct Optimizer {
    af: AdaFactor,
    state: ModelOptState,
}

impl Optimizer {
    /// Build with default AdaFactor hyperparameters, sizing state from `model`.
    pub fn new(model: &Model) -> Self {
        Self { af: AdaFactor::default(), state: model.init_opt() }
    }

    /// Build with explicit AdaFactor hyperparameters.
    pub fn with_config(model: &Model, af: AdaFactor) -> Self {
        Self { af, state: model.init_opt() }
    }

    /// Apply one optimizer step to `model` from `grads` at learning rate `lr`.
    pub fn step(&mut self, model: &mut Model, grads: &ModelGrads, lr: f32) {
        model.apply_grad(grads, &mut self.state, &self.af, lr);
    }

    /// Borrow the AdaFactor state tree (for checkpointing).
    pub fn state(&self) -> &ModelOptState {
        &self.state
    }

    /// Rebuild an optimizer from a configured AdaFactor and a restored state tree.
    pub fn from_parts(af: AdaFactor, state: ModelOptState) -> Self {
        Self { af, state }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::ModelConfig;
    use crate::model::model::cross_entropy;

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
        c.vocab = 16;
        c.max_seq = 16;
        c.validate();
        c
    }

    /// The closure proof: forward → cross-entropy → backward → AdaFactor step must
    /// drive the loss on one fixed batch sharply down (the model overfits it). This
    /// exercises every parameter's `apply_grad` path end to end. Relative-step is
    /// disabled so descent is reliable — its behaviour is checked in `optimizer.rs`.
    #[test]
    fn overfits_a_fixed_batch() {
        let mut model = Model::new(tiny_cfg(), 0x0FF1);
        let vocab = model.config().vocab;
        let ids = [1usize, 5, 2, 9, 3, 0, 7, 4];
        let targets = [5usize, 2, 9, 3, 0, 7, 4, 1];

        let af = AdaFactor { relative_step: false, ..AdaFactor::default() };
        let mut opt = Optimizer::with_config(&model, af);

        let l0 = cross_entropy(&model.forward(&ids).logits, vocab, &targets).0;
        let mut last = l0;
        for _ in 0..300 {
            let f = model.forward(&ids);
            let (loss, d_logits) = cross_entropy(&f.logits, vocab, &targets);
            let g = model.backward(&f, &d_logits, None);
            opt.step(&mut model, &g, 0.05);
            last = loss;
        }
        assert!(last.is_finite(), "loss diverged to {last}");
        assert!(last < l0 * 0.5, "loss did not descend enough: {l0} -> {last}");
    }

    /// Gradient accumulation over micro-batches must equal the mean of the
    /// per-batch gradients: `add` then `scale(1/n)` == elementwise average.
    #[test]
    fn grad_accum_equals_mean() {
        let mut model = Model::new(tiny_cfg(), 0x5EED);
        let vocab = model.config().vocab;
        let mut batch = |ids: &[usize], tgt: &[usize]| {
            let f = model.forward(ids);
            let (_, dl) = cross_entropy(&f.logits, vocab, tgt);
            model.backward(&f, &dl, None)
        };
        let g1 = batch(&[1, 2, 3, 4, 5, 6, 7, 8], &[2, 3, 4, 5, 6, 7, 8, 1]);
        let g2 = batch(&[8, 7, 6, 5, 4, 3, 2, 1], &[7, 6, 5, 4, 3, 2, 1, 8]);

        let mut acc = batch(&[1, 2, 3, 4, 5, 6, 7, 8], &[2, 3, 4, 5, 6, 7, 8, 1]);
        acc.add(&g2);
        acc.scale(0.5);

        for i in 0..acc.d_embed.len() {
            let want = 0.5 * (g1.d_embed[i] + g2.d_embed[i]);
            assert!((acc.d_embed[i] - want).abs() < 1e-6, "d_embed[{i}]: {} vs {want}", acc.d_embed[i]);
        }
        for i in 0..acc.d_dict.len() {
            let want = 0.5 * (g1.d_dict[i] + g2.d_dict[i]);
            let got = acc.d_dict[i];
            let err = (got - want).norm();
            assert!(err < 1e-6, "d_dict[{i}] mismatch: got {got} want {want}");
        }
        let a = &acc.layers[0].d_wq;
        for i in 0..a.len() {
            let want = 0.5 * (g1.layers[0].d_wq[i] + g2.layers[0].d_wq[i]);
            assert!((a[i] - want).abs() < 1e-6, "d_wq[{i}]: {} vs {want}", a[i]);
        }
        // No probe loss was supplied, so probe grads stay empty and untouched.
        assert!(acc.layers[0].d_probe_w.is_empty());
    }
}
