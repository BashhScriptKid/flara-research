//! The training loop: micro-batch gradient accumulation, the WSD learning-rate
//! schedule, a two-phase sequence-length curriculum, and annealed CALM probe
//! supervision, driving an [`Optimizer`] over a [`Model`].
//!
//! Data arrives through the [`BatchSource`] trait so the loop is exercised on a
//! synthetic, fully deterministic source in tests; a real tokenized-corpus reader
//! plugs in behind the same trait without touching the loop.

use crate::model::model::{Model, ModelGrads, cross_entropy};
use crate::train::optim::Optimizer;
use crate::train::schedule::WsdSchedule;

/// A source of training batches. `next_batch` returns `(ids, targets)` of length
/// `seq_len` (targets are the next-token shift of ids), or `None` when exhausted.
pub trait BatchSource {
    fn next_batch(&mut self, seq_len: usize) -> Option<(Vec<usize>, Vec<usize>)>;
}

/// Static configuration for a training run.
pub struct TrainConfig {
    /// Number of optimizer steps to run.
    pub total_steps: usize,
    /// Micro-batches accumulated per optimizer step (the effective-batch multiplier).
    pub micro_batches: usize,
    /// Sequence length before the curriculum switch.
    pub seq_len_init: usize,
    /// Sequence length at/after the curriculum switch.
    pub seq_len_final: usize,
    /// Step at which the sequence length jumps from init to final.
    pub curriculum_switch: usize,
    /// Learning-rate schedule (queried per step).
    pub schedule: WsdSchedule,
    /// Peak CALM probe-loss coefficient (λ) once fully annealed in.
    pub probe_weight: f32,
    /// Steps over which the probe coefficient ramps linearly from 0 to `probe_weight`.
    pub probe_anneal_steps: usize,
}

impl TrainConfig {
    /// Sequence length in effect at `step`.
    pub fn seq_len(&self, step: usize) -> usize {
        if step < self.curriculum_switch { self.seq_len_init } else { self.seq_len_final }
    }

    /// Annealed CALM probe coefficient at `step`.
    pub fn probe_coeff(&self, step: usize) -> f32 {
        let frac = (step as f32 / self.probe_anneal_steps.max(1) as f32).min(1.0);
        self.probe_weight * frac
    }
}

/// One step's reported metrics.
#[derive(Clone, Copy, Debug)]
pub struct StepMetrics {
    pub step: usize,
    pub lr: f32,
    /// Mean cross-entropy over the step's micro-batches.
    pub ce: f32,
    /// Mean (unweighted) CALM probe BCE over the step's micro-batches.
    pub probe_bce: f32,
}

/// Run the training loop, returning per-step metrics. Stops early if the source
/// is exhausted mid-step.
///
/// `checkpoint` optionally saves model + optimizer state every `n` steps to a path,
/// so a long run survives interruption (see [`crate::train::checkpoint`]).
pub fn train<B: BatchSource>(
    model: &mut Model,
    opt: &mut Optimizer,
    src: &mut B,
    cfg: &TrainConfig,
    checkpoint: Option<(usize, &std::path::Path)>,
) -> Vec<StepMetrics> {
    let vocab = model.config().vocab;
    let mut history = Vec::with_capacity(cfg.total_steps);

    for step in 0..cfg.total_steps {
        let seq_len = cfg.seq_len(step);
        let probe_coeff = cfg.probe_coeff(step);

        let mut acc: Option<ModelGrads> = None;
        let mut ce_sum = 0.0f32;
        let mut probe_sum = 0.0f32;
        let mut nb = 0usize;

        for _ in 0..cfg.micro_batches {
            let Some((ids, targets)) = src.next_batch(seq_len) else { break };
            let fwd = model.forward(&ids);
            let (ce, d_logits) = cross_entropy(&fwd.logits, vocab, &targets);
            // CALM: supervise every 8th layer's exit probe (cost control), and skip
            // probe decoding entirely while the anneal coefficient is ~0.
            let (probe_bce, d_probe_p) = if probe_coeff > 0.0 {
                const PROBE_STRIDE: usize = 8;
                let (b, g) = model.probe_consistency(&fwd, probe_coeff, PROBE_STRIDE);
                (b, Some(g))
            } else {
                (0.0, None)
            };
            let g = model.backward(&fwd, &d_logits, d_probe_p.as_deref());

            ce_sum += ce;
            probe_sum += probe_bce;
            nb += 1;
            match acc.as_mut() {
                Some(a) => a.add(&g),
                None => acc = Some(g),
            }
        }

        if nb == 0 {
            break;
        }
        let mut grads = acc.unwrap();
        grads.scale(1.0 / nb as f32);
        let lr = cfg.schedule.lr(step);
        opt.step(model, &grads, lr);

        history.push(StepMetrics {
            step,
            lr,
            ce: ce_sum / nb as f32,
            probe_bce: probe_sum / nb as f32,
        });

        if let Some((every, path)) = checkpoint {
            if every > 0 && (step + 1) % every == 0 {
                let m = *history.last().unwrap();
                match crate::train::checkpoint::save(model, opt.state(), path) {
                    Ok(()) => eprintln!(
                        "step {:>7} | lr {:.3e} | ce {:.4} | probe {:.4} | saved {}",
                        m.step,
                        m.lr,
                        m.ce,
                        m.probe_bce,
                        path.display()
                    ),
                    Err(e) => eprintln!("checkpoint save failed at step {}: {e}", m.step),
                }
            }
        }
    }

    history
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::optimizer::AdaFactor;
    use crate::model::config::ModelConfig;

    /// Deterministic source: a fixed token pattern, with targets the next-token shift.
    struct SyntheticSource {
        pattern: Vec<usize>,
    }

    impl BatchSource for SyntheticSource {
        fn next_batch(&mut self, seq_len: usize) -> Option<(Vec<usize>, Vec<usize>)> {
            let n = self.pattern.len();
            let ids = (0..seq_len).map(|i| self.pattern[i % n]).collect();
            let targets = (0..seq_len).map(|i| self.pattern[(i + 1) % n]).collect();
            Some((ids, targets))
        }
    }

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

    /// The full loop must drive cross-entropy down on a fixed synthetic batch,
    /// with CALM probe supervision active. Relative-step is disabled so descent is
    /// reliable (its behaviour is validated separately in `optimizer.rs`).
    #[test]
    fn loop_descends_on_synthetic_data() {
        let mut model = Model::new(tiny_cfg(), 0xC0FFEE);
        let af = AdaFactor { relative_step: false, ..AdaFactor::default() };
        let mut opt = Optimizer::with_config(&model, af);
        let mut src = SyntheticSource { pattern: vec![1, 5, 2, 9, 3, 0, 7, 4] };

        let cfg = TrainConfig {
            total_steps: 120,
            micro_batches: 2,
            seq_len_init: 8,
            seq_len_final: 8,
            curriculum_switch: 60,
            schedule: WsdSchedule::new(0.05, 0.0, 5, 30, 120),
            probe_weight: 0.1,
            probe_anneal_steps: 40,
        };

        let hist = train(&mut model, &mut opt, &mut src, &cfg, None);
        assert_eq!(hist.len(), 120);
        let first = hist[0].ce;
        let last = hist.last().unwrap().ce;
        assert!(last.is_finite(), "ce diverged to {last}");
        assert!(last < first * 0.5, "ce did not descend: {first} -> {last}");
        // The probe loss must be finite and present (supervision wired through).
        assert!(hist.last().unwrap().probe_bce.is_finite());
    }

    /// The sequence-length curriculum must switch at the configured step.
    #[test]
    fn curriculum_switches_seq_len() {
        let cfg = TrainConfig {
            total_steps: 10,
            micro_batches: 1,
            seq_len_init: 4,
            seq_len_final: 8,
            curriculum_switch: 5,
            schedule: WsdSchedule::new(0.05, 0.0, 1, 1, 10),
            probe_weight: 0.0,
            probe_anneal_steps: 1,
        };
        assert_eq!(cfg.seq_len(0), 4);
        assert_eq!(cfg.seq_len(4), 4);
        assert_eq!(cfg.seq_len(5), 8);
        assert_eq!(cfg.seq_len(9), 8);
    }
}
