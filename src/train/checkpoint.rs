//! Checkpoint persistence: serialize a model's learned parameters together with
//! the optimizer state to a single bincode file, and restore both.
//!
//! Only learned tensors and config are written; the FFT plans, RoPE tables, and
//! attention runners are rebuilt from config on load (see [`Model::from_checkpoint`]),
//! so they never bloat the checkpoint nor need to be serializable.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::model::model::{Checkpoint, Model, ModelOptState};

fn to_io(e: bincode::Error) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
}

/// Write the model's parameters and the optimizer state to `path`.
pub fn save(model: &Model, opt_state: &ModelOptState, path: impl AsRef<Path>) -> std::io::Result<()> {
    let ckpt = model.to_checkpoint();
    let mut w = BufWriter::new(File::create(path)?);
    bincode::serialize_into(&mut w, &(&ckpt, opt_state)).map_err(to_io)
}

/// Read a checkpoint from `path`, returning a fully reconstructed model (FFT plans
/// rebuilt from config) and the restored optimizer state.
pub fn load(path: impl AsRef<Path>) -> std::io::Result<(Model, ModelOptState)> {
    let r = BufReader::new(File::open(path)?);
    let (ckpt, opt): (Checkpoint, ModelOptState) = bincode::deserialize_from(r).map_err(to_io)?;
    Ok((Model::from_checkpoint(&ckpt), opt))
}

/// Sidecar path for the step counter — the checkpoint proper has no step field
/// (it's a snapshot of *parameters*, not training progress), so the step this
/// checkpoint was saved at is tracked alongside it instead.
fn step_path(ckpt_path: &Path) -> std::path::PathBuf {
    ckpt_path.with_extension("step")
}

/// Write the step counter alongside a checkpoint, for resume.
pub fn save_step(ckpt_path: impl AsRef<Path>, step: usize) -> std::io::Result<()> {
    std::fs::write(step_path(ckpt_path.as_ref()), step.to_string())
}

/// Read the step counter saved alongside a checkpoint. `Ok(0)` if no sidecar
/// exists (e.g. a checkpoint saved before this feature existed).
pub fn load_step(ckpt_path: impl AsRef<Path>) -> std::io::Result<usize> {
    match std::fs::read_to_string(step_path(ckpt_path.as_ref())) {
        Ok(s) => s.trim().parse().map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::optimizer::AdaFactor;
    use crate::model::config::ModelConfig;
    use crate::model::model::cross_entropy;
    use crate::train::optim::Optimizer;

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

    fn af() -> AdaFactor {
        AdaFactor { relative_step: false, ..AdaFactor::default() }
    }

    fn train_step(model: &mut Model, opt: &mut Optimizer, ids: &[usize], targets: &[usize]) {
        let vocab = model.config().vocab;
        let f = model.forward(ids);
        let (_, dl) = cross_entropy(&f.logits, vocab, targets);
        let g = model.backward(&f, &dl, None);
        opt.step(model, &g, 0.05);
    }

    /// A saved checkpoint must restore both parameters (identical logits) and the
    /// optimizer state (an identical follow-up step keeps the two models in lockstep).
    #[test]
    fn roundtrip_preserves_params_and_opt_state() {
        let ids = [1usize, 5, 2, 9, 3, 0, 7, 4];
        let targets = [5usize, 2, 9, 3, 0, 7, 4, 1];

        let mut model = Model::new(tiny_cfg(), 0x1234);
        let mut opt = Optimizer::with_config(&model, af());
        for _ in 0..5 {
            train_step(&mut model, &mut opt, &ids, &targets);
        }
        let logits_before = model.forward(&ids).logits;

        let path = std::env::temp_dir().join(format!("fydel_ckpt_{}.bin", std::process::id()));
        save(&model, opt.state(), &path).unwrap();
        let (mut model2, opt_state2) = load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        // Parameters restored: bit-identical logits.
        let logits_after = model2.forward(&ids).logits;
        assert_eq!(logits_before.len(), logits_after.len());
        for (a, b) in logits_before.iter().zip(&logits_after) {
            assert_eq!(a.to_bits(), b.to_bits(), "param mismatch after reload");
        }

        // Optimizer state restored: one more identical step must agree bit-for-bit.
        let mut m1 = model;
        let mut o1 = opt;
        let mut m2 = model2;
        let mut o2 = Optimizer::from_parts(af(), opt_state2);
        train_step(&mut m1, &mut o1, &ids, &targets);
        train_step(&mut m2, &mut o2, &ids, &targets);
        let l1 = m1.forward(&ids).logits;
        let l2 = m2.forward(&ids).logits;
        for (a, b) in l1.iter().zip(&l2) {
            assert_eq!(a.to_bits(), b.to_bits(), "post-step mismatch ⇒ optimizer state not restored");
        }
    }
}
