//! Training entrypoint for Fydel Jumping Seedling.
//!
//! Fetches a tokenizer + text dataset from the Hugging Face hub, builds the model
//! (vocab driven by the tokenizer), and runs the training loop with periodic
//! checkpointing. Knobs are plain constants below — shrink the architecture in
//! `ModelConfig` for a quick proof run before committing to the full 1B.

use std::path::PathBuf;

use fydel::kernels::optimizer::AdaFactor;
use fydel::model::config::ModelConfig;
use fydel::model::model::Model;
use fydel::train::data::{self, hf_corpus};
use fydel::train::optim::Optimizer;
use fydel::train::r#loop::{TrainConfig, train};
use fydel::train::schedule::WsdSchedule;

fn main() {
    let cache_dir = PathBuf::from(".data-cache");
    let ckpt_path = PathBuf::from("checkpoints/fydel.ckpt");
    if let Some(p) = ckpt_path.parent() {
        let _ = std::fs::create_dir_all(p);
    }

    // --- data: tokenizer + corpus from the Hugging Face hub ---
    // `max_chars` caps the tokenized text for a first run; raise/remove for the
    // full corpus.
    eprintln!("fetching tokenizer ({}) + dataset …", data::DEFAULT_TOKENIZER_REPO);
    let (tokenizer, mut reader) =
        hf_corpus(data::DEFAULT_TOKENIZER_REPO, data::DEFAULT_DATASET_URL, &cache_dir, Some(2_000_000))
            .expect("failed to fetch/tokenize corpus");
    let vocab = tokenizer.get_vocab_size(true);
    eprintln!("corpus ready: {} tokens, vocab {}", reader.len(), vocab);

    // --- model: Fydel-1B architecture, vocab driven by the tokenizer ---
    let mut cfg = ModelConfig::default();
    cfg.vocab = vocab;
    cfg.validate();
    let mut model = Model::new(cfg, 0xF1DE1);
    let mut opt = Optimizer::with_config(&model, AdaFactor::default());

    // --- schedule + loop config ---
    let total_steps = 100_000;
    let train_cfg = TrainConfig {
        total_steps,
        micro_batches: 8,
        seq_len_init: 512,
        seq_len_final: 1024,
        curriculum_switch: 20_000,
        schedule: WsdSchedule::new(3e-3, 3e-4, 2_000, total_steps / 5, total_steps),
        probe_weight: 0.1,
        probe_anneal_steps: 5_000,
    };

    eprintln!("training for {total_steps} steps (checkpoint every 1000) …");
    let hist = train(&mut model, &mut opt, &mut reader, &train_cfg, Some((1_000, &ckpt_path)), 25, 0);

    if let Some(last) = hist.last() {
        eprintln!("done: final ce {:.4} at step {}", last.ce, last.step);
    }
    fydel::train::checkpoint::save(&model, opt.state(), &ckpt_path)
        .expect("final checkpoint save failed");
    eprintln!("final checkpoint written to {}", ckpt_path.display());
}
