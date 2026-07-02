//! Small real-architecture baseline: same Full/Sliding attention split as the
//! Fydel-1B spec (ratio-matched: 3 Full / 9 Sliding vs. production's 24 / 96),
//! shrunk to train in minutes on CPU. Exists to answer one question the
//! block-sparse-attention TODO in RESEARCH_LOG.md needs answered first: how much
//! held-out quality does the *existing* sliding-window design already cost,
//! relative to full attention on the same trained weights?
//!
//! Uses the real GPT-2 BPE tokenizer/vocab (50257) over a Hugging Face corpus
//! fetch, so it's the closer proxy to the actual 1B target's tokenization
//! scheme. See `train_small_lod` for a byte-level-vocab fast-iteration
//! variant of this same architecture — faster to run, but not a stand-in for
//! this binary's quality conclusions (char-level LMs behave differently
//! w.r.t. context length than subword LMs).
//!
//! `cargo run --release --bin train_small` trains + checkpoints.
//! `cargo run --release --bin train_small -- --eval` loads the checkpoint and
//! reports held-out CE for both the as-trained (sliding) and an all-full-attention
//! reconstruction sharing the identical trained weights.

use std::path::PathBuf;

use fydel::kernels::optimizer::AdaFactor;
use fydel::model::config::ModelConfig;
use fydel::model::model::{Model, cross_entropy};
use fydel::train::data::{self, hf_corpus, CorpusReader};
use fydel::train::optim::Optimizer;
use fydel::train::r#loop::{BatchSource, TrainConfig, train};
use fydel::train::schedule::WsdSchedule;

fn small_cfg(vocab: usize) -> ModelConfig {
    let mut c = ModelConfig::default();
    c.n_layers = 12;
    c.full_attn_layers = 3; // 3/12 = 1/4, same ratio as production's 24/96
    c.hidden = 256;
    c.n_q_heads = 4;
    c.n_kv_heads = 1;
    c.head_dim = 64;
    c.ffn_dim = 768;
    c.block = 64;
    c.n_active = 3;
    c.dict_k = 8;
    c.kv_block = 64;
    c.window = 64;
    c.vocab = vocab;
    c.max_seq = 512;
    c.validate();
    c
}

const EVAL_SEQ_LEN: usize = 256;
const EVAL_WINDOWS: usize = 200;

fn eval_loss<B: BatchSource>(model: &Model, src: &mut B, seq_len: usize, n_windows: usize) -> f32 {
    let vocab = model.config().vocab;
    let mut total = 0.0f32;
    let mut n = 0usize;
    for _ in 0..n_windows {
        let Some((ids, targets)) = src.next_batch(seq_len) else { break };
        let fwd = model.forward(&ids);
        let (ce, _) = cross_entropy(&fwd.logits, vocab, &targets);
        total += ce;
        n += 1;
    }
    total / n.max(1) as f32
}

fn load_val_reader(cache_dir: &std::path::Path) -> CorpusReader {
    let (_tok, reader) =
        hf_corpus(data::DEFAULT_TOKENIZER_REPO, data::DEFAULT_DATASET_URL, cache_dir, Some(500_000))
            .expect("failed to fetch/tokenize corpus");
    let (_train_r, val_r) = reader.split_val(0.1);
    val_r
}

fn main() {
    let cache_dir = PathBuf::from(".data-cache");
    let ckpt_path = PathBuf::from("checkpoints/fydel_small.ckpt");
    if let Some(p) = ckpt_path.parent() {
        let _ = std::fs::create_dir_all(p);
    }

    if std::env::args().nth(1).as_deref() == Some("--eval") {
        let (model, _opt) =
            fydel::train::checkpoint::load(&ckpt_path).expect("no checkpoint found — run training first");
        eprintln!("checkpoint loaded: n_layers={} full_attn_layers={} window={}",
            model.config().n_layers, model.config().full_attn_layers, model.config().window);

        let mut val_r = load_val_reader(&cache_dir);
        eprintln!("held-out set: {} tokens", val_r.len());

        let sliding_val_r = val_r.clone();
        let sliding_loss = eval_loss(&model, &mut val_r, EVAL_SEQ_LEN, EVAL_WINDOWS);

        let mut ckpt = model.to_checkpoint();
        ckpt.cfg.full_attn_layers = ckpt.cfg.n_layers; // same trained weights, all layers Full
        let full_model = Model::from_checkpoint(&ckpt);
        let mut full_val_r = sliding_val_r;
        let full_loss = eval_loss(&full_model, &mut full_val_r, EVAL_SEQ_LEN, EVAL_WINDOWS);

        eprintln!("sliding-window (as trained)        held-out CE: {sliding_loss:.4} nats");
        eprintln!("all-full-attention (same weights)  held-out CE: {full_loss:.4} nats");
        eprintln!("quality cost of sliding-window:     {:.4} nats", sliding_loss - full_loss);
        return;
    }

    eprintln!("fetching tokenizer ({}) + dataset …", data::DEFAULT_TOKENIZER_REPO);
    let (tok, reader) =
        hf_corpus(data::DEFAULT_TOKENIZER_REPO, data::DEFAULT_DATASET_URL, &cache_dir, Some(500_000))
            .expect("failed to fetch/tokenize corpus");
    let vocab = tok.get_vocab_size(true);
    eprintln!("corpus ready: {} tokens, vocab {}", reader.len(), vocab);
    let (mut train_r, val_r) = reader.split_val(0.1);
    eprintln!("train/val split: {} / {} tokens", train_r.len(), val_r.len());

    // RESUME=0 forces a fresh run even if a checkpoint exists (default: resume).
    let resume = std::env::var("RESUME").ok().as_deref() != Some("0");
    let (mut model, mut opt, start_step) = if resume && ckpt_path.exists() {
        let (m, opt_state) = fydel::train::checkpoint::load(&ckpt_path)
            .expect("failed to load checkpoint for resume (set RESUME=0 to start fresh instead)");
        let step = fydel::train::checkpoint::load_step(&ckpt_path).unwrap_or(0);
        eprintln!("resuming from {} at step {step} (set RESUME=0 to start fresh)", ckpt_path.display());
        (m, Optimizer::from_parts(AdaFactor::default(), opt_state), step)
    } else {
        let m = Model::new(small_cfg(vocab), 0xF1DE1_5EED);
        let opt = Optimizer::with_config(&m, AdaFactor::default());
        (m, opt, 0)
    };

    let total_steps = 3000;
    let train_cfg = TrainConfig {
        total_steps,
        micro_batches: 4,
        seq_len_init: 128,
        seq_len_final: 256,
        curriculum_switch: 1500,
        schedule: WsdSchedule::new(3e-3, 3e-4, 200, total_steps / 5, total_steps),
        probe_weight: 0.1,
        probe_anneal_steps: 500,
    };

    eprintln!("training small real-arch model ({} layers, {} full / {} sliding) from step {start_step} to {total_steps} …",
        model.config().n_layers, model.config().full_attn_layers,
        model.config().n_layers - model.config().full_attn_layers);
    let hist = train(&mut model, &mut opt, &mut train_r, &train_cfg, Some((500, &ckpt_path)), 5, start_step);

    if let Some(last) = hist.last() {
        eprintln!("done: final ce {:.4} at step {}", last.ce, last.step);
    }
    fydel::train::checkpoint::save(&model, opt.state(), &ckpt_path)
        .expect("final checkpoint save failed");
    fydel::train::checkpoint::save_step(&ckpt_path, hist.last().map_or(start_step, |m| m.step + 1))
        .expect("final step-sidecar save failed");
    eprintln!("checkpoint written to {}", ckpt_path.display());
}
