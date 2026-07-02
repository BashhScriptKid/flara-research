//! Low-LOD fast-iteration variant of `train_small`: same architecture (same
//! Full/Sliding attention split, same shrunk dims), but byte-level vocab (128,
//! printable ASCII, same scheme as `train_char.rs`) over the local
//! `data/input.txt` corpus instead of `train_small`'s full GPT-2 BPE vocab
//! (50257). The GPT-2 vocab makes the tied embedding/LM-head ~99% of the toy
//! model's params and (uncompressed) likely most of its step time —
//! dominating the very kernel work this project is trying to exercise. Good
//! for fast kernel iteration; NOT a stand-in for how the real BPE-tokenized
//! 1B model would behave (char-level LMs have a different relationship to
//! context length than subword LMs) — any quality-cost conclusions here are
//! specific to this config, not directly transferable to `train_small`'s.
//!
//! `--full-attn` trains/evaluates a *separately trained* all-Full-attention
//! model instead of the default sliding-window split, to a distinct
//! checkpoint. This exists because swapping `full_attn_layers` post-hoc on a
//! single trained checkpoint (tried first) is confounded: `window` isn't a
//! learned parameter, so forcing full attention on weights trained under
//! sliding-window puts every downstream layer out of the distribution it was
//! trained on — that measures brittleness to the swap, not the quality
//! sliding-window actually costs. Two independently trained models, evaluated
//! separately on the same held-out set, is the honest comparison.
//!
//! `cargo run --release --bin train_small_lod [-- --full-attn]` trains + checkpoints.
//! `cargo run --release --bin train_small_lod -- --eval [--full-attn]` loads
//! the corresponding checkpoint and reports its held-out CE.

use std::path::PathBuf;

use fydel::kernels::optimizer::AdaFactor;
use fydel::model::config::ModelConfig;
use fydel::model::model::{Model, cross_entropy};
use fydel::train::data::CorpusReader;
use fydel::train::optim::Optimizer;
use fydel::train::r#loop::{BatchSource, TrainConfig, train};
use fydel::train::schedule::WsdSchedule;

const VOCAB: usize = 128; // printable ASCII, byte-level — see module doc.
const DATA_PATH: &str = "data/input.txt";

fn load_byte_corpus() -> CorpusReader {
    let text = std::fs::read_to_string(DATA_PATH)
        .unwrap_or_else(|e| panic!("failed to read {DATA_PATH}: {e}"));
    let tokens: Vec<u32> = text.bytes().map(|b| (b as u32).min(VOCAB as u32 - 1)).collect();
    CorpusReader::new(tokens)
}

fn small_cfg(vocab: usize, full_attn: bool) -> ModelConfig {
    let mut c = ModelConfig::default();
    c.n_layers = 12;
    c.full_attn_layers = if full_attn { 12 } else { 3 }; // 3/12 matches production's 24/96 ratio
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

fn load_val_reader() -> CorpusReader {
    let reader = load_byte_corpus();
    let (_train_r, val_r) = reader.split_val(0.1);
    val_r
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let full_attn = args.iter().any(|a| a == "--full-attn");
    let ckpt_path = PathBuf::from(if full_attn {
        "checkpoints/fydel_small_lod_full.ckpt"
    } else {
        "checkpoints/fydel_small_lod.ckpt"
    });
    if let Some(p) = ckpt_path.parent() {
        let _ = std::fs::create_dir_all(p);
    }

    if args.iter().any(|a| a == "--eval") {
        let (model, _opt) =
            fydel::train::checkpoint::load(&ckpt_path).expect("no checkpoint found — run training first");
        eprintln!("checkpoint loaded: n_layers={} full_attn_layers={} window={}",
            model.config().n_layers, model.config().full_attn_layers, model.config().window);

        let mut val_r = load_val_reader();
        eprintln!("held-out set: {} tokens", val_r.len());
        let loss = eval_loss(&model, &mut val_r, EVAL_SEQ_LEN, EVAL_WINDOWS);
        eprintln!("held-out CE ({}): {loss:.4} nats",
            if full_attn { "all-full-attention, separately trained" } else { "sliding-window, as trained" });
        return;
    }

    let reader = load_byte_corpus();
    let vocab = VOCAB;
    eprintln!("corpus ready: {} tokens, vocab {} (byte-level, {})", reader.len(), vocab, DATA_PATH);
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
        let m = Model::new(small_cfg(vocab, full_attn), 0xF1DE1_5EED);
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
