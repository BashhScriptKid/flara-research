//! No-network profiling harness: builds a representative mid-size model, runs a
//! synthetic forward/backward/optimizer loop, and reports per-phase timing.
//!
//! Run `cargo run --release --bin profile` for in-process numbers, or under a
//! sampling profiler for function-level attribution:
//!   cargo build --release --bin profile
//!   perf record -g ./target/release/profile && perf report
//!
//! The config keeps the same head_dim / block / dict_k / full-vs-sliding mix as
//! Fydel-1B so the per-op cost shape is realistic, just with fewer/narrower layers
//! so a step is fast enough to iterate on.

use std::time::Instant;

use fydel::kernels::optimizer::AdaFactor;
use fydel::kernels::profiling;
use fydel::model::config::ModelConfig;
use fydel::model::model::{Model, cross_entropy};
use fydel::train::optim::Optimizer;

/// Read an environment variable as `usize`, falling back to `default`.
fn envu(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn profile_cfg() -> ModelConfig {
    // FULL=1 uses the production 1B spec (96L, hidden 896) as-is.
    if envu("FULL", 0) == 1 {
        let c = ModelConfig::default();
        c.validate();
        return c;
    }
    // TRAIN_SMALL_LOD=1 exactly matches src/bin/train_small_lod.rs's shape,
    // for profiling the config the real end-to-end synthetic bench uses.
    if envu("TRAIN_SMALL_LOD", 0) == 1 {
        let mut c = ModelConfig::default();
        c.n_layers = 12;
        c.full_attn_layers = 3;
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
        c.vocab = envu("VOCAB", 128);
        c.max_seq = 512;
        c.validate();
        return c;
    }
    let mut c = ModelConfig::default();
    c.n_layers = envu("N_LAYERS", 12);
    c.hidden = envu("HIDDEN", 512);
    c.n_q_heads = envu("N_Q_HEADS", 8); // must divide HIDDEN with HEAD_DIM
    c.n_kv_heads = envu("N_KV_HEADS", 2);
    c.head_dim = envu("HEAD_DIM", 64);
    c.ffn_dim = envu("FFN_DIM", 2048);
    c.block = envu("BLOCK", 64);
    c.n_active = envu("N_ACTIVE", 12);
    c.dict_k = envu("DICT_K", 32);
    c.kv_block = envu("KV_BLOCK", 64);
    c.window = envu("WINDOW", 256);
    c.full_attn_layers = envu("FULL_ATTN", 3);
    c.vocab = envu("VOCAB", 8192);
    c.max_seq = 512;
    c.validate();
    c
}

fn main() {
    let seq = envu("SEQ", 256);
    let warmup = envu("WARMUP", 5);
    let steps = envu("STEPS", 50);
    let fwd_only = envu("FWD_ONLY", 0) == 1;

    let cfg = profile_cfg();
    let vocab = cfg.vocab;
    eprintln!(
        "profile: {} layers, hidden {}, ffn {}, vocab {}, seq {}, {} steps",
        cfg.n_layers, cfg.hidden, cfg.ffn_dim, vocab, seq, steps
    );

    let mut model = Model::new(cfg, 0x1234);
    let mut opt = Optimizer::with_config(&model, AdaFactor::default());

    // Deterministic synthetic batch (next-token shift).
    let ids: Vec<usize> = (0..seq).map(|i| i.wrapping_mul(2654435761) % vocab).collect();
    let targets: Vec<usize> = (0..seq).map(|i| (i + 1).wrapping_mul(2654435761) % vocab).collect();

    let mut pool = fydel::kernels::scratch::BufPool::new();
    let run_step = |model: &mut Model, opt: &mut Optimizer, t: &mut [f64; 3], pool: &mut fydel::kernels::scratch::BufPool| {
        let a = Instant::now();
        let fwd = model.forward(&ids, pool);
        t[0] += a.elapsed().as_secs_f64();

        if fwd_only { return; }

        let (_, d_logits) = cross_entropy(&fwd.logits, vocab, &targets);

        let b = Instant::now();
        let g = model.backward(fwd, &d_logits, None, pool);
        t[1] += b.elapsed().as_secs_f64();

        let c = Instant::now();
        opt.step(model, &g, 1e-3);
        t[2] += c.elapsed().as_secs_f64();
    };

    let mut warm = [0.0f64; 3];
    for _ in 0..warmup {
        run_step(&mut model, &mut opt, &mut warm, &mut pool);
    }
    profiling::reset(); // drop warmup's contribution to the sub-block breakdown

    let mut t = [0.0f64; 3];
    let wall = Instant::now();
    for _ in 0..steps {
        run_step(&mut model, &mut opt, &mut t, &mut pool);
    }
    let wall = wall.elapsed().as_secs_f64();

    let n = steps as f64;
    let [fwd, bwd, optt] = t;
    let total = fwd + bwd + optt;
    eprintln!("--- ms/step (mean over {steps} steps) ---");
    eprintln!("forward   {:8.2}  ({:4.1}%)", fwd / n * 1e3, fwd / total * 100.0);
    if !fwd_only {
        eprintln!("backward  {:8.2}  ({:4.1}%)", bwd / n * 1e3, bwd / total * 100.0);
        eprintln!("optimizer {:8.2}  ({:4.1}%)", optt / n * 1e3, optt / total * 100.0);
    }
    eprintln!("total     {:8.2}", total / n * 1e3);
    eprintln!("wall      {:8.2}  ({:.1} steps/s)", wall / n * 1e3, n / wall);
    profiling::report();
}
