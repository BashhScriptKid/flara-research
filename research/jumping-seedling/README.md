# Fydel — Jumping Seedling

A 1B-parameter transformer designed to be trained *and* run entirely on a CPU, hand-written in Rust with custom kernels. The target machine is an AMD Ryzen 5 5500U (6 Zen 2 cores, 8MB L3, AVX2 + F16C, no AVX-512, no native BF16) — a consumer laptop CPU, not a datacenter part.

## Thesis

The default assumption in 2026 is that serious transformer work requires a GPU, because the binding constraint is assumed to be floating-point throughput. On a CPU that's usually not the actual bottleneck — memory bandwidth is. CPUs have latent advantages GPUs lack: deep out-of-order execution over independent dependency chains, a real cache hierarchy, and cheap divergent control flow. This project is a from-scratch attempt to design a transformer architecture and kernel set around those constraints instead of fighting them.

## Layout

- `src/model/`, `src/kernels/`, `src/train/` — the architecture, hand-written kernels, and training loop
- `src/bin/` — training and benchmarking entry points (`train.rs` is the main binary; `profile.rs`, `layer_bench.rs`, etc. are diagnostic binaries)
- `RESEARCH_LOG.md` — append-only running journal: why each design decision was made, what was validated, what was deliberately deferred
- `OPTIMIZATION_NOTES.md` — kernel-level performance notes
- `HANDOFF.md` — cross-session handoff notes (this project is worked on across multiple AI coding sessions; append-only, don't overwrite)

## Running it

```bash
cargo run --release --bin train
```

See `src/bin/` for other available binaries (profiling, kernel probes, benchmarks).
