# Handoff — What happened while you were frozen (2026-06-27)

Claude hit the weekly limit mid-operation. Another model picked up the profiling
work and completed the analysis below. This is what changed and what was decided
in your absence.

---

## Process note for non-Claude models

If you are NOT Claude: **append a new section at the end of this file** after
every change you make to the kernel (code, benchmarks, analysis — regardless of
scale). Timestamp it. Claude is the de facto lab assistant/executor on this
project and needs a paper trail to reorient when it resumes. Do NOT overwrite
existing sections. Append only.

---

## What ran

Full 1B profile test completed:

```
FULL=1 SEQ=256 cargo run --release --bin profile

  forward    7481.60 ms  (23.6%)
  backward  23653.67 ms  (74.6%)
  optimizer   584.82 ms  ( 1.8%)
  total     31720.10 ms
  wall      31780.35 ms  (0.0 steps/s)
```

## What was diagnosed

Backward is 3.16× forward. The analysis traced it to a structural mismatch:
your kernels are decode-shaped (single-token), but the profile runs
training/prefill (256 tokens). Specifics:

1. **No token batching in BasisMatmul** — every token triggers independent
   forward/backward. Lambdas (token-independent) recomputed 256× per layer.
   Per-token gradient buffers allocated and zeroed 1,440 times per step.
   Call sites: `layer.rs:326,354,367` (forward), `layer.rs:440,478,510` (backward).

2. **rayon completely unused** — in Cargo.toml, never imported. Everything serial.

3. **Attention uses scalar dot** — `attn_flash.rs:37` is a plain iterator sum,
   not `gemm::dot`. ~21.6M scalar dots across full-attention layers.

## What was decided

Implementation in three phases:

- **Phase 1** (quick wins): AVX2 dot in attention, rayon over token loops
- **Phase 2** (structural): `forward_seq` / `backward_seq_accum` batch kernels
  in `fft.rs`, wired into `layer.rs`
- **Phase 3** (later): FFN block-route grouping

Start with Phase 1a+1b, then Phase 2. Do not touch Phase 3 until Phase 2 is
stable.

## What was written

`HANDOFF.md` — detailed analysis with file:line references, code path
summaries, build commands. Read it before resuming work. It has everything
you need to pick up where you left off.

---

## Phase 1 attempt — REVERTED (2026-06-28)

Attempted both Phase 1a (gemm::dot in attention) and Phase 1b (rayon over
token loops). Results on FULL=1B SEQ=256 STEPS=3:

```
forward   11497.97 ms  (was 7481 — +54% SLOWER)
backward  33167.98 ms  (was 23653 — +40% SLOWER)
total     45259.35 ms  (was 31720 — +43% SLOWER)
```

**Why it failed:** Token-level parallelism is counterproductive at this scale.

1. **L3 cache thrashing** — all 6 cores hammer the shared dictionary `G`
   simultaneously. Serial execution keeps `G` in L3; parallel execution
   evicts it across cores. This is the dominant regression cause.

2. **Sync overhead** — 96 layers × 4 `par_iter` calls = 384 fork-join
   barriers per step. Per-token BasisMatmul work is too cheap to amortize.

3. **gemm::dot at head_dim=64** — 64 elements is too short for AVX2 to
   overcome dispatch overhead. Neutral at best.

**Conclusion:** Phase 1 is the wrong approach. The kernels are too cheap
per-token for token-level parallelism to help. The real fix is **Phase 2
batch kernels** — reuse lambdas across tokens, which makes each unit of
work heavier and reduces redundant `G` fetches, making parallelism
worthwhile.

**Current state:** All code reverted to baseline. 76/76 tests green.

---

## Serial performance audit (2026-06-28)

Three subagents investigated: memory access patterns, heap allocation
profiling, and cache efficiency. The bottleneck is NOT parallelism — it's
serial overhead that makes each step take 31.7s when arithmetic says ~0.4s.

### The math says it should be fast

BasisMatmul is **strongly compute-bound**: arithmetic intensity ~39-61
flops/byte vs Zen 2 ridge point 1.24. Total forward FLOPs for 96 layers ×
256 tokens ≈ 21 GFLOPs. At peak 56 GFLOPS, that's 0.37s. Actual forward:
7.5s. Something is 20× off.

### Killer #1: 24.4 million heap allocations per step

This is the dominant bottleneck. Every BasisMatmul call allocates fresh
Vecs on the heap. Every FFN backward inner loop allocates 512-byte scratch
buffers. Total:

| Source | Allocs/step | Fix |
|--------|-------------|-----|
| FFN backward inner loops (`psign`/`psa`/`psb` in `accum_*_block_grads`, `fft.rs:194,232-233`) | ~12.2M | Hoist buffers out of inner loops, pass as `&mut [f32]` |
| BasisMatmul forward fresh Vecs (`xq`, `lambda`, `acc` — `fft.rs:271-281`) | ~1.8M | Pre-allocate scratch on BasisMatmul struct |
| BasisMatmul backward fresh Vecs (`xq`, `ap`, `cq`, `d_coeffs`, `d_dict`, etc. — `fft.rs:522-548`) | ~11.2M | Same arena/scratch approach |
| Layer output clones (`lf.out.clone()` `model.rs:217`, `d_x = lg.d_hidden.clone()` `model.rs:387`) | 192 × 230K f32 = 90 MB memcpy | Double-buffer, pass ownership without clone |

Specific hotspots in `fft.rs`:
- `accum_block_grads` (line 184-213): allocates `psign = vec![0.0; 2*b]`
  inside a P × |active_q| loop = 168 allocs per `backward_cols` call
- `accum_pair_block_grads` (line 219-253): allocates `psa` + `psb` inside
  |active_p| × Q loop = 336 allocs per `backward_rows_pair` call

### Killer #2: `logits_from_embed` streams 112 MB per token

At `gemm.rs:55-88`: for each of 256 tokens, the vocab projection reads
the entire 32,768×896 embed matrix (112 MB). Total: **28 GB of DRAM
reads** per step. At ~40 GB/s bandwidth: ~700ms minimum for forward alone.
Backward does it again.

### Killer #3: rustfft is entirely scalar

Every FFT processes one Complex32 at a time. No SIMD. The FFTs consume
~30% of BasisMatmul compute but get zero vectorization benefit. 96 layers
× 256 tokens × 7 projections × ~392 FFT calls per token = ~67M FFT calls
per step, all scalar.

### Killer #4: FFN wastes memory on sparse activation

`ffn.rs:177` allocates `act = vec![0.0; F]` (3,072 floats = 12 KB) per
token, but only n_active×b = 12×64 = 768 elements are non-zero. 75% is
zero-filled and never touched.

### What is NOT the bottleneck

- **Dictionary `G` layout** — `(K, b)` is correct, 16 KB fits L2
- **Coefficient access** — sequential, L1-resident (21 KB working set)
- **Flash attention tiling** — kv_block=64 keeps working set in L1/L2
- **RMSNorm** — allocation-free, well-optimized
- **`prefetch_coeffs`** — real x86 prefetch into L1, working correctly
- **Attention K/V access** — strided (512B stride) but tiled, L2-resident

### Why Phase 1 parallelism regressed

The 43% regression was NOT primarily L3 cache thrashing on `G`. `G` is
only 16 KB and read-only (MESI Shared state — no invalidation). The real
causes:

1. **`d_dict` false-sharing** — every token's backward calls
   `add_c(&mut g.d_dict, &fg.d_dict)` at `layer.rs:452,481,512-516`.
   In parallel, T cores do read-modify-write on the same 16 KB buffer.
2. **Per-layer working set overflow** — dict (16 KB) + attn coeffs
   (4×25 KB) + FFN coeffs (3×86 KB) + router (172 KB) = ~547 KB/layer.
   In serial, fits L3 (8 MB). In parallel, 6 cores' L2s (3 MB) can't
   hold it.
3. **FFT scratch competition** — 6 cores × ~1.2M FFT calls/step competing
   for L1D.
4. **Attention `dk/dv` coupling** — `attn_flash.rs:206-212` accumulates
   across query positions. Inherently cross-token, cannot be parallelized.

### Updated implementation plan

The old 3-phase plan is obsolete. New plan, in priority order:

**Phase A — Eliminate allocation churn (target: -50% wall time)**

1. Add a `Scratch` struct to `BasisMatmul` with pre-allocated buffers:
   `xq: Vec<Vec<Complex32>>`, `lambda: Vec<Complex32>`, `acc: Vec<Complex32>`,
   `d_coeffs: Vec<f32>`, `d_dict: Vec<Complex32>`, etc. Reuse across calls.
2. Hoist `psign`/`psa`/`psb` out of `accum_block_grads` and
   `accum_pair_block_grads` — pass as `&mut [f32]` parameters. Caller
   owns the buffer.
3. Eliminate `lf.out.clone()` in `model.rs:217` — pass `out` directly as
   next layer's input via double-buffering or `std::mem::take`.
4. Same for `d_x = lg.d_hidden.clone()` in `model.rs:387`.

**Phase B — Sparse FFN activation (target: -15% FFN memory)**

5. Replace `act = vec![0.0; F]` with a sparse representation that only
   stores the n_active×b non-zero elements. Adjust `forward_cols` to
   operate on the sparse buffer.

**Phase C — Logits kernel optimization (target: -700ms forward)**

6. Tile the vocab dimension in `logits_from_embed` to improve L2 temporal
   reuse. Or: reduce vocab projection to a learned low-rank map if
   accuracy permits.

**Phase D — SIMD FFT (long-term, target: -30% BasisMatmul)**

7. Replace scalar rustfft with SIMD-optimized FFT for b=64. This is a
   significant undertaking but addresses the 30% of BasisMatmul time that
   is currently scalar.

**Phase E — Batch kernels (revised from old Phase 2)**

8. `forward_seq` / `backward_seq_accum` batch kernels that process all
   T tokens in one call. This enables lambda reuse (token-independent),
   eliminates per-token allocator churn (already fixed in Phase A), and
   makes the work heavy enough for parallelism to pay off.
9. After Phase E, revisit rayon — the per-unit work will be heavy enough
   that fork-join overhead is amortized.

### Key file references

| File | Lines | What |
|------|-------|------|
| `fft.rs:163-177` | `block_eigs` — dict access pattern |
| `fft.rs:263-305` | `BasisMatmul::forward` — per-token hot loop |
| `fft.rs:506-599` | `BasisMatmul::backward` — per-token backward |
| `fft.rs:184-213` | `accum_block_grads` — #1 alloc hotspot |
| `fft.rs:219-253` | `accum_pair_block_grads` — #2 alloc hotspot |
| `fft.rs:755-811` | `forward_rows_pair` — FFN up+gate |
| `fft.rs:1070-1114` | `forward_cols` — FFN down |
| `layer.rs:306-394` | `TransformerLayer::forward` |
| `layer.rs:401-553` | `TransformerLayer::backward` |
| `layer.rs:452,481,512-516` | `d_dict` accumulation (false-sharing point) |
| `model.rs:217` | `lf.out.clone()` — unnecessary memcpy |
| `model.rs:387` | `d_x = lg.d_hidden.clone()` — unnecessary memcpy |
| `gemm.rs:55-88` | `logits_from_embed` — 112 MB/token bandwidth monster |
| `ffn.rs:177` | Sparse activation waste |
| `attn_flash.rs:206-212` | `dk/dv` cross-token coupling |
| `attn_flash.rs:70-136` | FlashAttention forward — kv_block=64 tiling |

### Build and test

```bash
# Profile
FULL=1 SEQ=256 cargo run --release --bin profile

# Tests (must all pass after any change)
cargo test --release
```
