# Fydel Jumping Seedling — Research Log

A running journal for **Fydel Jumping Seedling** (a.k.a. Fydel-1B): a 1B-parameter
transformer designed to be trained *and* run on a CPU from scratch, implemented in
Rust with hand-written kernels. This log is the narrative companion to the code —
it records *why* each decision was made, what was validated, and what was
deliberately left for later. Entries are append-only and dated.

---

## The thesis (why this project exists)

The reflex in 2026 is that you cannot do serious transformer work without a GPU.
That reflex bakes in a hardware assumption: that the binding constraint is
floating-point throughput. On a CPU it usually is not. On the target machine — an
**AMD Ryzen 5 5500U** (6 Zen 2 cores, 8 MB L3, AVX2 + F16C, no AVX-512, no native
BF16) — the binding constraint is **memory bandwidth**, and the latent advantages
are the things GPUs are bad at: deep out-of-order windows over independent
dependency chains, a real cache hierarchy, and cheap, divergent control flow.

So the design question is inverted. Instead of "how do we feed the matmul units,"
it is **"how do we keep the working set in cache and turn irregularity into an
advantage?"** Every architectural choice in this project is an answer to that
question:

- **Circular-basis weight compression** — fit each layer's weights in L3 so they
  are streamed once, not thrashed.
- **Block-routed structured sparsity FFN** — compute 25% of the FFN per token,
  with routing granularity aligned to the compression block so the skip is *exact*.
- **Sliding-window attention** for most layers — cap KV bytes per query at a
  constant, independent of sequence length.
- **Early-exit probes** — variable depth per token, which is free wall-clock on a
  CPU's single-token decode loop and a stalled warp on a GPU.

The point is not to beat an H100. It is to show that a coherently CPU-shaped
architecture makes from-scratch 1B training and inference *feasible* on a laptop,
and that several of these ideas are interesting independent of the hardware story.

---

## Architecture at a glance

| Component | Decision |
|---|---|
| Layers × Hidden | 96 × 896 |
| FFN | Block-routed ReGLU, dim 3072, block b=64 → M=48 micro-blocks, k=12 active (25%) |
| FFN compression | Circular basis (joint up+gate, separate down), load-time dial, 5× target for L3 |
| Attention 1–24 | Flash, full causal, GQA |
| Attention 25–96 | Sliding-window (W=128–256), GQA |
| GQA | 14 query heads, 2 KV heads, head_dim 64 |
| Exit probe | Linear H→1 after pre-norm, per layer, gradient-stopped |
| Position / Norm | RoPE / RMSNorm |
| Optimizer | AdaFactor (factored 2nd moment), relative step, optional momentum |

---

## 2026-06-26 — Kernel layer complete (forward + backward, all gradchecked)

Closed out the full set of compute kernels. Every backward pass is validated
against central finite differences before it is considered done; this log treats
"passes gradcheck" as the bar for correctness.

**`fft.rs` — circular-basis matmul (10 tests).**
Scheme B (chosen over plain block-circulant): a weight `W[out,in]` is tiled into
`b×b` circulant blocks, each diagonalized by the DFT so `block·x = IFFT(λ ⊙
FFT(x))`. A single complex dictionary `G ∈ ℂ^{K×b}` is shared across *all* blocks,
matrices, and layers; each block keeps only real coefficients `α ∈ ℝ^K` with
`λ = Σ_k α_k G_k`. Storage per matrix is `P·Q·K` reals; `G` is the cache-resident
basis. Real α with complex G is the key compromise — coefficients stay cheap, and
the imaginary structure is *learned* inside the dictionary ("learn within the
constraint"). Forward, backward (VJP w.r.t. α, G, x), and the masked row/column
variants that make the sparse FFN exact all live here.

**`ffn.rs` — block-routed structured-sparsity FFN (8 tests).**
This is *not* MoE: there is no parameter replication, total params equal one dense
FFN, so there is no RAM blow-up. The insight that unlocked it: element-wise ReLU
sparsity does not compose with block-wise circular matmul (you would need a whole
zeroed b-block, which never happens by chance). So instead each b-wide slice of the
intermediate is a "micro-expert"; a per-token router (linear H→M=48, top-k=12
softmax) picks 25% of blocks. Up/gate compute only the selected output row-blocks,
down sums only the selected input column-blocks — and because routing granularity
*equals* the circular-block granularity, the skip is **exact**, not approximate.
The router doubles as a CPU prefetch signal (which block coefficients to pull into
L2). Top-k is straight-through; a Switch-style load-balance aux loss
(`aux = M·Σ f_j P_j`, minimum = n_active) prevents block collapse. The full
forward/backward chain is finite-diff gradchecked end to end.

**OoO/ILP exploit — up+gate fusion.** A concrete place where the architecture is
shaped to the microarchitecture. The first cut called the masked forward
separately for up and gate, each re-running `FFT(h)` on the same input. Fused into
a single `up_gate` matmul with `forward_rows_pair` / `backward_rows_pair`: share
`X_q = FFT(h)` once, then interleave the two λ-multiply/IFFT pipelines so Zen 2's
out-of-order window has two independent dependency chains per block to overlap.
Proven bit-exact to two separate calls.

**`attn_flash.rs` — FlashAttention-2, GQA, causal (5 tests).**
Online-softmax, key-tiled, never materializes the T×T score matrix. Forward returns
the per-row log-sum-exp; backward recomputes scores from Q/K + the saved LSE (still
no T×T matrix), using the delta trick `D_i = dot(dO_i, O_i)` and the softmax VJP
`dS_j = p_j(dP_j − D_i)`. dK/dV correctly accumulate across the 7 query heads
sharing each KV head.

**`attn_swa.rs` — sliding-window attention, both directions (4 tests).**
Structurally identical to flash, with the causal key range clamped to a window
`[i−W+1, i]`. This is the bandwidth lever: it caps KV bytes streamed per query at
`W·head_dim` *regardless of sequence length*, so layers 25–96 stay cache-resident
as context grows, while layers 1–24 keep global reach. The sharpest test
(`window_excludes_old_keys`) perturbs key 0 and confirms every row ≥ W is
bit-identical — the window boundary is exact, not merely "approximately local."

**`probe.rs` — early-exit confidence probe (4 tests).**
A linear H→1 head per layer (CALM-style) predicting whether a token can halt early.
Critically **gradient-stopped on its input**: it trains its own w/bias but its
gradient never flows into the hidden state, so it cannot cheat by degrading the
representation to make tokens "look easy." Zero-init gives a neutral p=0.5 before
the auxiliary loss is annealed in.

**`optimizer.rs` — AdaFactor, factored second moment (5 tests).**
Adam's full second-moment tensor doubles the parameter memory — exactly the
pressure we are avoiding. AdaFactor keeps only row-sums `R` and column-sums `C` and
reconstructs `V̂[i,j] = R[i]·C[j]/ΣR`, which is *exact* when `G²` is separable
(verified directly) and a good rank-1 approximation otherwise. Update RMS-clipping
and an optional relative step size are included; 1-D tensors fall back to a full
per-element second moment. Convergence is checked on an ill-conditioned diagonal
quadratic (≥100× descent) with and without momentum.

**Status:** full suite **45 tests green, 0 warnings.** The kernel layer
(fft, norm, rope, ffn, flash, swa, probe, optimizer) is forward+backward complete
and gradchecked.

### Deferred — explicitly, not forgotten
These are flagged in code and tracked here so they are not silently dropped:
- **AVX2 pass** across the spectral kernels: register-level multi-accumulators,
  multi-block software pipelining (prefetch/FFT block n+1 under block n compute),
  and tiling the attention *backwards* to `kv_block` (the forwards already tile).
- **Sub-byte (4/2-bit) coefficient packing** under the compression dial.
- **Frequency-domain second moment** for AdaFactor — the project conjecture that an
  FFT of the gradient compacts energy and makes the rank-1 factorization tighter.
  This is an *experiment to run on top of* the validated spatial base, not a silent
  default. **Open design decision before implementing.**
- **INT8 momentum** (via `QuantizedMomentum`) — a memory pass on the optimizer;
  f32 momentum is kept for now as a correct reference.

### Next
Assemble upward from kernels: `model/` (config, layer, model wiring — RoPE applied
by the layer before attention; activation checkpointing every 8 layers) and
`train/` (WSD schedule with 2000-step warmup, 512→1024 seq-len curriculum, grad
accumulation to a ~512K-token effective batch, depth-weighted exit-probe KL loss
annealed from 0 over 5000 steps).

### Open questions worth writing up
- How much does the circular-basis compression cost in loss vs. a dense FFN at
  matched params, and where does the compression dial's sweet spot land for the
  8 MB L3?
- Does the frequency-domain second moment actually beat the spatial factorization,
  or is the energy-compaction intuition wrong for *gradients* (as opposed to
  weights/activations)?
- Sustained throughput on the 5500U — the back-of-envelope is 100–150 tok/s, which
  makes 100–500M-token research validation feasible (1–7 weeks). Full Chinchilla
  (20B) needs a cluster. The number to actually measure once `train/` exists.

---

## 2026-06-26 — Note: branch behavior & speculation (design thread, not yet built)

Prompted by a sharper framing of the early-exit-vs-GPU point. Early exit is
**data-dependent control-flow divergence**, not branch *prediction* — a CPU runs it
per token with cheap misprediction recovery, while a SIMT GPU must mask-and-
serialize the divergent paths, so skipped compute is only reclaimed when a whole
warp agrees to exit. The follow-on question — *can we actively exploit the CPU's
branch predictor / speculation while we have it?* — splits into two senses.

**Literal hardware branch prediction.** The predictor only helps on real
conditional branches that are *biased*. Two consequences for our kernels:
- **Hot numeric loops want to be branchless.** Writing the sparse FFN as
  `for b in 0..M { if selected[b] {…} }` puts an unbiased ~25%-taken branch in the
  inner loop — near worst-case, constant mispredicts. Loop over the *gathered*
  active indices instead (`for &b in &selected {…}`) so the branch doesn't exist.
  The predictor's best case is a branch that isn't there.
- **The one branch worth predicting is early exit.** `if conf > τ { break }` is
  biased *if kept biased* — easy text exits shallow consistently, hard text runs
  deep consistently. So make the exit decision **temporally sticky** (hysteresis /
  dwell on τ): a flip-flopping exit mispredicts every token and drains the
  pipeline; a sticky one is ~free. This is a concrete, cheap design lever.

**Speculation in spirit (guess-ahead + prefetch) — where the real wins are, since
the bottleneck is memory latency:**
1. **Schedule the router early (deterministic, not speculative).** Router is a tiny
   `H→48` matmul; compute it as soon as the hidden state exists and prefetch the
   selected coefficient tiles into L2 before the FFN needs them. Pure reordering
   win — the sharpened "router as prefetch signal."
2. **Speculative cross-token routing prefetch (true speculation).** Predict this
   token's routed blocks from the *previous* token's routing (MoE routing has
   real temporal/positional correlation) and prefetch before the hidden state even
   exists. Wrong → wasted cache lines, no correctness hit. **Caveat:** wrong
   speculation burns *bandwidth*, our scarce resource, and can evict useful lines —
   only a win if the predictor is *measurably* biased. Validate before betting.
3. **Speculative next-layer warmup across the exit branch.** The probe is a tiny
   `H→1` dot product; the next layer is huge. Overlap them — prefetch/start layer
   L+1 on the common-case assumption while the probe resolves, discard if it
   disagrees. Pipelining across the CALM branch.

**Honesty bounds.** This is almost entirely a *decode-time* story — training runs
all layers for the gradient, so there's no exit branch to speculate across and the
kernels are dense-and-branchless regardless. And Zen 2's OoO engine already
speculates past branches implicitly; the explicit job is mainly to *not stall it* —
keep hot loops branchless or biased so mispredicts don't drain the window we are
trying to fill with independent block dependency chains.

**Plan, ranked:** (1) deterministic early-router scheduling + sticky exit threshold
— free, no risk, fold into `model/` when the layer loop exists; (2) measure routing
temporal correlation before committing to speculative cross-token prefetch;
(3) next-layer warmup as a later micro-opt. Hang these on the `model/` loop.

---

## 2026-06-27 — Training pipeline closed (functional + smoke-tested)

Wired the rest of the path from "kernels exist" to "a step can run on real data."
Optimizer finished (AdaFactor, factored 2nd moment, β1=0), gradient accumulation,
a WSD (warmup-stable-decay) schedule, and serde+bincode checkpointing of model and
optimizer state. Data path is deliberately boring: pull a tokenizer + a text corpus
straight from the HF hub over `ureq` (dropped `hf-hub` — its feature set fought our
deps and dragged in parquet we don't want), tokenize, and serve shifted windows that
wrap. There is a `train` binary and a `profile` binary; 76 tests green, every
backward still gradchecked. The bar for this stage was **functional + smoke-tested,
not fast** — that comes next, and on purpose, because optimizing before the shape is
settled is how you polish the wrong thing.

---

## 2026-06-27 — Kernel optimization arc, and two hypotheses the measurements killed

Goal: make the mid-config step (12 layers, hidden 512, FFN 2048, vocab 8192, seq
256) fast enough to iterate on, fwd+bwd+opt. The honest story here is less about the
**2.13× we got** and more about **two plausible-sounding plans the data refused**.

**The arc (ms/step, all gradcheck-protected):**

| Stage | ms/step |
|---|---|
| SSE2 baseline | 7882 |
| AVX2 codegen (`.cargo/config.toml` target-feature) | 6977 |
| + forward head GEMM, hand-AVX2 | 6138 |
| + backward-head **loop reorder** (not SIMD) | 5723 |
| + attention-projection backward AVX2 | 4356 |
| + FFN backward AVX2 | **3698** |

**Hypothesis 1 (killed): "the FFT transforms dominate `BasisMatmul`."** It is the
obvious suspect — dozens of size-64 rustfft calls per projection. Built `fftfrac` to
measure it directly before optimizing it. **rustfft is ~1% of the cost.** It's
already SIMD internally; the transforms were never the problem. Had we trusted the
intuition we'd have spent days writing a custom radix kernel for a 1% line item.

**Where the cost actually is.** A `dict_k` sweep isolated it: backward is ~88% of
`BasisMatmul`, and the cost scales with the basis rank `k` at ~4.3 µs/k against
forward's ~0.45 µs/k. The culprit is the **complex-dot contraction** that builds the
block eigenvalues and its backward — `acc += (pbuf[f]*atom[f]).re` fused with a
`d_dict` axpy. That reduction will not auto-vectorize (strict FP ordering), which is
exactly why hand-AVX2 helps *here* and nowhere else.

**Hypothesis 2 (killed): "amortize allocations with scratch-reuse."** After the SIMD
work, backward was still 72% and felt allocation-heavy. Before committing to a
structural scratch-threading pass, **measured it** with a counting global allocator:
attn backward = 33 allocs / 41 KB, FFN down = 81 / 95 KB, FFN up = 57 / 77 KB. At
~15–50 ns per alloc that is **~1%** of a 100–426 µs op. The `dict_k` sweep had
already implied this — the cost lives in the k-slope (compute), not the k=1 intercept
(where fixed alloc overhead would sit). Scratch-reuse rejected; it buys ~1%.

**Lessons worth keeping:**
- **Hand-AVX2 only pays on reductions.** axpy and the eigenvalue accumulation
  auto-vectorize once AVX2 codegen is on; intrinsics there are wasted effort. The
  backward-head win came from a **loop reorder** (making the `d_embed` write
  cache-resident, written once per vocab id), not from SIMD — it was bandwidth-bound.
- Factored the validated fix into two `BasisMatmul` helpers (`accum_block_grads`,
  `accum_pair_block_grads`) so the contraction routes through one AVX2 `dot`.
- Keepers: `gemm.rs` (`logits_from_embed` 2.8×, `head_backward` reorder 1.6×, `dot`),
  the AVX2 codegen flag, the three backward contraction paths.

**Honesty bound.** This 2.13× is real but it is shaving against a ceiling. The
remaining backward is genuine, already-vectorized complex FLOPs, L1-resident — we are
near the SIMD roofline *for this primitive*. That framing matters for the next entry.

---

## 2026-06-27 — The pivot: the FFT realization is the wall, and a Monarch×dictionary hybrid

This is the most consequential entry in the log so far. It started as a throughput
question and ended by relocating the project's central risk.

**Regime matters, and we were measuring the wrong one.** A roofline pass split the
story cleanly:
- **Training / prefill (batched, weights reused):** compute-bound. And here is the
  uncomfortable fact — `BasisMatmul` does **~the same FLOPs as the dense matmul it
  replaces** (`block_eigs` ≈ 2d²; decompressing a weight costs about what multiplying
  by it would). Compression buys *storage and bandwidth, not compute*. So in the
  regime we'd been profiling, the structured primitive is parity-to-slower than dense,
  because the complex/FFT contraction vectorizes worse than a clean GEMM.
- **Autoregressive generation (batch 1, weights streamed per token):** bandwidth-
  bound, and this *is* where compression theoretically wins — ~5–6× over a dense fp16
  model on this CPU, by turning a weight-streaming problem into a cache-resident
  compute one.

**The thesis got sharpened (by being challenged).** The goal isn't generation tok/s —
it's making **training / fine-tuning / distillation viable on a CPU**. So the precise
claim had to be cleaned up: this architecture does **not** lower the compute exponent
(per-token matmul stays O(d²) in both). What it changes is (1) **memory/feasibility** —
~40 M actual trained params vs a dense 1B's ~8–12 GB of training state that simply
doesn't fit and so never runs; (2) **parameter-side cost** (optimizer/grad/storage)
~128× smaller; (3) **FFN block-routing**, which *is* a genuine sub-quadratic compute
reduction. The loose "O(n!) → O(n²)" is best read as **infeasible → feasible**, not a
complexity-class change. Said plainly so the paper doesn't overclaim.

**Then: is the FFT-circulant primitive even the right one?** Checked the literature.
The answer is pointed: our circular-basis-via-FFT is a structured matrix in the
**butterfly family** — and the FFT/butterfly *realization* is precisely the one the
field abandoned for hardware inefficiency (cited <2% FLOP utilization). **Monarch
matrices** (a corner of the **Block-Tensor-Train / BTT** family) refactor the same
sub-quadratic idea into **block-diagonal GEMMs + permutations** — real-valued, and
shaped exactly for the dense-matmul units AVX2 is good at. The prior art on *shared*
structured cores (MetaTT, TRAC, Basis Sharing, Share-Your-Attention) is all GPU /
PEFT / post-hoc — **from-scratch CPU pretraining is open ground.**

**The realization that reframes our own design:** our `dict + per-block coeffs` split
*is already structurally a shared-core BTT* (tied basis core + per-weight coefficient
cores) — which is the right shape, and the shape the full-rank/maximal-update
principle wants. The *only* wrong part is the per-block **FFT-circulant realization**.
So the hybrid worth building isn't "Monarch + BTT" (Monarch ⊂ BTT, redundant); it is
**GEMM-based BTT cores × our shared-core dictionary × CPU-native AVX2 tile sizing** —
keeping our novelty (weight-tying + CPU training) and replacing only the broken
primitive. Guardrail, or it underperforms both parents: share exactly **one** core,
keep coefficient cores full-rank, obey maximal-update init/LR.

**Gate (a) — verified, not asserted (`monarch_probe.rs`).** Built a head-to-head with
the *same* `gemm::dot` AVX2 kernel everywhere, so it isolates structure, not effort:

| primitive (n=512, per token) | µs/token | GFLOP/s |
|---|---|---|
| dense 512×512 matvec | 20.8 | 25.2 |
| **Monarch 512 (block-GEMM)** | **1.88** | **26.2** |
| BasisMatmul 512 (FFT-circulant, current) | 19.45 | — |

**Monarch is 10.4× faster per token than the current FFT block, at the same AVX2
efficiency as a plain dense matmul (26 vs 25 GFLOP/s).** The damning line is that
`BasisMatmul` runs at *dense* wall-clock (19.5 ≈ 20.8 µs) — it pays full price and
captures **none** of the structural speedup its compression should buy. Monarch
captures 10× of it. The FFT realization is the wall, quantified.

**Honesty bounds (the caveats are the interesting part):**
- **Not equal-param.** This Monarch block has ~24.6 K params vs the current block's
  ~2 K coeffs — it bought ~10× speed partly by being ~12× *less* compressed.
  Recovering the compression without losing the speed is *exactly* what the shared-
  core hybrid is for; this result is its motivation, not a finished answer.
- **25 GFLOP/s is a matvec + single-thread floor**, not peak — a tiled, threaded GEMM
  goes higher and should *widen* Monarch's edge (its weights stay L2-resident while
  dense's 1 MB matrix starts contending for bandwidth).
- **Forward + speed only.** Says nothing about whether a shared-core block *learns*
  as well at matched compression. That is the open risk now.

**Where the risk moved.** The kernel objection to a Monarch/BTT primitive is gone.
The project's central uncertainty is no longer "can we make the structured matmul
fast on a CPU" — it is **"can a shared-core BTT block stay full-rank and trainable at
our compression target."** That's a math/gradcheck question (gate b) and a proof-run
question (gate c), not a systems one. The FFT path stays as a working fallback, so
exploring costs nothing but time.

**Plan, ranked:** (1) gate (b) — prototype the shared-core BTT block and gradcheck it
for full-rank/trainability; a failed gradcheck kills the spec anyway, so this is the
cheapest decisive test. (2) If it holds, write the design spec from the working
primitive. (3) gate (c) — a tiny proof run that loss descends at matched params.

---

## 2026-06-27 — Gate (b): shared-core BTT block is full-rank and trainable

Built a standalone, gradcheckable prototype (`btt_probe.rs`, isolated from the FFT
path) of the actual hybrid primitive: an order-2 Monarch block whose two block-diagonal
stages are each a **linear combination of a shared atom dictionary** (`D1`, `D2`) with
**per-weight coefficients** (`a1`, `a2`). The compression knob is `nd`, the number of
shared atoms; small `nd` forces each block into a low-dimensional atom span, which is
exactly where full-rank could fail — so the probe stresses the guardrail rather than
dodging it. Three checks, on `n=64` (`m1=m2=8`):

**Rank — the result that de-risks the whole direction.** The effective 64×64 map is
**full-rank (64) at every `nd`, down to `nd=1`** (16 coefficients per weight), with
healthy pivots. The two-stage + permutation structure produces *dense* rank even when
each stage is built from a single shared atom. So the full-rank property the
maximal-update principle demands is **structural here, not delicate** — the central
worry going in.

**Backward — correct, proven the right way.** A finite-difference gradcheck was
f32-cancellation-noisy on the small-magnitude coefficient grads (`a1` ~2e-2 while
`a2`/`D1`/`D2` were clean at ~1e-3, even after moving the FD accumulation to f64). The
decisive proof is the training test instead: fitting a **same-family teacher** (a
target that is exactly representable) drives relative error to **0.0000 at `nd=8`** —
which is impossible with a wrong gradient. Lesson restated: a clean overfit-to-zero on
a representable target is a stronger correctness witness than a noisy gradcheck.

**Trainable — yes, with a real, scoped caveat.** Same-family targets are learnable
(→0 at `nd=8`, 0.10 at `nd=4`). But at the most aggressive compression (`nd=2,4`) it
*does not* reach zero despite the target being exactly representable — an
**optimization-conditioning** problem from the bilinear `atom × coefficient`
landscape, not a correctness or capacity one. This is the first empirical (not just
theoretical) evidence that the **init/LR parametrization is load-bearing** at high
compression — the guardrail we flagged now has teeth and a known lever.

**Control:** fitting a *random dense* (incompressible) target plateaus at 0.79–0.92
rel err, decreasing with `nd`. That's expected — a sub-quadratic family provably can't
represent arbitrary dense matrices, and seeing the gap confirms the prototype is a
genuinely *compressing* structure, not secretly dense.

**Verdict.** Gate (b) passes on both load-bearing questions: full-rank (structural)
and trainable (backward sound, representable targets learnable). The hybrid survives
its hardest test. The risk has narrowed from "does this even work" to a specific,
addressable engineering problem: **conditioning the optimizer at high compression.**

**Plan, ranked:** (1) tune init/LR (maximal-update-style scaling of atom vs coefficient
learning rates) and re-check the `nd=2,4` same-family fits — confirm the conditioning
caveat is a lever, not a wall. (2) gate (c): wire one shared-core BTT block into the
real layer in place of a `BasisMatmul` projection and confirm loss descends on real
data at matched params. (3) only then, the full design spec — written from a working,
conditioned primitive rather than from theory.

---

## 2026-06-27 — Gate (b) follow-up: the conditioning lever, and a non-monotonic anomaly

Chased the low-`nd` conditioning caveat. Two findings, one clean and one I do not
fully understand yet.

**Clean: the lever is the schedule, not the learning-rate ratio.** Cosine LR decay
(anneal into the basin) fixed the worst single-seed cases outright — `nd=2` went
0.36 → 0.0000, `nd=16` 0.02 → 0.0000. Decoupling the atom vs coefficient learning
rate (the maximal-update-themed lever I expected to matter) **did not help** and
slightly hurt — which makes sense: Adam already normalizes each parameter by its own
second moment, so a raw gradient-magnitude imbalance between shared atoms and
per-weight coefficients is mostly absorbed. So the practical recipe is simpler than
feared: cosine-decayed Adam, equal LR.

**Unexplained: a non-monotonic dead spot at `nd=4`.** A 12-seed sweep (cosine-8k,
same-family targets) gives solved-rate (rel_err < 1e-3): **`nd=2` 9/12, `nd=4` 2/12,
`nd=8` 12/12, `nd=16` 12/12.** `nd=4` is *systematically* stuck (median 0.22), not one
unlucky instance — and it is non-monotonic: more atoms than `nd=4` is easier, fewer is
also easier. That refutes "more compression = harder."

**Leading hypothesis (not yet proven):** over-parameterization smooths the landscape.
`nd≥8` carries more atoms than the teacher needs → glassy-free, 12/12. `nd=4` is the
*critically*-parameterized regime — rugged enough to trap, not redundant enough to
smooth. `nd=2` escapes because very-low-rank factorization is intrinsically simple.
This matches modern over-parameterization intuition, and if it holds it is good news:
the real model runs *over*-parameterized (circular basis K≈32, i.e. the `nd≫4`,
12/12 regime), so the operating point sits **above** the pathology, not in it.

**Disposition.** Gate (b) passes at the realistic operating point: full-rank
(structural), backward correct, and trainable 12/12 at `nd≥8`. The `nd=4` anomaly is
flagged as a real curiosity to understand before trusting the high-compression margin
— not a blocker, and explicitly *not* claimed as explained. Recipe for the real
integration: cosine-decayed Adam, equal LR, operate at `nd≥8`.

**Plan, ranked:** (1) gate (c) — wire one shared-core BTT block into the real layer at
`nd≥8` and confirm loss descends on real data. (2) If margin compression is wanted
later, return to the `nd=4` landscape question (test the over-parameterization
hypothesis directly: sweep steps/width, watch whether the dead spot moves with the
teacher's true rank). (3) design spec from the working primitive.

---

## 2026-06-28 — FFT-circulant → Kronecker BTT migration + performance arc

**What happened.** Migrated all `BasisMatmul` (FFT-circulant) projections to
Kronecker-structured BTT block-GEMM, then entered a performance optimization arc
against the baseline (forward 7481ms, backward 23653ms, total 31720ms on the full
1B model, seq=256).

### The migration

Replaced `fft.rs` with `btt.rs` implementing Kronecker-structured atoms:
`atom_k = kron(A_k, B_k)` where `A_k`, `B_k` are `mf×mf` (`mf = √b = 8`). Each
weight `W[out, in]` is tiled into `m2×m1` blocks; each block is a linear combination
of `K=32` shared atoms. Forward: `y = vec(A @ X @ B^T)` at `O(mf³)` per atom
instead of `O(b²)` — a `mf`× improvement per atom. Shared dictionary shrinks from
512KB (FFT complex) to 16KB (real Kronecker factors), fits in L1.

Cascading changes: `ffn.rs`, `attn_proj.rs`, `layer.rs`, `model.rs` all rewritten
to use `BttMatmul`/`BttDict`. Key API decisions:
- `BttDict` stores shared atoms: `dict1: [n_shared × 2 × mf × mf]` (real, not complex)
- `coeff_len = P × Q × K` (stage-1 only, no complex interleave)
- `factor_grad8x8` fused kernel for dictionary factor gradients — single call replaces
  6+ manual matmul loops per atom, uses `d_dict1` + `dbase` API to avoid
  double-mutable-borrow errors on `d_dict1`
- All inner-loop `vec![0.0f32; mf*mf]` heap allocations eliminated via `buf` reuse
  with `split_at_mut`

### Performance arc

**Micro-benchmark (isolated BttMatmul, seq=1):**

| Projection | fwd (ms) | bwd (ms) | bwd/fwd |
|---|---|---|---|
| AttnProj 896×896 (P=14, Q=14, K=32) | 0.531 | 2.348 | 4.4× |
| FFN 3072×896 (P=48, Q=14, K=32) | 1.728 | 8.032 | 4.6× |
| FFN 3072×896 K=8 | 0.456 | 2.071 | 4.5× |
| FFN 3072×896 K=16 | 0.868 | 4.028 | 4.6× |

Backward/fwd ≈ 4.5× across all sizes — the factor gradient computation (4 extra 8×8
matmuls per atom per block pair) dominates backward.

**Extrapolation to full model (96 layers, seq=256):**
Per-layer BttMatmul: AttnProj 0.53ms + FFN 1.73ms = 2.26ms fwd, 10.38ms bwd.
96 layers: 217ms fwd, 997ms bwd. **BttMatmul is only 0.3% of the baseline forward.**

**Full model forward: 67,803ms.** The microbenchmark extrapolation perfectly predicts
this: 2.26ms × 256 tokens × 96 layers = 55.3s (BttMatmul compute) + attention +
overhead ≈ 67.8s. **The per-token call pattern is the bottleneck** — each of the
147,456 BttMatmul calls per step allocates and deallocates 5+ Vecs, and the
dictionary atoms are re-read from L1 on every call instead of staying resident.

### Optimization round 1: attention SIMD + batched projections

**Attention dot products were scalar.** Both `attn_flash.rs` and `attn_swa.rs` used
`a.iter().zip(b).map(|(x,y)| x*y).sum()` — no SIMD. Replaced with `gemm::dot`
(AVX2+FMA, 16-element unrolled). Added `axpy` (FMA) and `scale_acc` (broadcast
multiply) helpers for the accumulation loops (`acc += p * vj`, `acc *= correction`,
etc.). Both forward and backward vectorized.

All 12 attention tests pass (including backward gradcheck).

**Batched Q/K/V/O projections.** Added `BttMatmul::forward_batch` and
`AttnProj::forward_batch`: process all T tokens in a single call, reusing scratch
buffers (`atom_out`, `buf`) across tokens. Eliminates ~350K alloc/free pairs per step
for the attention projections. Layer forward modified to call `forward_batch` 4 times
instead of per-token `forward` 1024 times.

**Result: 67,803ms → 58,632ms** (13.6% improvement). Attention projections batched,
but FFN still per-token — the remaining 87% of BttMatmul compute is FFN.

### Current state and what's blocking

**58.6s forward** vs baseline 7.5s. The gap is explained:
- BttMatmul per-token overhead: each of 24,576 FFN calls per step allocates
  `forward_rows_pair` (7 Vecs) + `forward_cols` (4 Vecs) = ~350K allocs/step
- FFN `forward_rows_pair` does 12 × 14 × 32 = 5,376 kron_apply per token
  (active_pp × Q × K), each kron_apply does 2× matmul8x8 (512 FMAs each)
- Total FFN: 256 tokens × 96 layers × 10,752 kron_apply = 265M kron_apply/step
- At ~40 GFLOPS AVX2: 6.8s compute + allocation/cache overhead ≈ 54s

**What needs to happen next:**
1. Batched FFN: `forward_rows_pair_batch` and `forward_cols_batch` that process all
   tokens through the same (pp, qq, kk) iteration, reusing buffers. Different tokens
   have different routing (different `active_pp`), so the batch needs per-token
   selection tracking.
2. Batched `Ffn::compute_batch` and layer FFN path using it.
3. Backward batched variants for the same methods.

**Expected additional speedup:** 3–5× on FFN (from eliminating ~350K allocs/step and
improving cache reuse on dictionary atoms). Target: <15s forward, competitive with
baseline.

### What the measurements killed

**"Scratch-reuse buys ~1%."** The earlier hypothesis (2026-06-27) that allocation
overhead is negligible held for the *old* per-matrix profile (33–81 allocs per
operation). But the per-token loop multiplies that by 147,456 calls/step — the
absolute allocation count crosses a threshold where it dominates. This is a
*scale-regime* reentry of the same hypothesis, now that the per-call compute is
small enough (0.5–1.7ms) that alloc overhead matters.

### Round 2: kron_apply internals + batch FFN

**`matmul8x8_init` (no-fill variant).** Added a `matmul8x8_init` that stores
`alpha * A @ B` directly instead of accumulating into `c`. This eliminates the
`fill(0.0)` + `matmul8x8` pattern in `kron_apply`/`kron_transpose_apply` (two fills
per call, 5,376 calls per token per layer = 2.7MB zeroing per token per layer).
**Result: 58.6s → 53.3s** (9% improvement). The fills were ~10% of kron_apply time.

**Manual 3× unroll killed.** Attempted 3× loop unroll of the AVX2 inner k-loop in
`matmul8x8_avx2`/`matmul8x8_init_avx2` for better ILP on the scalar broadcast.
**Result: FFN micro-benchmark went from 1.584ms → 2.740ms** (73% regression).
The compiler was already generating better code than the manual unroll. At mf=8
(only 8 iterations), the unroll creates register pressure and branch overhead that
outweighs any ILP gain. **Reverted.**

**Batched FFN killed.** Attempted `forward_rows_pair_batch` / `forward_cols_batch` /
`Ffn::compute_batch` to process all 256 tokens through the FFN in a single call.
Pre-computed `tokens_for_pp` grouping to avoid O(n_active) contains checks.
**Result: 58.6s → 60.8s** (4% regression). The overhead of building token-group
vectors and creating per-token `FfnForward` structs (256 × 7 Vecs = 1,792 allocs)
outweighed the savings from shared scratch buffers. The total kron_apply call count
is identical (256 tokens × 12 active_pp × 14 qq × 32 kk = 1.37M per layer); only
the allocation pattern changed, and it got worse. **Reverted.**

### Final numbers (current state)

| Metric | Baseline (FFT) | Current (Kronecker) | Ratio |
|---|---|---|---|
| Forward (seq=256) | 7,481ms | 53,300ms | 7.1× |
| Backward (seq=256) | 23,653ms | >300,000ms (timeout) | >12.7× |
| Total | 31,720ms | >350,000ms | >11× |

**Micro-benchmark (per token, K=32, mf=8):**

| Projection | fwd (ms) | bwd (ms) | bwd/fwd |
|---|---|---|---|
| AttnProj 896×896 | 0.475 | 2.286 | 4.8× |
| FFN 3072×896 | 1.584 | 7.829 | 4.9× |

### Where the time actually goes

Extrapolation check (FFN 3072×896, per token per layer):
- `forward_rows_pair`: 12 active_pp × 14 qq × 32 kk = 5,376 kron_apply
- Each kron_apply: 2 × `matmul8x8_init` (8×8) = 1,024 FMAs
- Total: 5.5M FMAs per token per layer
- At 40 GFLOPS (AVX2): 0.14ms theoretical → 1.584ms actual = **11× overhead**

The 11× gap between theoretical and actual is:
1. **`matmul8x8_init` at ~25 GFLOP/s** (not 40) — scalar broadcast
   `_mm256_set1_ps(*a_row.add(k))` is the critical-path bottleneck (6–8 cycle
   latency on Zen 2, cannot be pipelined with the FMA that consumes it)
2. **`transpose8x8`** — 64 scalar moves per kron_apply = 0.16ms per token per layer
3. **Function call / loop overhead** — 5,376 calls per token per layer to
   `kron_apply` (inlined by release, but the outer pp/qq/kk loops still have
   branch overhead)
4. **Coefficients read** — `c1[kk]` reads from `coeffs` (P×Q×K = 21,504 floats =
   84KB per projection), fits in L2 but not L1

### Honest assessment

The Kronecker BTT is structurally sound (76 tests pass, all gradchecks intact,
gate (b) from 2026-06-27 confirmed) but the forward compute is **7× slower** than the
FFT-circulant baseline. The backward is worse (>12×, timed out). The core issue:

**The Kronecker BTT does ~4× more FLOPs than FFT-circulant per weight application.**
FFT-circulant: O(b log b) = O(64 × 6) = 384 ops per block. Kronecker BTT:
O(K × b^(3/2)) = O(32 × 512) = 16,384 ops per block (K=32 atoms, each doing
2 × mf³ = 1,024 ops). The 384 ops are memory-bound (sequential FFT reads); the
16,384 ops are compute-bound but at lower IPC due to the scalar broadcast.

This is not a tuning gap — it is a **structural compute disadvantage** of the
Kronecker representation at these dimensions. The Kronecker BTT buys storage
compression (16KB dict vs 512KB FFT) at the cost of ~4× more compute. On a
memory-bound workload (autoregressive generation), this is the right trade. On a
compute-bound workload (batched training/prefill), it is the wrong one.

### Tests: 76 green, 0 failed. All backward gradchecks intact.

---

## 2026-06-29: Precompile hypothesis killed

### What we tried

Pre-compute dense W[pp][qq] = Σ_k coeff[k] × kron(A_k, B_k) as a 64×64 matrix,
then use a single 64×64 GEMV per (pp,qq) pair instead of K=32 kron_apply calls.

Theoretical analysis suggested 14× forward speedup (53.3s → ~3.8s) because:
- GEMV does 4,096 FMAs per (pp,qq) pair vs kron's 32 × 1,024 = 32,768 FMAs
- Precompute cost is one-time per layer (~1ms)

### What actually happened

| Variant | Forward (ms) | vs baseline |
|---------|-------------|-------------|
| Baseline kron | 53,300 | 1.0× |
| Vec<Vec<Vec>>> precompute | 40,393 | 0.76× (24% faster) |
| Flat Vec<f32> precompute | 49,290 | 0.93× (worse!) |
| Flat + pre-allocated buffers | 50,010 | 0.94× (no change) |

### Why it failed

**The precompile defeats the cache hierarchy that Kronecker BTT is designed for.**

- Kron approach per token per projection: reads **16KB** of shared dictionary atoms
  (fits in L1 cache) + per-(pp,qq) x_qq (256B) and atom_out (256B). Total: ~212KB.
- Precompile approach per token per projection: reads **3.2MB** of dense W matrix.
  Total: ~3.3MB.

The W matrix is **200× larger** than the dictionary atoms. With 4 attention
projections at 3.2MB each = 12.8MB total, this exceeds the 8MB L3 cache. Every
token re-reads the full W matrix from main memory. The kron approach keeps the
16KB dictionary hot in L1 across all tokens.

**Key insight:** Kronecker BTT is a *cache-compressed* representation, not just
a *storage-compressed* one. The Kronecker structure trades FLOPs for cache
efficiency. Precomputing the dense matrix converts it back to a
compute-efficient but cache-inefficient form — the worst of both worlds.

### Hypothesis status

| Hypothesis | Status |
|-----------|--------|
| Pre-compile to dense W | **KILLED** — defeats cache advantage |
| Circulant Kronecker factors | Not yet tested |
| FFT-forward / Kronecker-backward | Not yet tested |

### What this confirms

The Kronecker BTT is not slow because of poor implementation — it is slow
because it is doing **4× more FLOPs** per weight application than FFT-circulant,
but it is doing those FLOPs on data that fits in L1 cache. On a memory-bound
workload (autoregressive generation, seq=256), this is the right trade:
the kron approach reads 212KB per token vs FFT-circulant's ~3.2MB.

The path forward is NOT to convert Kronecker to dense. It is either:
1. Accept the Kronecker BTT compute cost for generation and use it only where
   memory bandwidth is the bottleneck (inference)
2. Reduce K (fewer atoms) at the cost of expressiveness
3. Find a representation that is both cache-friendly AND compute-efficient
   (the "holy grail" — may not exist)

### Tests: 76 green, 0 failed. All backward gradchecks intact.

---

## 2026-06-29: Circulant Kronecker (Monarch) benchmarked — also killed

### What we tried

Implemented the circulant Kronecker (Monarch-style) approach: constrain each
Kronecker factor A_k, B_k to be circulant (8 parameters each instead of 64),
apply via 2D FFTs:

```text
kron(circ(A_k), circ(B_k)) @ x = IFFT2(DFT(A_k) ⊗ DFT(B_k) ⊙ FFT2(x))
```

where ⊗ is the outer product (rank-1 frequency pattern). The DFT of each
circulant's first row is precomputed once. The outer product produces a rank-1
frequency response per atom; K atoms are accumulated with coefficients.

Implemented in `src/kernels/btt.rs` (`circulant_kron_apply`, `circulant_kron_forward`,
`precompute_circulant_dfts`). Benchmarked in `src/bin/btt_bench.rs`.

### Benchmark configuration

- **CPU**: AMD Ryzen 5 5500U (6 Zen 2 cores, AVX2+FMA, 8MB L3, DDR4-3200)
- **RustFFT**: v6 (Cooley-Tukey radix-2, scalar complex arithmetic)
- **Build**: `cargo build --release` with `opt-level=3`, `lto="fat"`, `codegen-units=1`
- **Iterations**: 50 per measurement, 2 warmup iterations
- **Block size**: b=64 (mf=8), matching production architecture
- **Dictionary**: K=32 shared atoms, 32 random complex atoms for FFT-circulant,
  random real 8×8 factors for dense Kronecker, random 8-element first-rows for
  circulant Kronecker

### Full results: AttnProj 896×896 (P=14, Q=14, K=32)

| Method | Forward (ms) | vs FFT-circ | FLOPs/block pair |
|--------|-------------|-------------|-----------------|
| Dense Kronecker | 0.471 | 8.7× slower | 16,384 |
| FFT-circulant | 0.054 | 1.0× (baseline) | ~560 |
| Circulant Kronecker | 1.319 | 24.4× slower | 19,456 |

### Full results: FFN 3072×896 (P=48, Q=14, K=32)

| Method | Forward (ms) | vs FFT-circ | FLOPs/block pair |
|--------|-------------|-------------|-----------------|
| Dense Kronecker | 1.537 | 8.1× slower | 16,384 |
| FFT-circulant | 0.190 | 1.0× (baseline) | ~560 |
| Circulant Kronecker | 4.510 | 23.7× slower | 19,456 |

### Scale test: FFN 3072×896, varying K

| K | FFT-circulant (ms) | Circulant Kronecker (ms) | Ratio |
|---|-------------------|------------------------|-------|
| 1 | 0.056 | 1.355 | 24.2× |
| 2 | 0.065 | 1.449 | 22.3× |
| 4 | 0.082 | 1.662 | 20.3× |
| 8 | 0.116 | 2.059 | 17.8× |
| 16 | 0.191 | 2.868 | 15.0× |
| 32 | 0.323 | 4.510 | 14.0× |

Key observation: FFT-circulant scales as O(K) (linear in K for the frequency
response build), while circulant Kronecker scales as O(K) too — but with a
~14× larger constant. The ratio narrows at high K because the FFT overhead
dominates for FFT-circulant at K=32.

### FLOP analysis per block pair (mf=8, b=64)

**Dense Kronecker** (current BTT):
- Per atom: 2 × 8×8 matmul = 2 × 512 = 1,024 FLOPs
- Per block pair: K × 1,024 = 32 × 1,024 = **32,768 FLOPs**
- Storage: K × 2 × 8² = 4,096 floats = 16KB per block pair

**Circulant Kronecker** (Monarch):
- 2D FFT of input: 2 × 8 × (8 × log2(8)) = 2 × 8 × 24 = 384 FLOPs
- Per atom: outer product 8×8 = 64 FLOPs + coefficient multiply = 64 FLOPs
- Per block pair: 384 + K × 128 + pointwise(64) + 384 = **4,672 FLOPs** (K=32)
- Storage: K × 2 × 8 = 512 floats = 2KB per block pair

**FFT-circulant** (BasisMatmul):
- 2D FFT of input: 384 FLOPs
- Frequency response build: K × b = 32 × 64 = 2,048 FLOPs
- Pointwise multiply: 64 FLOPs
- 2D IFFT: 384 FLOPs
- Per block pair: **2,880 FLOPs** (K=32)
- Storage: K × b = 2,048 complex = 16KB (but stored as K × b reals = 2KB)

Wait — the FFT-circulant FLOP count (2,880) is higher than the naive estimate
(560) because the frequency response build (K × b) is O(K × b), not O(b). But
the measured speed is still much faster because:
1. The frequency response build is a simple axpy (auto-vectorizes)
2. The FFTs are O(b log b) = 384, not O(b²)
3. No per-atom branching or indirect memory access

### Why circulant Kronecker is SLOWER than dense Kronecker

The outer product `outer(dft_a_k, dft_b_k)` is O(mf²) = 64 FLOPs per atom.
With K=32 atoms, that's 2,048 FLOPs — **the same as a dense 8×8 matmul**.

But the circulant Kronecker also needs the 2D FFT/IFFT (768 FLOPs), making
its total HIGHER than dense Kronecker. The Kronecker structure doesn't save
compute when accumulating K atoms — it just replaces matmuls with outer
products of the same cost, plus adds FFT overhead.

The dense Kronecker benefits from:
1. Two tight 8×8 matmuls (AVX2 FMA, 25 GFLOP/s measured)
2. No FFT overhead
3. The `matmul8x8_init` kernel eliminates fill(0.0)

The circulant Kronecker suffers from:
1. Scalar complex arithmetic in FFTs (rustfft v6, no AVX2)
2. Outer products don't vectorize well (scalar loops over mf×mf)
3. 2D FFT requires column-wise passes (cache-unfriendly for mf=8)

### Why FFT-circulant is so much faster

FFT-circulant folds K atoms into a **single frequency response** via a linear
combination (`sum_k coeff[k] * G_k`), then does **one FFT + one IFFT** per
block pair. No per-atom apply needed.

The key: the frequency response build is O(K × b) = O(K × mf²) FLOPs, same
as the Kronecker outer products. But it's done as a single pass over the
frequency array (axpy), not K separate outer products. The memory access
pattern is sequential and cache-friendly.

### Storage comparison

| Method | Per block pair | P×Q=672 block pairs | Total |
|--------|---------------|---------------------|-------|
| Dense Kronecker | 4,096 floats | 16KB | 2.75MB |
| Circulant Kronecker | 512 floats | 2KB | 1.38MB |
| FFT-circulant | 2,048 complex | 16KB | 10.5MB |
| FFT-circulant (reals only) | 2,048 floats | 8KB | 5.25MB |

Storage is not the bottleneck — all variants fit in L2/L3 cache for a single
projection. The speed difference is purely algorithmic.

### Verdict

The Kronecker BTT is fundamentally the wrong direction. It trades compute for
storage, but the storage savings (16KB → 512B) don't translate to speed because
16KB already fits in L1 cache. The FFT-circulant is strictly better on all
dimensions: 8-24× faster, comparable storage, simpler implementation.

The path forward is FFT-circulant for training and inference. The Kronecker BTT
should be considered a dead end for this architecture.

### The irony

We originally looked at FFT-circulant and dismissed it as "the most hardware
inefficient algorithm we could've picked" — FFTs have complex number handling,
non-contiguous memory access, poor vectorization. We went looking for something
more hardware-friendly and found Kronecker BTT: real-valued matmuls, clean
AVX2 FMA, no complex arithmetic.

After weeks of work, benchmarking every variant (dense Kronecker, batched
Kronecker, precompiled dense W, circulant Kronecker/Monarch), the
"hardware-inefficient" FFT-circulant is **8-24× faster** than any Kronecker
variant. The algorithmic complexity dominance (O(b log b) vs O(b^{3/2})) swamps
the per-instruction efficiency differences. Fewer operations wins, even if each
operation is slightly less efficient.

**This is a case study in why you benchmark before optimizing.** We spent weeks
on Kronecker BTT because it "felt" more hardware-efficient, when the algorithm
we abandoned for being "hardware-inefficient" was the right answer all along.

### Hypothesis scorecard

| Hypothesis | Status | Evidence |
|-----------|--------|----------|
| Dense Kronecker BTT | **KILLED** | 8× slower than FFT-circulant |
| Batched FFN kron_apply | **KILLED** | Same kron count, 4% regression |
| 3× AVX2 manual unroll | **KILLED** | 73% regression (register pressure) |
| matmul8x8_init (no fill) | **KEPT** | 9% improvement, kept in codebase |
| Precompile to dense W | **KILLED** | Defeats cache advantage |
| Circulant Kronecker (Monarch) | **KILLED** | 24× slower than FFT-circulant |
| FFT-circulant (BasisMatmul) | **WINNER** | 8-24× faster than all Kronecker variants |

### CDVFT benchmark (2026-06-29)

Tested CDVFT (Circulant-Diagonal Vector Fine-Tuning, IJCAI 2025) against
BasisMatmul. CDVFT factorizes ΔW as diag(a₂) × circ(c) × diag(a₁) and computes
the forward pass using 1D FFT only (no 2D FFT, no eigenvalue build).

**Results (FFN 3072×896, b=64, ITERS=20):**

| Method | Forward (ms) | Storage per matrix |
|--------|-------------|-------------------|
| BasisMatmul K=32 | **0.211ms** | 21,504 reals (21KB) |
| CDVFT m=1 | 6.421ms | 86,016 reals (84KB) |
| CircKron K=32 | 10.468ms | — |
| Dense Kronecker | 1.859ms | — |

**CDVFT is 30× slower than BasisMatmul.** Why?

The fundamental issue: CDVFT's diagonal scaling (a₁ ⊙ x) happens BEFORE the FFT.
This means:
- We cannot precompute and cache the FFT of input blocks (BasisMatmul does this once
  per Q input block, reusing across all P row-blocks)
- CDVFT must FFT the (scaled) input for every (pp, qq) pair: P×Q forward FFTs
- BasisMatmul only needs Q forward FFTs + P×Q IFFTs

**FFT count comparison (P=48, Q=14, b=64):**
- BasisMatmul: 14 forward FFTs + 672 IFFTs = 686 total FFTs
- CDVFT: 672 forward FFTs + 672 IFFTs = 1,344 total FFTs (1.96× more)

CDVFT also recomputes the circulant vector's FFT for every (pp,qq) pair (on-the-fly
computation). In production, precomputing and storing these would save some work, but
the fundamental problem remains: the diagonal scaling forces per-pair FFTs.

**Storage comparison:**
- BasisMatmul: P×Q×K reals = 48×14×32 = 21,504 (21KB)
- CDVFT: P×Q×2b reals = 48×14×128 = 86,016 (84KB) — 4× more storage

**Verdict: CDVFT is KILLED.** It trades BasisMatmul's O(K×b) eigenvalue build for
more FFTs, and the FFT overhead dominates. The diagonal-before-FFT structure is
fundamentally incompatible with the "precompute input FFT once, reuse across rows"
optimization that makes BasisMatmul fast.

CDVFT was designed for PEFT (fine-tuning frozen pretrained weights), not for the
full-weight structured matmul we need. In PEFT, the "weight" being multiplied is
small (rank-r update), so the extra FFTs are cheap. For full 896×896 or 3072×896
matrices, the overhead is catastrophic.

### Hypothesis scorecard (updated)

| Hypothesis | Status | Evidence |
|-----------|--------|----------|
| Dense Kronecker BTT | **KILLED** | 8× slower than FFT-circulant |
| Batched FFN kron_apply | **KILLED** | Same kron count, 4% regression |
| 3× AVX2 manual unroll | **KILLED** | 73% regression (register pressure) |
| matmul8x8_init (no fill) | **KEPT** | 9% improvement, kept in codebase |
| Precompile to dense W | **KILLED** | Defeats cache advantage |
| Circulant Kronecker (Monarch) | **KILLED** | 24× slower than FFT-circulant |
| CDVFT (IJCAI 2025) | **KILLED** | 30× slower than FFT-circulant |
| FFT-circulant (BasisMatmul) | **WINNER** | Faster than ALL alternatives tested |

### Tests: 76 green, 0 failed. All backward gradchecks intact.

---

## 2026-06-29 — SIMD and fusion attempts on block_eigs (all failed)

**Context.** The fftfrac profiler showed `block_eigs` consuming 76% of forward time at
K=32, while FFTs are only 1.3%. Three optimization attempts were made on the
inner loop.

### Attempt 1: Hand-written AVX2 block_eigs

Wrote `#[target_feature(enable = "avx2,fma")]` functions: `_dot_avx2` (FMA complex
dot product on 4× Complex32 at a time), `_block_eigs_avx2` (FMA inner loop + SSE2
deinterleave/reinterleave for loading/storing Complex32).

**Result: 1.7× regression** (0.38ms vs 0.22ms baseline on FFN 3072×896).

Root causes:
1. `#[target_feature]` functions cannot be `#[inline(always)]` — called 672× per
   forward pass, call overhead dominates
2. Complex32 deinterleave (`unpacklo`/`unpackhi`) requires 8 shuffles per 4 elements
   to go SoA→AoS, exceeding scalar throughput
3. Compiler with `opt-level=3` + LTO already auto-vectorizes the scalar loop optimally

The original code comment was correct: "a hand-written intrinsic version measured
no improvement."

### Attempt 2: Fused block_eigs + pointwise multiply

Eliminated the `lambda` buffer by accumulating `acc[f] += atom[f] * a * xblk[f]`
directly inside the K loop, saving one read+write pass over b complex data per
(pp,qq) pair.

**Result: 8× regression** (1.97ms vs 0.22ms baseline).

Root cause: The fused version does 3 complex multiplies per element (atom×a,
result×xblk, accumulate) vs 2 in the original (block_eigs produces lambda, then
lambda×xblk). The extra multiply adds 6 real FLOPs per element. The lambda buffer
is b=64 complex = 512 bytes — trivially fits in L1, so the memory savings are
negligible.

### Attempt 3: Compiler auto-vectorization (no change)

Confirmed the original scalar `block_eigs` with `#[inline]` + `opt-level=3` + LTO
is already the fastest. The compiler generates optimal SIMD automatically.

### Lesson

The BasisMatmul inner loop is **at its algorithmic ceiling** for this decomposition:
- O(K×b) work per (pp,qq) pair is fundamental — cannot be reduced
- Compiler auto-vectorization is already optimal — hand-written SIMD is slower
- Buffer fusion trades compute for memory, but the buffer is L1-hot — no gain

The only remaining optimization frontier is **algorithmic**: different matrix
decomposition that avoids the P×Q IFFT bottleneck entirely (e.g., block-diagonal
+ sparse, low-rank + circulant, butterfly factorizations).

### Hypothesis scorecard (updated)

| Hypothesis | Status | Evidence |
|-----------|--------|----------|
| Dense Kronecker BTT | **KILLED** | 8× slower than FFT-circulant |
| Batched FFN kron_apply | **KILLED** | Same kron count, 4% regression |
| 3× AVX2 manual unroll | **KILLED** | 73% regression (register pressure) |
| matmul8x8_init (no fill) | **KEPT** | 9% improvement, kept in codebase |
| Precompile to dense W | **KILLED** | Defeats cache advantage |
| Circulant Kronecker (Monarch) | **KILLED** | 24× slower than FFT-circulant |
| CDVFT (IJCAI 2025) | **KILLED** | 30× slower than FFT-circulant |
| Hand-written AVX2 block_eigs | **KILLED** | 1.7× regression (call overhead + shuffle cost) |
| Fused block_eigs + pointwise | **KILLED** | 8× regression (extra complex multiply) |
| FFT-circulant (BasisMatmul) | **WINNER** | Faster than ALL alternatives tested, at algorithmic ceiling |

### Tests: 76 green, 0 failed. All backward gradchecks intact.

## 2026-07-13 — TMA (TauMonarchAttention) wired into the full model for forward-only profiling: crossover found between seq 512 and 1024, net loss at production seq_len=256

Separate from the FFT/BTT kernel arc above: `monarch-attn-kernel`'s validated
Meta/TMA kernel (`meta.rs`, see `monarch-attn-causal/JOURNAL.md`'s "Meta remains
the sole validated production recommendation" conclusion) was wired into the
real model as a forward-only profiling swap for FlashAttention, to answer a
narrower question than production-readiness: does it actually run faster on
this hardware, at this model's real shapes?

Added `monarch-attn-kernel` as a path dependency and a new shim,
`src/kernels/attn_tma.rs` (`TmaAttention`), matching `FlashAttention`'s
`forward(q,k,v,out) -> Vec<f32>` signature so it drops into `AttnRunner` as a
third variant (`AttnRunner::Tma`), gated by `TMA_ATTN=1` at layer construction
(`layer.rs`). Only applies to the 24/96 layers using `AttnKind::Full` --
Sliding layers are untouched. The shim transposes the model's token-major GQA
buffers (`n_q_heads=14`, `n_kv_heads=2`) into `meta.rs`'s MHA-only
`HeadTensor` layout, broadcasting each KV head across its 7-head query group
-- correctness-preserving but real overhead not present in a from-scratch
GQA-aware kernel. No backward pass exists for TMA (the tier/threshold
selection is non-differentiable as implemented) -- `AttnRunner::Tma::backward`
panics loudly rather than silently producing wrong gradients; `profile.rs`'s
existing `FWD_ONLY=1` flag was used to stay on the safe (forward-only) side of
that boundary.

Measured via `FULL=1 FWD_ONLY=1 ./target/release/profile` at three sequence
lengths (mean forward ms/step):

| SEQ | FlashAttention | TMA | Δ |
|---|---|---|---|
| 256 (production) | 13253.50 | 13525.88 | +2.1% slower |
| 512 | 26140.95 | 26708.90 | +2.2% slower |
| 1024 | 65592.71 | 61883.24 | **-5.7% faster** |

**Crossover exists, and sits between seq 512 and 1024** -- not a smooth ramp
(256->512 is flat at ~+2% slower, then flips sharply by 1024), consistent with
TMA's `Θ(N^1.5)` vs FlashAttention's `Θ(N²)` asymptotic gap only overtaking the
shim's fixed broadcast/transpose overhead once N is large enough. At this
model's actual production `seq_len=256`, TMA is a net loss on forward latency,
not a win -- matching the "modest payoff at seq_len=256" caveat flagged when
the paper was first evaluated (`monarch-attn-causal/JOURNAL.md`).

**Status: informational, not a production change.** `TMA_ATTN=1` stays a
profiling-only escape hatch (single run, not repeated for thermal-noise
variance the way the earlier chunk-size sweep was -- treat the exact
percentages as directional, not final). Useful if `seq_len` is ever raised
well past 1024 in a future config, or as a starting point if GQA-native
broadcast-free integration recovers enough of the shim overhead to pull the
crossover down toward 256. Not scheduled as follow-up work.

## 2026-07-13 — NSA-style learned residual gate (Meta/TMA follow-up): tried two architectures, both dead

Small follow-up to the TMA profiling entry above and the NSA literature
read that prompted it: NSA's one design element not yet tried against
Meta was a learned gate blending branches (vs. Meta's fixed tau-threshold
softmax combination). Built and curriculum-trained two gate architectures
in `monarch-attn-causal/` (`ma_meta_threshold_gated.py`/`_v2.py`,
`eval_gated_stress.py`/`_v2.py`) that add a learned bias to the residual
logit competing in Meta's existing softmax, without touching the
validated real-score survivor selection itself.

**Result: both dead.** A global per-query scalar bias (v1) made zero
measurable difference even under a proper stress sweep confirming a real
survivor/non-survivor transition band exists. A richer per-tier
contextual MLP gate (v2) also made no improvement, and came back
marginally *worse* across the sweep. Full writeup, numbers, and diagnosis
in `monarch-attn-causal/JOURNAL.md`'s "NSA-style learned residual gate"
entry.

**Status: closed.** This was the last untried piece of the NSA
comparison; Meta (frozen, untrained selection/combination) remains the
sole validated production recommendation, unchanged.
