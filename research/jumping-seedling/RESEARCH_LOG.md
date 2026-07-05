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

---

## 2026-06-30 — SharedMonarchMatmul: the pivot that actually worked

After exhausting all alternatives to BasisMatmul (every Kronecker variant, CDVFT,
hand-written SIMD on block_eigs — all dead-ended), the bottleneck analysis shifted.
BasisMatmul is at its **algorithmic ceiling**: compiler auto-vectorization already
produces optimal SIMD, and no buffer fusion or hand-written intrinsic has moved it.
The only lever left is a different decomposition.

The candidate that survived theoretical analysis: a **shared-atom Monarch matrix**
(`SharedMonarchMatmul`). The structure differs from the earlier Monarch probe
(`monarch_probe.rs`, killed 2026-06-29 as 10.4× faster but not-equal-param) in one
key way — a shared atom dictionary is introduced at the BTT level, giving the
compression knob we had in BasisMatmul without the FFT overhead.

**Structure:**
```
y[pp] = Σ_{qq} block(pp, qq, x[qq])
block: two-stage block-diagonal GEMM with m=8 (b = m² = 64)
  stage-1: y1[i, r] = Σ_d a1[pp,qq,i,d] · (D1[d, :, :] · x_i)   for i in 0..m
  transpose z[j][i] = y1[i][j]
  stage-2: out[j, r] = Σ_d a2[pp,qq,j,d] · (D2[d, :, :] · z_j)   for j in 0..m
```
Parameters: `D1, D2 ∈ ℝ^{nd × b}` shared across all (pp, qq); `a1, a2 ∈ ℝ^{P×Q×m×nd}`
per-block coefficients. Compression knob is `nd`: at nd=8, each block uses 8 shared
atoms per stage instead of a full 64×64 matrix. AVX2 hand-kernels: `fwd_block_avx2`,
`bwd_block_avx2` — explicit intrinsics because the inner reduction over `nd` atoms
doesn't auto-vectorize.

**Gate (c) result — first end-to-end training test:** The gradcheck on the full
projection (finite differences on MSE loss vs analytical backward) passed with
`max_err < 3e-5`. Loss descends on real character data. Gate (c) passed.

**Why this beats BasisMatmul in the training regime:** BasisMatmul did O(K×b) real
multiplies per (pp,qq) pair for the frequency-response build, then one IFFT. The
shared-atom Monarch does O(nd×b) per block per stage — structurally the same count,
but in **real-valued block-GEMM** instead of complex FFT. For nd=8, b=64: each stage
does 8×64 = 512 real MACs per block, which composes cleanly into AVX2 FMA without
the complex-deinterleave shuffle overhead that killed BasisMatmul's vectorization.

---

## 2026-06-30 — First end-to-end character-level training run

Built `src/bin/train_char.rs` (727 lines): a complete character-level training binary
for a 2-layer, 256-hidden, 4-head Fydel model using `SharedMonarchMatmul` for all
projections. Purpose: validate the full forward + backward + optimizer loop on real
data at small scale before scaling.

**Architecture (train_char config):**
```
HIDDEN = 256, FFN_DIM = 1024, N_HEADS = 4, N_LAYERS = 2
SEQ_LEN = 128, VOCAB = 128 (ASCII), ACCUM_STEPS = 4
B = 64, ND = 8, M = 8   →   113K total parameters
```

Data: TinyShakespeare (`Trelis/tiny-shakespeare`, 1,115,394 bytes). ASCII 0–127
tokenizer. Cross-entropy with tied output weights (embed^T reused as output head).

**Key implementation decisions:**
- `cross_entropy` returns `(mean_loss, dlogits)` where `dlogits = (softmax - one_hot) / SEQ_LEN`
- Tied weight backward: `d_embed` accumulates from *both* the output head gradient
  (`d_embed[v] += dlogits[t,v] · h_norm[t]`) and the embedding scatter-add
  (`d_embed[tokens[t]] += dh[t]`). Both paths are correct because the embedding
  participates in both operations.
- `scale_model_grads` divides by `ACCUM_STEPS` — the SEQ_LEN normalization is already
  inside `cross_entropy`, no double-scaling.
- `clip_grads` clips the **global** L2 norm across all parameters to 1.0.
- Adam per parameter group (d1/d2/a1/a2 separate states per projection, plus norms).
- LR schedule: linear warmup for 100 steps, then cosine decay to 0 over `N_OPT_STEPS`.

**Run 1 (fixed LR = 3e-4, no accumulation):** Descent to ~2.45 by step 300, then
oscillation amplitude ~0.35 nats that never narrowed. The model walked the loss basin
rather than settling — classic too-large LR combined with single-sample gradient
variance.

**Run 2 (cosine LR + ACCUM_STEPS=4):** Oscillation envelope compressed to
~0.10–0.15 nats. Reached floor ~2.30 nats by step 2100, held through step 3000.
Both runs reached the **same floor** (~2.30) despite dramatically different dynamics.

**Two-runs-same-floor is the tell.** A floor sensitive to model capacity would shift
when you change training dynamics — different trajectory lands in a different local
minimum. The fact that maximum-variance training (batch=1, fixed LR, large
oscillations) and minimum-variance training (accum=4, cosine, tight oscillations)
both park at 2.30 nats means something structural is capping learning, not the
optimizer. Expected floor for a well-tuned 113K-param Shakespeare model: ~1.8–2.0
nats. The gap pointed to a systematic error.

---

## 2026-06-30 — Kernel scaling bug: missing 1/√Q in atom initialization

**The bug (one line, `monarch.rs:225`):**
```rust
// before (wrong):
let s_atom = 1.0 / (m as f32).sqrt();

// after (correct):
let s_atom = 1.0 / (m as f32 * q as f32).sqrt();
```

**Root cause:** The initialization scaled atoms by `1/√m` to make each sub-matmul
variance-preserving at the block level. But it did not account for the Q-block
**summation** in the forward pass: `y[pp] = Σ_{qq=0..Q} block_out(pp, qq)`. Summing
Q independent blocks multiplies output variance by Q.

**Variance analysis with old init (s_atom = 1/√m, s_coeff = 1/√nd):**
- Var(D1, D2 entries) = `(2/√m)² / 3 = 4/(3m)`
- Var(y1[i,r]) = `m × nd × Var(a1) × Var(D1) × Var(x) = 16/9 ≈ 1.78`
- Var(out_block) = `(16/9)² ≈ 3.16` after stage 2
- Var(y[pp]) = `Q × 3.16` — **grows linearly with Q**

For Q=4 (hidden→hidden, FFN up/gate at hidden=256): **12.5× amplification**.
For Q=16 (FFN down at hidden=256): **50× amplification**.

The residual stream `x + Monarch(x)` was dominated by projection outputs (std ≈ 3.5).
The optimizer had to fight variance mismatch rather than learn structure. RMSNorm
mitigates this within each layer, but the distortion is visible at init and slows
convergence enough to create an artificial floor that two training runs with very
different dynamics both hit at the same value.

**With the fix (s_atom = 1/√(m×Q)):**
- Var(out_block) = `(16/9)² / Q²` per block
- Var(y[pp]) = `Q × (16/9)² / Q² = (16/9)² / Q ≈ 3.16 / Q`
- For Q=4: `≈ 0.79` — just under unit variance ✓

**Fix #1 result (s_atom = 1/√(m·Q)):** 500 steps, step 100 loss 3.16 vs old 4.50.
Floor at step 400: 2.78. Better, but the formula was still wrong — Q^{-1/2} instead of
Q^{-1/4}, and missing the bilinear composition factor of 3.

**Correct derivation:**

From the two-stage composition (Var ∝ s_atom⁴):
```
Var(E1 entry)  = nd × Var(a) × Var(D) = s_atom²/9
Var(y1[i,r])  = m × s_atom²/9
Var(out_block) = (m × s_atom²/9)²    ← bilinear composition squares it
Var(y[pp])    = Q × (m × s_atom²/9)²
```
Setting Var = 1 and solving: **s_atom = 3 · m^{-1/2} · Q^{-1/4}**

For m=8, Q=4: s_atom = 0.75 (not 0.177 from fix #1, not 0.354 from original).

**Depth scaling:** For a residual network of depth n_layers, target Var(output) =
1/(2·n_layers) to prevent exponential norm growth. Since Var ∝ s_atom⁴, the depth
factor on atom values is `(1/(2·n_layers))^(1/4)`. Applied in `monarch_new()` in
`train_char.rs` — the kernel stays pure, depth is a model-level concern.

**Three-way comparison at 500 steps:**

| step | gnorm | fix #1 | fix #2 (no depth) | fix #3 (+ depth) |
|------|-------|--------|-------------------|------------------|
| 100  | 0.54  | 3.16   | 3.20              | 3.33             |
| 200  | 1.25  | 3.08   | 2.77              | 2.98             |
| 300  | 1.54  | 2.87   | **2.57** ← peak   | 2.67             |
| 400  | 1.48  | 2.78   | 2.59 ↑ bounce     | **2.66** stable  |

Fix #2 hit the lowest point (2.57 at step 300) but gnorm climbed to 2.18 and clipping
distorted the update — causing the step 400 rebound. Fix #3 (Var = 0.25, std = 0.5)
peaked at gnorm 1.54 and held. Clipping noise compounds over long runs; fix #3 is the
right call for sustained training.

**Status:** Fix #3 current. 3000-step run completed — see entry below.

---

## 2026-07-01 — 3000-step floor run + eval mode

### 3000-step run results (fix #3)

Full cosine schedule run to 3000 opt-steps (each = 4 forward+backward passes,
512 tokens/update). Key printouts:

| step | loss | gnorm | lr |
|------|------|-------|-----|
| 100  | 3.33 | 0.54  | 3.00e-4 |
| 500  | 2.55 | 1.55  | 2.86e-4 |
| 1000 | 2.34 | 1.56  | 2.34e-4 |
| 2100 | **2.2975** | 2.23 | 6.58e-5 |
| 2500 | 2.44 | 2.25  | 2.15e-5 |
| 2900 | 2.42 | 2.07  | 8.79e-7 |

Single-step minimum: **2.2975** (step 2100). End-of-training average (steps 2500–2900): ~2.38–2.42.

### The oscillation mystery — resolved

Gnorm at step 2900 is still 2.067 even at LR=8.79e-7. That looks like instability, but
it is **batch variance**. Each reported loss is on 512 randomly-sampled tokens — 0.05%
of the 1.1M-token dataset. Batch-to-batch σ at this regime is ±0.10–0.15 nats, which
is exactly the oscillation amplitude seen. The gradient norm is large because random
128-token windows have high per-sequence variance, not because the model is unstable.
`clip_grads` absorbs this by capping step magnitude without distorting direction.

The implication: the printed per-step loss during training is **not a reliable floor
estimate**. A single lucky window (low-entropy passage) can show 2.30; a hard passage
can show 2.50. Neither is the true expected loss.

### Added `--eval` mode

Added `load_checkpoint` + `eval_loss` to `train_char.rs`. Usage:

```
cargo run --bin train_char --release -- data/input.txt --eval
```

Averages `Model::forward()` over 500 non-training windows (64,000 tokens) with a
fixed seed, σ ≈ 0.013 nats. No backward pass. Runs in ~41 seconds.

### True floor (honest measurement)

```
checkpoint: step 3000
running eval over 500 windows (64000 tokens) …
eval loss: 2.3932  (41.4s)
```

**True floor at step 3000: 2.3932 nats.** The 2.2975 seen at step 2100 was a single
lucky batch. The 2.38–2.45 oscillations at the tail were the real range.

For context, the theoretical floor for a 113K-param character model on TinyShakespeare
is roughly 1.7–2.0 nats (entropy of natural English text ≈ 1.0 nat/char, but a small
model can't fully exploit structure). 2.39 is above this — there is still headroom,
likely from model capacity (2 layers, 256 hidden is small) rather than from the
kernel/init.

**Next:** dense baseline comparison (see entry below), then fast-exp for SwiGLU.

---

## 2026-07-01 — Dense baseline comparison

To quantify the expressiveness cost of the Monarch structure, a dense linear-layer
baseline (`train_char_dense.rs`) was run at two configs: same hidden dimension as
Monarch (Dense-256, the expressiveness ceiling) and roughly matched parameter count
(Dense-64, ~139K params vs Monarch's 113K).

### Configs

| Model | HIDDEN | Params | Init |
|-------|--------|--------|------|
| Monarch-256 | 256 | 113K | variance-preserving + depth scale (fix #3) |
| Dense-256 | 256 | 2131K | Xavier uniform, residual projections scaled ×1/√N_LAYERS |
| Dense-64 | 64 | 139K | same as Dense-256 |

All runs: same LR (3e-4 cosine, 100-step warmup), ACCUM_STEPS=4, 3000 opt-steps,
TinyShakespeare. Eval: 500 fixed-seed windows (64K tokens), σ ≈ 0.013 nats.

### Results

| Model | Params | Eval loss | ms/step |
|-------|--------|-----------|---------|
| Dense-64 | 139K | 2.4552 | 57 |
| **Monarch-256** | **113K** | **2.3932** | **1100** |
| Dense-256 | 2131K | 2.3409 | 1262 |

### What this shows

**Monarch-256 beats Dense-64 by 0.062 nats at matched params.** This is the key
result: the structured compression isn't just a parameter count story. Monarch uses
its compression budget to *keep the hidden dimension high* (H=256), while a dense
model at 113K params can only afford H≈64. A 64-wide residual stream is a genuine
bottleneck — the model can't route information richly between tokens regardless of
how expressive the individual projections are. Monarch avoids this by using low-rank
*weight structure* instead of a narrow *stream*.

**Dense-256 is 0.053 nats better than Monarch-256** despite having 19× more
parameters and being 15% slower. The compression tax at this scale is small in
absolute terms. The speed gap being only 15% (not proportional to param count) is
explained by the O(S²·H) attention being identical for both — at H=256, seq=128,
attention dominates and projection cost is secondary.

**The scaling concern is real but manageable.** At 1B scale, projections dominate
(attention becomes cheap relative to d_model), so Dense-1B would be ~10-20× slower
than Monarch-1B on CPU — effectively impractical. Whether the 0.05 nat quality gap
at small scale grows, stays, or shrinks at 1B is unknown. The param-matched result
(Monarch beating Dense-64) suggests the hidden-dimension advantage compounds with
scale, but this is extrapolation.

**What is not explained by this comparison:** The Dense-64 training curve was
notably slower to converge (step 100: 3.84 nats vs 3.33 for Monarch), suggesting
the narrow hidden dim also hurts optimization dynamics, not just final capacity.
This could be a confound — a 2-layer 64-hidden model may be underparameterised in
the optimizer sense, not just the representation sense.

### Honest limitations

- No matched-compute comparison (Monarch at 1100ms/step could train for 20× more
  steps than Dense-256 in the same wall-clock time — a fairer comparison than
  matched steps).
- The quality gap at 1B scale is unknown and cannot be measured without a GPU.
- Dense-64's H=64 means HEAD_DIM=16 (vs 64 for Monarch), which may independently
  hurt attention quality.

**Next:** fast-exp polynomial approximation for SwiGLU sigmoid.

## 2026-07-01 — fast-exp SwiGLU kernel: no measurable win at this scale

Implemented `src/kernels/fastmath.rs`: a Cephes-derived polynomial `fast_exp`
(range-reduce to `x = n·ln2 + r`, build `2^n` via direct float-exponent bit
packing, approximate `e^r` with a degree-5 minimax polynomial — the standard
avx_mathfun-style approach). Wrapped as vectorized AVX2/FMA 8-lane
`swiglu_forward`/`swiglu_backward` (Monarch's `up·gate·sigmoid(gate)`) and
`glu_forward`/`glu_backward` (dense baseline's `up·sigmoid(gate)`), each with a
scalar tail for non-multiple-of-8 remainders. `fast_exp` measured at <1e-5
relative error vs `std::exp` across the tested range; unit tests confirm the
vectorized SwiGLU/GLU forward+backward match the scalar reference to <1e-4
absolute error. Replaced the four `.exp()` call sites in `train_char.rs` and
`train_char_dense.rs` (SwiGLU/GLU forward and backward).

### Before/after benchmark (150 opt-steps, same seed, isolated via git-patch
revert/reapply so only the exp() call sites changed)

| Config | Before (scalar exp) | After (fast-exp AVX2) | Δ |
|--------|---------------------|------------------------|---|
| Monarch-256 | 922 ms/step | 956 ms/step | +3.7% (noise) |
| Dense-64 | 60 ms/step | 60 ms/step | 0% |

### What this shows

**No measurable speedup at this scale, in either direction outside noise.**
The FFN activation exp() calls are not the bottleneck for either model at
HIDDEN=256/FFN_DIM=1024 (Monarch) or HIDDEN=64/FFN_DIM=256 (Dense-64) — this
is consistent with the earlier finding that O(S²·H) attention dominates step
time at these small hidden dims and short sequence length (seq=128). The
kernel is *correct* (verified against reference) but currently *inert*: it
replaces a cost that was already too small to show up in wall-clock time.

**Why this might still matter at scale.** The original motivation was that at
1B params, FFN_DIM would be ~3072 and the FFN is expected to be 25-35% of step
time (per the earlier scaling-concern analysis) — a regime this toy-scale
benchmark cannot exercise. Whether fast-exp helps there is still unverified;
it would need a run at closer-to-target FFN_DIM/HIDDEN to know, which isn't
feasible to bench end-to-end at 1B on this CPU. The honest conclusion is: this
kernel is a correctly-implemented but so-far unproven optimization — validated
for correctness, not yet validated for the regime it was intended to help.

### Honest limitations

- Benchmarked at N_LAYERS=2, seq=128 — far from the 1B target config
  (N_LAYERS=96, FFN_DIM=3072) where the hypothesis says this should matter.
- 150-step benchmarks have some step-to-step timing noise (visible in the
  ms/step still drifting at step 9 vs the step-100 steady-state value); the
  ±3.7% Monarch delta is within that noise band, not a real regression.
- Accidentally deleted the Dense-256/Dense-64 checkpoint files
  (`checkpoint_dense.bin`, `checkpoint_dense64.bin`) while clearing state for
  a clean benchmark run — they were regenerated by this session's dense-64
  reruns, but the original Dense-256 checkpoint is gone and would need
  retraining to reproduce bit-for-bit.

**Next:** either validate at closer-to-1B FFN_DIM/HIDDEN before investing more
in this direction, or move to a part of the step budget known to dominate at
small scale (attention) — e.g. profiling to confirm where the 900+ ms/step
for Monarch-256 is actually being spent before optimizing further blind.

## 2026-07-01 — tree-shaped profiler + batched forward: real 15-24% win

Followed up on the flat profiler's finding (FFN backward = 4-6x its forward,
attention only ~4% of step time) by rewriting the profiler to be tree-shaped
instead of flat: `src/bin/train_char.rs` now has a RAII span stack
(`let _s = span("name");`, thread-local, gated by `PROFILE=1`) that nests
child spans under their caller and prints an indented tree with per-node
inclusive time and % of parent. This let sub-phases of `ffn_block_bwd` and
`qkv_proj_bwd` (down/gate/up-proj, wq/wk/wv) be isolated individually instead
of only seeing the coarse parent bucket.

### Diagnosis

The nested breakdown showed FFN backward costing 4.5-6x its forward, but QKV
backward (square 256→256 projections) only costing ~2x — the ratio a normal
dW+dx backward should show. Per-call timing analysis (total phase time ÷
number of `SharedMonarchMatmul::forward` calls) showed near-flat ~19-34µs/call
across projections with wildly different block counts (QKV: 16 `(pp,qq)`
blocks; FFN projections: 64 blocks, 4x more) — if time scaled with block
count as it should for real compute, FFN's per-call cost should have been
~4x QKV's, not ~1.7x. That flat-regardless-of-workload signature pointed at a
large **fixed per-call overhead**, not FLOPs — most likely rayon's
work-stealing dispatch/thread-wake cost, since `forward` was being called
once *per token* (`SEQ_LEN=128` separate rayon dispatches per projection per
layer per opt-step), each dispatching only 16-64 units of genuinely tiny AVX2
work.

### Fix: batch tokens through one rayon dispatch instead of 128

Added `SharedMonarchMatmul::forward_batch(x, n_tokens)` (`src/kernels/
monarch.rs`) — same per-block math as `forward`, but parallelizes over the
flattened `(token, pp)` space in one `into_par_iter()` call instead of
dispatching separately per token. `backward`'s signature changed from taking
`cache: &FwdCache` to `zs: &[f32]` (only `zs` was ever read) plus a new
`zs_at(cache, token)` helper, so a per-token slice of a *batched* cache can
still be fed into the (still serial, unbatched) `backward` — Phase 2 of
batching backward itself was deliberately deferred; see below.

Verified against the production model's `AttnProj::forward_batch` in
`btt.rs` (already does per-layer token-batched calls, though without inner
rayon) — confirms the "batch across tokens" direction matches established
convention in this codebase, not a one-off guess.

Correctness: `SharedMonarchMatmul` had zero existing tests. Added two —
`forward_batch_matches_looped_forward` (exact equality vs. the per-token
loop) and `backward_from_batched_cache_matches_backward_from_single_cache`
(gradients from a `zs_at`-sliced batched cache match a single-token cache) —
both pass. Also re-ran `gate_c`'s existing finite-difference gradcheck after
changing `backward`'s signature (max_err ~1e-5, PASS) to make sure the
signature change didn't silently break the standalone `SharedMonarchMatmul`
consumers (`gate_c.rs`).

### Results (150 opt-steps, Monarch-256, same seed)

| | Before (per-token dispatch) | After (batched) | Reduction |
|---|---|---|---|
| qkv_proj_fwd | 9229 ms | 1341 ms | 6.9× |
| ffn_block_fwd | 13877 ms | 4211 ms | 3.3× |
| wo_proj_fwd | 3007 ms | 505 ms | 6.0× |
| **step time** | **868-956 ms/step** | **717-723 ms/step** | **~15-24%** |

Loss trajectory is bit-identical to the pre-refactor run through step 100
(loss 3.3266 both before and after) — the batching is a pure dispatch-
granularity change, not a numerics change.

Backward is untouched (still one rayon-free serial call per token), so it's
now ~92% of tracked step time (up from ~79%, purely because forward got
cheaper). `ffn_block_bwd` alone is 67.0% of the step.

### What this shows

The dispatch-overhead hypothesis from the previous entry was correct and
was the dominant cost in forward — not attention, not FLOPs, not something
inherent to the Monarch factorization. This is now the clearest lever left:
backward is where nearly all remaining step time lives, and it's still
being called once per token per projection.

### Honest limitations

- Backward batching (Phase 2 — parallelize `(token, pp)` for `dx`, plus a
  `fold`/`reduce`-based accumulation for `da1`/`da2`/`dd1`/`dd2` since those
  gradients sum across tokens, not just across `(pp,qq)`) was scoped but not
  implemented. It's expected to be riskier: dx is embarrassingly parallel
  per-token like forward, but the weight-gradient accumulation needs an
  actual parallel reduction, and there's no existing gradcheck test to lean
  on beyond the two added this session.
- Didn't verify whether rayon dispatch overhead is the *sole* explanation
  for the remaining ~1.3-1.7x sub-linear-with-blockcount scaling in forward's
  per-call cost, or whether some of that was noise/thread-pool warm-up
  ordering (up_proj was consistently ~30% slower than gate_proj despite
  identical shape, unexplained).
- Tree-profiler spans themselves add measurable overhead when enabled
  (visible as the profiled run's step time being close to but not identical
  to the unprofiled benchmark) — fine for relative-proportion analysis, not
  for absolute ms claims at leaf granularity.

**Next:** backward batching (Phase 2) is the highest-leverage remaining
target, given backward is now ~92% of step time. Needs a deliberate design
for the da1/da2/dd1/dd2 parallel reduction before touching code, plus a
proper gradcheck since none exists for `SharedMonarchMatmul` beyond what
this session added.

## 2026-07-01 — Phase 2: parallelize backward across tokens

Backward was ~92% of step time after Phase 1 (batched forward). Considered
two designs for parallelizing it:

- **Block-axis (`qq`) parallelism inside `SharedMonarchMatmul::backward`**
  (parallelize by block instead of token, since `da1`/`da2` land in disjoint
  memory per `(pp,qq)` regardless of iteration order — only `dd1`/`dd2`,
  small shared-atom-dictionary gradients, would need cross-thread reduction).
  Lower reduction cost, but requires restructuring the kernel internals.
- **Token-axis parallelism in the caller** (`Layer::backward` in
  `train_char.rs`): compute each token's full gradient contribution in
  parallel via rayon, `collect()` into an order-preserving `Vec` (rayon's
  indexed iterators preserve input order), then merge with the exact same
  sequential `acc_grads` loop the code already had. Reuses
  `SharedMonarchMatmul::backward` completely unchanged — already
  gradcheck-verified, no new kernel-internals risk.

Went with the caller-side (token-axis) design: estimated the block-axis
version's reduction-cost saving at ~0.2% of backward's runtime (negligible),
so the added implementation risk of touching kernel internals wasn't
justified by the win. Key insight that made this safe: the non-determinism
concern I'd initially raised (rayon's `fold`/`reduce` don't guarantee summation
order) isn't actually a property of *which axis* you parallelize over — it's
specifically about using `fold`/`reduce`. Using `collect()` (order-preserving)
+ a manual sequential merge avoids it entirely, at any parallelism granularity.

### Implementation

Restructured the three per-token serial loops in `Layer::backward`
(`ffn_block_bwd`, `wo_proj_bwd`, `qkv_proj_bwd`) to use `rayon`'s
`par_chunks`/`par_chunks_mut` + `.map().collect()` for the independent
per-token compute, keeping the existing `acc_grads` merge loop unchanged
afterward. Disjoint mutable buffers (`d_h_mid`, `d_attn_out`, `d_h_attn`,
`dx`) are threaded through as `par_chunks_mut` zip members rather than
indexed from inside the closure, so each token's write target is provably
disjoint from every other token's — no unsafe code needed.

Dropped the fine-grained sub-phase spans (`down_proj_bwd`, `wq_bwd`, etc.)
from inside the now-parallel closures — each rayon worker thread has its own
thread-local span stack, so spans recorded there would be invisible to
`print_profile_summary` (which only reads the calling thread's stack) and
would silently undercount. Kept the outer per-block spans
(`ffn_block_bwd`/`wo_proj_bwd`/`qkv_proj_bwd`), which still give an accurate
total since they're measured from the calling thread around the whole
dispatch+merge.

### Results (150 opt-steps, Monarch-256, same seed)

| | Baseline (original) | Phase 1 (batched fwd) | Phase 2 (+ parallel bwd) |
|---|---|---|---|
| step time | 868-956 ms | 717-723 ms | **296 ms** |
| vs baseline | 1.0× | ~1.2-1.3× | **~2.9-3.2×** |
| vs Phase 1 | — | 1.0× | **~2.4×** |

Loss trajectory bit-identical to both the original and Phase-1 runs through
step 100 (loss 3.3266, exact match) — confirms the collect+sequential-merge
design achieves the parallelism with zero floating-point reordering, exactly
as the design predicted. All 50 kernel-level unit tests pass, including the
two `SharedMonarchMatmul` tests added in the Phase 1 entry.

### What this shows

The combination of both phases took Monarch-256 from ~900ms/step to
~300ms/step on this 6-core/12-thread Ryzen 5500U — almost entirely a
dispatch-granularity and parallelism fix, not a numerics or algorithm change.
Neither phase touched the actual math; both were "call the existing,
already-correct kernels more efficiently." The fast-exp kernel from the
earlier session, by contrast, changed the numerics for a part of the step
that turned out not to matter — a useful contrast in what kind of
optimization pays off in this codebase at this scale.

### Honest limitations

- Didn't re-validate the earlier tree-profile phase breakdown after Phase 2
  — the sub-phase spans were intentionally removed from the parallel
  closures (see above), so there's no fine-grained breakdown of where the
  remaining 296ms goes; only the coarse per-block totals are still
  accurate.
- Rayon parallelism at this token-count (128 tokens ÷ ~6-12 threads) is
  still fairly coarse-grained; haven't checked whether there's a sweet spot
  or diminishing/negative returns at other `SEQ_LEN` values.
- The block-axis (`qq`) design was reasoned about but not implemented or
  benchmarked — the "~0.2% reduction cost" estimate for the token-axis
  design's merge overhead was a rough calculation, not measured directly,
  so the actual comparison between the two designs is theoretical, not
  empirical.
- At 1B scale (96 layers, hidden 896) the thread-pool dispatch/parallelism
  behavior may not extrapolate cleanly from this 2-layer/256-hidden toy
  config — untested.

**Next:** open — both obvious optimization phases for this training loop
are done. Possible directions: revisit the fast-exp kernel now that step
time has changed shape (it may matter more/less proportionally now), attempt
a closer-to-1B-scale profiling run to check whether the phase proportions
found here hold up, or step back to the model-quality side of the project
(the dense-baseline comparison) now that iteration speed is much better.

## 2026-07-01 — fast-exp re-check + parallelization sanity check

Two follow-ups after Phase 2: re-tested whether fast-exp matters now that
step time is ~3x smaller (Amdahl's-law reasoning: a fixed-cost op becomes a
bigger fraction of a smaller total), and empirically verified the Phase 2
parallel-backward determinism claim rather than just trusting the design
argument.

### Fast-exp, re-benchmarked post-Phase-2

Temporarily swapped the two `fastmath::swiglu_forward`/`swiglu_backward`
call sites back to scalar `.exp()` (same technique as the original
before/after bench — direct swap, rebuild, run, revert), now on top of the
batched+parallelized forward/backward:

| | scalar exp | fast-exp (AVX2) |
|---|---|---|
| 150-step run | 295 ms/step | 296 ms/step |

Still no measurable difference (within noise). Confirms the earlier finding
holds even after the step-time shape changed dramatically — `exp()` was
never the bottleneck at this scale, in either the unparallelized or
parallelized version. The kernel stays in the codebase since it's correct
and harmless, but there's no evidence it's earned its complexity at this
scale; would need actual 1B-scale FFN_DIM to know if that changes.

### Parallelization determinism, verified empirically

The Phase 2 design argument was that `collect()` (order-preserving) +
sequential merge gives determinism regardless of thread count, unlike
`fold`/`reduce`. Tested this directly rather than just trusting the
reasoning: ran 20 steps at `RAYON_NUM_THREADS=1`, `=4`, and `=12`, and
diffed the loss/gnorm/lr columns (excluding the timing column, which is
expected to differ). All three runs produced **bit-identical** loss and
gradient-norm values at every step, confirming the merge order really is
independent of how rayon schedules the parallel work. Also re-ran `gate_c`'s
finite-difference gradcheck (max_err ~1e-5, PASS) as a final correctness
signal — no regressions from either phase of this session's work.

### What this shows

Both checks came back clean/negative in a good way: fast-exp confirmed
inert (again), and parallelization confirmed sound. No further action
needed on either front unless the scale changes enough to revisit fast-exp
(1B-scale FFN_DIM) or the token count changes enough to revisit rayon
granularity (this was tested at SEQ_LEN=128 only).

**Next:** open — same options as the prior entry (1B-scale profiling
validation, or shift to the model-quality/dense-comparison track).

## 2026-07-01 — decode/prefill tok/s: distinct from training throughput

Clarified scope: today's earlier Phase 1/2 work optimized **training**
throughput (`Layer::forward`/`backward` over a known `SEQ_LEN=128` batch).
**Decode** (`layer_bench.rs`'s `forward_decode`, autoregressive, one token
at a time via `SharedMonarchMatmul::forward_inference`) is a separate code
path with a different bottleneck shape — Phase 1's core trick (batch many
tokens into one rayon dispatch) doesn't apply, since decode only ever has
one new token per step; there's nothing to batch. Training throughput in
tok/s terms: ~1730 tok/s (512 tokens/opt-step at 296ms/step, fwd+bwd) — up
from ~570-590 tok/s pre-optimization.

### Thread-count test (decode doesn't behave like training did)

| Threads | Decode tok/s | Prefill tok/s |
|---|---|---|
| 1 | 11 | 10 |
| 2 | 24 | 22 |
| 12 (default) | 29 | 50 |

Unlike training's forward (which was almost entirely fixed-dispatch-overhead,
confirmed by near-flat per-call cost regardless of block count), decode's
parallelism gives real, if diminishing, benefit — 1→2 threads nearly doubles
tok/s, 2→12 only adds ~20%. Rules out "just remove rayon entirely" as the
fix; decode is doing genuine useful parallel work, just with a different
efficiency curve than training's per-token dispatch problem.

### Decode phase profile (1B-scale layer: HIDDEN=896, FFN=3072)

Added a flat profiler to `layer_bench.rs` (`PROFILE=1`, same pattern as
`train_char.rs`'s early flat profiler — simpler than the tree version since
decode has no nested per-token loop to profile). Per-layer breakdown
(before fast-exp):

| Phase | Time | % |
|---|---|---|
| down_proj | 93.3 µs | 21.9% |
| up_proj | 75.0 µs | 17.6% |
| gate_proj | 72.4 µs | 17.0% |
| wo | 40.6 µs | 9.5% |
| wq | 40.0 µs | 9.4% |
| wv | 38.8 µs | 9.1% |
| wk | 36.2 µs | 8.5% |
| swiglu | 25.5 µs | 6.0% |
| attn (seq=1, trivial) | 2.5 µs | 0.6% |
| norm1+norm2 | 2.5 µs | 0.6% |

FFN (up+gate+down) is 56.5% combined — same qualitative pattern as training.
Notable: `down_proj` (93.3µs) is slower than `up_proj`/`gate_proj` (75.0/
72.4µs) despite identical total `(pp,qq)` block count (672 each, just
transposed shapes: down has P=14/Q=48, up/gate have P=48/Q=14) — fewer
top-level parallel units (P=14 vs P=48) means worse core utilization for the
same rayon dispatch, the same parallelism-granularity effect found in
training's Phase 1 investigation, showing up here even without any
per-token dispatch multiplication.

### fast-exp re-tested at 1B-scale FFN_DIM — different verdict this time

`layer_bench.rs` had never had `fastmath` wired in (separate binary from
`train_char.rs`) — was still using scalar `.exp()` in both `forward_decode`
and `forward_prefill`'s SwiGLU. Wired in `fastmath::swiglu_forward` at both
call sites.

Paired comparison (same profiled run, `swiglu` phase specifically):
**25.46µs → 4.86µs, a 5.2× reduction** — a real, measurable win at this
FFN_DIM (3072), unlike the earlier training-config result (FFN_DIM=1024,
where the fastmath swap showed no measurable difference). This confirms the
"maybe it matters at 1B scale" caveat from the original fast-exp entry.

However: this ~20.6µs/layer saving **did not show up reliably in the overall
decode/prefill numbers** — 3 repeated un-profiled runs of the whole decode
benchmark gave 362.7/389.9/381.4 µs/layer, a ~27µs spread from run-to-run
noise alone, larger than the swiglu savings. Net effect on 96-layer decode
tok/s: 29 tok/s before and after (no change outside noise), though prefill
showed no clear change either (49-50 tok/s both ways).

### What this shows

Two real, opposite-direction findings depending on granularity: fast-exp
**is** a genuine win at the single-op level at 1B-scale FFN_DIM (contradicts
the training-config "inert" finding — confirms it was scale-dependent, as
flagged), but that win is currently too small relative to this CPU's
run-to-run timing noise to show up as a measurable tok/s change. Worth
keeping the fastmath swap (it's strictly not worse, and lines up with where
the bottleneck grows as FFN_DIM scales further), but it's not the lever that
moves decode/prefill tok/s today.

### Honest limitations

- Only 3 repeat runs for the noise-floor estimate — not a rigorous
  variance characterization, just enough to show the swiglu saving is
  within the same order of magnitude as run-to-run noise.
- Didn't test whether the swiglu win becomes clearly visible at even larger
  FFN_DIM (e.g. artificially inflating FFN_DIM further) or whether
  something else (background CPU load, thermal throttling on this laptop)
  is inflating the noise floor beyond what's inherent to the benchmark.
- The QKV/up-gate dispatch-fusion idea (combine wq/wk/wv sharing the same
  input into one rayon dispatch, similarly up/gate) — proposed as the next
  lever for decode tok/s specifically, since it targets dispatch count
  directly rather than per-op numerics — was not attempted this entry.

**Next:** QKV/up-gate dispatch fusion (reduce decode's 672 sequential
per-token rayon dispatches at 1B scale down toward ~384 by merging
same-input projections into single dispatches) is the untested, more
promising lever for decode/prefill tok/s specifically, versus fast-exp which
is now confirmed real-but-small at this scale.

## 2026-07-01 — QKV/up-gate dispatch fusion: real 13-24% decode win

Implemented the fusion idea from the previous entry. Added
`SharedMonarchMatmul::forward_inference_grouped(projs, x)` in `monarch.rs` —
takes several weight-disjoint projections that share the same input (e.g.
wq/wk/wv all read `h_norm`), and dispatches them in **one** rayon
`into_par_iter()` over the flattened `(projection, pp)` space instead of one
dispatch per projection. Requires all `projs` to share `p`/`q`/`m`/`nd`
(asserted at runtime) — true for wq/wk/wv (all HIDDEN→HIDDEN) and for
up/w_gate (both HIDDEN→FFN).

Added a correctness test
(`forward_inference_grouped_matches_individual_calls`, 3 differently-seeded
projections, exact match vs. looping `forward_inference` individually,
<1e-6 tolerance) before wiring it in — same "test before touching the
call site" pattern as the earlier `forward_batch` work.

Wired into `layer_bench.rs`'s `forward_decode`: replaced 3 separate
`wq`/`wk`/`wv` calls with one `forward_inference_grouped(&[wq,wk,wv], ...)`,
and the 2 separate `up`/`w_gate` calls similarly. `wo` and `w_down` weren't
touched — they're the only consumer of their respective inputs
(`attn_out`, `act`), so there's no sibling to fuse with. `forward_prefill`
wasn't touched this entry (only decode).

### Results (1B-scale layer: HIDDEN=896, FFN=3072, 96-layer extrapolation)

| | Before fusion (fast-exp only) | After fusion | Change |
|---|---|---|---|
| decode 1-layer | 360.9-389.9 µs (noisy) | 324.2-332.9 µs (3 repeats) | ~13-17% faster |
| decode tok/s | 29 | **31-32** | +7-10% |
| qkv_group (profiled) | 104.6 µs (wq+wk+wv summed) | 63.0 µs | **40% less** |
| upgate_group (profiled) | 131.8 µs (up+gate summed) | 107.0 µs | 19% less |

The pre-fusion decode-time range (362.7-389.9µs across 3 repeats) and the
post-fusion range (324.2-332.9µs across 3 repeats) don't overlap — this is
a real win, clearly outside the ~27µs run-to-run noise band established in
the previous entry, unlike the fast-exp result. Prefill wasn't fused this
entry; its numbers moved slightly (49-50→45-47 tok/s) but that's ordinary
noise since its code path is unchanged.

Verified no regressions: all 51 kernel unit tests pass (including the 2 new
`forward_batch`/`backward` tests from the earlier entry and this entry's new
grouped-forward test), `gate_c`'s finite-difference gradcheck still passes
(max_err ~1e-5).

### What this shows

Confirms the hypothesis from the last entry: decode's bottleneck genuinely
was partly dispatch-count (672 sequential per-token rayon calls at 1B
scale), and merging same-input siblings into fewer, larger dispatches
recovers real throughput — the QKV group alone got 40% faster despite doing
the exact same arithmetic, just fewer rayon dispatch/wake cycles. This is
the decode-specific analog of Phase 1's training fix: same underlying
mechanic (dispatch overhead not amortized over enough work per call), a
different fusion axis (sibling projections sharing an input, since decode
has no token batch to fuse across).

### Honest limitations

- `forward_prefill` has the same fusion opportunity (its per-token closure
  also calls wq/wk/wv and up/gate separately) but wasn't updated — untested
  whether the same ~13-40% win applies there, though the *reasoning* should
  transfer since it's the same per-token-call structure just wrapped in an
  outer `par_iter_mut` over tokens.
  `wo` and `w_down` have no fusion partner in this layer's structure — not
  a limitation exactly, just confirms this is a bounded optimization (goes
  from 7 dispatches/layer to 4, not further, without changing what each
  projection's input actually is).
- The 40%/19% per-group numbers are from a single profiled run (with
  profiling overhead present); the top-line decode tok/s number is the
  more trustworthy one since it's from repeated unprofiled runs.
- Didn't check whether grouping introduces any first-touch/cache-locality
  penalty from writing 3x/2x more output into one larger buffer versus 3/2
  smaller separate ones — plausible secondary factor in the win, not
  isolated from the dispatch-count effect.

**Next:** apply the same fusion to `forward_prefill`, and/or consider
whether `wq`/`wk`/`wv`/`wo` could all be fused together if `wo`'s input
dependency (needs `attn_out`, computed from q/k/v) were restructured —
likely not worth the complexity given wo is only 9-11% of decode time.

## 2026-07-01 — fusion applied to prefill: regression, reverted

Tried applying the same `forward_inference_grouped` fusion to
`forward_prefill`'s per-token closure (QKV group + up/gate group), matching
what worked for `forward_decode`. Result: **a small regression, not a win**
— reverted.

### What happened

Initial unpaired comparison (fused prefill, 3 repeats: 44-47 tok/s) looked
roughly flat against the established noise floor (44-50 tok/s across prior
sessions), too noisy to call. Ran a proper same-session paired A/B instead
(same technique as the fast-exp checks — build fused, bench, revert, build
unfused, bench, same session):

| | Unfused (paired, this session) | Fused (paired, this session) |
|---|---|---|
| 1-layer prefill | 55.4 / 57.3 / 56.1 ms (avg 56.3ms) | 60.8 / 60.7 / 56.7 ms (avg 59.4ms) |

Fused is consistently slower (~5% worse) in the fair paired comparison.
Reverted `forward_prefill` back to individual `forward_inference` calls;
`forward_decode`'s fusion (the entry above) is unaffected and still gives
36 tok/s decode / 50 tok/s prefill (prefill's number is from the *unfused*
prefill path — the two functions are independent, only decode is fused).

### Why decode won but prefill didn't

`forward_prefill` already parallelizes over the outer token dimension
(`tokens.par_iter()`, 256-way) before each per-token closure makes its own
inner `forward_inference` calls — nested rayon parallelism. Decode has no
such outer parallelism (`seq=1`, nothing to parallelize over except the
`pp` blocks within a single call). Working hypothesis: the outer 256-way
parallelism in prefill was already keeping cores busy enough that the inner
per-call dispatch overhead was substantially hidden/amortized by rayon's
work-stealing scheduler interleaving many independent token-tasks; fusing
QKV/up-gate into a wider *inner* dispatch per token didn't remove overhead
so much as add nested-scheduling contention between the outer (256 tokens)
and inner (now 3x/2x wider per-token) parallel regions. Not verified
directly (would need rayon-internal tracing/instrumentation to confirm the
mechanism) — this is the most plausible explanation given the facts, not a
proven one.

### What this shows

The dispatch-fusion fix is real but **context-dependent**: it helps when
there's no other parallelism to amortize dispatch overhead (decode), and
can hurt when there's already coarse-grained parallelism doing that job
(prefill). Confirms this wasn't a universal "always fuse projections" rule
— the win in the previous entry was specific to decode's structure, exactly
as flagged as a limitation there ("reasoning *should* transfer" — it
didn't, and that's a useful correction).

### Honest limitations

- The nested-parallelism-contention explanation is a plausible hypothesis
  based on the structural difference (outer parallelism present vs.
  absent), not something directly measured or traced.
- Only tested prefill fusion at `seq=256`; didn't check whether smaller
  prefill batches (e.g. seq=8, seq=32) — where the outer parallelism has
  fewer units to hide overhead with — might flip the result back toward a
  win, matching the decode case's `seq=1` extreme more closely.

**Next:** open. Both training throughput (Phase 1+2) and decode tok/s
(fusion) have real, validated wins in hand. Prefill is confirmed not to
benefit from this specific fix. Remaining open threads: whether small-seq
prefill (partway between decode's seq=1 and this entry's seq=256) shows
different behavior, or shifting back to the model-quality track.

## 2026-07-01 — Opus review + dead `y1s` cache buffer removed

Asked Opus for a fresh, unconstrained second opinion on the session's work
so far (full `RESEARCH_LOG.md` + relevant source as context). Ranked
suggestions, highest priority first:

1. **Correctness gap, not perf**: `layer_bench.rs`'s decode bench has no
   real KV-cache — `forward_decode` self-attends to a single token, so
   `weight = score.exp()/score.exp()` is *always* 1.0 by construction. The
   36 tok/s decode number excludes all KV-cache streaming, which is the
   actual CPU-bandwidth constraint the whole project is about — attention
   shows as 0.6% of decode time only because there's no context to attend
   over. Flagged as higher priority than further speed work; not yet acted
   on this entry.
2. **Free win**: `FwdCache.y1s` is allocated and fully written by
   `forward`/`forward_batch` but never read by `backward` (only `zs` is).
   Pure wasted write bandwidth. — **done this entry, see below.**
3. Prefill's regression (previous entry) is likely nested-rayon contention,
   not "fusion doesn't work" — suggested a `forward_inference_serial`
   variant for use inside already-parallel contexts (untested, not
   attempted this entry).
4. `down_proj` (21.9% of decode) is parallelism-starved (P=14 vs 48 for
   up/gate) despite equal total block count — same fix as Phase 1's
   training batching, applied to `forward_inference`. Untested.
5. Confirmed `wo`/`w_down` genuinely can't be input-fused (they consume
   `attn_out`/`act`, which don't exist until the thing they'd fuse with
   finishes) — a real data dependency, not an oversight from the earlier
   investigation.
6. Fast-exp: confirmed dead end, stop reinvesting (matches this session's
   own findings).

### Removed the dead `y1s` buffer

`fwd_block`'s `y1` output was only ever needed as scratch to build `z` (via
transpose) within a single block computation — `forward`/`forward_batch`
were allocating and fully writing a `[p*q*b]` (or `[n_tokens*p*q*b]`)
buffer, storing it in `FwdCache`, and nothing ever read it back (confirmed
via `grep -rn "y1s\b"` across the whole repo — zero read sites outside the
write itself). Changed both to use a local per-`pp`-worker scratch `y1`
(same pattern `forward_inference` already used), matching how the value
was actually consumed. `FwdCache` now only holds `zs`.

Verified: all 51 kernel unit tests pass, `gate_c` gradcheck still passes
(max_err ~1e-5), and `train_char`'s 10-step loss trajectory is bit-identical
to before the change (pure memory-layout change, no numerics touched).

Training: 296ms → 285ms/opt-step (~3.7% faster, single-run measurement, not
independently re-verified against noise this entry). Decode/prefill numbers
are unaffected by this change — they use `forward_inference`/
`forward_inference_grouped`, which never allocated `y1s` in the first place;
any fluctuation observed there this entry is ordinary run-to-run noise, not
a consequence of this fix.

**Next:** the KV-cache correctness gap (Opus suggestion #1) is flagged as
higher-priority than any further speed work — decode's 36 tok/s number may
not reflect real generation cost. Otherwise: `forward_inference_serial` for
prefill, or parallelizing `down_proj`'s `forward_inference` over the full
`(pp,qq)` space, are the remaining untested perf levers.

## 2026-07-01 — both remaining perf levers tried, both regressed, reverted

Implemented the two remaining Opus-suggested levers from the previous
entry. Both made things worse, not better, despite sound-sounding
reasoning — a useful reminder (consistent with this session's whole
methodology) that these hypotheses need actual measurement, not just
plausible mechanism stories.

### Lever A: `forward_inference_wide` (parallelize `w_down` over `qq` instead of `pp`)

Implemented as a `q > p` branch that parallelizes over `qq` instead of `pp`
(48 units instead of 14 for `w_down` at 1B scale), collecting per-`qq`
partial output buffers and summing them serially afterward (same
determinism pattern as the Phase 2 training backward work). Added as the
default behavior of `forward_inference` when `q > p`.

**Result: decode got consistently worse** (29 tok/s baseline → 22-25 tok/s
initially, though variance was high — later runs closer to baseline, but
never clearly better, and often worse). Root cause suspected: the `qq`-
parallel path allocates a fresh `Vec<f32>` partial buffer per unit (`q=48`
separate heap allocations) instead of writing into a pre-allocated shared
buffer via `chunks_mut` (what the `pp`-parallel path does with zero extra
allocation) — at this problem size (each unit does only a few hundred FMA
ops), allocator overhead plus the serial merge cost apparently exceeds
what the extra parallel units recover. **Reverted**: pulled the `q>p`
branch out into a separately-named `forward_inference_wide` (not called by
default anywhere), kept `forward_inference` as the original unconditional
`pp`-parallel implementation. The method stays in the codebase, correctness-
tested, in case it's worth revisiting with a scratch-buffer pool instead of
fresh allocations per unit.

### Lever B: `forward_inference_serial` for prefill (avoid nested-rayon contention)

Implemented a fully serial (no rayon) `forward_inference_serial`, wired
into all of `forward_prefill`'s projection calls (which already run inside
the outer `tokens.par_iter()`, 256-way). Hypothesis: inner rayon dispatches
nested inside outer parallelism are pure contention, not useful work, since
the outer parallelism should already be keeping cores busy.

**Result: prefill got consistently worse** (50 tok/s baseline → 34-46
tok/s across repeats). The hypothesis was wrong, or at least incomplete:
apparently the inner parallelism *was* doing useful work even nested inside
the outer 256-way loop — going fully serial per-token made each outer
task heavier without a compensating benefit, since raw compute reduction
from splitting a P=48-block projection across multiple threads outweighs
the nested-dispatch overhead cost at this problem size. **Reverted**: all
`forward_prefill` call sites back to plain `forward_inference`.
`forward_inference_serial` stays in the codebase (correctness-tested
against the parallel version) but isn't used by default anywhere.

### Verification after reverting both

All 53 kernel unit tests pass (5 new this session: `forward_inference_wide
_axis_matches_forward`, `forward_inference_serial_matches_parallel`, plus
the 3 from earlier entries), `gate_c` gradcheck passes (max_err ~1e-5),
`train_char`'s 10-step loss trajectory is bit-identical to every prior
entry. Post-revert decode/prefill: 28-36 tok/s / 44-49 tok/s across 3
repeats — back in the established baseline range (36 tok/s / 50 tok/s).

### What this shows

Two for two: both Opus-suggested perf levers, despite each having a
specific, mechanistically plausible failure mode identified in the
original review (parallelism-granularity starvation for lever A, nested-
dispatch contention for lever B), turned out to be regressions when
actually measured. The QKV/up-gate fusion from two entries ago is the only
decode-side win that's actually held up under paired measurement — everything
else tried since has either been confirmed-inert (fast-exp) or
confirmed-regression (these two). This is the same lesson as the prefill
fusion regression: a plausible mechanism story is a hypothesis to test, not
a result.

### Honest limitations

- Didn't isolate *why* each regression happens beyond a plausible
  explanation (allocation overhead for lever A, useful-nested-parallelism
  for lever B) — neither was directly measured/traced (e.g. no allocator
  profiling, no rayon internal tracing). Could be wrong about the specific
  mechanism even though the empirical regression itself is solid (multiple
  repeats, consistent direction).
- Lever A's variance was notably higher than other measurements this
  session (22-34 tok/s spread on what should be a fairly deterministic
  workload) — worth understanding if pursued further, since high variance
  itself might be a symptom of allocator contention under concurrent load.
- Neither lever was tried at other problem sizes (e.g. lever A at a less
  extreme P/Q ratio, lever B at smaller/larger seq) — possible either
  helps in a different regime even though both failed at the specific
  1B-scale shapes tested.

**Next:** the KV-cache correctness gap (Opus suggestion #1) remains the
highest-priority open item — no further validated perf levers are known at
this point. Current best-known numbers: training ~1730 tok/s (toy config,
fwd+bwd), decode 36 tok/s, prefill 50 tok/s (both 1B-scale, unfused-prefill/
fused-decode).

## 2026-07-01 — real KV-cache decode benchmark: 36 tok/s only holds at ctx=0

Built the benchmark Opus flagged as higher-priority than further speed
work. This is the most significant finding of the session — the headline
decode number the whole session has been reporting turns out to only be
true for the very first generated token.

### What was built

`layer_bench.rs`'s old `forward_decode` self-attends to a single token —
softmax over 1 position is always weight=1.0 by construction, so attention
was structurally free (0.6% of decode time) regardless of any real context.
Added `forward_decode_cached(h, k_cache, v_cache, ctx_len, window)`: real
causal attention over `ctx_len` cached positions (or only the last
`window` if set — sliding-window attention), streamed from actual
`[ctx_len * HIDDEN]` K/V buffers rather than assumed away. Returns
`(output, new_k, new_v)` for the caller to append to the cache.

Two correctness tests before trusting the number: (1) `ctx_len=0` should
reduce exactly to the old self-attend stub's math (softmax-over-1 case) —
confirmed, two independently-written attention implementations agree
exactly; (2) perturbing cache content *outside* the sliding window
shouldn't change windowed-attention output at all — confirmed. Both pass.

Benchmarked using the production config's actual defaults (found in
`src/model/config.rs`, not the toy `profile.rs` defaults used elsewhere
this session): **24 full-attention layers, 72 sliding-window layers
(window=256), out of 96 total** — `full_attn_layers: 24` is the real
default, not the `profile.rs` benchmark harness's `FULL_ATTN=3`.

### Results (96-layer extrapolation, 24 full + 72 windowed[256])

| Context length | Full-attn layer | Windowed layer | 96-layer decode | tok/s |
|---|---|---|---|---|
| 0 (no history) | 282-308 µs | 305-307 µs | 28.7-29.5 ms | 34-36 |
| 512 | 1258-1392 µs | 590-926 µs | 72.7-100.1 ms | 10-14 |
| 2048 | 4093-4099 µs | 730-738 µs | 151.0-151.4 ms | 7 |
| 8192 | 22108-22996 µs | 640-665 µs | 578.5-598.0 ms | **2** |

Consistent across 2 repeats (values above are the observed range). The
sliding-window layers behave exactly as designed — their cost stays flat
(~600-900µs) regardless of context length, since they only ever attend to
the last 256 positions. The **24 full-attention layers are the entire
story**: at ctx=8192 they alone account for ~530ms of the ~580ms total
(≈92%), because full attention cost scales with context length while
window-limited layers don't.

### What this shows

Every decode tok/s number reported earlier this session (29 → 36 tok/s
after the QKV/up-gate fusion) was measured at **ctx=0** — generating the
very first token, with no conversation history. That's not what "decode
tok/s" means in practice; real generation happens *after* a prompt/prior
turns exist. At a realistic mid-conversation context (2048 tokens), decode
is **7 tok/s, not 36** — a 5x difference. At a long context (8192), it's
**2 tok/s** — 18x worse than the headline number. All the fusion/dispatch
optimization work this session (Phase 1, Phase 2, QKV/up-gate fusion) was
real and correctly measured, but it was optimizing a part of decode that
becomes proportionally *irrelevant* once real context exists — the
projections that got 24-40% faster are a shrinking fraction of a growing
total dominated by full attention's O(context) cost.

This reframes the priority question for the whole project: the 24
full-attention layers are the single largest lever for decode tok/s at any
realistic context length, dwarfing every fusion/dispatch fix combined.
Reducing `full_attn_layers` (fewer full-attention layers, more windowed)
would directly and predictably improve this — the current 24/96 split is
inherited from the `config.rs` default, not something this session's
measurements had validated until now.

### Honest limitations

- K/V cache content is synthetic (deterministic sine/cosine patterns, not
  from real generated tokens) — sufficient for a bandwidth/compute cost
  measurement (the numbers don't depend on content, only cache size), but
  doesn't validate output *quality*, only throughput.
- Didn't test intermediate context lengths between 2048 and 8192, or
  beyond 8192 — the scaling isn't perfectly linear in the data collected
  (512→2048 is ~3.3x for 4x context; 2048→8192 is ~5.4x for 4x context),
  and I haven't identified whether that's cache-locality effects (L2/L3
  spill at some threshold), genuine noise, or something else.
- Didn't explore whether `full_attn_layers` could be reduced without
  hurting model quality — that's a model-architecture/quality question,
  not a systems one, and outside what this session's tooling can answer.
- The K/V cache itself isn't reused across the benchmark's `bench_one`
  calls in a way that simulates real incremental generation (each
  measurement uses a freshly-sized, static cache rather than growing it
  one token at a time across many steps) — appropriate for measuring
  steady-state cost at a given context length, not for measuring
  cache-append overhead itself (append cost is O(HIDDEN), trivially cheap,
  not expected to matter, but not directly measured here).

**Next:** this finding changes the priority order for the whole project.
Options: (a) investigate whether `full_attn_layers` can be reduced (a
model-quality tradeoff question, needs domain judgment not just systems
profiling), (b) optimize the full-attention path itself now that it's known
to dominate (e.g. the causal loop in `forward_decode_cached` is a plain
scalar loop, not vectorized — real headroom likely exists there,
unexplored), or (c) accept the current architecture and treat 7 tok/s
(2048 ctx) as the honest number going forward instead of 36.

## TODO (not started) — block-sparse full attention for the 24 full-attn layers

All of Opus's original suggestions are now exhausted: KV-cache gap
investigated (found the ctx=0-only-measurement issue above), `y1s` removed,
`forward_inference_serial` and the `down_proj` wide-axis fix both tried and
reverted (regressions), `wo`/`w_down` fusion confirmed structurally
impossible (real data dependency), fast-exp confirmed dead end. This is the
next idea, refined into something concrete enough to pick up later, but
**deliberately not started** — flagging why below.

**The idea:** the 24 full-attention layers do exact O(context) dense
attention — every query attends to every past position. Sliding-window
attention (the other 72 layers) is already a *content-blind* pruning
strategy: always attend to exactly the last 256 positions, nothing else,
regardless of relevance. The natural next step for the full-attention
layers is a *content-aware* pruning strategy: cheaply score which blocks of
past keys are likely to matter for a given query *before* paying for exact
attention on them — conceptually the same two-phase structure as AABB
broad-phase collision detection (cheap bounding-test to rule out most
candidates, expensive precise test only on survivors). This is an
established technique family (block-sparse attention, routing transformers,
LSH-based attention like Reformer), not a novel idea — the point of writing
it down here is to scope it for *this* codebase specifically, not to invent
a new algorithm.

**Why it's the right next target, mechanically:** the previous entry showed
the 24 full-attention layers are ~92% of decode cost at ctx=8192. Anything
that reduces their effective attention set from O(context) toward something
closer to O(window) — even approximately — has more headroom than any
dispatch/fusion fix touched this session, all of which affected the
projection cost that's now known to be a shrinking fraction of the total.

**Why it's not started, deliberately:** every optimization this session
(Phase 1, Phase 2, QKV/up-gate fusion, the `y1s` removal) was *exact* —
verified bit-identical or gradient-checked against a known-correct
reference. Block-sparse attention is *approximate* by construction — it
changes what the model actually attends to, which changes model quality,
not just speed. That can't be validated with the benchmarking/testing tools
used all session (timing + numerical equality checks); it needs an actual
quality eval (perplexity on held-out data, same methodology as the
dense-baseline comparison entry from earlier), which is a different kind of
work than everything else in this log.

**What "picking this up" would concretely involve**, when it's time:
1. A cheap block-relevance scoring function (e.g. a low-dimensional summary
   per key-block — mean-pooled key vector, or a small learned/fixed
   projection — compared against the query to estimate attention mass
   before computing it exactly).
2. A decision rule for which blocks to keep (top-k blocks by score? a
   score threshold? fixed budget matching the sliding-window layers'
   O(window) cost for a fair comparison?).
3. Validation split into two independent questions that shouldn't be
   conflated: (a) does it actually speed up decode at long context
   (systems question, benchmarkable the same way as this session's other
   work), and (b) does it preserve output quality closely enough to be
   worth the complexity (research question, needs perplexity/eval work,
   not benchmarking).
4. A honest baseline: how much quality, if any, does the *existing*
   sliding-window design already sacrifice vs. full attention on the 72
   windowed layers? That's presumably already an accepted tradeoff in the
   architecture, and knowing its magnitude would calibrate how much
   additional approximation on the 24 full-attention layers might be
   tolerable.

## Side note — paper found: "Prism Transformer: Progressive Head Schedules" (arXiv 2606.27449)

Surfaced mid-session, not investigated beyond a single low-confidence
WebFetch summary (small-model summary of a PDF, unverified against the
source — treat the description below as provisional, not fact-checked).

Per that summary: reduces the number of *active attention heads*
progressively by layer depth (fewer heads in deeper layers), decided at
training time, and deliberately avoids content-aware/dynamic sparse
indexing in favor of static, hardware-regular memory access.

**Relation to the block-sparse TODO above:** not directly adaptable — it's
a different axis (whole-head pruning by depth vs. which-keys-within-a-head
pruning by content) and philosophically closer to what sliding-window
already does (static, content-blind) than to the content-aware pruning
proposed above. Possible orthogonal lever worth a skeptical look later:
production config already has `n_kv_heads=2` (aggressive GQA), but
`n_q_heads=14` is unexamined — if query-head redundancy does grow with
depth the way this paper claims, progressively dropping query heads in
deeper layers would be a separate, static (no per-token eval-of-quality
risk beyond the head-pruning itself) lever alongside block-sparse
attention. Not started; source needs verification before any of this is
taken as ground truth.

## TODO (not started) — matmul-free kernels/attention, per "Scalable MatMul-free Language Modeling" (arXiv 2406.02528)

Surfaced mid-session. Higher confidence than the Prism note above — this is
a paper I recognize independently of the WebFetch summary (Zhu et al.,
June 2024), not just a single-source unverified fetch.

**Core method, two independent pieces:**
1. Ternary weight quantization (BitNet-1.58-style, weights in `{-1, 0, +1}`)
   — every matmul becomes pure add/subtract/skip, no multiplication.
2. MLGRU — replaces self-attention with a matmul-free recurrent linear
   gating mixer; token-mixing becomes O(1) per token instead of O(context).

**Relevance, two distinct angles:**

- **Kernel-level, on `SharedMonarchProj` (the kernel just wired into
  `AttnProj` this session):** `fwd_block`'s innermost loop is
  `eff[e] += a * atom[e]` (both stages) — a multiply-accumulate over the
  coefficient `a`. If `a1`/`a2` were ternary instead of float, this
  collapses to conditional add/subtract/skip, no multiply/FMA needed.
  This layers *on top of* the existing block-structured compression — not
  a competing architecture, an orthogonal precision/quantization axis.
- **Architecture-level, on the block-sparse-attention TODO above:** MLGRU
  doesn't prune which keys get attended to — it removes the O(context)
  attention cost entirely, replacing it with a constant-cost recurrent
  update. That would obsolete the "24 full-attention layers are ~92% of
  decode cost at ctx=8192" problem rather than mitigate it (as block-sparse
  pruning would). A more radical alternative worth weighing against
  block-sparse pruning, not a refinement of it.

**Why not started:** both pieces are architecture changes with real
quality-vs-speed tradeoffs (same category as the block-sparse TODO above —
needs eval methodology, not just benchmarking), and MLGRU in particular
would replace a whole model component (self-attention), not just optimize
an existing one. Flagging for later comparison against block-sparse
pruning once eval infra exists, not implementing now.

## 2026-07-02 — The AttnProj swap (BasisMatmul → SharedMonarchProj), never logged until now

A structural gap flagged by a Fable 5 review (below): the real model's
attention projections (`AttnProj`, in `src/model/attn_proj.rs`) were switched
from `BasisMatmul` (FFT-based, single global complex dictionary shared across
*every* projection in the model) to a new `SharedMonarchProj` (Monarch-style,
same block math as `SharedMonarchMatmul` — the kernel all of this session's
earlier optimization work targeted — but with the atom dictionary owned
externally instead of per-instance, so it can still be shared model-wide).
This finally connects the fast, parallelized, gradchecked Monarch kernel to
the actual model being trained; before this, every earlier speedup in this
log (batched forward, parallel backward, QKV fusion) lived only in
`train_char.rs`'s toy proof-of-concept, never in `src/model/*`.

Mechanically: `SharedMonarchMatmul`'s block math (`fwd_block`, the AVX2
backward kernels) already took `d1`/`d2` as plain slice parameters rather
than reading `self.d1`/`self.d2` directly, so `SharedMonarchProj` reuses that
math unchanged — only the wrapper (ownership, dispatch, dict threading) is
new code. `Model` gained `mono_d1`/`mono_d2` (shared real atom dictionary,
alongside the FFN's existing shared complex `dict`), and `LayerForward`
gained per-projection `FwdCache` fields (`wq_fc`/`wk_fc`/`wv_fc`/`wo_fc`) so
`backward` doesn't need to recompute Monarch's stage-1 intermediate from
scratch. `Ffn` (up/gate/down) was deliberately left on `BasisMatmul` — its
MoE-style block routing (`n_active` of `ffn_dim/block` blocks per token)
wasn't analyzed against Monarch's tiling in the same sitting.

Verified via: a new `shared_monarch_proj_matches_shared_monarch_matmul` test
(bit-identical output/grads vs. the known-good per-instance kernel, same
seed) and a new `shared_monarch_proj_gradcheck` (finite-difference, covering
the `dd1`/`dd2` dictionary-gradient path that `SharedMonarchMatmul` never
exercised). All 85 lib tests pass.

**What was missing, per the Fable review below**: no before/after throughput
or convergence benchmark of the swap itself was ever run. Correctness was
verified; whether it's actually faster on the real model was not measured at
the time. That gap is still open — everything below this entry chases a
different, related question (sliding-window's quality cost) that surfaced
before the throughput question got answered.

## 2026-07-02 — Fable 5 kernel/architecture review

Asked a fresh Fable 5 agent (no prior context, briefed from this log + the
code) to review the current kernel/architecture state and help sequence the
open TODOs. Three findings acted on immediately:

1. **Reintroduced init-scaling bug.** The 2026-06-30 entry's derived
   `s_atom = 3·m^-½·Q^-¼` (fixing a real convergence problem, verified
   empirically) only ever landed in `SharedMonarchMatmul::new` — the new
   `init_shared_atoms`/`SharedMonarchProj::new` path the real model actually
   uses dropped the `Q^-¼` term entirely. Since the dictionary is now shared
   across projections that may have different `q`, the compensation can't
   live on the atoms anymore (no single `q` to scale by) — it has to move to
   the per-projection coefficients instead. Fixed: `SharedMonarchProj::new`'s
   `s_coeff` now carries `q^-¼`; `init_shared_atoms`'s `s_atom` stays
   `3·m^-½` (Q-independent, matches its "fix #1" precursor from 06-30).
   Atom/coefficient scale enter the composed variance as a product
   (`Var(E1 entry) = nd × Var(a) × Var(D)`), so relocating the exponent
   preserves the same solved target.
2. **Phase 2's parallel-backward win was never ported to the real model.**
   `TransformerLayer::backward`'s three per-token loops (FFN, `wo`, QKV) were
   still fully serial — exactly the pre-Phase-2 shape from `train_char.rs`.
   Ported the same collect-then-sequential-merge pattern (parallel `map`
   into an order-preserving `Vec`, then a plain sequential accumulation
   loop — deterministic by construction, no `fold`/`reduce` reordering).
   Verified via the existing `layer_backward_d_hidden_gradchecks` finite-
   difference test, plus a manual re-run under `RAYON_NUM_THREADS=1` and
   `=4` with no failures.
3. **Ternary quantization's "eliminates multiply" pitch is GPU/ASIC
   reasoning, doesn't hold on this target.** On AVX2, FMA costs the same as
   a plain add — ternary coefficients wouldn't save compute in `fwd_block`'s
   inner loop, only bandwidth/footprint (2-bit vs 32-bit storage). Correction
   folded into the matmul-free TODO above (not re-quoted here); the honest
   framing is a bandwidth experiment, not a "remove multiplies" one.

Also flagged, not yet acted on: `Ffn`'s MoE routing composes with Monarch's
tiling as an *exact* zero-skip (same trick `BasisMatmul`'s
`forward_rows`/`forward_cols` already uses) — the "blocked, unanalyzed"
framing in the matmul-free TODO's item (a) was overly pessimistic; the FFN
swap is a re-parameterization + quality A/B, not blocked on a routing
compatibility question. Left for a future session.

Also added: `TrainConfig`-independent `log_every` parameter to
`train()` (`src/train/loop.rs`) — prints a lightweight progress line every
`n` steps, decoupled from `checkpoint`'s (much less frequent) save cadence,
so a long run doesn't go silent for tens of minutes between checkpoints the
way earlier runs in this log did.

## 2026-07-02 — Sliding-window quality-cost baseline: the swap-based eval was confounded; fixed with two separately trained models

The block-sparse-attention TODO's prerequisite (how much quality does
sliding-window already cost vs. full attention) finally got a real
measurement — after finding the first methodology was wrong.

**First attempt (confounded).** `train_small`'s `--eval` mode took one
trained checkpoint and swapped `full_attn_layers` post-hoc (same weights,
reconstructed with a different config) to compare sliding-as-trained against
a forced-full-attention variant. Result: sliding-window's "cost" came back
**negative** (-0.27 nats — sliding looked *better*). This is not a real
finding: `window` isn't a learned parameter, so forcing full attention onto
weights trained under sliding-window puts every downstream layer (norms,
FFN, later attention layers) out of the distribution it was actually trained
on. The swap measures brittleness to an inference-time attention-pattern
change, not the quality cost of the design choice itself.

**Fix**: train two models completely independently (same seed, same
architecture, same everything except `full_attn_layers`), evaluate each on
the same held-out set with its own weights, no post-hoc swapping. Also
switched to a byte-level-vocab "low LOD" config for this
(`train_small_lod`, new binary — see below) purely for iteration speed;
`train_small` (real GPT-2 vocab) is unaffected and still exists as the
closer proxy to the production tokenization scheme.

**Result**, two from-scratch 3000-step runs, 12 layers/hidden=256/window=64,
byte-level vocab, identical everything except `full_attn_layers`:

```
sliding-window (full_attn_layers=3, separately trained):   1.8884 nats
all-full-attention (full_attn_layers=12, separately trained): 1.8858 nats
quality cost of sliding-window:                              0.0026 nats
```

Essentially zero — and unlike the very first (pre-session) 0.0000 result
(which was plausibly a kernel/measurement artifact at an even more
undertrained scale), this one is from a clean, non-confounded comparison:
both models converged normally to nearly identical train loss (1.6777 vs
1.6688). The 0.0026 nats gap is smaller than this project's own measured
eval noise floor (~0.013 nats at a comparable eval-window count elsewhere in
this log), so it reads as noise, not signal.

**What this does and doesn't tell us.** Consistent with (not proof of) the
hypothesis that a model this small/undertrained hasn't learned to need
long-range context yet, so restricting the window costs nothing measurable.
It does **not** answer the question for the real production config (96
layers, window=256, much larger scale/budget) — whether long-range context
matters presumably *does* scale with model capacity and training budget, and
this toy setup structurally can't speak to that. The block-sparse-attention
TODO's quality-budget question remains open at the scale that actually
matters; this entry closes it out only for the toy scale, with a
methodology now worth reusing once a larger checkpoint exists.

**Byproduct**: `train_small_lod` (`src/bin/train_small_lod.rs`) now exists
as a permanent low-LOD sibling to `train_small` — same architecture, local
byte-level corpus (`data/input.txt`, vocab 128) instead of the GPT-2 BPE
fetch (vocab 50257). Motivation: at this toy model size, the GPT-2 vocab
made the tied embedding/LM-head ~99% of total params and (fully dense,
uncompressed) plausibly most of the step time — dominating the very kernel
work this project exists to speed up, and swamping any signal from the
Monarch/parallel-backward changes above. Confirmed empirically: byte-vocab
3000-step run completed in 49m21s (0.99s/step avg) vs. the GPT-2-vocab
config's one completed data point (~9.25s/step, from before the AttnProj
swap) — roughly 9x faster, consistent with removing the LM head as the
dominant cost. Explicitly not a stand-in for `train_small`'s quality
conclusions — character-level LMs have a different relationship to context
length than subword LMs, so results here are informative about kernel
iteration speed and about *this* toy config specifically, not directly
transferable to what the real BPE-tokenized model would show.

## 2026-07-02 — Kernel throughput benchmark scare, resolved: SharedMonarch vs BasisMatmul was never fairly compared until now

Chasing item 1 from the Fable review (never benchmarked whether the AttnProj
swap actually helped) via `gate_c.rs`'s existing `bench()` at
`train_small_lod`'s actual projection shapes (`256x256`, `dict_k=8`)
produced an alarming result: SharedMonarch **6x slower** than BasisMatmul at
that scale, despite winning 1.3-1.9x at production scale (`896x896`,
`896x3072`) in `bench()`'s original, long-standing calls. A grid-size
crossover search (square P×Q grids from 256 up to 896, `nd=k=8` throughout)
never found a crossover — Monarch lost at *every* size tested, including
896x896, contradicting the original "production scale wins" result.

**Root cause (found via Opus, asked specifically to verify the parameter
accounting rather than trust either result at face value):** `nd=k` is not
an equal-capacity comparison. `SharedMonarchMatmul`/`SharedMonarchProj`
spend `2·m·nd` coefficients per block-pair (separate `a1`/`a2`, each
`m·nd`); `BasisMatmul` spends only `K`. The original "production scale
wins" benchmark used `nd=8, K=32` — Monarch actually had *more* capacity
there (`2·8·8=128` vs `32`), so that comparison wasn't measuring what it
looked like it was measuring either. The `nd=k` sweep that seemed to
contradict it was, symmetrically, starving BasisMatmul relative to Monarch's
FLOP cost, not testing equal capacity.

The FLOP cost ratio is `m·nd/K` (Monarch: rebuilds an `m×m` matrix from `nd`
atoms per row-group, twice; BasisMatmul: `K`-atom axpy per block-pair plus
amortized FFT/IFFT) — this ratio alone correctly retrodicts every prior
data point, including the previously-puzzling fact that the "6x slower"
result was ~constant across `nd=k=8/16/32` (the ratio `m` is independent of
the shared `nd=k` value, which a naive "cost scales with coefficient count"
model would not predict). True equal capacity requires `K = 2·m·nd = 16·nd`
(at `m=8`); `K=64` is the equal-*FLOP* point instead (BasisMatmul
unpadded — worth checking separately since equal-*params* can pad
BasisMatmul past its expressibility ceiling, `monarch.rs` itself notes
Monarch needs `nd≥8` for full rank).

A secondary bug in the bench harness itself was also fixed while at it:
it timed `SharedMonarchMatmul::forward` (which allocates and writes the
backward cache) against `BasisMatmul::forward` (which writes no cache) —
an unfair tax on Monarch for work the forward-only comparison wasn't asking
for. Switched to `forward_inference`.

**Result, true equal capacity, `nd=8`, both threading regimes** (ruling out
"it's just more cores" by checking `RAYON_NUM_THREADS=1` too — `BasisMatmul`
was confirmed single-threaded, no rayon anywhere in it, so raw multi-thread
numbers alone would have been confounded):

```
K=128 (equal param count):
  896x3072: full-threads 8.1x faster  |  single-thread 4.4x faster
  896x896:  full-threads 6.5x faster  |  single-thread 3.9x faster
K=64 (equal FLOPs, BasisMatmul unpadded — the harder/fairer test):
  896x3072: full-threads 3.2x faster  |  single-thread 1.8x faster
```

Monarch wins decisively, in every configuration, even single-threaded. The
scare resolves entirely: there was never a grid-size crossover, only an
unmatched-capacity artifact in every prior comparison (in both directions).
The AttnProj swap (BasisMatmul → SharedMonarchProj) is confirmed as a real,
substantial kernel-level win, not scale-dependent or risky the way the
raw `train_small_lod`-scale numbers briefly suggested. Item 1 from the
Fable review (benchmark the swap) is now genuinely closed, at the kernel
level — a full-model wall-clock A/B (old BasisMatmul-integrated model vs.
current) was not run and would require reverting the model-level
integration, judged not worth doing given the kernel-level result is this
clear.

---

## 2026-07-03 — The full-model A/B *was* run after all: real training got slower, not faster

The previous entry judged a full-model wall-clock A/B "not worth doing." It
turned out to be necessary: `train_small` (real GPT-2-vocab config,
`hidden=256, ffn_dim=768, block=64, dict_k=8`) finished its post-swap
3000-step run cleanly (held-out CE 5.4773 nats — the first honest number
post-swap; `--eval`'s confounded post-hoc-full-attention-swap comparison
was also deleted from `train_small.rs` in this session, matching the fix
already applied to `train_small_lod`).

Prompted by "do you mind booting Fable" once the accessibility framing
came up ("Monarch should be ≤ BasisMatmul at *any* scale, not just
production scale — that's the pitch"), a real old-vs-new comparison was
run via a throwaway synthetic bench (`git worktree` checkout of the
pre-swap commit `6ae62cb`, a scratch `bench_efficiency.rs` doing N
forward+backward+AdaFactor steps on random token ids — no corpus fetch
needed, never committed):

```
train_small config (vocab=50257, real GPT-2 vocab):
  old (BasisMatmul):        2650.8 ms/step, 51.72 MB checkpoint
  new (SharedMonarchProj):  2826.3 ms/step, 52.79 MB checkpoint   (+6.6% slower)

train_small_lod config (vocab=128, byte-level — embedding no longer dominates):
  old (BasisMatmul):         306.5 ms/step,  0.39 MB checkpoint
  new (SharedMonarchProj):   438.4 ms/step,  1.46 MB checkpoint   (+43% slower, 3.7x bigger)
```

Not a contradiction of the previous entry's kernel-level result — a scale
mismatch. `train_small`'s actual dims (`hidden=256, ffn_dim=768`) sit
*below* the crossover point found in `gate_c.rs`'s own grid search
(roughly `768-896`). The isolated equal-capacity wins (8.1x/3.2x etc.) were
all measured at production-scale shapes (`896x896`, `896x3072`); nothing
in this project has trained at that scale yet. `train_small` is currently
the wrong regime to expect a win in — a "sports car on a driveway" situation,
not a broken swap.

### Fable 5 consultation: is the low-x loss fixable, or a hard floor?

Asked Fable to verify two hypotheses from reading the code directly
(`monarch.rs`, `fft.rs`, `gate_c.rs`) and give a verdict on whether
"Monarch ≤ BasisMatmul at every scale" — the actual accessibility goal, not
just "wins at 1B scale" — is achievable without sacrificing the
parameter-efficiency the whole scheme exists for.

**Verified, and a bigger bug found than either hypothesis:** `fwd_block_avx2`/
`bwd_block_avx2` really do pay an ~8:1 reconstruction:compute FLOP ratio at
`nd=8` (confirms the "inherent atom-reconstruction tax" hypothesis) — but
the actual defect isn't that the tax exists, it's *where* it's paid.
`SharedMonarchProj::forward_batch` and the per-token `backward` calls in
`layer.rs`/`ffn.rs` reconstruct the effective `eff` blocks **from scratch
for every token**, even though `eff` depends only on weights, which are
fixed for an entire training step (or forever, at inference). This is
loop-ordering, not inherent math — training should reconstruct each block
once per step and reuse it across all `t_len` tokens; it currently doesn't.
Rayon dispatch overhead (the original hypothesis (b)) is real but
secondary — `forward_batch` already amortizes dispatch across
`n_tokens × P` units, so it's a minor contributor to the real 43%
regression, confined mostly to the isolated single-call `gate_c.rs`
benchmark.

A second confound was flagged in the same pass: the toy-scale `gate_c.rs`
benchmark compared `nd=8` against `K=8` "to isolate the kernel" — but per
block-pair that hands Monarch `2·m·nd=128` coefficients against
BasisMatmul's `8`, a 16x capacity difference, not an isolated kernel
comparison at all.

**Verdict:** no fundamental floor. Both kernels pay an unhoisted
per-call reconstruction tax (BasisMatmul's is FFT/IFFT + `K`-atom axpy,
same defect, never hoisted either); at matched capacity the floors are the
same order. Training amortizes trivially once fixed (reconstruct once,
reuse across the whole batch); inference can precompute the effective
blocks once at model load and never reconstruct again, which would make
Monarch strictly cheaper than BasisMatmul at every shape. Explicit
recommendation: **do not reintroduce BasisMatmul as a scale-dependent
hybrid fallback** — once the reconstruction is hoisted, there's no regime
left where it wins at equal capacity, and reintroducing it has real
complexity cost (it was fully deleted from the live model this session).

Prioritized fix list from the review (highest impact first): (1) reorder
`forward_batch`/backward to reconstruct each block once, loop tokens
inside — the actual fix for the training regression, not yet started;
(2) sequential fallback below a small-work threshold — cheap, done (see
below); (3) cache reconstructed blocks at inference load time — not yet
started, blocked on there being an inference/decode path at all; (4) a
proper micro-GEMM restructure of `matvec8` — smaller win, not started;
(5) fix `gate_c.rs`'s toy-scale benchmark to use equal capacity — done
(see below).

### Two cheap fixes landed today

**`PARALLEL_THRESHOLD` sequential fallback** (`monarch.rs`): both
`SharedMonarchProj` and `SharedMonarchMatmul` now skip rayon and run
inline when the number of parallel units (`p`, or `n_tokens*p` for
`forward_batch`) is below 8 — rayon's fork-join wake cost was dominating
actual per-unit work at toy scale. Reuses the existing
`forward_inference_serial` pattern already established for `prefill`.
11/11 monarch tests still pass.

**`gate_c.rs` equal-capacity toy-scale benchmark added** (`K=128` at
`train_small_lod`'s exact shapes, alongside the existing `K=8`
capacity-mismatched row, so both are visible together).

**Measured effect — isolated kernel benchmark, clear win:**

```
256x256, nd=k=8 (same-nd/k, capacity-mismatched):  fwd 20.94µs → 5.28µs (4x), speedup 0.2x → 0.5x
256x256, K=128 (true equal capacity):               speedup 1.6x → 4.3x
```

**Measured effect — real training throughput (train_small_lod config,
same synthetic bench as above): no change (438.4ms → 450.1ms, within
noise).** Root cause: `forward_batch`'s parallel unit count is
`n_tokens × P` — at training scale (`seq_len=256, P=4` → 1024 units) this
is always far above the threshold, so the sequential fallback never
triggers during real training. It only helps single-token-scale calls
(future autoregressive decode), which don't exist as a code path yet. This
is a clean negative result, not a wasted effort: it directly confirms
Fable's diagnosis that rayon dispatch was never the dominant cost in the
training regression — the still-unfixed per-token reconstruction
redundancy (fix #1 above) is.

**Status:** the accessibility goal ("Monarch ≤ BasisMatmul at any scale")
looks achievable per Fable's analysis, but is not yet delivered — the
high-impact fix (reconstruction-hoisting across the batch, in both
`forward_batch` and a new batched `backward`) has not been started. This
is now the clear next item, ahead of block-sparse attention and the
ternary/matmul-free TODOs.

---

## 2026-07-03 — Reconstruction hoisting, take two: parallelize over tokens, not blocks

Implemented the fix flagged above (`expand_block`/`apply_block`/
`backward_block_hoisted` primitives in `monarch.rs`, splitting weight
reconstruction from per-token application), gradcheck-verified against
summed independent per-token `backward()` calls (new test:
`backward_batch_matches_summed_looped_backward`, 12/12 monarch tests
pass) — but the first version shipped with a real bug, caught before it
reached `layer.rs`... and then again after it did.

**First version parallelized `forward_batch`/`backward_batch` over `P`
(the block-row count), matching the file's existing `forward`/
`forward_inference` convention.** Wrong choice for this problem: `P` is
small (4 at `train_small_lod`'s shapes, low tens even at production
scale), while `n_tokens` (the sequence length, 128-256+) is the axis with
real work to spread across 12 cores. Wired into `AttnProj`/`layer.rs`
(new `AttnProj::backward_batch`, replacing the per-token collect+merge
loop for wq/wk/wv/wo — 89/89 full test suite passes, including the
whole-model gradcheck tests, so correctness was never in question) — real
training throughput got *worse* (438ms baseline → 698-734ms/step,
repeated). A focused isolated-kernel bench pinpointed why: at `P=4` (below
`PARALLEL_THRESHOLD=8`), `backward_batch` always took the sequential
branch — losing the old per-token loop's real 12-core parallelism over
256 tokens entirely, for a hoisting saving that turned out to be much
smaller than assumed (reconstruction is *not* backward's dominant cost —
the per-token gradient-accumulation math, `dot8`+`axpy8` over `m·nd`
pairs, is; isolated looped-vs-batched kernel bench at matched work showed
only **1.00x**, i.e. no win at all from hoisting alone at this shape).

**Fix: decouple hoisting from parallelization axis entirely.** New
`expand_all_blocks` reconstructs every `(pp,qq)` block once, unconditionally
(cheap — `P×Q` is always small), before either forward or backward touches
a single token. Application/gradient computation then always parallelizes
over **tokens** — `forward_batch` directly via `par_chunks_mut` (each
token's output is exclusive, no merge needed); `backward_batch` via
token-*chunks* sized to `rayon::current_num_threads()` (not one partial
per token — every token touches every block, so per-token partials would
mean `n_tokens` full-sized `da1`/`da2` copies; chunking to ~12 groups
keeps that bounded, each chunk still processes its tokens sequentially,
accumulating locally, merged at the end — same collect+merge shape used
elsewhere in this codebase).

**Result, isolated kernel bench (`P=4,Q=4,m=8,nd=8`, `n_tokens=256`,
`train_small_lod`'s actual wq/wo shape):**

```
looped (old, per-token):  45.4µs/token
batched (new):             10.2µs/token
speedup: 4.45x
```

**Result, real end-to-end training throughput (same synthetic bench,
byte-vocab config): noisy, 387-464ms/step across 5 runs, no longer a
regression (was 698-734ms with the broken P-parallel version) but also
not a decisive win over the 438-450ms pre-this-fix baseline.** Consistent
with expectation, not a failure: `Ffn`'s backward (up/gate/down, three
larger projections — `768×256`/`256×768` vs AttnProj's `256×256`/`64×256`
— plus MoE routing) is still fully unhoisted, per-token, untouched by this
pass. At this toy scale FFN is the larger cost center; AttnProj's real,
verified 4.45x win is now a smaller fraction of a step dominated by
something this pass didn't touch. Extending the same hoist to `Ffn` (open
question: how routing interacts with hoisting when different tokens
activate different blocks — flagged as the "trickier part" before this
pass started) is the next item, and is where a real full-step win should
actually show up, if the diagnosis holds.

**Status:** `AttnProj`'s reconstruction-hoisting is done, tested (89/89 full
suite + new kernel-level backward-batch test), and wired into
`layer.rs`. `Ffn` is not yet touched. The accessibility goal ("Monarch ≤
BasisMatmul at any scale") is not yet demonstrated end-to-end — closer
than before, but still gated on the FFN half of this same fix.

---

## 2026-07-03 — Ffn gets the same hoist: real batching for the routed projections

Extended the reconstruction-hoisting to `Ffn`, which — unlike `AttnProj` —
had *zero* real batching before this: `select_batch`/`compute_batch` existed
but were pure per-token loops calling the single-token `select`/`compute`
(confirmed via grep: neither was called from `layer.rs`, which ran its own
per-token `select`+`compute` loop directly).

The wrinkle this session's earlier "trickier part" flag anticipated: each
token routes to a different `n_active`-of-`M` block subset, so there's no
shared row/col-selection axis the way dense `AttnProj` had. Resolution:
reconstruct **every** block densely regardless of routing (via the existing
`expand_all_blocks`) rather than tracking a per-batch active set — at any
real batch size, the union of blocks touched across tokens tends to cover
most/all of `M` anyway (`train_small_lod`: `n_active=3` of `M=12`, `t_len=256`
tokens — the union is essentially guaranteed to be complete), so this is
both simpler and not meaningfully more work than a "smart" union-only
reconstruction would be. Added four new `SharedMonarchProj` methods —
`forward_rows_batch`/`forward_cols_batch`/`backward_rows_batch`/
`backward_cols_batch` — mirroring the dense `forward_batch`/`backward_batch`
(same token-chunked-parallelism design from the fix earlier today), but
restricted per token to that token's own routed subset when applying/
accumulating. New test `routed_batch_matches_looped_forward_rows_and_cols`
verifies against per-token loops with genuinely different routing per token
(13/13 monarch tests pass).

Wired through `Ffn::compute_batch`/new `Ffn::backward_batch` (real batched
implementations, replacing the old loop-only stubs), with a new
`FfnForwardBatch`/`FfnGradsBatch` pair mirroring the per-token
`FfnForward`/`FfnGrads`. New test `compute_batch_matches_looped_compute_and_backward`
(7/7 ffn tests pass). Wired into `layer.rs`'s FFN sub-block: the per-token
`select`+`compute` loop and the per-token `backward`-in-a-`par_iter` block
are both replaced with single batched calls; `norm::backward` (nonlinear,
genuinely per-token) stays in a collect+merge loop, same pattern as the
`AttnProj` wiring earlier today. Full suite: 91/91 pass, including the
layer- and model-level gradcheck tests exercising the FFN routing path
end-to-end.

**Result, real end-to-end training throughput** (same synthetic
byte-vocab bench as the AttnProj-only measurement): **408-420ms/step**,
3 runs — tighter (lower variance: was 387-464ms with AttnProj hoisted
alone) and modestly faster on average, but not a dramatic win, and still
short of the pre-swap BasisMatmul baseline (306.5ms/step) at this
specific toy scale. Consistent with the very first finding this project
made about Monarch vs. BasisMatmul (`gate_c.rs`'s own crossover search):
`train_small_lod`'s dimensions sit below the point where Monarch wins at
all, hoisting or not — reconstruction-hoisting removes a real, measured
inefficiency (the per-token reconstruction bug), it does not change which
side of the crossover a given shape sits on. The accessibility goal
("Monarch ≤ BasisMatmul at *any* scale", not just production scale) needs
one more thing this pass didn't touch: the crossover point itself is a
function of block size `m` (Fable's option #2 from two entries back,
"increase block size relative to matrix size at low x") — reconstruction-
hoisting fixes the *implementation* tax, not the *inherent* per-block
overhead that determines where the crossover sits.

**Status:** both `AttnProj` and `Ffn` are now real, batched, and hoisted
(no known un-hoisted per-token reconstruction left in the training path).
This closes the loop-ordering bug Fable found. What's left for the
accessibility goal is a genuinely different lever — tuning `m` (block
size) to move the crossover point itself — not more hoisting.

---

## 2026-07-03 — Opus review: block-size tuning was the wrong lever; the "crossover" was another capacity-mismatch artifact

Asked Opus to investigate whether tuning `m` (block size) could push the
small-scale crossover down, per the previous entry's framing (bigger `m` →
fewer, bigger blocks → less relative reconstruction overhead at small
matrix sizes). Two corrections to that framing, both verified against the
actual code and prior log entries, not just asserted:

1. **The reconstruction:compute FLOP ratio is `nd`, flat in `m`** — my
   "~nd/m" was an arithmetic slip. Reconstructing one `m×m` eff-block from
   `nd` atoms costs `nd·m²` MACs against a `m²`-MAC matvec; the ratio is
   `nd`, independent of `m`. Block size doesn't touch this at all.
2. **`nd≥8` is not a rank requirement.** The code comment and `gate_c.rs`
   both say "for full rank," but the 2026-06-27 log entry (verified
   directly, not just cited) shows the effective map is full-rank at
   *every* `nd` down to 1 — the two-stage-plus-permutation structure
   produces dense rank structurally. `nd≥8` is an empirically-observed
   *optimization/trainability* threshold (same-family overfit solved-rate:
   `nd=2` 9/12, `nd=4` 2/12 — a dead spot — `nd=8` 12/12), measured only at
   `m=8`, never shown to scale with `m`.

Bigger `m` also actively hurts the small-matrix regime this is supposed to
help (collapses the P×Q grid — a 256×256 matrix at `m=16` gives `P=Q=1`,
destroying the block-sharing structure that makes Monarch cheap in the
first place), and any `m≠8` loses the AVX2 fast path entirely (hardcoded
`M=8` in `fwd_block_avx2`/`bwd_block_avx2`/etc.) — so testing other `m`
values would require writing new AVX2 kernels for a change that, per the
FLOP analysis, doesn't even address the thing it was proposed to fix.
**Verdict: drop block-size tuning as a lever.**

The more consequential finding: **`gate_c.rs`'s crossover-search benchmark
never timed `BasisMatmul::backward`** — the printed "speedup" ratio was
forward-only (`basis_us / mon_us`, line ~140), while Monarch's backward
carries a real, structural `nd`-scaled per-token gradient-accumulation cost
(the `dot8`+`axpy8` loops over `m·nd` pairs in `bwd_block_avx2`) that
forward-only reconstruction-hoisting can't touch (it's inherently per-token
— depends on that token's own `x`/`z`/`dout`). The reported 768-896
crossover was therefore measuring the wrong thing for a training-cost
question.

### Re-measured, with backward included

Added `BasisMatmul::backward` timing to `gate_c.rs::bench()` alongside the
existing forward timing, reporting both a forward-only and a fwd+bwd
speedup ratio.

**Capacity-mismatched `nd=k=8` (the original crossover-search setting):**
no crossover found in the tested range at all — flat at **0.26-0.31x**
(3-4x *slower*) from `384×384` through `896×896`. The apparent
forward-only "improvement toward parity" (0.4x → 0.8x across that range)
disappears once backward is included — it was riding on top of the same
capacity mismatch (Monarch getting 16x more per-block params than K=8)
this session already diagnosed once before (2026-07-02, "kernel throughput
benchmark scare" entry), recurring here in a spot nobody had re-checked.

**True equal capacity (`K=2·m·nd=128`) — the fair comparison, backward
included:**

```
                    fwd-only    fwd+bwd
256x256 (toy)         3.55x       2.73x
64x256  (toy)         3.69x       3.07x
896x896 (production)  6.33x       2.75x
896x3072 (production) 9.46x       2.80x
```

Monarch wins at **every** shape tested, toy scale included, once capacity
is compared fairly — margins compressed substantially from the
forward-only numbers (backward's `nd`-tax is real, as predicted), but
never flip. **There is no real crossover in the tested range; the
appearance of one was entirely a capacity-accounting artifact, the same
class of bug as before.** The accessibility goal ("Monarch ≤ BasisMatmul
at any scale") is already met once measured correctly — no hybrid dispatch
or block-size work needed.

### Frozen-dictionary experiment (Opus's other recommendation)

Opus's alternative lever, if Monarch itself needed to be faster at small
scale: freeze the shared atom dictionary (learn coefficients only), which
eliminates the `dd1`/`dd2` accumulation — exactly half of backward's
`(r,d)`-loop body (`axpy8(dd*, ...)` alongside the `da*`/`dot8` computation
that must stay, since it's needed regardless of whether the dictionary is
learned).

Implemented as research-only kernels (`bwd_block_avx2_hoisted_frozen`,
`SharedMonarchMatmul::backward_block_hoisted_frozen`, `m=8`-only, no
scalar fallback — not wired into the live model, since freezing the
dictionary changes model capacity/expressiveness and is a training
decision separate from a performance one) and measured via a throwaway
standalone bench (isolated backward-only timing, normal vs. frozen, same
reconstructed blocks):

```
256x256 (toy):          2.06x speedup
896x896 (production):   2.11x speedup
896x3072 (production):  2.20x speedup
```

Confirms Opus's prediction cleanly — consistent ~2x backward speedup from
freezing the dictionary, at every scale tested. Not wired into the live
model or evaluated for quality impact this session (that's the real
question before adopting it — does freezing the dictionary cost held-out
loss, and is it worth the tradeoff given the equal-capacity result above
already closes the accessibility question without it). Flagged as a
genuine, cheap-to-test future lever if more speed is wanted, not something
this session decided to adopt.

**Status:** the accessibility goal is resolved (Monarch wins at equal
capacity, at every scale tested, once backward is correctly included).
Block-size tuning is closed as a dead end. Frozen-dictionary is an open,
promising, *not yet evaluated for training quality* future option.

---

## 2026-07-03 — Fable review #2 + implementation: real in-process profiling, then the actual backward fix

Asked Fable to investigate why real training throughput (408-420ms/step)
still trailed the isolated kernel win (2.7-3x) so badly, given both
`AttnProj` and `Ffn` were already hoisted+batched. Fable read
`layer.rs`/`ffn.rs`/`optim.rs` directly, confirmed the batching really is
exercised at this shape (no silent fallback), ruled out the AdaFactor
optimizer (~1% of a step), and profiled a standalone (non-LTO) scratchpad
bench to attribute time: **Monarch backward ≈ 72% of backward, ~half the
whole step** — driven by `backward_block_hoisted`'s `da/dd` (coefficient +
dictionary gradient) accumulation, which costs ~16x the data-gradient
(`dx/dz`) per token and — critically — runs **per token**, even though
`da`/`dd` are weight gradients that don't depend on any individual token.
Root cause, precisely stated: forward reconstruction was hoisted (last
entry); the corresponding backward *contraction* with the dictionary was
not — it was still happening `n_tokens` times per block instead of once.

### Landed in-repo instrumentation first (verification before the rewrite)

Added `src/kernels/profiling.rs` (named `AtomicU64` counters + an RAII
`Timer`, ~zero overhead — a handful of `Instant::now()` calls per layer per
step against work measured in milliseconds) and wired it into every named
sub-block of `TransformerLayer::forward`/`backward`
(QKV/WO/attn-core/FFN-select/FFN-compute/norm, fwd and bwd separately).
Extended `src/bin/profile.rs` with `TRAIN_SMALL_LOD=1` to exactly replicate
`train_small_lod`'s config, and to print the accumulated breakdown.
**Confirmed Fable's attribution precisely, in the real LTO'd release
binary** (not a scratchpad probe): of a 663ms/step total, Monarch backward
(FFN+QKV+WO combined) was **374.9ms — 56.5% of the entire step**, with
allocation/glue overhead actually small (~5%, smaller than Fable's rougher
estimate) — the dominant cost really was exactly where Fable said.

### The fix: defer the dictionary contraction to once per block, not once per token

Algebraic reassociation, not an approximation. Per block, per stage, the
weight-gradient contribution from token `t` is `dy_t[r] · dot(dict_row,
z_t)` (da) and `a · dy_t[r] · z_t` (dd) — both linear in `z_t`/`x_t`, so
summing over tokens can move inside: `Σ_t dy_t[r]·dot(dict_row, z_t) =
dot(dict_row, Σ_t dy_t[r]·z_t)`. Define `S[r] = Σ_t dy_t[r] ⊗ z_t` (an
`m×m` accumulator per block per stage — literally an outer-product sum,
cheap: `O(m²)` per token, same cost class as the `dz`/`dx` propagation that
was already cheap) — then the `nd`-scaled dictionary contraction
(`dot`/`axpy` over `nd` atoms) runs **once per block, after the whole
token batch**, instead of once per token.

Implemented as a genuine two-phase split in `monarch.rs`:
- **Phase 1** (`backward_block_phase1`, AVX2 `m=8` + scalar fallback):
  computes `dz`/`dx` exactly as before (unchanged, cheap) and accumulates
  `s1`/`s2` (`Σ dy1_i⊗x_i`, `Σ dout_j⊗z_j`) via plain `axpy8` — no `nd` loop
  at all in the per-token path anymore.
- **Phase 2** (`contract_block`/`contract_all_blocks`, AVX2 + scalar): the
  `nd`-scaled `da`/`dd` computation, structurally identical to the old
  per-token inner loop but now reading `s1`/`s2` (the batch-summed outer
  product) instead of a single token's `z`/`x`, run exactly once per block.

Rewired `SharedMonarchProj::backward_batch`/`backward_rows_batch`/
`backward_cols_batch` (the dense and routed variants both) to this
two-phase shape — same token-chunked parallelism structure as before
(chunks accumulate local `s1`/`s2` partials, merged via `+=`, contraction
runs once sequentially at the end since its total cost — `P·Q·m·nd`, done
once — is now small enough not to need parallelizing).

**Correctness**: new test `phase1_plus_contract_matches_backward_block_hoisted`
proves phase1+contract over N tokens exactly equals N old-style
per-token `backward_block_hoisted` calls summed, at both `m=8` (AVX2) and
`m=4` (scalar) — algebraic reassociation confirmed, not just "the numbers
came out close." Full suite: 92/92 pass, including every existing
gradcheck (`backward_batch_matches_summed_looped_backward`,
`routed_batch_matches_looped_forward_rows_and_cols`, layer/model-level
gradchecks) — all still validated against the *original* single-token
`backward()`/`backward_rows()`/`backward_cols()` methods, so this landed
without touching (or needing to trust) any of the reference implementations.

**Result, sub-block breakdown, same shape as above:** total step time
663ms → **364ms** (45% reduction). Monarch backward specifically: 374.9ms
→ **~108ms** (3.5x reduction — close to Fable's estimate of "toward
40-70ms" per the less-optimized scratchpad build). Attention core is now
the next-largest single item (~89ms/step, unparallelized — Fable's
secondary recommendation, not done this pass).

**Result, real end-to-end synthetic bench** (same `train_small_lod`-shape
byte-vocab bench used throughout this session): **353-393ms/step**, down
from 408-420ms before this fix. Relative to the pre-swap BasisMatmul
baseline (306.5ms/step): was 33-37% slower, now **~15-28% slower** — real,
substantial progress, not fully closed. The remaining gap is consistent
with the now-larger relative weight of the fully-serial attention core
(no rayon anywhere in `attn_flash.rs`/`attn_swa.rs`) and residual
allocation/glue overhead, both flagged but not addressed this pass.

**Status:** the backward gradient-accumulation bug (the actual dominant
cost of the post-swap regression) is fixed and verified. Training
throughput has closed roughly two-thirds of the gap to the pre-swap
baseline. Remaining known levers, not yet done: parallelize the attention
core; reduce/reuse the `FwdCache.zs` allocation footprint (flagged by
Fable, secondary at these shapes per the confirmed profiling — allocation
overhead measured much smaller than Fable's estimate, ~5% not ~20%, so
lower priority than initially thought).

---

## 2026-07-04 — Opus review: single-thread audit before parallelizing, then the regression fully closes

Explicit instruction going in: don't reach for multi-core parallelism of
the attention core until single-threaded waste in the hot path is
genuinely exhausted, not just "probably fine." Asked Opus to audit
`monarch.rs`'s two-phase backward/AVX2 kernels, `attn_flash.rs`/
`attn_swa.rs`, the routed FFN batch paths, and the layer/attn_proj batched
wrappers specifically for single-thread inefficiency — algorithmic or
micro-architectural — before recommending parallelization.

Opus's verdict: the big single-thread win (per-token reconstruction,
per-token dictionary contraction) was already harvested. What was left was
mostly small, with one exception. Prioritized findings:

1. **`gemm::dot` had no `#[inline]` and re-checked
   `is_x86_feature_detected!` on every call** — called O(n_q·T²) times on
   64-element dot products inside the attention core's inner loop, where
   the call/branch overhead is non-trivial relative to the work itself.
   Rated "small-to-moderate."
2. Register-blocking the attention score/`dv` dot products (keeping `qi`
   resident across the inner `j` loop instead of reloading it) — a real
   lever, but a new SIMD kernel with correctness risk, orthogonal to
   parallelism. Deferred, not done.
3. Per-token `vec![0.0f32; b]` scratch allocation inside the monarch apply
   closures (`forward_batch`/`forward_rows_batch`/`forward_cols_batch`) —
   rated negligible (well under 1%), and hoisting it safely under rayon's
   work-stealing would need thread-local scratch, not worth the
   complexity for the estimated win. Skipped.
4. Redundant `h_mid.clone()` in `layer.rs`'s FFN residual write (clone
   then add-in-place, when a single combined write does the same job in
   one pass) — rated small.

Opus's explicit final call: single-threaded optimization was close to
exhausted; attention-core parallelization (embarrassingly parallel over
query rows, ~89ms/step fully serial, largest remaining single sub-block)
was the correct next lever — but only *after* grabbing the cheap, safe
items above, and it explicitly checked the attention algorithm itself
wasn't wasteful before recommending parallelism by default.

### Landed the two cheap, safe fixes (#1 and #4)

- `#[inline]` on `gemm::dot` (`src/kernels/gemm.rs`).
- `layer.rs`: replaced `h_mid.clone()` + in-place residual add with a
  single combined write (`out[j] = h_mid[j] + scale * ffn_out[j]`),
  dropping one full `T·H` memcpy per layer.

Skipped #2 (real gain, real risk, orthogonal to this pass) and #3
(negligible, needs thread-local scratch to do safely).

**Result — far larger than either fix looked in isolation.** Measured
twice for reproducibility (`TRAIN_SMALL_LOD=1 STEPS=20 WARMUP=5 SEQ=256
./target/release/profile`, LTO release binary): total step time
**364ms → ~264-270ms/step (28% reduction)**. The `dot()` inlining was the
real story — attention's inner loops call it O(T²) times per step, and the
call + feature-detection overhead on a 64-element dot dominated the actual
FMA work at this shape. Committed as `4fdd016`.

### The regression is now fully closed and reversed

`profile.rs`'s `TRAIN_SMALL_LOD=1` mode already *is* a real end-to-end
forward+backward+optimizer step on the exact `train_small_lod` shape with
real (synthetic) data — no need to reconstruct a separate `bench_efficiency.rs`
throwaway. Wall-clock: **269.67ms/step**, vs. the pre-swap BasisMatmul
baseline of **306.5ms/step**. The Swap-to-Monarch move — which regressed
to 420ms/step immediately after landing — is now **~12% faster than the
original pre-swap baseline**, not slower.

**Decision point, asked explicitly:** proceed with attention-core
parallelization now (a pure additional win, no longer closing a
regression), or stop here. User chose to stop here — the accessibility and
efficiency goals that motivated this entire multi-session effort are met,
and further parallelization work is deferred as an explicitly flagged,
not-yet-started future lever rather than continued speculatively.

**Status:** the post-swap regression is closed and reversed end-to-end.
Remaining known levers, flagged but explicitly deferred, not started:
parallelize the attention core (~89ms/step serial, embarrassingly
parallel over query rows per Opus); register-block the attention
score/`dv` dot products; evaluate the frozen-dictionary experiment for
training-quality impact (2x additional backward speedup measured, not yet
adopted — separate training decision from this performance thread).

### Fable second opinion: "is there anything left?"

Asked Fable to independently re-verify (not just trust the summary) and
specifically hunt for siblings of the `gemm::dot` inlining fix that Opus's
audit might have missed. Fable re-derived everything from the current
code (confirmed `#[inline]` is really on `dot`, the two-phase backward is
really what's in `monarch.rs`), then found one real miss:

**`Ffn::select` (`src/kernels/ffn.rs:197-204`) computed router logits with
a plain scalar `row.iter().zip(h).map(|(w,x)| w*x).sum()`** instead of
`gemm::dot` — the exact same class of oversight (missing the AVX2+FMA
path), just at a different call site. 3072 calls/step (256 tokens × 12
layers) in scalar code fully accounted for `FFN_SELECT`'s measured cost.
One-line fix, near-zero risk (same function, same shapes, already
correctness-proven elsewhere in the codebase).

Also flagged, correctly rated as marginal and not applied: `backward_batch`
(and its rows/cols siblings) re-runs `expand_all_blocks` on the same
`d1`/`d2`/`a1`/`a2` the forward pass already expanded — redundant
`~1.7M MACs/layer`, an estimated 1-2% of a step. Fixable by caching `eff`
in `FwdCache`, but a real memory-vs-compute tradeoff, not a free win — left
alone.

Fable's tok/s sanity check: a step is ~4-5 GFLOP, so ~1050 tok/s (after
the fix below) implies ~18-20 GFLOP/s achieved on this AVX2 CPU with a
~30%-serial attention core — plausible, no hidden order-of-magnitude waste
found.

**Applied the one confirmed fix:** `Ffn::select` now calls `gemm::dot`
(`src/kernels/ffn.rs`). 92/92 tests still pass. Measured (same
`TRAIN_SMALL_LOD=1` harness): `FFN_SELECT` 8.8ms/step → 1.7ms/step; total
step **238-244ms/step** (down from 264-270ms, ~10% further reduction).
Wall-clock throughput now **~1050 tok/s** at seq_len=256 (up from ~949).
Committed as `aae5657`.

**Status, final for this thread:** two independent reviews (Opus, then
Fable specifically checking Opus's blind spots) agree single-threaded
optimization in the hot training-step path is now genuinely exhausted at
this shape — the remaining sub-block breakdown is dominated by FFN (~44%)
and the still-serial attention core (~30%), exactly the known, explicitly
deferred levers (attention-core parallelization, register-blocked
attention dot products), not undiscovered waste. From the original
420ms/step post-swap regression to **238-244ms/step (~1050 tok/s)** — a
1.7-1.8x improvement, and comfortably faster than the 306.5ms/step
pre-swap baseline this entire effort was trying to reach parity with.

---

## 2026-07-04 — Attention-core parallelization + register-blocking (the two deferred levers, landed)

With single-threaded optimization exhausted, picked up the two levers
explicitly deferred above.

### Parallelize `attn_flash.rs`/`attn_swa.rs` over query rows

Both kernels had zero rayon usage — every query row computed serially,
despite forward being embarrassingly parallel (row `i`'s output depends
only on causal/windowed keys, never on another row). Backward's dK/dV do
accumulate across rows sharing a key, so the parallel split reused the
same per-chunk-local-accumulator-then-merge pattern already established
for the Monarch two-phase backward: row-chunk workers (`rayon::current_num_threads()`-sized
chunks, same as `monarch.rs`) write to local `dk`/`dv` buffers, summed
afterward; `dq` needs no merge since each row's gradient is independent.
Below `PARALLEL_THRESHOLD=8` rows, the unmodified sequential path runs —
same threshold/rationale as the Monarch kernels.

Correctness: added `forward_parallel_matches_naive_softmax` (t=16, checked
against the existing `naive()` O(T²) reference — untouched by the change)
and `backward_parallel_chunking_matches_single_chunk` (compares a forced
1-thread run against a forced 8-thread run via local `rayon::ThreadPool`
instances, isolating just the chunk/merge logic) to both files. Along the
way, an FD-based gradcheck attempt at t=16 failed — but reproduced
identically on the *original, unmodified* file (confirmed via `git
stash`), revealing a pre-existing finite-difference tolerance limitation
at that scale unrelated to this change. Dropped the FD approach at t=16 in
favor of the chunk-comparison test, which sidesteps FD precision entirely.

**Measured** (train_small_lod shape, LTO release binary): `ATTN_CORE_FWD`
21.1ms/step → 10.2ms/step, `ATTN_CORE_BWD` 44ms/step → 15.8ms/step. Total
step 238-244ms/step → 216-230ms/step; wall-clock throughput **~1100
tok/s** (up from ~1050) — a smaller relative gain than attention's own
drop, because FFN (Monarch) is now the larger share of the step. Committed
`49e9de2`.

### Register-block the score/dp dot products (Opus's flagged real-but-risky lever)

Added `gemm::dot4(a, b0,b1,b2,b3)` — loads the shared vector `a` once and
reuses it across four FMA accumulators, instead of reloading it once per
`dot()` call. Wired into both kernels' forward score/block-max loop and
backward score+dp loop, processing keys in groups of 4 with a scalar
remainder loop for counts not divisible by 4.

Correctness: `dot4_matches_four_dot_calls` in `gemm.rs`; all existing
attention tests (naive/gradcheck/parallel-chunking) still pass unmodified
(97/97 total).

**Measured:** `ATTN_CORE_FWD`/`ATTN_CORE_BWD` changed within run-to-run
noise — no clear win at this scale. Consistent with Opus's original
caveat that this lever's payoff was smaller and less certain than
row-parallelization. Kept anyway since it's correctness-verified, not a
regression, and should matter more at larger `head_dim`/production scale
where the `qi`/`doi` reload cost is a larger fraction of per-key work.
Committed `8d8c5fd`.

**Status:** both explicitly-deferred perf levers from the prior entry are
now landed and verified. Attention is no longer a meaningfully serial
bottleneck. The only performance-adjacent item left on the board is the
frozen-dictionary experiment — which is a training-quality question, not
a kernel-speed one, and needs real training runs (not benchmarks) to
answer honestly.

---

## 2026-07-04 — Frozen-dictionary experiment: the quality side, finally measured

Closed the last open item from the 2026-07-03 frozen-dictionary entry: the
~2x backward speedup was measured in isolation back then, but never
evaluated for training quality (freezing the shared atom dictionary
changes model capacity/expressiveness, not just speed). Wired it into the
live model to test that honestly.

Added `Model::FREEZE_DICT` (`src/model/model.rs`) — same const-ablation
pattern as `attn_proj.rs`'s `DENSE_ATTN`. When `true`, `apply_grad` simply
skips the `af.step(...)` calls for both shared Monarch dictionaries
(attention and FFN); only the per-block coefficients (`a1`/`a2`) keep
learning. This deliberately reuses the *existing, unmodified* two-phase
backward — `dd1`/`dd2` are still computed exactly as before, just
discarded here — so testing the quality question didn't require touching
or risking the hot backward kernels at all. (The actual `dd1`/`dd2`-skipping
kernel from the 2026-07-03 entry, `bwd_block_avx2_hoisted_frozen`, remains
unwired — it predates the two-phase restructuring and doesn't fit the
current batched architecture; wiring it in for the real speed win is a
separate follow-up, only worth doing if the quality result below were more
favorable.)

Also added two env knobs to `train_small_lod.rs` for a fast, isolated A/B:
`CKPT_TAG` (routes a run to its own checkpoint file and always starts
fresh, never resuming from or touching the real `fydel_small_lod.ckpt`)
and `TOTAL_STEPS` (env-overridable step count, for a shorter run than the
default 3000).

**Experiment:** two runs, same config/seed, 1200 steps each — one with
`FREEZE_DICT=false` (baseline), one with `true`.

```
                held-out CE (nats)   final training-window CE
baseline               2.0640              1.8113
frozen                 2.0871              1.8786
```

Frozen costs **~0.023 nats (~1.1% relative) worse held-out CE** — a real,
measurable quality cost, not free. Training-CE gap (1.88 vs 1.81) points
the same direction. Small at this step count/scale, but not nothing, and
there's no reason to expect it to shrink at longer training — if anything
a frozen random dictionary caps representable directions permanently,
so the gap plausibly persists or widens over a full run.

**Decision:** `FREEZE_DICT` stays `false` (off) by default. The
equal-capacity finding from 2026-07-03 already closed the
accessibility/speed question (Monarch beats BasisMatmul at every scale,
backward included) without needing this lever, and freezing the dictionary
has a real, measured quality cost. `FREEZE_DICT` remains in the codebase
as a documented, available ablation — not adopted, not recommended as a
default — for anyone who wants to trade a chunk of that ~1% quality for
the ~2x backward speedup this specific sub-block later, with eyes open.

**Status:** this closes the full arc from the 2026-07-02/03/04 sessions.
Every item raised — the post-swap regression, the equal-capacity
crossover scare, the single-thread audit, attention parallelization, and
the frozen-dictionary quality question — has been measured (not
guessed) and resolved. Nothing performance-related remains open on this
thread.

---

## 2026-07-04 — `FwdCache.zs` allocation footprint: reused via a recycling pool, ~2x step time

Picked up the one item still sitting on the shelf from Fable's original
2026-07-03 review: the `FwdCache.zs`/`y` buffers were re-`vec![0.0; n]`'d
fresh every call, every layer, every step. Originally estimated ~5% of a
step (Fable, revised down from a rougher ~20%) — re-checked with a
standalone alloc+dealloc+zero-fill microbenchmark at matching scale before
starting, since that 5% estimate predated every speedup landed since, and
the same *fixed* allocation cost is a bigger fraction of a much smaller
step now. Measured ~19ms/step, ~8-9% of the (pre-this-change) ~216-230ms
step — worth doing.

### Scoping arrived at the real design in stages

First pass assumed a naive shared/global scratch buffer would work; investigation
found `zs` genuinely needs to survive from one layer's forward call until
that *same* layer's backward call, potentially 20+ other calls later —
ruling out a single shared buffer (each layer/projection needs its own
slot) and any lifetime-borrowed design (Rust can't express "mutate now,
read later, guaranteed non-overlapping" without `unsafe` pointers or
`RefCell`, not without threading real lifetimes through `FwdCache`,
`LayerForward`, `FfnForwardBatch`, `ModelForward`). User confirmed the
"full redesign" scope after this was made explicit (asked twice, as the
real scope kept growing beyond the initial estimate).

### The actual mechanism: a take/give recycling pool, not literal lifetimes

Landed `src/kernels/scratch.rs`'s `BufPool`: buffers stay fully owned
(`Vec<f32>`, moved around exactly as before) — the only change is where a
fresh buffer's memory comes from. `take_zeroed`/`take_uninit` pull a
same-length buffer back out of a free list if one was previously `give`n
back (a `Vec::resize`-cost operation, no syscall); `give` returns a buffer
for the next call to reuse. Sidesteps the lifetime problem entirely: no
`unsafe`, no `RefCell`, no generic lifetime parameters anywhere.

Threaded explicitly through the whole call chain (the "explicit
workspace/arena parameter" option from the initial scoping question):
`SharedMonarchProj`'s six batch methods (drawing `y`/`zs` from the pool;
`zs` specifically switches `backward_batch`/`backward_rows_batch`/
`backward_cols_batch` from taking `zs: &[f32]` to taking `cache: FwdCache`
by value, so the buffer can be extracted and given back right after use)
→ `AttnProj` (copies `y` out immediately and gives it straight back — same
call, no cross-step lifetime needed) → `Ffn` (up/gate/out escape into
`FfnForwardBatch` for the whole step, so aren't pooled the same way; their
`zs`-equivalent caches still are) → `TransformerLayer`/`Model`
(forward/backward both take `pool: &mut BufPool`) → the training loop
(owns one `BufPool` for the entire run, not per-step).

`take_uninit` (for `zs`, pure write-before-read — every element assigned
exactly once across the block loop, never accumulated into) skips the
zero-fill pass entirely, not just the allocation; `take_zeroed` (for `y`,
which *is* accumulated into) still zeros, same as before.

About 40 test call sites across `monarch.rs`/`ffn.rs`/`layer.rs`/
`model.rs`/`train/{optim,checkpoint,loop}.rs` needed updating — mechanical
but not risk-free: a few tests read a `LayerForward`/`FfnForwardBatch`
field *after* the call that now consumes it by value, needing the read
reordered before the consuming call (e.g. cloning `fwd.ffn_fwds.selected`
before `backward` moves `fwd`), and one test calling `backward` twice on
the same forward cache needed a second, independent `forward` call instead
(neither `LayerForward` nor `FwdCache` derive `Clone`).

**Result: 100/100 tests pass, and the win is much bigger than scoped.**
Measured (train_small_lod shape, LTO release binary): total step time
**216-230ms/step → ~114-122ms/step — roughly 2x**, far past the ~8-9% the
standalone microbenchmark predicted. The gap: `take_uninit` also skips
`zs`'s zero-fill pass entirely (the microbenchmark zeroed every buffer,
including ones that don't need it), and the isolated microbenchmark
almost certainly understated real allocator/page-fault cost under an
actual 12-layer model's live memory pressure versus a clean, repeated
alloc/dealloc loop with nothing else competing for the allocator. Wall-clock
throughput: **~2100-2200 tok/s** at seq_len=256, up from ~1100 —
roughly double, again.

**Status:** this was the last item on the board from the 2026-07-02
through 2026-07-04 arc. Every flagged item has now been measured and
resolved, several (this one included) by a wide margin past the original
estimate. Nothing performance-related remains open on this thread.

---

## 2026-07-05 — Fable training-commitment review, then closing the two hard blockers

Asked Fable a differently-framed question than the perf arc above: not
"what's the next optimization" but "is this ready for someone to actually
commit real time to training it." Fable profiled a production-shaped
8-layer slice directly (rebuilt, ran `TRAIN_SMALL_LOD` and a
production-shaped config itself) and found the real bottleneck nobody had
looked at: **`FULL=1` (the true 96-layer, hidden-896 production spec)
doesn't fit in this machine's 14GB RAM at all**, and a measurable
production-shaped step spent **~82% of its time in the tied LM head**
(`gemm.rs`'s `logits_from_embed`/`head_backward`), completely
unparallelized and outside every existing profiling counter — every
optimization in the arc above had targeted the ~18% of a production step
that stays small at that shape. Fable's verdict: **not ready to commit
yet** — fix the head parallelism and verify checkpoint resume first, since
at the then-current throughput a week of training bought only ~8-10M
tokens, too few to prove the architecture learns anything.

### Parallelized the tied LM head

`logits_from_embed` (forward) and `head_backward`'s two passes sweep the
full `[vocab, hidden]` embedding table for every row — memory-bandwidth-
not compute-bound. Chunked over token rows (forward, pass A) and vocab
rows (pass B) via rayon, same threshold/pattern as attention/Monarch;
split the combined AVX2 kernel into independently-parallelizable
`head_backward_pass_a_avx2`/`_pass_b_avx2`. New parallel-path correctness
tests; 102/102 total.

**Measured, honestly:** ~30-35% reduction at a production-shaped 8-layer
slice (11.66s/step -> ~7.6-8.3s/step) — real, but well short of a naive
"82% single-threaded on 12 threads ⇒ big multiple" expectation, because
memory-bandwidth-bound work doesn't scale with core count the way
compute-bound work does; the shared memory controller becomes the limit.

### Verified checkpoint resume for real

The existing `roundtrip_preserves_params_and_opt_state` test only
exercises save/load in-process — never a real kill-and-restart. Added
`CKPT_EVERY` (checkpoint interval override) and let `RESUME=1` opt a
`CKPT_TAG`'d run back into resuming (tagged runs default to fresh, to
protect the real checkpoint). Ran a real training process, `SIGKILL`'d it
mid-run past a checkpoint save, restarted, confirmed correct resume from
the saved step with sensible continued loss descent (no reset, no
divergence). Note: the data reader's position isn't checkpointed, so a
resumed run's batch sequence isn't bit-identical to an uninterrupted one
— state-consistent, not replay-exact, fine for training.

### RAM-fitting config, then the user caught the real lever: activation checkpointing

Measured empirically (not estimated): ~354MB/layer, ~420MB fixed, at
production width/seq. 12 layers (4.68GB) verified safe; 16 layers (5.96GB)
right at the edge with no margin. This would have been the plan — a
reduced-depth config — until the user pointed out `model.rs`'s own
docstring already flagged the real fix: **"activation checkpointing —
trading storage for recompute — is a deferred memory optimization."**

Implemented it. `Model::forward` no longer retains every layer's full,
expensive `LayerForward` (the Monarch `zs` caches + FFN intermediates that
account for essentially all of that ~354MB/layer); it keeps only a cheap
per-layer `LayerReplayState` (input, output, probe probability, FFN
routing decision — a handful of `t·hidden`-sized buffers, not the
`t·hidden²/block`-sized `zs` caches). `Model::backward` recomputes each
layer's full `LayerForward` from its saved input immediately before that
layer's own backward call, then discards it the same way — so peak
activation memory is `O(1 layer)` instead of `O(n_layers)`. Recompute is
exact (deterministic, no dropout anywhere in this codebase), so this is a
pure memory/speed tradeoff, not a quality one — confirmed by the existing
gradcheck tests passing unchanged. Added `discard_into_pool` on
`LayerForward`/`FfnForwardBatch` so the throwaway initial-forward pass's
pooled buffers (`zs`, FFN `up`/`gate`/`act`) get returned to `BufPool`
instead of leaking to a plain drop, preserving the pooling win built
2026-07-04. (Named the new struct `LayerReplayState`, not `LayerCheckpoint`
— that name was already taken by an unrelated, pre-existing concept, a
*weight* snapshot for disk save/load.)

**Result: 16-layer slice peak RSS 5.96GB -> 1.07GB (~5.5x). The actual
96-layer `FULL=1` spec — previously unable to even allocate — now runs
successfully at 2.34GB RSS**, ~34.5s/step, ~14.8 tok/s. Honestly: the
throughput number lands back almost exactly at Fable's original
pre-any-fix estimate, since the head parallelization's ~30% reduction and
this change's ~30-50% recompute overhead largely offset each other. **The
win is feasibility, not speed** — this config could not run at all
before, and now runs with 3-4GB of headroom to spare.

**Status:** two of Fable's three hard training-commitment blockers are
now resolved (LM head parallelism, checkpoint resume) and the third
(RAM feasibility) is resolved by a better mechanism than originally
planned (activation checkpointing instead of a reduced-layer-count
workaround) — the full production spec now actually runs on this machine.
Remaining, per Fable's punch list: a longer sanity run past the ~1200-step
toy horizon, and confirming eval/logging is trustworthy unattended — both
flagged as good-to-have insurance, not blockers.

### Follow-up: a memory-specific Fable review (not throughput)

With activation checkpointing making peak activation memory O(1) layer
instead of O(n_layers), asked Fable to review memory specifically — is
there real headroom left, now that the model actually fits (`FULL=1`
measured 1.90GB peak RSS at seq=256, comfortably inside 14GB)? Fable
profiled fresh (didn't trust the summary above), traced every claimed
buffer to a specific line, and checked the obvious suspects: AdaFactor is
factored everywhere including the embedding matrix (not a full
`[vocab,hidden]` second moment); the embedding table isn't duplicated in
the LM head; `BufPool` is bounded in practice; checkpointing correctly
holds only one layer's Monarch caches at a time. All clean.

Found two genuine, zero-tradeoff wastes in `src/model/model.rs`:
1. `LayerReplayState.input` was never read anywhere — every consumer
   already derives a layer's input from the previous layer's `.out` (or
   the embedding lookup for layer 0). Pure duplication, ~86MB at
   production shape.
2. `backward`'s per-layer loop cloned `lg.d_hidden` into `d_x`, then
   retained the *original* `lg` (d_hidden included) inside
   `layer_grads_rev` until the whole `ModelGrads` dropped — but
   `d_hidden` is a per-token activation gradient, deliberately never read
   again downstream (`LayerGrads::add`/`scale` explicitly skip it).
   Switched to `std::mem::take` — moves instead of clones, and leaves
   nothing retaining a dead copy.

Removed both (102/102 tests still pass — both were correctness-neutral,
confirmed by grep showing zero readers before removing). Measured,
`FULL=1` seq=256: peak RSS **1.90GB → 1.73GB** (~170MB, matching Fable's
estimate almost exactly).

**Status:** Fable's verdict, corroborated by the measurement: memory is
now at genuine diminishing returns. What's left (weights, one gradient
copy, already-factored optimizer state, the embed table) is inherently
sized, not wasteful. Compute — not memory — is this model's binding
constraint at production shape (confirmed separately: ~34.5s/step, ~14.8
tok/s at seq=512, 96 layers).

### Follow-up: a compute-focused Fable review (back to throughput, at production scale specifically)

With memory closed out, asked Fable one more time — compute throughput
specifically at production shape, now that toy scale is exhausted and
`FULL=1` actually runs. Fable profiled `FULL=1` fresh and made a sharp
catch: **the profiling counters double-count the activation-checkpointing
recompute** the same way the LM head was invisible before — every `*_FWD`
Timer lives inside `layer.forward`, which `Model::backward` now calls a
second time per layer (the recompute), so the `*_FWD` rows in the report
count two passes, not one. Decomposed correctly: recompute tax is only
**~12% of the step** (one forward pass ≈ 3.6s of the ~29.5s measured),
not the dominant cost people might assume from "doubles every layer's
forward compute." The real dominant cost is FFN_BWD gradient math itself
(~35% of the step) — genuine, already-parallelized, bandwidth-bound
compute, not overhead.

Fable also did the honest math on selective/partial checkpointing (the
obvious "trade some of the ~12GB spare RAM for less recompute" idea): it
doesn't work the way it sounds. The current scheme already recomputes
each layer exactly once — the theoretical minimum for single-pass
recompute. Coarsening to "checkpoint every Kth layer" doesn't reduce
recompute, it only *raises peak memory* (since retained layers skip
recompute, at cost of holding their full activation cache). At current
headroom (~12GB ÷ ~354MB/layer ≈ 33 layers retainable), retaining 33/96
layers would cut recompute to 63 layers and save an estimated ~4%
(~16.3 → ~17.0 tok/s) — real but marginal, for real complexity and 4-5x
more activation RAM. Not worth it.

The one thing Fable flagged as genuinely unexamined: `contract_all_blocks`
(the dictionary-contraction tail of the two-phase Monarch backward,
`monarch.rs`) is a serial `P×Q` double loop whose own doc comment
correctly said "too small to parallelize" — at *toy* `dict_k=8`. Nobody
had revisited that claim at production `dict_k=32` (4x heavier,
independent of sequence length).

**Measured before acting** (established practice this session — verify
Fable's flags against real in-repo numbers before implementing): added a
`MONARCH_CONTRACT` profiling counter. Confirmed: ~2.1s/step, ~6.6% of a
whole production-shaped step, 100% serial.

**Fixed:** parallelized `contract_all_blocks` over `(pp,qq)` blocks —
`da1`/`da2` write disjointly per block (direct `par_chunks_mut`); `dd1`/
`dd2` are shared dictionary-gradient accumulators every block contributes
to, so each chunk accumulates a local copy, merged afterward — the same
shape as the existing two-phase backward's `s1`/`s2` token-chunk
accumulation. 102/102 tests pass unchanged (including the existing
`backward_batch` gradcheck tests, which exercise this function
indirectly).

**Result:** `MONARCH_CONTRACT` itself dropped **~2.2x** (2086ms →
~955ms/step). Since it was only ~6.6% of the whole step, overall step
time (~30.7-32s/step, within this machine's run-to-run noise floor at
this scale) didn't move measurably outside that noise — exactly matching
Fable's own framing going in ("a clean, in-character win," never promised
as a big multiplier).

**Status:** per Fable's explicit, corroborated verdict: **~16 tok/s is
close to the floor for this architecture/hardware/scale combination.**
FFN_BWD is genuine, well-parallelized, bandwidth-bound compute on 6
cores — not waste. No further cheap compute lever was found. This closes
the compute side the same way the memory review closed the memory side:
diminishing returns confirmed by measurement, not assumed.

---

## 2026-07-05/06 — `fp16-migration` branch: storage-format experiment, four stages, diminishing returns confirmed

With both compute and memory declared at genuine diminishing returns on
`master`, opened a new branch (`fp16-migration`) to explore a different
axis entirely: fp16 storage. Verified upfront (not assumed) that this CPU
(AVX2, Zen2, no AVX-512FP16/BF16) has **no native fp16 arithmetic units**
— only `F16C`'s conversion instructions (`VCVTPH2PS`/`VCVTPS2PH`). So
unlike GPU/TPU mixed-precision training (which gets its win from native
low-precision tensor-core hardware), fp16 here is a **storage format
only**: every kernel converts to fp32 immediately before computing and
back before storing. The benefit is halved memory/bandwidth footprint,
not faster FMA.

### Stage 1: conversion primitives (`src/kernels/f16_simd.rs`)

`f16_to_f32`/`f32_to_f16` (bulk AVX2+F16C conversion, scalar fallback) and
`dot_f16`, using the `half` crate (a previously-declared-but-unused
Cargo dependency). Tests confirm AVX2 conversion matches `half::f16`'s
own conversion bit-for-bit.

### Stage 2: the tied embedding table (117MB, the highest-bandwidth-impact target)

Converted `Model.embed` to `Vec<half::f16>`. `logits_from_embed`/
`head_backward`'s embed-reading pass convert each row into a small reused
fp32 scratch buffer right before the existing FMA loop, which runs
unchanged — the hot AVX2 math itself was never touched, only the
conversion at its boundary.

**Caught a real footgun before reporting a win**: the first
implementation's `apply_grad` (AdaFactor needs fp32 for its update math)
allocated a *fresh* fp32 scratch buffer every step — touching a new
~117MB region every call, which raised peak RSS by roughly what fp16
storage was supposed to save. Confirmed via worktree A/B against the
pre-fp16 commit (1.99GB before → 2.05GB after, a real *increase*, not a
decrease). Fixed by giving `Model` a persistent `embed_scratch` field,
reused every step — this eliminated the repeated allocator churn but
(learned honestly, not assumed) did *not* reduce peak RSS further, since
`/usr/bin/time`'s max-RSS is a high-water mark that never decreases once
a region is touched, transient or persistent.

**Result**: throughput ~5-7% faster (interleaved A/B, twice, consistent).
Peak memory unchanged net (the embed table's own ~59MB reduction
canceled by the footgun above).

### Stage 3: Monarch's `zs` activation cache (the single largest buffer this whole arc's optimization work centered on)

Cleaner case than the embed table: `zs` is write-once/read-once per step
with **no gradient/accumulator counterpart**, so no fp32-master-copy
footgun risk at all. Critically, `apply_block`/`apply_block_avx2` — the
hottest kernel in the codebase, called hundreds of thousands of times per
step at production scale — was **deliberately left untouched**: all fp16
conversion happens at the caller boundary (the three
`forward_*_batch`/`backward_*_batch` methods' per-token/per-block loops),
never inside the dense AVX2 math itself. `BufPool` gained a parallel fp16
free-list (`take_f16_uninit`/`give_f16`) since fp32 and fp16 buffers of
the same element count aren't interchangeable.

**Result**: combined with stage 2, throughput ~8-9% faster (two
interleaved A/B pairs, consistent both times); peak RSS 1.94GB, close to
the original 1.99GB pre-fp16 baseline — `zs`'s real reduction finally
showing through, unlike stage 2 alone.

### Stage 4: FFN's `up`/`gate`/`act` (same write-once/read-once shape as `zs`)

Landed for consistency (correctness-neutral, `BufPool` infrastructure
already existed) but **confirmed diminishing returns**: these buffers are
~18.9MB/layer combined at production shape — roughly 14x smaller than
`zs`'s per-projection footprint. Measured: essentially flat versus stage
3 (~31.4-32.1s vs ~31.5-31.7s), no meaningful further improvement. This
is the expected, honestly-reported result, not a surprise — the whole
point of measuring each stage separately (rather than batching all four
and reporting one combined number) was to find exactly this: which
specific buffers matter and which don't.

**Correctness throughout**: 106/106 tests pass at every stage, including
the full gradcheck suite (`backward_batch_matches_summed_looped_backward`,
`shared_monarch_proj_gradcheck`, `compute_batch_matches_looped_compute_and_backward`,
and the rest) — all still validated against the *original, untouched*
fp32 reference implementations, confirming fp16 storage changes memory
and bandwidth only, not the computed gradients (as expected for a
storage-format change with fp32 compute throughout).

**Status:** `fp16-migration` branch has captured its real value (~8-9%
throughput, roughly memory-neutral) via stages 2-3. Stage 4 confirmed the
remaining untouched candidates (Monarch's `a1`/`a2` coefficients, `d1`/
`d2` shared dictionaries, attention's `q`/`k`/`v`/`attn_out`) would likely
show similarly small returns — `d1`/`d2` especially are already tiny and
cache-resident (nd·b ≈ 8KB), so fp16 there would add conversion overhead
for no bandwidth benefit at all. Not merged to `master` — remains an
experimental branch pending a decision on whether an 8-9% throughput
win is worth the added `half`-crate dependency and fp16-conversion
surface area across the codebase permanently.

---

## 2026-07-05 — `feat/int16-quant`: true-int16 optimizer state falsified, deferred

Explored pushing further than fp16 — full int16 (SIMD-width motivated:
`i16×i16→i32` AVX2 accumulation is 16-wide vs. fp32's 8-wide) across
weights, activations, gradients, and optimizer state, with quantization-
aware training (fake-quant forward, straight-through backward). Before
touching the real 17K-line codebase, built two standalone falsification
probes (`src/bin/fixedpoint_adam_probe.rs`,
`src/bin/fixedpoint_adafactor_probe.rs`) to test the riskiest piece —
int16 optimizer state — on a synthetic quadratic-bowl problem, per the
project's usual practice of proving a numerical scheme cheaply before
wiring it into production code.

**Design tested:** dynamic block fixed-point — int16 storage + a shared
power-of-two scale per tensor/vector, recalibrated from the observed max
magnitude every 8 steps (mirrors dynamic loss-scaling, applied per-tensor
instead of just to the loss).

**First probe (plain Adam, wrong optimizer — see below):** catastrophic
divergence. Root cause: Adam's second moment `v` is intentionally
*per-parameter* — it's what gives each parameter its own effective
learning rate. A single shared scale across a tensor whose per-parameter
gradient magnitude spans orders of magnitude (realistic — different
rows/channels of a real weight matrix see different gradient statistics)
forces the smallest-magnitude entries to quantize to exactly zero
(measured: ~55-98% of low-curvature parameters' `v` dead by step 100).
Once `v_i = 0`, Adam's denominator collapses to `eps` (1e-8), amplifying
that parameter's update by ~1e8x — not noise, a structural blowup.

**Correction:** the actual optimizer in this codebase is AdaFactor
(`src/kernels/optimizer.rs`), not Adam — factored row/column second-moment
sums (`O(rows+cols)` state) instead of full per-parameter `v`
(`O(rows·cols)`), plus a global RMS-clip on the update. Re-ran the same
falsification against AdaFactor's real update math (mirrored from
`optimizer.rs::step`) on a matrix problem with per-row/per-column
curvature spread.

**Second probe (real AdaFactor):** same underlying cause (shared scale
can't span the row/column factors' dynamic range — measured ~31%/50% of
`R`/`C` quantized to zero), but a *different* failure signature:
stagnation, not explosion. When `C[j] = 0` zeroes an entire column's
`vhat`, that column's update spikes toward infinity, and AdaFactor's
**global** RMS-clip divides the *whole tensor's* update — including every
healthy parameter — down to compensate. One dead factor entry silently
throttles the entire tensor's effective learning rate; the loss curve
still trends downward, just glacially, which is a harder failure to
notice by eye than a divergence. Factoring shrinks the *state size* but
not the *dynamic-range problem* that breaks single-scale int16 storage.

**Decision, backed by the existing profiler data (2026-07-03 entry,
above): the AdaFactor step is ~1% of a training step**, so even a working
int16 optimizer-state scheme would optimize a cost center that isn't one.
Deferred at best — not worth the precision risk just demonstrated twice.
Any future int16 work on this branch should target the matmul/kernel path
instead (Monarch backward alone is ~72% of backward, ~half the whole
step — the actual FLOPs and the only place the int16 SIMD width
advantage would matter), keeping weights' master copy, gradients, and
AdaFactor's `R`/`C`/`mom` state in fp32.

**Status:** `feat/int16-quant` branch open with both probes committed as
a record. No changes to `master`/production code paths.
