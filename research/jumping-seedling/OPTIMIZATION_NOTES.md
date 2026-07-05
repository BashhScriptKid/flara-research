# Jumping Seedling — Optimization Notes (Stacking)

This file accumulates optimization ideas, implementation details, and performance
targets for future versions of Jumping Seedling. Claude reads this to resume work.
Notes are append-only. Scope each version before implementing.

---

## Baseline numbers (for calibration)

All single-thread estimates unless noted. Measured on R5 5500U (6C Zen 2, AVX2,
8 MB L3, 36 GB/s DDR4).

| Config | ms/step | tok/s | GFLOP/s eff | Notes |
|---|---|---|---|---|
| Mid-config (12L, H=512, FFN=2048, vocab=8192, seq=256) | 3698 | 69 | ~9 | Current FFT-circulant kernels |
| Full 1B estimated (96L, H=896, FFN=3072, vocab=32768, seq=1024) | ~37s | ~7 | ~9 | Memory-bound, weights scale 10× |

Post-Monarch estimates (from monarch_probe.rs):

| Primitive | µs/token (n=512) | GFLOP/s |
|---|---|---|
| Dense 512×512 matvec | 20.8 | 25.2 |
| Monarch 512 (block-GEMM) | 1.88 | 26.2 |
| BasisMatmul 512 (FFT-circulant) | 19.45 | — |

Monarch is 10.4× faster per token than BasisMatmul at the same AVX2 efficiency.
The FFT-circulant realization captures 0% of structural speedup. Monarch
captures ~100%.

### Chinchilla-optimal ceiling

The Chinchilla scaling law: N params needs O(N) tokens, O(N^1.5) total FLOPs.
At fixed throughput T tok/s for 30 days, the ceiling scales as:

  N_max ≈ (T × 30 × 86400)^0.67

On the 5500U with JS1 (~7 tok/s), this gives ~670M params in 30 days.
Each kernel optimization moves this up proportionally to the throughput gain.
Doubling throughput gives ~40% more params (sqrt scaling).

---

## JS1.1 — Monarch/BTT migration (highest priority)

**Goal:** Replace every `BasisMatmul` projection with a GEMM-based BTT core.

**Why:** BasisMatmul runs at dense wall-clock (19.5 µs vs 20.8 µs for dense
matvec at n=512) and captures 0% of structural speedup. Monarch captures 100%
(1.88 µs). This is the single highest-leverage change in the entire project.

**Status:** Gate (a) passed (monarch_probe.rs). Gate (b) passed (btt_probe.rs:
full-rank at nd≥8, backward correct, trainable). Gate (c) pending — wire one
shared-core BTT block into a real layer and confirm loss descends on real data.

### Implementation plan

#### Step 1: Shared-core BTT matmul primitive

Replace `BasisMatmul` with a new `BttMatmul` in `kernels/`:

```
struct BttMatmul {
    m1: usize,          // block rows (hidden / block)
    m2: usize,          // block cols (hidden / block)
    n_shared: usize,    // number of shared atoms per stage (nd)
    dict1: Vec<f32>,    // shared atom dictionary for stage 1, m2 × n_shared
    dict2: Vec<f32>,    // shared atom dictionary for stage 2, m1 × n_shared
}
```

Per-matrix storage: `P × Q × n_shared` real coefficients (same as current
`BasisMatmul`), but the matmul itself is block-diagonal GEMM, not FFT-circulant.

The forward pass for a weight `W[out, in]` expressed as order-2 Monarch:

```text
y = Monarch(x) = B @ transpose(A @ x)
where A = block_diag(A_0, ..., A_{m1-1}) with A_i ∈ R^{m2×m2}
      B = block_diag(B_0, ..., B_{m2-1}) with B_j ∈ R^{m1×m1}
```

Each `A_i` and `B_j` is a linear combination of shared atoms:

```text
A_i = Σ_k coeff_a[i][k] * dict1[k]    (dict1 is m2 × n_shared)
B_j = Σ_k coeff_b[j][k] * dict2[k]    (dict2 is m1 × n_shared)
```

**Key difference from FFT-circulant:** The atom reconstruction is a real
matrix-sum (not FFT/IFFT + complex multiply). The matmul itself is
block-diagonal GEMM (real-valued), which AVX2/AVX-512 handles natively.

**Geometry for full 1B spec (H=896, block=64):**
- m1 = H/block = 14
- m2 = block = 64
- Stage 1: 14 blocks of 64×64 GEMV
- Stage 2: 64 blocks of 14×14 GEMV
- Transpose: [14][64] → [64][14]

**FLOPs per matvec:** 14 × 64² × 2 + 64 × 14² × 2 = 114,688 + 25,088 = 139,776
Dense equivalent: 896² × 2 = 1,602,816
Ratio: 139,776 / 1,602,816 = **8.7% of dense FLOPs**

But the measured speedup is 10.4× (not 11.5×) because Monarch has transpose
overhead and the second-stage blocks are small (14×14).

#### Step 2: Backward pass

The backward needs gradients w.r.t. coefficients, shared atoms, and input.

**Coefficient gradients:** For each block `(i,j)`, `d_coeff[i][k] =
dot(dA_i, dict1[k])` where `dA_i` is the gradient w.r.t. the block matrix.
This is a small dot product (m2² = 4096 elements for stage 1, m1² = 196 for
stage 2). Auto-vectorizes with AVX2.

**Shared atom gradients:** `d_dict1[k] += coeff_a[i][k] * dA_i` for each block.
This is an axpy — auto-vectorizes.

**Input gradient:** Chain rule through the two stages + transpose. Each stage's
backward is a block-diagonal GEMV transpose (same structure as forward, different
blocks).

**Full backward FLOPs:** ~2× forward (same as standard backward).

#### Step 3: Wire into the model

Replace `BasisMatmul` in:
- `attn_proj.rs`: Q/K/V/O projections
- `ffn.rs`: up/down projections (gate shares up's geometry)
- Keep `router_w` as dense (tiny, H→48, not worth compressing)

The `BttMatmul` API should match `BasisMatmul`'s interface:
- `forward(dict, coeffs, x) -> y`
- `backward(dict, coeffs, x, dy) -> BttGrads`
- `forward_rows(dict, coeffs, x, selected_rows) -> y` (for FFN routing)
- `backward_rows(dict, coeffs, x, dy, selected_rows) -> BttGrads`

#### Step 4: gradcheck and benchmark

- Finite-difference gradcheck of all backward paths
- Head-to-head benchmark against BasisMatmul on the same hardware
- Target: 10× per-block speedup, same GFLOP/s as dense matvec

### Open design decisions

1. **Atom count (nd):** Gate (b) showed full-rank at nd≥8, 12/12 trainable at
   nd≥8, non-monotonic dead spot at nd=4. Real model uses K=32 (equivalent to
   nd=32), well above the pathology. Could potentially reduce to nd=16 for more
   compression — but this is a later experiment, not a blocker.

2. **Shared atoms vs per-block atoms:** Currently one shared dictionary G across
   all blocks. The BTT hybrid keeps this: dict1 and dict2 are shared across all
   blocks of all matrices and all layers. Per-block coefficients vary.

3. **Dictionary init:** Current `init_dict_random` uses splitmix64. For BTT,
   could init dict1/dict2 as random Gaussian or use a structured init (e.g.,
   DCT basis vectors). The research log says the init doesn't matter much at
   nd≥8 (over-parameterized regime).

---

## JS1.1 — AVX-512 kernel port

**Goal:** Port the hot inner loops from AVX2 to AVX-512 on Zen 4.

**Why:** AVX-512 is 16-wide vs AVX2's 8-wide. For the complex-dot reduction in
`accum_block_grads` (the contraction that won't auto-vectorize), this is a
direct 2× or more. For auto-vectorized code (axpy, pointwise multiply), the
compiler handles it with `target-feature=avx512f`.

### What to port

**Priority 1 — hand-written intrinsics (won't auto-vectorize):**

1. `gemm::dot` — the AVX2 dot product used everywhere. Port to AVX-512:
   - Use `_mm512_fmadd_ps` (512-bit FMA)
   - 32-wide accumulators (two `_mm516` for latency hiding)
   - Horizontal sum via `_mm512_reduce_add_ps` (AVX-512 DQ)
   - Expected: ~1.8-2× over AVX2 version

2. `fft::accum_block_grads` — the complex-dot reduction `acc += (pbuf[f] *
   atom[f]).re`. This is the hot inner loop of BasisMatmul backward. With
   Monarch, this becomes a standard block-GEMM backward, but the coefficient
   gradient computation still needs dot products.

3. `fft::accum_pair_block_grads` — fused-pair variant for up+gate backward.

**Priority 2 — compiler-generated (auto-vectorizes with AVX-512 codegen):**

4. `.cargo/config.toml`: add `target-feature=+avx512f,+avx512bw,+avx512dq,+avx512vl`
   for release builds on Zen 4 targets.

5. All axpy operations, pointwise complex multiply, FFT butterfly operations —
   these auto-vectorize once AVX-512 codegen is enabled.

**Priority 3 — structural changes:**

6. **Attention tiling backwards:** Currently `attn_flash.rs` backward
   recomputes scores from Q/K + saved LSE. The forward tiles to `kv_block`;
   the backward should tile the same way to keep working set in L1.

7. **Multi-block software pipelining:** Prefetch block n+1's coefficients into
   L2 while computing block n. Use `_mm_prefetch` intrinsics or compiler
   builtins. The router already identifies which blocks to prefetch
   (`prefetch_coeffs`); this extends the pattern to the matmul loop itself.

### AVX-512 vs AVX2 on Zen 4

Important: Zen 4 has TWO 256-bit FMA units, not one 512-bit. So AVX-512 on Zen 4
is actually 2×256=512 bit per cycle, same throughput as two AVX2 FMAs. The
advantage of AVX-512 is:
- Wider registers reduce loop overhead (fewer iterations)
- `_mm512_reduce_add_ps` for horizontal sums (vs manual extract+add for AVX2)
- Masked operations for tail handling (no scalar remainder loop)
- `_mm512_fmadd_ps` has same latency as `_mm256_fmadd_ps` on Zen 4

Net: ~1.5-2× over AVX2, not 2×. The 2× comes from reducing loop iterations and
eliminating the scalar tail.

---

## JS1.1 — Branchless FFN routing

**Goal:** Eliminate the 25%-taken branch in the FFN inner loop.

**Why:** The research log identified this: writing `for b in 0..M { if
selected[b] {…} }` puts an unbiased branch in the inner loop — near worst-case
for branch prediction, constant mispredicts. Loop over gathered active indices
instead.

### Implementation

In `ffn.rs`, the `compute` method currently iterates over all M blocks and
checks if each is selected. Change to:

```rust
// Before (branchy):
for blk in 0..self.m {
    if selected.contains(&blk) {
        // compute this block
    }
}

// After (branchless):
for &blk in &selected {
    // compute this block — no branch, contiguous iteration
}
```

The `selected` array is already sorted by logit (from `route()`). Iterating over
it directly gives:
- No branch in the inner loop
- Contiguous memory access pattern
- Predictable iteration count (always `n_active = 12`)

**Estimated gain:** 10-20% on FFN forward/backward. The FFN is ~40% of total
layer cost, so ~4-8% end-to-end. More importantly, it eliminates pipeline stalls
from mispredicts.

### Extension: branchless prefix-sum for block selection

The `top-k` selection in `route()` currently uses `sort_by` + `truncate`. This
is O(M log M) where M=48. Could be replaced with a partial sort or nth_element
for O(M), but M=48 is small enough that the constant dominates. Leave as-is
unless profiling shows it matters.

---

## JS1.1 — Sticky exit threshold (hysteresis)

**Goal:** Make the early-exit branch prediction-friendly.

**Why:** The CALM probe produces `p = sigmoid(w·h + bias)` per token. The exit
decision `if p > τ { break }` flips between exit/no-exit on nearly every token
for borderline-confidence tokens — the branch predictor's worst case (unbiased
alternating). Making the decision **temporally sticky** (hysteresis) keeps it
biased: once a sequence commits to exiting, it stays exited for a window.

### Implementation

In `model/model.rs` or `train/loop.rs`, add a dwell counter per sequence:

```rust
struct ExitState {
    dwell: usize,         // tokens since last state change
    exited: bool,         // current exit decision
    threshold: f32,       // base exit probability
    hysteresis: usize,    // min tokens before re-evaluating (e.g., 8)
}
```

Decision logic:
```rust
if self.dwell < self.hysteresis {
    // Stay in current state — don't re-evaluate
    self.dwell += 1;
    return self.exited;
}
// Re-evaluate
let new_exited = probe_p > self.threshold;
if new_exited != self.exited {
    self.exited = new_exited;
    self.dwell = 0;  // reset dwell on state change
} else {
    self.dwell += 1;
}
self.exited
```

**Estimated gain:** Minimal wall-clock improvement (early exit is already free
on CPU decode). The gain is making the branch predictor's job easy instead of
impossible. On a 1000-token sequence with ~30% exit rate, this reduces
mispredicts from ~300 to ~50.

---

## JS1.1 — Router early scheduling + coefficient prefetch

**Goal:** Compute the router and prefetch selected coefficient tiles before the
FFN's heavy spectral compute.

**Why:** The router is a tiny H→48 linear (896×48 = 43K FLOPs). The FFN's
spectral matmuls are the bottleneck. If we compute the router as soon as the
hidden state exists and prefetch the selected coefficient tiles into L2, the
prefetch overlaps with the attention sub-block's tail.

### Implementation

In `model/layer.rs`, the current flow is:
```text
attention → O proj → + residual → FFN norm → select → prefetch → compute
```

Restructure to:
```text
attention → O proj → + residual → FFN norm
                                      ↓
                                   select (tiny, H→48)
                                      ↓
                                   prefetch selected coeff tiles into L2
                                      ↓
                                   compute (heavy, waits for prefetch)
```

The `Ffn::select` and `Ffn::prefetch_coeffs` methods already exist. The change
is to call `select` immediately after FFN norm, before any other computation
in the FFN sub-block. The `compute` call then finds the tiles already warm in L2.

**Estimated gain:** 5-15% on FFN if the prefetch hides the memory latency. On
Zen 4 with 384 MB L3, most tiles are already L3-resident; the gain comes from
L3→L2 promotion latency.

---

## JS1.1 — Sub-byte coefficient packing

**Goal:** Reduce coefficient storage from fp32 to 4-bit or 2-bit under the
compression dial.

**Why:** The compression dial (`quantize_coeffs` in `fft.rs`) already picks 8-bit
or 16-bit per group. Sub-byte (4/2-bit) would cut storage 2-4× more, increasing
effective bandwidth and fitting more of the model in L3.

### Implementation sketch

The existing `CompressedWeight` type already supports per-group bit-widths:
```rust
pub struct CompressedWeight {
    pub bit_widths: AlignedVec<u8>,   // 4, 8, or 16 per group
    pub scales: AlignedVec<f32>,
    pub packed: AlignedVec<u8>,
    pub shape: [usize; 2],
    pub group_size: usize,
}
```

Add 4-bit and 2-bit packing:
- 4-bit: 2 coefficients per byte, symmetric quantization → scale = maxabs / 7
- 2-bit: 4 coefficients per byte, symmetric quantization → scale = maxabs / 1

Dequantization in the inner loop:
```rust
// 4-bit dequant (two coefficients per byte)
let lo = (byte & 0x0F) as i8 - 8;  // unsigned to signed
let hi = ((byte >> 4) & 0x0F) as i8 - 8;
let coeff_lo = lo as f32 * scale;
let coeff_hi = hi as f32 * scale;
```

**Cost:** One extra shift+mask+subtract per coefficient in the inner loop. On
AVX-512 this is ~1 cycle per 32 coefficients (vectorized unpacking).

**Risk:** At 2-bit, quantization error may hurt model quality. The compression
dial should auto-select: 8-bit where high precision is needed, 4-bit for
mid-range, 2-bit only for low-sensitivity coefficients. Profile on real training
to validate.

---

## JS1.1 — Activation checkpointing (implemented separately)

**Goal:** Reduce activation memory from ~6 GB to ~500 MB for full 1B training.

**Why:** Currently every layer's forward intermediates are stored for backward.
With checkpointing every 8 layers, only 8 layers' activations are live at once.
The other 88 layers recompute their forward during backward.

**Status:** Flagged as deferred in `model/model.rs`. The forward already stores
everything needed; checkpointing requires:
1. Storing only every 8th layer's `LayerForward`
2. In the backward loop, recomputing forward for layers between checkpoints
3. This costs ~12% more compute (recompute 7/8 of forward) but saves ~88%
   activation memory

### Implementation

In `model/model.rs` forward:
```rust
// Store only checkpoint layers
for (i, layer) in self.layers.iter().enumerate() {
    let lf = layer.forward(...);
    if i % CHECKPOINT_EVERY == 0 {
        layer_fwds.push(lf.clone());
    } else {
        // Don't store — will recompute in backward
        layer_fwds.push(LayerForward::placeholder(lf.out.clone()));
    }
}
```

In backward, recompute forward for non-checkpoint layers:
```rust
for l in (0..self.layers.len()).rev() {
    let fwd = if l % CHECKPOINT_EVERY == 0 {
        &fwd.layer_fwds[l]
    } else {
        // Recompute forward from previous checkpoint
        recompute_forward(&self.layers, &self.dict, &self.rope, l, ...)
    };
    // ... backward as usual
}
```

---

## JS1.2 — Frequency-domain second moment (experiment)

**Goal:** Test whether FFT of the gradient makes AdaFactor's rank-1
factorization tighter.

**Why:** The project conjecture (from RESEARCH_LOG.md): "an FFT of the gradient
compacts its energy into fewer coefficients, which should make G² more separable
in the spectral domain and the rank-1 factorization tighter." This is a genuine
experiment, not a guaranteed win.

### Experiment design

1. Take a gradient `g` from a real training step (any parameter tensor)
2. Compute `G = FFT(g)` (use existing rustfft infrastructure)
3. Measure the rank-1 approximation error:
   - Spatial: `||g² - R*C/ΣR||_F` (current AdaFactor)
   - Spectral: `||G² - R_s*C_s/ΣR_s||_F` (proposed)
4. If spectral is tighter, the second-moment estimate is more accurate → better
   update scaling → faster convergence

**Expected outcome:** The gradient is NOT as sparse/compressible as weights or
activations. The energy compaction argument works for smooth signals but
gradients are noisy. The spectral factorization might be WORSE, not better.
This is why it's an experiment, not a default.

### If it works

Modify `optimizer.rs` to optionally FFT the gradient before computing row/column
sums. The FFT is cheap (same rustfft infrastructure, block size = parameter
block size). The factorization runs on the spectral coefficients. The inverse
FFT is not needed because the update is applied in spectral domain (or
equivalently, the Adam update in spatial domain is equivalent to spectral update
via convolution theorem).

**If it doesn't work:** Document the result and move on. The spatial
factorization is fine.

---

## JS1.2 — INT8 momentum

**Goal:** Quantize AdaFactor's momentum buffer from f32 to INT8.

**Why:** `QuantizedMomentum` type already exists in `types.rs` but is unused.
The momentum buffer `mom` is the second-largest optimizer state after the
factored second moment. Quantizing to INT8 with per-group scales cuts its
memory 4×.

### Implementation

The existing `QuantizedMomentum` struct:
```rust
pub struct QuantizedMomentum {
    pub data: AlignedVec<i8>,
    pub scales: AlignedVec<f32>,
    pub group_size: usize,
}
```

In `optimizer.rs`, replace `mom: Vec<f32>` with `mom: QuantizedMomentum`:
- Store: quantize `mom[i] = (value / scale).round().clamp(-127, 127) as i8`
- Load: dequantize `value = mom[i] as f32 * scale`
- Update: `mom[i] = (beta1 * dequant(mom[i]) + (1-beta1) * grad).requant()`

The requantization is the tricky part — the scale changes every step because
the momentum magnitude grows. Two options:
1. **Fixed scale:** Set scale at init based on expected gradient magnitude.
   Simple but loses precision over time.
2. **Adaptive scale:** Re-quantize every N steps based on current maxabs.
   More accurate but adds overhead.

**Recommendation:** Start with fixed scale (set from first gradient's maxabs),
profile memory savings, then add adaptive scale if precision matters.

---

## Scaling the architecture

**Prerequisites:** JS1.1 fully landed (Monarch + AVX-512 + pipelining).

### Memory scaling

Per-token FLOPs scale as O(N). Weight memory scales as O(N). With compressed
weights (Monarch + sub-byte coefficients), the constant is much smaller than
dense fp32:

| Params | Dense fp32 weights | Compressed (BTT + 8-bit) | With ckpt/8 activations |
|---|---|---|---|
| 1B | 4 GB | ~150 MB | ~150 MB + 500 MB |
| 5B | 20 GB | ~750 MB | ~750 MB + 500 MB |
| 10B | 40 GB | ~1.5 GB | ~1.5 GB + 500 MB |
| 20B | 80 GB | ~3 GB | ~3 GB + 500 MB |

The compression ratio improves with sub-byte coefficients (JS1.1 section).
The activation memory floor (~500 MB with ckpt/8) is fixed regardless of model
size — it depends on sequence length and hidden dim, not parameter count.

### Throughput scaling

Throughput (tokens/sec) scales as O(1/N) for a fixed hardware budget.

From the 1B baseline (measured on 5500U, JS1):
- 1B: ~7 tok/s
- Post-Monarch: ~25 tok/s (3.6×)

At 5B (5× the FLOPs per token):
- ~5 tok/s (post-Monarch on same hardware)

At 10B:
- ~2.5 tok/s

Chinchilla-optimal: N params needs O(N) tokens, O(N^1.5) total FLOPs.
For a 30-day training run at fixed throughput T tok/s:
- Max tokens D = T × 30 × 86400
- Max params N ≈ (D)^0.67 (Chinchilla scaling law)
- Doubling params needs ~3× more tokens → 3× more wall-clock time

### Multi-threaded layer parallelism

The current training loop processes one token at a time through each layer
(sequential). Within a layer, there are parallelism opportunities:

**Option A: Head-level parallelism (attention)**

Use `rayon::scope` or `std::thread::scope` to process query heads in parallel:

```rust
std::thread::scope(|s| {
    for (head_idx, head_slice) in q.chunks_mut(head_dim).enumerate() {
        s.spawn(|| {
            let kv_head = head_idx / group;
            flash.forward_single_head(head_slice, &k[kv_head*hd..], &v[kv_head*hd..]);
        });
    }
});
```

This gives n_heads× parallelism for attention (~40% of layer cost).

**Option B: Block-level parallelism (FFN)**

Process multiple active FFN blocks in parallel:

```rust
std::thread::scope(|s| {
    for &blk in &selected {
        s.spawn(|| {
            // Compute this block's up/gate/down independently
        });
    }
});
```

This gives n_active× parallelism for FFN (~40% of layer cost).

**Option C: Micro-batch parallelism (training)**

Process multiple tokens through the same layer simultaneously:

```rust
// Instead of: for token in tokens { layer.forward(token) }
// Do: layer.forward_batch(tokens)  // process all tokens in parallel
```

The Monarch block-GEMM naturally supports batching: each block processes
batch_size vectors simultaneously, giving batch_size× more work per block.
Scales linearly with cores until memory bandwidth becomes the bottleneck.

### Which to implement first

1. **Micro-batch parallelism** (Option C) — highest impact, most natural for
   training. Scales linearly with cores until bandwidth-bound.
2. **Head-level parallelism** (Option A) — easy, gives n_heads× for attention.
3. **Block-level parallelism** (Option B) — harder (load balancing), gives
   n_active× for FFN.

---

## JS2 — Multi-threaded layer parallelism

**Goal:** Parallelize within a single layer across multiple cores.

**Why:** The current training loop processes one token at a time through each
layer (sequential). Within a layer, there are parallelism opportunities:
- Attention heads are independent (14 query heads → 14-way parallelism)
- FFN blocks are independent (48 blocks, 12 active → 12-way parallelism)
- Monarch block-GEMM stages are independent

### Implementation approach

**Option A: Head-level parallelism (attention)**

Use `rayon::scope` or `std::thread::scope` to process query heads in parallel:

```rust
std::thread::scope(|s| {
    for (head_idx, head_slice) in q.chunks_mut(head_dim).enumerate() {
        s.spawn(|| {
            // Process this head independently
            let kv_head = head_idx / group;
            flash.forward_single_head(head_slice, &k[kv_head*hd..], &v[kv_head*hd..]);
        });
    }
});
```

This gives 14× parallelism for attention (limited by group size of 7, but the
query heads within a group are still independent).

**Option B: Block-level parallelism (FFN)**

Process multiple active FFN blocks in parallel:

```rust
std::thread::scope(|s| {
    for &blk in &selected {
        s.spawn(|| {
            // Compute this block's up/gate/down independently
        });
    }
});
```

This gives 12× parallelism for FFN (number of active blocks).

**Option C: Micro-batch parallelism (training)**

Process multiple tokens through the same layer simultaneously. This is the most
natural form of data parallelism and gives the best scaling:

```rust
// Instead of: for token in tokens { layer.forward(token) }
// Do: layer.forward_batch(tokens)  // process all tokens in parallel
```

The Monarch block-GEMM naturally supports batching: each block processes
batch_size vectors simultaneously, giving batch_size× more work per block.

### Which to implement first

1. **Micro-batch parallelism** (Option C) — highest impact, most natural for
   training. Scales linearly with cores until memory bandwidth becomes the
   bottleneck.

2. **Head-level parallelism** (Option A) — easy to implement, gives 14× for
   attention (which is ~40% of layer cost).

3. **Block-level parallelism** (Option B) — harder (load balancing across
   active blocks), gives 12× for FFN (which is ~40% of layer cost).

---

## JS2 — Mixed precision (F16/BF16) training

**Goal:** Train in F16/BF16 to halve memory bandwidth and potentially double
throughput.

**Why:** The model weights are stored in f32. If we train in F16, we halve the
memory traffic per token (weights, gradients, optimizer state). On a
bandwidth-bound workload, this doubles throughput.

### F16 support on x86-64

Modern x86-64 CPUs (Zen 4, Ice Lake, Sapphire Rapids) have:
- `_mm512_f16_ps` / `_mm512_ps_f16` conversion intrinsics (F16C extension)
- No native BF16 (that's Zen 5 / Granite Ridge)
- VNNI for INT8 dot products (not directly useful for F16)

So F16 training requires:
- Store weights in F16
- Convert to F32 for compute (AVX-512 F16→F32 conversion)
- Accumulate in F32 (loss scaling)
- Convert back to F16 for storage

The conversion overhead is ~1 cycle per 16 floats (trivial compared to the
compute). The real win is halving memory bandwidth.

### Implementation sketch

1. Store model weights as `Vec<half::f16>` (already a dependency: `half = "2"`)
2. In the matmul inner loop, load F16 weights, convert to F32, multiply,
   accumulate in F32
3. Store gradients in F32 (need precision for accumulation across micro-batches)
4. Optimizer state stays in F32 (AdaFactor's factored moment needs precision)

**Risk:** F16 has limited range (65504 max). Need loss scaling to prevent
gradient underflow. Standard practice: dynamic loss scaling that doubles the
loss every N steps until Inf/NaN appears, then halves.

---

## Open research questions

1. **Frequency-domain second moment:** Does FFT-ing the gradient before
   AdaFactor's rank-1 factorization make the factorization tighter? The
   conjecture is yes (energy compaction), but gradients are noisy — might be
   worse. Design an experiment in `optimizer.rs` and measure factorization
   error.

2. **INT8 momentum:** `QuantizedMomentum` exists in `types.rs` but is unused.
   Can we quantize the momentum buffer to INT8 with acceptable quality loss?
   Memory savings: 4× on the momentum buffer.

3. **Shared-core BTT nd tuning:** Gate (b) showed full-rank at nd≥8, but nd=4
   has anomalous conditioning. The real model uses nd=32 (over-parameterized).
   Can we reduce to nd=16 or nd=8 for more compression without hurting loss?
   This is a real experiment, not a guaranteed win.
