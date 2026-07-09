# Causal MonarchAttention — working journal

This is a scoped sub-investigation living entirely under
`monarch-attn-causal/`, separate from the project's main
`RESEARCH_LOG.md`. It picks up where that log's 2026-07-08
"naive drop-in doesn't work" entry left off: MonarchAttention
(Yaras et al., NeurIPS'25, `cjyaras/monarch-attention`) approximates
softmax attention via a two-stage Monarch-matrix factorization, but the
reference implementation only accepts a padding-style per-position
mask, never a real pairwise `(query, key)` causal mask — confirmed
empirically by feeding it a causal `(N,N)` mask (rejected on shape) and
then a padding-shaped mask (accepted, but leaks future tokens). This
journal tracks the attempt to build an actual causal variant, not just
document the dead end. Entries below are chronological within this
sub-investigation; nothing here is merged upstream or wired into the
project's Rust training loop yet.

---

## Mechanism trace

Traced `ma/ma_torch.py` exactly rather than reasoning about the "Sinkhorn-style"
label abstractly. Reshape `(E,H,N,D) -> (E,H,M,B,D)` is row-major:
sequence position `n -> (m = n // B, b = n % B)`. The algorithm alternates
two stages per iteration:

- **Stage 1, `al_cl_ref` (local):** within each block `m`, a query
  representative attends over keys `j` in the *same* block only
  (`B x B`, `M` batched). Produces `al[b, m]`, one summary per
  (intra-block slot `b`, block `m`).
- **Stage 2, `ar_cr_ref` (cross-block):** for each intra-block slot `b`,
  the actual query at block `m'` attends across all blocks' summaries
  `al[b, m]` (`M x M`, `B` batched), column-softmax'd.

Because the reshape is row-major, row-blocks are contiguous chunks of
the real sequence (causal cut = plain lower-triangular on intra-block
index) and column-blocks (fixed slot `b`, varying `m`) are strictly
increasing in sequence order with `m` (causal cut = plain
lower-triangular on block index). Composition is leak-free by
construction when both cuts are applied.

## Single-representative causal implementation

`ma_causal.py`: added `causal: bool` to a standalone copy of
`monarch_attention_torch`. Row-block mask = intra-block triangular
(`query_b >= key_b`), column-block mask = block triangular
(`key_block <= query_block`). `causal=False` reproduces the upstream
reference bit-for-bit (MSE 0.0 sanity check against `ma/ma_torch.py`).

**Result:** causal validity holds exactly — leakage ~1e-38 (float-min
underflow noise, not real leakage), row sums = 1.0. But approximation
quality vs. `F.scaled_dot_product_attention(..., is_causal=True)` was
consistently 4-8x worse (MSE) than the non-causal baseline's own
quality vs. its ground truth, and the gap widened with `N`:

| N | causal MSE / cos | non-causal baseline MSE / cos |
|---|---|---|
| 16 | 0.0795 / 0.796 | 0.0202 / 0.911 |
| 64 | 0.0974 / 0.713 | 0.0214 / 0.739 |
| 128 | 0.0488 / 0.650 | 0.0058 / 0.771 |

**Diagnosis:** not a `-inf` numerics problem. `al[b, m]` gets built
once per (b, m) with the row-causal mask applied, then *reused* by
every later query block `m' > m` in the cross-block pass — but a query
in a later block should see all of block `m` (entirely in its past),
not just the first `b` positions. Leak-free by construction, but
systematically under-attends to valid past context, worse for early
slots and more blocks — matching the observed trend of the gap growing
with `N`.

## Dual-representative fix (Option 1)

Cost estimate before building: doubling stage 1 (compute both a
causal-masked and a full/unmasked local representative per block) while
leaving stage 2's asymptotic shape unchanged should cost ~1.3-1.5x the
single-representative version per iteration, still `O(N^1.5 D)` overall,
not a new complexity class.

`ma_causal_dual.py`: computes `al_causal`/`al_full` (and at the final
step `y_causal`/`y_full`) per block each iteration, combines per
`(key_block, query_block)` pair via a diagonal selector — causal source
where `key_block == query_block`, full source where `key_block <
query_block`, masked out where `key_block > query_block`. The
value-aggregation split uses a masked-matmul identity
(`l * diag_mask @ y_causal + l * off_mask @ y_full`) rather than an
explicit gather, since `l * diag_mask` is nonzero only at the single
diagonal entry per query row.

**Result:** dual-rep roughly halves MSE and closes most of the cosine
gap versus single-rep, and at N=16/128 actually *exceeds* the
non-causal baseline's own cosine similarity to its ground truth:

| N | single MSE / cos | dual MSE / cos | non-causal baseline MSE / cos |
|---|---|---|---|
| 16 | 0.0795 / 0.796 | 0.0179 / 0.949 | 0.0202 / 0.911 |
| 64 | 0.0974 / 0.713 | 0.0429 / 0.843 | 0.0214 / 0.739 |
| 128 | 0.0488 / 0.650 | 0.0152 / 0.865 | 0.0058 / 0.771 |

Causal validity unchanged (still exact, same float-min noise floor).
Implementation note: used the masked-matmul combine (not the cheaper
diagonal-gather shortcut from the cost estimate), so actual overhead
here is closer to 2x than the theoretical 1.5x — an optimization to
revisit if this heads toward production, not done now.

**Status:** forward-pass only, verified numerically on toy sizes
(N=16/64/128, B=4, T=3). Not yet checked: gradients/autograd
correctness (nothing here has touched backward), and behavior at this
project's actual operating point (production B/T/N, not these toy
values). Both open as of this entry.

## Dual-rep optimization (`ma_causal_dual_opt.py`)

Question: was the dual-rep's measured ~2x overhead (vs. single-rep) a
real cost of the design, or implementation slack? The naive combine
(`ma_causal_dual.py`) computed two full `M x M` matmuls for the causal
branch just to keep a single diagonal entry per row. Replaced with the
algebraic shortcut: the diagonal of `al_c @ q^T` is just an elementwise
row-wise dot product (`(al_c * q).sum(-1)`), no `M x M` matmul needed,
combined into the full logits via `torch.diagonal_scatter`. Same trick
applied to the final step's value combination (diagonal weight extract
+ scale, instead of a masked matmul against `y_c`).

**Result:** numerically identical to the naive dual-rep (`ma_causal_dual.py`)
to fp precision (exact at N=16, ~1e-7 max diff at N=64/128 -- fp32
noise, not a discrepancy). Wall-clock (single-threaded CPU, no thermal
control -- same noise caveat as the main `RESEARCH_LOG.md`'s prior
benchmarking entries):

| N | B | single | dual (naive) | dual_opt | opt/single |
|---|---|---|---|---|---|
| 256 | 16 | 15.0ms | 18.8ms | 16.6ms | 1.10 |
| 512 | 16 | 15.6ms | 31.5ms | 28.5ms | 1.83 |
| 1024 | 32 | 42.3ms | 69.5ms | 52.5ms | 1.24 |
| 2048 | 32 | 91.6ms | 177.8ms | 149.3ms | 1.63 |

Optimized version lands around 1.1-1.8x single-rep cost (noisy band,
not a precise constant), closer to the ~1.3-1.5x theoretical estimate
than the naive version's ~1.6-2x. Conclusion: the original ~2x was
mostly implementation waste, not an inherent cost of the dual-rep
design -- the cost estimate roughly holds once the obvious redundant
matmuls are removed.

## Version B: local-exact + linear-recurrent-global hybrid (open, not resolved)

Different question, prompted by: could the off-diagonal (cross-block)
term be a running/incremental recurrent state a query reads directly,
instead of Monarch's per-iteration block-representative lookup? Two
sub-variants were possible:

- **Version A (exact running accumulator, flash-attention-style):**
  mathematically valid but gives no FLOP win -- it's dense causal
  attention rescheduled, same `O(N^2 D)` total, since every query still
  needs a weighted contribution from every past key. Not built.
- **Version B (kernelized/linear-attention recurrent state):** real
  complexity win -- off-diagonal drops from dual-rep's `O(B*M^2*D)` to
  `O(M*D*Dv)`, linear in `M`. Chosen to build. Known, not novel, risk
  going in: linear/kernelized attention is documented to underperform
  softmax attention on recall-heavy, copying-style tasks (same failure
  class as Infini-attention / gated-linear-attention hybrids) --
  explicitly accepted as the risk to test for, not treated as
  disqualifying up front. Also explicitly given up: the `T`-iteration
  alternating refinement doesn't carry over to a single left-to-right
  recurrent pass; only the (now much simpler, exact) diagonal block
  keeps any refinement, and even that's moot since exact softmax on a
  small block needs no iterative approximation at all.

Built `ma_causal_linear_hybrid.py`: exact causal softmax on the
diagonal block (`O(B^2 D)`, no Monarch approximation needed at that
size), `phi(x) = elu(x) + 1` (Katharopoulos et al., "Transformers are
RNNs") kernelized state for everything before it. Test plan: causal
validity (as before) + aggregate MSE/cos vs. exact causal softmax on
random Q/K/V + a **needle-in-haystack probe** (distinctive key/value
inserted at a fixed past position, query aligned with it at increasing
distances, measured against ground truth and a mean-V control) --
specifically because aggregate MSE on unstructured random data can't
surface a "did the model forget the one thing that mattered" failure,
which is the actual risk being tested for.

**Three combination attempts, three different failure modes:**

| variant | dist-0 cos (should be ~1.0, same-block) | dist-14 cos | aggregate cos @ N=1024 |
|---|---|---|---|
| naive additive sum | 0.1718 | 0.4222 | -- |
| count-normalized (S/Z / running count) | 0.5664 | 0.3380 | 0.601 |
| log-domain joint softmax (mean-key pseudo-token) | **1.0000** | -0.1217 | 0.220 |
| dual-rep Monarch (`dual_opt`), for reference | 1.0000 | 0.3831 | 0.771 |

1. **Naive** (`out = (num_diag+num_off)/(denom_diag+denom_off)`): bug
   was scale, not concept -- `denom_off` is a raw running sum that grows
   with sequence length while `denom_diag`'s exp-scores are capped near
   O(1) per key after max-subtraction. Let background noise swamp a
   correct local answer even at zero distance.
2. **Count-normalized** (divide S/Z by running key-count before
   reading, so the global branch reads as *one averaged pseudo-key*):
   fixed the growth-with-length artifact (dist-0 cos roughly tripled)
   but didn't fully fix it -- `phi(q)*phi(k)` is still an unbounded raw
   dot product with no saturation cap the way `exp(score)` has, so no
   constant rescaling makes the two families genuinely comparable.
3. **Log-domain joint softmax** (real dot-product logit against a mean
   of past raw keys, `k_bar = mean(K)`, folded into the *same* softmax
   as local scores instead of combined after normalization): fixed the
   calibration problem completely (dist-0 now exact, 1.0000) but
   revealed a different, deeper problem -- `k_bar` is a flat average
   of every past key, and averaging is exactly the operation that
   destroys a single outlier's signature. The needle's direction
   becomes `~1/count` of the mean as history grows, and for generic
   zero-mean random keys `k_bar -> 0` as count grows, so the model
   defaults to "trust local only, ignore global" almost everywhere --
   explaining both the needle collapse at distance >0 *and* the
   aggregate-quality drop (0.68 -> 0.41 -> 0.22 as N grows: the model
   is increasingly attending to nothing but its own diagonal block).

**Read:** this isn't a calibration bug anymore, it's a capacity limit --
a single averaged summary vector cannot simultaneously represent
"nothing notable happened" and "there was one specific important thing
here." A real fix needs a summary that preserves outliers instead of
averaging them away: multiple summary slots with routing/clustering
(Compressive-Transformer / Memorizing-Transformer style), or a
max-pooled rather than mean-pooled key sketch. That's a materially
bigger build than anything tried so far, not a follow-up tweak.

**Status: open, not closed.** Three combination attempts have now
landed on three distinct failure modes (scale bug -> partial fix ->
capacity limit), and `dual_opt` (Monarch) has beaten every hybrid
variant on both aggregate quality and the needle test at every distance
tried. The multi-slot fix is the identified next move if this direction
gets picked back up, but it hasn't been attempted -- explicitly not
treating the current state as a verdict against Version B, just as
where the investigation paused.

---

## Closing entry: status at end of tonight's session

Full arc, for the record. Not a final verdict on causal Monarch
attention as a research direction -- a checkpoint of what's solved,
what's ruled out and why, and what's scoped but unbuilt.

### Solved: causal masking for MonarchAttention

Row-block triangular masking (contiguous chunks of the sequence,
free/exact by the reshape's own structure) + column-block masking
(exploiting that the row-major reshape makes column order equal to
sequence order within a fixed column, so it's triangular too, not
scrambled -- see "Mechanism trace" above). The dual-representation fix
(`ma_causal_dual.py`, optimized in `ma_causal_dual_opt.py`) resolved
the specific recall bug this uncovered: a single row-causal-masked
block representative was being reused by every downstream causal query
regardless of how much of that block they were actually causally
entitled to see.

**Verified:** 0 leakage (float-min noise floor, not real leakage), row
sums exactly 1.0, `causal=False` reproduces the upstream reference
bit-for-bit (MSE 0.0). Cost overhead measured at ~1.1-1.8x the
non-causal baseline (averaging ~1.3-1.5x across N=256..2048) -- a
constant factor, confirmed via direct scaling checks not to erode
Monarch's `O(N^1.5)` vs. dense `O(N^2)` asymptotic advantage as N
grows.

### Ablation study: three off-diagonal alternatives, all non-competitive with `dual_opt`

Each ruled out for a distinct, well-characterized reason -- not "didn't
work," but a specific, reproducible failure mode:

1. **Compression** (mean-pooled fixed-size state; 3 combination
   attempts -- naive sum, count-normalized, log-domain joint softmax).
   Fails structurally: a single averaged summary vector cannot
   simultaneously represent "nothing important happened" and "one
   specific important thing happened." Not a calibration bug -- a
   capacity limit. Not fixable by recalibration alone (all three
   attempts tried gradually more careful calibration and still failed,
   just via different mechanisms each time).
2. **Exact top-k retrieval.** Excellent when uncontested -- perfect
   recall (cos 1.0000) at every distance tested, no dilution, no
   retrieval miss with a strong needle even at k=8 against a 240-key
   pool. But has two real, sharp (cliff, not gradual) failure modes:
   a **weak-signal cliff** (recall collapses below signal_scale ~1.5,
   going *negative* -- actively retrieves anti-correlated noise, not
   just "loses the signal") and a **decoy/rank-competition cliff**
   (recall collapses from 0.95 to negative once decoys outnumber k,
   with the needle's own signal strength held completely unchanged --
   it's relative rank among competitors, not absolute strength, that
   breaks it). Larger k does not reliably rescue weak-signal recall
   (k=64 measured *worse* than k=8 at one tested point -- diluting
   softmax weight even when technically still "in" the candidate set),
   though it does help aggregate quality monotonically on unstructured
   data. Cost: genuinely `O(N^2)` by design (brute-force, deliberately
   unoptimized for this probe) -- already 2-3x slower than `dual_opt`
   at N=256..2048 and the gap widens with N.
3. **Approximate/ANN top-k** (coarse dot-product k-means clustering,
   `sqrt(P)` buckets, probe nearest 3). Inherits both of exact top-k's
   cliffs, with somewhat added noise on top (e.g. non-monotonic recall
   vs. distance at low signal_scale, worse than exact top-k's own
   already-noisy pattern there). Its cost numbers are not trustworthy
   as ANN cost data: this reference implementation still computes the
   full `B x P` exact score matrix and masks it down to the probed
   clusters, rather than skipping scoring for unprobed points the way
   a real ANN index would -- so the ~2-5x-slower-than-exact-top-k
   numbers measured reflect clustering overhead *added on top of*
   exact top-k's own cost, not what a well-engineered ANN would
   actually cost. Useful as an early signal that approximation doesn't
   obviously *fix* either cliff, not useful as real performance data.

### Derived principle

Fixed, content-independent scope (the Monarch / sparse-attention /
sliding-window-attention family) is the reliability backbone attention
needs: it can degrade in *weight* (approximation error on which
in-scope tokens get how much attention) but it never silently loses
something that was already in scope -- scope is decided by position,
not by a runtime competition. Rank-based / contested-scope methods
(top-k retrieval, ANN, and by extension anything that decides *what's
visible* based on a runtime similarity competition rather than a fixed
structural rule) can lose the correct answer entirely under
competition, independent of the target's own actual signal strength --
this is a failure of attention's core reliability promise, not a
tunable inefficiency to optimize away. Conclusion for this project:
rank-based retrieval methods should be non-load-bearing supplements at
most, never the backbone attention mechanism.

### Scoped for later, not pursued tonight

Potential follow-up contributions to MonarchAttention's own open
questions (paper-level, not just this project's training loop):

- **SlidingMonarchAttention** -- decouple the local exact-window size
  from the block size used for the off-diagonal/column-pass term (a
  wider exact SWA-style local window, independent of Monarch's own
  block granularity). Fixed-scope, consistent with the derived
  principle above. Not yet implemented or tested in any form.
- **FlashMonarchAttention** -- kernel-fusion (FlashAttention-style
  memory-I/O optimization) applied to Monarch's block-local softmax.
  Plausible, well-precedented as a technique category, but expected
  payoff is likely GPU-dominant rather than CPU-relevant for this
  project's actual target hardware (R5500U) -- the CPU cache hierarchy
  doesn't have the HBM/SRAM gap that motivates FlashAttention on GPU.
  Distinct from Monarch's own FLOP-reduction advantage, which *is*
  CPU-relevant and already evidenced by SharedMonarch's existing wins
  elsewhere in Jumping Seedling.
- The Sinkhorn-iteration connection to the `T`-round refinement loop --
  relevant if a richer/wider fixed sparsity pattern gets explored
  later, not investigated tonight.

### Explicitly deferred, not decided against

Real next steps that need a trained model or the actual production
kernel stack, not synthetic Q/K/V probes:

- Validating against actual next-token loss in a trained model, rather
  than synthetic-data approximation-quality/needle metrics.
- Head-to-head wall-clock vs. the current inference stack's actual
  attention kernel on the 5500U (everything measured tonight was
  single-threaded, unoptimized PyTorch reference code, not the
  project's Rust/AVX2 kernels).

**Status: paused, not concluded.** Causal masking is a solved,
verified building block (`ma_causal_dual_opt.py`). The off-diagonal
mechanism question has three ruled-out alternatives with clear reasons
and one open, well-scoped direction (fixed-scope extensions like
SlidingMonarchAttention) that hasn't been touched yet.

---

## SlidingMonarchAttention: first build, correctness-only, strong result

Follow-up session, picking up the one open direction from the closing
entry above. Decouples the local exact-window size from Monarch's
block size, per the derived reliability principle: fixed,
content-independent scope for both the window boundary and the
Monarch block-triangular boundary -- never a runtime rank competition.

**Design simplification found while building it:** the earlier
dual-representation trick (`ma_causal_dual_opt.py`) was needed only
because, at window=1 block, the diagonal block had to serve both its
own causal self-attention *and* be reused unmasked by later blocks --
two conflicting visibility requirements on one representative. With a
genuine multi-block sliding window (`W_blocks >= 1`) handling all
self-visibility exactly, every block Monarch ever serves is already
permanently "closed" (past the window) for every query that reads it --
so the far/Monarch part needs only ONE representative, not two.
Structure: (1) local window, exact causal softmax over the trailing
`W_blocks` blocks, no Monarch approximation at all; (2) far blocks
(strictly before the window), Monarch's single-representative
mechanism, refined over `T-1` iterations using a block-triangular mask
SHIFTED by `W_blocks` (`m_key <= m_query - W_blocks`) so the iterative
refinement itself stays causally safe -- an unmasked refinement would
let future blocks contaminate a past block's representative through
the alternating rounds, the same leak the very first `causal_probe.py`
caught for the fully non-causal reference. Far-block logits are kept
UNCOLLAPSED (one real candidate per far block) and combined with the
local window's per-token logits in one joint softmax -- avoiding
Version B's mean-pooling trap at block granularity instead of token
granularity.

**Bug found and fixed during implementation:** early query blocks
(`m_q < W_blocks`) have no valid far candidates yet -- an all `-inf`
softmax column, which raw `F.softmax` turns into 0/0 = NaN that then
cascades through later `T`-iterations via the shared `ar`/`cr` state.
Fixed with the same manual max-subtract + `nan_to_num` pattern already
used elsewhere in this codebase instead of raw `F.softmax`.

**Results (correctness only, cost not yet measured):**

- Causal validity exact at every `W_blocks` tested (1/2/4/8): 0
  leakage, row sums 1.0.
- Needle-in-haystack (signal_scale=6.0): W=1 tracks `dual_opt` almost
  exactly (0.380 vs 0.383 cos at distance 14 blocks) -- sanity check
  that the single-representative simplification didn't break anything.
  Widening the window recovers recall **smoothly and monotonically,
  no cliffs**: W=8 hits 0.9842 cos even 14 blocks out, purely from a
  fixed, position-derived rule.
- Signal-strength sweep (W=4, hardest distance): even ground truth
  degrades hard at low signal beyond the window (this is inherently a
  hard retrieval problem past W blocks, not a method-specific
  failure). Critically, W=4 converges toward roughly GT's own noise
  floor (~0.27 vs ~0.28) rather than diverging negative the way exact
  top-k's weak-signal and decoy cliffs did -- the qualitative
  difference the derived principle predicted: approximate in *weight*,
  never catastrophically wrong.
- Aggregate quality (random Q/K/V): monotonic improvement with W
  (0.965 -> 1.000 cos at N=64 for W=1->8), and even W=1 beat
  `dual_opt` (0.965 vs 0.917 at N=64) -- mechanism for that specific
  gap not investigated, noted but not explained.

**Status: correctness validated, cost not yet measured.** This is the
first mechanism tested tonight that shows no cliff behavior anywhere --
consistent with the derived principle that fixed-scope methods degrade
in weight, not in correctness. Files: `ma_sliding_monarch.py`,
`eval_sliding_monarch.py`. Next: wall-clock cost (current local-window
implementation is an unoptimized per-block Python loop, no efficiency
attempt made yet).

**Cost (follow-up, same session):**

| N | noncausal | dual_opt | W=1 | W=2 | W=4 | W=8 | W=4/dual_opt |
|---|---|---|---|---|---|---|---|
| 256 | 6.4ms | 10.8ms | 17.5ms | 18.5ms | 16.6ms | 18.2ms | 1.54x |
| 512 | 25.2ms | 21.8ms | 40.2ms | 36.7ms | 46.4ms | 45.1ms | 2.12x |
| 1024 | 29.3ms | 42.3ms | 64.1ms | 79.1ms | 74.5ms | 96.3ms | 1.76x |
| 2048 | 64.1ms | 118.9ms | 300.9ms | 287.7ms | 264.3ms | 354.3ms | 2.22x |

Roughly 1.5-2.2x `dual_opt`'s cost (usual noisy single-threaded CPU
caveat -- no thermal/affinity control). Cost barely changes across
W=1->8 at these sizes: the per-block Python loop overhead currently
dominates over the actual window-size matmul, so widening the window
is close to free in this unoptimized reference implementation. Scaling
check (W=4) shows ~2.0-2.45x growth per N-doubling -- below the ~4x a
true `O(N^2)` method would show, closer to Monarch's own `O(N^1.5)`
(~2.8x), consistent with the design (fixed window adds only an
`O(N*W)` term on top of the Monarch far branch's `O(N^1.5)`), though
not a rigorous asymptotic proof given the noise. Files:
`eval_sliding_monarch_cost.py`.

**Net for tonight:** SlidingMonarchAttention is the strongest result of
the whole causal-MonarchAttention investigation -- no correctness
cliffs anywhere tested, a genuine controllable recall/window knob, and
cost that's a modest constant-factor premium over `dual_opt` rather
than a different complexity class.

**Decoy-pressure stress test (parity check against the same test that
broke exact top-k):** needle scale fixed at 3.0, query at distance 14
blocks (beyond the window at every W tested), decoy count swept
0/5/20/50, same decoy generation as `eval_topk_stress.py`.

| num_decoys | W=1 | W=4 | W=8 |
|---|---|---|---|
| 0 | 0.2520 | 0.2868 | 0.3218 |
| 5 | 0.2381 | 0.2592 | 0.2562 |
| 20 | 0.1972 | 0.1637 | 0.1186 |
| 50 | 0.1800 | 0.1444 | 0.1135 |

(0-decoy values match the earlier signal-strength sweep exactly --
determinism check.) Confirms the derived principle directly: recall
declines gradually and monotonically under mounting decoy pressure but
**stays positive throughout, even at 50 decoys** -- sharply unlike
exact top-k's collapse to -0.10/-0.11 under the identical setup.
Fixed-scope block candidates degrade in weight under competition, they
don't get bumped out of relevance entirely the way a rank-based top-k
slot can. Files: `eval_sliding_monarch_decoy.py`.

---

## Multi-slot compression (Version B-2): the identified fix for
## Version B doesn't work cleanly, and here's the likely reason

Follow-up attempt at the specific fix Version B's closing read named:
"a real fix needs a summary that preserves outliers instead of
averaging them away: multiple summary slots with routing/clustering."
Built `ma_causal_multislot.py` -- same joint-softmax combination
validated in Version B's attempt 3 (real dot-product logits, one
shared softmax with the local diagonal block, no scale mismatch), but
with `n_slots` independent running `(K_sum, V_sum, count)`
accumulators instead of one. Each causal key routes online to its
nearest existing slot centroid (argmax dot-product similarity).

**Causal design note:** deliberately did NOT pre-seed slot centroids
from a fixed set of early keys -- that would bake future-relative-to-
early-queries content into the slot state before block 0's own causal
window is processed, a real leak caught during design, not by
accident. Slots start genuinely empty.

**Causal validity: exact** (0 leakage, row sums 1.0) at n_slots =
4/8/16, same as every other mechanism tonight -- the correctness
scaffolding transfers cleanly regardless of the routing decision.

**Results: does NOT cleanly fix the capacity limit.**

| dist | GT | 1slot | 4slot | 8slot | 16slot |
|---|---|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 | 1.0000 | 0.1200 | 0.0657 | 0.0657 | 0.0657 |
| 5 | 1.0000 | 0.0074 | -0.1257 | -0.1257 | -0.1257 |
| 9 | 1.0000 | 0.3631 | 0.3872 | 0.3872 | 0.3872 |
| 14 | 1.0000 | -0.1217 | 0.0202 | 0.2082 | 0.2082 |

Decoy-pressure baseline (0 decoys, needle scale=3.0, dist=14) is
already near-zero or negative (-0.07 at 4 slots, 0.03 at 8/16 slots) --
worse than SlidingMonarchAttention's 0.29 baseline at the identical
distance/scale, with no clean monotonic trend as decoys are added
(bounces between -0.05 and 0.10). Aggregate quality improved modestly
over single-slot (0.38 -> 0.47 cos at N=256) but nowhere near
SlidingMonarchAttention's 0.87-0.96 range.

**Diagnosis:** 8-slot and 16-slot results are IDENTICAL in every row
tested -- strong evidence most slots never actually get used. This
matches a cold-start problem flagged as a known caveat during design:
an empty slot's centroid is exactly zero, so argmax-routing dumps
every early key into slot 0 (via tie-breaking) until its centroid
moves away from zero and later keys start routing elsewhere. The
needle sits at position 18 -- inside that cold-start window -- so this
test may be measuring the routing-collapse artifact more than the real
"does multi-slot fix capacity" question. A fair test needs either a
smarter causal seeding/routing scheme (real engineering -- e.g.
k-means++-style causal seeding) or a needle placed later in the
sequence to isolate steady-state routed behavior from cold-start noise.
Neither attempted yet.

**Status: negative result, not fully conclusive.** The "obvious" fix
for Version B's capacity limit doesn't work out of the box, likely
confounded by a routing cold-start bug rather than proof the multi-slot
idea itself is unsound. SlidingMonarchAttention remains the only
mechanism in this whole investigation with clean, graceful, no-cliff
behavior end to end. Files: `ma_causal_multislot.py`,
`eval_multislot.py`.

---

## SlidingMonarchAttention structural variant: decoupled refinement
## shift -- causally safe, empirically worse (a validated negative result)

Follow-up on the "different far-block shift scheme" option from the
earlier menu. First checked whether dual-representation (the
single-block-diagonal trick from `ma_causal_dual_opt.py`) was secretly
still needed somewhere in `ma_sliding_monarch.py` -- traced it through
and confirmed NO: `al_full[m]`/`ar[m]` only ever pool from blocks `<=
m` by induction, for ANY shift value >= 0, so a single representative
is provably sufficient regardless of window width. Nothing to fix
there; the original simplification was correct, not a shortcut.

**The actual variant tried:** the internal T-iteration cross-block
refinement was using the same wide `W_blocks` shift as the final read
step, but the causal-safety argument above shows the internal shift
doesn't need to match -- `al_full[m]`'s safety doesn't depend on it.
Hypothesis: decoupling them (`W_refine=0`, plain causal, for the
internal refinement; `W_blocks` kept only at the final read) should
let the refinement use more legitimately-available context and
therefore improve quality, for free.

**Measured: the hypothesis was wrong.** `W_refine=0` is consistently
*worse* than `W_refine=W_blocks`, not better:

Needle test (signal_scale=6.0), W_blocks=4:
| dist | GT | old (Wr=W) | new (Wr=0) |
|---|---|---|---|
| 5 | 1.0000 | 0.9989 | 0.7269 |
| 9 | 1.0000 | 0.8830 | 0.3597 |
| 14 | 1.0000 | 0.6443 | 0.3479 |

Aggregate quality also drops slightly and consistently (e.g. 0.958 ->
0.951 cos at N=256, W=4) -- smaller effect than the needle test but
the same direction.

**Read:** causally safe is not the same as quality-neutral. Letting
the internal refinement pull in blocks closer to each round's own
reference point than the eventual final-read shift allows lets those
nearer blocks dominate the cross-block attention mass during
refinement, pulling the representative away from a more evenly-
considered view across the full valid far range -- a real quality
cost from an optimization that has zero correctness cost. Worth
recording precisely because it's a trap: nothing about causal_probe.py-
style leakage checks would ever catch this, since there IS no leak --
only a live needle/aggregate-quality test surfaces it.

**Outcome:** kept `W_refine` as an explicit, optional parameter (useful
for further experimentation) but changed the default to `None` ->
resolves to `W_blocks`, i.e. the original, validated, better-performing
behavior -- not 0. Every existing caller and eval script is unaffected
(same default behavior as before this variant was tried). Files:
`ma_sliding_monarch.py` (parameter added), `eval_sliding_monarch_refine.py`.

---

## Two more variants: B independent of window width (confirms the
## founding premise), and T-sensitivity (the strongest lever found yet)

**A) Window width held constant in tokens, `B` varied.** Tests the
actual founding premise of SlidingMonarchAttention -- "decouple window
size from Monarch's block size" -- which no prior test had isolated
(everything so far varied `W_blocks` with `B` fixed at 16). At a fixed
64-token window: aggregate quality N=256 -- B=16,Wb=4: 0.9514; B=8,Wb=8:
**0.9539**; B=32,Wb=2: 0.9475. Needle test makes the trend clearer:

| dist | B=16,Wb=4 | B=8,Wb=8 | B=32,Wb=2 |
|---|---|---|---|
| 9 | 0.8830 | **0.9491** | 0.7177 |
| 14 | 0.6443 | **0.7445** | 0.6833 |

Finer `B` consistently helps, coarser `B` consistently hurts, at
identical window width -- confirms the founding premise directly, not
just as a proxy for window width. Smaller `B` gives the far branch's
per-block representative less internal averaging blur even though the
exact window covers the same token span either way. Design guidance:
default `B` smaller rather than larger for any fixed window budget.

**B) T-sensitivity (`B=16, W_blocks=4` fixed), the strongest lever
found in the whole investigation:**

| dist | T=1 | T=2 | T=3 | T=5 | T=8 |
|---|---|---|---|---|---|
| 5 | 0.0984 | 0.9630 | 0.9989 | 1.0000 | 1.0000 |
| 9 | 0.0986 | 0.5900 | 0.8830 | 0.9980 | 1.0000 |
| 14 | 0.2295 | 0.4939 | 0.6443 | 0.8043 | **0.9387** |

`T=1` (zero refinement rounds -- `ar` never leaves its raw initial
value) is nearly useless at distance. Recall climbs steeply and
monotonically with `T`, reaching near-perfect by `T=8` even 14 blocks
out -- beating what widening `W_blocks` alone achieved at `T=3`.
Aggregate quality saturates faster (0.91 -> 0.94 cos at N=256, mostly
done by `T=5`), so the needle-specific gain is the more dramatic
effect. Unlike the `W_refine` shift experiment, this is a clean,
monotonic, no-trap result -- more refinement rounds is unambiguously
good here, not just theoretically safe.

**Design guidance for if this gets built out further:** `T` is the
cheap, high-leverage quality knob (diminishing but still positive
returns past `T=5`); `B` should default smaller rather than larger for
a fixed window budget; widening `W_blocks` itself remains useful but is
not the only, or even the strongest, lever available. Files:
`eval_sliding_monarch_variants.py`.

---

## FlashMonarchAttention: initial result looked like a reversal of the
## earlier "GPU-dominant, not CPU-relevant" call -- isolation showed it wasn't

Picked up the one item from the original closing entry's "scoped for
later" list that had been explicitly deprioritized: kernel fusion
(FlashAttention-style, avoid materializing intermediate score tensors)
applied to Monarch's block-local softmax. Original reasoning for
deprioritizing it: FlashAttention's core motivation is avoiding
HBM<->SRAM round-trips on GPU, and this project's CPU target doesn't
have that specific memory-hierarchy gap. Went back to test that
skepticism empirically rather than leave it asserted.

**First test (naive eager PyTorch, 3 separate ops: matmul, softmax,
matmul) vs `F.scaled_dot_product_attention` (PyTorch's real fused/
memory-efficient CPU kernel, a legitimate stand-in for "the flash
technique" -- not GPU-only) showed a 2-9x speedup for fusion, INCLUDING
at Monarch's actual small operating scale (B=16-64, 2-5x)** -- looked
like a genuine reversal of the deprioritization, with a plausible
mechanism (CPUs have their own fast/slow memory tiers -- L1 vs L2/L3/
RAM -- so the "avoid re-reading intermediate tensors" argument isn't
inherently GPU-specific).

**Isolation test before trusting it:** naive is always exactly 3
op-launches, fused always 1, regardless of batch size `M` (number of
parallel blocks). If the speedup were a genuine memory-traffic/
algorithmic effect, it should persist or grow as `M` scales. If it's
mostly fixed per-launch dispatch overhead, it should shrink toward 1x
as `M` grows large enough to amortize 3-vs-1 launches away.

| M | naive | fused | speedup | naive/block | fused/block |
|---|---|---|---|---|---|
| 1 | 0.089ms | 0.028ms | 3.23x | 88.7us | 27.5us |
| 16 | 0.123ms | 0.033ms | 3.71x | 7.7us | 2.1us |
| 64 | 0.171ms | 0.083ms | 2.07x | 2.7us | 1.3us |
| 256 | 0.382ms | 0.311ms | 1.23x | 1.5us | 1.2us |
| 4096 | 11.269ms | 12.132ms | **0.93x** | 2.8us | 3.0us |

**Result: the speedup collapses toward 1x and then goes BELOW 1x** --
at M=4096, naive separate ops are slightly FASTER than the fused
kernel per block. This is the signature of dispatch overhead, not a
persistent algorithmic advantage: at small M, three op launches each
pay a fixed cost that dominates the tiny real work per call; fusion
mostly just avoids paying that fixed cost three times, not "moving
data through cache more efficiently" (the actual FlashAttention claim).

**Corrected conclusion: the original deprioritization was right.** Once
dispatch overhead is controlled for via scaling, kernel fusion shows no
genuine, scale-persistent advantage at Monarch's actual block size and
realistic block counts on this CPU target -- consistent with the CPU
cache hierarchy not having GPU's dramatic HBM/SRAM bandwidth gap. The
first test's apparent reversal was a PyTorch-dispatch-overhead
artifact, not a real signal about the underlying technique's
CPU-relevance.

**Tooling note:** `torch.compile` (the originally planned isolation
method) failed outright in this environment -- the project path
contains a space (`! Codes`), and torch's C++ JIT linker step doesn't
quote its `-L` flags, so it can't find `libtorch`. Not attempted to fix
(would mean moving the project directory mid-investigation). Pivoted to
the scale-based isolation above instead, which needed no JIT compiler
and gave a cleaner, more directly interpretable result anyway.

**Status: CPU verdict resolved, GPU question left explicitly open --
not the same thing.** What's actually settled: on THIS CPU target, at
Monarch's real operating scale, kernel fusion shows no genuine
scale-persistent win once dispatch overhead is controlled for -- tested,
not assumed. What's NOT settled: whether FlashMonarchAttention would
win on GPU, where the HBM<->SRAM gap that motivates FlashAttention in
the first place actually exists. That question is left open deliberately
-- not because it can't be reasoned about algorithmically (the
underlying argument for why it should plausibly help on GPU is fine),
but because there's no GPU available in this environment to actually
measure it, and asserting an answer either way without being able to
test it would be exactly the kind of unverified claim this whole
session has been trying not to make. Don't read "deprioritized for THIS
project's CPU target" as "the technique doesn't work" -- those are
different claims, and only the first one has evidence behind it here.
The process -- surprising result, then isolate before trusting it -- is
itself worth having on record too, since the naive first measurement
would have been actively misleading taken at face value. Files:
`eval_flash_hypothesis.py`, `eval_flash_isolation.py` (failed, kept for
the tooling note), `eval_flash_isolation2.py`.

**GPU question closed (follow-up, real hardware).** Built a standalone
script + Colab notebook (`gpu_flash_benchmark.py`,
`gpu_flash_benchmark_colab.ipynb`) running the identical M-scaling
isolation methodology, and got it run on a real GPU via Colab (T4-class).
Real results, M fixed B=16:

| M | speedup |
|---|---|
| 1 | 10.95x |
| 4 | 6.54x |
| 16 | 9.54x |
| 64 | 8.02x |
| 256 | 3.23x |
| 1024 | 1.34x |
| 4096 | **0.69x** |
| 16384 | **0.84x** |

Same collapse pattern as the CPU result: high speedup at small M
(dispatch-overhead territory), falling toward and then BELOW 1x by
M=4096-16384 -- the fused kernel ends up slightly slower than naive
separate ops once the fixed per-launch cost is amortized away. (A crash
at M=65536 is a separate, unrelated CUDA grid-dimension hardware limit
-- 65536 is exactly one past the 65535 cap on a grid axis -- not a real
result; doesn't affect the trend, which is already clean and monotonic
from M=1 through M=16384.)

**This closes the open question, on real hardware, not just CPU
extrapolation:** kernel fusion does NOT show a persistent,
scale-independent advantage at Monarch's actual block granularity
(B=16), even on a GPU where the HBM<->SRAM gap that motivates
FlashAttention genuinely exists. Likely reason: a 16x16 score tensor
(1KB) is small enough to stay in fast on-chip memory/registers
regardless of fusion -- there's little HBM traffic to save at this
block size in the first place, on CPU or GPU.

**Part 1 (straight size sweep, single big attention call, N=16..8192)
results came back too, and they complete the picture rather than just
adding a data point:**

| size | speedup |
|---|---|
| 16 | 6.02x |
| 32 | 5.33x |
| 64 | 4.74x |
| 128 | 2.96x |
| 256 | **1.42x** (minimum) |
| 512 | 1.84x |
| 1024 | 3.54x |
| 2048 | 4.04x |
| 4096 | 4.59x |
| 8192 | 5.09x |

A U-shape: high at tiny N (dispatch overhead, same effect as everywhere
else in this investigation), dips to a minimum around N=256, then rises
steadily back up through N=8192 -- because a SINGLE attention call's
intermediate score matrix grows quadratically with N, so at large N
materializing and re-reading it becomes genuinely expensive, and fusion's
advantage (avoiding that) grows right along with it.

**The contrast between the two tests is the actual finding:** the
M-scaling isolation test grows the *number of independent small blocks*
(M) -- that doesn't make any single intermediate tensor bigger, just
adds more small, cheap, parallel work, so fusion's advantage collapsed
toward/below 1x there. This test grows the *size of one attention call*
(N) -- that does make the intermediate tensor bigger, so fusion's
advantage grows here instead. FlashAttention's real strength is
specifically about the second axis (large single-sequence attention, the
classic long-context LLM use case it was built for). Monarch's actual
shape -- many small, fixed-size B=16 blocks, batched over M -- lives
entirely on the first axis, where fusion doesn't help. Same conclusion
as the CPU result, but now with a concrete mechanistic explanation for
*why* the two shapes behave so differently, not just an empirical
correlation.

**Status: fully resolved.** FlashMonarchAttention is deprioritized on
tested grounds now, CPU and GPU both, with both the "it doesn't help
here" and "it clearly would help at a different shape" halves of the
story on record -- not an assumption, not an extrapolation from
CPU-only data. Files: `gpu_flash_benchmark.py`,
`gpu_flash_benchmark_colab.ipynb`.

---

## MetaMonarchAttention: multi-scale hierarchy over SlidingMonarchAttention

Design attempt to replace SlidingMonarchAttention's flat far-branch
(one Monarch representative per block, same resolution regardless of
distance) with a fixed geometric hierarchy -- finer near the window,
coarser further out. Hard constraints carried over from the whole
night's findings: block assignment fixed by geometry alone (never
content/rank -- the property that made Causal and Sliding Monarch work
and every rank-based alternative fail), and every block's representative
must stay an uncollapsed, genuine per-block candidate combined via one
joint softmax (never pre-blended across blocks within a level, or this
just reinvents Version B's mean-pooling failure at that level's
granularity).

**Design found in three real problems, each surfaced and resolved (or
not) in sequence, by hand-tracing before writing code each time:**

1. **Naive "reach" scheme (token-distance-based level boundaries) --
   broken.** Traced on `B=4,K=2,R=2,t=50`: level-1's intended 16-token
   range spanned THREE of level-1's actual global blocks, not two,
   because `t=50` wasn't aligned to level-1's grid. Position-dependent
   gaps/double-counting near level boundaries, not a rounding nicety.

2. **Fixed via canonical binary (Fenwick-tree / BP-Transformer-style)
   decomposition.** For query base-block `m0`, decompose the causal
   prefix using globally-fixed dyadic blocks per level: level `l`
   contributes a candidate iff bit `l` of `n = m0 - W_blocks + 1` is
   set, with a closed-form block index `(n >> (l+1)) << 1`. Verified by
   hand on `n=13` (`1101` -> blocks `12,2,0` at levels `0,2,3`,
   matching `[12,13)+[8,12)+[0,8)`, zero gaps/overlaps) and `n=20`
   (`10100` -> blocks `4,0` at levels `2,4`). Provably exact, globally
   reusable blocks (unlike query-relative boundaries, which would have
   defeated precomputation). This part held up completely -- never
   revisited or found wrong later.

3. **Found DURING implementation: O(N^2/B) read-cost wall.** Monarch's
   own local pass structurally produces `B_l` (block size) representatives
   per block, and reading ALL of them per query, when roughly half of
   all queries touch each level regardless of that level's size, sums
   to `O(N^2/B)` total -- silently destroying the `O(log N)` win the
   binary decomposition was supposed to buy. Caught before writing
   further code, not discovered via a failing benchmark after the fact.

**Reconsideration (user-prompted) resolved this cleanly:** decouple
per-block representative count from block size via a FIXED `R`,
precomputed once per block (not per query) and reused by every future
query, giving `O(R * N * log(N/B))` total cost -- genuinely
sub-quadratic. Checked explicitly whether this reintroduces Version B's
capacity problem: partially, honestly -- attention-based `R` slots are
categorically different from Version B's flat mean (a slot CAN
concentrate sharply on one standout key, the same mechanism that gave
Monarch's window-level local pass clean 1.0 needle recall all night),
but `R` fixed slots still have a hard ceiling: more than `R`
genuinely-distinct simultaneous needles in one block WILL collide. That
was the thing the multi-needle stress test below was built to measure,
not assume.

### Implementation (`ma_meta_monarch.py`) and what the stress test actually found

Built: local exact window (unchanged from SlidingMonarchAttention) +
binary-decomposition tier selection + per-tier `R`-landmark compression
(R evenly-spaced real keys from the block, each attending over all of
the block's own keys, Nystrom-style -- reusing real content as the
"pseudo-query" since this is an untrained probe with no learned
landmark parameters available) + one joint softmax combining window and
all active tiers via logsumexp composition (exact, not an
approximation, given the per-tier internal softmax). T-iteration
cross-block refinement (SlidingMonarchAttention's strongest lever) was
deliberately NOT included in this build -- kept the scope to the
R-capacity question specifically.

**Causal validity: exact** (0 leakage, row sums 1.0) -- the geometric
scaffolding transferred cleanly despite the much more involved
selection logic.

**Multi-needle stress test (first run) had its own bug, caught before
trusting it:** needle positions were placed via the same `linspace()`
formula used for landmark positions, so at `K=R` a landmark's position
literally coincided with a needle's position -- trivial perfect
self-retrieval that looked like a clean "R meets K" threshold but was a
test-construction artifact, not a real result (results jumped to
1.0000 exactly at `K=R` and only there, including going 1.0 -> bad ->
1.0 non-monotonically as R grew past K, the tell that something was
wrong). Fixed by placing needles at random (not evenly-spaced)
positions, independent of landmark placement.

**Corrected test revealed something more fundamental than the capacity
question it was built to answer.** Even a SINGLE isolated needle (zero
competition, `K=1`) was recalled poorly at small `R` -- 0.10 to 0.39
cosine across `R=2` to `R=32` out of a 128-token block. Ruled out an
implementation bug directly: swept `R` up toward the full block size
(`R=64,100,128`) with one fixed needle and confirmed clean convergence
to 1.0000 exactly where it should (every position eventually becomes
its own landmark) -- so the mechanism is correctly implemented, the
*small-R* regime is where it fails, and it fails long before any
multi-needle competition even enters the picture.

**Root cause:** landmarks are built by reusing the block's own random
content as pseudo-queries. A landmark only picks up a needle if the
landmark's own essentially-random direction happens to have a decent
dot product with the needle's direction -- for high-dimensional random
vectors that's rare by chance, so unless `R` is a large FRACTION of the
block size (empirically needed somewhere between 32 and 64 out of 128,
i.e. roughly a quarter to half the block, to reliably catch one needle
here), compression misses salient content regardless of how many other
needles are or aren't competing for space. This is a detection failure,
not a capacity failure -- categorically different from what the test
was designed to measure, and it defeats the entire point of decoupling
`R` from `B_l`: if `R` must scale with block size to work at all, the
sub-quadratic complexity win evaporates.

**Explicit scope boundary on this result:** this is a finding about the
specific untrained, content-reused-landmark construction used here, not
a general verdict on "R fixed representatives per block." A real
trained model would have LEARNED landmark/pseudo-query parameters,
explicitly optimized to notice salient content -- a fundamentally
different mechanism than reusing whatever random real keys happen to
sit at fixed geometric positions. This probe cannot speak to that
version at all; it only rules out the specific choice made here.

**Status: MetaMonarchAttention paused, not concluded.** The
geometric/causal scaffolding (binary decomposition, one-time
precomputation, joint softmax combination) is sound and reusable --
verified exact, no leakage, no gaps or overlaps. The specific
landmark-compression mechanism built to populate it is not, for the
reason found above. If this gets picked back up, the open question is
squarely about landmark construction (learned parameters being the
obvious next thing to try, if this were ever wired into an actual
trainable model rather than probed standalone) -- not about the
selection/combination scaffolding around it, which held up under
hand-tracing, causal-validity testing, and the R-sweep sanity check.
Files: `ma_meta_monarch.py`, `eval_meta_stress.py`.

---

## Structural probe battery: landmark mechanics, block structures,
## parameter regimes -- mapping where the R-representatives idea works

Broader sweep following up the single detection-failure finding above,
to check whether it was specific to the one landmark mechanic tried or
a more general property. Built `landmark_mechanics.py` (five pluggable
representative-construction functions, isolating selection strategy as
the only variable) and generalized `ma_meta_monarch.py` to accept a
`landmark_fn` and a `structure` parameter.

### Axis 1: landmark construction (5 mechanics)

**A correction mid-sweep changed the whole picture.** First pass:
`top_magnitude` (R landmarks = highest-norm keys) and `fps`
(farthest-point sampling) both passed the single-needle sanity check
cleanly at R=2, while `random_reuse`, `kmeans`, `maxpool` all failed.
Before trusting that, ran a same-norm control (needle key norm matched
to background ~2.0, only direction distinguishes it, vs. the usual
6.0x scale-up): **both "winners" collapsed to the same poor performance
as everything else** (mean cos ~0.00-0.18). Root cause: FPS's "farthest
pairwise point" correlates strongly with "largest magnitude" for a
roughly zero-centered background, so both mechanics were reading off
the needle test's OWN magnitude-scaling convention, not detecting
genuine salience.

| mechanic | uncorrected | same-norm corrected | actual verdict |
|---|---|---|---|
| random_reuse | fails | fails | detection failure |
| top_magnitude | passes R=2 | fails | was a magnitude-artifact, not real |
| kmeans | fails | (not retested, already failed) | detection failure |
| fps | passes R=2 | fails | was a magnitude-artifact, not real |
| maxpool | fails | (not retested, already failed) | detection failure + broken value-correspondence (no leak, but row sums badly miscalibrated, up to 15.8 vs expected ~1.0 -- max-pooling isn't a convex combination of values) |

**All five fail once the magnitude shortcut is controlled for.** Per
the task's own gating rule (multi-needle sweep only if sanity check
passes), none proceed further -- there's no capacity ceiling worth
measuring on a mechanism that can't detect an uncontested single
needle. This is a STRONGER result than the original single-mechanic
finding: five structurally different, reasonable untrained selection
heuristics (random reuse, magnitude, clustering, diversity sampling,
max-pooling) all converge on the same failure once the test doesn't
hand them a shortcut -- real evidence the problem is architectural (no
query-awareness during compression) rather than a matter of picking a
cleverer untrained heuristic.

### Axis 2: block/level structure

Tested by direct combinatorics on the candidate-selection logic itself
(not needle recall -- broken landmark quality would mask any structural
signal). For each query, does its candidate set exactly tile its own
causal past with no gaps and no double-counting?

| structure | gaps (any of 199 queries) | overlap (any of 199 queries) |
|---|---|---|
| binary (Fenwick) | 0 | 0 |
| kary (always-immediately-preceding-block per level) | 0 | **192** |

**Correction to the earlier hand-trace:** that trace (in the previous
entry) was for a different, more naive reach-based scheme (fixed
token-distance thresholds) and found both gaps and overlaps. THIS kary
implementation (deterministic: always the single immediately-preceding
block at each level, if causally valid) turns out to have zero gaps --
every causally-valid block is eventually reachable by some query -- but
massive overlap: the same near blocks get redundantly covered by
multiple tiers simultaneously in 192/199 queries. Its actual failure
mode is different from what was predicted: not silent blindness to
content, but wasting candidate budget on duplicate near-coverage
instead of efficiently reaching further into the past -- it forfeits
binary's `O(log N)` disjoint-coverage property without buying any
implementation simplicity in return (same per-level loop either way).
Binary decomposition remains the only structure tested that is both
gap-free and non-redundant.

### Axis 3: parameter sensitivity

R-sweep already covered by Axis 1's sanity checks (no mechanic crosses
~1.0 in any tested range, R=2 through 64). Block-size sweep (`B=4,8,16`,
holding the ABSOLUTE stressed-block size fixed at 128 tokens) gave
IDENTICAL results across all three `B` values -- confirms what governs
difficulty is the absolute size of the specific tier block being
compressed, not the base block parameter `B` itself. Rules out "just
pick a different base B" as a lever for the detection-failure problem.

### New failure mode found (distinct from capacity-limit and detection-failure)

Yes, one: kary's redundancy problem is a third category -- a structural
INEFFICIENCY (wasted candidate budget on duplicate coverage), not a
signal-loss problem like the other two. Everything else in this sweep
reinforced rather than added to what was already known: the
detection-failure diagnosis generalizes across five different untrained
selection heuristics, not just the one originally tried -- a converging,
not isolated, result.

**Status: mapping complete for the axes tested.** No variant picked as
a winner (not the goal of this pass, per the task). The geometric
scaffolding (binary decomposition specifically) remains the strongest
piece to reuse if this direction is picked back up; every tested
landmark-construction heuristic and the kary structural alternative are
now ruled out with specific, distinct reasons on record. Files:
`landmark_mechanics.py`, `eval_landmark_sanity.py`,
`eval_landmark_samenorm.py`.

---

## Query-awareness confirmed as the load-bearing variable

Direct isolation test: all five Axis-1 mechanics share one property --
the landmark is built with zero knowledge of any future query. Tested
whether THAT property (not heuristic choice) is the actual cause, by
replacing the precomputed-landmark read entirely with real, fresh,
block-local exact attention per query (no precompute, no compression --
same window + binary-decomposition tier selection as
`ma_meta_monarch.py`, unchanged and still verified exact/gap-free/
leak-free; only the tier READ step changed). Causal validity held (0
leakage, row sums 1.0).

**Same-norm single-needle test (identical control as Axis 1):**

| approach | mean cos | min cos |
|---|---|---|
| best precomputed landmark (any of 5 mechanics, R=64) | ~0.17-0.18 | -- |
| query-aware block-local exact attention | **0.9306** | **0.8845** |

Clean, decisive recovery -- all 10 trials landed 0.885-0.963, not a
lucky average. Confirms the diagnosis directly: it was never heuristic
choice (five different selection strategies all failed identically in
Axis 1), it was the shared "zero query-awareness during compression"
property. Letting the query see the block's real keys directly recovers
detection.

**Cost, reported plainly, not yet a verdict:**

| N | query-aware | landmark R=8 | landmark R=32 | ratio (qa/R=8) |
|---|---|---|---|---|
| 256 | 35.17ms | 35.49ms | 40.72ms | 0.99x |
| 512 | 86.13ms | 79.81ms | 102.40ms | 1.08x |
| 1024 | 194.41ms | 186.23ms | 200.73ms | 1.04x |
| 2048 | 611.77ms | 381.60ms | 459.81ms | 1.60x |

Gap smaller than the `O(B_l)` vs `O(R)` asymptotic difference implies
at these sizes -- likely Python per-block-loop overhead dominating
wall-clock the same way it did for every other reference implementation
tonight, not yet a clean read on the real asymptotic cost. Real and
growing (1.60x at the largest N tested), not yet the dramatic wall the
theory predicts.

**Confirms the next question is well-posed:** with (1) settled and cost
named rather than hidden, "is there a cheap, fixed-cost, QUERY-INFORMED
sketch that approximates full block-local attention without paying
`O(B_l)`" is now worth pursuing rather than a premature optimization.
Files: `ma_meta_query_aware.py`, `eval_query_aware.py`.

---

## Query-informed bucket routing: a genuinely new positive result, not
## just a confirmation

Follow-up to the query-awareness confirmation: precompute only
STRUCTURE per block (a k-means partition into `n_buckets` buckets, done
once, shared across every query that later touches that block -- same
compute-once-reuse-many shape as everything else tonight), then at
QUERY TIME the query itself picks which bucket to read (via similarity
to bucket centroids) and gets real exact attention over that bucket's
actual member keys -- never a static precomputed value. Distinguishes
this cleanly from Axis 1's failed landmarks: those had zero query input
into what got summarized; this has the query choosing its own scope,
just a SUBSET of the block rather than the whole thing. Causal validity
held (0 leakage, row sums 1.0) at every `n_buckets` tested.

**Same-norm single-needle test, sweep n_buckets (block size 128):**

| n_buckets | avg bucket size | mean cos | min cos |
|---|---|---|---|
| 2 | 64.0 | 0.9584 | 0.9184 |
| 4 | 32.0 | 0.9716 | 0.9466 |
| 8 | 16.0 | 0.8194 | **-0.5796** (outlier) |
| 16 | 8.0 | 0.9813 | 0.9580 |
| 32 | 4.0 | 0.9858 | 0.9593 |
| 64 | 2.0 | 0.9913 | 0.9712 |

**Stronger than expected, and it reframes the whole earlier finding.**
At `n_buckets=64` (average bucket size 2 out of 128 keys -- a tiny
fraction), recall is 0.99, comfortably beating every static landmark
tested at ANY R, including R=64 (half the block, mean cos ~0.17-0.18).
The determining factor was never compression ratio -- a size-2
QUERY-ROUTED bucket beats a size-64 STATIC summary by a wide margin.
Whether the query gets to influence what it reads matters far more than
how much gets kept.

**One real anomaly, flagged rather than averaged over:** `n_buckets=8`
(bucket size 16) has mean 0.82 but min **-0.58** -- at least one seed
hit a genuine mis-routing failure (query's centroid-similarity pointed
to the wrong bucket, missing the needle entirely) while every other
bucket count tested was consistently strong. Not a monotonic trend --
a specific fragility at that size, unexplained, worth further look if
this direction continues rather than smoothed into the mean.

**Cost: not measured here, honestly, not just deferred.** This
reference implementation computes full `(B, B_l)` scores and masks to
the chosen bucket rather than skipping computation for non-bucket keys
-- the same caveat already applied to the ANN top-k reference
implementation earlier tonight (`ma_causal_topk_ann.py`). Correctly
tests recall; wall-clock numbers from this version would not
demonstrate the real savings a proper sparse-gather implementation
could show, so none were run to avoid reporting a misleading number.

**Status: open, promising, not yet cost-validated.** Recall is
resolved and strong across most tested configurations. What's still
needed before this could be called a real answer to the sketch/hash
question: (1) understand the `n_buckets=8` anomaly, (2) a real
sparse-gather (not full-score-then-mask) implementation to get an
honest cost number, (3) the multi-needle competition test this whole
arc was originally chasing before the detection-failure diagnosis
superseded it -- not yet run for bucket routing specifically. Files:
`ma_meta_bucket_route.py`, `eval_bucket_route.py`.

### `n_buckets=8` anomaly resolved: not size-specific, a real baseline mis-routing rate everywhere

Re-ran with 100 seeds per `n_buckets` (vs. the original 10) specifically
to check whether the earlier -0.58 min was a one-off or reproducible,
and whether it was specific to bucket size 16 or present more broadly.

| n_buckets | mean | min | frac cos < 0.5 | frac cos < 0.0 |
|---|---|---|---|---|
| 2 | 0.8952 | -0.5715 | 6.00% | 3.00% |
| 4 | 0.8909 | -0.2739 | **8.00%** | 4.00% |
| 8 | 0.9380 | -0.1327 | 4.00% | 3.00% |
| 16 | 0.9533 | -0.4858 | 2.00% | 2.00% |
| 32 | 0.9614 | -0.1765 | 2.00% | 2.00% |
| 64 | 0.9680 | -0.1336 | 2.00% | 2.00% |

**Resolved: not a `n_buckets=8`-specific resonance.** Every bucket count
fails 2-8% of the time with a large enough sample; `n_buckets=8` is
actually one of the BETTER configurations (4%), not an outlier --
`n_buckets=4` has the worst rate (8%). The original 10-seed run simply
happened to draw a failure for n_buckets=8 and not (by chance) for the
others. Failure rate trends downward as bucket count grows (roughly,
not perfectly monotonic) but never reaches zero in the tested range,
floors around 2%.

**Root cause, confirmed via the four `n_buckets=8` failing seeds
(needle positions 92, 57, 92, 19 -- no positional pattern):** this is a
genuine, structural routing-accuracy limit, not a bug or a size
artifact. The query and the needle key share direction but not identity
(query scaled to 6.0, needle key deliberately held at background norm
~2.0 for the same-norm control) -- so the query's own similarity to the
k-means centroids doesn't always rank the needle's true bucket highest;
occasionally an unrelated background centroid scores marginally higher
by chance. A real, quantified failure mode (2-8% baseline mis-routing),
now separated cleanly from the earlier concern that it might be tied to
a specific bucket size. Files: `eval_bucket_route_manyseeds.py`.

---

## Fable consultation + four follow-up probes

Consulted Claude Fable 5 as an external research advisor, floor
constrained to "Monarch-inspired" (grounded in the block-factorization
family and this session's own findings, not generic attention-paper
suggestions). Produced a 38-item probe list across mathematics of the
factorization, lever interactions, causal/hierarchical edge cases,
stress-testing the reliability principle, literature cross-checks, and
systems/cost honesty -- plus a concrete design suggestion for
MetaMonarchAttention ("Fenwick-of-buckets": make every completed
Fenwick tier node a routing structure -- centroids + bucket-contiguous
real keys -- rather than a static summary, combined with the exact
window via one joint streaming softmax; keep bucket count fixed per
tier so read cost doesn't regrow with block size; the decode-time
argument that Fenwick nodes are immutable once complete, so generation
completes at most O(log N) nodes total over a whole sequence, amortized
O(1) per token, a property query-relative hierarchies could never have).

Not all 38 probes are runnable in this environment -- several need a
trained model (#1, #26, #36, #38), real hardware profiling tools
(#32, #33, #35), or a training loop (#13). Worked through four of the
highest-value, cheapest ones from the feasible subset:

**1. Multi-needle competition for bucket routing (open item 2,
finally run).** Fable's pre-registered hypothesis: different-bucket
needles shouldn't meaningfully compete (unlike Version B's single-
representative capacity limit), so the real hard case should be
routing-stage competition, not raw needle count. **Confirmed on both
counts.** Recall stayed strong (mean 0.85-0.99) across K=1 to K=32
needles at both n_buckets=8 and n_buckets=32 -- no capacity-limit-style
collapse with more needles. But a targeted adversarial test (a
mass-heavy decoy deliberately placed to pull the local centroid away
from the needle) produced an **83.33% failure rate** over 30 trials --
far more severe than the natural 2-8% baseline. This is a real,
distinct, and much sharper cliff than anything the natural-distribution
stress tests had found: bucket routing is robust to incidental
multi-needle crowding but has a serious, exploitable weakness if
something (adversarial or just unlucky data) can shift a local
centroid away from rare content. Files: `eval_bucket_multineedle.py`.

**2. Margin/outlier diagnosis of the natural mis-routing rate
(probe 30).** Nuances Fable's "MIPS/outlier-mismatch" hypothesis rather
than cleanly confirming it: the dominant signal in wrong routes is a
much smaller top1-top2 centroid margin (0.0475 vs 0.5240 for correct
routes, ~11x), while needle-to-own-centroid distance is only modestly
elevated (1.8219 vs 1.5380 correct, vs 1.5015 background baseline,
~18%). So natural failures are predominantly close-call, near-boundary
routing decisions, not needles being severe outliers that centroids
systematically under-rank -- a more precise diagnosis than "outlier
mismatch" alone, and distinct from probe 1's adversarial mechanism
(which deliberately manufactures exactly this kind of close call).
Small sample caveat: only 4 wrong routes out of 100 trials backing the
"wrong" statistics. Files: `eval_bucket_margin_diagnosis.py`.

**3. T under decoy pressure (probe 11).** Clean negative result: T does
NOT rescue decoy/rank-competition pressure, only pure distance. At
W_blocks=4, needle scale=3.0, dist=14 blocks: T=1->8 with 20 decoys
present moves recall only 0.1729->0.1635 (saturates by T=2, then
completely flat) -- a tiny fraction of T's effect on pure distance with
no decoys (~0.10->~0.94 across the same T range, established earlier).
T and decoy-robustness are separate axes needing separate remedies, not
"T helps everything." Files: `eval_t_under_decoy.py`.

**4. Backward-pass gradient check (probe 34, the acknowledged
session-wide gap).** Initial gradcheck FAILED for both
`ma_causal_dual_opt.py` and `ma_sliding_monarch.py`. Diagnosed before
concluding there's a real bug: both hardcode `.to(torch.float)`
(forces float32) in several internal steps regardless of the actual
input dtype -- silently downcasts mid-graph, which breaks float64
numerical gradient checking even with a correct formula. Patched copies
replacing the hardcoded cast with the actual input dtype (`ar.dtype` /
`q.dtype` as appropriate per function scope) **passed gradcheck cleanly
on both.** Gradients are correct; the failure was a dtype artifact, not
a formula bug. Real, minor, actionable footnote for the actual shipped
files: the hardcoded `torch.float` casts are harmless no-ops in normal
float32 training but would silently downcast precision in any float64
or mixed-precision context -- a one-line fix (not applied to the
committed files, since it wasn't asked for and isn't urgent) whenever
that starts to matter. Not run: eval scripts, only the direct
gradcheck comparison.

**Status: 4 of the feasible ~25 probes done, rest not started.** The
adversarial bucket-routing finding (#1) is the standout result --
a real, severe, distinct cliff worth treating as a first-class finding
alongside the reliability principle itself, not a footnote. The
gradient check (#4) closes a real gap that had been open since the very
first causal-masking entry. Remaining probes not yet attempted.

---

## Eight more probes: built as a Colab notebook, smoke-tested locally
## (results already in hand, not just a handoff)

Built `probe_battery_colab.ipynb` (self-contained port of
`sliding_monarch_causal` and `monarch_meta_bucket_route`, GPU-optional)
covering 8 more of Fable's probes: Sinkhorn convergence
characterization (#3), T x B surface (#9), T x window width (#10),
ragged N (#15), query-position boundary sawtooth (#18), adversarial
low-norm/high-alignment needle (#20), bucket-routing capacity ceiling
at larger scale (#23), and hot-bucket load imbalance under anisotropic
keys (#35). Smoke-tested the extracted code locally on CPU before
handoff -- it ran clean end to end, which means these are real results,
not just a JSON-validity check.

**#3 Sinkhorn convergence -- propagation, not pure convergence.**
Per-iteration delta norms shrink monotonically but consecutive ratios
drift (0.111 -> 0.527) rather than staying constant -- not clean
geometric convergence. Block-hop test: at T=1, only hop-distances 1-2
succeed (1.0, 1.0) while 4/8/16 lag (0.40/0.42/0.23); increasing T
progressively "unlocks" farther hops IN ORDER (T=3-4 rescues hop=4,
T=6-8 rescues hop=8), but hop=16 plateaus around 0.56 even at T=12,
never fully rescued in the tested range. Real evidence for multi-hop
information propagation through the block structure, not just
convergence to a fixed point.

**#9 T x B surface -- run has a real methodology caveat, not clean
evidence either way.** Query distance was scaled with B
(`14 * B`), confounding block size with absolute token distance across
the sweep. B=8 improved dramatically with T (0.32->0.99), B=32 stayed
capped (0.20->0.36) -- plausible that finer B benefits more from T, but
this specific run can't distinguish that from "smaller B was tested at
a shorter absolute distance." Needs a rerun holding absolute distance
fixed before trusting the direction.

**#10 T x window width -- real positive interaction, not independent
axes.** Wider windows benefited MORE from T, not less/independently
(W=8: 0.52->0.97; W=1: 0.57->0.67) -- contradicts a clean "window
handles near-field, T handles far-field" separation.

**#15 Ragged N -- clean.** Zero leakage, row sums ~1.0 across six N
values not divisible by B. No padding bug.

**#18 Boundary sawtooth -- inconclusive.** Recall varies noisily
(0.64-0.93) across intra-block query offsets with no obvious pattern at
this sample size (1 trial/offset) -- needs more seeds before concluding
anything about a real positional effect.

**#20 Low-norm adversarial needle -- confirms a general dot-product
property, not Monarch-specific.** Recall degrades steadily as key norm
drops (0.20->0.14) even with the query staying strongly aligned --
expected for any dot-product attention mechanism (magnitude matters,
not just alignment), now directly measured in this context.

**#23 Bucket-routing capacity at larger scale -- a genuine new wrinkle,
not a clean extension of the earlier positive finding.** At
BLOCK_SPAN=256 (vs. 128 locally), `n_buckets=64` performed WORSE and
more erratically than `n_buckets=16` in several cells (K=4: mean
0.71/min -0.09 at nb=64 vs. mean 0.97/min 0.96 at nb=16) --
contradicting the earlier "more buckets is better" trend. Likely
k-means instability with very few points per cluster (~4 average) at
this specific scale combination, not a fundamental reversal of the
earlier finding -- but real, and not smoothed over.

**#35 Hot-bucket load imbalance -- confirms the concern.** Imbalance
ratio (max/mean bucket occupancy) grows from 1.62 (near-isotropic keys)
to 7.38 (maximally anisotropic) as key distribution rank-bias
decreases -- real trained-model-like anisotropic key distributions
would plausibly produce meaningfully hotter buckets than tonight's
uniform Gaussian probes suggested, relevant to real 6-core load
balancing on the target hardware.

**Status: 12 of ~25 feasible probes now done (4 from the direct batch +
8 from this notebook).** Notebook still available for GPU reruns at
larger scale/more trials if the noisier results (#9's confound, #18's
inconclusiveness, #23's new wrinkle) warrant a cleaner follow-up. Files:
`probe_battery_colab.ipynb`.

---

## Fable reconsultation: the adversarial routing failure reprioritizes everything

Sent the 12-probe results back to the same Fable session (resumed from
transcript, full prior context intact) and asked for a refined
MetaMonarchAttention recommendation given the new data. Substantially
reprioritizes rather than restating the earlier plan.

**Headline reframing: the adversarial mass-heavy-decoy routing failure
(83%) is now the most urgent open item, ahead of layout/cost.** Key
distinction drawn from the margin data: natural mis-routing (~4%, small
n) and adversarial mis-routing (83%) are DIFFERENT mechanisms, not two
severities of the same thing. Natural failures are near-boundary ties
(margin ~11x smaller than correct routes, needle-to-centroid distance
only mildly elevated). The adversarial case is genuine CENTROID
CAPTURE -- a large-mass decoy dragging a mean-based centroid away from
the true content. Fable's read: this reproduces the exact character of
the top-k/decoy-competition cliff from earlier in the session (correct,
in-scope answer structurally outvoted) -- precisely the failure class
the reliability principle exists to rule out, now found inside the
mechanism recommended as the way forward. Treated as the single most
important result in the batch, not a footnote.

**Revised priority order:**
1. Floor-read rescue test against the adversarial case specifically --
   promoted from "opportunistic, listed last" to load-bearing: does an
   always-on mean-summary read (in addition to the routed bucket) turn
   the 83% FAILURE into 83% DILUTION (needle still contributes some
   weight) or does it stay a true miss? Determines whether the design
   can honestly keep the reliability-principle claim under adversarial
   content, not just adversarial distance/count.
2. Robust centroid construction (geometric median / trimmed-mean /
   magnitude-capped contribution) -- attacks the capture mechanism
   directly instead of only bounding damage.
3. Re-run the margin diagnosis at larger n (only 4 wrong routes backed
   the natural-failure characterization).
4. NEW: a minimum-viable-bucket-occupancy probe, motivated by probe
   23's scale wrinkle -- "keep k fixed per tier" isn't just a cost
   decision anymore, there's an apparent k-means stability floor
   (points-per-bucket) to respect too.
5. Layout/cost implementation (former top item) -- now sequenced AFTER
   stability is characterized, since building around an unstable
   (B_l, k) schedule would be wasted work.
6. Trained-landmark hedge -- unchanged, now clearly lower urgency.

**New design idea surfaced, not present in the original consultation:**
probe 3's un-cleared propagation ceiling (hop=16 plateaus at 0.56 even
at T=12) applies specifically to SlidingMonarch's chained multi-hop
Sinkhorn-style refinement. Fenwick-of-buckets reads real keys directly
at each dyadic node in one shot, no chained hops -- Fable's read is
that it may not inherit this ceiling at all, and flags a cheap,
high-value direct comparison: does bucket routing avoid the plateau
chained-T hits at the same hop distance?

**Corrections to the original 38-probe list, not just new findings:**
- Probe 30's outlier-mismatch hypothesis is weakened for the NATURAL
  rate but validated for the ADVERSARIAL rate -- needs splitting into
  two separate findings, not one, going forward.
- Probe 10 directly contradicts the implicit "window=near-field,
  T=far-field, independent axes" assumption baked into earlier
  thinking -- any future default-picking needs a joint grid, not
  independent per-axis tuning.
- Probe 9 invalidated by its own confound (Fable's own probe spec
  should have fixed absolute distance) -- own error acknowledged, not
  blamed on the run.
- Probe 23 corrects the core design recommendation, not just adds a
  caveat -- "keep k fixed per tier" now needs a stability floor
  attached.
- Probe 35 escalates from "flagged concern" to "confirmed, and
  compounds with probe 23" -- anisotropic keys make bucket instability
  AND load imbalance worse simultaneously, raising the priority of
  getting real trained-model K/V statistics before finalizing defaults
  (the Gaussian-probe defaults used all session may be systematically
  too optimistic on this specific axis).
- Probe 34's dtype-cast bug pattern flagged as worth sweeping the rest
  of the reference codebase for, before trusting any future float64
  measurement.

**New probes suggested for later:** a decoy-SEVERITY sweep (not just
count) to find the actual capture threshold, mirroring how the
weak-signal cliff was characterized for top-k; window-width x
routing-quality interaction, parallel to the T x window coupling;
extending the hop-ceiling test to T=16/24 to determine if hop=16's
plateau is a hard structural ceiling or just needs more iterations;
balanced/capacity-constrained clustering (cap max bucket occupancy,
spill overflow to a neighbor or the floor-read) as a direct fix for the
load-imbalance finding, tested jointly with mis-routing rate since they
may trade off.

**Status: recommendation refined, floor-read rescue test (new #1
priority) up next.**

---

## Floor-read rescue test: clean failure, and it's Version B's failure
## mode all over again

Implemented `ma_meta_bucket_route_floor.py`: an always-on mean-summary
candidate per tier block, precomputed once, competing as one more
uncollapsed candidate in the same joint softmax alongside the routed
bucket and the window -- not a fallback triggered on low confidence, a
permanent extra candidate, exactly as Fable specified. Causal validity
held (0 leakage, row sums 1.0).

**Result: zero effect, to four decimal places.** Same adversarial
mass-heavy-decoy harness that gave 83.33% failure:

| use_floor | mean cos | min cos | fail rate | true-miss rate |
|---|---|---|---|---|
| False | 0.3006 | -0.2248 | 83.33% | 23.33% |
| True | 0.3007 | -0.2248 | 83.33% | 23.33% |

**Why, and it traces directly back to the very first finding of the
whole MetaMonarchAttention arc:** the floor candidate's value is a flat
mean over the WHOLE block (~127 mostly-irrelevant keys plus the
decoy). Even setting aside whether the decoy also distorts the floor's
own logit, the floor's VALUE contribution is structurally incapable of
carrying the needle's specific signal, regardless of how much softmax
weight it wins -- this is Version B's exact mean-pooling failure,
reintroduced as a safety net. Insurance built from a mechanism already
known not to carry signal doesn't insure anything. Files:
`ma_meta_bucket_route_floor.py`, `eval_bucket_floor_rescue.py`.

## Robust (geometric-median) centroids: modest improvement, then a
## diagnosis that overturns Fable's "centroid capture" framing entirely

Implemented `ma_meta_bucket_route_robust.py`: geometric-median
centroids via Weiszfeld iteration (iteratively reweighted mean, weight
= 1/distance) run after standard k-means assignment settles --
principled choice given geometric median's up-to-50%-breakdown-point
robustness property vs. the arithmetic mean's 0%. Causal validity held.

**Same adversarial harness: only a small improvement, not a rescue.**

| method | mean cos | min cos | fail rate | true-miss rate |
|---|---|---|---|---|
| arithmetic-mean | 0.3006 | -0.2248 | 83.33% | 23.33% |
| geometric-median | 0.3533 | -0.1615 | 80.00% | 20.00% |

**Diagnosed before accepting "robust centroids barely help" at face
value -- and the diagnosis overturns the whole "centroid capture"
framing, not just the mitigation.** Directly measured routing accuracy
(does the query's centroid-similarity correctly identify the needle's
actual bucket) separately from final recall:

| method | routing correct | needle+decoy same bucket |
|---|---|---|
| arithmetic-mean | 29/30 (96.7%) | 18/30 (60.0%) |
| geometric-median | 29/30 (96.7%) | 18/30 (60.0%) |

**Routing was never broken.** Both centroid methods route to the
needle's actual bucket 96.7% of the time, identically -- geometric
median changed nothing about routing because routing wasn't the
problem. The real mechanism: when the needle and decoy land in the SAME
bucket (60% of trials, by construction of the adversarial test), the
decoy -- built with 3x magnitude and a correlated direction -- LEGITIMATELY
outscores the needle in the real exact-attention step within that
correctly-selected bucket. This is not centroid capture. It is the SAME
decoy/rank-competition cliff exact top-k hit earlier tonight, just
relocated inside one bucket's real attention instead of the whole
block's. Bucket routing narrows the contested candidate pool (which is
exactly why the NATURAL baseline failure rate is only 2-8%, not 83% --
fewer chances for a decoy to land in the same small bucket as a given
needle by accident) but never eliminated the underlying vulnerability:
once two items share a bucket, the final exact-attention step is still
genuine rank-based competition among real keys -- the same contested-
scope mechanism the original reliability principle was built to
distinguish from fixed, content-independent scope. Robust centroid
construction cannot fix this because it only affects WHICH bucket gets
chosen, never what happens once you're reading it. Files:
`ma_meta_bucket_route_robust.py`, `eval_bucket_robust_adversarial.py`,
`eval_bucket_robust_diagnosis.py`.

**Status: both of Fable's top-2 mitigations (floor-read, robust
centroids) tested and ruled out, for two DIFFERENT reasons -- floor-read
because it recreates Version B's flat-mean failure, robust centroids
because they fix a stage (routing) that was never actually broken. The
real vulnerability has been re-identified as bucket-INTERNAL
rank-competition, structurally the same class as exact top-k's decoy
cliff, not a routing-stage problem at all. This is a correction to send
back to Fable, not just more data -- it changes what "the mitigation"
should even target.**

---

## Fable's third refinement, and the decisive dense-attention control it called for

Sent the routing-diagnosis correction back to Fable. Response was
notably self-critical: named the floor-read design-consistency lapse
directly (proposed reusing mean-pooling -- the exact mechanism Version
B's very first failure proved incapable of carrying a specific signal
-- as a safety net for a signal-carrying failure; "a fallback should
never be built from a mechanism already falsified for the property
it's meant to insure").

**Identified the decisive missing experiment, calling it what should
have been probe 1:** run the identical needle+decoy construction
through PLAIN DENSE CAUSAL SOFTMAX ATTENTION -- no Monarch, no
bucketing, the actual ground truth. Reasoning: "3x magnitude and
correlated direction legitimately outscores the needle" describes a
property of dot-product softmax scoring itself, not of bucket routing,
and is the same axis as probe 20 (recall degrading with LOW needle-key
norm) attacked from the decoy's side instead. This single test would
reclassify the whole finding: if dense attention also collapses, bucket
routing is no worse than the ground truth it approximates; if it
doesn't, small-candidate-pool dynamics specifically amplify domination
-- a genuinely bucket-routing-specific problem.

**Ran it. Decisive, and it flips the interpretation in bucket routing's
favor:**

| method | mean cos | fail rate (cos<0.5) | true-miss rate (cos<0.0) |
|---|---|---|---|
| dense causal softmax (ground truth) | 0.2181 | **90.00%** | 23.33% |
| arithmetic-mean bucket routing | 0.3006 | 83.33% | 23.33% |
| geometric-median bucket routing | 0.3533 | 80.00% | 20.00% |

**Dense attention collapses too -- worse than either bucket-routing
variant.** Confirms Fable's hypothesis decisively: the cliff is
inherited from softmax scoring itself, not introduced or amplified by
bucketing. This specific needle+decoy construction defeats real, exact,
full-context softmax attention just as effectively as it defeats bucket
routing -- bucket routing's "narrows exposure" property gave it a real,
measurable edge here (83.33%/80.00% fail rate vs. dense's 90.00%),
since it sometimes avoids putting the decoy in the same candidate pool
as the needle at all, while dense attention always sees everything.

**This was never a MetaMonarchAttention-specific vulnerability.** It's
a fundamental property of dot-product softmax attention that no
scope-selection mechanism -- geometric, learned, or otherwise -- can
escape, and bucket routing is demonstrably no worse than, and modestly
better than, the ground truth it approximates. Substantially changes
the verdict on the whole adversarial-routing thread: from "here's a new
cliff bucket routing introduced" to "here's a known limitation of
attention itself, which bucket routing partially mitigates rather than
inherits or worsens." Files: `eval_dense_attention_control.py`.

**Status: sending this back to Fable to close the loop on the
reliability-principle reframing (structural inclusion vs. outcome
guarantee) and the "still the right headline design, weaker but more
honest claim" verdict from the prior refinement.**

---

## Fable's close-out: a real positive claim, not just a null result, plus a targeted follow-up

**True-miss rate (cos<0) is identical between dense and arithmetic-mean
bucket routing: 23.33% both.** Validates the reframe exactly: once
genuine contested co-location occurs, bucket routing offers zero extra
protection over dense attention -- same rank-competition dynamics, same
worst case.

**But the soft-fail-rate gap (dense 90% vs. bucket routing 83.33-80%)
is a real, mechanistically-grounded positive claim, not passive
"narrows exposure."** Fable's framing: restricting candidate scope has
a nonzero chance of removing a competitor from the contest ENTIRELY --
something no full-context mechanism can offer by construction, since
dense attention structurally cannot avoid exposing the needle to the
decoy (every query sees every key, always), while bucket routing
sometimes does avoid it as a byproduct of routing. Generalizes beyond
bucket routing specifically -- same underlying logic as a sliding
window helping against a decoy outside the window, arguably the same
logic that motivated the original reliability principle. Scoped
honestly, not overclaimed: "a genuine, partial, probabilistic
mitigation against moderate dilution, with no advantage once true
contested co-location occurs" -- not "solves the decoy problem."

**Adversarial-cliff thread closed. MetaMonarchAttention priority list
returns to the original sequence, reordered by one new fact:** bucket
size is now a THREE-way tradeoff (cost, k-means stability from probe
23, and now adversarial-exposure reduction), not two -- makes the
occupancy sweep more valuable, since one probe now resolves three
competing considerations instead of one:
1. Margin diagnosis at larger n (still open, unaffected by any of this).
2. Minimum-viable-bucket-occupancy probe, MERGED with a fail-rate-vs-
   bucket-size sweep on the adversarial construction (was going to be
   two separate asks, now one sweep resolves both).
3. Layout/cost implementation, once a real (B_l, k) operating point exists.
4. Trained-landmark hedge (unchanged, still last).

**On repeating the dense-attention-control pattern elsewhere:** Fable's
answer is targeted, not blanket. Worth running exactly once more --
against exact top-k's decoy cliff specifically -- because top-k's
failure mode (hard exclusion once decoy count exceeds k, deterministic
cutoff) is mechanistically different from bucket routing's (soft
dilution within a contested pool), so the outcome isn't predictable
from this result. If top-k also partially benefits from exposure-
reduction logic, that would soften the session's harshest verdict on
it; if its true-miss rate spikes instead of staying flat, that sharpens
the hard-exclusion-vs-soft-dilution distinction with real data instead
of a qualitative argument. Explicitly NOT worth re-running for probe 20
(low-norm cliff) -- that failure is about the needle's own signal
strength, not competitive exposure, so scope-narrowing has no clear
mechanism to help or hurt there.

---

## Dense-attention control vs. exact top-k's decoy cliff: the OPPOSITE
## result from bucket routing, sharpening the distinction with real data

Ran the targeted follow-up on the exact original top-k decoy-pressure
construction (needle scale=3.0, query at distance 14 blocks, same seeds
as the original test tonight).

| num_decoys | dense (ground truth) | topk8 | topk16 |
|---|---|---|---|
| 0 | 0.8229 | 0.9539 | 0.9421 |
| 5 | 0.6517 | 0.7085 | 0.7068 |
| 20 | 0.4479 | **-0.0073** | 0.3134 |
| 50 | 0.2634 | **-0.1063** | **-0.0996** |

**Opposite pattern from the bucket-routing case.** Dense attention
degrades gracefully and stays positive throughout the whole decoy
sweep (0.82 -> 0.65 -> 0.45 -> 0.26). Exact top-k crashes to negative at
both k values once decoy count exceeds k. This time the approximation
is genuinely WORSE than the ground truth -- unlike bucket routing,
where dense attention collapsed just as much or more than the
approximation did.

**Confirms Fable's predicted distinction with real, evidence-backed
data rather than a qualitative argument:** bucket routing's soft
dilution is no worse than (and sometimes better than) dense attention
under adversarial pressure -- an inherited softmax-attention property,
partially mitigated by scope-narrowing. Exact top-k's hard-exclusion
cutoff is a genuine, approximation-SPECIFIC vulnerability that dense
attention does not share -- not inherited, actually introduced by the
top-k mechanism itself. Two cliffs that both "involve a decoy" turn out
to be fundamentally different in kind, now with a controlled comparison
establishing which is which rather than resting on the earlier
qualitative "rank-based methods can lose the correct answer entirely"
framing alone. This also retroactively strengthens the original
derived reliability principle from earlier tonight (fixed-scope methods
degrade gracefully, rank-based/contested-scope methods can fail
outright) -- it's not just a pattern observed once for top-k, it's now
shown to be the actual DIFFERENCE between top-k and everything else
that's been tested against dense attention as a control. Files:
`eval_dense_vs_topk_decoy.py`.

**Status: dense-attention-control thread closed, per Fable's own scope
recommendation (worth running for bucket routing and top-k specifically,
not blanket-repeated elsewhere). MetaMonarchAttention priority list
returns to: margin diagnosis at larger n, merged occupancy/exposure
sweep, layout/cost implementation, trained-landmark hedge.**
