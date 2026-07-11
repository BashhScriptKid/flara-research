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

---

## Margin diagnosis at larger n (item 1 of the returned priority queue)

Re-ran the natural (non-adversarial) mis-routing margin analysis at
n=1000 instead of the original n=100 (which had only 4 wrong routes --
too few to trust). Got 36 wrong routes this time, a real sample.

| | correct routes (n=964) | wrong routes (n=36) |
|---|---|---|
| top1-top2 centroid margin | mean 0.527, median 0.504 | mean 0.089, median 0.068 |
| needle-to-own-centroid distance | mean 1.534, median 1.550 | mean 1.814, median 1.800 |

**Routing accuracy: 96.40% (3.60% mis-routing)** -- lands in the middle
of the earlier 2-8% estimate range, now pinned more precisely.

**Margin ratio 5.96x (down from the original 11x estimate at n=100, but
still clearly the dominant signal).** Confirms margin-tightness as the
primary driver of natural failures, as the smaller sample suggested,
just with a less inflated ratio now that n is real.

**Distance ratio 1.18x -- identical to the n=100 estimate, but the
larger sample reveals something the mean-only comparison couldn't
show: the distributions barely overlap.** Wrong routes' distances
cluster tightly in a right-shifted range (1.60-2.12, std 0.125), while
correct routes span a much wider range (0.92-2.07, std 0.188) that dips
well below where any wrong route lands. So the outlier-distance effect
isn't just "mean 18% higher" noise that would wash out with more data
-- it's a real, distributionally-separated secondary contributor.
Refined verdict: BOTH mechanisms (margin-tightness, outlier-distance)
are genuine and robust at scale; margin-tightness dominates in
magnitude, distance-elevation is a real secondary signal, not noise.
Files: `eval_bucket_margin_diagnosis_large.py`.

**Status: item 1 done. Moving to item 2 (merged minimum-viable-bucket-
occupancy + adversarial-exposure-vs-bucket-size sweep).**

---

## Item 2: merged occupancy/exposure sweep -- contradicts Fable's
## exposure-reduction prediction, non-monotonic with a real mechanism

Swept n_buckets = 4/8/16/32/64/128 at BLOCK_SPAN=256 (the scale where
probe 23 found nb=64 unstable vs nb=16), tracking bucket-occupancy
stats, natural single-needle recall, and the same adversarial mass-
heavy-decoy fail rate together in one sweep.

| n_buckets | avg occupancy | natural recall | adversarial fail rate |
|---|---|---|---|
| 4 | 64.0 | 0.8612 | 75.00% |
| 8 | 32.0 | 0.8088 | 45.00% (best) |
| 16 | 16.0 | 0.9005 | 80.00% |
| 32 | 8.0 | 0.8056 | 75.00% |
| 64 | 4.0 | 0.8822 | 80.00% |
| 128 | 2.0 | 0.9881 | 90.00% (worst -- worse than nb=4)|

**Adversarial fail rate is NON-MONOTONIC in bucket size, with a minimum
around n_buckets=8 and the WORST result at the finest granularity
tested (n_buckets=128).** This contradicts Fable's exposure-reduction
prediction that smaller buckets should monotonically reduce adversarial
exposure -- the opposite happens at the fine end.

**Plausible mechanism, two competing effects:** reduced co-location
PROBABILITY favors smaller buckets (fewer other members means less
chance the decoy happens to land with the needle), but reduced within-
bucket DILUTION once co-located favors larger buckets (at nb=128,
avg occupancy ~2, once needle+decoy do share a bucket there's almost no
other background content to dilute the decoy's built-in 3x magnitude
advantage -- closer to a raw 1-on-1 competition the decoy wins even
more cleanly than in a larger, more diluted bucket). These trade off
with a real sweet spot, not a monotonic relationship.

**Probe 23's stability wrinkle (nb=64 notably worse than nb=16) did not
clearly reproduce here** -- natural recall stayed in a similar 0.81-0.99
range across all bucket sizes tested, though this run used fewer trials
(10) than probe 23's original test. Inconclusive rather than
contradicted; the occupancy-instability question and the adversarial-
exposure question turned out to be less coupled than expected -- worth
tracking as two separate open questions rather than one.

**Status: real correction to send back to Fable -- challenges the core
mechanism behind the "narrow bucket size for exposure reduction"
recommendation from the prior refinement.** Files:
`eval_bucket_occupancy_exposure_sweep.py`.

---

## Fable's response, and the larger split sweeps it recommended

Fable's refinement: the co-location-probability half of the two-effects
hypothesis holds, but the dilution half needed sharpening into a
softmax entropy/normalization effect (low-scoring background
contributes negligibly to the softmax denominator regardless of count,
so "more background helps" isn't quite the right frame) -- proposed
testing with orthogonalized background keys to isolate this. Flagged a
real confound: n_buckets=128 (avg occupancy ~2) is exactly probe 23's
original instability regime, so results there could be routing noise,
not pure post-routing competition. Statistical caution: n=10 fail-rate
estimates (45/75/80/90%) have binomial confidence intervals too wide to
trust the "sweet spot" shape -- flagged as the same small-sample
curve-shape risk that already bit probe 23 once. Most importantly:
even at its best point, fail rate was 45% -- catastrophic regardless of
exact curve shape, reinforcing "treat as bounded accepted risk, choose
bucket size on cost/stability grounds, report adversarial number as a
footnote" rather than tuning architecture to chase this curve.
Recommended splitting into two separate, larger sweeps rather than one
merged low-n one.

**Stability-only sweep, n=50 (`eval_bucket_stability_large.py`):**
natural recall stays FLAT across all bucket sizes (mean 0.87-0.93, no
degradation trend) -- CONTRADICTS probe 23's original "nb=64 unstable"
finding at real sample size, confirming Fable's suspicion the original
read was itself noise. But a more precise, different pattern emerged:
worst-case outcomes get notably worse at smaller bucket sizes even
though the MEAN doesn't move (min -0.52 at nb=8, -0.44 at nb=128) --
not "smaller buckets have worse average quality," but "smaller buckets
have more occasional bad outliers." A real finding, just not the one
probe 23 originally suggested.

**Adversarial exposure sweep, n=25, two background variants
(`eval_bucket_exposure_large.py`):**

| n_buckets | normal background fail rate | orthogonalized background fail rate |
|---|---|---|
| 4 | 80.00% | 96.00% |
| 8 | 44.00% | 96.00% |
| 16 | 76.00% | 96.00% |
| 32 | 76.00% | 88.00% |
| 64 | 72.00% | 96.00% |
| 128 | 84.00% | 68.00% |

Normal-background curve reproduces the earlier sweet spot reasonably
well (44% at nb=8, close to the earlier 45%). Orthogonalized-background
does NOT show the predicted monotonic decrease -- it's mostly WORSE
(96% at most bucket sizes) than normal background, with only one drop
(68% at nb=128).

**Diagnosed as a likely confound in the test construction, not a clean
refutation of Fable's hypothesis:** orthogonalizing background keys
against the query direction makes their dot-product score exactly
zero -- but `exp(0)=1` is not negligible softmax weight. Ordinary
random background keys often have NEGATIVE dot products with the
query, and `exp(negative)` genuinely is near-zero -- so normal
background may actually contribute LESS aggregate competing mass than
a bucket full of exactly-zero-scored orthogonalized keys, the opposite
of what the manipulation intended to isolate. "Score near zero" and
"softmax weight near zero" are not the same thing once sign isn't also
controlled. A cleaner isolation would need background keys with
STRONGLY NEGATIVE dot product against the query, not merely orthogonal
ones.

**Status: this specific sub-thread (precise exposure-curve mechanism)
closed as inconclusive rather than resolved, per explicit user
direction ("okay, so be it") after the confound was identified --
not pursued further with a corrected re-test.** What's actually settled
and usable: (1) probe 23's original stability-mean claim doesn't
reproduce at n=50 (real finding), (2) worst-case variance does increase
at small occupancy (real, different finding), (3) Fable's standing
recommendation from before this sub-thread started still holds
regardless of the exact curve mechanism -- treat the adversarial cliff
as a bounded, accepted risk, pick bucket size on cost/stability/natural-
recall grounds, report the adversarial number as a footnote rather than
a design driver. Returning to the main priority queue: item 3 (layout/
cost implementation) is next.

## Fable's close-out on the split sweeps

Sent the sweep results (including the honest confound writeup) back to
Fable rather than skipping that step -- caught mid-thread when asked
directly whether I had.

**Adversarial thread: no change, reinforced, not reopened.** Every
bucket size in the n=25 run is catastrophic (44-84%), so the exact
non-monotonicity mechanism doesn't matter for any design decision. The
nb=8 dip replicated at both n=10 and n=25 (45% -> 44%) -- probably real
and mild, usable as a last-resort tiebreaker if cost/stability leave
options tied, but should never outrank them.

**Stability thread: genuinely updated, in a useful and actionable
direction.** Confirms probe 23's mean-based read was noise, but
reframes the real finding as a methodology fix: bucket size should be
selected on TAIL-RISK/WORST-CASE recall, not mean recall -- mean is
flat and uninformative across the whole range (0.87-0.93), but the
min/std pattern found in the n=50 stability sweep is a real,
discriminating signal a mean-only criterion would have missed entirely.

**Confound-test rerun: deprioritized, not blocked on.** The "even at
best it's catastrophic" argument makes the exact entropy mechanism
moot for design purposes. Parked as an optional footnote-level
experiment if there's appetite for the mechanistic story itself later,
not a gate before further work.

**Durable methodology lesson, independent of this specific test:**
"near-zero score" and "near-zero softmax weight" are NOT
interchangeable -- exp(0)=1 is not negligible, only genuinely negative
logits contribute near-zero weight via exp(). Any future synthetic
probe claiming to isolate "negligible background contribution" needs
to control the SIGN of the dot product, not just its magnitude. Worth
carrying forward to any future probe construction in this arc.

**Status: both sub-threads closed. Proceeding to item 3 (layout/cost
implementation) with a refined three-part bucket-size selection
criterion going forward -- cost, tail-risk stability (not mean), and
the adversarial rate as a reported footnote -- replacing the
mean-recall-only criterion implicitly assumed earlier in this arc.**

---

## External research artifact: Louver (threshold-based selection) and
## Multipole Attention, synthesized against the whole session

A user-provided research document surfaced several external findings
directly relevant to this arc, most notably **Louver / "Sparse
Attention as a Range Searching Problem"** (arXiv 2605.06763, preprint,
single-workstation benchmarks, not clearly peer-reviewed): reframes
sparse attention as halfspace range searching -- given threshold tau,
return exactly {k : <q,k> >= tau}, a fixed geometric predicate with a
proven zero-false-negative guarantee (relative to tau), via bounding
balls over fixed-size key groups pruned only when provably below tau.
Reports real CPU C++/AVX/FP16 kernels, "faster than highly optimized
dense attentions such as FlashAttention" on an AMD Threadripper.
Also **Multipole Attention** (Hooper et al., NeurIPS 2025): exact
attention for near/important keys + CENTROID-approximated (never
dropped) attention for far keys, progressively coarser with distance --
close to this session's Fenwick design but never routes-and-discards.
Also independent literature corroboration of THREE of this session's
own negative results: MQAR/Jelassi formally prove the mean-pooling
capacity ceiling; Landmark Attention's own literature found untrained
landmarks underperform, matching all-five-heuristics-failed; MoBA
derives a closed-form SNR (~sqrt(d/B)) for exactly the router this
session built via k-means bucket routing.

**Ran a fresh (new-transcript) Fable consultation to synthesize this
against everything found tonight**, specifically asking for the same
skepticism that caught the floor-read and robust-centroid failures
earlier rather than accepting the artifact's framing at face value.

**Verdict: Louver genuinely fixes one failure family, provably
relocates a second, and cannot touch a third.**

- **Genuinely fixes:** the EXCLUSION failure -- top-k's decoy cliff and
  bucket routing's 3.60% natural mis-routing floor. Under a threshold
  predicate, the needle's inclusion depends only on its own score
  against a fixed bar; a decoy can't evict it, since there's no budget
  being competed for. Matches the STRUCTURAL half of the refined
  reliability principle exactly.
- **Relocates, not resolves:** tau-estimation. Budget-driven tau is
  top-k in disguise (vacuous guarantee); absolute tau makes the
  guarantee real but makes survivor count adversary-controlled --
  degrading toward dense O(N^2) under pressure. Fable's framing:
  "Louver converts a correctness cliff into a cost cliff, plus a
  residual soft correctness dependence on tau-estimation. That is
  progress, not a solution."
- **Cannot touch, using data already in hand:** the adversarial
  construction. Dense attention is effectively tau=-infinity (zero
  false negatives, everything computed) and it STILL failed at 90%,
  worse than bucket routing (already established in this session). The
  decoy legitimately exceeds any sane tau, enters the survivor set, and
  outscores the needle inside the exact softmax -- a SCORING-stage
  failure no SELECTION-stage guarantee can reach. Sharpened principle:
  "Louver addresses the structural half; nothing selection-side
  addresses the outcome half."

**Produced a clean, falsifiable, cheap-to-run prediction using the
existing harness, no new kernels needed:** threshold selection with
ORACLE tau should drive the 3.60% natural mis-routing floor to ~0, and
should NOT materially improve the 83% adversarial number. If either
half fails, the mental model is wrong -- explicitly the most valuable
possible outcome either way.

**Revised queue:** don't replace the sparse-gather cost work (item 2),
generalize it -- the gather machinery (take indices, exact-attend over
them) is identical for both bucket routing and threshold selection, so
building it selection-agnostic serves both without duplicating work.
Sequence: (1) the oracle-tau vs reservoir-tau correctness experiment,
cheap, no kernels -- up next; (2) the now-broadened cost
implementation; (3) full Fenwick integration only if (1) is positive;
(4) trained-landmark hedge stays parked but with improved priors from
the Landmark Attention literature match.

**Other findings:** MoBA's SNR formula adopted as a closed-form
explanation of the measured 3.6% floor, with a real tension flagged for
later reconciliation -- SNR says smaller blocks help, but this
session's own sweep found tail-risk WORSE at small occupancy, so
something outside the SNR model (centroid variance, boundary effects)
is driving that. Titans/Gated DeltaNet parked (different paradigm,
requires training, abandons zero-shot). NoMAD-Attention judged relevant
to the SEPARATE Jumping Seedling CPU-kernel track, not this attention-
mechanism thread.

**Status: proceeding to the oracle-tau vs reservoir-tau correctness
experiment next.**

---

## Oracle-tau vs reservoir-tau experiment: both halves of Fable's
## prediction confirmed cleanly

Implemented threshold-based (Louver-style) selection in isolation
(exact scores over the whole stressed block, no bucketing, no new
kernels) and ran it on the exact same natural and adversarial harnesses
used throughout this arc.

**Part 1, natural mis-routing floor, n=200:**

| tau_mode | mean cos | fail rate | avg survivors |
|---|---|---|---|
| oracle | 0.9991 | 0.00% | 1.0/128 |
| reservoir | 0.9620 | 0.00% | 13.0/128 |

Both drive the natural 3.60% mis-routing floor to exactly 0% -- even
the realistic, needle-blind reservoir estimator (top-10% quantile of
the block's own scores) works essentially perfectly here.

**Part 2, adversarial construction, n=30:**

| tau_mode | mean cos | fail rate |
|---|---|---|
| oracle | 0.2162 | 90.00% |
| reservoir | 0.2154 | 90.00% |

Identical to each other, and identical to the DENSE-ATTENTION baseline's
own 90.00% fail rate found earlier in this arc -- actually worse than
bucket routing's 83.33%. Makes sense in hindsight: oracle tau set just
below the needle's own (unremarkable) score includes essentially
everything scoring at or above it, including the decoy -- converging
toward dense attention's own behavior in this regime. Threshold
selection buys nothing here because the adversarial failure was never
an exclusion problem.

**Both halves of Fable's falsifiable prediction confirmed exactly, not
overturned.** Threshold selection is a real, clean fix for the
EXCLUSION failure family (top-k's decoy cliff, bucket routing's natural
floor) and genuinely useless against the INHERITED scoring-stage
vulnerability (the adversarial construction) -- validates the
sharpened reliability principle ("structural inclusion guarantee !=
outcome quality guarantee") with a precise, falsifiable test rather
than accepting the artifact's framing at face value. Files:
`eval_threshold_selection.py`.

**Status: mental model validated. Sending back to Fable to close the
loop, then proceeding to the broadened (selection-agnostic) sparse-
gather cost implementation.**

---

## Fable's close-out, plus the tau-inflation follow-up: resolved in
## the reservoir estimator's favor

Fable's response to the oracle/reservoir results: the exact 90%/90%
symmetry on the adversarial case is itself informative -- it means
tau-estimation quality was never the bottleneck there, since the single
strong decoy clears tau under both estimators equally, so the failure
happens downstream in scoring, not admission. Flagged one real gap:
that construction never exercised tau-estimation's precision on the
EXCLUSION side. Proposed the specific test: inject k MODERATE
(individually-unremarkable) decoys, sweep k and strength, oracle tau
held fixed at the true no-decoy quantile as a control, reservoir tau
re-estimated from the corrupted sample -- the "many weak attackers
shift the population" attack shape, different from "one strong decoy."
Also sharpened the integration recommendation: threshold selection
should be the DEFAULT front-end now (strictly dominates bucket routing
on every natural-case number, no natural-case regime where bucket
routing wins), bucket routing kept only as fallback pending this test;
cost/cache-blocking plans should be built around the RESERVOIR
estimator's survivor count (~13/128), not oracle's (1/128, a
correctness ceiling not a deployable estimator).

**Ran the tau-inflation test (`eval_tau_inflation.py`): resolved
cleanly in the reservoir estimator's favor, with an unexpectedly
positive mechanism, not just "no problem found."**

Needle exclusion rate: **0.0% across every configuration tested** --
both oracle and reservoir tau, decoy count 0-40, decoy strength
0.8x-1.5x the needle's own norm. The needle is never pushed below
threshold by many moderate decoys. What happens instead is graceful
dilution (mean cos 0.957 -> 0.51 at the harshest setting, 40 decoys at
1.5x strength) -- a weight-competition effect, not exclusion.

**Reservoir tau consistently OUTPERFORMED oracle tau, not just matched
it** (e.g. 0.5643 vs 0.5084 mean cos at the harshest setting).
Mechanism: as decoys accumulate, reservoir tau (estimated from the
corrupted sample) rises ABOVE the fixed oracle baseline, becoming more
selective. Since these decoys are individually weaker than the
needle's own score, the rising threshold filters out more of the WEAK
competing decoys while the needle keeps clearing it easily -- reducing
dilution rather than risking exclusion. The adaptive estimator is
mildly SELF-CORRECTING in this regime, not fragile.

**Conclusion: the relocated-attack-surface concern doesn't materialize
as feared.** Tau-inflation from many moderate decoys doesn't drag the
needle below threshold -- it only becomes a real problem once a SINGLE
decoy is individually stronger than the needle itself, which is exactly
the original single-strong-decoy construction already classified as an
inherited, unfixable-at-selection-layer scoring vulnerability (matches
dense attention's own 90% failure rate). No new crack found; an
additional piece of clean good news for threshold selection. Files:
`eval_tau_inflation.py`.

**Status: threshold selection (Louver-style) now the validated default
front-end for MetaMonarchAttention's read step -- exclusion failures
resolved (0% floor, both natural mis-routing and moderate-decoy
inflation), scoring-stage vulnerability correctly identified as
inherited and out of scope for any selection-layer fix. Proceeding to
the broadened, selection-agnostic sparse-gather cost implementation
next, built around the reservoir estimator's real survivor-count
operating point (~13/128, ~10%).**

---

## Real sparse-gather cost implementation: honest result, same lesson
## FlashMonarchAttention taught earlier

Built two implementations of the identical threshold-selection math:
`full_mask` (dense (B,Bl) scores, mask non-survivors to -inf, single
dense matmul against the full v_block -- what every reference
implementation in this session has actually done) and `sparse_gather`
(real gather via `torch.nonzero` + `scatter_reduce` + `index_add`,
touching ONLY survivor (query,key) pairs in the value-aggregation step,
never the full block). Correctness verified first (max abs diff
1.49e-07 between the two -- same computation, different execution path).

**Wall-clock: sparse_gather is SLOWER than full_mask at every block
size tested (0.40x-0.79x, i.e. sparse takes 1.3-2.5x longer), despite
touching only ~10% of the entries:**

| Bl | full_mask | sparse_gather | speedup | avg survivors |
|---|---|---|---|---|
| 128 | 0.107ms | 0.136ms | 0.79x | 12.8 |
| 512 | 0.137ms | 0.197ms | 0.70x | 51.2 |
| 2048 | 0.387ms | 0.959ms | 0.40x | 204.8 |
| 8192 | 2.108ms | 3.366ms | 0.63x | 819.2 |
| 32768 | 6.985ms | 12.477ms | 0.56x | 3276.8 |

**Same lesson as FlashMonarchAttention's naive first benchmark:**
`torch.nonzero`/`scatter_reduce`/`index_add` each carry real per-op
dispatch overhead and irregular memory access; `full_mask`'s value
aggregation is a single, highly-optimized dense matmul. Doing
genuinely less arithmetic doesn't win if it costs more, smaller,
irregular ops in a high-level runtime.

**Important caveat, not a final verdict, stated precisely:** this is
PyTorch reference code specifically. The earlier FlashMonarchAttention
investigation found that PyTorch op-dispatch overhead can make a
genuinely-less-work approach look worse than compiled code would show.
This result says "the naive PyTorch sparse-gather path isn't a free
win" -- it does NOT say "sparse gather can't work on the actual
Rust/AVX2 target." Those are different claims; only the first has
evidence here. Files: `eval_threshold_sparse_gather.py`.

**Status: complicates the "threshold selection as default" recommendation
on the COST axis specifically -- the QUALITY case (0% exclusion floor
vs bucket routing's 3.6%, robust to moderate-decoy inflation) remains
completely intact and unaffected by this result. Sending back to Fable
for reaction before deciding how to proceed.**

---

## Fable's response, plus the padded-dense follow-up: decisive, PyTorch
## reference timing hits a genuine wall here

**Fable's reaction, precise:** the recommendation splits into two
independently-decided halves. The QUALITY case (0% exclusion floor,
now doubly load-bearing after the clean tau-inflation result) is a
property of the selection predicate -- confirmed, unaffected. The COST
case is a property of THIS specific PyTorch realization -- not yet
informative either way, same posture the FlashMonarchAttention
investigation eventually settled into (don't retract a correctness win
over a dispatch-overhead artifact, don't claim an unearned cost win
either). Proposed one more cheap, diagnostic test before escalating to
a real kernel: pad to a fixed max-survivor-count and do a single dense-
but-smaller matmul (torch.topk + fixed-shape gather, no scatter_reduce/
index_add) -- if THAT also loses, that's a stronger, more decisive
result than the first.

**Ran it. Decisive, and it's the stronger result Fable predicted.**
Padded-dense (`padded_dense_attention`, pad_width ~1.5-2x average
survivors) verified correct first (max diff 1.19e-07 vs full_mask),
then benchmarked:

| Bl | pad_width | full_mask | sparse_gather | padded_dense | pd speedup |
|---|---|---|---|---|---|
| 128 | 24 | 0.092ms | 0.140ms | 0.138ms | 0.67x |
| 512 | 96 | 0.133ms | 0.198ms | 0.288ms | 0.46x |
| 2048 | 384 | 0.331ms | 0.390ms | 0.874ms | 0.38x |
| 8192 | 1536 | 1.318ms | 1.872ms | 2.976ms | 0.44x |
| 32768 | 6144 | 4.876ms | 11.361ms | **18.197ms** | **0.27x** |

**Padded-dense also loses, and by MORE at scale -- actually worse than
sparse_gather at the largest size tested** (18.2ms vs 11.4ms at
Bl=32768). `torch.topk` itself carries real overhead (partial sorting,
not free), making the "smarter," dispatch-minimized approach the WORST
of all three at large Bl. Two genuinely different sparse strategies
(ragged scatter-based, fixed-shape topk-based) both lose to naive dense
masking, at every scale tested. Exactly Fable's predicted stronger
conclusion: PyTorch's fixed per-op dispatch costs dominate here
regardless of gather strategy -- real evidence (not just diagnosed
cause) that no further PyTorch-level rewrite will reveal sparse
gather's real advantage. A compiled kernel is the honest next step to
evaluate this cost question at all, matching the pattern already
established for FlashMonarchAttention's own eventual CPU/GPU
resolution. Files: `eval_threshold_sparse_gather.py` (updated).

**Standing methodology note, written down per Fable's suggestion so
this doesn't get re-litigated a third time:** PyTorch-level timing in
this arc has now produced one false negative (Flash fusion, reversed
once shape/dispatch was isolated) and one confirmed-genuine wall
(sparse gather, two independent strategies both lose, real signal not
just an artifact). Reference-implementation wall-clock numbers in this
codebase should be treated as CORRECTNESS SCAFFOLDING ONLY -- never a
cost verdict about the Rust/AVX2 target -- until the actual kernel is
written. Any cost claim made from PyTorch timing alone should be
labeled provisional in the same entry that makes it, not corrected
retroactively.

**Status: MetaMonarchAttention's design is now settled on quality
grounds (threshold selection, Louver-style, is the validated default
read mechanism, exclusion failures resolved, scoring-stage
vulnerability correctly scoped as inherited and out of the selection
layer's reach) and explicitly UNSETTLED on cost grounds pending a real
Rust/AVX2 kernel -- not a gap to keep chasing in PyTorch further, a
genuine boundary of what this reference-code investigation can honestly
answer.**

---

## Fable's overall-next-move check-in, and the first full integration
## test -- T-iteration structurally superseded, plus a real new trade-off

Asked Fable for the overall next step given both halves of threshold
selection (quality settled, cost unsettled pending a real kernel).
Identified three items still doable in Python (none requiring the Rust
kernel): (1) a full end-to-end integration test -- every prior
threshold-selection result was measured on an isolated single block,
never the assembled system (Fenwick tiers + threshold-selection read +
Multipole-style residual centroid + window + whatever's left of T),
flagged as the last real correctness gap; (2) an analytical (not
wall-clock) roofline cost model for the 5500U, using numbers already in
hand; (3) a quick re-check of whether the MoBA-SNR-vs-tail-risk tension
resurfaces at the threshold-selection ball-tree's leaf-size parameter.
Trained-landmark hedge and the remaining 38-probe-list items stay
parked. Recommended sequence: (1) then (3) then (2), full handoff to
kernel-writing after.

**Built and ran item 1 (`ma_meta_threshold.py`,
`eval_meta_threshold_integration.py`): Fenwick dyadic tier selection +
threshold-selection read (real exact attention over survivors) +
Multipole-style residual-centroid for non-survivors + the exact local
window -- with NO Monarch/Sinkhorn T-iteration refinement anywhere.**

**Deliberate architectural claim tested directly, not assumed:**
Monarch's T-iteration machinery exists to build a good APPROXIMATE
representative cheaply, avoiding an O(B_l) real read. Threshold
selection already reads REAL keys directly for the survivor set -- so
once every tier is read this way, there is no more Monarch
representative left to refine. This is a stronger claim than "reduces
dependence on T" -- it says T becomes structurally UNUSED, not just
less important. Causal validity held (0 leakage, row sums 1.0).

**Distance axis: confirmed decisively, in the strongest possible
form.** Threshold-selection-without-T matches ground truth EXACTLY at
every distance tested, including the hardest (1.0000 at dist=14) --
where SlidingMonarch needed T=8 just to reach 0.9387 without ever
quite hitting perfect, and was near-useless at T=1 (0.0984-0.2295).
Zero Monarch refinement iterations, better result than eight.

**Decoy axis: a genuine, non-obvious trade-off, not another clean win.**
Threshold selection starts far stronger (0.9580 vs 0.2868 at 0 decoys)
but degrades faster under decoy pressure and goes NEGATIVE at 50
decoys (-0.1312), while SlidingMonarch's much-weaker-baseline
T-refined representative stays weakly positive throughout (0.1444 at
50 decoys). Plausible mechanism: many decoys individually strong enough
to also clear tau become real competing survivor candidates in the
same joint softmax (threshold selection guarantees inclusion, not
favorable competition once included -- the already-established
limitation). Monarch's compression, while much weaker at capturing a
genuine distant signal, may incidentally average away some decoy-
specific signal too, giving accidental damping against high decoy
counts that real-key reads don't get for free. Not predicted going in;
a real result from actually building and running the assembled system,
exactly the kind of interaction effect component-level tests couldn't
have shown. Files: `ma_meta_threshold.py`,
`eval_meta_threshold_integration.py`.

**Status: sending this to Fable -- it both confirms the strongest form
of the T-supersession hypothesis and surfaces a genuine new open
question (does the design need BOTH mechanisms, threshold selection for
distance and something Monarch-like for decoy-damping, rather than one
replacing the other outright?) before calling MetaMonarchAttention's
integration complete.**

---

## Fable's response, plus the tier-concentration diagnosis: a genuinely
## new, structural failure mode confirmed, not a re-derivation

**Fable's reframing of the damping hypothesis:** correct in substance,
relocated in mechanism. Not "incidental averaging helps" -- the SAME
capacity ceiling from the very first Version B finding, now showing
its other side. A fixed-size compressed representation can't capture a
real signal well (the disqualifying half, already established), but
that same boundedness caps how much adversarial mass can ever enter the
final read, no matter how many decoys exist upstream. Threshold
selection has no such cap by design -- exactly what makes it win on
distance (every real signal gets an uncompressed read) and lose on
decoy count (every survivor, real or decoy, adds full uncapped weight
to the same joint softmax). The 0-decoy and 50-decoy numbers are "the
same lever, cutting both ways," not two separate phenomena.

**Corrected the T-supersession claim rather than retracting it:** what's
unused is ITERATIVE REFINEMENT for the distance problem specifically --
holds cleanly, confirmed by the distance table. What's still needed is
something with Monarch's BOUNDEDNESS property as a counterweight to
threshold selection's unboundedness -- a property of fixed-capacity
compression in general, not Sinkhorn iteration specifically.

**Flagged one diagnostic before trusting the framing fully:** are the
50 decoys concentrated in the needle's own tier (re-derivation of the
already-known same-bucket finding) or spread across many tiers (a
genuinely new failure mode -- each tier independently applies its own
tau test with no knowledge that a sibling tier already admitted several
decoys, so survivor count accumulates ADDITIVELY across the whole tree)?

**Ran it (`eval_tier_concentration_diagnosis.py`): confirms the new,
more concerning failure mode, not a re-derivation.**

```
tier l=0 (Bl=16):  2 survivors, 2 decoys
tier l=1 (Bl=32):  4 survivors, 4 decoys
tier l=2 (Bl=64):  7 survivors, 7 decoys
tier l=3 (Bl=128): 13 survivors, 13 decoys  <- contains needle
```

**Concentration: exactly 50%.** The needle's own tier has 13 decoy
survivors, but ANOTHER 13 are spread across the three sibling tiers
that have nothing to do with the needle at all -- each tier
independently admits ~10% of its own block as survivors, unaware three
other tiers are doing the same into the SAME final softmax. This
confirms the structurally new failure: total competing candidate pool
scales with the NUMBER OF ACTIVE TIERS (L), not just with decoy density
wherever the needle happens to sit. A decoy in a tier that doesn't even
contain the needle still contributes real competing mass to the same
softmax the needle's output comes from -- dilution without co-location.
For longer sequences with deeper Fenwick hierarchies this would amplify
further, independent of where the needle actually is. This would not
show up in ANY single-tier test -- exactly why the integration test
mattered; the reliability principle's structural-inclusion guarantee
holds per-tier but the aggregate competitive pressure across tiers
compounds in a way isolated component tests couldn't reveal. Files:
`eval_tier_concentration_diagnosis.py`.

**Status: sending this confirmed-new-failure-mode result to Fable before
deciding whether to build the proposed hybrid (capacity-bounded
aggregation over the query-aware survivor set, post-selection) --
the diagnostic outcome changes how urgently that mitigation is needed,
not just whether it's worth trying.**

---

## Fable's response: reframes the mitigation design, but flags the
## real gating question first -- and it resolves in the reassuring
## direction

**Fable's read on the confirmed cross-tier finding:** raises the
stakes, since it's a property of hierarchy DEPTH (L, active tier count)
rather than decoy placement -- meaning it scales with exactly the axis
MetaMonarchAttention exists to handle (longer context -> deeper
hierarchy -> more tiers -> more independent tau-admissions feeding one
softmax).

**On mitigation design (global vs per-tier cap):** global, in
principle -- a per-tier cap of size C still permits a total pool of
L*C, exactly what was measured (four tiers, each independently
admitting ~10%, summing to 50%). But flagged a real trap: if "global
cap" means rank-and-truncate the union of all tiers' survivors, that's
top-k with extra steps -- the SAME exclusion cliff this session already
ruled out twice, now relocated to the top of the hierarchy. The
non-regressive version: extend the post-selection, query-aware
compression hybrid to operate over the UNION of survivors across all
active tiers, not each tier separately -- keeps the structural
inclusion guarantee at the tier level while capping total mass
globally, the way Monarch's boundedness did before, but now
post-selection instead of pre-selection (the piece that was missing
when blind pre-selection compression failed identically five times
earlier this session).

**On a simpler fix:** a cheaper variant worth trying before full
aggregation -- one SHARED reservoir-tau estimate across the pooled
score distribution of all active tiers, instead of each tier
independently computing its own local 90th percentile blind to its
siblings. Still an absolute threshold (not rank-truncation, so it
doesn't reintroduce the exclusion cliff), directly targets the
"zero awareness of siblings" gap, much cheaper than the aggregation
hybrid.

**But identified the actually-most-urgent question, ahead of building
either fix:** does cross-tier accumulation also degrade the NATURAL
(zero-decoy) case as tier count grows with sequence length? If yes,
this stops being a bounded/accepted adversarial risk and becomes a
core-value-proposition problem -- the architecture's own long-context
scaling axis would ALSO be its core vulnerability, since tau is
quantile-based (relative, not absolute) and every tier structurally
admits ~10% of its own block regardless of relevance, even with zero
adversary.

**Ran it (`eval_natural_tier_scaling.py`): resolved decisively in the
reassuring direction.** Swept N=256 through N=8192 (active tier count
4 through 9), needle strength and position fixed, ZERO decoys:

| N | active tiers | mean cos | min cos |
|---|---|---|---|
| 256 | 4 | 1.0000 | 1.0000 |
| 512 | 5 | 1.0000 | 1.0000 |
| 1024 | 6 | 1.0000 | 1.0000 |
| 2048 | 7 | 1.0000 | 1.0000 |
| 4096 | 8 | 1.0000 | 1.0000 |
| 8192 | 9 | 1.0000 | 1.0000 |

**Zero degradation at any tier count tested, min identical to mean at
every configuration.** Background survivors admitted by other tiers
are still ordinary random content, not adversarially engineered -- so
even as more of them accumulate with growing tier count, their
aggregate weight relative to a strongly-aligned needle score stays
negligible. Confirms the failure is ADVERSARIAL-PRESSURE-SPECIFIC, not
a core-scaling-axis vulnerability -- exactly the outcome Fable was
hoping to rule out, and it did. Files: `eval_natural_tier_scaling.py`.

**Status: the cross-tier accumulation finding stays filed as a
bounded, accepted risk -- same category as every other adversarial
cliff in this session -- rather than a blocking architectural problem.
The natural case, which is the vast majority of real usage and the
architecture's actual value proposition, is completely unaffected by
tier count growth. Mitigation work (shared-tau estimate, or the
post-selection global aggregation hybrid) remains worth doing
eventually but is no longer urgent -- can be sequenced behind the
cost/kernel work rather than ahead of it.**

## Shared-tau: the last correctness-side experiment

Fable's final recommendation before declaring the design phase closed:
build the shared-tau estimate now anyway ("cheap insurance, not
deferred work"), since the fix is small and the natural-case result
already showed it isn't urgent -- rather than leaving it as unfinished
business. Implemented `ma_meta_threshold_shared_tau.py`: identical
Fenwick-tier/threshold-selection/residual-centroid/window structure to
`ma_meta_threshold.py`, but restructured into two passes per query --
pass 1 computes raw scores for every active tier's candidate block and
pools them into one combined sample; pass 2 computes a SINGLE
`torch.quantile` from that pooled sample and applies the same tau to
every tier's own survivor test, instead of each tier independently
computing its own local 90th percentile with zero awareness of
siblings (the mechanism the tier-concentration diagnosis pinned down).

**Causal validity confirmed first:** leak = 0.00000000 (perturbing a
future key at position 200 leaves all outputs at positions <190 exactly
unchanged), all outputs finite. Files: `ma_meta_threshold_shared_tau.py`.

**Ran the 50-decoy cross-tier adversarial sweep** (`eval_shared_tau_check.py`,
same construction as `eval_tier_concentration_diagnosis.py` /
`eval_meta_threshold_integration.py`'s decoy sweep), comparing
shared-tau against the existing independent-per-tier-tau baseline:

| num_decoys | independent-tau | shared-tau |
|---|---|---|
| 0 | 0.9580 | 0.9399 |
| 5 | 0.7205 | 0.7290 |
| 20 | 0.4445 | 0.4358 |
| 50 | -0.1312 | -0.1228 |

**Null result -- shared-tau does not meaningfully narrow the gap.**
Differences at every decoy count are within trial noise (largest gap
0.018, no consistent direction: shared-tau is fractionally worse at 0
and 20 decoys, fractionally better at 5 and 50). At the worst case (50
decoys) both variants land in the same negative-cosine failure regime.
Sharing tau across tiers does not change the fundamental problem:
adversarially-crafted decoys score high enough to clear ANY reasonable
absolute threshold, pooled or per-tier -- the scoring-stage
vulnerability threshold selection inherits from dense attention itself
(established earlier: matches dense attention's 90% adversarial fail
rate exactly) dominates over the cross-tier-awareness gap it was meant
to close. Files: `eval_shared_tau_check.py`.

**Status: shared-tau implemented, validated, and tested -- confirmed as
cheap insurance that costs nothing and closes out the mitigation
question, but does not measurably improve the crafted-adversarial
case, consistent with that case being accepted as unfixable at the
selection layer. Sent to Fable.**

**Fable's response: agrees this is a clean null and closes the thread.**
Frames it as the FOURTH independent confirmation that the
crafted-adversarial gap sits at the scoring layer, not the selection
layer -- dense attention 90%, bucket routing 83%/80%, oracle/reservoir
threshold-selection 90%/90%, now shared-tau 90%-equivalent -- and says
this classification should now be treated as settled, not tentative;
further selection-layer mitigation attempts would be re-testing an
already-falsified hypothesis. Cross-tier accumulation stays filed as
adversarial-only and non-blocking (confirmed zero-impact in the natural
case regardless of tau scheme, per the N=256->8192 sweep).

**Status: MetaMonarchAttention design-quality thread declared COMPLETE.**
Final state carried forward: threshold selection (shared-tau variant)
as the read mechanism, no T-iteration, Multipole-style residual
centroid for non-survivors, cross-tier accumulation filed as
adversarial-only/non-blocking, crafted-adversarial gap filed as
inherited-and-accepted with four-way cross-validation. Fable's
explicit go-ahead: proceed to item 2, the analytical (not wall-clock)
roofline cost model for the 5500U, using this confirmed read pattern.
Flagging for user check-in before starting item 2, since it's a new
substantial piece of design/analysis work.

## Item 2 detour: the O(N^2) complexity finding, and the bounding-ball recheck

Before building the roofline model, the FLOPs accounting for
`ma_meta_threshold.py` exposed a complexity-class problem, not a
constant-factor one. The Fenwick tiering picks O(log N) *blocks* per
query, but each tier's block size grows geometrically (`Bl = B*2^l`),
and the block sizes across active tiers sum to `n*B` -- the FULL causal
prefix. Threshold selection needs a real q.k score for every key in
every active tier's block before applying the quantile cutoff, so QK^T
scoring is O(N) per query, O(N^2) total -- the SAME complexity class as
dense causal attention. Sent to Fable.

**Fable's response: this is real, and traced to a specific gap.** The
original Stage-1 recommendation ("threshold/bounding-ball test -- skip
a block only if PROVABLY below tau, contribute centroid instead of
nothing for skipped blocks") only got its second half implemented --
the Multipole-style centroid -- never the actual pruning half. The
current build scores every key first, then filters -- masking, not
range-search. Fable's proposed fix: precompute a bounding ball per tier
block (center = mean of keys, radius = max distance from center to any
key), once, query-independent; at read time, a single O(1)
Cauchy-Schwarz test (`sm_scale*(<q,center> + ||q||*radius) < tau` =>
provably prunable) skips real scoring for blocks that fail it. Fable
draws a real distinction from the five prior failed centroid-style
mitigations (floor-read, robust/geometric-median centroid, etc.): those
were CONTENT proxies (guess what's probably in the block, fails when
the needle is atypical); a bounding ball is a PROVABLE UPPER BOUND (by
convexity, nothing inside can score above the bound, regardless of how
atypical the true point is) -- a different mathematical object, not the
same failure mode recurring a sixth time.

**Implemented and tested** (`ma_meta_threshold_ball_prune.py`,
`eval_ball_prune_check.py`). Necessary structural change beyond just
adding the bound test: the bound test needs an ABSOLUTE, pre-score tau
-- the original per-tier quantile is computed FROM that tier's own
scores, a chicken-and-egg problem for pruning. Seeded tau from the
local window's own exact, always-computed scores instead, used as one
shared threshold for both pruning and survivor selection.

Results:
- Causal validity: leak=0.00000000, all outputs finite. Clean.
- Natural (N=256, n=200): 0.00% fail rate, matches baseline exactly.
- 50-decoy adversarial sweep: did NOT come back identical to baseline
  (0.8188/0.6495/0.4496/0.2652 vs. baseline's
  0.9580/0.7205/0.4445/-0.1312 at 0/5/20/50 decoys) -- worse at low
  decoy counts, better at 50. Diagnosed as a side effect of switching
  tau's SOURCE (local window, only 16 keys) rather than a bug in the
  bound math, and flagged as a real design delta rather than assumed
  benign.
- **Prune rate across N=256->8192, zero decoys: 0.00% at every single
  scale tested.** The bound test never fired once.

**Ran the follow-up oracle-tau isolation Fable requested**
(`eval_oracle_tau_prune_check.py`): tau computed from the FULL pooled
real scores across all active tiers (diagnostic-only, cheats with full
knowledge, isolates "are the balls too loose" from "was the local-
window tau_seed biased/noisy"). **Result: 0.00% prune rate at every N
from 256 to 8192, again.** Confirms the pessimistic branch decisively
and independent of tau source: the bounding balls are genuinely too
loose on this key geometry (D=16, the toy dimension used throughout
this session for fast iteration -- flagged as a real transferability
caveat, not chased further here).

**Status: ball-pruning branch discarded, confirmed nonviable on this
data. The O(N)-per-query / O(N^2)-total complexity of threshold
selection stands as a genuine property of the design, not an
implementation gap.** No complexity-class recovery is available via
bounding-ball pruning. Item 2 reverts to being a constant-factor cost
model (dense O(N^2) vs. threshold-selection O(N^2) with a smaller
AV/softmax coefficient from the ~10% survivor rate), scoped over the
original confirmed design: shared-tau threshold selection, no
T-iteration, Multipole-style residual centroid, no pruning. Sent to
Fable.

**Fable's correction, before any roofline write-up started:** the
"~10% survivors" figure only applies to the AV/softmax stage, not to
total attention cost. Dense attention splits roughly evenly between two
comparable-size stages per query: QK^T (score every key) and AV
(weight-and-sum every value). In this design, QK^T gets ZERO savings
(every key in every active tier is still scored, per the O(N^2)
finding above) -- only AV+softmax benefits from the survivor rate, and
even there total survivors summed across all active tiers still scale
~O(N) (dominated by the largest/coarsest tier), just with a constant
discount (~5-10x). Net: honest blended speedup ceiling is ~1.5-2x, not
the order-of-magnitude the AV-only figure would suggest in isolation.
Also recommended memory traffic / working-set footprint be reported as
its own axis in the roofline (separate from FLOP count), given the
actual target is cache-constrained (8MB L3) -- the AV stage's smaller
value working set could matter for cache residency independent of the
FLOP ratio.

**User asked to verify this empirically before proceeding.** Built
`eval_flop_accounting.py`: instruments a real run of
`monarch_meta_threshold_shared_tau` (final confirmed design) with real
random data at N=256 through 4096, counting actual QK^T terms (every
real key scored, every active tier) and actual AV terms (local window +
real survivors + one residual centroid per tier with any non-survivor),
compared against dense attention's `N(N+1)/2` causal pair count.

| N | QK ratio | AV ratio | blended ratio |
|---|---|---|---|
| 256 | 1.000 | 0.178 | 0.589 |
| 512 | 1.000 | 0.141 | 0.571 |
| 1024 | 1.000 | 0.122 | 0.561 |
| 2048 | 1.000 | 0.111 | 0.556 |
| 4096 | 1.000 | 0.106 | 0.553 |

**Confirms Fable's correction precisely.** QK ratio is exactly 1.000 at
every N (zero savings on scoring, exact match to prediction). AV ratio
converges toward ~0.10 as N grows (matches the estimated 5-10x
discount). Blended ratio settles around 0.55 -- a real, honest **~1.8x
speedup ceiling**, landing inside Fable's predicted 1.5-2x range, not
the misleading order-of-magnitude figure the AV-only fraction would
imply read in isolation. Files: `eval_flop_accounting.py`.

**Status: stage-separated FLOP accounting empirically verified.**
Roofline write-up scope is now locked: two-stage accounting (QK^T: 1.0x
unchanged, AV+softmax: ~0.1-0.2x discount, blended: ~0.55x / ~1.8x
speedup ceiling) plus memory-traffic/working-set as a separate axis,
over the confirmed design (shared-tau threshold selection, no
T-iteration, Multipole-style residual centroid, no pruning). Fable
requested the memory-traffic axis be measured, not just proposed --
sent to Fable.

## Memory-traffic axis: measured, and it reframes the headline number

Fable's ask (measure, don't estimate, since the instrumentation harness
already existed) surfaced a real scoping question: a naive byte-count
extension of the FLOP counters would just reproduce the FLOP ratios
(bytes-per-term and FLOPs-per-term both scale linearly with D), adding
no information. Real memory-traffic analysis requires modeling cache
REUSE, not just counting touches -- scoped as two honest BOUNDS rather
than a full LRU/eviction simulator (a simulated cache model in Python
carries the same false-precision risk already ruled out for PyTorch
wall-clock timing -- not trustworthy until there's a real Rust/AVX2
kernel to profile with real hardware counters on the actual 5500U).

Key reframing: threshold selection reads REAL keys/values directly
(`k_flat.view(...)` is a reshaped VIEW over the same underlying K,V
tensors, not a separate compressed copy per tier) -- so DRAM-resident
DATA VOLUME is identical between dense and threshold selection: N*D*4 +
N*Dv*4 bytes per head. The real question is access pattern / reuse, not
footprint.

Built `eval_memory_traffic_bounds.py` at PRODUCTION-scale parameters
(D=Dv=64, H=8 -- not the toy D=16 used elsewhere in this session for
fast iteration, since toy-D trivially fits in L3 and would make the
axis uninformative). Two bounds: FLOOR (every K/V byte read exactly
once, best case, identical for both mechanisms) and NAIVE (actual
measured traffic in the current query-major loop, zero reuse credit
beyond within-query-group batching).

**First pass had a bug, caught before reporting**: the dense baseline
used raw O(N^2) per-query-pair counting while the threshold-selection
count credited free reuse within each batch of B queries sharing a
key-block -- an apples-to-oranges comparison that produced a misleading
~60x-favorable ratio. Fixed by applying the SAME block-tiled convention
(FlashAttention-style: a block of B queries shares one read of its
causal-prefix keys/values) to dense's baseline too.

**Corrected results:**

| N | K+V floor (MB) | fits L3? | dense naive (MB) | threshold naive (MB) | ratio |
|---|---|---|---|---|---|
| 1024 | 4.19 | yes | 35.7 | 39.8 | 1.117 |
| 2048 | 8.39 | yes | 138.4 | 148.7 | 1.074 |
| 4096 | 16.78 | no | 545.3 | 569.6 | 1.045 |
| 8192 | 33.55 | no | 2164.3 | 2219.8 | 1.026 |
| 16384 | 67.11 | no | 8623.5 | 8744.8 | 1.014 |

**Threshold selection shows essentially ZERO memory-traffic advantage
over a properly block-tiled dense implementation -- ratio converges to
~1.0 as N grows, slightly WORSE at smaller N.** Once dense is also
tiled (queries in blocks of B, sharing one causal-prefix key read per
block), its own traffic already drops to O(N^2/B), the same order
threshold selection achieves. The earlier ~1.8x figure was a
COMPUTE-only win; on bytes moved, threshold selection and well-tiled
dense are roughly break-even. K+V no longer fits in L3 past N~2048 at
this production scale, confirming genuine cache pressure exists (unlike
the toy-D case, where everything trivially fit). Files:
`eval_memory_traffic_bounds.py`.

**Status: this reframes the item-2 headline.** Since attention is
typically memory-bandwidth-bound rather than compute-bound on real
hardware, and threshold selection's memory-traffic profile is roughly
break-even with well-tiled dense, the realizable wall-clock benefit on
the 5500U could sit much closer to ~1.0x than the ~1.8x compute
ceiling -- a materially more sobering number than what was sent to
Fable in the previous round. Sending this to Fable now, before writing
the roofline document, since it may bear on whether the Rust/AVX2
investment is worth it at all.

## Ridge-point check: resolves whether the FLOP win is realizable

**Fable's response**: don't treat "close to 1.0x" as final -- one more
cheap, purely arithmetic step (no new instrumentation, no simulator)
resolves it: compare the kernel's implied arithmetic intensity (FLOPs
per byte, from numbers already measured) against the 5500U's roofline
ridge point (peak AVX2 FLOP/s / peak DRAM bandwidth). Below the ridge =
memory-bound, cutting FLOPs buys ~nothing; above it = compute-bound,
the FLOP win shows up directly. Also flagged a real practical concern:
the memory-traffic ratio was WORST exactly at N=512-4096 (1.117 down to
1.045), the range most plausibly relevant to this project, not the
N=16384+ range where it approaches 1.0 -- worth checking Fydel's actual
target context length against that table before writing the headline
number.

**Checked**: `RESEARCH_LOG.md`'s full-attention-layer benchmarks use
context lengths 512/2048/8192, with full-attention layers identified as
~92% of decode cost at ctx=8192 -- confirming the realistic operating
range is exactly where the memory-traffic ratio was weakest, not the
asymptotic-comfort zone.

**Built `eval_roofline_ridge_point.py`**: computes 5500U peak
FLOPs/bandwidth from public specs (Zen 2, 6 cores, AVX2 256-bit, dual-
channel DDR4-3200), both a theoretical-peak bound and a
conservative-efficiency estimate (flagged explicitly as unmeasured,
same caveat as everywhere else in this session PyTorch/analytical
numbers stand in for real hardware profiling): ridge point ~4.6-7.9
FLOPs/byte. Computes implied AI for dense (tiled) and threshold
selection from the FLOP counts (`eval_flop_accounting.py`) and byte
counts (`eval_memory_traffic_bounds.py`) already measured.

**First pass had a bug, caught before reporting**: used H=8 uniformly
for both compute and memory scaling, but forgot to multiply
`dense_flops` by H at all (single-head FLOPs vs. multi-head bytes) --
produced a spurious result where threshold selection's AI looked HIGHER
than dense's, backwards from the structural expectation (fewer FLOPs +
~same bytes should give LOWER AI). Fixed the H-scaling bug, then went
further: checked the real production config (`src/model/config.rs`)
instead of assuming uniform H=8. Found `head_dim=64` and `kv_block=64`
matched the assumption, but the model uses **grouped-query attention**:
`n_q_heads=14`, `n_kv_heads=2` (7 query heads share each KV head) --
compute scales with n_q_heads, K/V memory traffic scales with the much
smaller n_kv_heads. Rebuilt the script on the real GQA config.

**Corrected, production-config results:**

| N | dense AI | thresh AI | regime |
|---|---|---|---|
| 512 | 199.5 | 104.5 | both compute-bound |
| 1024 | 211.0 | 109.8 | both compute-bound |
| 2048 | 217.3 | 114.4 | both compute-bound |
| 4096 | 220.6 | 117.8 | both compute-bound |
| 8192 | 222.3 | 120.1 | both compute-bound |

**Both mechanisms land 15-50x above the ridge point at every target
context length -- decisively compute-bound, not memory-bound.** GQA's
low n_kv_heads=2 (vs n_q_heads=14) makes K/V memory traffic much
smaller relative to compute than a uniform-head model would suggest,
pushing the result further into compute-bound territory than the
initial H=8 approximation showed. Files: `eval_roofline_ridge_point.py`.

**Status: the sobering "close to 1.0x" memory-traffic framing does NOT
dominate -- the ~1.8x FLOP reduction should be realizable in wall-clock
time**, because bytes were never the bottleneck at this design's real
GQA configuration. This reverses the previous round's tentative
conclusion, now on a decisively stronger empirical footing (real
production config, not an assumed uniform-head approximation). Sent to
Fable.

## The residual-centroid cost: the omission that erases the whole win

**Fable's response gave the write-up go-ahead, with one required
addition**: before presenting the 1.8x figure as a wall-clock
expectation, account for threshold selection's own bookkeeping/overhead
(tau maintenance, Fenwick traversal, gather/index logic, residual-
centroid combination) as O(log N) or O(1) against the O(N) dominant
term, since being compute-bound means any uncounted FLOPs now show up
directly in wall time with no memory-stall slack to hide behind. Framed
as a short analytical paragraph, not a new experiment -- "plausibly
negligible is exactly the kind of claim this thread has learned not to
accept without checking."

**Checked it properly instead of asserting it -- and it was NOT
negligible.** The original FLOP accounting (`eval_flop_accounting.py`)
only counted terms entering the FINAL combined softmax (real survivors
+ one residual slot per tier) -- it never counted the cost of
COMPUTING that residual slot: `mean_k`/`mean_v` is a masked reduction
over ALL of a tier's Bl keys (~90% are non-survivors), an O(Bl*D)
operation, the SAME ORDER as that tier's own QK scoring matmul.

Built `eval_flop_accounting_with_residual.py` to measure this
directly:

| N | blended (QK+AV+residual) | blended (+ quantile-sort) |
|---|---|---|
| 256 | 1.056 | 1.167 |
| 512 | 1.054 | 1.184 |
| 1024 | 1.052 | 1.201 |
| 2048 | 1.051 | 1.217 |
| 4096 | 1.051 | 1.232 |

**Including the residual-centroid reduction cost, the blended FLOP
ratio flips from ~0.55 (the reported 1.8x speedup) to ~1.05 --
essentially ZERO net FLOP savings versus dense.** Including the
quantile-sort overhead (torch.quantile is sort-based, O(n log n), not
the O(1)-amortized reservoir-sampling Louver's own design calls for --
the validated shared-tau implementation uses full-pool sorting for
correctness-checking purposes, not the cheaper streaming estimator) it
comes out ~1.05-1.23, i.e. **worse than dense** at every N tested.

The residual-centroid reduction roughly DOUBLES the "unavoidable O(N)"
portion of the kernel (QK scoring + residual reduction, both O(N) per
query), which swallows essentially all of the AV-stage savings that
produced the earlier 1.8x figure. This is the single largest previously
-uncounted cost anywhere in this cost-accounting arc -- not a rounding
error, and it directly validates Fable's own warning about not
accepting "negligible" without checking.

**Status: the 1.8x headline does not survive a complete FLOP account.**
On FLOPs alone, threshold selection as currently designed (with a real
per-tier residual centroid, recomputed fresh per query) offers little
to no computational advantage over dense attention -- independent of
the separately-resolved compute-bound/memory-bound question, which is
now moot if there's no FLOP saving to realize. Open question: whether
the residual-centroid mechanism itself needs to be redesigned (e.g.
computed incrementally/maintained rather than recomputed fresh per
query) to recover any real advantage. Sent to Fable.

## Exact-algebraic residual fix: recovers most of the FLOP saving

**Fable's response**: this is fixable, and NOT via another content-
approximation heuristic (the category that already failed five times
this session) -- an exact algebraic identity instead:
`sum(non-survivors) = sum(all keys in block) - sum(survivors)`.
`sum(all keys in block)` is query-independent (same value for every
query touching that tier's block), so precompute it ONCE when the
block finalizes -- same reuse pattern already validated for bounding
balls and k-means centroids. `sum(survivors)` costs O(num_survivors*D),
same order as work already paid for the AV gather. Since this computes
the IDENTICAL mathematical quantity, not an approximation, none of the
reliability/quality harnesses need rerunning -- only the cost harness.
Also recommended swapping the sort-based `torch.quantile` (O(n log n))
for the reservoir-sampling tau estimator, whose QUALITY was already
validated in the earlier oracle-vs-reservoir round -- pure cost
substitution, no new correctness question.

**Implemented** (`ma_meta_threshold_fast_residual.py`): precomputes
per-tier full-block K/V sums once, computes residual mean via
subtraction instead of a full masked reduction.

**Verified numerically exact first** (`eval_fast_residual_check.py`,
N=256 to 8192): max abs diff ~3-6e-8 (float32 rounding-noise level, not
a real discrepancy) at every N. Causal validity: leak=0.00000000, all
outputs finite. Confirmed before trusting any FLOP-cost claim.

**Re-ran the FLOP accounting** (`eval_flop_accounting_fast_residual.py`)
with both fixes applied -- residual cost counted at its REAL achievable
O(num_survivors*D) (gather-equivalent, same order as already-paid AV
work) plus a one-time O(Bl*D) full-block-sum cost amortized across the
whole sweep (not per query), and tau cost counted at O(reservoir_size)
instead of O(n log n):

| N | QK+AV only | +survivor-gather residual | +onetime block-sum | +reservoir-tau | **Full blended** |
|---|---|---|---|---|---|
| 256 | 0.589 | 0.637 | 0.645 | +0.0007 | **0.646** |
| 512 | 0.571 | 0.620 | 0.625 | +0.0005 | **0.625** |
| 1024 | 0.561 | 0.610 | 0.613 | +0.0003 | **0.614** |
| 2048 | 0.556 | 0.605 | 0.607 | +0.0002 | **0.607** |
| 4096 | 0.553 | 0.603 | 0.604 | +0.0001 | **0.604** |
| 8192 | 0.552 | 0.601 | 0.602 | +0.0001 | **0.602** |

**Full blended ratio settles around ~0.60-0.65 -- a real ~1.5-1.7x
speedup ceiling.** Not the original naive 1.8x (the survivor-gather
residual term is a genuine, non-zero marginal cost -- not literally
free), but a substantial recovery from the ~1.05-1.23x (no-savings-or-
worse) result found before the fix. Files:
`ma_meta_threshold_fast_residual.py`, `eval_fast_residual_check.py`,
`eval_flop_accounting_fast_residual.py`.

**Status: FLOP saving recovered and re-verified at ~1.5-1.7x, on a
design that is numerically identical to the already-validated
reliability results (exact algebraic identity, not a new heuristic).**
This is the number the ridge-point/compute-bound result should now be
checked against, since that check was previously answering "is the
1.8x realizable" -- a premise that no longer holds unmodified. Sent to
Fable.

**Fable's response**: no full re-run needed, and directionally the
correction can only widen the existing margin, never shrink it -- the
fix recovers savings but adds FLOPs versus the naive 0.55x estimate
(0.55x -> ~0.60-0.65x), while bytes moved are completely unaffected (a
separate measurement the residual-computation method doesn't touch).
More FLOPs over the same bytes means HIGHER arithmetic intensity, i.e.
further right on the roofline, deeper into compute-bound territory.
Still asked for the five-line arithmetic recompute rather than skipping
it on the strength of the direction alone -- consistent with the
thread's own discipline ("a plausible-sounding argument gets verified,
not trusted," which is exactly what caught the residual-cost omission
two rounds ago).

**Ran it** (`eval_roofline_ridge_point_v2.py`): recomputed AI using the
corrected fast-residual+reservoir-tau FLOP figures against the same
unchanged byte counts and real GQA production config:

| N | AI (old, pre-fix) | AI (corrected) | regime |
|---|---|---|---|
| 512 | 104.5 | 112.6 | compute-bound |
| 1024 | 109.8 | 119.1 | compute-bound |
| 2048 | 114.4 | 124.5 | compute-bound |
| 4096 | 117.8 | 128.4 | compute-bound |
| 8192 | 120.1 | 130.9 | compute-bound |

**Confirms Fable's directional prediction arithmetically: margin
widened at every N, both stay ~15-28x above the realistic ridge point
(4.62 FLOPs/byte).** Compute-bound conclusion holds on the corrected
FLOP figures. Files: `eval_roofline_ridge_point_v2.py`.

**Status: item 2 is now fully resolved and internally consistent.**
Final numbers for the roofline document: ~1.5-1.7x FLOP-level speedup
ceiling (fast-residual + reservoir-tau design, both fixes verified
numerically exact against the reliability-validated implementation),
confirmed compute-bound at the real production GQA config with a wide
margin (15-28x above ridge point), ball-pruning branch confirmed dead
and discarded (0% prune rate under oracle tau, any N). Proceeding to
write the roofline document itself.

## Item 2 deliverable: ROOFLINE_5500U.md written

Wrote `ROOFLINE_5500U.md`: confirmed final design (Fenwick tiers,
shared-tau threshold selection, reservoir-tau, exact-algebraic
fast-residual, no T-iteration, no pruning, exact local window),
headline ~1.5-1.7x FLOP-level speedup ceiling with the full
stage-by-stage table (QK 1.0x, AV ~0.10-0.18x, residual-centroid cost
called out explicitly as the largest correction in this arc, tau
estimation cost), roofline placement (compute-bound, 15-28x margin at
real GQA production config, held robust across the residual-cost
correction), memory-traffic axis reported as secondary/non-deciding,
bounding-ball pruning documented as tried-and-ruled-out, and an
explicit limitations section (analytical not measured-on-hardware,
D=16 ball-pruning caveat, adversarial-scoring-stage vulnerability filed
separately as a quality property). Sending to Fable for final review.

**Status: item 2 (analytical roofline cost model) complete**, pending
Fable's final pass. This closes the Fable-artifact-driven segment that
began with the Louver/threshold-selection research synthesis --
MetaMonarchAttention's design-quality thread (declared complete
earlier) and its cost thread now both resolved and cross-validated.

## Final review: two precision fixes, both applied

**Fable's final pass**: overall a strong, honest closing document --
corrections represented in the right order and weight, residual-
centroid catch genuinely called out rather than softened. Two issues
flagged, both fixed directly (precision fixes to already-verified
content, not new findings needing re-verification):

1. **Real overstatement**: the document stated "both dense and
   threshold selection land 15-28x above the ridge point," but only
   threshold selection's AI was ever shown in that table -- dense's AI
   (~199.5-222.3, from the earlier round) against the same ridge range
   actually gives ~25-48x, a meaningfully wider margin than threshold
   selection's. Single shared range implied the two mechanisms sit
   closer to the ridge than they actually do. Fixed: table now shows
   both AI columns side by side, prose states both ranges separately
   (thresh ~15-28x, dense ~25-48x) with threshold selection correctly
   identified as the tighter of the two margins.

2. **Completeness gap**: "no T-iteration" was presented as a clean win
   without the caveat, established earlier in this same investigation,
   that removing it also removes Monarch's incidental bounded-
   compression damping against decoy pressure (T-refined
   SlidingMonarchAttention stayed weakly positive at 50 decoys; pure
   threshold selection went negative at the same decoy count). Fixed:
   added to the limitations section, explicitly scoped as a quality
   trade-off (not a cost consideration) with the global-aggregation
   mitigation noted as parked future work, not resolved.

Everything else in the document checked out without changes: the
1.8x -> 1.05-1.23x -> 1.5-1.7x progression, the memory-traffic
numbers (exact match to the earlier measured round), the H=8-uniform
caveat, the bounding-ball dead-end (both local-window and oracle-tau
citations), the floating-point cancellation check, and the
adversarial-scoring-vulnerability scoping were all confirmed sound as
written.

**Status: ROOFLINE_5500U.md finalized. Fable-artifact-driven segment
CLOSED.** MetaMonarchAttention's design-quality thread and its cost
thread are both complete, cross-validated, and internally consistent.
Final architecture: Fenwick dyadic tiers, shared-tau threshold
selection with reservoir-tau, exact-algebraic fast-residual centroid,
no T-iteration (quality trade-off documented), no bounding-ball
pruning (confirmed dead), exact local window. Confirmed ~1.5-1.7x FLOP
speedup ceiling, decisively compute-bound at the real production GQA
config with a wide margin. Adversarial-scoring-stage vulnerability and
the no-T-iteration decoy-damping trade-off both remain filed as
accepted, cross-validated quality properties, separate from this cost
analysis.

## Fable's sign-off

Confirmed both fixes land correctly. Final summary of what "closed"
means for this arc: design settled as threshold selection (shared-tau,
reservoir estimation, exact-algebraic residual, no T-iteration, no
pruning) with a real, honestly-derived ~1.5-1.7x FLOP-level ceiling,
confirmed compute-bound at the real production GQA config with wide
margin. Every mitigation that didn't survive scrutiny -- bounding-ball
pruning, the naive 1.8x figure, memory-traffic as an independent win,
per-tier tau, floor-reads, robust centroids, static landmarks -- is
documented as tried-and-ruled-out with the specific evidence that
killed it, not quietly dropped. Crafted-adversarial scoring-stage
vulnerability correctly filed as inherited-and-accepted, cross-
validated four separate ways, not claimed fixed.

One forward-looking note (already covered by the limitations section,
not a new action item): the very first real Rust/AVX2 hardware
profiling run, whenever that phase begins, should be treated as a
CHECK on this document's central claims (compute-bound regime,
~1.5-1.7x ceiling), not an assumption they'll hold -- every number here
is an analytical estimate against public 5500U specs, never measured
on the actual chip.

**ARC CLOSED.** Fable-artifact-driven MetaMonarchAttention investigation
(Louver/threshold-selection synthesis through the roofline cost model)
is complete: 10 mechanisms tried in sequence (bucket routing -> floor-
read -> robust centroids -> threshold selection -> T-iteration removal
-> full integration -> shared-tau -> bounding-ball pruning -> roofline/
FLOP accounting -> exact-algebraic fast-residual), each tested
adversarially rather than accepted on plausibility, with two rounds of
self-caught errors (the tiling-convention bug in the memory-traffic
comparison, the H-scaling bug and missing GQA correction in the
ridge-point check, and the omitted residual-centroid cost in the FLOP
accounting) corrected before being reported rather than after.

## New phase: empirical Rust validation (monarch-attn-kernel)

Per user request, started building a standalone Rust crate
(`../monarch-attn-kernel/`, sibling to Jumping Seedling, independent
`Cargo.toml`) to empirically test Causal, Sliding, and Meta
MonarchAttention against ROOFLINE_5500U.md's analytical claims --
the "first real measurement" that document's own limitations section
called for. Installed `perf` (`pkexec pacman -S perf`) for hardware-
counter profiling.

**Scalar-correctness phase complete, all three kernels validated:**
- Causal: dense causal attention with GQA support, validated against
  real PyTorch `scaled_dot_product_attention(is_causal=True)` at the
  production config (head_dim=64, n_q_heads=14, n_kv_heads=2), max abs
  diff <1e-3.
- Sliding: faithful scalar port of `ma_sliding_monarch.py`'s T-iteration
  Sinkhorn-style cross-block refinement (read the exact reference file
  rather than reimplementing from memory, given the algorithm's
  complexity) -- validated against the real PyTorch reference, max abs
  diff 1.49e-7 (float32 rounding-noise level) on first attempt.
- Meta: faithful scalar port of
  `ma_meta_threshold_fast_residual.py` (the final confirmed design:
  Fenwick tiers, shared-tau, exact-algebraic fast-residual, sort-based
  quantile matching `torch.quantile`'s linear-interpolation default) --
  validated against the real PyTorch reference, max abs diff 1.19e-7.

Every cross-validation follows the same pattern: Python export script
(imports the actual validated reference function, not a reimplementation)
-> raw f32 binary test vectors -> Rust integration test diffing against
them. Files: `monarch-attn-kernel/src/{causal,sliding,meta}.rs`,
`export_{causal,sliding,meta}_vectors.py`, `tests/*_reference_check.rs`.

**Status: correctness foundation complete for all three kernels.**
Remaining: AVX2 versions of each (correctness-first design explicitly
chosen so SIMD work stays separable from correctness debugging), then
criterion.rs wall-clock benchmarks + `perf stat` hardware-counter
measurements (L2/L3 miss rate, instructions retired) comparing all
three against each other and against ROOFLINE_5500U.md's predictions
(compute-bound, ~1.5-1.7x FLOP-level ceiling for Meta vs. Causal).

## Bench/profiler infrastructure verified before AVX2

Per user request: verify the benchmark/profiler binaries themselves
before adding AVX2 complexity on top, rather than assuming they call
the kernels correctly.

Built `src/bin/verify.rs`: runs the same reference-vector checks as
`tests/*_reference_check.rs`, but as a `--release`-mode binary (LTO,
opt-level=3, codegen-units=1) -- `cargo test` always uses the `test`
profile, never `release`, so it alone couldn't confirm optimized
codegen preserves correctness. Ran it: **all three kernels PASS in
release mode**, same diffs as debug (causal/sliding 1.49e-7, meta
1.19e-7) -- confirms no fast-math/reordering issues from optimization
(expected, since no fast-math flags are used, but verified rather than
assumed).

Built `src/bin/profile.rs`: the actual `perf stat` / wall-clock
profiling target (deliberately not criterion -- criterion's harness
overhead and statistical resampling make it a poor target for
hardware-counter attachment; a plain fixed-iteration-count binary is
standard practice for this). Takes `<kernel> <seq_len> <iterations>`,
uses a dependency-free deterministic xorshift RNG for inputs, prints a
checksum (sum of output) both to prevent dead-code elimination and as
a sanity signal.

**Sanity-checked the profiler binary itself** across edge cases (seq_len
1, 63, 64/block-aligned, 65, 512) for all three kernels: no crashes, all
checksums finite and non-zero. Notable internal-consistency signal:
Sliding and Meta produce IDENTICAL checksums at seq_len<=64 (single
block, no far-context mechanism ever activates for either design) --
both degenerate to the same local-window-only computation when there's
nothing to route to a far mechanism, exactly as expected structurally.
They diverge starting at seq_len=65 (second block appears, far
mechanisms start differing). Also confirmed determinism: repeated runs
at the same (kernel, seq_len) produce bit-identical checksums.

**Status: bench/profiler infrastructure verified correct and
deterministic.** Proceeding to AVX2 versions next, now that the
harness they'll be measured with is itself trustworthy.

## AVX2 wired into all three kernels, correctness re-verified

Built `src/simd.rs`: `dot()` and `axpy()` primitives, AVX2/FMA with
runtime feature detection and scalar fallback, correctness-tested
against scalar reference at both AVX2-aligned and remainder-tail
lengths (1,7,8,9,15,16,17,64,100) before touching any kernel.

Wired into causal.rs, sliding.rs, meta.rs by replacing their inner
dot-product and value-accumulation loops. Re-ran the full PyTorch
cross-validation suite plus the release-mode `verify` binary after each
kernel's conversion -- all still pass, diffs unchanged at float32
rounding-noise level (1e-7 to 2e-7).

Measured real speedups via `profile` (seq_len=512): **causal
80ms->28ms (~2.8x), sliding 90ms->44ms (~2x)**. Meta only improved
modestly at first (99ms->80ms, ~1.25x) -- its actual bottleneck turned
out to be elsewhere.

**Found and fixed a real inefficiency in meta.rs while investigating
the smaller SIMD win**: the pass-2 survivor loop was storing a value
clone and a logit for EVERY key in every tier block (survivor or not),
even though non-survivors get a -inf logit contributing exactly 0
softmax weight -- an O(Bl) storage cost per tier instead of the
O(num_survivors) the FLOP accounting had already established as the
real cost. Skipping non-survivor storage entirely is mathematically
identical to the reference (removing zero-weight terms from a softmax
doesn't change the result) -- verified this is true, not assumed: full
test suite + release verify binary still pass at identical precision.
Meta improved further: 80ms->50.9ms (~1.9x total from the pre-SIMD
baseline), checksum bit-identical before/after confirming the
optimization is a pure efficiency fix, not a behavior change.

**Status: all three kernels AVX2-accelerated and re-verified correct.**
Proceeding to the criterion.rs + perf-stat benchmark harness -- the
actual empirical check on ROOFLINE_5500U.md's predictions.

## First empirical result: ROOFLINE_5500U.md's headline number does NOT survive real measurement

Built `benches/attention_bench.rs` (criterion, sample_size=10 given
seq_len=8192's O(N^2)-ish cost) and `perf_measure.sh` (perf stat,
cycles/instructions and cache-references/cache-misses in separate
runs -- combining them hit this CPU's limited PMC slots and dropped
counters, confirmed empirically; asked about disabling the NMI
watchdog to free a slot but that request was correctly denied by the
auto-mode classifier as exceeding the perf-install pkexec authorization
-- worked around it by splitting event groups instead, no system change
needed). Ran both across seq_len 512/2048/8192 (the project's actual
target range per RESEARCH_LOG.md) for all three kernels, in sequence
(not concurrently, to avoid CPU contention skewing results).

**Wall-clock (criterion):**

| N | Causal | Sliding | Meta |
|---|---|---|---|
| 512 | 28.8ms | 35.8ms | 38.4ms |
| 2048 | 460ms | 172ms | 658ms |
| 8192 | 9.11s | 1.04s | 11.69s |

**Meta is SLOWER than Causal at every N tested -- the opposite of
ROOFLINE_5500U.md's ~1.5-1.7x speedup prediction.** Sliding wins
decisively at every scale (8-9x faster than both at N=8192), despite
T-iteration being "structurally superseded" in the design-quality
thread's conclusion (a claim about quality under threshold-selection
vs. Monarch-representative reads, never about implementation
efficiency of the actual code).

**Hardware counters (perf stat) confirm this isn't a wall-clock
artifact:**

| kernel/N | instructions/iter (billions) | IPC | cache-miss rate |
|---|---|---|---|
| causal/8192 | 51.79 | 2.00 | 1.30% |
| sliding/8192 | 10.04 | 2.67 | 22.40% |
| meta/8192 | 82.11 | 1.89 | 14.57% |

Meta executes **1.59x MORE real instructions than Causal** at N=8192 --
`instructions retired` is a direct hardware count, not subject to
wall-clock noise. Directly contradicts the FLOP-based accounting, which
only counted mathematically-necessary QK/AV/residual operations, never
implementation overhead.

**Ran `perf record`/`perf report` on meta at N=8192 to find the actual
cause rather than continuing to hypothesize:** ~56% of ALL CPU cycles
are spent in `core::slice::sort::stable::quicksort` and related sort
functions. **The sort-based tau quantile -- ROOFLINE_5500U.md's own
flagged placeholder ("this is the CORRECTNESS reference; the
reservoir-sampling cost optimization... is a follow-up, not yet
implemented") -- turned out to be the single largest real-world cost in
the entire kernel.** Exactly the "plausibly negligible, actually
dominant" pattern this whole research arc has repeatedly caught
(residual-centroid cost, H-scaling bug, tiling-convention bug) --
except this time caught via real hardware profiling rather than
analytical FLOP accounting, which is precisely why Fable's own
limitations-section caveat ("treat the first real profiling run as a
check on this document's claims, not an assumption they'll hold") was
right to include.

**Status: ROOFLINE_5500U.md's headline ~1.5-1.7x speedup claim for Meta
is EMPIRICALLY FALSIFIED as currently implemented** -- the sort-based
tau is real, dominant, unaccounted-for cost, not a rounding error.
Files: `benches/attention_bench.rs`, `perf_measure.sh`,
`/tmp/meta_perf.data` (perf record capture, not checked in). Sending
this finding to Fable now (fresh transcript, per established pattern
for a result this consequential to a previously-declared-closed arc).

## Fable's read: correction not retraction, but bigger than expected -- and the real headline is Sliding

**Fable's response** (fresh transcript, grounded in the actual repo
files rather than just the summary given): two separable claims in the
roofline doc's headline have different fates. "~1.5-1.7x FLOP reduction
vs dense" is plausibly NOT falsified -- stripping the sort's ~56% cycle
share from meta/8192's 11.69s gives ~5.1s, vs causal's 9.11s, landing
almost exactly in the predicted band. What's actually wrong is
"compute-bound -> FLOPs translate to wall-clock": instructions != FLOPs,
and the roofline model never accounted for per-key branchy threshold
tests or per-query heap allocation churn. Also identified something the
JOURNAL narrative missed: the sort as written is an O(N^2 log N)
COMPLEXITY-CLASS regression (sorts the full pooled tier scores, O(N)-
sized for the largest tier, once per query), not just a constant-factor
placeholder. And: reservoir-tau would close most of the gap to CAUSAL,
but a zero-cost-tau bound still leaves Meta ~5x behind SLIDING -- a
comparison the roofline document never made at all (it only ever
benchmarked Meta against dense). Fable's recommended sequencing: a
free-tau control first (cheap, ~zero cost) to get the hard upper bound
before implementing anything, or a one-line `select_nth_unstable_by`
(quickselect, O(n)) swap as a real intermediate fix.

**Tried the free-tau control first, as recommended -- and it was
confounded, caught before trusting it.** Fixed tau to a constant
(0.0) to skip the sort entirely: `meta_freetau` came back SLOWER
(28.9s) than the real sort-based version (18.4s at that run), the
opposite of expected. Diagnosed directly: scores are roughly symmetric
around 0, so tau=0.0 makes ~50% of keys "survive" instead of the real
~10% (90th-percentile threshold) -- and since survivor storage was
already optimized to scale with num_survivors, 5x more survivors means
~5x more storage/accumulation work. The "control" wasn't isolating sort
cost at all, it was testing a completely different, heavier workload.
Discarded this approach rather than report a misleading number.

**Implemented Fable's quickselect suggestion properly instead**
(`TauMode::Quickselect` in `meta.rs`, using `select_nth_unstable_by` for
both the floor and ceil interpolation points of the linear-
interpolation quantile). This produces the mathematically IDENTICAL tau
value to sort-based tau -- same survivor rate, same output -- just via
O(n) partial reordering instead of O(n log n) full sort. **Verified
numerically identical before trusting any timing** (new test
`quickselect_tau_matches_sort_based_tau`, max abs diff <1e-5) and
release-mode `verify` binary still passes at unchanged precision for
all three kernels.

**Clean, apples-to-apples timing comparison at N=8192** (all four from
the same `profile` binary, single-iteration, immediately sequential to
minimize system-state drift):

| | Causal | Sliding | Meta (sort) | Meta (quickselect) |
|---|---|---|---|---|
| time | 10.9s | 1.42s | 16.5s | ~8.9s |
| vs Causal | -- | 7.7x faster | 1.51x slower | **1.22x faster** |
| vs Sliding | -- | -- | -- | still **6.3x slower** |

Checksums identical between sort-based and quickselect meta (confirms
correctness -- same tau, same survivors, same output, only the
algorithm computing tau changed). **Confirms Fable's prediction almost
exactly**: quickselect took Meta from losing to Causal to beating it by
~1.2x (short of the original naive 1.8x/1.5-1.7x prediction -- real
non-sort overhead remains, exactly as Fable flagged), and did
essentially nothing to close the gap to Sliding, which stays at ~6.3x.
Files: `src/meta.rs` (TauMode enum, quickselect implementation),
`tests/meta_reference_check.rs` (quickselect correctness test),
`src/bin/profile.rs` / `benches/attention_bench.rs` (meta_quickselect
variant).

**Status: quickselect fix implemented, verified correct, and measured.**
Meta now legitimately beats Causal on FLOPs-that-matter, closing the
gap the roofline document originally claimed (in a different, more
honest way than the document's own analytical argument). The gap to
Sliding remains wide and is NOT a tau-cost problem -- per Fable's
structural read, Sliding's Monarch-representative reads never score
every key (avoiding threshold selection's O(N)-per-query QK scoring
entirely), which is a genuine complexity-class advantage threshold
selection gave up when it replaced representative-reads with real-score
reads (the original design-quality tradeoff, now shown to also be a
real cost tradeoff, not just a quality one). Reporting full results
back to Fable now.

## Fable pushes back: reservoir-tau confirmed dead, but "recenter on Sliding" not yet earned

**Fable's read**: agrees the reservoir-tau question is closed (the 6.3x
gap to Sliding originates at QK scoring, a stage no tau estimator
touches). Also identifies the "parked hybrid" I'd referenced was
Meta-side (global-aggregation on top of Meta, to recover T-iteration's
decoy-damping) -- since it only ADDS cost to Meta's already-capped
stage, it's foreclosed for the same structural reason, not worth
parking further.

**The real pushback**: Sliding's founding correctness claim (JOURNAL
line 364, "first build, correctness-only, strong result") predates the
SAME-NORM CONTROL this session later introduced (`eval_bucket_
multineedle.py`'s `BACKGROUND_NORM` fix) -- a control that subsequently
unmasked a magnitude-scaling artifact silently inflating "passes" for
FIVE separate untrained representative-construction mechanics tried for
Meta's R-landmark generalization (all five failed single-needle
detection outright once needle-key norm was forced equal to background
norm). Sliding, using a single fixed representative (R=1, the TIGHTEST
version of the arc's own established capacity ceiling), was never
re-checked under this corrected methodology. Recommended: rerun a
same-norm-controlled single/multi-needle probe against Sliding
specifically before treating its cost win as a production decision.

**Ran it** (`eval_sliding_multineedle_samenorm.py`): needle(s) placed in
a FAR block (outside the local exact window, so only the T-iteration-
refined Monarch representative can carry the signal), same-norm control
applied to needle keys (`BACKGROUND_NORM = 0.5*sqrt(D)`, matching
background key norm -- no magnitude shortcut available).

| K needles | mean cos | min cos | frac>0.5 |
|---|---|---|---|
| 1 | -0.2152 | -0.2152 | 0.00 |
| 2 | 0.4188 | 0.2132 | 0.50 |
| 4 | 0.2823 | 0.1717 | 0.00 |
| 8 | 0.1370 | -0.1109 | 0.00 |

**K=1 (single needle, ZERO competition) fails outright -- mean cos
negative.** Not a capacity-ceiling degradation pattern (that would show
K=1 passing cleanly, then degrading as K grows) -- a complete detection
failure with no competition at all.

**Sanity-checked before trusting this** (`eval_sliding_multineedle_
sanity_check.py`): identical far-block placement, window boundary, and
T-iteration wiring, but WITHOUT the same-norm control (needle key at
large magnitude, matching the style used throughout most of this
session's earlier SlidingMonarchAttention work). Result: K=1 passes
cleanly (cos=0.548), K=2/K=4 hold up reasonably, K=8 degrades (the
expected capacity-ceiling pattern). **Confirms the probe mechanics are
correct and the same-norm version's K=1 failure is real, specifically
caused by removing the magnitude shortcut -- not a bug in the adapted
script.**

**Status: Sliding's founding correctness claim does not survive the
same-norm control.** Its T-iteration-refined block representative
relies on a magnitude shortcut to "find" a needle -- exactly the same
artifact that killed all five untrained landmark mechanics tried for
Meta. Under a fair test (needle indistinguishable from background by
magnitude), Sliding cannot do single-needle retrieval from its far
region at all. This disqualifies Sliding as a "leading production
candidate" on the strength of its cost advantage alone -- the 8x cost
win is moot if basic retrieval fails under realistic key-norm
conditions. Files: `eval_sliding_multineedle_samenorm.py`,
`eval_sliding_multineedle_sanity_check.py`. Sending this to Fable now.

## Fable catches a real methodological gap: the K-sweep was single-trial, not statistically established

**Fable's response**: first verified the same-norm control itself was
correctly calibrated (BACKGROUND_NORM=2.0 at D=16 matches background
keys' actual expected norm -- that part is solid). The real problem:
`make_scene(seed=1, K=K)` used ONE seed per K value -- K=1's -0.2152 was
a single trial, not an average, and K values weren't even independently
comparable (directions/values drawn as separate list comprehensions, so
the RNG stream diverges after the first draw -- K=1's needle isn't the
same draw as K=2's first needle despite sharing a seed). The apparent
K=1->K=2 non-monotonicity could be single-draw noise, not a real
mechanism crossover. Recommended: rerun with 20-50 independent seeds
per K, report mean/min/CI, before hand-tracing anything.

**Ran it** (`eval_sliding_multineedle_samenorm_multiseed.py`, 30
independent seeds per K, all recalls per scene included):

| K | n | mean cos | +-1 SE | min | max | frac>0.5 |
|---|---|---|---|---|---|---|
| 1 | 30 | 0.2150 | 0.0414 | -0.44 | 0.71 | 7% |
| 2 | 60 | 0.1886 | 0.0314 | -0.34 | 0.65 | 10% |
| 4 | 120 | 0.1814 | 0.0233 | -0.48 | 0.68 | 10% |
| 8 | 240 | 0.2023 | 0.0162 | -0.52 | 0.84 | 12% |

**Materially different, less alarming finding than the single-trial
run suggested.** Mean cos is POSITIVE across all K (~0.19-0.22, tight
SE) -- the original -0.2152 was an unlucky single draw, exactly the
"wide variance straddling zero" alternative Fable flagged as the
less-alarming outcome. This is NOT systematic detection failure or
anti-alignment. But it's also not good: mean cos ~0.2 is weak signal,
and only 7-12% of trials clear the cos>0.5 "good recall" bar, roughly
FLAT regardless of competition level (K=1 through K=8 all land in the
same mediocre range -- no clear capacity-ceiling degradation pattern
either). Files: `eval_sliding_multineedle_samenorm_multiseed.py`.

**Status: corrected finding -- Sliding's far-region retrieval under
same-norm control is chronically WEAK and UNRELIABLE (not a clean
detection failure), a different and less dramatic problem than
"collapse."** Still a real correctness concern (7-12% good-recall rate
is far short of what a production attention mechanism needs), but the
"systematic anti-alignment" framing from the single-trial result does
not hold up. Sending this corrected picture to Fable.

## Fable: don't conclude anything without a GT/Meta baseline on the same scenes -- and it's decisive

**Fable's response**: cited the arc's own established practice from
Sliding's ORIGINAL validation (JOURNAL line 412-419: "even ground truth
degrades hard [at low signal]... rather than diverging negative") -- a
weak absolute cosine number is uninterpretable without knowing the
achievable ceiling at that signal strength. Two very different stories
are consistent with "~0.2 mean cos": (a) Sliding leaves real signal on
the table (GT scores much higher -> genuine defect) or (b) this exact
scene is intrinsically hard even for exact attention (GT also lands
near ~0.2 -> a harness finding, not a Sliding finding). Recommended
running GT (exact dense attention) and Meta (threshold selection) on
the IDENTICAL seeds/scenes before drawing any conclusion, and before
the hand-trace.

**Ran it** (`eval_sliding_multineedle_samenorm_gt_meta.py`, same
generator/seeds/needle-placement/norm-control, all three mechanisms on
every scene):

| K | mech | n | mean cos | +-1 SE | min | max | frac>0.5 |
|---|---|---|---|---|---|---|---|
| 1 | GT | 30 | 0.8686 | 0.0082 | 0.7705 | 0.9458 | 1.00 |
| 1 | Meta | 30 | 0.9207 | 0.0057 | 0.8367 | 0.9820 | 1.00 |
| 1 | Sliding | 30 | 0.2150 | 0.0414 | -0.4383 | 0.7050 | 0.07 |
| 2 | GT | 60 | 0.8619 | -- | -- | -- | 1.00 |
| 2 | Meta | 60 | 0.9122 | -- | -- | -- | 1.00 |
| 2 | Sliding | 60 | 0.1886 | -- | -- | -- | 0.10 |
| 4 | GT | 120 | 0.8588 | -- | -- | -- | 1.00 |
| 4 | Meta | 120 | 0.9119 | -- | -- | -- | 1.00 |
| 4 | Sliding | 120 | 0.1814 | -- | -- | -- | 0.10 |
| 8 | GT | 240 | 0.8592 | -- | -- | -- | 1.00 |
| 8 | Meta | 240 | 0.9126 | -- | -- | -- | 1.00 |
| 8 | Sliding | 240 | 0.2023 | -- | -- | -- | 0.12 |

**Decisive: GT retrieves the needle cleanly and reliably at every K
(mean ~0.86-0.87, 100% frac>0.5) -- the scene is NOT intrinsically
hard.** Meta matches or slightly EXCEEDS GT (~0.91-0.92, also 100%
frac>0.5). Sliding sits at ~0.18-0.22 with only 7-12% frac>0.5 -- a
massive, reproducible gap against both baselines on the exact same
scenes, not an artifact of scene difficulty. Files:
`eval_sliding_multineedle_samenorm_gt_meta.py`.

**Status: Sliding's disqualification is now confirmed on solid
empirical footing** (not a single-trial artifact, not an ambiguous
"maybe the scene is hard" explanation) -- its T-iteration-refined
representative genuinely loses real, retrievable signal that both
dense attention and threshold selection successfully capture. This ALSO
empirically confirms the structural hypothesis about Meta: real-per-
key-score selection does NOT inherit Sliding's weakness, matching or
exceeding dense attention's own retrieval quality on the identical
adversarial-norm scenes.

## Closing polish: two cheap sweeps confirm the mechanism precisely

**Fable's response**: production call doesn't change (Meta ships
either way), but flagged one specific, cheap, testable hypothesis worth
15 minutes before fully closing the entry: CROSS-BLOCK dilution, not
just within-block. The far region isn't one block's representative vs.
the window -- EVERY Monarch block strictly before the window
contributes its own representative, all combined in ONE joint softmax
(~31 competitors at QUERY_POS=256, B=8). A needle's signal gets diluted
TWICE: once within its own block (Sinkhorn-averaged with background
keys, pre-score), then again competing against every other purely-
background far-block representative (post-score, already-diluted).
Directly echoes Meta's own earlier cross-tier-accumulation finding, but
Sliding is far more exposed since it dilutes PRE-score, not post-score.
Proposed two sweeps: far-region-length (tests dilution-by-competition
directly) and T-iteration budget (distinguishes "needs more compute"
from "structural ceiling regardless of budget" -- matters given how
much of this arc's effort went into T-iteration specifically).

**Ran both** (`eval_sliding_dilution_sweeps.py`, K=1, 30 seeds/config):

Sweep 1 (far-region length, T=3 fixed):
| QUERY_POS | n_far_blocks | mean cos | frac>0.5 |
|---|---|---|---|
| 80 | 2 | 0.4307 | 0.40 |
| 96 | 4 | 0.4126 | 0.43 |
| 128 | 8 | 0.3702 | 0.27 |
| 192 | 16 | 0.2958 | 0.10 |
| 256 | 24 | 0.2661 | 0.10 |
| 384 | 40 | 0.2242 | 0.07 |
| 480 | 52 | 0.1872 | 0.10 |

Clean, monotonic degradation as far-region grows -- confirms cross-
block dilution as the dominant mechanism.

Sweep 2 (T-iteration budget, QUERY_POS=256 fixed):
| T | mean cos | frac>0.5 |
|---|---|---|
| 3 | 0.2661 | 0.10 |
| 5 | 0.2661 | 0.10 |
| 10 | 0.2661 | 0.10 |
| 20 | 0.2661 | 0.10 |

**Perfectly flat -- identical to four decimal places at every T.** Zero
effect from additional refinement iterations. Confirms a STRUCTURAL
ceiling, not an iteration-budget problem: no amount of T-iteration
compute would ever have fixed Sliding at this configuration. Files:
`eval_sliding_dilution_sweeps.py`.

**Status: mechanism fully understood and confirmed.** Sliding's
disqualification is precise, not just empirical: R=1 block-
representative collapse, pre-score, competing against every other far
block in one joint softmax, is a structural ceiling independent of
refinement budget. The closing statement is "the R=1 block-
representative design class is disqualified regardless of budget," not
merely "Sliding as configured underperforms." Sending final results to
Fable, plus a new question: is there an alternative way to build a
"sliding" (window + far-region) Monarch-family mechanism that avoids
this specific pre-score collapse-then-compete structure?

## Fable's final answer: salvageable in principle, but the fix converges back to Meta -- category closed for production

**Why the flat T-sweep is earned, not just observed**: T-iteration only
ever refines representatives AGAINST OTHER REPRESENTATIVES under a
block-triangular mask -- it never reaches back to raw per-key K/V
content. Once a block's content is collapsed to one vector at
construction time, no amount of representative-to-representative
message-passing can recover information never carried into that vector
in the first place. The flat-to-four-decimals result is exactly what
this architecture predicts, not a surprising empirical fact needing its
own explanation -- confirmation that the loss happens at CONSTRUCTION,
and iteration downstream of construction is structurally incapable of
fixing it. This is the precise justification for "regardless of
refinement budget," now earned by the sweep rather than asserted.

**Two separable defects were conflated in "Sliding fails"**:
1. **Content-blind (pre-score) construction** -- the representative is
   built without knowing what the query is looking for, so it's
   whatever T-iteration's Sinkhorn dynamics converge to over the
   block's random content, uncorrelated with any particular query's
   needle. FIXABLE: build the representative AFTER computing real
   per-key scores against the actual query (the scoring step Meta
   already does) -- e.g. hard top-1 or a real-score-weighted average.
2. **Collapse to exactly one candidate per block, however constructed**
   -- the generic R=1 capacity ceiling already derived earlier in this
   arc (line 871: "more than R genuinely-distinct simultaneous needles
   in one block WILL collide"). Separate from defect 1, and NOT fixable
   by better representative construction.

**A genuinely new observation**: the K-sweep (K=1..8 needles in the
SAME far block) never actually measured defect 2 in isolation -- it
was flat at ~0.2 across all K because defect 1 was already flooring
performance before defect 2 ever became the binding constraint. A
real-score-fixed variant would, for the first time in this arc, let the
true R=1 capacity ceiling be observed cleanly (expected: K=1 jumps to
~GT/Meta level, K=2/4/8 show genuine monotonic degradation this time).
Flagged as a real, cheap, well-motivated experiment -- but explicitly
OPTIONAL, non-blocking, documentation-value only.

**Does the fix rescue Sliding as a genuine alternative? No.** Fixing
defect 1 properly just converges back to Meta's own construction
principle (score first, decide what matters using real content, only
THEN compress) applied one level coarser (collapse survivors to one
representative per block instead of keeping them all, as Meta does).
There's no way to fix defect 1 without reintroducing Meta's own
machinery, and no way to fix defect 2 without abandoning the collapse
-- which is Sliding's entire reason for existing. The fixed variant
would be a strictly WORSE version of Meta: no path to beating Meta on
quality (Meta has zero collapse-loss for survivors by design) and no
cost advantage either (any real-score-gating step it needs is the same
QK-scoring cost Meta already pays).

**Status: "sliding window + far-region compression" is CLOSED as a
production category, unconditionally.** Meta (shared-tau threshold
selection, quickselect tau, exact-algebraic fast-residual, no
T-iteration) remains the confirmed production recommendation on both
correctness (matches/exceeds dense attention's own quality, including
on the exact adversarial-norm scenes that break Sliding) and cost
(honest ~1.22x wall-clock win over dense, hardware-measured on the real
5500U). The real-score-fixed-representative variant is noted as an
optional future experiment for isolating the R=1 capacity ceiling
analytically derived at line 871 but never yet directly observed --
standalone documentation value only, not gating, not a production path.
This closes the Sliding-vs-Meta investigation.

## Design-space closure check: is any other variant possible?

Asked Fable directly whether the whole design space (not just Sliding)
has been exhausted, given every mechanism ruled out in this arc shares
one thing: untrained representative-construction heuristics.

**R>1 real-score-gated representatives**: provably dominated by Meta,
not just similar. Building even one query-aware representative requires
real per-key QK scoring for the whole block first -- the expensive
stage Meta already fully pays (ROOFLINE_5500U.md line 35). Collapsing
scored candidates down to R representatives only touches the AV/softmax
stage, which is already the CHEAP stage (~0.10-0.18x of dense, line
36). Saves nothing on the bottleneck, adds risk on the stage that's
already cheap.

**Multi-scale block sizes with real scoring at every tier**: this isn't
a variant to test, it IS Meta's own definition (Fenwick/binary-
decomposition tiers + real per-key threshold selection at each tier,
already built and shipping). No daylight to explore.

**Cheap-prefilter-then-exact-attention**: same failure family one level
up. Any signature cheap enough to avoid full per-key scoring is some
kind of block summary (mean centroid, robust centroid, magnitude,
FPS-style spread) -- every one tried independently in this arc diluted
a strong outlier into a coarse aggregate and lost it before the query
got a real look. Block-mean prefilter = bucket routing with a different
backend (dead). Bounding-ball prefiltering = another cheap-signature
family (dead, confirmed 0% prune rate). One genuinely UNTESTED specific
mechanism: LSH-style hashing (per-key randomized-hyperplane sign, not
block-pooling) -- technically distinct, never tried, but Fable predicts
(lower confidence, genuinely untested) it fails the same same-norm
control since it's still a proxy computed without seeing real
query-key relevance -- structurally the same exclusion-cliff risk
threshold selection was invented to eliminate at the very start of this
arc. Burden of proof is now on any cheap-proxy scheme, given 4-5
independent proxy families have all failed the identical control.

**The one axis that's genuinely still open**: every mechanism ruled out
used an UNTRAINED construction heuristic (k-means, magnitude, FPS,
maxpool, random-reuse, Sinkhorn-refined single representative). This was
a deliberate methodology choice for fast standalone iteration, flagged
explicitly at the time (line 930-937: "a real trained model would have
LEARNED landmark/pseudo-query parameters... this probe cannot speak to
that version at all"). A TRAINED summarizer (gradient-optimized to
produce a useful block representative, not reusing random content or an
unweighted statistic) has never been tested here. Literature precedent
noted in passing (line 1947, Landmark Attention) is that real published
efficient-attention systems using this general shape of idea use
trained landmarks, not untrained ones -- suggestive, not proof.
"Compress before scoring" is NOT provably doomed as a category -- only
the untrained-construction version actually tested here is.

**Practical read: don't chase it.** Validating a trained-representative
variant means an entirely different, much more expensive research arc
(trainable summarizer module, real gradient training, generalization
validation across real training data) -- disproportionate scope for a
from-scratch single-developer CPU-target 1B model that already has a
fully validated, real-hardware-measured, honest production candidate on
the table. File the trained-representative axis explicitly as
"genuinely open, deliberately unexplored by this arc's methodology,"
NOT "ruled out" -- but it should not gate or delay shipping Meta. If
ever worth revisiting, it's a training-loop co-design question for a
later phase, not a next step in this standalone-probe arc.

**Status: design-space exploration for standalone (untrained,
probe-testable) MonarchAttention variants is COMPLETE.** Meta ships as
the production recommendation. The only unexplored axis (trained
representative construction) is out of scope for this arc by
deliberate methodology, not oversight, and is documented as future
work rather than a blocker. This closes the entire Fable-artifact-
driven MetaMonarchAttention investigation, from the original Louver/
threshold-selection synthesis through the Rust/AVX2 empirical
validation phase.

**User's call**: pursue the two flagged-as-open items (LSH-style
per-key hashing prefilter, genuinely untested; trained representative
construction, deliberately out of scope for this arc) as a follow-up
phase AFTER Meta ships, not before. Consistent with Fable's own
recommendation not to let either gate or delay the production
recommendation already on the table.

## Cleanup pass: promote quickselect to the crate's "meta" default, revise ROOFLINE_5500U.md

Two loose ends flagged when asked "what's left": (1) `monarch-attn-kernel`'s
"meta" entry point (profiler binary, verify binary, criterion bench) still
defaulted to `TauMode::SortBased` -- the slower reference mode -- while the
actual production-recommended Quickselect variant only existed as a separate
`meta_quickselect` entry. (2) `ROOFLINE_5500U.md` still reflected the
pre-quickselect-fix, pre-Sliding-disqualification picture: the stale ~1.5-1.7x
headline, no mention of the sort-cost discovery, no mention of Sliding at all.

**Fixed (1)**: swapped `profile.rs`/`benches/attention_bench.rs`'s "meta"
entries to `TauMode::Quickselect`, renamed the reference-only sort-based
entry to `meta_sortref` (kept for direct before/after comparison, not a
recommendation). `verify.rs`'s correctness check deliberately LEFT on
SortBased, since its job is comparing against the PyTorch ground truth
reference specifically (which itself is sort-based) -- a different purpose
from the production/profiling entries. Rebuilt, reran full test suite +
release verify binary: all still pass, checksums identical between `meta`
and `meta_sortref` confirming the rename didn't change behavior.

**Fixed (2)**: substantial revision to `ROOFLINE_5500U.md`. Added a "Real
hardware validation" section up front (before the analytical section) with
the actual measured numbers: Meta initially falsified at 11.69s vs Causal's
9.11s, the perf-record sort-cost discovery (~56% of cycles), the quickselect
fix bringing it to a real ~1.22x win, and a full SlidingMonarchAttention
subsection (7.7x faster than Meta, then disqualified on the same-norm needle
probe, cross-block-dilution mechanism, structural-ceiling confirmation via
the flat T-sweep). Relabeled the original analytical section as "superseded
headline, kept for its derivation" rather than removing it -- it's still a
correct and informative FLOP-counting result, just not the final wall-clock
number. Corrected the τ-estimation table row, which had asserted reservoir
sampling without ever building or measuring it (the actual gap that caused
the whole falsification). Updated "Known limitations" to remove the
now-stale "not measured on real hardware" caveat (most of the document now
IS real hardware measurement) and added the design-space-closure and
trained-representative-axis findings as documented future work. Fixed a
typo (`n_kv_keads` -> `n_kv_heads`).

**Status: both cleanup items complete.** `monarch-attn-kernel`'s "meta"
now means what it should (the actual production recommendation), and
`ROOFLINE_5500U.md` is internally consistent with the full validation
history in this JOURNAL rather than frozen at the pre-hardware-check
snapshot. This is the true final closing state of the Fable-artifact-driven
MetaMonarchAttention investigation.

## The stash, item 1: LSH-style per-key hashing prefilter -- tested, dead

Per user direction, picking up the deferred design-space items (LSH-style
hashing, trained representative construction) as their own follow-up work,
independent of Jumping Seedling's model design (which is complete).

**First pass** (`eval_lsh_prefilter_samenorm.py`): needle at a fixed
position, query built maximally aligned with the needle's exact direction
(same construction as every other same-norm probe in this arc), SimHash-
style bucketing (sign of dot product against 8 random hyperplanes),
Hamming-distance-gated survivor filter feeding real softmax over survivors.
Result: 100% needle survival, mean cos 0.96-1.0 at every Hamming threshold
tested -- APPEARED to contradict Fable's prediction that LSH would fail
the same-norm control like every other cheap-proxy family.

**Recognized this was the easiest possible case, not a fair test** (per
this arc's own established discipline: bucket routing passed its natural
case at 96.4% before failing 83% under adversarial construction -- a
favorable natural-case result alone proves nothing). Query-maximally-
aligned-with-needle is exactly the case LSH is BUILT to handle well
(angular-similarity preservation). Built two harder tests before drawing
any conclusion (`eval_lsh_prefilter_adversarial.py`):

**Test 1 (boundary case)**: needle direction perturbed off the query
direction by increasing amounts (0.0 to 0.5), simulating realistic
imperfect correlation instead of artificial perfect alignment.

| perturb | needle_survival | mean cos | frac>0.5 |
|---|---|---|---|
| 0.0 | 100.00% | 0.9844 | 1.00 |
| 0.1 | 73.33% | 0.7332 | 0.73 |
| 0.2 | 40.00% | 0.4102 | 0.40 |
| 0.3 | 26.67% | 0.2575 | 0.27 |
| 0.5 | 20.00% | 0.2094 | 0.20 |

**A real, sharp exclusion cliff** -- survival rate collapses as soon as
the needle isn't perfectly aligned with the query, tracking mean cosine
almost exactly. This is the same failure mode Louver-style threshold
selection was specifically invented to eliminate: LSH is a PROBABILISTIC
proxy (can silently exclude a real needle before scoring), not a
guaranteed-inclusion test.

**Test 2 (adversarial decoy pressure)**: many same-hash-bucket decoys
competing in the post-filter softmax.

| n_decoys | needle_survival | mean cos | frac>0.5 |
|---|---|---|---|
| 0 | 100.00% | 0.9838 | 1.00 |
| 5 | 100.00% | 0.7806 | 1.00 |
| 20 | 100.00% | 0.5623 | 0.67 |
| 50 | 100.00% | 0.4452 | 0.43 |

Needle survives the FILTER 100% of the time even under heavy decoy
pressure, but still loses the downstream softmax competition -- this is
the ALREADY-KNOWN, already-accepted scoring-stage vulnerability (matches
dense attention's own inherited ~90% adversarial ceiling), not a new
LSH-specific problem. Test 1 is the decisive result; Test 2 just
reconfirms the pre-existing, inherited-and-accepted finding.

**Status: LSH-style per-key hashing prefilter CONFIRMED DEAD**, via a
real exclusion cliff (Test 1), consistent with every other untrained
cheap-proxy family already ruled out in this arc (k-means centroids,
robust centroids, magnitude, FPS, bounding balls). Fable's prediction
holds -- the natural-case-only result was misleading exactly as this
arc's own discipline would predict it might be; the harder boundary test
was necessary to see the real failure. Files:
`eval_lsh_prefilter_samenorm.py`, `eval_lsh_prefilter_adversarial.py`.
This closes item 1 of the stash. Item 2 (trained representative
construction) remains genuinely open, out of scope for standalone-probe
methodology per Fable's read -- a training-loop co-design question, not
a next step here.
