# MetaMonarchAttention (threshold selection) — cost model and hardware validation for the 5500U

Item 2 of the Fable-artifact consultation arc, now closed out with real
hardware measurement (see `../monarch-attn-kernel/`, a standalone Rust crate
built specifically to check the claims below). The original version of this
document was analytical only (PyTorch timing was correctness scaffolding, not
a cost verdict, until a real Rust/AVX2 kernel existed). **That kernel now
exists, and the first real measurement corrected the headline number** — see
"Real hardware validation" below before trusting the analytical section that
follows it, which is kept for its derivation but is no longer the final word.

## Confirmed final design

- Fenwick dyadic tier selection (causal prefix covered by O(log N) blocks per query)
- Real threshold selection per tier: shared τ pooled across all active tiers
  (`ma_meta_threshold_shared_tau.py`), not independent per-tier quantiles
- τ computed via **quickselect** (`select_nth_unstable_by`, O(n) average) —
  mathematically identical value to the sort-based reference (verified,
  `quickselect_tau_matches_sort_based_tau`, max abs diff <1e-5), just a
  cheaper algorithm. Reservoir sampling was analyzed as an alternative
  O(1)-amortized estimator and never built once quickselect's real
  hardware measurement (below) showed it already captured the realistic
  ceiling of tau-cost optimization.
- Exact residual centroid for non-survivors, computed via the exact algebraic
  identity `sum(non-survivors) = sum(all keys in block) − sum(survivors)`
  (`ma_meta_threshold_fast_residual.py`) rather than a full masked reduction —
  verified numerically identical to the reliability-validated reference
  (max abs diff ~3-6e-8, float32 rounding-noise level)
- No T-iteration / Sinkhorn refinement (structurally superseded once every tier
  reads real keys instead of a Monarch representative) — **also empirically
  confirmed disqualified as a design alternative**, see "SlidingMonarchAttention"
  below
- No bounding-ball pruning (tried, confirmed dead — see below)
- Exact local sliding window

## Real hardware validation (the actual empirical result)

Built and measured all three MonarchAttention variants (Causal, Sliding, Meta)
as scalar-then-AVX2 Rust kernels, cross-validated numerically against the real
PyTorch reference implementations (diffs at float32 rounding-noise level, 1e-7
to 1e-8) before trusting any timing. Full detail in `JOURNAL.md`; summary here.

**First measurement (criterion + `perf stat`, N=8192, real production GQA
config) falsified the analytical ~1.5–1.7x headline.** Meta was measured
*slower* than Causal (11.69s vs 9.11s), and `perf record` found ~56% of all
CPU cycles were spent in `core::slice::sort::stable::quicksort` — the
sort-based τ this document's earlier draft called "not O(n log n) sort," when
that was in fact exactly what the implementation did. The analytical FLOP
model never accounted for τ-estimation cost, non-survivor storage overhead,
or per-query allocation churn — instructions retired ≠ FLOPs, and the roofline
argument implicitly assumed parity between the two that doesn't hold.

**Fixed the τ-estimation cost with quickselect** (see "Confirmed final
design" above), verified mathematically identical to the sort-based reference
before trusting any new timing. Re-measured, apples-to-apples, same tool,
same run, N=8192:

| | Causal | Sliding | Meta (sort) | Meta (quickselect) |
|---|---|---|---|---|
| time | 10.9s | 1.42s | 16.5s | ~8.9s |
| vs Causal | — | 7.7x faster | 1.51x slower | **1.22x faster** |

**Honest headline: Meta is ~1.22x faster than dense causal attention,
hardware-measured on the actual 5500U — not the ~1.5–1.7x analytical
estimate below**, which undercounted real implementation overhead. This is
a smaller number, earned through actual measurement rather than FLOP
accounting alone, consistent with this whole arc's discipline of not trusting
a plausible-sounding claim until it's checked.

### SlidingMonarchAttention: tried, faster, then disqualified on correctness

Sliding (Monarch block representative + T-iteration Sinkhorn refinement, no
threshold selection) measured **7.7x faster than Meta** at N=8192 (1.42s vs
~8.9s) — dramatically faster than either Causal or Meta, and briefly considered
the leading production candidate on cost alone. It does not ship, because it
fails correctness under a same-norm-controlled needle probe (same methodology
that earlier unmasked a magnitude-scaling artifact in five separate untrained
representative-construction mechanics tried for Meta's R-landmark
generalization): on identical adversarial-norm scenes, dense attention (GT)
and Meta both retrieve a single needle cleanly and reliably (mean cosine
~0.86–0.92, 100% success rate), while Sliding manages only ~0.2 mean cosine,
7–12% success rate. Two follow-up sweeps confirmed the mechanism precisely:
degradation is driven by cross-block dilution (a needle's signal is collapsed
into one representative *before* any real scoring happens, then that
already-diluted representative competes against every other far-block
representative in one joint softmax — a needle's signal gets diluted twice,
neither dilution stage paid by Meta), and the effect is a **structural
ceiling, not a refinement-budget problem** — T=3, 5, 10, and 20 iterations
produced identical results to four decimal places. No amount of additional
T-iteration compute would have fixed it. Full derivation, sweep data, and the
broader design-space closure check (is any untested variant salvageable —
answer: only a fundamentally different, out-of-scope trained-representative
approach, not any untrained heuristic) in `JOURNAL.md`.

## Analytical FLOP accounting (superseded headline, kept for its derivation)

The following section is the ORIGINAL analytical argument for a ~1.5–1.7x
speedup ceiling. It is a genuine, verified FLOP-counting result — but "FLOPs
reduced" turned out not to equal "wall-clock reduced" once real implementation
overhead (τ-estimation algorithm, storage, allocation) was measured on real
hardware, as detailed above. Kept below because the stage-by-stage FLOP
breakdown and the roofline/compute-bound placement are still correct and
informative on their own terms — they explain why the *quickselect-fixed*
version's real ~1.22x gain is realizable in wall-clock time at all (compute-
bound, not memory-bound), even though the original ~1.5–1.7x number itself
did not survive contact with a real implementation.

Stage-separated FLOP accounting, verified empirically against real runs of the
final design (not assumed from the quantile's nominal 90th-percentile rate):

| Stage | Ratio vs. dense | Note |
|---|---|---|
| QK^T scoring | 1.0x (zero savings) | Every key in every active Fenwick tier gets a real dot product — this is a structural property of threshold selection needing real scores to apply τ, not an implementation gap. Same complexity class as dense (O(N) per query, O(N²) total). |
| AV + softmax | ~0.10–0.18x | Only ~10% survivors + one residual slot per tier get value-weighted. Total survivors still scale ~O(N) (dominated by the largest active tier), so this is a constant-factor discount (~5–10x), not a complexity-class win. |
| Residual-centroid computation | Real, non-trivial if done naively | A masked reduction over ~90% of a tier's keys costs the same order as that tier's own QK matmul — omitting this cost was the single largest error caught in this arc (see below). Fixed via an exact algebraic identity that reduces it to ~O(num_survivors·D) marginal cost plus a one-time O(Bl·D) per-block sum, amortized to ~0 over the many queries that reuse it. |
| τ estimation | O(reservoir_size) per tier per query | **This row was the original document's mistake**: assumed reservoir sampling without building or measuring it. The implementation actually used sort-based quantile (O(n log n)) at this point, and that gap was never checked until real hardware profiling found it accounted for ~56% of total cycles — see "Real hardware validation" above. Corrected via quickselect (O(n) average, mathematically identical tau value), not reservoir sampling. |

**Blended ratio (all stages, both fixes applied): ~0.60–0.65 at N=256–8192,
converging toward ~0.60 as N grows** — i.e. a **~1.5–1.7x** realizable FLOP
reduction versus dense causal attention. Not the naive ~1.8x first estimated
(the AV-stage discount read in isolation, before the residual-centroid cost was
counted) — that number silently omitted a real cost that materially eroded it
to ~1.05–1.23x (no savings, or worse than dense) before the exact-subtraction
fix recovered most of it.

## Roofline placement: decisively compute-bound, wide margin

5500U specs (Zen 2 "Lucienne", 6 cores, AVX2 256-bit, no AVX-512, dual-channel
DDR4-3200 or quad-channel LPDDR4-4266, 8MB shared L3) sourced from public specs,
not measured on the actual chip — flagged as an estimate pending real profiling.

- Ridge point: ~4.6 FLOPs/byte (conservative-efficiency estimate) to ~7.9
  FLOPs/byte (theoretical peak)
- Implied arithmetic intensity of the final design, at the **real production
  config** (`src/model/config.rs`: `head_dim=64`, `n_q_heads=14`,
  `n_kv_heads=2` GQA, `kv_block=64`), corrected FLOP figures:

| N | AI, dense (tiled) | AI, threshold selection | Regime |
|---|---|---|---|
| 512 | 199.5 | 112.6 | both compute-bound |
| 1024 | 211.0 | 119.1 | both compute-bound |
| 2048 | 217.3 | 124.5 | both compute-bound |
| 4096 | 220.6 | 128.4 | both compute-bound |
| 8192 | 222.3 | 130.9 | both compute-bound |

Both mechanisms are comfortably compute-bound at every context length this
project actually targets (`RESEARCH_LOG.md`'s full-attention benchmarks use
512/2048/8192 — full-attention layers are ~92% of decode cost at ctx=8192), but
the margins differ: threshold selection lands **~15–28x above** the realistic
ridge point (4.62 FLOPs/byte), while dense — doing more total work, unreduced
by the AV-stage discount — lands even further above it, **~25–48x**. Neither
figure depends on optimistic assumptions about kernel efficiency; even a
hand-written AVX2 kernel running at a fraction of theoretical throughput would
need to be off by well over an order of magnitude to flip either regime.
Threshold selection is the tighter of the two margins, but still wide.

**Consequence, as originally argued: the FLOP reduction should be realizable
in wall-clock time, because the design is compute-bound, not memory-bound**
(the model's `n_kv_heads=2` vs. `n_q_heads=14` keeps K/V memory traffic small
relative to compute). This compute-bound placement held up under real
measurement — it's *why* the quickselect-fixed version's real ~1.22x gain
shows up in wall-clock time at all. What did NOT hold up was the specific
~1.5–1.7x magnitude, which undercounted real per-query overhead the FLOP
model never captured (see "Real hardware validation" above).

## Memory-traffic axis (secondary, does not change the verdict)

Measured (not estimated) at production scale (`head_dim=64`, `H=8` uniform-head
approximation for this specific table — see GQA correction above for the
regime-determining numbers):

| N | K+V footprint | Fits L3 (8MB)? | Naive DRAM ratio (thresh / dense, block-tiled) |
|---|---|---|---|
| 1024 | 4.19 MB | yes | 1.117 |
| 2048 | 8.39 MB | yes | 1.074 |
| 4096 | 16.78 MB | no | 1.045 |
| 8192 | 33.55 MB | no | 1.026 |
| 16384 | 67.11 MB | no | 1.014 |

Threshold selection's memory traffic is roughly break-even with a properly
block-tiled dense baseline (ratio → 1.0, slightly worse at small N) — this
axis offers no independent advantage. It doesn't need to: the roofline
placement above already establishes the kernel is compute-bound, so bytes
moved were never going to be the deciding factor. This table is reported for
completeness, not as a second speedup claim.

## What was tried and ruled out

**Bounding-ball pruning** (Cauchy-Schwarz upper-bound test, meant to recover
O(N log N) complexity by proving whole tier blocks skippable before scoring):
confirmed dead. 0% prune rate at every N from 256 to 8192, under both a
local-window-seeded τ and an oracle τ (full pooled real scores, cheating with
complete knowledge). The bounding balls are genuinely too loose on this key
geometry — not an estimator artifact. O(N)-per-query QK scoring stands as a
real property of this design, not an implementation gap to be closed later.

## Known limitations / what this analysis does not establish

- The 5500U peak-FLOPs/bandwidth figures used for the ridge-point calculation
  are public-spec estimates, not measured on the actual chip — everything
  ELSE in the "Real hardware validation" section above (wall-clock,
  instructions retired, cache-miss rate) is real hardware measurement, not
  an estimate.
- The ball-pruning null result used the toy `D=16` head dimension used
  throughout this session for fast iteration; the roofline/ridge-point section
  above uses the real production `head_dim=64` and GQA config, but ball-pruning
  itself was not re-tested at production scale — filed as a caveat, not chased
  further since the pruning branch is closed regardless.
- The adversarial-scoring-stage vulnerability (threshold selection matches
  dense attention's ~90% crafted-adversarial fail rate exactly, four
  independently cross-validated ways) is a quality property, separate from
  this cost analysis, and remains filed as inherited-and-accepted — shared by
  every mechanism including dense itself, not fixable by attention-mechanism
  choice.
- T-iteration's incidental decoy-pressure damping (T-refined
  SlidingMonarchAttention stayed weakly positive at 50 decoys, where pure
  threshold selection went negative) is now moot as a design trade-off, not
  just accepted: SlidingMonarchAttention was subsequently disqualified
  outright on a more fundamental correctness failure (real-hardware needle
  probe, see above) that no amount of T-iteration budget fixes. The
  global-aggregation-hybrid mitigation once parked as future work is likewise
  closed — it would only add cost to Meta's already-capped QK-scoring stage,
  with no path to recovering Sliding's (disqualified) damping property.
- Design-space closure check (Fable consultation, `JOURNAL.md`): every
  MonarchAttention variant tried in this arc — bucket routing, floor-read,
  robust/geometric-median centroids, bounding-ball pruning, R-landmark
  generalizations with untrained construction, SlidingMonarchAttention's
  T-iteration-refined representative — used an UNTRAINED representative-
  construction heuristic and failed a same-norm adversarial control. A
  gradient-TRAINED summarizer has never been tested (deliberate methodology
  scope, not oversight) and is not provably doomed as a category — filed as
  genuinely open future work, explicitly out of scope for this standalone-
  probe arc, not gating the Meta recommendation above.
