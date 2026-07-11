"""Same-norm-controlled single/multi-needle probe against
SlidingMonarchAttention specifically, per Fable's flagged gap: Sliding's
original correctness validation (JOURNAL.md line 364, first build,
"strong result") happened BEFORE the same-norm control existed in this
session's methodology. That control (introduced later, see
eval_bucket_multineedle.py) unmasked a magnitude-scaling artifact that
had been silently inflating earlier "passes" for FIVE separate untrained
representative-construction mechanics tried for Meta's R-landmark
generalization -- once needle keys were forced to the SAME norm as
background keys (so only DIRECTION, not magnitude, can carry signal),
all five failed single-needle detection outright.

Sliding uses a single fixed representative per block (R=1) -- the
TIGHTEST possible version of the "more than R genuinely-distinct
needles collide" capacity ceiling already established in this arc, and
its founding correctness claim has never been re-checked under this
control. This probe places needle(s) in a FAR block (outside the local
exact window, so only Sliding's T-iteration-refined Monarch
representative -- not the exact window -- can carry the signal),
holding needle key norm equal to background key norm throughout.
"""
import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_sliding_monarch import sliding_monarch_causal

D, Dv = 16, 16
B = 8            # Monarch block size
W_blocks = 1     # local exact window: ONLY the query's own block -- needle
                 # block must therefore be strictly before it to land in
                 # the far/Monarch-representative region, not the exact window
T = 3            # validated default T-iteration count
N = 512
FAR_BLOCK_START = 64   # far block spans positions [64, 64+B) = [64,72)
QUERY_POS = 256         # well past the far block + window boundary
BACKGROUND_NORM = 0.5 * (D ** 0.5)  # same-norm control: matches background key norm


def make_scene(seed, K):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    needle_positions = FAR_BLOCK_START + torch.randperm(B, generator=g)[:K].sort().values
    directions = [F.normalize(torch.randn(D, generator=g), dim=0) for _ in range(K)]
    values = [F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0 for _ in range(K)]
    k_full, v_full = bk.clone(), bv.clone()
    for pos, e, val in zip(needle_positions.tolist(), directions, values):
        k_full[0, 0, pos] = e * BACKGROUND_NORM  # same-norm control -- no magnitude shortcut
        v_full[0, 0, pos] = val
    return bq, k_full, v_full, directions, values


print("=== Same-norm-controlled single/multi-needle probe: SlidingMonarchAttention ===")
print(f"(needle(s) placed in FAR block [{FAR_BLOCK_START},{FAR_BLOCK_START+B}), query at pos {QUERY_POS},")
print(f" only the T-iteration-refined Monarch representative can carry the signal, not the exact window)")
print()
print(f"{'K needles':>10} | {'mean cos':>9} {'min cos':>8} {'frac>0.5':>9}")
for K in (1, 2, 4, 8):
    recalls = []
    bq, k_full, v_full, directions, values = make_scene(seed=1, K=K)
    for e, val in zip(directions, values):
        q_full = bq.clone()
        q_full[0, 0, QUERY_POS] = e * 6.0
        z = sliding_monarch_causal(q_full, k_full, v_full, B=B, W_blocks=W_blocks, T=T)[0, 0, QUERY_POS]
        recalls.append(F.cosine_similarity(z, val, dim=0).item())
    mean_r = sum(recalls) / len(recalls)
    min_r = min(recalls)
    frac_good = sum(1 for r in recalls if r > 0.5) / len(recalls)
    print(f"{K:>10} | {mean_r:>9.4f} {min_r:>8.4f} {frac_good:>9.2f}")

print()
print("If K=1 (single needle, no competition at all) fails under this control,")
print("that reproduces the magnitude-artifact pattern found for Meta's untrained")
print("landmark mechanics -- a fundamental detection failure, not a capacity limit.")
print("If K=1 passes but recall degrades sharply as K grows, that's the EXPECTED")
print("R=1 capacity ceiling (multiple needles in one block competing for one")
print("representative slot), a known and different failure mode.")
