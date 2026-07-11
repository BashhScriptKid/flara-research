"""Fable's flagged-as-most-urgent diagnostic: does cross-tier
accumulation degrade the NATURAL (zero-decoy) case as active tier
count L grows with sequence length, or is the accumulation effect
specific to adversarial decoy pressure? This determines whether the
cross-tier finding is a bounded/accepted risk (filed alongside every
other cliff this session) or a core-value-proposition problem (the
architecture's own scaling axis is also its vulnerability).

Mechanism worth checking directly: tau is QUANTILE-based (90th
percentile of each tier's own block), which is RELATIVE, not absolute
-- by construction, ~10% of every tier's block "survives" regardless of
whether anything in that tier is actually relevant. If so, more active
tiers should structurally mean more background-noise competing mass in
the same final softmax even with zero decoys, since ordinary background
keys clear their own tier's local quantile by definition.
"""

import sys, math
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_meta_threshold import monarch_meta_threshold

D, Dv = 16, 16
B = 16
W_blocks = 1
needle_pos = 18
NEEDLE_SCALE = 6.0


def make_base(seed, N):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    v_needle = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    return bq, bk, bv, e, v_needle


def active_tier_count(query_pos, B, W_blocks, N):
    M_base_needed = (N + B - 1) // B
    L = max(1, math.ceil(math.log2(max(M_base_needed, 2))))
    m0 = query_pos // B
    n = m0 - W_blocks + 1
    return sum(1 for l in range(L) if (n >> l) & 1), L


print("=== Natural (zero-decoy) recall vs active tier count L ===")
print("(needle strength and position fixed; query position varied to increase")
print(" the number of active Fenwick tiers via longer sequences / larger distances)")
print()
print(f"{'N':>6} {'query_pos':>10} {'active_tiers':>13} {'L(max)':>7} | {'mean cos (n=15)':>16} {'min cos':>9}")

# sweep N (and correspondingly query position near the end) to grow active tier count
configs = [
    (256, 240),
    (512, 496),
    (1024, 1008),
    (2048, 2032),
    (4096, 4080),
    (8192, 8176),
]

for N, qp in configs:
    n_active, L_max = active_tier_count(qp, B, W_blocks, N)
    coses = []
    for trial in range(15):
        bq, bk, bv, e, v_needle = make_base(2000 + trial, N)
        k_full = bk.clone(); k_full[0, 0, needle_pos] = e * NEEDLE_SCALE
        v_full = bv.clone(); v_full[0, 0, needle_pos] = v_needle
        q_full = bq.clone(); q_full[0, 0, qp] = e * NEEDLE_SCALE
        z = monarch_meta_threshold(q_full, k_full, v_full, B=B, W_blocks=W_blocks)[0, 0, qp]
        coses.append(F.cosine_similarity(z, v_needle, dim=0).item())
    t = torch.tensor(coses)
    print(f"{N:>6} {qp:>10} {n_active:>13} {L_max:>7} | {t.mean().item():>16.4f} {t.min().item():>9.4f}")

print()
print("If mean cos degrades measurably as active_tiers grows (even at signal_scale=6.0,")
print("the easiest case, zero decoys), that confirms the tau-is-relative-not-absolute")
print("mechanism: every tier admits ~10% of its own block regardless of relevance,")
print("so more tiers structurally means more background-noise competing mass, even")
print("with no adversary at all -- the core-scaling-axis-is-the-vulnerability result.")
