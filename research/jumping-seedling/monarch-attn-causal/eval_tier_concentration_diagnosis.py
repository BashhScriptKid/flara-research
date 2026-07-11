"""Diagnostic: at 50 decoys, is decoy-survivor mass concentrated in ONE
Fenwick tier (re-derivation of the known same-bucket contested-
colocation finding) or spread ACROSS MANY tiers (a genuinely new
failure mode -- each tier independently applies its own tau test with
no knowledge that a sibling tier already admitted several decoys, so
total admitted survivor count across the whole tree can accumulate
additively in a way no single-tier analysis predicts)?
"""

import sys, math
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F
from math import sqrt

D, Dv = 16, 16
B = 16
W_blocks = 1
N = 256
needle_pos = 18
NEEDLE_SCALE = 3.0
qp_fixed = 240
sm_scale = 1 / sqrt(D)


def make_base(seed):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    v_needle = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    return bq, bk, bv, e, v_needle


print("=== Tier-concentration diagnosis at 50 decoys ===")
print()

M_base_needed = (N + B - 1) // B
L = max(1, math.ceil(math.log2(max(M_base_needed, 2))))
N_padded = B * (1 << L)
print(f"L={L} tiers, N_padded={N_padded}")

m0 = qp_fixed // B
n = m0 - W_blocks + 1
candidates = []
for l in range(L):
    if (n >> l) & 1:
        candidates.append((l, (n >> (l + 1)) << 1))
print(f"query block m0={m0}, n={n} (binary {bin(n)}), active tiers/blocks: {candidates}")
print()

total_survivors_per_trial = []
concentration_per_trial = []  # fraction of decoy survivors in the single tier containing the needle

for trial in range(10):
    g = torch.Generator().manual_seed(1000 + trial)
    bq, bk, bv, e, v_needle = make_base(1)
    k_full = bk.clone(); k_full[0, 0, needle_pos] = e * NEEDLE_SCALE
    v_full = bv.clone(); v_full[0, 0, needle_pos] = v_needle
    decoy_positions = torch.randperm(qp_fixed - 1, generator=g)[:50]
    decoy_positions = decoy_positions[decoy_positions != needle_pos]
    decoy_scales = 0.9 + 0.4 * torch.rand(len(decoy_positions), generator=g)
    for pos, dscale in zip(decoy_positions.tolist(), decoy_scales.tolist()):
        k_full[0, 0, pos] = e * (NEEDLE_SCALE * dscale)
    q_full = bq.clone(); q_full[0, 0, qp_fixed] = e * NEEDLE_SCALE

    pad = N_padded - N
    k_flat = F.pad(k_full, (0, 0, 0, pad))
    query = q_full[0, 0, qp_fixed]

    needle_tier = None
    per_tier_survivors = []
    for l, block_idx in candidates:
        Bl = B * (1 << l)
        Ml = N_padded // Bl
        k_block = k_flat.view(1, 1, Ml, Bl, D)[0, 0, block_idx]  # (Bl,D)
        scores = sm_scale * (k_block @ query)  # (Bl,)
        tau = torch.quantile(scores, 0.90).item()
        survivors = (scores >= tau)
        n_surv = survivors.sum().item()
        # does this tier's block contain the needle position?
        block_start = block_idx * Bl
        block_end = block_start + Bl
        contains_needle = block_start <= needle_pos < block_end
        needle_survives_here = False
        if contains_needle:
            local_idx = needle_pos - block_start
            needle_survives_here = bool(survivors[local_idx].item())
            needle_tier = l
        # count decoys among survivors in this tier (all survivors except needle if present)
        n_decoy_surv = n_surv - (1 if needle_survives_here else 0)
        per_tier_survivors.append((l, Bl, n_surv, n_decoy_surv, contains_needle, needle_survives_here))

    total_decoy_surv = sum(t[3] for t in per_tier_survivors)
    needle_tier_decoy_surv = sum(t[3] for t in per_tier_survivors if t[0] == needle_tier) if needle_tier is not None else 0
    concentration = needle_tier_decoy_surv / total_decoy_surv if total_decoy_surv > 0 else float("nan")

    total_survivors_per_trial.append(total_decoy_surv)
    concentration_per_trial.append(concentration)

    if trial < 3:
        print(f"-- trial {trial} -- needle in tier l={needle_tier}")
        for l, Bl, n_surv, n_decoy_surv, contains_needle, needle_ok in per_tier_survivors:
            marker = " <- contains needle" if contains_needle else ""
            print(f"   tier l={l} (Bl={Bl}): {n_surv} survivors, {n_decoy_surv} decoys{marker}")
        print()

avg_total = sum(total_survivors_per_trial) / len(total_survivors_per_trial)
valid_conc = [c for c in concentration_per_trial if c == c]  # filter nan
avg_conc = sum(valid_conc) / len(valid_conc) if valid_conc else float("nan")
print(f"avg total decoy survivors across ALL tiers: {avg_total:.1f}")
print(f"avg fraction of decoy survivors concentrated in the needle's OWN tier: {avg_conc:.2%}")
print()
print("If avg_conc is close to 100%, decoy pressure is concentrated in one tier")
print("(re-derivation of the known same-bucket finding). If avg_conc is well below")
print("100%, decoy survivor mass is spread across multiple tiers -- confirming the")
print("new additive-across-tree failure mode Fable flagged.")
