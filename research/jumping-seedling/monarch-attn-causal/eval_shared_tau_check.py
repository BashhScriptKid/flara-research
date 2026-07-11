"""Verify causal validity of the shared-tau variant, then re-run the
50-decoy cross-tier adversarial sweep comparing shared-tau against the
existing independent-per-tier-tau baseline (ma_meta_threshold.py).
"""

import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_meta_threshold import monarch_meta_threshold
from ma_meta_threshold_shared_tau import monarch_meta_threshold_shared_tau

D, Dv = 16, 16
B = 16
W_blocks = 1
N = 256
needle_pos = 18
NEEDLE_SCALE = 3.0
qp_fixed = 240


def make_base(seed):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    v_needle = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    return bq, bk, bv, e, v_needle


print("=== Causal validity check: monarch_meta_threshold_shared_tau ===")
g = torch.Generator().manual_seed(7)
q = torch.randn(1, 2, N, D, generator=g)
k = torch.randn(1, 2, N, D, generator=g)
v = torch.randn(1, 2, N, Dv, generator=g)
z = monarch_meta_threshold_shared_tau(q, k, v, B=B, W_blocks=W_blocks)

# leak test: perturb a future key, confirm earlier outputs unchanged
k2 = k.clone()
k2[0, 0, 200] += 100.0
z2 = monarch_meta_threshold_shared_tau(q, k2, v, B=B, W_blocks=W_blocks)
leak = (z[0, 0, :190] - z2[0, 0, :190]).abs().max().item()
print(f"leak (max abs diff on positions <190 after perturbing pos 200): {leak:.8f}")

# rowsum test via softmax weights isn't directly exposed; use output boundedness/finite check instead
finite_ok = torch.isfinite(z).all().item()
print(f"all outputs finite: {finite_ok}")
print()

print("=== 50-decoy cross-tier adversarial sweep: shared-tau vs independent-tau ===")
print(f"{'num_decoys':>10} | {'independent-tau':>16} | {'shared-tau':>10}")
for num_decoys in (0, 5, 20, 50):
    cos_indep, cos_shared = [], []
    for trial in range(10):
        g = torch.Generator().manual_seed(1000 + trial)
        bq, bk, bv, e, v_needle = make_base(1)
        k_full = bk.clone(); k_full[0, 0, needle_pos] = e * NEEDLE_SCALE
        v_full = bv.clone(); v_full[0, 0, needle_pos] = v_needle
        if num_decoys > 0:
            decoy_positions = torch.randperm(qp_fixed - 1, generator=g)[:num_decoys]
            decoy_positions = decoy_positions[decoy_positions != needle_pos]
            decoy_scales = 0.9 + 0.4 * torch.rand(len(decoy_positions), generator=g)
            for pos, dscale in zip(decoy_positions.tolist(), decoy_scales.tolist()):
                k_full[0, 0, pos] = e * (NEEDLE_SCALE * dscale)
        q_full = bq.clone(); q_full[0, 0, qp_fixed] = e * NEEDLE_SCALE

        z1 = monarch_meta_threshold(q_full, k_full, v_full, B=B, W_blocks=W_blocks)[0, 0, qp_fixed]
        z2 = monarch_meta_threshold_shared_tau(q_full, k_full, v_full, B=B, W_blocks=W_blocks)[0, 0, qp_fixed]
        cos_indep.append(F.cosine_similarity(z1, v_needle, dim=0).item())
        cos_shared.append(F.cosine_similarity(z2, v_needle, dim=0).item())
    print(f"{num_decoys:>10} | {sum(cos_indep)/len(cos_indep):>16.4f} | {sum(cos_shared)/len(cos_shared):>10.4f}")
