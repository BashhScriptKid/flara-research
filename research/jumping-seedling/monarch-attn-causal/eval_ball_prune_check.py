"""Correctness rerun + prune-rate measurement for the bounding-ball
pruned threshold-selection variant (ma_meta_threshold_ball_prune.py),
per Fable's recommended sequencing: (1) causal validity, (2) natural +
adversarial harnesses should come back IDENTICAL in outcome to the
existing flat threshold-selection numbers (a regression means a bug in
the bound math, not a reopened design question), (3) only then measure
actual prune rate across the N=256->8192 sweep to determine whether
item 2 should be a constant-factor or complexity-class cost model.
"""

import sys, math
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_meta_threshold_ball_prune import monarch_meta_threshold_ball_prune

D, Dv = 16, 16
B = 16
W_blocks = 1
needle_pos = 18


def make_base(seed, N=256):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    v_needle = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    return bq, bk, bv, e, v_needle


print("=== (1) Causal validity check ===")
N = 256
g = torch.Generator().manual_seed(7)
q = torch.randn(1, 2, N, D, generator=g)
k = torch.randn(1, 2, N, D, generator=g)
v = torch.randn(1, 2, N, Dv, generator=g)
z, _ = monarch_meta_threshold_ball_prune(q, k, v, B=B, W_blocks=W_blocks)
k2 = k.clone(); k2[0, 0, 200] += 100.0
z2, _ = monarch_meta_threshold_ball_prune(q, k2, v, B=B, W_blocks=W_blocks)
leak = (z[0, 0, :190] - z2[0, 0, :190]).abs().max().item()
print(f"leak (positions <190 after perturbing pos 200): {leak:.8f}")
print(f"all outputs finite: {torch.isfinite(z).all().item()}")
print()

print("=== (2a) Natural needle-in-haystack recall, N=256 (n=200 trials) ===")
NEEDLE_SCALE = 6.0
qp = 240
fail = 0
n_trials = 200
for trial in range(n_trials):
    bq, bk, bv, e, v_needle = make_base(2000 + trial, N=256)
    k_full = bk.clone(); k_full[0, 0, needle_pos] = e * NEEDLE_SCALE
    v_full = bv.clone(); v_full[0, 0, needle_pos] = v_needle
    q_full = bq.clone(); q_full[0, 0, qp] = e * NEEDLE_SCALE
    z, _ = monarch_meta_threshold_ball_prune(q_full, k_full, v_full, B=B, W_blocks=W_blocks)
    cos = F.cosine_similarity(z[0, 0, qp], v_needle, dim=0).item()
    if cos < 0.5:
        fail += 1
print(f"natural fail rate: {fail}/{n_trials} = {100*fail/n_trials:.2f}%  (existing baseline: 0.00%)")
print()

print("=== (2b) 50-decoy cross-tier adversarial sweep vs existing baselines ===")
NEEDLE_SCALE_ADV = 3.0
qp_fixed = 240
print(f"{'num_decoys':>10} | {'ball-prune (this)':>18}")
for num_decoys in (0, 5, 20, 50):
    cos_bp = []
    for trial in range(10):
        g = torch.Generator().manual_seed(1000 + trial)
        bq, bk, bv, e, v_needle = make_base(1, N=256)
        k_full = bk.clone(); k_full[0, 0, needle_pos] = e * NEEDLE_SCALE_ADV
        v_full = bv.clone(); v_full[0, 0, needle_pos] = v_needle
        if num_decoys > 0:
            decoy_positions = torch.randperm(qp_fixed - 1, generator=g)[:num_decoys]
            decoy_positions = decoy_positions[decoy_positions != needle_pos]
            decoy_scales = 0.9 + 0.4 * torch.rand(len(decoy_positions), generator=g)
            for pos, dscale in zip(decoy_positions.tolist(), decoy_scales.tolist()):
                k_full[0, 0, pos] = e * (NEEDLE_SCALE_ADV * dscale)
        q_full = bq.clone(); q_full[0, 0, qp_fixed] = e * NEEDLE_SCALE_ADV
        z, _ = monarch_meta_threshold_ball_prune(q_full, k_full, v_full, B=B, W_blocks=W_blocks)
        cos_bp.append(F.cosine_similarity(z[0, 0, qp_fixed], v_needle, dim=0).item())
    print(f"{num_decoys:>10} | {sum(cos_bp)/len(cos_bp):>18.4f}")
print("(compare to independent-tau/shared-tau baseline: 0.9580/0.9399, 0.7205/0.7290,")
print(" 0.4445/0.4358, -0.1312/-0.1228 at 0/5/20/50 decoys respectively)")
print()

print("=== (3) Prune-rate measurement across N sweep (zero decoys, needle fixed) ===")
configs = [
    (256, 240),
    (512, 496),
    (1024, 1008),
    (2048, 2032),
    (4096, 4080),
    (8192, 8176),
]
print(f"{'N':>6} {'query_pos':>10} | {'prune_rate':>10} {'n_tested':>10} {'n_pruned':>10}")
for Nval, qpv in configs:
    bq, bk, bv, e, v_needle = make_base(2000, N=Nval)
    k_full = bk.clone(); k_full[0, 0, needle_pos] = e * NEEDLE_SCALE
    v_full = bv.clone(); v_full[0, 0, needle_pos] = v_needle
    q_full = bq.clone(); q_full[0, 0, qpv] = e * NEEDLE_SCALE
    z, stats = monarch_meta_threshold_ball_prune(q_full, k_full, v_full, B=B, W_blocks=W_blocks)
    print(f"{Nval:>6} {qpv:>10} | {stats['prune_rate']:>10.2%} {stats['n_tested']:>10} {stats['n_pruned']:>10}")
