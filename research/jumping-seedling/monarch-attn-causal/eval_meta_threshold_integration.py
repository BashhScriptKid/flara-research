import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_meta_threshold import monarch_meta_threshold
from ma_sliding_monarch import sliding_monarch_causal as sma

D, Dv = 16, 16
B = 16
W_blocks = 1
N = 256
needle_pos = 18

print("=== Full integration test: Fenwick tiers + threshold selection + residual")
print("    centroid + window, NO Monarch/T-iteration machinery -- first assembled run ===")
print()


def make_base(seed):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    v_needle = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    return bq, bk, bv, e, v_needle


print("--- Needle-in-haystack vs distance, signal_scale=6.0 (compare to SlidingMonarch at various T) ---")
distances_qp = {0: 20, 2: 48, 5: 96, 9: 160, 14: 240}
bq, bk, bv, e, v_needle = make_base(1)
k_full = bk.clone(); k_full[0, 0, needle_pos] = e * 6.0
v_full = bv.clone(); v_full[0, 0, needle_pos] = v_needle

print(f"{'dist':>5} | {'GT':>7} | {'threshold(no-T)':>16} | {'SlidingMonarch T=1':>18} {'T=3':>7} {'T=8':>7}")
for dist, qp in distances_qp.items():
    q_full = bq.clone(); q_full[0, 0, qp] = e * 6.0
    z_gt = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)[0, 0, qp]
    cos = lambda z: F.cosine_similarity(z, v_needle, dim=0).item()
    z_thresh = monarch_meta_threshold(q_full, k_full, v_full, B=B, W_blocks=W_blocks)[0, 0, qp]
    z_t1 = sma(q_full, k_full, v_full, B=B, W_blocks=4, T=1)[0, 0, qp]
    z_t3 = sma(q_full, k_full, v_full, B=B, W_blocks=4, T=3)[0, 0, qp]
    z_t8 = sma(q_full, k_full, v_full, B=B, W_blocks=4, T=8)[0, 0, qp]
    print(f"{dist:>5} | {cos(z_gt):>7.4f} | {cos(z_thresh):>16.4f} | {cos(z_t1):>18.4f} {cos(z_t3):>7.4f} {cos(z_t8):>7.4f}")

print()
print("--- Needle-vs-decoy-count (same construction as the exact top-k/SlidingMonarch decoy tests) ---")
NEEDLE_SCALE = 3.0
qp_fixed = 240
print(f"{'num_decoys':>10} | {'threshold(no-T)':>16} | {'SlidingMonarch T=3':>19}")
for num_decoys in (0, 5, 20, 50):
    cos_thresh, cos_sma = [], []
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
        z2 = sma(q_full, k_full, v_full, B=B, W_blocks=4, T=3)[0, 0, qp_fixed]
        cos_thresh.append(F.cosine_similarity(z1, v_needle, dim=0).item())
        cos_sma.append(F.cosine_similarity(z2, v_needle, dim=0).item())
    print(f"{num_decoys:>10} | {sum(cos_thresh)/len(cos_thresh):>16.4f} | {sum(cos_sma)/len(cos_sma):>19.4f}")
