"""Sanity check for eval_sliding_multineedle_samenorm.py: same far-block
placement, window, T-iteration config -- but WITHOUT the same-norm
control (needle key at large magnitude, matching the style used
throughout most of this session's earlier SlidingMonarchAttention
validation). If THIS version passes cleanly, that confirms the probe
mechanics (far-block placement, window boundary, T-iteration wiring)
are correct, and the same-norm probe's K=1 failure is specifically
caused by removing the magnitude shortcut -- not a bug in the adapted
script.
"""
import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_sliding_monarch import sliding_monarch_causal

D, Dv = 16, 16
B = 8
W_blocks = 1
T = 3
N = 512
FAR_BLOCK_START = 64
QUERY_POS = 256
NEEDLE_SCALE = 6.0  # old style: large magnitude, no same-norm control


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
        k_full[0, 0, pos] = e * NEEDLE_SCALE  # large magnitude, NO same-norm control
        v_full[0, 0, pos] = val
    return bq, k_full, v_full, directions, values


print("=== Sanity check: same far-block probe, WITHOUT same-norm control ===")
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
