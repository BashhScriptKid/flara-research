import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_sliding_monarch import sliding_monarch_causal as sma

D, Dv = 16, 16
B = 16
N = 256
needle_pos = 18


def make_base(seed):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    v_needle = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    return bq, bk, bv, e, v_needle


print("=== Decoy pressure, SlidingMonarchAttention (needle scale=3.0 fixed, query_pos=240, dist=14 blocks) ===")
NEEDLE_SCALE = 3.0
qp_fixed = 240
for W in (1, 4, 8):
    print(f"\n-- W_blocks={W} --")
    print(f"{'num_decoys':>10} | {'cos recall':>10}   (mean over 10 trials)")
    for num_decoys in (0, 5, 20, 50):
        cos_vals = []
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
            z = sma(q_full, k_full, v_full, B=B, W_blocks=W, T=3)[0, 0, qp_fixed]
            cos_vals.append(F.cosine_similarity(z, v_needle, dim=0).item())
        print(f"{num_decoys:>10} | {sum(cos_vals) / len(cos_vals):>10.4f}")
