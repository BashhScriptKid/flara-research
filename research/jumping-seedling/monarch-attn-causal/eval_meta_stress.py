import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_meta_monarch import monarch_meta

D, Dv = 16, 16
B = 4
W_blocks = 1

# By construction: n=32 (binary 100000) has ONLY bit 5 set, so this query
# selects EXACTLY tier l=5 (block size B*2^5=128), block_idx=0 (tokens
# [0,128)) -- no other tiers active, isolating tier 5's own R-capacity
# behavior with nothing else in the joint softmax to dilute the read.
N = 256
n = 32
m0 = n + W_blocks - 1  # = 32
QUERY_POS = m0 * B  # = 128
BLOCK_SPAN = 128  # tokens [0, 128) -- the block under stress

print(f"Query position: {QUERY_POS}, stressed block: tokens [0, {BLOCK_SPAN})")
print()


def make_scene(seed, K, needle_scale=6.0):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5

    # K distinct needles at RANDOM positions (deliberately NOT evenly-spaced
    # like the landmarks -- an earlier version used linspace() for both,
    # which made landmark positions coincide exactly with needle positions
    # at K=R, giving trivial perfect self-retrieval that had nothing to do
    # with real compression capacity. Random placement avoids that artifact.)
    needle_positions = torch.randperm(BLOCK_SPAN, generator=g)[:K].sort().values
    directions = [F.normalize(torch.randn(D, generator=g), dim=0) for _ in range(len(needle_positions))]
    values = [F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0 for _ in range(len(needle_positions))]

    k_full = bk.clone()
    v_full = bv.clone()
    for pos, e, val in zip(needle_positions.tolist(), directions, values):
        k_full[0, 0, pos] = e * needle_scale
        v_full[0, 0, pos] = val

    return bq, k_full, v_full, needle_positions.tolist(), directions, values


print("=== Multi-needle stress: how many simultaneous needles can R landmarks recover? ===")
print(f"{'R':>4} {'K needles':>10} | {'mean recall cos':>16} {'min recall cos':>15} {'frac > 0.7':>11}")
for R in (2, 4, 8, 16, 32):
    for K in (1, 2, 4, 8, 16, 32):
        bq, k_full, v_full, positions, directions, values = make_scene(seed=1, K=K)
        recalls = []
        for e, val in zip(directions, values):
            q_full = bq.clone()
            q_full[0, 0, QUERY_POS] = e * 6.0
            z = monarch_meta(q_full, k_full, v_full, B=B, W_blocks=W_blocks, R=R)[0, 0, QUERY_POS]
            recalls.append(F.cosine_similarity(z, val, dim=0).item())
        mean_r = sum(recalls) / len(recalls)
        min_r = min(recalls)
        frac_good = sum(1 for r in recalls if r > 0.7) / len(recalls)
        print(f"{R:>4} {K:>10} | {mean_r:>16.4f} {min_r:>15.4f} {frac_good:>11.2f}")
    print()
