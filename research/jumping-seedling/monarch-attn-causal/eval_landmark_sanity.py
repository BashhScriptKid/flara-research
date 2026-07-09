import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_meta_monarch import monarch_meta
from landmark_mechanics import MECHANICS

D, Dv = 16, 16
B = 4
N = 256
QUERY_POS = 128
BLOCK_SPAN = 128
W_blocks = 1


def make_single_needle(seed, needle_pos):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    val = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    k_full = bk.clone(); k_full[0, 0, needle_pos] = e * 6.0
    v_full = bv.clone(); v_full[0, 0, needle_pos] = val
    q_full = bq.clone(); q_full[0, 0, QUERY_POS] = e * 6.0
    return q_full, k_full, v_full, val


print("=== Single-needle sanity check (K=1, zero competition) across R, per mechanic ===")
print("Needle at a RANDOM position within the stressed block, averaged over 5 seeds/positions.")
print()
Rs = (2, 4, 8, 16, 32, 64)
results = {}
for name, fn in MECHANICS.items():
    print(f"-- {name} --")
    print(f"{'R':>4} | {'mean cos':>9} {'min cos':>8}")
    crossing_R = None
    for R in Rs:
        coses = []
        g = torch.Generator().manual_seed(42)
        for trial in range(5):
            needle_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=g).item()
            q_full, k_full, v_full, val = make_single_needle(seed=100 + trial, needle_pos=needle_pos)
            z = monarch_meta(q_full, k_full, v_full, B=B, W_blocks=W_blocks, R=R, landmark_fn=fn)[0, 0, QUERY_POS]
            coses.append(F.cosine_similarity(z, val, dim=0).item())
        mean_c, min_c = sum(coses) / len(coses), min(coses)
        print(f"{R:>4} | {mean_c:>9.4f} {min_c:>8.4f}")
        if crossing_R is None and mean_c > 0.9:
            crossing_R = R
    results[name] = crossing_R
    print(f"  -> crosses ~1.0 recall at R={crossing_R} (out of block size 128)" if crossing_R else "  -> NEVER reaches ~1.0 recall in tested range")
    print()

print("=== Summary ===")
for name, r in results.items():
    verdict = f"passes at R={r}" if r else "FAILS sanity check entirely"
    print(f"{name:>14}: {verdict}")
