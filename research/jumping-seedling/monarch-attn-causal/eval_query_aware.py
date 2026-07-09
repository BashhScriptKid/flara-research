import sys, time
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_meta_query_aware import monarch_meta_query_aware
from ma_meta_monarch import monarch_meta
from landmark_mechanics import MECHANICS

D, Dv = 16, 16
B = 4
N = 256
QUERY_POS = 128
BLOCK_SPAN = 128
W_blocks = 1
BACKGROUND_NORM = 0.5 * (D ** 0.5)


def make_same_norm_needle(seed, needle_pos):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    val = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    k_full = bk.clone(); k_full[0, 0, needle_pos] = e * BACKGROUND_NORM  # same norm as background
    v_full = bv.clone(); v_full[0, 0, needle_pos] = val
    q_full = bq.clone(); q_full[0, 0, QUERY_POS] = e * 6.0
    return q_full, k_full, v_full, val


print("=== Query-aware block-local exact attention: same-norm single-needle test ===")
print(f"(needle key norm matched to background ~{BACKGROUND_NORM:.2f} -- same control as Axis 1)")
print()
coses = []
g = torch.Generator().manual_seed(42)
for trial in range(10):
    needle_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=g).item()
    q_full, k_full, v_full, val = make_same_norm_needle(seed=100 + trial, needle_pos=needle_pos)
    z = monarch_meta_query_aware(q_full, k_full, v_full, B=B, W_blocks=W_blocks)[0, 0, QUERY_POS]
    coses.append(F.cosine_similarity(z, val, dim=0).item())
mean_c, min_c = sum(coses) / len(coses), min(coses)
print(f"query-aware block-local exact attention: mean cos={mean_c:.4f}, min cos={min_c:.4f}, all 10: {[f'{c:.3f}' for c in coses]}")
print()
print("(for reference, Axis 1's same-norm results for precomputed landmarks, R=64, mean cos:")
print(" random_reuse=0.1768, top_magnitude=0.1768, fps=0.1799 -- all near 0.15-0.18)")

print()
print("=== Cost: query-aware (O(Bl) per query, no precompute) vs precomputed landmarks (R fixed) ===")

def bench(fn, *args, reps=10, warmup=2):
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args)
        return (time.perf_counter() - t0) / reps

print(f"{'N':>5} | {'query-aware':>12} | {'landmark R=8':>13} {'landmark R=32':>14} | {'query-aware/R=8':>16}")
for N2 in (256, 512, 1024, 2048):
    q = torch.randn(1, 8, N2, 64); k = torch.randn(1, 8, N2, 64); v = torch.randn(1, 8, N2, 64)
    reps = 10 if N2 <= 1024 else 5
    t_qa = bench(monarch_meta_query_aware, q, k, v, B, 1, reps=reps)
    t_r8 = bench(monarch_meta, q, k, v, B, 1, 8, MECHANICS["random_reuse"], reps=reps)
    t_r32 = bench(monarch_meta, q, k, v, B, 1, 32, MECHANICS["random_reuse"], reps=reps)
    print(f"{N2:>5} | {t_qa*1e3:>11.2f}ms | {t_r8*1e3:>12.2f}ms {t_r32*1e3:>13.2f}ms | {t_qa/t_r8:>16.2f}")
