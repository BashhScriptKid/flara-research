import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_meta_bucket_route import monarch_meta_bucket_route
from ma_meta_bucket_route_robust import monarch_meta_bucket_route_robust

D, Dv = 16, 16
B = 4
N = 256
QUERY_POS = 128
BLOCK_SPAN = 128
W_blocks = 1
BACKGROUND_NORM = 0.5 * (D ** 0.5)
n_buckets = 8
n_trials = 30

print("=== Robust (geometric-median) centroid vs original (arithmetic-mean) centroid ===")
print("(same adversarial mass-heavy-decoy setup that gave 83.33% failure originally)")
print()
print(f"{'method':>20} | {'mean cos':>9} {'min cos':>8} {'fail rate (cos<0.5)':>20} {'true-miss rate (cos<0.0)':>25}")

for name, fn, kwargs in [
    ("arithmetic-mean", monarch_meta_bucket_route, {}),
    ("geometric-median", monarch_meta_bucket_route_robust, {"weiszfeld_iters": 5}),
]:
    recalls = []
    fails = 0
    true_misses = 0
    for trial in range(n_trials):
        seed = 3000 + trial
        gg = torch.Generator().manual_seed(seed)
        bq = torch.randn(1, 1, N, D, generator=gg) * 0.5
        bk = torch.randn(1, 1, N, D, generator=gg) * 0.5
        bv = torch.randn(1, 1, N, Dv, generator=gg) * 0.5
        e = F.normalize(torch.randn(D, generator=gg), dim=0)
        val = F.normalize(torch.randn(Dv, generator=gg), dim=0) * 5.0
        needle_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=gg).item()
        decoy_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=gg).item()
        while decoy_pos == needle_pos:
            decoy_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=gg).item()
        decoy_dir = F.normalize(e + 0.3 * torch.randn(D, generator=gg), dim=0)

        k_full, v_full = bk.clone(), bv.clone()
        k_full[0, 0, needle_pos] = e * BACKGROUND_NORM
        v_full[0, 0, needle_pos] = val
        k_full[0, 0, decoy_pos] = decoy_dir * BACKGROUND_NORM * 3.0

        q_full = bq.clone()
        q_full[0, 0, QUERY_POS] = e * 6.0
        z = fn(q_full, k_full, v_full, B=B, W_blocks=W_blocks, n_buckets=n_buckets, **kwargs)[0, 0, QUERY_POS]
        cos = F.cosine_similarity(z, val, dim=0).item()
        recalls.append(cos)
        if cos < 0.5:
            fails += 1
        if cos < 0.0:
            true_misses += 1

    mean_r, min_r = sum(recalls) / len(recalls), min(recalls)
    print(f"{name:>20} | {mean_r:>9.4f} {min_r:>8.4f} {fails/n_trials:>19.2%} {true_misses/n_trials:>24.2%}")
