import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_meta_bucket_route import monarch_meta_bucket_route

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
    k_full = bk.clone(); k_full[0, 0, needle_pos] = e * BACKGROUND_NORM
    v_full = bv.clone(); v_full[0, 0, needle_pos] = val
    q_full = bq.clone(); q_full[0, 0, QUERY_POS] = e * 6.0
    return q_full, k_full, v_full, val


print("=== Query-informed bucket routing: same-norm single-needle test, sweep n_buckets ===")
print(f"(block size = {BLOCK_SPAN}; n_buckets controls average bucket size = {BLOCK_SPAN}/n_buckets)")
print()
print(f"{'n_buckets':>10} {'avg bucket sz':>14} | {'mean cos':>9} {'min cos':>8}")
for nb in (2, 4, 8, 16, 32, 64):
    coses = []
    g = torch.Generator().manual_seed(42)
    for trial in range(10):
        needle_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=g).item()
        q_full, k_full, v_full, val = make_same_norm_needle(seed=100 + trial, needle_pos=needle_pos)
        z = monarch_meta_bucket_route(q_full, k_full, v_full, B=B, W_blocks=W_blocks, n_buckets=nb)[0, 0, QUERY_POS]
        coses.append(F.cosine_similarity(z, val, dim=0).item())
    mean_c, min_c = sum(coses) / len(coses), min(coses)
    print(f"{nb:>10} {BLOCK_SPAN/nb:>14.1f} | {mean_c:>9.4f} {min_c:>8.4f}")
