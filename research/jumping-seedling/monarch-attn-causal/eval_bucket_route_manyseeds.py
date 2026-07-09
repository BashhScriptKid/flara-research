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
N_TRIALS = 100


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


print(f"=== Query-informed bucket routing: {N_TRIALS}-seed sweep, all n_buckets ===")
print("(checking whether n_buckets=8's -0.58 min was a one-off, or reproducible,")
print(" and whether other bucket counts have hidden failures too small-sample missed)")
print()
print(f"{'n_buckets':>10} {'avg bkt sz':>11} | {'mean':>7} {'min':>7} {'p10':>7} | "
      f"{'frac<0.5':>9} {'frac<0.0':>9} {'n_fail(<0.5)':>13}")

FAIL_THRESH = 0.5
NEG_THRESH = 0.0

results = {}
for nb in (2, 4, 8, 16, 32, 64):
    coses = []
    g = torch.Generator().manual_seed(7)
    for trial in range(N_TRIALS):
        needle_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=g).item()
        q_full, k_full, v_full, val = make_same_norm_needle(seed=2000 + trial, needle_pos=needle_pos)
        z = monarch_meta_bucket_route(q_full, k_full, v_full, B=B, W_blocks=W_blocks, n_buckets=nb)[0, 0, QUERY_POS]
        coses.append(F.cosine_similarity(z, val, dim=0).item())
    coses_t = torch.tensor(coses)
    mean_c = coses_t.mean().item()
    min_c = coses_t.min().item()
    p10 = coses_t.kthvalue(max(1, int(0.1 * N_TRIALS))).values.item()
    frac_fail = (coses_t < FAIL_THRESH).float().mean().item()
    frac_neg = (coses_t < NEG_THRESH).float().mean().item()
    n_fail = int((coses_t < FAIL_THRESH).sum().item())
    results[nb] = coses
    print(f"{nb:>10} {BLOCK_SPAN/nb:>11.1f} | {mean_c:>7.4f} {min_c:>7.4f} {p10:>7.4f} | "
          f"{frac_fail:>9.2%} {frac_neg:>9.2%} {n_fail:>13}")

print()
print("=== Failing seeds for n_buckets=8 specifically (cos < 0.5) ===")
g = torch.Generator().manual_seed(7)
needle_positions = [torch.randint(0, BLOCK_SPAN, (1,), generator=g).item() for _ in range(N_TRIALS)]
for i, c in enumerate(results[8]):
    if c < FAIL_THRESH:
        print(f"  trial {i} (seed={2000+i}, needle_pos={needle_positions[i]}): cos={c:.4f}")
