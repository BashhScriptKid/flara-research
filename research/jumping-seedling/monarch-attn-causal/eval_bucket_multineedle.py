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


def make_multi_needle_scene(seed, K):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    needle_positions = torch.randperm(BLOCK_SPAN, generator=g)[:K].sort().values
    directions = [F.normalize(torch.randn(D, generator=g), dim=0) for _ in range(K)]
    values = [F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0 for _ in range(K)]
    k_full, v_full = bk.clone(), bv.clone()
    for pos, e, val in zip(needle_positions.tolist(), directions, values):
        k_full[0, 0, pos] = e * BACKGROUND_NORM  # same-norm control
        v_full[0, 0, pos] = val
    return bq, k_full, v_full, directions, values


print("=== Probe: multi-needle competition for query-informed bucket routing ===")
print("(hypothesis: buckets in DIFFERENT clusters don't compete for capacity;")
print(" the hard case should be same-bucket crowding, not raw needle count)")
print()
print(f"{'n_buckets':>10} {'K needles':>10} | {'mean cos':>9} {'min cos':>8} {'frac>0.5':>9}")
for nb in (8, 32):
    for K in (1, 2, 4, 8, 16, 32):
        recalls = []
        bq, k_full, v_full, directions, values = make_multi_needle_scene(seed=1, K=K)
        for e, val in zip(directions, values):
            q_full = bq.clone()
            q_full[0, 0, QUERY_POS] = e * 6.0
            z = monarch_meta_bucket_route(q_full, k_full, v_full, B=B, W_blocks=W_blocks, n_buckets=nb)[0, 0, QUERY_POS]
            recalls.append(F.cosine_similarity(z, val, dim=0).item())
        mean_r = sum(recalls) / len(recalls)
        min_r = min(recalls)
        frac_good = sum(1 for r in recalls if r > 0.5) / len(recalls)
        print(f"{nb:>10} {K:>10} | {mean_r:>9.4f} {min_r:>8.4f} {frac_good:>9.2f}")
    print()

print("=== Adversarial: needle near a Voronoi boundary, decoy anchoring the centroid ===")
print("(the pre-registered 'real' hard case: routing-stage competition, not attention-stage)")
n_buckets = 8
n_trials = 30
boundary_fail = 0
recalls = []
g = torch.Generator().manual_seed(99)
for trial in range(n_trials):
    seed = 3000 + trial
    gg = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=gg) * 0.5
    bk = torch.randn(1, 1, N, D, generator=gg) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=gg) * 0.5
    e = F.normalize(torch.randn(D, generator=gg), dim=0)
    val = F.normalize(torch.randn(Dv, generator=gg), dim=0) * 5.0
    needle_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=gg).item()
    # decoy: same block, slightly perturbed direction from the needle, placed
    # to compete for the same cluster region (a similar-but-not-identical key,
    # nudging the local centroid away from the needle's exact direction).
    decoy_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=gg).item()
    while decoy_pos == needle_pos:
        decoy_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=gg).item()
    decoy_dir = F.normalize(e + 0.3 * torch.randn(D, generator=gg), dim=0)

    k_full, v_full = bk.clone(), bv.clone()
    k_full[0, 0, needle_pos] = e * BACKGROUND_NORM
    v_full[0, 0, needle_pos] = val
    k_full[0, 0, decoy_pos] = decoy_dir * BACKGROUND_NORM * 3.0  # decoy has MORE mass nearby to pull the centroid

    q_full = bq.clone()
    q_full[0, 0, QUERY_POS] = e * 6.0
    z = monarch_meta_bucket_route(q_full, k_full, v_full, B=B, W_blocks=W_blocks, n_buckets=n_buckets)[0, 0, QUERY_POS]
    cos = F.cosine_similarity(z, val, dim=0).item()
    recalls.append(cos)
    if cos < 0.5:
        boundary_fail += 1

print(f"n_buckets={n_buckets}, {n_trials} trials with a mass-heavy decoy near the needle's cluster region:")
print(f"  mean cos={sum(recalls)/len(recalls):.4f}, min={min(recalls):.4f}, fail rate (cos<0.5)={boundary_fail/n_trials:.2%}")
