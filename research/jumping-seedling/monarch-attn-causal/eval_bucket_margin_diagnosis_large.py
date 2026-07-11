import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_meta_bucket_route import _kmeans_buckets

D, Dv = 16, 16
B = 4
N = 256
QUERY_POS = 128
BLOCK_SPAN = 128
BACKGROUND_NORM = 0.5 * (D ** 0.5)
sm_scale = 1 / (D ** 0.5)
N_BUCKETS = 16
N_TRIALS = 1000  # 10x the original run, specifically to get a real sample of "wrong" routes

print("=== Margin analysis at larger n (1000 vs. the original 100) ===")
print("(original run: 4/100 wrong routes -- too few to trust the characterization)")
print()

margins_correct, margins_wrong = [], []
dists_correct, dists_wrong = [], []
g = torch.Generator().manual_seed(7)
for trial in range(N_TRIALS):
    needle_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=g).item()
    gg = torch.Generator().manual_seed(2000 + trial)
    bk = torch.randn(1, 1, N, D, generator=gg) * 0.5
    e = F.normalize(torch.randn(D, generator=gg), dim=0)
    k_full = bk.clone()
    k_full[0, 0, needle_pos] = e * BACKGROUND_NORM
    k_block = k_full[:, :, :BLOCK_SPAN]
    centroids, assign = _kmeans_buckets(k_block, N_BUCKETS, 3, sm_scale)

    needle_key = k_block[0, 0, needle_pos]
    needle_bucket = assign[0, 0, needle_pos].item()

    query = e * 6.0
    sims_to_centroids = sm_scale * (query @ centroids[0, 0].T)
    ranked = sims_to_centroids.argsort(descending=True)
    top1_bucket = ranked[0].item()
    correct = (top1_bucket == needle_bucket)

    margin = (sims_to_centroids[ranked[0]] - sims_to_centroids[ranked[1]]).item()
    dist_to_own_centroid = (needle_key - centroids[0, 0, needle_bucket]).norm().item()

    if correct:
        margins_correct.append(margin)
        dists_correct.append(dist_to_own_centroid)
    else:
        margins_wrong.append(margin)
        dists_wrong.append(dist_to_own_centroid)

n_correct, n_wrong = len(margins_correct), len(margins_wrong)
print(f"routing accuracy: {n_correct}/{N_TRIALS} correct ({n_wrong} mis-routed, {n_wrong/N_TRIALS:.2%})")
print()


def stats(name, xs):
    t = torch.tensor(xs)
    print(f"{name}: n={len(xs)}, mean={t.mean().item():.4f}, std={t.std().item():.4f}, "
          f"median={t.median().item():.4f}, min={t.min().item():.4f}, max={t.max().item():.4f}")


print("--- top1-top2 centroid margin ---")
stats("correct routes", margins_correct)
stats("WRONG routes  ", margins_wrong)
print()
print("--- needle-to-OWN-centroid distance ---")
stats("correct routes", dists_correct)
stats("WRONG routes  ", dists_wrong)

g2 = torch.Generator().manual_seed(55)
bg_dists = []
for trial in range(200):
    gg = torch.Generator().manual_seed(4000 + trial)
    bk = torch.randn(1, 1, N, D, generator=gg) * 0.5
    k_block = bk[:, :, :BLOCK_SPAN]
    centroids, assign = _kmeans_buckets(k_block, N_BUCKETS, 3, sm_scale)
    pos = torch.randint(0, BLOCK_SPAN, (1,), generator=g2).item()
    d = (k_block[0, 0, pos] - centroids[0, 0, assign[0, 0, pos]]).norm().item()
    bg_dists.append(d)
print()
stats("background (typical, non-needle) key distance to own centroid", bg_dists)

print()
print("=== Verdict, at n=1000 rather than n=100 ===")
if margins_wrong and margins_correct:
    mc = sum(margins_correct) / len(margins_correct)
    mw = sum(margins_wrong) / len(margins_wrong)
    dc = sum(dists_correct) / len(dists_correct)
    dw = sum(dists_wrong) / len(dists_wrong)
    print(f"margin ratio (correct/wrong): {mc/mw:.2f}x")
    print(f"distance ratio (wrong/correct): {dw/dc:.2f}x")
    print("(original n=100 result: margin ratio ~11x, distance ratio ~1.18x --")
    print(" does the larger sample confirm margin-tightness dominates, or shift the picture?)")
