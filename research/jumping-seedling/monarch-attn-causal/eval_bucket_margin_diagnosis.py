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
N_TRIALS = 100


def make_scene(seed, needle_pos):
    g = torch.Generator().manual_seed(seed)
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    k_full = bk.clone()
    k_full[0, 0, needle_pos] = e * BACKGROUND_NORM
    return k_full, e


print("=== Margin analysis: does the MIPS/outlier-mismatch hypothesis explain baseline mis-routing? ===")
print("(hypothesis: needle keys are atypical for their bucket, so mean centroids")
print(" systematically under-rank the buckets that actually contain them)")
print()

margins_correct, margins_wrong = [], []
dists_correct, dists_wrong = [], []
g = torch.Generator().manual_seed(7)
for trial in range(N_TRIALS):
    needle_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=g).item()
    k_full, e = make_scene(seed=2000 + trial, needle_pos=needle_pos)
    k_block = k_full[:, :, :BLOCK_SPAN]  # (1,1,128,D)
    centroids, assign = _kmeans_buckets(k_block, N_BUCKETS, 3, sm_scale)  # (1,1,Nb,D), (1,1,128)

    needle_key = k_block[0, 0, needle_pos]
    needle_bucket = assign[0, 0, needle_pos].item()

    # query aligned with the needle
    query = e * 6.0
    sims_to_centroids = sm_scale * (query @ centroids[0, 0].T)  # (Nb,)
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
print(f"routing accuracy: {n_correct}/{N_TRIALS} correct ({n_wrong} mis-routed)")
print()
if margins_correct:
    print(f"top1-top2 centroid margin | correct routes: mean={sum(margins_correct)/n_correct:.4f}")
if margins_wrong:
    print(f"top1-top2 centroid margin | WRONG routes:   mean={sum(margins_wrong)/n_wrong:.4f}")
print()
if dists_correct:
    print(f"needle-to-OWN-centroid distance | correct routes: mean={sum(dists_correct)/n_correct:.4f}")
if dists_wrong:
    print(f"needle-to-OWN-centroid distance | WRONG routes:   mean={sum(dists_wrong)/n_wrong:.4f}")

# background baseline for comparison: how far is a typical (non-needle) key from its own centroid?
g2 = torch.Generator().manual_seed(55)
bg_dists = []
for trial in range(30):
    gg = torch.Generator().manual_seed(4000 + trial)
    bk = torch.randn(1, 1, N, D, generator=gg) * 0.5
    k_block = bk[:, :, :BLOCK_SPAN]
    centroids, assign = _kmeans_buckets(k_block, N_BUCKETS, 3, sm_scale)
    pos = torch.randint(0, BLOCK_SPAN, (1,), generator=g2).item()
    d = (k_block[0, 0, pos] - centroids[0, 0, assign[0, 0, pos]]).norm().item()
    bg_dists.append(d)
print()
print(f"(baseline: typical background key's distance to its own centroid, mean={sum(bg_dists)/len(bg_dists):.4f})")
