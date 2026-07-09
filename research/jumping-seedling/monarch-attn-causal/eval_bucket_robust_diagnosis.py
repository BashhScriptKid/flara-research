import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_meta_bucket_route import _kmeans_buckets
from ma_meta_bucket_route_robust import _kmeans_buckets_robust

D, Dv = 16, 16
N = 256
BLOCK_SPAN = 128
BACKGROUND_NORM = 0.5 * (D ** 0.5)
n_buckets = 8
sm_scale = 1 / (D ** 0.5)
n_trials = 30

print("=== Diagnosis: is the bottleneck ROUTING (wrong bucket chosen) or")
print("    IN-BUCKET COMPETITION (right bucket, but decoy outscores needle within it)? ===")
print()

for name, kmeans_fn, kwargs in [
    ("arithmetic-mean", _kmeans_buckets, {}),
    ("geometric-median", _kmeans_buckets_robust, {"weiszfeld_iters": 5}),
]:
    routing_correct = 0
    same_bucket_as_decoy = 0
    for trial in range(n_trials):
        seed = 3000 + trial
        gg = torch.Generator().manual_seed(seed)
        bk = torch.randn(1, 1, N, D, generator=gg) * 0.5
        e = F.normalize(torch.randn(D, generator=gg), dim=0)
        needle_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=gg).item()
        decoy_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=gg).item()
        while decoy_pos == needle_pos:
            decoy_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=gg).item()
        decoy_dir = F.normalize(e + 0.3 * torch.randn(D, generator=gg), dim=0)

        k_full = bk.clone()
        k_full[0, 0, needle_pos] = e * BACKGROUND_NORM
        k_full[0, 0, decoy_pos] = decoy_dir * BACKGROUND_NORM * 3.0
        k_block = k_full[:, :, :BLOCK_SPAN]

        centroids, assign = kmeans_fn(k_block, n_buckets, 3, sm_scale, **kwargs)

        sims = sm_scale * ((e * 6.0) @ centroids[0, 0].T)  # (n_buckets,)
        chosen_bucket = sims.argmax().item()
        needle_bucket = assign[0, 0, needle_pos].item()
        decoy_bucket = assign[0, 0, decoy_pos].item()

        if chosen_bucket == needle_bucket:
            routing_correct += 1
        if needle_bucket == decoy_bucket:
            same_bucket_as_decoy += 1

    print(f"-- {name} --")
    print(f"  routing correctly finds needle's bucket: {routing_correct}/{n_trials} ({routing_correct/n_trials:.1%})")
    print(f"  needle and decoy land in the SAME bucket: {same_bucket_as_decoy}/{n_trials} ({same_bucket_as_decoy/n_trials:.1%})")
    print()

print("Interpretation: if 'routing correct' is already high (~needle's bucket IS chosen)")
print("but final recall was still bad (from the previous test), the bottleneck is")
print("IN-BUCKET exact-attention competition (decoy legitimately outscoring the needle")
print("within a correctly-selected bucket), not centroid-based routing at all -- a")
print("different problem than 'centroid capture', closer to the top-k decoy-pressure")
print("cliff found earlier tonight, just relocated inside one bucket's exact attention.")
