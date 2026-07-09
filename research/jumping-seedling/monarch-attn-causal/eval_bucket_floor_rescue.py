import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_meta_bucket_route_floor import monarch_meta_bucket_route_floor

D, Dv = 16, 16
B = 4
N = 256
QUERY_POS = 128
BLOCK_SPAN = 128
W_blocks = 1
BACKGROUND_NORM = 0.5 * (D ** 0.5)
n_buckets = 8
n_trials = 30

print("=== Floor-read rescue test: same adversarial decoy setup that gave 83.33% failure ===")
print("(mass-heavy decoy deliberately placed to capture the local centroid)")
print()
print(f"{'use_floor':>10} | {'mean cos':>9} {'min cos':>8} {'fail rate (cos<0.5)':>20} {'true-miss rate (cos<0.0)':>25}")

for use_floor in (False, True):
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
        z = monarch_meta_bucket_route_floor(
            q_full, k_full, v_full, B=B, W_blocks=W_blocks, n_buckets=n_buckets, use_floor=use_floor
        )[0, 0, QUERY_POS]
        cos = F.cosine_similarity(z, val, dim=0).item()
        recalls.append(cos)
        if cos < 0.5:
            fails += 1
        if cos < 0.0:
            true_misses += 1

    mean_r, min_r = sum(recalls) / len(recalls), min(recalls)
    print(f"{str(use_floor):>10} | {mean_r:>9.4f} {min_r:>8.4f} {fails/n_trials:>19.2%} {true_misses/n_trials:>24.2%}")

print()
print("Interpretation: if use_floor=True shows a HIGHER mean cos and LOWER true-miss")
print("rate than use_floor=False, even if fail rate (cos<0.5) stays similar, that's")
print("the FAILURE-vs-DILUTION distinction Fable's rescue test was designed to check --")
print("floor read converts total misses into diluted-but-nonzero recall.")
