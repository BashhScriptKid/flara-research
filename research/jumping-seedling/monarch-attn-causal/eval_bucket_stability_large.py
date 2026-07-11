import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_meta_bucket_route import _kmeans_buckets, monarch_meta_bucket_route

D, Dv = 16, 16
B = 4
N = 512
BLOCK_SPAN = 256
QUERY_POS = 256
W_blocks = 1
BACKGROUND_NORM = 0.5 * (D ** 0.5)
sm_scale = 1 / (D ** 0.5)
N_TRIALS = 50

print("=== Stability-only sweep, n=50 (vs. the earlier merged sweep's n=10) ===")
print(f"(BLOCK_SPAN={BLOCK_SPAN}, isolating natural recall + occupancy from adversarial exposure)")
print()
print(f"{'n_buckets':>10} {'avg occ':>8} | {'occ std (mean)':>15} | {'natural recall mean':>20} {'std':>7} {'min':>7}")

n_buckets_sweep = (4, 8, 16, 32, 64, 128)

for n_buckets in n_buckets_sweep:
    occ_stds = []
    for trial in range(N_TRIALS):
        gg = torch.Generator().manual_seed(5000 + trial)
        bk = torch.randn(1, 1, N, D, generator=gg) * 0.5
        k_block = bk[:, :, :BLOCK_SPAN]
        _, assign = _kmeans_buckets(k_block, n_buckets, 3, sm_scale)
        counts = torch.bincount(assign[0, 0], minlength=n_buckets).float()
        occ_stds.append(counts.std().item())

    natural_recalls = []
    for trial in range(N_TRIALS):
        gg = torch.Generator().manual_seed(6000 + trial)
        bq = torch.randn(1, 1, N, D, generator=gg) * 0.5
        bk = torch.randn(1, 1, N, D, generator=gg) * 0.5
        bv = torch.randn(1, 1, N, Dv, generator=gg) * 0.5
        e = F.normalize(torch.randn(D, generator=gg), dim=0)
        val = F.normalize(torch.randn(Dv, generator=gg), dim=0) * 5.0
        needle_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=gg).item()
        k_full = bk.clone(); k_full[0, 0, needle_pos] = e * BACKGROUND_NORM
        v_full = bv.clone(); v_full[0, 0, needle_pos] = val
        q_full = bq.clone(); q_full[0, 0, QUERY_POS] = e * 6.0
        z = monarch_meta_bucket_route(q_full, k_full, v_full, B=B, W_blocks=W_blocks, n_buckets=n_buckets)[0, 0, QUERY_POS]
        natural_recalls.append(F.cosine_similarity(z, val, dim=0).item())

    t = torch.tensor(natural_recalls)
    avg_occ = BLOCK_SPAN / n_buckets
    print(f"{n_buckets:>10} {avg_occ:>8.1f} | {sum(occ_stds)/len(occ_stds):>15.2f} | "
          f"{t.mean().item():>20.4f} {t.std().item():>7.4f} {t.min().item():>7.4f}")

print()
print("If natural recall degrades meaningfully (mean drop or std spike) at small")
print("occupancy (nb=64/128), that confirms probe 23's original instability finding")
print("at real sample size. If it stays flat, probe 23's original read was itself noise.")
