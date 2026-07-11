import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_meta_bucket_route import _kmeans_buckets, monarch_meta_bucket_route

D, Dv = 16, 16
B = 4
N = 512
BLOCK_SPAN = 256  # the scale where probe 23 found instability
QUERY_POS = 256
W_blocks = 1
BACKGROUND_NORM = 0.5 * (D ** 0.5)
sm_scale = 1 / (D ** 0.5)

print("=== Merged sweep: bucket-occupancy stability floor + adversarial-exposure vs bucket size ===")
print(f"(BLOCK_SPAN={BLOCK_SPAN}, the scale where probe 23 found nb=64 unstable vs nb=16)")
print()
print(f"{'n_buckets':>10} {'avg occ':>8} | {'occ min/max':>13} {'occ std':>8} | "
      f"{'natural recall':>15} | {'adversarial fail rate':>22}")

n_buckets_sweep = (4, 8, 16, 32, 64, 128)

for n_buckets in n_buckets_sweep:
    # --- stability: bucket occupancy distribution over several random blocks ---
    occ_means, occ_mins, occ_maxs, occ_stds = [], [], [], []
    for trial in range(10):
        gg = torch.Generator().manual_seed(5000 + trial)
        bk = torch.randn(1, 1, N, D, generator=gg) * 0.5
        k_block = bk[:, :, :BLOCK_SPAN]
        _, assign = _kmeans_buckets(k_block, n_buckets, 3, sm_scale)
        counts = torch.bincount(assign[0, 0], minlength=n_buckets).float()
        occ_mins.append(counts.min().item())
        occ_maxs.append(counts.max().item())
        occ_stds.append(counts.std().item())
    avg_occ = BLOCK_SPAN / n_buckets

    # --- natural single-needle recall (no decoy, uncontested) ---
    natural_recalls = []
    for trial in range(10):
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
    mean_natural = sum(natural_recalls) / len(natural_recalls)

    # --- adversarial exposure: same mass-heavy-decoy construction, fail rate vs bucket size ---
    fails = 0
    n_adv_trials = 20
    for trial in range(n_adv_trials):
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
        z = monarch_meta_bucket_route(q_full, k_full, v_full, B=B, W_blocks=W_blocks, n_buckets=n_buckets)[0, 0, QUERY_POS]
        cos = F.cosine_similarity(z, val, dim=0).item()
        if cos < 0.5:
            fails += 1
    fail_rate = fails / n_adv_trials

    print(f"{n_buckets:>10} {avg_occ:>8.1f} | "
          f"{sum(occ_mins)/len(occ_mins):>5.1f}/{sum(occ_maxs)/len(occ_maxs):>5.1f} "
          f"{sum(occ_stds)/len(occ_stds):>8.2f} | "
          f"{mean_natural:>15.4f} | {fail_rate:>22.2%}")

print()
print("Reading the table: 'stability floor' = where occ std blows up relative to avg occ")
print("(probe 23 found nb=64 unstable at avg_occ~4 -- does natural recall or adversarial")
print("fail rate visibly degrade at/below that same occupancy?); 'exposure curve' = does")
print("adversarial fail rate actually decrease as n_buckets grows (smaller buckets),")
print("as Fable's exposure-reduction mechanism predicts?")
