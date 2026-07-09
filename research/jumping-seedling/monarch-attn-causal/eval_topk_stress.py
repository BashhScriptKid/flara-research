import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_causal_topk import monarch_causal_topk as topk_hybrid

D, Dv = 16, 16
B = 16
N = 256
needle_pos = 18  # block 1


def make_base(seed):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    v_needle = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    return bq, bk, bv, e, v_needle


print("=== 1. Signal-strength sweep (grid: signal_scale x distance) ===")
distances_qp = {0: 20, 2: 48, 5: 96, 9: 160, 14: 240}
signal_scales = [6.0, 3.0, 1.5, 1.0, 0.5, 0.25]

for kk in (8, 16):
    print(f"\n-- k={kk} -- cosine recall vs needle value, rows=signal_scale, cols=distance(blocks)")
    header = "scale  | " + " ".join(f"d={d:>3}" for d in distances_qp)
    print(header)
    for scale in signal_scales:
        bq, bk, bv, e, v_needle = make_base(1)
        k_full = bk.clone()
        k_full[0, 0, needle_pos] = e * scale
        v_full = bv.clone()
        v_full[0, 0, needle_pos] = v_needle
        row = []
        for dist, qp in distances_qp.items():
            q_full = bq.clone()
            q_full[0, 0, qp] = e * scale
            z = topk_hybrid(q_full, k_full, v_full, B=B, topk=kk)[0, 0, qp]
            cos = F.cosine_similarity(z, v_needle, dim=0).item()
            row.append(cos)
        print(f"{scale:>6.2f} | " + " ".join(f"{c:>5.2f}" for c in row))

print()
print("=== 2. Distractor pressure (fixed needle signal_scale, varying decoy count) ===")
NEEDLE_SCALE = 3.0
qp_fixed = 240  # hardest distance, pool=240
for kk in (8, 16):
    print(f"\n-- k={kk}, needle signal_scale={NEEDLE_SCALE} (fixed), query_pos={qp_fixed} --")
    print(f"{'num_decoys':>10} | {'cos recall':>10}   (mean over 10 trials, decoy_scale ~ U[0.9,1.3]*needle_scale)")
    for num_decoys in (0, 5, 20, 50):
        cos_vals = []
        for trial in range(10):
            g = torch.Generator().manual_seed(1000 + trial)
            bq, bk, bv, e, v_needle = make_base(1)
            k_full = bk.clone()
            k_full[0, 0, needle_pos] = e * NEEDLE_SCALE
            v_full = bv.clone()
            v_full[0, 0, needle_pos] = v_needle

            if num_decoys > 0:
                decoy_positions = torch.randperm(qp_fixed - 1, generator=g)[:num_decoys]
                decoy_positions = decoy_positions[decoy_positions != needle_pos]
                decoy_scales = 0.9 + 0.4 * torch.rand(len(decoy_positions), generator=g)
                for pos, dscale in zip(decoy_positions.tolist(), decoy_scales.tolist()):
                    k_full[0, 0, pos] = e * (NEEDLE_SCALE * dscale)
                    # decoy VALUE is unrelated background, not v_needle -- if a
                    # decoy steals a top-k slot, recovered cos should drop
                    # since the model retrieves the decoy's unrelated value.

            q_full = bq.clone()
            q_full[0, 0, qp_fixed] = e * NEEDLE_SCALE
            z = topk_hybrid(q_full, k_full, v_full, B=B, topk=kk)[0, 0, qp_fixed]
            cos_vals.append(F.cosine_similarity(z, v_needle, dim=0).item())
        mean_cos = sum(cos_vals) / len(cos_vals)
        print(f"{num_decoys:>10} | {mean_cos:>10.4f}")

print()
print("=== 3. k sensitivity at a found-weak signal_scale ===")
WEAK_SCALE = 1.0  # pick based on part 1 results
for kk in (8, 16, 32, 64):
    bq, bk, bv, e, v_needle = make_base(1)
    k_full = bk.clone()
    k_full[0, 0, needle_pos] = e * WEAK_SCALE
    v_full = bv.clone()
    v_full[0, 0, needle_pos] = v_needle
    q_full = bq.clone()
    q_full[0, 0, 240] = e * WEAK_SCALE
    z = topk_hybrid(q_full, k_full, v_full, B=B, topk=kk)[0, 0, 240]
    cos = F.cosine_similarity(z, v_needle, dim=0).item()
    print(f"k={kk:>3}: signal_scale={WEAK_SCALE}, dist=14 blocks -> cos recall = {cos:.4f}")

print()
print("=== 3b. aggregate quality cost of larger k (random Q/K/V, no needle) ===")
print(f"{'N':>5} | " + " ".join(f"k={kk:>2} cos" for kk in (8, 16, 32, 64)))
for N2 in (64, 256, 1024):
    E, H = 1, 1
    q = torch.randn(E, H, N2, D)
    k = torch.randn(E, H, N2, D)
    v = torch.randn(E, H, N2, Dv)
    z_gt = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    coses = []
    for kk in (8, 16, 32, 64):
        z = topk_hybrid(q, k, v, B=B, topk=kk)
        cos = F.cosine_similarity(z.flatten(), z_gt.flatten(), dim=0).item()
        coses.append(cos)
    print(f"{N2:>5} | " + " ".join(f"{c:>7.4f}" for c in coses))
