import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_causal_multislot import monarch_causal_multislot as multislot
from ma_causal_linear_hybrid import monarch_causal_linear_hybrid as single_slot_hybrid

D, Dv = 16, 16
B = 16
N = 256
needle_pos = 18


def make_base(seed):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    v_needle = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    return bq, bk, bv, e, v_needle


print("=== 1. Needle-in-haystack, signal_scale=6.0 ===")
distances_qp = {0: 20, 2: 48, 5: 96, 9: 160, 14: 240}
SIGNAL = 6.0
bq, bk, bv, e, v_needle = make_base(1)
k_full = bk.clone(); k_full[0, 0, needle_pos] = e * SIGNAL
v_full = bv.clone(); v_full[0, 0, needle_pos] = v_needle

print(f"{'dist':>5} | {'GT':>7} | {'1slot':>7} {'4slot':>7} {'8slot':>7} {'16slot':>7}")
for dist, qp in distances_qp.items():
    q_full = bq.clone(); q_full[0, 0, qp] = e * SIGNAL
    z_gt = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)[0, 0, qp]
    cos = lambda z: F.cosine_similarity(z, v_needle, dim=0).item()
    z_1 = single_slot_hybrid(q_full, k_full, v_full, B=B)[0, 0, qp]
    z_ns = [multislot(q_full, k_full, v_full, B=B, n_slots=ns)[0, 0, qp] for ns in (4, 8, 16)]
    print(f"{dist:>5} | {cos(z_gt):>7.4f} | {cos(z_1):>7.4f} " + " ".join(f"{cos(z):>7.4f}" for z in z_ns))

print()
print("=== 2. Signal-strength sweep at n_slots=16, hardest distance (d=14) ===")
signal_scales = [6.0, 3.0, 1.5, 1.0, 0.5, 0.25]
for scale in signal_scales:
    bq, bk, bv, e, v_needle = make_base(1)
    k_full = bk.clone(); k_full[0, 0, needle_pos] = e * scale
    v_full = bv.clone(); v_full[0, 0, needle_pos] = v_needle
    q_full = bq.clone(); q_full[0, 0, 240] = e * scale
    z_gt = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)[0, 0, 240]
    z_16 = multislot(q_full, k_full, v_full, B=B, n_slots=16)[0, 0, 240]
    cos_gt = F.cosine_similarity(z_gt, v_needle, dim=0).item()
    cos_16 = F.cosine_similarity(z_16, v_needle, dim=0).item()
    print(f"scale={scale:>5.2f} | GT={cos_gt:.4f} | 16slot cos={cos_16:.4f}")

print()
print("=== 3. Decoy pressure (needle scale=3.0 fixed, query_pos=240) ===")
NEEDLE_SCALE = 3.0
qp_fixed = 240
for ns in (4, 8, 16):
    print(f"\n-- n_slots={ns} --")
    print(f"{'num_decoys':>10} | {'cos recall':>10}   (mean over 10 trials)")
    for num_decoys in (0, 5, 20, 50):
        cos_vals = []
        for trial in range(10):
            g = torch.Generator().manual_seed(1000 + trial)
            bq, bk, bv, e, v_needle = make_base(1)
            k_full = bk.clone(); k_full[0, 0, needle_pos] = e * NEEDLE_SCALE
            v_full = bv.clone(); v_full[0, 0, needle_pos] = v_needle
            if num_decoys > 0:
                decoy_positions = torch.randperm(qp_fixed - 1, generator=g)[:num_decoys]
                decoy_positions = decoy_positions[decoy_positions != needle_pos]
                decoy_scales = 0.9 + 0.4 * torch.rand(len(decoy_positions), generator=g)
                for pos, dscale in zip(decoy_positions.tolist(), decoy_scales.tolist()):
                    k_full[0, 0, pos] = e * (NEEDLE_SCALE * dscale)
            q_full = bq.clone(); q_full[0, 0, qp_fixed] = e * NEEDLE_SCALE
            z = multislot(q_full, k_full, v_full, B=B, n_slots=ns)[0, 0, qp_fixed]
            cos_vals.append(F.cosine_similarity(z, v_needle, dim=0).item())
        print(f"{num_decoys:>10} | {sum(cos_vals) / len(cos_vals):>10.4f}")

print()
print("=== 4. Aggregate quality (random Q/K/V) ===")
print(f"{'N':>5} | {'1slot':>9} {'4slot':>9} {'8slot':>9} {'16slot':>9}")
for N2 in (64, 256, 1024):
    E, H = 1, 1
    q = torch.randn(E, H, N2, D)
    k = torch.randn(E, H, N2, D)
    v = torch.randn(E, H, N2, Dv)
    z_gt = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    z_1 = single_slot_hybrid(q, k, v, B=B)
    cos_1 = F.cosine_similarity(z_1.flatten(), z_gt.flatten(), dim=0).item()
    coses = [cos_1]
    for ns in (4, 8, 16):
        z = multislot(q, k, v, B=B, n_slots=ns)
        coses.append(F.cosine_similarity(z.flatten(), z_gt.flatten(), dim=0).item())
    print(f"{N2:>5} | " + " ".join(f"{c:>9.4f}" for c in coses))
