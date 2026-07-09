import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_causal_dual_opt import monarch_attention_causal_dual_opt_torch as causal_dual_opt
from ma_sliding_monarch import sliding_monarch_causal as sma

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

print(f"{'dist':>5} | {'GT':>7} | {'W=1':>7} {'W=2':>7} {'W=4':>7} {'W=8':>7} | {'dual_opt':>8}")
for dist, qp in distances_qp.items():
    q_full = bq.clone(); q_full[0, 0, qp] = e * SIGNAL
    z_gt = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)[0, 0, qp]
    cos = lambda z: F.cosine_similarity(z, v_needle, dim=0).item()
    z_ws = [sma(q_full, k_full, v_full, B=B, W_blocks=w, T=3)[0, 0, qp] for w in (1, 2, 4, 8)]
    z_d = causal_dual_opt(q_full, k_full, v_full, None, T=3, B=B, pre_pad=False)[0, 0, qp]
    print(f"{dist:>5} | {cos(z_gt):>7.4f} | " + " ".join(f"{cos(z):>7.4f}" for z in z_ws) + f" | {cos(z_d):>8.4f}")

print()
print("=== 2. Signal-strength sweep at W=4, hardest distance (d=14) ===")
signal_scales = [6.0, 3.0, 1.5, 1.0, 0.5, 0.25]
for scale in signal_scales:
    bq, bk, bv, e, v_needle = make_base(1)
    k_full = bk.clone(); k_full[0, 0, needle_pos] = e * scale
    v_full = bv.clone(); v_full[0, 0, needle_pos] = v_needle
    q_full = bq.clone(); q_full[0, 0, 240] = e * scale
    z_gt = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)[0, 0, 240]
    z_w4 = sma(q_full, k_full, v_full, B=B, W_blocks=4, T=3)[0, 0, 240]
    cos_gt = F.cosine_similarity(z_gt, v_needle, dim=0).item()
    cos_w4 = F.cosine_similarity(z_w4, v_needle, dim=0).item()
    print(f"scale={scale:>5.2f} | GT={cos_gt:.4f} | W=4 cos={cos_w4:.4f}")

print()
print("=== 3. Aggregate quality vs exact causal softmax (random Q/K/V) ===")
print(f"{'N':>5} | {'W=1':>9} {'W=2':>9} {'W=4':>9} {'W=8':>9} | {'dual_opt':>9}")
for N2 in (64, 256, 1024):
    E, H = 1, 1
    q = torch.randn(E, H, N2, D)
    k = torch.randn(E, H, N2, D)
    v = torch.randn(E, H, N2, Dv)
    z_gt = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    coses = []
    for w in (1, 2, 4, 8):
        z = sma(q, k, v, B=B, W_blocks=w, T=3)
        coses.append(F.cosine_similarity(z.flatten(), z_gt.flatten(), dim=0).item())
    z_d = causal_dual_opt(q, k, v, None, T=3, B=B, pre_pad=False)
    cos_d = F.cosine_similarity(z_d.flatten(), z_gt.flatten(), dim=0).item()
    print(f"{N2:>5} | " + " ".join(f"{c:>9.4f}" for c in coses) + f" | {cos_d:>9.4f}")
