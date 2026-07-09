import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

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


print("=== 1. Needle-in-haystack, signal_scale=6.0: W_refine=W_blocks (old) vs W_refine=0 (new) ===")
distances_qp = {0: 20, 2: 48, 5: 96, 9: 160, 14: 240}
SIGNAL = 6.0
bq, bk, bv, e, v_needle = make_base(1)
k_full = bk.clone(); k_full[0, 0, needle_pos] = e * SIGNAL
v_full = bv.clone(); v_full[0, 0, needle_pos] = v_needle

for W in (2, 4):
    print(f"\n-- W_blocks={W} --")
    print(f"{'dist':>5} | {'GT':>7} | {'old(Wr=W)':>10} {'new(Wr=0)':>10}")
    for dist, qp in distances_qp.items():
        q_full = bq.clone(); q_full[0, 0, qp] = e * SIGNAL
        z_gt = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)[0, 0, qp]
        cos = lambda z: F.cosine_similarity(z, v_needle, dim=0).item()
        z_old = sma(q_full, k_full, v_full, B=B, W_blocks=W, T=3, W_refine=W)[0, 0, qp]
        z_new = sma(q_full, k_full, v_full, B=B, W_blocks=W, T=3, W_refine=0)[0, 0, qp]
        print(f"{dist:>5} | {cos(z_gt):>7.4f} | {cos(z_old):>10.4f} {cos(z_new):>10.4f}")

print()
print("=== 2. Aggregate quality (random Q/K/V) ===")
print(f"{'N':>5} | {'W=2 old':>9} {'W=2 new':>9} | {'W=4 old':>9} {'W=4 new':>9}")
for N2 in (64, 256, 1024):
    E, H = 1, 1
    q = torch.randn(E, H, N2, D)
    k = torch.randn(E, H, N2, D)
    v = torch.randn(E, H, N2, Dv)
    z_gt = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    cos = lambda z: F.cosine_similarity(z.flatten(), z_gt.flatten(), dim=0).item()
    z_2o = sma(q, k, v, B=B, W_blocks=2, T=3, W_refine=2)
    z_2n = sma(q, k, v, B=B, W_blocks=2, T=3, W_refine=0)
    z_4o = sma(q, k, v, B=B, W_blocks=4, T=3, W_refine=4)
    z_4n = sma(q, k, v, B=B, W_blocks=4, T=3, W_refine=0)
    print(f"{N2:>5} | {cos(z_2o):>9.4f} {cos(z_2n):>9.4f} | {cos(z_4o):>9.4f} {cos(z_4n):>9.4f}")
