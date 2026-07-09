import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_sliding_monarch import sliding_monarch_causal as sma

D, Dv = 16, 16
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


print("=== A. Window width (tokens) held constant, B varied -- does Monarch block")
print("       granularity matter independently of window width? ===")
combos_64 = [("B=16,Wb=4", 16, 4), ("B=8,Wb=8", 8, 8), ("B=32,Wb=2", 32, 2)]
combos_128 = [("B=16,Wb=8", 16, 8), ("B=8,Wb=16", 8, 16), ("B=32,Wb=4", 32, 4)]

print("\n-- Aggregate quality, window=64 tokens --")
print(f"{'N':>5} | " + " ".join(f"{name:>12}" for name, _, _ in combos_64))
for N2 in (64, 256, 1024):
    q = torch.randn(1, 1, N2, D); k = torch.randn(1, 1, N2, D); v = torch.randn(1, 1, N2, Dv)
    z_gt = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    coses = []
    for name, Bv, Wb in combos_64:
        z = sma(q, k, v, B=Bv, W_blocks=Wb, T=3)
        coses.append(F.cosine_similarity(z.flatten(), z_gt.flatten(), dim=0).item())
    print(f"{N2:>5} | " + " ".join(f"{c:>12.4f}" for c in coses))

print("\n-- Aggregate quality, window=128 tokens --")
print(f"{'N':>5} | " + " ".join(f"{name:>13}" for name, _, _ in combos_128))
for N2 in (256, 1024):
    q = torch.randn(1, 1, N2, D); k = torch.randn(1, 1, N2, D); v = torch.randn(1, 1, N2, Dv)
    z_gt = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    coses = []
    for name, Bv, Wb in combos_128:
        z = sma(q, k, v, B=Bv, W_blocks=Wb, T=3)
        coses.append(F.cosine_similarity(z.flatten(), z_gt.flatten(), dim=0).item())
    print(f"{N2:>5} | " + " ".join(f"{c:>13.4f}" for c in coses))

print("\n-- Needle test, signal_scale=6.0, window=64 tokens --")
distances_qp = {2: 48, 5: 96, 9: 160, 14: 240}
bq, bk, bv, e, v_needle = make_base(1)
k_full = bk.clone(); k_full[0, 0, needle_pos] = e * 6.0
v_full = bv.clone(); v_full[0, 0, needle_pos] = v_needle
print(f"{'dist':>5} | " + " ".join(f"{name:>12}" for name, _, _ in combos_64))
for dist, qp in distances_qp.items():
    q_full = bq.clone(); q_full[0, 0, qp] = e * 6.0
    cos = lambda z: F.cosine_similarity(z, v_needle, dim=0).item()
    row = []
    for name, Bv, Wb in combos_64:
        z = sma(q_full, k_full, v_full, B=Bv, W_blocks=Wb, T=3)[0, 0, qp]
        row.append(cos(z))
    print(f"{dist:>5} | " + " ".join(f"{c:>12.4f}" for c in row))

print()
print("=== B. T-sensitivity for the far branch (B=16, W_blocks=4 fixed) ===")
print("\n-- Needle test, signal_scale=6.0 --")
print(f"{'dist':>5} | " + " ".join(f"T={t:>5}" for t in (1, 2, 3, 5, 8)))
for dist, qp in distances_qp.items():
    q_full = bq.clone(); q_full[0, 0, qp] = e * 6.0
    cos = lambda z: F.cosine_similarity(z, v_needle, dim=0).item()
    row = []
    for t in (1, 2, 3, 5, 8):
        z = sma(q_full, k_full, v_full, B=16, W_blocks=4, T=t)[0, 0, qp]
        row.append(cos(z))
    print(f"{dist:>5} | " + " ".join(f"{c:>7.4f}" for c in row))

print("\n-- Aggregate quality --")
print(f"{'N':>5} | " + " ".join(f"T={t:>5}" for t in (1, 2, 3, 5, 8)))
for N2 in (64, 256, 1024):
    q = torch.randn(1, 1, N2, D); k = torch.randn(1, 1, N2, D); v = torch.randn(1, 1, N2, Dv)
    z_gt = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    coses = []
    for t in (1, 2, 3, 5, 8):
        z = sma(q, k, v, B=16, W_blocks=4, T=t)
        coses.append(F.cosine_similarity(z.flatten(), z_gt.flatten(), dim=0).item())
    print(f"{N2:>5} | " + " ".join(f"{c:>7.4f}" for c in coses))
