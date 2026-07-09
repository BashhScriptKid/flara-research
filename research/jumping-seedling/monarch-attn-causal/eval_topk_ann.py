import sys, time
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_causal_dual_opt import monarch_attention_causal_dual_opt_torch as causal_dual_opt
from ma_causal_topk import monarch_causal_topk as topk_exact
from ma_causal_topk_ann import monarch_causal_topk_ann as topk_ann

D, Dv = 16, 16
B = 16
N = 256
needle_pos = 18

N_PROBE = 3


def make_base(seed):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    v_needle = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    return bq, bk, bv, e, v_needle


print("=== 1. Needle-in-haystack, signal_scale=6.0 (ANN vs exact vs dual_opt) ===")
torch.manual_seed(1)
bq, bk, bv, e, v_needle = make_base(1)
distances_qp = {0: 20, 2: 48, 5: 96, 9: 160, 14: 240}
SIGNAL = 6.0
k_full = bk.clone(); k_full[0, 0, needle_pos] = e * SIGNAL
v_full = bv.clone(); v_full[0, 0, needle_pos] = v_needle
mean_v_other = torch.cat([bv[0, 0, :needle_pos], bv[0, 0, needle_pos + 1:]], dim=0).mean(dim=0)

print(f"{'query_pos':>9} {'dist':>5} | {'GT cos':>8} | {'ann k16':>8} {'ann k8':>7} | {'exact k16':>9} {'exact k8':>8} | {'dual_opt':>9} | {'mean-V':>7}")
for dist, qp in distances_qp.items():
    q_full = bq.clone(); q_full[0, 0, qp] = e * SIGNAL
    z_gt = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)[0, 0, qp]
    z_ann16 = topk_ann(q_full, k_full, v_full, B=B, topk=16, n_probe=N_PROBE)[0, 0, qp]
    z_ann8 = topk_ann(q_full, k_full, v_full, B=B, topk=8, n_probe=N_PROBE)[0, 0, qp]
    z_e16 = topk_exact(q_full, k_full, v_full, B=B, topk=16)[0, 0, qp]
    z_e8 = topk_exact(q_full, k_full, v_full, B=B, topk=8)[0, 0, qp]
    z_d = causal_dual_opt(q_full, k_full, v_full, None, T=3, B=B, pre_pad=False)[0, 0, qp]
    cos = lambda z: F.cosine_similarity(z, v_needle, dim=0).item()
    print(f"{qp:>9} {dist:>5} | {cos(z_gt):>8.4f} | {cos(z_ann16):>8.4f} {cos(z_ann8):>7.4f} | "
          f"{cos(z_e16):>9.4f} {cos(z_e8):>8.4f} | {cos(z_d):>9.4f} | {F.cosine_similarity(mean_v_other, v_needle, dim=0).item():>7.4f}")

print()
print("=== 2. Signal-strength sweep (grid: signal_scale x distance), ANN ===")
signal_scales = [6.0, 3.0, 1.5, 1.0, 0.5, 0.25]
for kk in (8, 16):
    print(f"\n-- ANN k={kk}, n_probe={N_PROBE} -- cosine recall, rows=signal_scale, cols=distance(blocks)")
    print("scale  | " + " ".join(f"d={d:>3}" for d in distances_qp))
    for scale in signal_scales:
        bq, bk, bv, e, v_needle = make_base(1)
        k_full = bk.clone(); k_full[0, 0, needle_pos] = e * scale
        v_full = bv.clone(); v_full[0, 0, needle_pos] = v_needle
        row = []
        for dist, qp in distances_qp.items():
            q_full = bq.clone(); q_full[0, 0, qp] = e * scale
            z = topk_ann(q_full, k_full, v_full, B=B, topk=kk, n_probe=N_PROBE)[0, 0, qp]
            row.append(F.cosine_similarity(z, v_needle, dim=0).item())
        print(f"{scale:>6.2f} | " + " ".join(f"{c:>5.2f}" for c in row))

print()
print("=== 3. Decoy pressure (fixed needle signal_scale=3.0), ANN ===")
NEEDLE_SCALE = 3.0
qp_fixed = 240
for kk in (8, 16):
    print(f"\n-- ANN k={kk}, n_probe={N_PROBE}, needle scale={NEEDLE_SCALE}, query_pos={qp_fixed} --")
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
            z = topk_ann(q_full, k_full, v_full, B=B, topk=kk, n_probe=N_PROBE)[0, 0, qp_fixed]
            cos_vals.append(F.cosine_similarity(z, v_needle, dim=0).item())
        print(f"{num_decoys:>10} | {sum(cos_vals) / len(cos_vals):>10.4f}")

print()
print("=== 4. Wall-clock cost, ANN vs exact top-k vs dual_opt ===")
print("(NOTE: this ANN reference implementation still computes the full B x P")
print(" score matrix and masks it -- see ma_causal_topk_ann.py docstring. These")
print(" numbers show clustering OVERHEAD added on top of exact scoring cost, not")
print(" the speed a real ANN index -- which skips scoring unprobed points -- would give.)")

def bench(fn, *args, reps=3, warmup=1):
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args)
        return (time.perf_counter() - t0) / reps

print(f"{'N':>5} | {'dual_opt':>9} | {'exact16':>9} {'exact8':>8} | {'ann16':>9} {'ann8':>8} | {'ann16/exact16':>13}")
for N2 in (256, 512, 1024, 2048):
    q = torch.randn(1, 8, N2, 64); k = torch.randn(1, 8, N2, 64); v = torch.randn(1, 8, N2, 64)
    reps = 3 if N2 <= 1024 else 2
    t_d = bench(causal_dual_opt, q, k, v, None, 3, B, False, reps=reps)
    t_e16 = bench(topk_exact, q, k, v, B, 16, reps=reps)
    t_e8 = bench(topk_exact, q, k, v, B, 8, reps=reps)
    t_a16 = bench(topk_ann, q, k, v, B, 16, N_PROBE, reps=reps)
    t_a8 = bench(topk_ann, q, k, v, B, 8, N_PROBE, reps=reps)
    print(f"{N2:>5} | {t_d*1000:>8.1f}ms | {t_e16*1000:>8.1f}ms {t_e8*1000:>7.1f}ms | "
          f"{t_a16*1000:>8.1f}ms {t_a8*1000:>7.1f}ms | {t_a16/t_e16:>13.2f}")
