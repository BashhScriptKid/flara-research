import sys, time
sys.path.insert(0, "repo")
import torch

from ma.ma_torch import monarch_attention_torch as orig_noncausal
from ma_causal_dual_opt import monarch_attention_causal_dual_opt_torch as causal_dual_opt
from ma_causal_topk import monarch_causal_topk as topk_hybrid

torch.manual_seed(0)
B, T = 16, 3

def bench(fn, *args, reps=5, warmup=2):
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args)
        return (time.perf_counter() - t0) / reps

print(f"{'N':>5} | {'noncausal':>10} {'dual_opt':>10} | {'topk8':>10} {'topk16':>10} {'topk32':>10} {'topk64':>10} | {'topk16/dual_opt':>16}")
for N in (256, 512, 1024, 2048):
    q = torch.randn(1, 8, N, 64); k = torch.randn(1, 8, N, 64); v = torch.randn(1, 8, N, 64)
    t_nc = bench(orig_noncausal, q, k, v, None, T, B, False)
    t_d = bench(causal_dual_opt, q, k, v, None, T, B, False)
    t_topk = {}
    for kk in (8, 16, 32, 64):
        reps = 5 if N <= 1024 else 2
        t_topk[kk] = bench(topk_hybrid, q, k, v, B, kk, reps=reps)
    print(f"{N:>5} | {t_nc*1000:>9.1f}ms {t_d*1000:>9.1f}ms | "
          f"{t_topk[8]*1000:>9.1f}ms {t_topk[16]*1000:>9.1f}ms {t_topk[32]*1000:>9.1f}ms {t_topk[64]*1000:>9.1f}ms | "
          f"{t_topk[16]/t_d:>16.2f}")

print()
print("=== scaling check: does topk16 grow like O(N^2/B) as predicted? ===")
import math
prev = None
for N in (256, 512, 1024, 2048):
    q = torch.randn(1, 8, N, 64); k = torch.randn(1, 8, N, 64); v = torch.randn(1, 8, N, 64)
    t = bench(topk_hybrid, q, k, v, B, 16, reps=3 if N <= 1024 else 2)
    ratio = (t / prev) if prev else float("nan")
    print(f"N={N:>5}: {t*1000:>8.1f}ms  (x{ratio:.2f} vs previous N, expect ~4x for O(N^2))")
    prev = t
