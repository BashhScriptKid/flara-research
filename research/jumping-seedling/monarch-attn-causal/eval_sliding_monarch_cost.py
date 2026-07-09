import sys, time
sys.path.insert(0, "repo")
import torch

from ma.ma_torch import monarch_attention_torch as orig_noncausal
from ma_causal_dual_opt import monarch_attention_causal_dual_opt_torch as causal_dual_opt
from ma_sliding_monarch import sliding_monarch_causal as sma

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

print(f"{'N':>5} | {'noncausal':>10} {'dual_opt':>9} | {'W=1':>8} {'W=2':>8} {'W=4':>8} {'W=8':>8} | {'W=4/dual_opt':>12}")
for N in (256, 512, 1024, 2048):
    q = torch.randn(1, 8, N, 64); k = torch.randn(1, 8, N, 64); v = torch.randn(1, 8, N, 64)
    reps = 5 if N <= 1024 else 3
    t_nc = bench(orig_noncausal, q, k, v, None, T, B, False, reps=reps)
    t_d = bench(causal_dual_opt, q, k, v, None, T, B, False, reps=reps)
    t_w = {}
    for w in (1, 2, 4, 8):
        t_w[w] = bench(sma, q, k, v, B, w, T, reps=reps)
    print(f"{N:>5} | {t_nc*1000:>9.1f}ms {t_d*1000:>8.1f}ms | "
          f"{t_w[1]*1000:>7.1f}ms {t_w[2]*1000:>7.1f}ms {t_w[4]*1000:>7.1f}ms {t_w[8]*1000:>7.1f}ms | "
          f"{t_w[4]/t_d:>12.2f}")

print()
print("=== scaling check: does W=4 grow like O(N^1.5) (Monarch-like) or O(N^2)? ===")
prev = None
for N in (256, 512, 1024, 2048):
    q = torch.randn(1, 8, N, 64); k = torch.randn(1, 8, N, 64); v = torch.randn(1, 8, N, 64)
    t = bench(sma, q, k, v, B, 4, T, reps=3 if N <= 1024 else 2)
    ratio = (t / prev) if prev else float("nan")
    print(f"N={N:>5}: {t*1000:>8.1f}ms  (x{ratio:.2f} vs previous N; O(N^1.5)~2.8x, O(N^2)~4x per doubling)")
    prev = t
