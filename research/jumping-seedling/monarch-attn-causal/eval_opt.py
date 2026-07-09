import sys, time
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma.ma_torch import monarch_attention_torch as orig_noncausal
from ma_causal import monarch_attention_causal_torch as causal_single
from ma_causal_dual import monarch_attention_causal_dual_torch as causal_dual
from ma_causal_dual_opt import monarch_attention_causal_dual_opt_torch as causal_dual_opt

torch.manual_seed(0)
B, T = 4, 3

print("=== numerical agreement: dual vs dual_opt (should match to fp precision) ===")
for N in (16, 64, 128):
    q = torch.randn(1, 1, N, 8); k = torch.randn(1, 1, N, 8); v = torch.randn(1, 1, N, 8)
    z1 = causal_dual(q, k, v, None, T, B, pre_pad=False)
    z2 = causal_dual_opt(q, k, v, None, T, B, pre_pad=False)
    print(f"N={N}: max abs diff = {(z1-z2).abs().max().item():.3e}")

print()
print("=== causal validity check (opt) ===")
N = 32
q = torch.randn(1, 1, N, 8); k = torch.randn(1, 1, N, 8)
eye = torch.eye(N).expand(1, 1, N, N)
A = causal_dual_opt(q, k, eye, None, T, B, pre_pad=False)
leak = torch.triu(A[0, 0], diagonal=1).abs().max().item()
print(f"max future weight: {leak:.6e}, row sums min/max: {A[0,0].sum(-1).min().item():.6f}/{A[0,0].sum(-1).max().item():.6f}")

print()
print("=== wall-clock benchmark (mean of 20 runs, torch.no_grad) ===")
def bench(fn, *args, reps=20):
    with torch.no_grad():
        for _ in range(3):  # warmup
            fn(*args)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args)
        return (time.perf_counter() - t0) / reps

print(f"{'N':>5} {'B':>4} {'noncausal':>11} {'single':>11} {'dual':>11} {'dual_opt':>11} {'opt/noncausal':>13} {'opt/single':>11}")
for N, Bb in [(256, 16), (512, 16), (1024, 32), (2048, 32)]:
    q = torch.randn(1, 8, N, 64); k = torch.randn(1, 8, N, 64); v = torch.randn(1, 8, N, 64)
    t_nc = bench(orig_noncausal, q, k, v, None, T, Bb, False)
    t_s = bench(causal_single, q, k, v, None, T, Bb, False, True)
    t_d = bench(causal_dual, q, k, v, None, T, Bb, False)
    t_o = bench(causal_dual_opt, q, k, v, None, T, Bb, False)
    print(f"{N:>5} {Bb:>4} {t_nc*1000:>9.2f}ms {t_s*1000:>9.2f}ms {t_d*1000:>9.2f}ms {t_o*1000:>9.2f}ms {t_o/t_nc:>13.3f} {t_o/t_s:>11.3f}")
