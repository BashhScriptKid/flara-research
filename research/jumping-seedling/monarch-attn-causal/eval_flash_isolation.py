import sys, time
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F
from math import sqrt

torch.manual_seed(0)


def naive_attention(q, k, v, sm_scale, causal: bool):
    scores = sm_scale * (q @ k.transpose(-1, -2))
    if causal:
        n = q.shape[-2]
        mask = torch.tril(torch.ones(n, n, dtype=torch.bool, device=q.device))
        scores = scores.masked_fill(~mask, -float("inf"))
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def fused_attention(q, k, v, sm_scale, causal: bool):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)


naive_compiled = torch.compile(naive_attention, dynamic=False)


def bench(fn, *args, reps=50, warmup=8):
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args)
        return (time.perf_counter() - t0) / reps


print("Isolating dispatch overhead from genuine memory-traffic savings:")
print("naive (eager, 3 separate ops) vs naive_compiled (torch.compile fuses the")
print("Python-level graph, removing per-op dispatch overhead but NOT changing the")
print("underlying algorithm) vs fused (real SDPA kernel, algorithmically flash-like).")
print("If compiled-naive closes most of the gap to fused, the earlier 2-9x was")
print("mostly dispatch overhead, not memory-traffic savings. If a real gap remains")
print("between compiled-naive and fused, that gap is the genuine algorithmic effect.")
print()
print(f"{'size':>6} | {'naive(eager)':>13} {'naive(compiled)':>16} {'fused(SDPA)':>12} | {'compiled/fused':>14}")

D = 64
H = 8
for size, reps in [(16, 200), (32, 200), (64, 100), (128, 100), (256, 50), (512, 30), (1024, 15)]:
    q = torch.randn(1, H, size, D)
    k = torch.randn(1, H, size, D)
    v = torch.randn(1, H, size, D)
    sm_scale = 1 / sqrt(D)

    t_naive = bench(naive_attention, q, k, v, sm_scale, True, reps=reps)
    t_compiled = bench(naive_compiled, q, k, v, sm_scale, True, reps=reps)
    t_fused = bench(fused_attention, q, k, v, sm_scale, True, reps=reps)

    print(f"{size:>6} | {t_naive*1e3:>12.3f}ms {t_compiled*1e3:>15.3f}ms {t_fused*1e3:>11.3f}ms | {t_compiled/t_fused:>13.2f}x")

print()
print("=== Same isolation at Monarch's batched-small-block shape ===")
print(f"{'B':>4} {'M':>4} | {'naive(eager)':>13} {'naive(compiled)':>16} {'fused(SDPA)':>12} | {'compiled/fused':>14}")
for B, M in [(16, 16), (16, 64), (32, 8), (32, 32), (64, 4), (64, 16)]:
    q = torch.randn(1, M, B, D)
    k = torch.randn(1, M, B, D)
    v = torch.randn(1, M, B, D)
    sm_scale = 1 / sqrt(D)
    reps = 200
    t_naive = bench(naive_attention, q, k, v, sm_scale, True, reps=reps)
    t_compiled = bench(naive_compiled, q, k, v, sm_scale, True, reps=reps)
    t_fused = bench(fused_attention, q, k, v, sm_scale, True, reps=reps)
    print(f"{B:>4} {M:>4} | {t_naive*1e3:>12.3f}ms {t_compiled*1e3:>15.3f}ms {t_fused*1e3:>11.3f}ms | {t_compiled/t_fused:>13.2f}x")
