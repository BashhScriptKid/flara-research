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


def bench(fn, *args, reps=30, warmup=5):
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args)
        return (time.perf_counter() - t0) / reps


print("Isolation via scale: naive is ALWAYS exactly 3 op-launches, fused ALWAYS 1,")
print("regardless of M (the batch dim). If the earlier 2-9x speedup was mostly fixed")
print("per-launch dispatch overhead, it should shrink toward 1x as M grows large enough")
print("to amortize 3-vs-1 launches away. If it persists/grows, that's a genuine")
print("memory-traffic/algorithmic effect, not dispatch overhead.")
print()

D = 64
B = 16  # fixed at Monarch's actual block size
sm_scale = 1 / sqrt(D)

print(f"B={B} (fixed, Monarch's block size), D={D}, scaling M (batch dim / number of blocks):")
print(f"{'M':>6} | {'naive':>12} {'fused':>12} | {'speedup':>8} | {'naive/M (per-block)':>20} {'fused/M (per-block)':>20}")
for M, reps in [(1, 300), (4, 300), (16, 200), (64, 100), (256, 50), (1024, 20), (4096, 10)]:
    q = torch.randn(1, M, B, D)
    k = torch.randn(1, M, B, D)
    v = torch.randn(1, M, B, D)
    t_naive = bench(naive_attention, q, k, v, sm_scale, True, reps=reps)
    t_fused = bench(fused_attention, q, k, v, sm_scale, True, reps=reps)
    speedup = t_naive / t_fused
    print(f"{M:>6} | {t_naive*1e3:>11.4f}ms {t_fused*1e3:>11.4f}ms | {speedup:>7.2f}x | "
          f"{t_naive/M*1e6:>17.3f}us {t_fused/M*1e6:>17.3f}us")
