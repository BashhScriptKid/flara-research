import sys, time
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F
from math import sqrt

torch.manual_seed(0)


def naive_attention(q, k, v, sm_scale, causal=True):
    scores = sm_scale * (q @ k.transpose(-1, -2))
    if causal:
        n = q.shape[-2]
        mask = torch.tril(torch.ones(n, n, dtype=torch.bool, device=q.device))
        scores = scores.masked_fill(~mask, -float("inf"))
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def fused_attention(q, k, v, sm_scale, causal=True):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)


def bench(fn, *args, reps=50, warmup=5):
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args)
        return (time.perf_counter() - t0) / reps


print("Testing: does fusion (PyTorch's real CPU flash/mem-efficient SDPA kernel)")
print("beat naive separate-ops attention at Monarch's actual operating scale")
print("(block/window sizes ~16-256), or only at much larger sizes (GPU-relevant regime)?")
print()
print(f"{'size':>6} {'H':>3} {'D':>4} | {'naive':>10} {'fused(SDPA)':>12} | {'speedup':>8}")

D = 64
H = 8
for size, reps in [(16, 200), (32, 200), (64, 100), (128, 100), (256, 50), (512, 30), (1024, 15), (2048, 8)]:
    q = torch.randn(1, H, size, D)
    k = torch.randn(1, H, size, D)
    v = torch.randn(1, H, size, D)
    sm_scale = 1 / sqrt(D)

    t_naive = bench(naive_attention, q, k, v, sm_scale, True, reps=reps)
    t_fused = bench(fused_attention, q, k, v, sm_scale, True, reps=reps)
    speedup = t_naive / t_fused
    print(f"{size:>6} {H:>3} {D:>4} | {t_naive*1e3:>9.3f}ms {t_fused*1e3:>11.3f}ms | {speedup:>7.2f}x")

print()
print("=== Same test at Monarch's typical SINGLE-HEAD-batched-many-blocks shape ===")
print("(i.e. what a block-local pass actually looks like: many small B x B")
print(" attentions computed in parallel via a batch dim, not one big attention)")
print()
print(f"{'B':>4} {'M(batch)':>9} | {'naive':>10} {'fused(SDPA)':>12} | {'speedup':>8}")
for B, M in [(16, 16), (16, 64), (32, 8), (32, 32), (64, 4), (64, 16)]:
    q = torch.randn(1, M, B, D)  # M as a "batch of heads" dim, standing in for many parallel blocks
    k = torch.randn(1, M, B, D)
    v = torch.randn(1, M, B, D)
    sm_scale = 1 / sqrt(D)
    reps = 200
    t_naive = bench(naive_attention, q, k, v, sm_scale, True, reps=reps)
    t_fused = bench(fused_attention, q, k, v, sm_scale, True, reps=reps)
    speedup = t_naive / t_fused
    print(f"{B:>4} {M:>9} | {t_naive*1e3:>9.3f}ms {t_fused*1e3:>11.3f}ms | {speedup:>7.2f}x")
