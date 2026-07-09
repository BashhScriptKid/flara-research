#!/usr/bin/env python3
"""
FlashMonarchAttention GPU benchmark -- standalone, single-file.

WHAT THIS IS FOR
-----------------
Part of a research investigation into a causal-capable variant of
MonarchAttention (a sub-quadratic attention approximation). One question
came up: would fusing Monarch's small block-local softmax computations
(FlashAttention-style, avoiding materializing intermediate score
tensors) actually help on GPU? On CPU, we tested this carefully and
found NO real win once dispatch overhead is controlled for -- the
"speedup" we first saw disappeared when isolated properly (see the
naive-vs-fused-vs-scale comparison below; same methodology, just run on
your GPU instead of a CPU). GPU is a different memory hierarchy
(HBM <-> SRAM) than what CPU has, so the answer might be different there
-- but nobody on our end has a GPU to check. That's what this script is
for.

WHAT IT DOES
------------
Compares "naive" attention (three separate ops: matmul, softmax, matmul
-- each one a separate GPU kernel launch, each one writing/reading its
full intermediate tensor to/from GPU memory) against
`torch.nn.functional.scaled_dot_product_attention` (PyTorch's real
fused/flash CPU+GPU kernel -- on GPU this dispatches to an actual
FlashAttention-family kernel). Two parts:

  1. A straight size sweep (16 up to 4096) at a realistic head_dim=64,
     8 heads -- mirrors how big a real attention call gets.
  2. An ISOLATION test: fixes Monarch's actual small block size (B=16)
     and scales the batch dimension M (number of blocks processed at
     once) way up. Naive always launches exactly 3 kernels regardless
     of M; fused always launches 1. If a speedup is just fixed
     per-launch overhead, it should shrink toward 1x as M grows large
     enough to amortize that fixed cost away. If a speedup PERSISTS or
     GROWS as M grows, that's a genuine memory-bandwidth/algorithmic
     effect -- which is exactly the FlashAttention claim, and exactly
     what we couldn't test without a GPU.

HOW TO RUN
----------
Needs Python 3.9+ and PyTorch with CUDA support. If you don't have
PyTorch installed:

    pip install torch --index-url https://download.pytorch.org/whl/cu121

(swap cu121 for whatever CUDA version matches your driver -- check
`nvidia-smi` if unsure; https://pytorch.org/get-started/locally/ has the
exact command for your setup).

Then just run:

    python gpu_flash_benchmark.py

It auto-detects CUDA and will refuse to run (with a clear message,
not a crash) if no GPU is found. Takes maybe 1-2 minutes. Please just
paste the full terminal output back -- that's all we need, no
interpretation required on your end.

Thank you for running this!
"""

import sys
import time
from math import sqrt

import torch
import torch.nn.functional as F


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


def bench(fn, *args, device, reps=50, warmup=8):
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps


def main():
    print("=" * 78)
    print("FlashMonarchAttention GPU benchmark")
    print("=" * 78)
    print(f"PyTorch version: {torch.__version__}")

    if not torch.cuda.is_available():
        print()
        print("!! No CUDA GPU detected by PyTorch. !!")
        print("If you DO have an NVIDIA GPU, this usually means the installed")
        print("PyTorch build doesn't have CUDA support -- reinstall via:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("(pick the cuXXX matching your driver -- see nvidia-smi / pytorch.org)")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version (PyTorch build): {torch.version.cuda}")
    print()

    torch.manual_seed(0)
    D = 64
    H = 8

    print("--- Part 1: size sweep, H=8 heads, D=64 (typical realistic shape) ---")
    print(f"{'size':>6} | {'naive':>12} {'fused(SDPA)':>13} | {'speedup':>8}")
    for size, reps in [
        (16, 300), (32, 300), (64, 200), (128, 200), (256, 100),
        (512, 100), (1024, 50), (2048, 30), (4096, 15), (8192, 8),
    ]:
        q = torch.randn(1, H, size, D, device=device)
        k = torch.randn(1, H, size, D, device=device)
        v = torch.randn(1, H, size, D, device=device)
        sm_scale = 1 / sqrt(D)

        t_naive = bench(naive_attention, q, k, v, sm_scale, True, device=device, reps=reps)
        t_fused = bench(fused_attention, q, k, v, sm_scale, True, device=device, reps=reps)
        speedup = t_naive / t_fused
        print(f"{size:>6} | {t_naive*1e3:>11.4f}ms {t_fused*1e3:>12.4f}ms | {speedup:>7.2f}x")

    print()
    print("--- Part 2: ISOLATION test -- fixed B=16 (Monarch's actual block size), ---")
    print("---          scaling M (number of blocks processed in one call)         ---")
    print("If speedup shrinks toward/below 1x as M grows: it's dispatch overhead,")
    print("not a real memory-bandwidth win (this is what we found on CPU).")
    print("If speedup holds or grows as M grows: that's the genuine FlashAttention")
    print("effect showing up -- the thing we couldn't test without a GPU.")
    print()
    B = 16
    sm_scale = 1 / sqrt(D)
    print(f"{'M':>6} | {'naive':>12} {'fused':>12} | {'speedup':>8} | {'naive/block':>14} {'fused/block':>14}")
    for M, reps in [
        (1, 400), (4, 400), (16, 300), (64, 200), (256, 100),
        (1024, 50), (4096, 20), (16384, 8), (65536, 4),
    ]:
        try:
            q = torch.randn(1, M, B, D, device=device)
            k = torch.randn(1, M, B, D, device=device)
            v = torch.randn(1, M, B, D, device=device)
        except RuntimeError as e:
            print(f"{M:>6} | (skipped -- out of GPU memory: {e})")
            continue
        t_naive = bench(naive_attention, q, k, v, sm_scale, True, device=device, reps=reps)
        t_fused = bench(fused_attention, q, k, v, sm_scale, True, device=device, reps=reps)
        speedup = t_naive / t_fused
        print(f"{M:>6} | {t_naive*1e3:>11.4f}ms {t_fused*1e3:>11.4f}ms | {speedup:>7.2f}x | "
              f"{t_naive/M*1e6:>11.4f}us {t_fused/M*1e6:>11.4f}us")

    print()
    print("=" * 78)
    print("Done. Please paste everything above back -- thank you!")
    print("=" * 78)


if __name__ == "__main__":
    main()
