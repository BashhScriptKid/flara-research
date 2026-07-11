"""Re-verify the ridge-point/compute-bound conclusion using the
CORRECTED FLOP figures (fast-residual + reservoir-tau, ~0.60-0.65
blended ratio) instead of the original ~0.55 figure that omitted the
residual-centroid cost. Per Fable: the correction only ADDS FLOPs
(recovers savings but doesn't erase the real survivor-gather marginal
cost) while leaving bytes moved completely unchanged (a separate,
unaffected measurement) -- so arithmetic intensity can only go UP,
widening the existing margin, never shrinking it. Confirming this
arithmetically rather than asserting it, same discipline used
throughout this session, and reporting the actual updated numbers
rather than reusing pre-fix figures.

Uses the real production config (src/model/config.rs): head_dim=64,
n_q_heads=14, n_kv_heads=2 (GQA), kv_block=64 -- same as the first
ridge-point check.
"""

import sys, math
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from eval_flop_accounting_fast_residual import instrumented_run_fast
from eval_memory_traffic_bounds import instrumented_run as byte_run

CORES = 6
AVX2_FLOPS_PER_CYCLE_PER_CORE = 32
CLOCK_THEORETICAL = 2.1e9
CLOCK_REALISTIC = 1.6e9
PEAK_FLOPS_THEORETICAL = CORES * AVX2_FLOPS_PER_CYCLE_PER_CORE * CLOCK_THEORETICAL
PEAK_FLOPS_REALISTIC = CORES * AVX2_FLOPS_PER_CYCLE_PER_CORE * CLOCK_REALISTIC * 0.5

DDR4_3200_MTS = 3200e6
CHANNELS = 2
BYTES_PER_TRANSFER = 8
PEAK_BW_THEORETICAL = DDR4_3200_MTS * BYTES_PER_TRANSFER * CHANNELS
PEAK_BW_REALISTIC = PEAK_BW_THEORETICAL * 0.65

RIDGE_THEORETICAL = PEAK_FLOPS_THEORETICAL / PEAK_BW_THEORETICAL
RIDGE_REALISTIC = PEAK_FLOPS_REALISTIC / PEAK_BW_REALISTIC

print("=== Ridge-point re-check with CORRECTED (fast-residual) FLOP figures ===")
print(f"Ridge point: theoretical={RIDGE_THEORETICAL:.2f} FLOPs/byte, realistic={RIDGE_REALISTIC:.2f} FLOPs/byte")
print()

D, Dv = 64, 64
N_Q_HEADS = 14
N_KV_HEADS = 2
B = 64
W_blocks = 1

print(f"Production config: head_dim={D}, n_q_heads={N_Q_HEADS}, n_kv_heads={N_KV_HEADS} (GQA), kv_block={B}")
print()
print(f"{'N':>7} | {'thresh AI (old, 0.55x)':>22} | {'thresh AI (corrected)':>21} | {'regime (realistic ridge)':>26}")

for N in (512, 1024, 2048, 4096, 8192):
    g = torch.Generator().manual_seed(42)
    q = torch.randn(1, N_KV_HEADS, N, D, generator=g) * 0.5
    k = torch.randn(1, N_KV_HEADS, N, D, generator=g) * 0.5
    v = torch.randn(1, N_KV_HEADS, N, Dv, generator=g) * 0.5

    qk_terms, av_terms, surv_gather_terms, onetime_terms, reservoir_terms = instrumented_run_fast(q, k, v, B=B, W_blocks=W_blocks)
    qk_terms_q = qk_terms * N_Q_HEADS / N_KV_HEADS
    av_terms_q = av_terms * N_Q_HEADS / N_KV_HEADS
    surv_gather_q = surv_gather_terms * N_Q_HEADS / N_KV_HEADS
    onetime_q = onetime_terms * N_Q_HEADS / N_KV_HEADS
    reservoir_q = reservoir_terms * N_Q_HEADS / N_KV_HEADS

    thresh_flops_old = qk_terms_q * 2 * D + av_terms_q * 2 * Dv  # old: QK+AV only, no residual/tau cost
    thresh_flops_corrected = (
        qk_terms_q * 2 * D
        + av_terms_q * 2 * Dv
        + surv_gather_q * 2 * D
        + onetime_q * 2 * D
        + reservoir_q * 1
    )

    qk_touches, av_touches, _, _, _, _ = byte_run(q, k, v, B=B, W_blocks=W_blocks)
    thresh_bytes = qk_touches * D * 4 + av_touches * Dv * 4

    ai_old = thresh_flops_old / thresh_bytes
    ai_corrected = thresh_flops_corrected / thresh_bytes

    regime = "compute-bound" if ai_corrected > RIDGE_REALISTIC else "memory-bound"

    print(f"{N:>7} | {ai_old:>22.3f} | {ai_corrected:>21.3f} | {regime:>26}")

print()
print("If ai_corrected > ai_old at every N (margin widens, not shrinks) and both stay")
print("well above the realistic ridge point, the compute-bound conclusion holds under")
print("the corrected FLOP figures -- confirms Fable's directional prediction arithmetically.")
