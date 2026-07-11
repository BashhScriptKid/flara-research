"""Ridge-point check: which side of the compute/bandwidth boundary does
this kernel actually fall on? This is the step that resolves whether
the ~1.8x FLOP reduction (verified) is realizable in wall-clock time
given the ~1.0x memory-traffic ratio (also verified) -- pure arithmetic
from numbers already measured, no new instrumentation or simulation.

Ridge point = peak AVX2 FLOP/s / peak DRAM bandwidth, for the 5500U.
Implied arithmetic intensity (AI) of the kernel = FLOPs / bytes moved,
computed here from the SAME production-scale (D=64,H=8,B=64) measured
counts used in eval_flop_accounting.py (FLOPs, per-individual-query,
no reuse credit -- compute doesn't get to skip work) and
eval_memory_traffic_bounds.py (bytes, block-tiled reuse-credited --
bytes DO benefit from batching B queries against one key read).

If implied AI < ridge point: memory-bandwidth-bound. The 5500U is
stalled on bytes regardless of FLOP count -- cutting FLOPs 1.8x while
leaving bytes ~unchanged buys close to nothing in wall-clock time.

If implied AI > ridge point: compute-bound. The FLOP reduction shows up
directly in wall-clock time, since bytes were never the bottleneck.

5500U figures (Zen 2 "Lucienne", 6 cores, AVX2 256-bit, no AVX-512,
TSMC 7nm, 15W nominal / 10-25W cTDP, dual-channel DDR4-3200 or
quad-channel LPDDR4-4266) -- sourced from public specs (TechPowerUp,
notebookcheck, cpu-monkey), NOT measured on real hardware. Two bounds
given (theoretical peak, and a realistic-efficiency estimate) since a
15W laptop part under sustained all-core AVX2 load will not hold boost
clock -- flagged explicitly as an estimate pending real profiling, same
methodology note as everywhere else PyTorch/analytical numbers have
stood in for real hardware measurement in this session.
"""

import sys, math
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from eval_flop_accounting import instrumented_run as flop_run
from eval_memory_traffic_bounds import instrumented_run as byte_run

# --- 5500U peak figures (see module docstring for sourcing/caveats) ---
CORES = 6
AVX2_FLOPS_PER_CYCLE_PER_CORE = 32  # 2x 256-bit FMA units, 8 floats/instr, 2 flops/FMA

CLOCK_THEORETICAL = 2.1e9   # base clock, best case
CLOCK_REALISTIC = 1.6e9     # conservative sustained all-core AVX2 estimate under 15W cTDP

PEAK_FLOPS_THEORETICAL = CORES * AVX2_FLOPS_PER_CYCLE_PER_CORE * CLOCK_THEORETICAL
PEAK_FLOPS_REALISTIC = CORES * AVX2_FLOPS_PER_CYCLE_PER_CORE * CLOCK_REALISTIC * 0.5  # 50% real-world matmul efficiency

DDR4_3200_MTS = 3200e6
CHANNELS = 2
BYTES_PER_TRANSFER = 8  # 64-bit per channel
PEAK_BW_THEORETICAL = DDR4_3200_MTS * BYTES_PER_TRANSFER * CHANNELS
PEAK_BW_REALISTIC = PEAK_BW_THEORETICAL * 0.65  # realistic STREAM-like achievable fraction

RIDGE_THEORETICAL = PEAK_FLOPS_THEORETICAL / PEAK_BW_THEORETICAL
RIDGE_REALISTIC = PEAK_FLOPS_REALISTIC / PEAK_BW_REALISTIC

print("=== 5500U roofline ridge point (estimated from public specs, not measured) ===")
print(f"Peak FLOPs: theoretical={PEAK_FLOPS_THEORETICAL/1e9:.1f} GFLOPs/s, realistic={PEAK_FLOPS_REALISTIC/1e9:.1f} GFLOPs/s")
print(f"Peak BW:    theoretical={PEAK_BW_THEORETICAL/1e9:.1f} GB/s, realistic={PEAK_BW_REALISTIC/1e9:.1f} GB/s")
print(f"Ridge point: theoretical={RIDGE_THEORETICAL:.2f} FLOPs/byte, realistic={RIDGE_REALISTIC:.2f} FLOPs/byte")
print()

# --- real Jumping Seedling production config (src/model/config.rs defaults) ---
D, Dv = 64, 64          # head_dim
N_Q_HEADS = 14
N_KV_HEADS = 2           # GQA: 7 query heads share each KV head -- compute scales
                          # with n_q_heads, K/V memory traffic scales with n_kv_heads
B = 64                    # kv_block
W_blocks = 1

print(f"Production config: head_dim={D}, n_q_heads={N_Q_HEADS}, n_kv_heads={N_KV_HEADS} (GQA), kv_block={B}")
print()
print("=== Implied arithmetic intensity vs ridge point, dense (tiled) vs threshold-selection ===")
print(f"{'N':>7} | {'dense AI':>10} {'thresh AI':>10} | {'regime (theoretical ridge)':>28} | {'regime (realistic ridge)':>26}")

for N in (512, 1024, 2048, 4096, 8192):
    # score/instrument against n_kv_heads worth of K/V (what's actually resident),
    # scale FLOPs up separately by n_q_heads since compute happens per query head
    g = torch.Generator().manual_seed(42)
    q = torch.randn(1, N_KV_HEADS, N, D, generator=g) * 0.5
    k = torch.randn(1, N_KV_HEADS, N, D, generator=g) * 0.5
    v = torch.randn(1, N_KV_HEADS, N, Dv, generator=g) * 0.5

    qk_terms, av_terms = flop_run(q, k, v, B=B, W_blocks=W_blocks)  # counted over n_kv_heads
    # rescale term counts from n_kv_heads to n_q_heads for the compute side
    qk_terms_q = qk_terms * N_Q_HEADS / N_KV_HEADS
    av_terms_q = av_terms * N_Q_HEADS / N_KV_HEADS
    thresh_flops = qk_terms_q * 2 * D + av_terms_q * 2 * Dv

    qk_touches, av_touches, _, _, _, _ = byte_run(q, k, v, B=B, W_blocks=W_blocks)  # counted over n_kv_heads (bytes actually moved)
    thresh_bytes = qk_touches * D * 4 + av_touches * Dv * 4

    dense_terms = N * (N + 1) // 2
    dense_flops = (dense_terms * 2 * D + dense_terms * 2 * Dv) * N_Q_HEADS  # compute: n_q_heads
    M_dense = (N + B - 1) // B
    dense_key_touches = sum(min((m0 + 1) * B, N) for m0 in range(M_dense))
    dense_bytes = dense_key_touches * D * 4 * N_KV_HEADS + dense_key_touches * Dv * 4 * N_KV_HEADS  # bytes: n_kv_heads

    dense_ai = dense_flops / dense_bytes
    thresh_ai = thresh_flops / thresh_bytes

    regime_theoretical = f"dense={'compute' if dense_ai>RIDGE_THEORETICAL else 'mem'}-bound, thresh={'compute' if thresh_ai>RIDGE_THEORETICAL else 'mem'}-bound"
    regime_realistic = f"dense={'compute' if dense_ai>RIDGE_REALISTIC else 'mem'}-bound, thresh={'compute' if thresh_ai>RIDGE_REALISTIC else 'mem'}-bound"

    print(f"{N:>7} | {dense_ai:>10.3f} {thresh_ai:>10.3f} | {regime_theoretical:>28} | {regime_realistic:>26}")

print()
print("If both dense and threshold-selection land on the SAME side of the ridge")
print("(both memory-bound, or both compute-bound), that determines whether the FLOP")
print("reduction is realizable: memory-bound => close to zero realizable wall-clock")
print("benefit despite the 1.8x FLOP win; compute-bound => the 1.8x should show up.")
