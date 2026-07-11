"""Real sparse-gather implementation for threshold selection, vs. the
full-score-then-mask reference used everywhere else in this session
(ma_meta_bucket_route.py, ma_causal_topk_ann.py). This is the item
Fable's synthesis specifically flagged as still owed: an honest cost
comparison, not another masked-dense reference.

Two implementations of the same math (threshold selection, reservoir
tau estimated per block, exact softmax over survivors, joint with a
window in the real pipeline -- here isolated to just the off-diagonal
block read since that's what's being cost-compared):

1. full_mask: compute (B, Bl) scores densely, mask non-survivors to
   -inf, softmax, matmul against the FULL v_block (wasted FLOPs on
   masked entries, but a single dense matmul -- what every reference
   implementation in this session has actually done).

2. sparse_gather: compute (B, Bl) scores densely (unavoidable -- the
   threshold test itself needs a score per key), but then GATHER only
   the (query, key) survivor pairs via torch.nonzero, and aggregate
   per-query outputs via index_add over the compact gathered set,
   never touching v_block entries for non-survivors in the aggregation
   step.

Honest question this answers: does avoiding "wasted" dense compute
actually pay off in PyTorch reference code, or does index/gather
overhead outweigh the savings at these sizes -- the same kind of
question that sank FlashMonarchAttention's naive first benchmark
earlier tonight. Reporting whichever answer comes out, not assuming
sparse is faster because it does less arithmetic.
"""

import sys, time
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F
from math import sqrt

sm_scale_d = lambda D: 1 / sqrt(D)


def full_mask_attention(q_m, k_block, v_block, tau, sm_scale):
    """q_m: (B,D), k_block: (Bl,D), v_block: (Bl,Dv), tau: scalar."""
    scores = sm_scale * (q_m @ k_block.T)  # (B,Bl)
    survivors = scores >= tau
    masked = scores.masked_fill(~survivors, -float("inf"))
    row_max = torch.clamp(masked.max(dim=-1, keepdim=True).values, min=-1e30)
    row_max = torch.nan_to_num(row_max, neginf=0.0)
    exp_s = torch.nan_to_num(torch.exp(masked - row_max), nan=0.0)
    denom = exp_s.sum(dim=-1, keepdim=True) + 1e-6
    weights = exp_s / denom
    return weights @ v_block  # (B,Dv)


def sparse_gather_attention(q_m, k_block, v_block, tau, sm_scale):
    """Same math, but the value-aggregation step only touches survivor
    (query, key) pairs via a real gather + index_add, not a dense matmul
    against the full v_block."""
    B = q_m.shape[0]
    Dv = v_block.shape[-1]
    scores = sm_scale * (q_m @ k_block.T)  # (B,Bl) -- scoring itself is unavoidable
    survivors = scores >= tau  # (B,Bl)

    query_idx, key_idx = torch.nonzero(survivors, as_tuple=True)  # (nnz,), (nnz,)
    surv_scores = scores[query_idx, key_idx]  # (nnz,)

    # per-query max for stable softmax (segment-max via scatter_reduce)
    row_max = torch.full((B,), -1e30, device=q_m.device, dtype=scores.dtype)
    row_max.scatter_reduce_(0, query_idx, surv_scores, reduce="amax", include_self=True)
    row_max = torch.nan_to_num(row_max, neginf=0.0)

    exp_s = torch.exp(surv_scores - row_max[query_idx])  # (nnz,)

    denom = torch.zeros(B, device=q_m.device, dtype=scores.dtype)
    denom.index_add_(0, query_idx, exp_s)
    denom = denom + 1e-6

    gathered_v = v_block[key_idx]  # (nnz,Dv) -- real gather, only survivor rows
    weighted_v = exp_s.unsqueeze(-1) * gathered_v  # (nnz,Dv)

    out = torch.zeros(B, Dv, device=q_m.device, dtype=v_block.dtype)
    out.index_add_(0, query_idx, weighted_v)
    out = out / denom.unsqueeze(-1)
    return out


def padded_dense_attention(q_m, k_block, v_block, tau, sm_scale, pad_width):
    """Fable's suggested third variant: pad to a FIXED max-survivor-count
    per query and do a single dense-but-smaller matmul, instead of true
    ragged gather via scatter_reduce/index_add. Take the top `pad_width`
    scores per query (torch.topk, one op, fixed shape), mask any that
    fall below tau within that padded set (handles the case where fewer
    than pad_width truly survive), gather their values via fixed-shape
    advanced indexing, weighted-sum over the small fixed width.

    Approximation note: if the TRUE survivor count for some query exceeds
    pad_width, this silently drops the excess (lowest-scoring true
    survivors beyond the padding width) -- an accuracy/cost tradeoff.
    pad_width should be chosen comfortably above the expected average
    survivor count to keep this rare."""
    top_scores, top_idx = torch.topk(q_m @ k_block.T * sm_scale, pad_width, dim=-1)  # (B,pad_width) each
    valid = top_scores >= tau
    masked = top_scores.masked_fill(~valid, -float("inf"))
    row_max = torch.clamp(masked.max(dim=-1, keepdim=True).values, min=-1e30)
    row_max = torch.nan_to_num(row_max, neginf=0.0)
    exp_s = torch.nan_to_num(torch.exp(masked - row_max), nan=0.0)
    denom = exp_s.sum(dim=-1, keepdim=True) + 1e-6
    weights = exp_s / denom  # (B,pad_width)
    gathered_v = v_block[top_idx]  # (B,pad_width,Dv) -- fixed-shape batched gather
    return (weights.unsqueeze(-1) * gathered_v).sum(dim=-2)  # (B,Dv)


def bench(fn, *args, reps=50, warmup=5):
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args)
        return (time.perf_counter() - t0) / reps


print("=== Correctness check: sparse_gather matches full_mask ===")
torch.manual_seed(0)
D, Dv, B, Bl = 16, 16, 16, 128
q_m = torch.randn(B, D)
k_block = torch.randn(Bl, D)
v_block = torch.randn(Bl, Dv)
sm_scale = sm_scale_d(D)
scores_check = sm_scale * (q_m @ k_block.T)
tau = torch.quantile(scores_check, 0.90).item()
z1 = full_mask_attention(q_m, k_block, v_block, tau, sm_scale)
z2 = sparse_gather_attention(q_m, k_block, v_block, tau, sm_scale)
max_diff = (z1 - z2).abs().max().item()
print(f"max abs diff: {max_diff:.2e} (should be ~0)")

print()
print("=== Wall-clock cost: full_mask vs sparse_gather, across block size Bl ===")
print("(query batch B fixed at 16, matching Monarch's own block size convention;")
print(" tau fixed at the 90th percentile -> ~10% survivor rate throughout)")
print()
print(f"{'Bl':>6} {'D':>4} | {'full_mask':>12} {'sparse_gather':>14} | {'speedup':>8} | {'avg survivors':>13}")

D, Dv, B = 64, 64, 16
for Bl in (128, 512, 2048, 8192, 32768):
    q_m = torch.randn(B, D)
    k_block = torch.randn(Bl, D)
    v_block = torch.randn(Bl, Dv)
    sm_scale = sm_scale_d(D)
    scores_ref = sm_scale * (q_m @ k_block.T)
    tau = torch.quantile(scores_ref, 0.90).item()
    n_surv = (scores_ref >= tau).float().sum(dim=-1).mean().item()

    reps = 100 if Bl <= 2048 else 30
    t_full = bench(full_mask_attention, q_m, k_block, v_block, tau, sm_scale, reps=reps)
    t_sparse = bench(sparse_gather_attention, q_m, k_block, v_block, tau, sm_scale, reps=reps)
    print(f"{Bl:>6} {D:>4} | {t_full*1e3:>11.4f}ms {t_sparse*1e3:>13.4f}ms | "
          f"{t_full/t_sparse:>7.2f}x | {n_surv:>13.1f}")

print()
print("=== Fable's follow-up: padded-dense (fixed-shape topk + gather, no scatter ops) ===")
print("(pad_width chosen ~1.5-2x average survivors; correctness check first)")
print()
D, Dv, B, Bl = 16, 16, 16, 128
torch.manual_seed(0)
q_m = torch.randn(B, D)
k_block = torch.randn(Bl, D)
v_block = torch.randn(Bl, Dv)
sm_scale = sm_scale_d(D)
scores_check = sm_scale * (q_m @ k_block.T)
tau = torch.quantile(scores_check, 0.90).item()
z1 = full_mask_attention(q_m, k_block, v_block, tau, sm_scale)
z3 = padded_dense_attention(q_m, k_block, v_block, tau, sm_scale, pad_width=32)
print(f"padded_dense (pad_width=32) vs full_mask max abs diff: {(z1-z3).abs().max().item():.2e}")

print()
print(f"{'Bl':>6} {'D':>4} {'pad_width':>10} | {'full_mask':>12} {'sparse_gather':>14} {'padded_dense':>13} | "
      f"{'pd speedup':>11} {'avg survivors':>13}")

D, Dv, B = 64, 64, 16
for Bl, pad_width in ((128, 24), (512, 96), (2048, 384), (8192, 1536), (32768, 6144)):
    q_m = torch.randn(B, D)
    k_block = torch.randn(Bl, D)
    v_block = torch.randn(Bl, Dv)
    sm_scale = sm_scale_d(D)
    scores_ref = sm_scale * (q_m @ k_block.T)
    tau = torch.quantile(scores_ref, 0.90).item()
    n_surv = (scores_ref >= tau).float().sum(dim=-1).mean().item()

    reps = 100 if Bl <= 2048 else 30
    t_full = bench(full_mask_attention, q_m, k_block, v_block, tau, sm_scale, reps=reps)
    t_sparse = bench(sparse_gather_attention, q_m, k_block, v_block, tau, sm_scale, reps=reps)
    t_padded = bench(padded_dense_attention, q_m, k_block, v_block, tau, sm_scale, pad_width, reps=reps)
    print(f"{Bl:>6} {D:>4} {pad_width:>10} | {t_full*1e3:>11.4f}ms {t_sparse*1e3:>13.4f}ms "
          f"{t_padded*1e3:>12.4f}ms | {t_full/t_padded:>10.2f}x {n_surv:>13.1f}")

print()
print("If padded_dense speedup < 1.0 too, that's Fable's predicted STRONGER result:")
print("PyTorch's per-op dispatch/kernel-launch fixed costs dominate at these sizes")
print("regardless of gather strategy -- a real signal that a compiled kernel, not")
print("another PyTorch rewrite, is the honest next step.")
