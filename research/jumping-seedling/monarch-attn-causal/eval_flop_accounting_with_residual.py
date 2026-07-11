"""Extends eval_flop_accounting.py to also count the residual-centroid
REDUCTION cost (mean_k, mean_v over ~90% of each tier's Bl keys) --
a real O(Bl*D) masked-sum operation per tier per query, same order as
that tier's own QK matmul, that the original FLOP accounting omitted
(it only counted terms entering the FINAL combined softmax: survivors
+ one residual slot per tier, not the cost of COMPUTING that slot).

Per Fable: "plausibly negligible is exactly the kind of claim this
thread has learned not to accept without checking" -- checking whether
this omitted cost materially erodes the ~1.8x blended FLOP ratio.
"""

import sys, math
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

D, Dv = 16, 16
B = 16
W_blocks = 1
QUANTILE = 0.90


def instrumented_run_full(q, k, v, B, W_blocks, quantile=QUANTILE):
    E, H, N, Dh = q.shape
    _, _, _, Dvh = v.shape
    sm_scale = 1 / math.sqrt(Dh)

    M_base_needed = (N + B - 1) // B
    L = max(1, math.ceil(math.log2(max(M_base_needed, 2))))
    N_padded = B * (1 << L)
    pad = N_padded - N

    qb = F.pad(q, (0, 0, 0, pad)).view(E, H, -1, B, Dh)
    kb = F.pad(k, (0, 0, 0, pad)).view(E, H, -1, B, Dh)
    M = qb.shape[2]
    valid_mb = (torch.arange(N_padded).view(M, B) < N)
    k_flat = F.pad(k, (0, 0, 0, pad))
    v_flat = F.pad(v, (0, 0, 0, pad))

    qk_terms = 0          # QK^T scoring
    av_terms = 0           # final-softmax terms (survivors + residual slots)
    residual_reduction_terms = 0  # NEW: cost of COMPUTING mean_k/mean_v per tier (masked reduction over Bl keys)
    quantile_sort_terms = 0        # NEW: cost of torch.quantile (sort-based) over the pooled score sample

    for m0 in range(M):
        q_m = qb[:, :, m0]
        w_start = max(0, m0 - W_blocks + 1)
        win_k = kb[:, :, w_start : m0 + 1].reshape(E, H, -1, Dh)
        n_win_blocks = m0 - w_start + 1
        win_valid = valid_mb[w_start : m0 + 1].reshape(-1)
        blk_idx = torch.arange(n_win_blocks).repeat_interleave(B)
        own_blk = n_win_blocks - 1
        intra = torch.arange(B).repeat(n_win_blocks)
        causal_win = (blk_idx < own_blk).unsqueeze(0) | (
            (blk_idx == own_blk).unsqueeze(0) & (intra.unsqueeze(0) <= torch.arange(B).unsqueeze(1))
        )
        win_mask = causal_win & win_valid.view(1, -1)

        n_local_valid = int(win_mask.sum().item()) * E * H
        qk_terms += n_local_valid
        av_terms += n_local_valid

        n = m0 - W_blocks + 1
        candidates = []
        if n > 0:
            for l in range(L):
                if (n >> l) & 1:
                    candidates.append((l, (n >> (l + 1)) << 1))
        if not candidates:
            continue

        per_tier_scores, per_tier_kv = [], []
        for l, block_idx in candidates:
            Bl = B * (1 << l)
            Ml = N_padded // Bl
            if block_idx >= Ml:
                continue
            k_block = k_flat.view(E, H, Ml, Bl, Dh)[:, :, block_idx]
            scores = sm_scale * (q_m @ k_block.transpose(-1, -2))
            per_tier_scores.append(scores)
            per_tier_kv.append((l, block_idx, Bl))
            qk_terms += scores.numel()
            # residual-centroid reduction: masked sum over ALL Bl keys (mean_k, mean_v),
            # per query row -- same shape as the QK matmul itself, real extra reduction work
            residual_reduction_terms += scores.numel()  # (E,H,B,Bl) -- one reduction term per (query,key) pair, per tier

        pooled = torch.cat(per_tier_scores, dim=-1)
        # quantile sort cost: n log n comparisons over the pooled sample size, per (E,H,B) row
        pool_size = pooled.shape[-1]
        n_rows = pooled.shape[0] * pooled.shape[1] * pooled.shape[2]
        quantile_sort_terms += n_rows * pool_size * max(1, math.ceil(math.log2(max(pool_size, 2))))

        shared_tau = torch.quantile(pooled, quantile, dim=-1, keepdim=True)
        for scores, (l, block_idx, Bl) in zip(per_tier_scores, per_tier_kv):
            survivors = scores >= shared_tau
            av_terms += int(survivors.sum().item())
            has_non_surv = (~survivors).sum(dim=-1) > 0
            av_terms += int(has_non_surv.sum().item())

    return qk_terms, av_terms, residual_reduction_terms, quantile_sort_terms


print("=== FLOP accounting INCLUDING residual-centroid reduction + quantile-sort overhead ===")
print(f"{'N':>6} | {'dense terms':>12} | {'QK':>10} | {'AV(final)':>10} | {'residual-reduction':>19} | {'quantile-sort(equiv-cmp)':>24} | {'blended (QK+AV+resid)':>22} | {'blended w/ D-scaled sort':>24}")

D_prod = 64  # production head_dim, for converting sort "comparisons" to FLOP-equivalent units fairly
for N in (256, 512, 1024, 2048, 4096):
    g = torch.Generator().manual_seed(42)
    q = torch.randn(1, 1, N, D, generator=g) * 0.5
    k = torch.randn(1, 1, N, D, generator=g) * 0.5
    v = torch.randn(1, 1, N, Dv, generator=g) * 0.5

    qk_terms, av_terms, resid_terms, sort_terms = instrumented_run_full(q, k, v, B=B, W_blocks=W_blocks)
    dense_terms = N * (N + 1) // 2

    # FLOPs: QK term = 2D flops, AV term = 2Dv flops, residual-reduction term = 2D flops (same shape as QK)
    # quantile sort: comparisons are ~O(1) each (not O(D)), so express as FLOP-equivalent using a single-compare cost of 1 flop
    qk_flops = qk_terms * 2 * D
    av_flops = av_terms * 2 * Dv
    resid_flops = resid_terms * 2 * D
    sort_flops_as_1op = sort_terms * 1  # each comparison ~1 flop-equivalent, dimension-independent

    dense_flops = dense_terms * 2 * (D + Dv)

    blended_no_resid = (qk_flops + av_flops) / dense_flops
    blended_with_resid = (qk_flops + av_flops + resid_flops) / dense_flops
    blended_with_resid_and_sort = (qk_flops + av_flops + resid_flops + sort_flops_as_1op) / dense_flops

    print(f"{N:>6} | {dense_terms:>12} | {qk_terms:>10} | {av_terms:>10} | {resid_terms:>19} | {sort_terms:>24} | {blended_with_resid:>22.3f} | {blended_with_resid_and_sort:>24.3f}")

print()
print(f"(for reference: blended ratio WITHOUT residual-reduction cost, as originally reported, was ~0.55-0.59)")
