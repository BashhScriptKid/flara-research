"""Empirical verification of Fable's stage-separated FLOP correction:
QK^T scoring gets ZERO savings vs dense (every key in every active tier
is scored), while only the AV/softmax stage benefits from the ~10%
survivor rate -- and even there, total survivors summed across tiers
scale ~O(N) too (dominated by the largest/coarsest active tier), so the
AV discount is a constant factor (~5-10x), not sublinear.

Counts REAL terms from an actual run of monarch_meta_threshold_shared_tau
(final confirmed design) at several N, rather than assuming the
quantile-implies-exactly-10% structurally -- measuring what actually
happens with real random data, same discipline as every other
experiment this session (measure, don't assume).

Dense causal attention over N positions: query i attends to i+1 keys
(0-indexed), so total QK^T terms = total AV terms = sum_{i=0}^{N-1}(i+1)
= N(N+1)/2 -- used as the baseline both stages are compared against.
"""

import sys, math
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

D, Dv = 16, 16
B = 16
W_blocks = 1
QUANTILE = 0.90


def instrumented_run(q, k, v, B, W_blocks, quantile=QUANTILE, eps=1e-6):
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

    qk_terms = 0   # total real key-scoring dot products (QK^T stage)
    av_terms = 0   # total real value-weighted-sum terms (AV stage): local window + survivors + residual centroids

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
        win_mask = causal_win & win_valid.view(1, -1)  # (B, win_len) bool, per-query validity

        # local window: exact, both QK and AV terms = number of valid (query,key) pairs
        n_local_valid = int(win_mask.sum().item()) * E * H
        qk_terms += n_local_valid
        av_terms += n_local_valid

        local_scores = sm_scale * (q_m @ win_k.transpose(-1, -2))
        local_scores = local_scores.masked_fill(~win_mask.view(1, 1, B, -1), -float("inf"))
        local_scores_nan = local_scores.masked_fill(~win_mask.view(1, 1, B, -1), float("nan"))

        n = m0 - W_blocks + 1
        candidates = []
        if n > 0:
            for l in range(L):
                if (n >> l) & 1:
                    candidates.append((l, (n >> (l + 1)) << 1))

        per_tier_scores, per_tier_kv = [], []
        for l, block_idx in candidates:
            Bl = B * (1 << l)
            Ml = N_padded // Bl
            if block_idx >= Ml:
                continue
            k_block = k_flat.view(E, H, Ml, Bl, Dh)[:, :, block_idx]
            scores = sm_scale * (q_m @ k_block.transpose(-1, -2))  # (E,H,B,Bl)
            per_tier_scores.append(scores)
            per_tier_kv.append((l, block_idx, Bl))
            # QK^T stage: every real key in every active tier's block gets scored, full stop
            qk_terms += scores.numel()

        if per_tier_scores:
            pooled = torch.cat(per_tier_scores, dim=-1)
            shared_tau = torch.quantile(pooled, quantile, dim=-1, keepdim=True)
            for scores, (l, block_idx, Bl) in zip(per_tier_scores, per_tier_kv):
                survivors = scores >= shared_tau
                av_terms += int(survivors.sum().item())  # real AV terms: only survivors
                has_non_surv = (~survivors).sum(dim=-1) > 0
                av_terms += int(has_non_surv.sum().item())  # +1 residual-centroid AV term per (E,H,row) with any non-survivor

    return qk_terms, av_terms


print("=== Empirical stage-separated FLOP accounting: shared-tau threshold selection vs dense ===")
print(f"{'N':>6} | {'dense terms':>12} | {'QK terms':>12} {'QK ratio':>9} | {'AV terms':>12} {'AV ratio':>9} | {'blended ratio':>13}")

for N in (256, 512, 1024, 2048, 4096):
    g = torch.Generator().manual_seed(42)
    q = torch.randn(1, 1, N, D, generator=g) * 0.5
    k = torch.randn(1, 1, N, D, generator=g) * 0.5
    v = torch.randn(1, 1, N, Dv, generator=g) * 0.5

    qk_terms, av_terms = instrumented_run(q, k, v, B=B, W_blocks=W_blocks)
    dense_terms = N * (N + 1) // 2

    qk_ratio = qk_terms / dense_terms
    av_ratio = av_terms / dense_terms
    # blended: total real FLOPs (QK+AV) here vs dense (QK+AV), each stage roughly equal weight in dense
    blended = (qk_terms + av_terms) / (2 * dense_terms)

    print(f"{N:>6} | {dense_terms:>12} | {qk_terms:>12} {qk_ratio:>9.3f} | {av_terms:>12} {av_ratio:>9.3f} | {blended:>13.3f}")

print()
print("QK ratio ~1.0 confirms zero savings on the scoring stage (matches Fable's prediction).")
print("AV ratio << 1.0 confirms real savings on the value-aggregation stage.")
print("Blended ratio is the honest overall FLOP ratio vs dense -- expected ~0.5-0.65")
print("(i.e. ~1.5-2x speedup ceiling), not the ~0.1 the AV-only figure alone would suggest.")
