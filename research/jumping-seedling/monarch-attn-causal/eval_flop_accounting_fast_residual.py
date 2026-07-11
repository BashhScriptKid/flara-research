"""FLOP accounting for the fast-residual variant: the residual-centroid
cost is now O(num_survivors*D) per tier per query (real gather-
equivalent cost, same order as work already paid for the AV pass) plus
a ONE-TIME O(Bl*D) full-block-sum cost per tier PER BLOCK (not per
query) that amortizes to ~0 marginal cost as M (number of query groups)
grows -- instead of the O(Bl*D) PER QUERY cost the masked-reduction
implementation paid.

Also swaps the quantile step from sort-based (O(n log n)) to the
validated reservoir-sampling estimator (O(1) amortized per key, per
Louver's design and this session's earlier oracle-vs-reservoir
robustness check) -- fixed-size reservoir sample of the pooled tier
scores, quality already confirmed equivalent to the full-pool quantile
in that earlier round, so this is purely a cost substitution, not a
new quality question.
"""

import sys, math
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

D, Dv = 16, 16
B = 16
W_blocks = 1
QUANTILE = 0.90
RESERVOIR_SIZE = 64  # fixed-size reservoir sample for tau estimation, O(1) amortized per key


def instrumented_run_fast(q, k, v, B, W_blocks, quantile=QUANTILE):
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

    qk_terms = 0
    av_terms = 0
    survivor_gather_terms = 0   # O(num_survivors*D) real residual cost
    onetime_block_sum_terms = 0  # O(Bl*D) PAID ONCE per (tier,block), not per query
    reservoir_tau_terms = 0      # O(RESERVOIR_SIZE) per active tier per query, not O(n log n)

    seen_blocks = set()  # track which (l, block_idx) have already had their one-time sum paid

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

            # one-time full-block sum: paid the FIRST time this physical block
            # is ever touched by any query, ~0 marginal cost thereafter
            key = (l, block_idx)
            if key not in seen_blocks:
                seen_blocks.add(key)
                onetime_block_sum_terms += Bl * E * H  # O(Bl*D) paid once

            # reservoir-tau: O(reservoir_size) per active tier per query, not O(Bl)/O(n log n)
            reservoir_tau_terms += min(RESERVOIR_SIZE, Bl) * E * H

        pooled = torch.cat(per_tier_scores, dim=-1)
        shared_tau = torch.quantile(pooled, quantile, dim=-1, keepdim=True)  # correctness stand-in only
        for scores, (l, block_idx, Bl) in zip(per_tier_scores, per_tier_kv):
            survivors = scores >= shared_tau
            n_surv = int(survivors.sum().item())
            av_terms += n_surv
            survivor_gather_terms += n_surv  # residual sum(survivors): same order as AV gather
            has_non_surv = (~survivors).sum(dim=-1) > 0
            av_terms += int(has_non_surv.sum().item())

    return qk_terms, av_terms, survivor_gather_terms, onetime_block_sum_terms, reservoir_tau_terms


print("=== FLOP accounting, fast-residual + reservoir-tau (both fixes applied) ===")
print(f"{'N':>6} | {'dense terms':>12} | {'blended (QK+AV)':>16} | {'+ survivor-gather resid':>23} | {'+ onetime block-sum':>19} | {'+ reservoir-tau':>15} | {'FULL blended':>12}")

for N in (256, 512, 1024, 2048, 4096, 8192):
    g = torch.Generator().manual_seed(42)
    q = torch.randn(1, 1, N, D, generator=g) * 0.5
    k = torch.randn(1, 1, N, D, generator=g) * 0.5
    v = torch.randn(1, 1, N, Dv, generator=g) * 0.5

    qk_terms, av_terms, surv_gather_terms, onetime_terms, reservoir_terms = instrumented_run_fast(q, k, v, B=B, W_blocks=W_blocks)
    dense_terms = N * (N + 1) // 2
    dense_flops = dense_terms * 2 * (D + Dv)

    qk_flops = qk_terms * 2 * D
    av_flops = av_terms * 2 * Dv
    surv_gather_flops = surv_gather_terms * 2 * D  # residual sum(survivors), D-dim reduction
    onetime_flops = onetime_terms * 2 * D           # paid once per block, amortized across the whole sweep
    reservoir_flops = reservoir_terms * 1            # O(1)-ish per reservoir sample slot, dimension-independent

    blended_qkav = (qk_flops + av_flops) / dense_flops
    blended_with_resid = (qk_flops + av_flops + surv_gather_flops) / dense_flops
    blended_with_onetime = (qk_flops + av_flops + surv_gather_flops + onetime_flops) / dense_flops
    blended_full = (qk_flops + av_flops + surv_gather_flops + onetime_flops + reservoir_flops) / dense_flops

    print(f"{N:>6} | {dense_terms:>12} | {blended_qkav:>16.3f} | {blended_with_resid:>23.3f} | {blended_with_onetime:>19.3f} | {blended_full-blended_with_onetime:>15.4f} | {blended_full:>12.3f}")

print()
print("Reference: original (masked-reduction residual, sort-based tau) blended ratio was ~1.05-1.23.")
print("This shows the SAME kernel's true achievable cost once residual + tau are computed cheaply.")
