"""Exact-algebraic residual-centroid fix, per Fable: the current
residual centroid (mean of NON-survivor keys/values) is computed via a
masked reduction over ALL Bl keys in a tier's block -- O(Bl*D) FLOPs
per tier per query, the cost that was found to erase the entire
AV-stage FLOP saving. This is not a new content-approximation heuristic
(the earlier floor-read/robust-centroid fixes that failed) -- it is the
SAME exact mathematical quantity, computed a cheaper way:

    sum(non-survivors) = sum(all keys in block) - sum(survivors)

sum(all keys in block) is query-independent -- identical for every
query that touches this tier's block -- so it is precomputed ONCE when
the block finalizes (paid a single O(Bl*D) cost, amortized to ~0 over
every future query that reads it), the same reuse pattern already
validated for bounding balls and k-means cluster centroids.

sum(survivors) costs O(num_survivors*D), the same order as work already
paid for the real AV gather (survivors are already the small ~10% set
being value-weighted into the output) -- NOT a new O(Bl*D) pass.

This file computes the mean_k/mean_v exactly as before (masked
reduction, for numerical-correctness verification against the existing
implementation) AND exposes the count of operations a real gather-based
implementation would need (O(num_survivors*D), not O(Bl*D)) so the FLOP
accounting can reflect the real achievable cost rather than the
PyTorch-masked-reduction stand-in used for correctness testing only --
same "PyTorch timing/masking is correctness scaffolding, not a cost
verdict" discipline used throughout this session.
"""

import math
from math import sqrt

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def monarch_meta_threshold_fast_residual(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    B: int,
    W_blocks: int,
    quantile: float = 0.90,
    eps: float = 1e-6,
):
    E, H, N, D = q.shape
    _, _, _, Dv = v.shape
    sm_scale = 1 / sqrt(D)

    M_base_needed = (N + B - 1) // B
    L = max(1, math.ceil(math.log2(max(M_base_needed, 2))))
    N_padded = B * (1 << L)
    pad = N_padded - N

    qb = F.pad(q, (0, 0, 0, pad)).view(E, H, -1, B, D)
    kb = F.pad(k, (0, 0, 0, pad)).view(E, H, -1, B, D)
    vb = F.pad(v, (0, 0, 0, pad)).view(E, H, -1, B, Dv)
    M = qb.shape[2]

    range_n = torch.arange(N_padded, device=q.device).view(M, B)
    valid_mb = range_n < N

    k_flat = F.pad(k, (0, 0, 0, pad))
    v_flat = F.pad(v, (0, 0, 0, pad))

    # --- precompute per-tier full-block sums, ONCE, query-independent ---
    sum_k_blocks, sum_v_blocks = {}, {}
    for l in range(L):
        Bl = B * (1 << l)
        Ml = N_padded // Bl
        k_block_l = k_flat.view(E, H, Ml, Bl, D)
        v_block_l = v_flat.view(E, H, Ml, Bl, Dv)
        sum_k_blocks[l] = k_block_l.sum(dim=-2)  # (E,H,Ml,D)
        sum_v_blocks[l] = v_block_l.sum(dim=-2)  # (E,H,Ml,Dv)

    n_survivor_gather_ops = 0  # instrumentation: O(num_survivors*D) real cost, not O(Bl*D)

    outputs = []
    for m0 in range(M):
        q_m = qb[:, :, m0]

        w_start = max(0, m0 - W_blocks + 1)
        win_k = kb[:, :, w_start : m0 + 1].reshape(E, H, -1, D)
        win_v = vb[:, :, w_start : m0 + 1].reshape(E, H, -1, Dv)
        n_win_blocks = m0 - w_start + 1
        win_valid = valid_mb[w_start : m0 + 1].reshape(-1)
        blk_idx = torch.arange(n_win_blocks, device=q.device).repeat_interleave(B)
        own_blk = n_win_blocks - 1
        intra = torch.arange(B, device=q.device).repeat(n_win_blocks)
        causal_win = (blk_idx < own_blk).unsqueeze(0) | (
            (blk_idx == own_blk).unsqueeze(0) & (intra.unsqueeze(0) <= torch.arange(B, device=q.device).unsqueeze(1))
        )
        win_mask = causal_win & win_valid.view(1, -1)
        local_scores = sm_scale * (q_m @ win_k.transpose(-1, -2))
        local_scores = local_scores.masked_fill(~win_mask.view(1, 1, B, -1), -float("inf"))

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
            k_block = k_flat.view(E, H, Ml, Bl, D)[:, :, block_idx]
            v_block = v_flat.view(E, H, Ml, Bl, Dv)[:, :, block_idx]
            scores = sm_scale * (q_m @ k_block.transpose(-1, -2))
            per_tier_scores.append(scores)
            per_tier_kv.append((l, block_idx, k_block, v_block))

        tier_logits, tier_values = [], []
        if per_tier_scores:
            pooled_scores = torch.cat(per_tier_scores, dim=-1)
            shared_tau = torch.quantile(pooled_scores, quantile, dim=-1, keepdim=True)

            for scores, (l, block_idx, k_block, v_block) in zip(per_tier_scores, per_tier_kv):
                Bl = B * (1 << l)
                survivors = scores >= shared_tau

                tier_logits.append(scores.masked_fill(~survivors, -float("inf")))
                tier_values.append(v_block.unsqueeze(2).expand(E, H, B, Bl, Dv))

                # --- fast residual via exact subtraction ---
                surv_f = survivors.to(k_block.dtype)  # (E,H,B,Bl)
                # sum(survivors): mathematically identical to a real gather-and-sum
                # over the ~10% survivor set -- computed here via masked reduction
                # for numerical correctness verification (matches PyTorch-timing-
                # is-correctness-scaffolding convention used throughout this
                # session), while instrumentation below counts it at its REAL
                # achievable cost, O(num_survivors*D), not O(Bl*D).
                sum_surv_k = (surv_f.unsqueeze(-1) * k_block.unsqueeze(2)).sum(dim=-2)  # (E,H,B,D)
                sum_surv_v = (surv_f.unsqueeze(-1) * v_block.unsqueeze(2)).sum(dim=-2)  # (E,H,B,Dv)
                n_survivor_gather_ops += int(surv_f.sum().item())

                sum_full_k = sum_k_blocks[l][:, :, block_idx].unsqueeze(2)  # (E,H,1,D)
                sum_full_v = sum_v_blocks[l][:, :, block_idx].unsqueeze(2)  # (E,H,1,Dv)

                sum_ns_k = sum_full_k - sum_surv_k  # (E,H,B,D)
                sum_ns_v = sum_full_v - sum_surv_v  # (E,H,B,Dv)

                count_ns = (Bl - surv_f.sum(dim=-1, keepdim=True)).clamp(min=1)  # (E,H,B,1)
                mean_k = sum_ns_k / count_ns
                mean_v = sum_ns_v / count_ns
                has_non_surv = (Bl - surv_f.sum(dim=-1)) > 0

                residual_logit = sm_scale * (mean_k * q_m).sum(dim=-1)
                residual_logit = residual_logit.masked_fill(~has_non_surv, -float("inf"))
                tier_logits.append(residual_logit.unsqueeze(-1))
                tier_values.append(mean_v.unsqueeze(-2))

        if tier_logits:
            all_tier_logits = torch.cat(tier_logits, dim=-1)
            all_tier_values = torch.cat(tier_values, dim=-2)
            combined = torch.cat([local_scores, all_tier_logits], dim=-1)
        else:
            all_tier_values = None
            combined = local_scores

        row_max = torch.clamp(combined.max(dim=-1, keepdim=True).values, min=-1e30)
        row_max = torch.nan_to_num(row_max, neginf=0.0)
        exp_c = torch.nan_to_num(torch.exp(combined - row_max), nan=0.0)
        denom = exp_c.sum(dim=-1, keepdim=True) + eps

        win_len = win_k.shape[-2]
        local_w = exp_c[..., :win_len]
        num_local = local_w @ win_v

        if tier_logits:
            tier_w = exp_c[..., win_len:]
            num_tier = (tier_w.unsqueeze(-1) * all_tier_values).sum(dim=-2)
        else:
            num_tier = 0.0

        out_m = (num_local + num_tier) / denom
        outputs.append(out_m)

    z = torch.stack(outputs, dim=2).view(E, H, N_padded, Dv)
    return z[..., :N, :], n_survivor_gather_ops
