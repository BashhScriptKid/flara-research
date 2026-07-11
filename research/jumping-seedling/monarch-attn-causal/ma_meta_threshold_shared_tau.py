"""Shared-tau variant (Fable's cheap, worth-doing-now fix): instead of
each active Fenwick tier independently computing its own local 90th-
percentile threshold -- the mechanism the tier-concentration diagnosis
showed lets four tiers each admit ~10% with zero awareness of the other
three, summing to 50% overall -- pool all active tiers' scores for a
given query into ONE combined distribution and compute a SINGLE shared
quantile threshold from that pool, then apply the same tau to every
tier's own survivor selection.

Still an ABSOLUTE threshold predicate (not rank-truncation), so this
does not reintroduce the exclusion cliff top-k already demonstrated
twice in this session -- it only removes the "tiers have zero awareness
of siblings" gap the diagnostic found. Not expected to close the
crafted-adversarial gap (that's accepted as unfixable at the selection
layer, per the dense-attention-control finding), only to narrow the
cross-tier accumulation effect.
"""

import math
from math import sqrt

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def monarch_meta_threshold_shared_tau(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    B: int,
    W_blocks: int,
    quantile: float = 0.90,
    eps: float = 1e-6,
) -> Tensor:
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

    causal_local = torch.tril(torch.ones(B, B, dtype=torch.bool, device=q.device))

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

        # --- pass 1: compute scores for every active tier, pool for shared tau ---
        per_tier_scores = []
        per_tier_kv = []
        for l, block_idx in candidates:
            Bl = B * (1 << l)
            Ml = N_padded // Bl
            if block_idx >= Ml:
                continue
            k_block = k_flat.view(E, H, Ml, Bl, D)[:, :, block_idx]
            v_block = v_flat.view(E, H, Ml, Bl, Dv)[:, :, block_idx]
            scores = sm_scale * (q_m @ k_block.transpose(-1, -2))  # (E,H,B,Bl)
            per_tier_scores.append(scores)
            per_tier_kv.append((k_block, v_block))

        tier_logits, tier_values = [], []
        if per_tier_scores:
            pooled_scores = torch.cat(per_tier_scores, dim=-1)  # (E,H,B, sum of Bl's)
            shared_tau = torch.quantile(pooled_scores, quantile, dim=-1, keepdim=True)  # (E,H,B,1)

            # --- pass 2: apply the SAME shared tau to every tier's own scores ---
            for (l, block_idx), scores, (k_block, v_block) in zip(candidates, per_tier_scores, per_tier_kv):
                Bl = B * (1 << l)
                survivors = scores >= shared_tau

                tier_logits.append(scores.masked_fill(~survivors, -float("inf")))
                tier_values.append(v_block.unsqueeze(2).expand(E, H, B, Bl, Dv))

                non_surv = ~survivors
                non_surv_f = non_surv.to(k_block.dtype)
                count_ns = non_surv_f.sum(dim=-1, keepdim=True).clamp(min=1)
                mean_k = (non_surv_f.unsqueeze(-1) * k_block.unsqueeze(2)).sum(dim=-2) / count_ns
                mean_v = (non_surv_f.unsqueeze(-1) * v_block.unsqueeze(2)).sum(dim=-2) / count_ns
                has_non_surv = (non_surv_f.sum(dim=-1) > 0)
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
    return z[..., :N, :]
