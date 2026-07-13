"""v2 of the ResidualGate probe (ma_meta_threshold_gated.py): the v1 gate
took ONLY the query as input and produced one scalar bias shared across
every tier's residual logit for that query -- diagnosed after training
as structurally unable to discriminate per-tier residual trustworthiness,
which is likely why it made zero measurable difference even under a
proper curriculum stress test (eval_gated_stress.py).

v2 gives the gate real per-tier context: query, the residual centroid
itself (mean_k), the non-survivor fraction (how much of the block's
content the centroid is actually summarizing), and the tier level. Still
frozen everywhere else (block structure, real-score survivor selection,
exact residual math) -- only the bias computation grows richer.
"""
import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class ResidualGateV2(nn.Module):
    """Per-tier learned bias: MLP([q_m, mean_k, non_surv_frac, tier_level])
    -> scalar, added to that tier's residual logit. Last layer zero-init
    so training starts identical to the frozen baseline."""

    def __init__(self, dim: int, max_tiers: int, hidden: int = 32):
        super().__init__()
        in_dim = dim * 2 + 1 + 1  # q_m, mean_k, non_surv_frac, tier_level (scalar, normalized)
        self.max_tiers = max_tiers
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, q_m: Tensor, mean_k: Tensor, non_surv_frac: Tensor, tier_level: int) -> Tensor:
        # q_m, mean_k: (E,H,B,D); non_surv_frac: (E,H,B,1)
        lvl = torch.full_like(non_surv_frac, tier_level / max(1, self.max_tiers - 1))
        feat = torch.cat([q_m, mean_k, non_surv_frac, lvl], dim=-1)
        return self.net(feat)  # (E,H,B,1)


def monarch_meta_threshold_gated_v2(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    B: int,
    W_blocks: int,
    gate: ResidualGateV2,
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

    sum_k_blocks, sum_v_blocks = {}, {}
    for l in range(L):
        Bl = B * (1 << l)
        Ml = N_padded // Bl
        k_block_l = k_flat.view(E, H, Ml, Bl, D)
        v_block_l = v_flat.view(E, H, Ml, Bl, Dv)
        sum_k_blocks[l] = k_block_l.sum(dim=-2)
        sum_v_blocks[l] = v_block_l.sum(dim=-2)

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

                surv_f = survivors.to(k_block.dtype)
                sum_surv_k = (surv_f.unsqueeze(-1) * k_block.unsqueeze(2)).sum(dim=-2)
                sum_surv_v = (surv_f.unsqueeze(-1) * v_block.unsqueeze(2)).sum(dim=-2)

                sum_full_k = sum_k_blocks[l][:, :, block_idx].unsqueeze(2)
                sum_full_v = sum_v_blocks[l][:, :, block_idx].unsqueeze(2)

                sum_ns_k = sum_full_k - sum_surv_k
                sum_ns_v = sum_full_v - sum_surv_v

                n_surv = surv_f.sum(dim=-1, keepdim=True)  # (E,H,B,1)
                count_ns = (Bl - n_surv).clamp(min=1)
                mean_k = sum_ns_k / count_ns
                mean_v = sum_ns_v / count_ns
                has_non_surv = (Bl - surv_f.sum(dim=-1)) > 0
                non_surv_frac = (Bl - n_surv) / Bl  # (E,H,B,1)

                gate_bias = gate(q_m, mean_k, non_surv_frac, l)  # (E,H,B,1)

                residual_logit = sm_scale * (mean_k * q_m).sum(dim=-1) + gate_bias.squeeze(-1)
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
