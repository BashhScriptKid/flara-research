"""NSA-inspired variant: replace the hard tau-threshold survivor/residual
split's implicit combination (whatever the raw softmax over
[local, survivor-logits, residual-logit] happens to produce) with a
small LEARNED per-query gate biasing how much weight the residual
centroid gets relative to real survivor scores -- the one piece of NSA's
design (arXiv 2502.11089) not yet tried here: a learned sigmoid-MLP gate
blending branches, as opposed to a fixed rule.

Everything else is untouched and frozen: block/tier structure, the
real-score-based survivor selection (shared-tau quantile), and the
exact-algebraic residual centroid math, all validated in
`ma_meta_threshold_fast_residual.py`. This file only adds one thing: a
learned scalar bias `gate_mlp(q_m)` added to each tier's residual_logit
before the shared softmax, trained end-to-end against ground-truth
causal attention output (see `train_meta_gate.py`).

Per NSA's own ablation (checked before building this: their comparison
of neural-importance-estimation selection vs real-score selection
favors real scores, matching this codebase's already-closed
trained-landmark-axis conclusion) -- this does NOT touch selection
itself, only the survivor/residual mixing weight, which NSA's
architecture doesn't have a direct analog for (its three branches are
compressed/selected/sliding, not survivor/residual within one branch).
This is an extrapolation of the "learned gate" idea to Meta's actual
structure, not a literal port.
"""
import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class ResidualGate(nn.Module):
    """Learned per-query scalar bias added to residual logits. Linear(D,1),
    initialized to output ~0 so the gated variant starts identical to the
    frozen baseline and only diverges as training finds a correction."""

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, 1, bias=True)
        nn.init.zeros_(self.w.weight)
        nn.init.zeros_(self.w.bias)

    def forward(self, q_m: Tensor) -> Tensor:
        return self.w(q_m)  # (..., 1)


def monarch_meta_threshold_gated(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    B: int,
    W_blocks: int,
    gate: ResidualGate,
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

            gate_bias = gate(q_m)  # (E,H,B,1)

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

                count_ns = (Bl - surv_f.sum(dim=-1, keepdim=True)).clamp(min=1)
                mean_k = sum_ns_k / count_ns
                mean_v = sum_ns_v / count_ns
                has_non_surv = (Bl - surv_f.sum(dim=-1)) > 0

                # the one addition vs the frozen baseline: a learned bias on
                # the residual logit, letting the model correct how much the
                # softmax trusts the residual centroid vs real survivors.
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
