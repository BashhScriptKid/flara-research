"""Query-awareness isolation probe. Same window + binary-decomposition
tier selection as ma_meta_monarch.py (unchanged, already verified exact/
gap-free/leak-free), but the tier read step is replaced entirely: instead
of reading a PRECOMPUTED landmark representative (built with zero
knowledge of any future query -- the shared property across all five
Axis-1 mechanics that failed), each query does REAL, fresh, block-local
exact attention over the selected block's actual B_l keys. No
precomputation, no compression -- the query looks directly.

This isolates one variable: does restoring query-awareness alone fix
Axis 1's uniform detection failure, independent of which heuristic was
used to pick landmarks? If yes, that confirms query-awareness (not
heuristic choice) was the load-bearing variable. If it still fails,
something else is broken too.

Cost is the deliberate, named tradeoff: this is O(B_l) per query per
active tier (no precomputation to amortize), the exact cost the
landmark-compression attempt in ma_meta_monarch.py existed to avoid.
"""

import math
from math import sqrt

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def monarch_meta_query_aware(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    B: int,
    W_blocks: int,
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

    # NO precompute loop here -- that's the whole point. Tier blocks are
    # read directly, per query, at read time below.

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

        tier_logits, tier_values = [], []
        for l, block_idx in candidates:
            Bl = B * (1 << l)
            Ml = N_padded // Bl
            if block_idx >= Ml:
                continue
            k_block = k_flat.view(E, H, Ml, Bl, D)[:, :, block_idx]  # (E,H,Bl,D) -- real keys, fresh read
            v_block = v_flat.view(E, H, Ml, Bl, Dv)[:, :, block_idx]  # (E,H,Bl,Dv)

            sub_logits = sm_scale * (q_m @ k_block.transpose(-1, -2))  # (E,H,B,Bl) -- REAL query attends ALL Bl keys
            sub_max = sub_logits.max(dim=-1, keepdim=True).values
            sub_exp = torch.exp(sub_logits - sub_max)
            level_logit = sub_max.squeeze(-1) + torch.log(sub_exp.sum(-1) + eps)  # (E,H,B)
            sub_weights = sub_exp / (sub_exp.sum(-1, keepdim=True) + eps)
            level_value = sub_weights @ v_block  # (E,H,B,Dv)
            tier_logits.append(level_logit)
            tier_values.append(level_value)

        if tier_logits:
            tier_logit_t = torch.stack(tier_logits, dim=-1)
            tier_value_t = torch.stack(tier_values, dim=-2)
            combined = torch.cat([local_scores, tier_logit_t], dim=-1)
        else:
            tier_value_t = None
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
            num_tier = (tier_w.unsqueeze(-1) * tier_value_t).sum(dim=-2)
        else:
            num_tier = 0.0

        out_m = (num_local + num_tier) / denom
        outputs.append(out_m)

    z = torch.stack(outputs, dim=2).view(E, H, N_padded, Dv)
    return z[..., :N, :]
