"""Full integration test (Fable's item 1): Fenwick dyadic tier selection
+ threshold-selection read (real exact attention over survivors, per
tier block) + Multipole-style residual-centroid contribution for
non-survivor (proven-below-tau) keys, so nothing is silently dropped +
the exact SlidingMonarchAttention local window. All assembled and run
together for the first time -- every prior threshold-selection result
was measured on an isolated single block, not this composed system.

DELIBERATE ARCHITECTURAL CHANGE, stated up front rather than buried:
this build has NO Monarch/Sinkhorn T-iteration refinement anywhere.
Reasoning, derived while designing this (not assumed): Monarch's T-
iteration machinery exists to build a GOOD approximate representative
for a block cheaply, avoiding an O(B_l) real read. Threshold selection
already reads REAL keys directly for the (small) survivor set -- so
once every tier is read via threshold selection instead of a Monarch
representative, there is no more Monarch representative left to refine.
T becomes structurally unused, not just less important. This is a
genuinely different, stronger claim than "threshold selection reduces
dependence on T" -- it says the Monarch machinery this whole
investigation started from may no longer be load-bearing once threshold
selection replaces its read mechanism. Testing this claim directly by
building the version WITHOUT it and checking whether quality holds or
improves relative to every prior threshold-selection result, not just
asserting the simplification is free.

Per-tier mechanics: for each Fenwick-selected candidate block, compute
exact scores for all its real keys (correctness-focused build, cost is
item 2's separate analytical question), select survivors via a
reservoir-estimated tau (90th percentile of that block's own scores,
matching every prior threshold-selection experiment). Survivors become
UNCOLLAPSED real candidates in the outer joint softmax (real logit,
real value, exactly like exact top-k). Non-survivors contribute ONE
additional residual-centroid candidate (mean key, mean value of the
sub-threshold keys) so the block's sub-threshold mass is never silently
dropped -- Multipole-style, but note its role here is calibration
(representing "there was some low-relevance background here"), not
needle-rescue, since a genuine needle is expected to already clear tau
and become a survivor (matches the near-0% exclusion floor already
established).
"""

import math
from math import sqrt

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def monarch_meta_threshold(
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

        tier_logits, tier_values = [], []
        for l, block_idx in candidates:
            Bl = B * (1 << l)
            Ml = N_padded // Bl
            if block_idx >= Ml:
                continue
            k_block = k_flat.view(E, H, Ml, Bl, D)[:, :, block_idx]  # (E,H,Bl,D)
            v_block = v_flat.view(E, H, Ml, Bl, Dv)[:, :, block_idx]  # (E,H,Bl,Dv)

            scores = sm_scale * (q_m @ k_block.transpose(-1, -2))  # (E,H,B,Bl)
            tau = torch.quantile(scores, quantile, dim=-1, keepdim=True)  # (E,H,B,1)
            survivors = scores >= tau

            # survivor candidates: real logits, real values, uncollapsed
            surv_scores = scores.masked_fill(~survivors, -float("inf"))
            surv_max = torch.clamp(surv_scores.max(dim=-1, keepdim=True).values, min=-1e30)
            surv_max = torch.nan_to_num(surv_max, neginf=0.0)
            surv_exp = torch.nan_to_num(torch.exp(surv_scores - surv_max), nan=0.0)
            # keep survivors uncollapsed: append their (max-shifted) logits/values directly
            tier_logits.append((scores.masked_fill(~survivors, -float("inf"))))  # (E,H,B,Bl)
            tier_values.append(v_block.unsqueeze(2).expand(E, H, B, Bl, Dv))

            # residual centroid: mean of NON-survivor keys/values, one extra candidate
            non_surv = ~survivors
            non_surv_f = non_surv.to(k_block.dtype)
            count_ns = non_surv_f.sum(dim=-1, keepdim=True).clamp(min=1)  # (E,H,B,1)
            mean_k = (non_surv_f.unsqueeze(-1) * k_block.unsqueeze(2)).sum(dim=-2) / count_ns  # (E,H,B,D)
            mean_v = (non_surv_f.unsqueeze(-1) * v_block.unsqueeze(2)).sum(dim=-2) / count_ns  # (E,H,B,Dv)
            has_non_surv = (non_surv_f.sum(dim=-1) > 0)  # (E,H,B)
            residual_logit = sm_scale * (mean_k * q_m).sum(dim=-1)  # (E,H,B)
            residual_logit = residual_logit.masked_fill(~has_non_surv, -float("inf"))
            tier_logits.append(residual_logit.unsqueeze(-1))  # (E,H,B,1)
            tier_values.append(mean_v.unsqueeze(-2))  # (E,H,B,1,Dv)

        if tier_logits:
            all_tier_logits = torch.cat(tier_logits, dim=-1)  # (E,H,B, sum of Bl's + n_tiers)
            all_tier_values = torch.cat(tier_values, dim=-2)  # (E,H,B, same, Dv)
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
            tier_w = exp_c[..., win_len:]  # (E,H,B, sum)
            num_tier = (tier_w.unsqueeze(-1) * all_tier_values).sum(dim=-2)
        else:
            num_tier = 0.0

        out_m = (num_local + num_tier) / denom
        outputs.append(out_m)

    z = torch.stack(outputs, dim=2).view(E, H, N_padded, Dv)
    return z[..., :N, :]
