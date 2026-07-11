"""Bounding-ball pruned threshold selection (Fable's correction to the
item-2 framing): the original Stage-1 recommendation this whole thread
has been chasing was "threshold/bounding-ball test -- skip a block only
if PROVABLY below tau, contribute centroid instead of nothing for
skipped blocks." ma_meta_threshold.py implemented the centroid half but
not the actual pruning half -- it scores every real key in every active
tier's block, then applies tau as a filter on already-computed scores.
That is threshold selection as a MASKING RULE, not as a RANGE-SEARCH
ALGORITHM, and it is the direct, correct reason the FLOP accounting for
that version came out O(N) per query / O(N^2) total -- same complexity
class as dense attention.

This variant restores the actual bound test: for each tier's block,
precompute a bounding ball (center = mean of keys, radius = max
distance from center to any key in the block) ONCE, query-independently.
At read time, test ONE O(1) inequality per block, by Cauchy-Schwarz:

    <q, k> <= <q, center> + ||q|| * radius   for every k in the ball

so if sm_scale * (<q,center> + ||q||*radius) < tau, NO key in that
block can possibly clear tau -- the block is PROVABLY prunable, and its
only contribution is the (already-known, precomputed) block-mean
centroid. No per-key scoring needed for a proven-prunable block.

STRUCTURAL PREREQUISITE this variant must satisfy that the earlier
per-tier-relative-quantile design did not: the bound test can only be
evaluated against an ABSOLUTE, PRE-SCORE tau, not a per-block quantile
computed FROM that block's own scores (chicken-and-egg -- you cannot
prove a block is below a threshold that is only defined after you have
already scored it). So this variant seeds ONE shared, absolute tau per
query from the local window's scores (the local window is always
scored exactly and causally regardless of any pruning decision, so it
is a real, already-available per-query score sample) and uses that same
tau_seed as BOTH the pruning threshold and the survivor threshold for
every tier -- replacing the original per-tier local 90th-percentile
entirely, not layering pruning on top of it unchanged.

This build still computes real per-key scores for every candidate block
regardless of the prune decision (correctness-focused: verifying the
bound test agrees with real scoring, and measuring how often it WOULD
have let you skip scoring, not actually skipping the PyTorch compute --
cost is the separate, later analytical question, per the same
methodology note used throughout this session for reference-
implementation timing). Prune decisions are enforced onto survivorship
(prunable => zero survivors) and instrumented via a running counter so
prune rate can be measured directly.
"""

import math
from math import sqrt

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def monarch_meta_threshold_ball_prune(
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

    # --- precompute per-tier bounding balls + centroids, ONCE, query-independent ---
    centers, radii, mean_vs = {}, {}, {}
    for l in range(L):
        Bl = B * (1 << l)
        Ml = N_padded // Bl
        k_block_l = k_flat.view(E, H, Ml, Bl, D)
        v_block_l = v_flat.view(E, H, Ml, Bl, Dv)
        center_l = k_block_l.mean(dim=-2)  # (E,H,Ml,D)
        radius_l = (k_block_l - center_l.unsqueeze(-2)).norm(dim=-1).max(dim=-1).values  # (E,H,Ml)
        mean_v_l = v_block_l.mean(dim=-2)  # (E,H,Ml,Dv)
        centers[l] = center_l
        radii[l] = radius_l
        mean_vs[l] = mean_v_l

    n_tested = 0
    n_pruned = 0

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

        # tau_seed: single absolute per-query threshold, seeded from the
        # always-exact local window's own real score distribution --
        # available BEFORE any tier is touched, unlike a per-tier quantile.
        local_scores_nan = local_scores.masked_fill(~win_mask.view(1, 1, B, -1), float("nan"))
        tau_seed = torch.nanquantile(local_scores_nan, quantile, dim=-1, keepdim=True)  # (E,H,B,1)

        n = m0 - W_blocks + 1
        candidates = []
        if n > 0:
            for l in range(L):
                if (n >> l) & 1:
                    candidates.append((l, (n >> (l + 1)) << 1))

        q_norm = q_m.norm(dim=-1, keepdim=True)  # (E,H,B,1)

        tier_logits, tier_values = [], []
        for l, block_idx in candidates:
            Bl = B * (1 << l)
            Ml = N_padded // Bl
            if block_idx >= Ml:
                continue

            center = centers[l][:, :, block_idx]  # (E,H,D)
            radius = radii[l][:, :, block_idx]  # (E,H)
            mean_v_block = mean_vs[l][:, :, block_idx]  # (E,H,Dv)

            # Cauchy-Schwarz upper bound on the max real score in this block
            bound = sm_scale * (
                (q_m * center.unsqueeze(2)).sum(-1, keepdim=True)
                + q_norm * radius.view(E, H, 1, 1)
            )  # (E,H,B,1)
            prunable = bound < tau_seed  # (E,H,B,1) bool

            n_tested += prunable.numel()
            n_pruned += int(prunable.sum().item())

            k_block = k_flat.view(E, H, Ml, Bl, D)[:, :, block_idx]  # (E,H,Bl,D)
            v_block = v_flat.view(E, H, Ml, Bl, Dv)[:, :, block_idx]  # (E,H,Bl,Dv)

            # real scoring still computed here for correctness verification only
            # (this build measures the prune decision, it does not yet skip the
            # PyTorch compute for proven-prunable blocks -- see module docstring)
            scores = sm_scale * (q_m @ k_block.transpose(-1, -2))  # (E,H,B,Bl)
            survivors = scores >= tau_seed
            survivors = survivors & (~prunable)  # enforce: prunable => zero survivors

            tier_logits.append(scores.masked_fill(~survivors, -float("inf")))
            tier_values.append(v_block.unsqueeze(2).expand(E, H, B, Bl, Dv))

            non_surv = ~survivors
            non_surv_f = non_surv.to(k_block.dtype)
            count_ns = non_surv_f.sum(dim=-1, keepdim=True).clamp(min=1)
            mean_k = (non_surv_f.unsqueeze(-1) * k_block.unsqueeze(2)).sum(dim=-2) / count_ns
            mean_v = (non_surv_f.unsqueeze(-1) * v_block.unsqueeze(2)).sum(dim=-2) / count_ns
            has_non_surv = non_surv_f.sum(dim=-1) > 0
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
    prune_stats = {"n_tested": n_tested, "n_pruned": n_pruned, "prune_rate": n_pruned / n_tested if n_tested else 0.0}
    return z[..., :N, :], prune_stats
