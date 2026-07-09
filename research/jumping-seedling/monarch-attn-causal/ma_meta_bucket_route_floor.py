"""Floor-read rescue test (Fable's reprioritized #1): adds an always-on
mean-summary candidate per tier block, competing in the SAME joint
softmax as the routed bucket and the exact window -- not a fallback
triggered on low confidence, a permanent extra candidate.

Purpose: the adversarial mass-heavy-decoy test on ma_meta_bucket_route.py
found an 83.33% failure rate when a decoy is deliberately placed to
capture the local centroid away from the true needle -- genuine
centroid capture, not the natural ~2-8% near-boundary mis-routing rate
(different mechanism, confirmed via margin analysis). This tests
whether a permanent, query-informed-but-content-independent floor read
(a flat mean of the WHOLE block's keys/values, same failure family as
Version B's single-representative compression -- known individually
weak at detection) can still act as insurance: does adding it turn the
83% FAILURE (needle essentially absent from the output) into 83%
DILUTION (needle still contributes some weight, just outcompeted), or
does it stay a true miss? This determines whether the reliability
principle ("degrade in weight, never silently drop in-scope content")
can still be honestly claimed under adversarial content, not just
adversarial distance/count -- the thing Fable identified as the
load-bearing question before more design investment goes into routing
mitigations.

Structurally identical to ma_meta_bucket_route.py except: precompute
one extra (mean_key, mean_value) pair per tier block alongside the
bucket centroids, and add it as one more real, uncollapsed candidate in
the same joint softmax -- never pre-blended with the routed-bucket
candidate before that softmax.
"""

import math
from math import sqrt

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def _kmeans_buckets(k_block: Tensor, n_buckets: int, iters: int, sm_scale: float):
    Bl = k_block.shape[-2]
    n_buckets = min(n_buckets, Bl)
    idx0 = torch.linspace(0, Bl - 1, n_buckets, device=k_block.device).round().long().clamp(max=Bl - 1)
    centroids = k_block.index_select(-2, idx0).clone()
    assign = None
    for _ in range(iters):
        sims = sm_scale * (k_block @ centroids.transpose(-1, -2))
        assign = sims.argmax(dim=-1)
        onehot = F.one_hot(assign, n_buckets).to(k_block.dtype)
        sums = onehot.transpose(-1, -2) @ k_block
        counts = onehot.sum(dim=-2).unsqueeze(-1)
        new_centroids = sums / counts.clamp(min=1)
        centroids = torch.where(counts == 0, centroids, new_centroids)
    return centroids, assign


def monarch_meta_bucket_route_floor(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    B: int,
    W_blocks: int,
    n_buckets: int,
    use_floor: bool = True,
    kmeans_iters: int = 3,
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

    tier_centroids, tier_assign, tier_floor_k, tier_floor_v = [], [], [], []
    for l in range(L):
        Bl = B * (1 << l)
        Ml = N_padded // Bl
        k_l = k_flat.view(E, H, Ml, Bl, D)
        v_l = v_flat.view(E, H, Ml, Bl, Dv)
        centroids, assign = _kmeans_buckets(k_l, n_buckets, kmeans_iters, sm_scale)
        tier_centroids.append(centroids)
        tier_assign.append(assign)
        tier_floor_k.append(k_l.mean(dim=-2))  # (E,H,Ml,D) -- one flat mean per block
        tier_floor_v.append(v_l.mean(dim=-2))  # (E,H,Ml,Dv)

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
            k_block = k_flat.view(E, H, Ml, Bl, D)[:, :, block_idx]
            v_block = v_flat.view(E, H, Ml, Bl, Dv)[:, :, block_idx]
            centroids = tier_centroids[l][:, :, block_idx]
            assign = tier_assign[l][:, :, block_idx]

            centroid_scores = sm_scale * (q_m @ centroids.transpose(-1, -2))
            chosen_bucket = centroid_scores.argmax(dim=-1)

            full_scores = sm_scale * (q_m @ k_block.transpose(-1, -2))
            member_mask = assign.unsqueeze(2) == chosen_bucket.unsqueeze(-1)
            scores = full_scores.masked_fill(~member_mask, -float("inf"))

            sub_max = scores.max(dim=-1, keepdim=True).values
            sub_max = torch.nan_to_num(sub_max, neginf=0.0)
            sub_exp = torch.nan_to_num(torch.exp(scores - sub_max), nan=0.0)
            level_logit = sub_max.squeeze(-1) + torch.log(sub_exp.sum(-1) + eps)
            sub_weights = sub_exp / (sub_exp.sum(-1, keepdim=True) + eps)
            level_value = sub_weights @ v_block
            tier_logits.append(level_logit)
            tier_values.append(level_value)

            if use_floor:
                floor_k = tier_floor_k[l][:, :, block_idx]  # (E,H,D)
                floor_v = tier_floor_v[l][:, :, block_idx]  # (E,H,Dv)
                floor_logit = sm_scale * (q_m @ floor_k.unsqueeze(-1)).squeeze(-1)  # (E,H,B)
                floor_value = floor_v.unsqueeze(2).expand(E, H, B, Dv)  # (E,H,B,Dv)
                tier_logits.append(floor_logit)
                tier_values.append(floor_value)

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
