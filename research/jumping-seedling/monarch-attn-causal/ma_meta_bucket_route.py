"""Sketch/hash follow-up to the query-awareness confirmation: is there a
cheap, fixed(er)-cost, QUERY-INFORMED operation that approximates full
block-local exact attention without paying O(B_l) per query?

Key distinction from Axis 1's failed landmarks: those precomputed a
STATIC VALUE with zero knowledge of any future query -- the query never
gets a say in what gets summarized. This mechanism precomputes only
STRUCTURE (a k-means partition of the block's keys into `n_buckets`
buckets, done once, reused by every query touching that block -- same
"compute once, share across queries" cost shape as everything else
tonight), then at QUERY TIME the query itself picks which bucket to
read (via similarity to the bucket centroids) and gets REAL exact
attention over that bucket's actual member keys -- not a static summary
vector. The query is genuinely informing what it sees, just scoped to a
subset instead of the whole block.

COST CAVEAT, stated up front (same one flagged for the ANN top-k
reference implementation earlier tonight): this implementation computes
scores against ALL B_l keys and MASKS to the selected bucket, rather
than skipping computation for non-bucket keys entirely. That means it
correctly tests RECALL/correctness, but its wall-clock numbers do NOT
demonstrate the real cost savings a proper sparse-gather implementation
would show -- reported honestly as a recall-only probe, not a
performance benchmark, unless stated otherwise.
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
        sims = sm_scale * (k_block @ centroids.transpose(-1, -2))  # (...,Bl,n_buckets)
        assign = sims.argmax(dim=-1)  # (...,Bl)
        onehot = F.one_hot(assign, n_buckets).to(k_block.dtype)  # (...,Bl,n_buckets)
        sums = onehot.transpose(-1, -2) @ k_block
        counts = onehot.sum(dim=-2).unsqueeze(-1)
        new_centroids = sums / counts.clamp(min=1)
        centroids = torch.where(counts == 0, centroids, new_centroids)
    return centroids, assign  # (...,n_buckets,D), (...,Bl)


def monarch_meta_bucket_route(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    B: int,
    W_blocks: int,
    n_buckets: int,
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

    # precompute bucket STRUCTURE only (centroids + assignment), once per
    # tier block, shared across every query that later reads it. No value
    # is precomputed -- the query always reads real member keys at read time.
    tier_centroids, tier_assign = [], []
    for l in range(L):
        Bl = B * (1 << l)
        Ml = N_padded // Bl
        k_l = k_flat.view(E, H, Ml, Bl, D)
        centroids, assign = _kmeans_buckets(k_l, n_buckets, kmeans_iters, sm_scale)
        tier_centroids.append(centroids)  # (E,H,Ml,n_buckets,D)
        tier_assign.append(assign)  # (E,H,Ml,Bl)

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
            centroids = tier_centroids[l][:, :, block_idx]  # (E,H,n_buckets,D)
            assign = tier_assign[l][:, :, block_idx]  # (E,H,Bl)

            # query picks its own bucket (query-informed routing)
            centroid_scores = sm_scale * (q_m @ centroids.transpose(-1, -2))  # (E,H,B,n_buckets)
            chosen_bucket = centroid_scores.argmax(dim=-1)  # (E,H,B)

            # exact attention scoped to the chosen bucket's real members
            # (COST CAVEAT: computed via full-scores-then-mask -- see docstring)
            full_scores = sm_scale * (q_m @ k_block.transpose(-1, -2))  # (E,H,B,Bl)
            member_mask = assign.unsqueeze(2) == chosen_bucket.unsqueeze(-1)  # (E,H,B,Bl)
            scores = full_scores.masked_fill(~member_mask, -float("inf"))

            sub_max = scores.max(dim=-1, keepdim=True).values
            sub_max = torch.nan_to_num(sub_max, neginf=0.0)
            sub_exp = torch.nan_to_num(torch.exp(scores - sub_max), nan=0.0)
            level_logit = sub_max.squeeze(-1) + torch.log(sub_exp.sum(-1) + eps)
            sub_weights = sub_exp / (sub_exp.sum(-1, keepdim=True) + eps)
            level_value = sub_weights @ v_block
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
