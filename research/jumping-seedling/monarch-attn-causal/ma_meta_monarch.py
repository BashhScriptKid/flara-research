"""MetaMonarchAttention: multi-scale hierarchical extension of
SlidingMonarchAttention. See chat log for the full design derivation
(three real design problems found and resolved in sequence: naive
reach-based block boundaries create gaps/overlaps; fixed by a
Fenwick-tree-style canonical binary decomposition of the causal prefix;
that in turn had a hidden O(N^2/B) read-cost wall because Monarch's own
per-block representative count (B_l, growing with level) was being used
directly at read time; fixed by decoupling read cost from block size via
a fixed small R representatives-per-block, built once and reused by
every downstream query -- NOT recomputed per query).

Structure: local exact window (identical to SlidingMonarchAttention) +
a fixed number of hierarchy tiers l=0..L-1, tier l using GLOBALLY FIXED
blocks of size B_l = B * 2^l. For a query in base block m0, which tier
blocks are valid causal candidates is determined ENTIRELY by the binary
representation of n = m0 - W_blocks + 1 (the causal prefix length before
the window, in base-block units) -- geometry only, zero content/rank
dependence: tier l contributes a candidate iff bit l of n is set, and
the candidate's block index (in tier l's own fixed grid) is exactly
`(n >> (l+1)) << 1`. Verified by hand on n=13 and n=20 before writing
any code (see chat).

Per-block compression (the R-representative fix): each tier-l block is
compressed, ONCE, into R fixed "landmark" representatives via genuine
softmax attention -- NOT a flat mean (that would be Version B's
mean-pooling failure again) and NOT all B_l raw sub-representatives
(that reintroduces the O(B_l) read-cost wall). Landmarks are R
evenly-spaced real keys from within the block, each attending over all
B_l keys in the block (Nystrom-style: reuse real content as the
"pseudo-query" rather than needing trained landmark parameters, since
this is an untrained probe). This is a REAL, acknowledged, unresolved
partial risk: R fixed slots can capture at most R genuinely-distinct
sharp signals per block; a block with more than R separately-important
needles will see collisions. That's exactly what eval_meta_stress.py is
built to measure, not assume.

Combination: for a query, window logits (exact, per-token) and each
active tier's SINGLE combined logit (via logsumexp over that tier's R
landmark sub-logits -- an EXACT composition, not an approximation, of
"attend over R landmarks with weights w, contribute logsumexp as this
tier's outer-softmax logit") go into ONE joint softmax, output combines
window values and each tier's landmark-weighted value accordingly.

NOT included in this build (explicitly out of scope, to keep this
tractable as a stress-test-focused first version): the T-iteration
cross-block refinement that SlidingMonarchAttention's far branch used.
Landmark compression here is purely local to each block's own content --
no cross-block mixing. T was found to be a strong lever for
SlidingMonarchAttention's far-branch quality; omitting it here is a
real simplification, not a claim that it wouldn't help, just orthogonal
to the R-capacity question this build is testing.
"""

import math
from math import sqrt

import torch
import torch.nn.functional as F

from landmark_mechanics import landmarks_random_reuse

Tensor = torch.Tensor


def monarch_meta(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    B: int,
    W_blocks: int,
    R: int,
    landmark_fn=landmarks_random_reuse,
    structure: str = "binary",
    eps: float = 1e-6,
) -> Tensor:
    """landmark_fn: (k_block,v_block,R,sm_scale) -> (al_R,y_R), any of the
    functions in landmark_mechanics.py (or a custom one with the same
    signature). structure: 'binary' (Fenwick-tree-style, tonight's
    verified-correct decomposition, each query gets a variable but exact,
    gap-free/overlap-free set of tier candidates) or 'kary' (always uses
    the single immediately-preceding block at every level, deterministic
    fixed candidate count -- simpler, but reintroduces the gaps/overlaps
    the binary decomposition was specifically built to fix; kept as a
    deliberate comparison point, not a recommended default)."""
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

    tier_al, tier_y = [], []
    for l in range(L):
        Bl = B * (1 << l)
        Ml = N_padded // Bl
        k_l = k_flat.view(E, H, Ml, Bl, D)
        v_l = v_flat.view(E, H, Ml, Bl, Dv)
        al_R, y_R = landmark_fn(k_l, v_l, R, sm_scale)  # (E,H,Ml,R,D)/(E,H,Ml,R,Dv)
        tier_al.append(al_R)
        tier_y.append(y_R)

    causal_local = torch.tril(torch.ones(B, B, dtype=torch.bool, device=q.device))

    outputs = []
    for m0 in range(M):
        q_m = qb[:, :, m0]
        valid_m = valid_mb[m0].view(1, 1, 1, B)

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

        n = m0 - W_blocks + 1  # causal prefix length before the window, in base blocks
        tier_logits, tier_values = [], []
        if structure == "binary":
            candidates = []
            if n > 0:
                for l in range(L):
                    if (n >> l) & 1:
                        candidates.append((l, (n >> (l + 1)) << 1))
        elif structure == "kary":
            # deterministic: always the single immediately-preceding block
            # at every level (if causally valid) -- simpler, fixed
            # candidate count, but reintroduces gaps/overlaps (see docstring).
            candidates = []
            for l in range(L):
                c_l = m0 // (1 << l)
                block_idx = c_l - 1
                if block_idx >= 0 and (block_idx + 1) * (1 << l) <= w_start:
                    candidates.append((l, block_idx))
        else:
            raise ValueError(f"unknown structure {structure!r}")

        for l, block_idx in candidates:
            Ml = M // (1 << l)
            if block_idx < Ml:
                al_R = tier_al[l][:, :, block_idx]  # (E,H,R,D)
                y_R = tier_y[l][:, :, block_idx]  # (E,H,R,Dv)
                sub_logits = sm_scale * (q_m @ al_R.transpose(-1, -2))  # (E,H,B,R)
                sub_max = sub_logits.max(dim=-1, keepdim=True).values
                sub_exp = torch.exp(sub_logits - sub_max)
                level_logit = sub_max.squeeze(-1) + torch.log(sub_exp.sum(-1) + eps)  # (E,H,B)
                sub_weights = sub_exp / (sub_exp.sum(-1, keepdim=True) + eps)
                level_value = sub_weights @ y_R  # (E,H,B,Dv)
                tier_logits.append(level_logit)
                tier_values.append(level_value)

        if tier_logits:
            tier_logit_t = torch.stack(tier_logits, dim=-1)  # (E,H,B,num_tiers)
            tier_value_t = torch.stack(tier_values, dim=-2)  # (E,H,B,num_tiers,Dv)
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
            tier_w = exp_c[..., win_len:]  # (E,H,B,num_tiers)
            num_tier = (tier_w.unsqueeze(-1) * tier_value_t).sum(dim=-2)
        else:
            num_tier = 0.0

        out_m = (num_local + num_tier) / denom
        outputs.append(out_m)

    z = torch.stack(outputs, dim=2).view(E, H, N_padded, Dv)
    return z[..., :N, :]
