"""SlidingMonarchAttention: decouples the local exact-window size from
Monarch's block size, per the "Solved / Ablation / Derived principle"
closing entry in JOURNAL.md. Fixed, content-independent scope for both
parts (window boundary and Monarch block-triangular boundary are both
position-derived, never a runtime competition) -- consistent with the
derived reliability principle: weight can be approximate, but what's
"in scope" is never decided by rank/competition.

Design (see chat for the full derivation): the single-block-diagonal
dual-representation trick (ma_causal_dual_opt.py) was needed because,
at W=1 block, the diagonal block had to serve BOTH its own causal
self-attention AND be reused, unmasked, by all later blocks -- two
conflicting visibility requirements on the same representative. With a
genuine multi-block sliding window (W_blocks >= 1) handling all
self-visibility exactly, every block Monarch ever serves is already
permanently "closed" (past the window) for every query that reads it --
so the far/Monarch part needs only ONE representative, not two.

What must NOT happen (the mistake this design avoids): collapsing the
far blocks into a single blended vector before combining with the local
window would just reinvent Version B's mean-pooling trap at block
granularity. Instead, far-block logits stay UNCOLLAPSED -- one real,
sm_scale-consistent candidate per far block -- and compete in the SAME
joint softmax as the local window's per-token logits, following the
same pattern validated in ma_causal_topk.py (real logits, real units,
no calibration mismatch, unlike Version B's flat mean-key).

Two parts:
1. Local window (exact): for query block m_q, keys/values from blocks
   [max(0, m_q - W_blocks + 1), m_q], real causal-masked softmax,
   O(N * W) work, no Monarch approximation -- W_blocks=1 reduces this
   to exactly the old single-block-diagonal case.
2. Far (Monarch, single representative): blocks strictly before the
   window, i.e. m_key <= m_q - W_blocks. `al_full`/`cl_full` computed
   via non-causal LOCAL passes (no within-block causal mask needed --
   window owns all self-visibility), refined over T-1 iterations using
   a block-triangular mask SHIFTED by W_blocks (`m_key <= m_query -
   W_blocks`) in the cross-block pass -- this shift is what keeps the
   iterative refinement itself causally safe (an unmasked global
   refinement would let future blocks contaminate a past block's
   representative through the alternating rounds, the same leak the
   very first causal_probe.py caught for the fully non-causal
   reference). Final step keeps per-far-block logits/values uncollapsed
   for the joint softmax instead of aggregating them into one z.

Caveat, not yet resolved: whether q . al_full (Monarch's own refined
block representative) is genuinely scale-comparable to raw q . k_j
(local window logits) the way exact top-k's per-token logits were --
al_full is an attention-WEIGHTED combination of a block's real keys
(via Monarch's own internal local softmax), not a flat mean, so it may
retain more signal than Version B's k_bar did, but this hasn't been
measured. That's exactly what the needle test below is for.
"""

import torch
import torch.nn.functional as F
from math import sqrt

Tensor = torch.Tensor


def _local_pass(ar, k, cr, sm_scale, mask, eps=1e-12):
    r_hat = sm_scale * (ar @ k.transpose(-1, -2)).to(torch.float)
    r_hat = r_hat / (cr[..., :, None] + eps)
    r_hat = r_hat + torch.where(mask, 0.0, -float("inf"))
    r_hat = torch.exp(
        r_hat - torch.clamp(torch.max(r_hat, dim=-1, keepdim=True).values, min=eps)
    )
    r = r_hat / (torch.sum(r_hat, dim=-1, keepdim=True) + eps)
    r = torch.clamp(r, min=torch.finfo(r.dtype).tiny)
    cl = torch.sum(torch.special.xlogy(r, r), dim=-1).transpose(-1, -2)
    al = sm_scale * (r.to(k.dtype) @ k).transpose(-2, -3)
    return al, r, cl


def _local_pass_with_v(ar, k, v, cr, sm_scale, mask, eps=1e-12):
    al, r, cl = _local_pass(ar, k, cr, sm_scale, mask, eps)
    y = (r.to(v.dtype) @ v).transpose(-2, -3)
    return al, y, cl


def sliding_monarch_causal(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    B: int,
    W_blocks: int,
    T: int,
    W_refine: int | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """W_refine: shift used for the INTERNAL T-iteration cross-block
    refinement, decoupled from W_blocks (the shift used at the FINAL
    read step). Both are causally SAFE for any value >= 0 -- al_full[m]
    only ever pools from blocks <= m by induction regardless of shift,
    so this parameter cannot introduce a leak. It was hypothesized that
    a tighter W_refine (e.g. 0 = plain causal) would improve quality by
    letting the refinement use more available context. Measured FALSE:
    W_refine=0 is consistently worse than W_refine=W_blocks (needle cos
    0.34 vs 0.64 at W_blocks=4, dist=14; small but consistent aggregate
    quality drop too) -- letting nearer blocks dominate the cross-block
    attention mass during refinement pulls the representative away from
    an evenly-considered view across the full valid far range. Default
    is None, which resolves to W_blocks (the validated, better choice)
    -- kept as an explicit parameter for further experimentation, not
    because 0 is recommended.
    """
    if W_refine is None:
        W_refine = W_blocks
    E, H, N, D = q.shape
    _, _, _, Dv = v.shape
    M = (N + B - 1) // B
    N_padded = M * B
    sm_scale = 1 / sqrt(D)

    pad = N_padded - N
    qb = F.pad(q, (0, 0, 0, pad)).view(E, H, M, B, D)
    kb = F.pad(k, (0, 0, 0, pad)).view(E, H, M, B, D)
    vb = F.pad(v, (0, 0, 0, pad)).view(E, H, M, B, Dv)

    range_n = torch.arange(N_padded).view(M, B)
    valid_mb = (range_n < N)  # (M,B) bool

    # ---- far/Monarch branch: single representative, causal-safe refinement ----
    ar = qb
    cr = torch.ones(E, H, M, B, device=q.device, dtype=torch.float)
    q_t = qb.transpose(-2, -3)  # (E,H,B,M,D)

    row_full = valid_mb.view(1, 1, M, 1, B).expand(E, H, M, B, B)  # padding only, no causal cut

    mk = torch.arange(M, device=q.device).view(1, M, 1)
    mq = torch.arange(M, device=q.device).view(1, 1, M)
    far_lower_refine = (mk <= mq - W_refine)  # internal refinement: tighter shift OK

    for _ in range(T - 1):
        al, _, cl = _local_pass(ar, kb, cr, sm_scale, row_full)
        l_hat = (al @ q_t.transpose(-1, -2)).to(torch.float) - cl[..., :, None]
        l_hat = l_hat + torch.where(far_lower_refine, 0.0, -float("inf"))
        # early query blocks (m_q < W_blocks) have NO valid far candidates yet
        # -- an all -inf column -- which raw softmax turns into 0/0 = NaN that
        # then cascades through later iterations. Guard explicitly.
        row_max = torch.clamp(l_hat.max(dim=-2, keepdim=True).values, min=-1e30)
        row_max = torch.nan_to_num(row_max, neginf=0.0)
        exp_l = torch.nan_to_num(torch.exp(l_hat - row_max), nan=0.0)
        l = exp_l / (exp_l.sum(dim=-2, keepdim=True) + eps)
        cr = torch.sum(l, dim=-1).transpose(-1, -2)
        ar = (l.to(q_t.dtype) @ q_t).transpose(-2, -3)

    al_full, y_full, cl_full = _local_pass_with_v(ar, kb, vb, cr, sm_scale, row_full)
    # al_full/y_full/cl_full: (E,H,B,M,*) -- per-block representative, value, log-norm

    far_lower_t = (mk <= mq - W_blocks).transpose(-2, -1)  # (1,Mq,Mk) query-axis first
    l_hat_far = (q_t @ al_full.transpose(-1, -2)).to(torch.float) - cl_full[..., None, :]  # (E,H,B,Mq,Mk)
    l_hat_far = l_hat_far.masked_fill(~far_lower_t, -float("inf"))

    # ---- local window branch: exact causal softmax over trailing W_blocks blocks ----
    causal_local = torch.tril(torch.ones(B, B, dtype=torch.bool, device=q.device))
    outputs = []
    for m_q in range(M):
        w_start = max(0, m_q - W_blocks + 1)
        win_k = kb[:, :, w_start : m_q + 1].reshape(E, H, (m_q - w_start + 1) * B, D)
        win_v = vb[:, :, w_start : m_q + 1].reshape(E, H, (m_q - w_start + 1) * B, Dv)
        q_m = qb[:, :, m_q]  # (E,H,B,D)

        n_win_blocks = m_q - w_start + 1
        win_valid = valid_mb[w_start : m_q + 1].reshape(-1)  # (n_win_blocks*B,)
        # causal: key position (block-local index) valid if its block < current block,
        # or (== current block AND intra-block index <= query's own b)
        blk_idx = torch.arange(n_win_blocks, device=q.device).repeat_interleave(B)
        own_blk = n_win_blocks - 1
        intra = torch.arange(B, device=q.device).repeat(n_win_blocks)
        causal_win = (blk_idx < own_blk).unsqueeze(0) | (
            (blk_idx == own_blk).unsqueeze(0) & (intra.unsqueeze(0) <= torch.arange(B, device=q.device).unsqueeze(1))
        )  # (B, n_win_blocks*B)
        win_mask = causal_win & win_valid.view(1, -1)

        local_scores = sm_scale * (q_m @ win_k.transpose(-1, -2))  # (E,H,B,win_len)
        local_scores = local_scores.masked_fill(~win_mask.view(1, 1, B, -1), -float("inf"))

        far_logits_m = l_hat_far[:, :, :, m_q, :]  # (E,H,B,Mk)
        combined = torch.cat([local_scores, far_logits_m], dim=-1)  # (E,H,B,win_len+M)

        row_max = torch.clamp(combined.max(dim=-1, keepdim=True).values, min=-1e30)
        row_max = torch.nan_to_num(row_max, neginf=0.0)
        exp_combined = torch.exp(combined - row_max)
        exp_combined = torch.nan_to_num(exp_combined, nan=0.0)
        denom = exp_combined.sum(dim=-1, keepdim=True) + eps

        win_len = win_k.shape[-2]
        local_w = exp_combined[..., :win_len]
        far_w = exp_combined[..., win_len:]  # (E,H,B,Mk)

        num_local = local_w @ win_v  # (E,H,B,Dv)
        y_full_m = y_full[:, :, :, :]  # (E,H,B,Mk,Dv) -- same for every m_q, mask does the work
        num_far = (far_w.unsqueeze(-1) * y_full_m).sum(dim=-2)  # (E,H,B,Dv)

        out_m = (num_local + num_far) / denom
        outputs.append(out_m)

    z = torch.stack(outputs, dim=2).view(E, H, N_padded, Dv)
    return z[..., :N, :]
