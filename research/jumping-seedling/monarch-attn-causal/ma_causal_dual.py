"""Dual-representative causal MonarchAttention (Option 1 from the cost
discussion): computes two per-block local representatives each
iteration -- causal-masked (for the block's own diagonal self-attention)
and unmasked/full (for use by strictly-later query blocks, which should
see an earlier block in full, not truncated to key <= query's own slot).

Builds on ma_causal.py's single-representative scheme (see that file's
docstring for the row/column-block causal-cut derivation). The only
structural change is: everywhere the single-representative version used
one `al`/`cl` (or `al,y,cl` at the final step), this version computes
two and combines them per (key_block, query_block) pair via a diagonal
selector -- causal source at key_block == query_block, full source at
key_block < query_block, masked out entirely at key_block > query_block.

Combination trick (avoids needing per-pair gather): for a combined
attention distribution `l` over (key_block, query_block) built from
selected logits, and two possible "value" tensors to aggregate
(y_causal, y_full), the correct per-query output is:

    z = (l * diag_mask) @ y_causal + (l * ~diag_mask) @ y_full

which works because `l * diag_mask` is nonzero only at the single
key_block == query_block entry per query, so this matmul reduces to
exactly `l[q,q] * y_causal[q]` for each query row, matching the desired
per-pair selection without an explicit gather.
"""

from math import sqrt

import torch
import torch.nn.functional as F

Tensor = torch.Tensor
xlogy = torch.special.xlogy


def _local_pass(ar, k, cr, sm_scale, mask, eps=1e-12):
    """One local (row-block) attention pass. mask: (...,M,Bq,Bk) bool."""
    r_hat = sm_scale * (ar @ k.transpose(-1, -2)).to(torch.float)
    r_hat = r_hat / (cr[..., :, None] + eps)
    r_hat = r_hat + torch.where(mask, 0.0, -float("inf"))
    r_hat = torch.exp(
        r_hat - torch.clamp(torch.max(r_hat, dim=-1, keepdim=True).values, min=eps)
    )
    r = r_hat / (torch.sum(r_hat, dim=-1, keepdim=True) + eps)
    r = torch.clamp(r, min=torch.finfo(r.dtype).tiny)

    cl = torch.sum(xlogy(r, r), dim=-1).transpose(-1, -2)
    al = sm_scale * (r.to(k.dtype) @ k).transpose(-2, -3)
    return al, r, cl


def _local_pass_with_v(ar, k, v, cr, sm_scale, mask, eps=1e-12):
    al, r, cl = _local_pass(ar, k, cr, sm_scale, mask, eps)
    y = (r.to(v.dtype) @ v).transpose(-2, -3)
    return al, y, cl


def _cross_pass_dual(al_c, al_f, q, cl_c, cl_f, diag_mask, causal_lower):
    """Stage-2 cross-block pass, combining causal (diagonal) and full
    (strictly-earlier) representatives into one softmax distribution.

    al_c/al_f, cl_c/cl_f: (...,B,M,D) / (...,B,M). diag_mask, causal_lower:
    (1,M,M) bool over (key_block, query_block).
    """
    l_hat_c = (al_c @ q.transpose(-1, -2)).to(torch.float) - cl_c[..., :, None]
    l_hat_f = (al_f @ q.transpose(-1, -2)).to(torch.float) - cl_f[..., :, None]
    l_hat = torch.where(diag_mask, l_hat_c, l_hat_f)
    l_hat = l_hat + torch.where(causal_lower, 0.0, -float("inf"))
    l = F.softmax(l_hat, dim=-2)

    cr = torch.sum(l, dim=-1).transpose(-1, -2)
    # value being aggregated is q itself (shared, not causal/full-dependent)
    ar = (l.to(q.dtype) @ q).transpose(-2, -3)
    return ar, cr


def _final_dual(al_c, y_c, cl_c, al_f, y_f, cl_f, q, diag_mask_t, causal_lower_t):
    """Final step, mirrors z_ref but with dual (causal/full) source
    selection since here the aggregated *value* (y) genuinely differs
    between the two sources -- needs the masked-matmul split.
    diag_mask_t/causal_lower_t: (1,Mq,Mk) bool (query axis first).
    """
    l_hat_c = (q @ al_c.transpose(-1, -2)).to(torch.float) - cl_c[..., None, :]
    l_hat_f = (q @ al_f.transpose(-1, -2)).to(torch.float) - cl_f[..., None, :]
    l_hat = torch.where(diag_mask_t, l_hat_c, l_hat_f)
    l_hat = l_hat + torch.where(causal_lower_t, 0.0, -float("inf"))
    l = F.softmax(l_hat, dim=-1)

    z = (l * diag_mask_t).to(y_c.dtype) @ y_c + (
        l * (~diag_mask_t & causal_lower_t)
    ).to(y_f.dtype) @ y_f
    return z.transpose(-2, -3).contiguous()


def monarch_attention_causal_dual_torch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None,
    T: int,
    B: int,
    pre_pad: bool,
) -> Tensor:
    E, H, N, D = q.shape
    _, _, _, Dv = v.shape
    M = (N + B - 1) // B
    N_padded = M * B

    if pre_pad:
        raise NotImplementedError("dual causal variant only supports pad_type='post'")

    sm_scale = 1 / sqrt(D)
    pad_t_2d = (0, 0, 0, N_padded - N)

    q = F.pad(q, pad_t_2d).view(E, H, M, B, D)
    k = F.pad(k, pad_t_2d).view(E, H, M, B, D)
    v = F.pad(v, pad_t_2d).view(E, H, M, B, Dv)

    ar = q
    cr = torch.ones(E, H, M, B, device=q.device, dtype=torch.float)
    q = q.transpose(-2, -3)

    range_n = torch.arange(M * B).view(M, B).to(q.device)
    valid = range_n < N  # (M, B)
    if attn_mask is not None:
        attn_mask = F.pad(attn_mask, (0, N_padded - N)).view(E, 1, M, B)
        valid = torch.logical_and(valid, attn_mask)

    bq = torch.arange(B, device=q.device).view(1, B, 1)
    bk = torch.arange(B, device=q.device).view(1, 1, B)
    row_causal = (bq >= bk).unsqueeze(0).unsqueeze(0)  # (1,1,1,B,B) broadcast over M
    row_full = torch.ones_like(row_causal)
    key_valid = valid[..., None, :]  # (...,M,1,B)
    row_mask_causal = torch.logical_and(row_causal, key_valid)
    row_mask_full = torch.logical_and(row_full, key_valid)

    mk = torch.arange(M, device=q.device).view(1, M, 1)
    mq = torch.arange(M, device=q.device).view(1, 1, M)
    diag_mask = (mk == mq)          # (1,M,M), key-axis first (matches ar_cr_ref l_hat)
    causal_lower = (mk <= mq)       # (1,M,M)

    mq2 = torch.arange(M, device=q.device).view(1, M, 1)
    mk2 = torch.arange(M, device=q.device).view(1, 1, M)
    diag_mask_t = (mq2 == mk2)      # (1,M,M), query-axis first (matches z_ref l_hat)
    causal_lower_t = (mk2 <= mq2)

    for _ in range(T - 1):
        al_c, _, cl_c = _local_pass(ar, k, cr, sm_scale, row_mask_causal)
        al_f, _, cl_f = _local_pass(ar, k, cr, sm_scale, row_mask_full)
        ar, cr = _cross_pass_dual(al_c, al_f, q, cl_c, cl_f, diag_mask, causal_lower)

    al_c, y_c, cl_c = _local_pass_with_v(ar, k, v, cr, sm_scale, row_mask_causal)
    al_f, y_f, cl_f = _local_pass_with_v(ar, k, v, cr, sm_scale, row_mask_full)
    z = _final_dual(al_c, y_c, cl_c, al_f, y_f, cl_f, q, diag_mask_t, causal_lower_t)
    z = z.view(E, H, N_padded, Dv)
    return z[..., :N, :]
