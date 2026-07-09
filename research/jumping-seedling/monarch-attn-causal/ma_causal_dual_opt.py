"""Optimized dual-representative causal MonarchAttention.

Same math as ma_causal_dual.py, but avoids the two wasteful full M x M
matmuls that version spent on the *causal* (diagonal-only) branch. The
diagonal of `al_c @ q^T` (or `q @ al_c^T`) is just an elementwise
row-wise dot product -- `(al_c * q).sum(-1)` -- since both tensors are
already aligned by block index for the diagonal case. No need to
materialize a full M x M matmul just to throw away every entry except
the diagonal, and no need for the masked-matmul value-combination trick
either (that also computed a full matmul only to keep one nonzero entry
per row) -- a diagonal extract + elementwise scale replaces it.

This should bring stage 2 and the final step's causal-branch cost down
from O(B*M^2*D) to O(B*M*D), leaving only stage 1's genuine 2x (two
full local B x B passes, unavoidable -- both representatives are
needed everywhere) as real overhead. Verified against ma_causal_dual.py
for exact numerical agreement, then benchmarked for wall-clock cost.
"""

from math import sqrt

import torch
import torch.nn.functional as F

Tensor = torch.Tensor
xlogy = torch.special.xlogy


def _local_pass(ar, k, cr, sm_scale, mask, eps=1e-12):
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


def _cross_pass_dual_opt(al_c, al_f, q, cl_c, cl_f, off_lower):
    """al_c/al_f, q: (...,B,M,D); cl_c/cl_f: (...,B,M); off_lower: (1,M,M)
    bool, True where key_block < query_block (strict)."""
    diag_logit = (al_c * q).sum(-1) - cl_c  # (...,B,M), elementwise dot, no MxM matmul

    l_hat_f = (al_f @ q.transpose(-1, -2)).to(torch.float) - cl_f[..., :, None]
    l_hat = torch.where(off_lower, l_hat_f, -float("inf"))
    l_hat = torch.diagonal_scatter(l_hat, diag_logit, dim1=-2, dim2=-1)
    l = F.softmax(l_hat, dim=-2)

    cr = torch.sum(l, dim=-1).transpose(-1, -2)
    ar = (l.to(q.dtype) @ q).transpose(-2, -3)
    return ar, cr


def _final_dual_opt(al_c, y_c, cl_c, al_f, y_f, cl_f, q, off_lower_t):
    """q, al_c/al_f: (...,B,M,D); off_lower_t: (1,M,M) bool, True where
    key_block(mk) < query_block(mq), axes (mq, mk)."""
    diag_logit = (q * al_c).sum(-1) - cl_c  # (...,B,M)

    l_hat_f = (q @ al_f.transpose(-1, -2)).to(torch.float) - cl_f[..., None, :]
    l_hat = torch.where(off_lower_t, l_hat_f, -float("inf"))
    l_hat = torch.diagonal_scatter(l_hat, diag_logit, dim1=-2, dim2=-1)
    l = F.softmax(l_hat, dim=-1)

    diag_weight = torch.diagonal(l, dim1=-2, dim2=-1)  # (...,B,M)
    l_off = l * off_lower_t
    z = l_off.to(y_f.dtype) @ y_f + diag_weight[..., None].to(y_c.dtype) * y_c
    return z.transpose(-2, -3).contiguous()


def monarch_attention_causal_dual_opt_torch(
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
    valid = range_n < N
    if attn_mask is not None:
        attn_mask = F.pad(attn_mask, (0, N_padded - N)).view(E, 1, M, B)
        valid = torch.logical_and(valid, attn_mask)

    bq = torch.arange(B, device=q.device).view(1, B, 1)
    bk = torch.arange(B, device=q.device).view(1, 1, B)
    row_causal = (bq >= bk).unsqueeze(0).unsqueeze(0)
    row_full = torch.ones_like(row_causal)
    key_valid = valid[..., None, :]
    row_mask_causal = torch.logical_and(row_causal, key_valid)
    row_mask_full = torch.logical_and(row_full, key_valid)

    mk = torch.arange(M, device=q.device).view(1, M, 1)
    mq = torch.arange(M, device=q.device).view(1, 1, M)
    off_lower = mk < mq  # (1,M,M), strict: key-block < query-block

    mq2 = torch.arange(M, device=q.device).view(1, M, 1)
    mk2 = torch.arange(M, device=q.device).view(1, 1, M)
    off_lower_t = mk2 < mq2  # (1,M,M), query axis first

    for _ in range(T - 1):
        al_c, _, cl_c = _local_pass(ar, k, cr, sm_scale, row_mask_causal)
        al_f, _, cl_f = _local_pass(ar, k, cr, sm_scale, row_mask_full)
        ar, cr = _cross_pass_dual_opt(al_c, al_f, q, cl_c, cl_f, off_lower)

    al_c, y_c, cl_c = _local_pass_with_v(ar, k, v, cr, sm_scale, row_mask_causal)
    al_f, y_f, cl_f = _local_pass_with_v(ar, k, v, cr, sm_scale, row_mask_full)
    z = _final_dual_opt(al_c, y_c, cl_c, al_f, y_f, cl_f, q, off_lower_t)
    z = z.view(E, H, N_padded, Dv)
    return z[..., :N, :]
