"""Causal-capable variant of MonarchAttention's torch reference impl.

Standalone copy of `ma/ma_torch.py` (cjyaras/monarch-attention), NOT an
edit of the pristine upstream clone in repo/. Adds a `causal: bool` flag.

Derivation (see RESEARCH_LOG.md 2026-07-08 entries for the earlier dead
end and the mechanism trace that unblocked this):

  Reshape (E,H,N,D) -> (E,H,M,B,D) is row-major: sequence position
  n -> (m = n // B, b = n % B). So:

  - Row-blocks (al_cl_ref, local B x B attention within block m) are
    CONTIGUOUS chunks of the real sequence -> causal cut is a plain
    lower-triangular mask on (query_b, key_b), identical for every m.

  - Column-blocks (ar_cr_ref, cross-block M x M attention at fixed
    intra-block slot b) group positions {b, b+B, b+2B, ...} -> strictly
    increasing sequence order with increasing m -> causal cut is a
    plain lower-triangular mask on (query_m, key_m), identical for
    every b.

  Composition is leak-free: al[b, m] (row-causal-masked) only ever pools
  from key positions m*B + c, c <= b. The column-block causal mask then
  additionally excludes m_key > m_query outright.

  KNOWN APPROXIMATION CAVEAT (not a correctness bug): al[b, m] is built
  ONCE per (b, m) and reused by every later query block m' > m in the
  cross-block pass. For m' > m, block m is entirely in the past and
  should ideally be visible in full, but al[b, m] only saw keys <= b
  within its own block. So off-diagonal attention under-attends to the
  tail of earlier blocks, worse for small b. This is a recall/quality
  effect, separate from the -inf-vs-annealed numerics question -- both
  are measured independently in eval_causal.py.

  pre_pad interacts with block-position semantics in a way not verified
  here -- causal=True only supports pad_type="post" (right-padding),
  which is the standard causal-LM case anyway. Passing pre_pad=True with
  causal=True raises.
"""

from math import sqrt

import torch
import torch.nn.functional as F

Tensor = torch.Tensor
xlogy = torch.special.xlogy


def al_cl_ref(ar, k, cr, sm_scale, mask, eps=1e-12):
    """mask: (..., M, Bq, Bk) bool, True = key visible to that query."""
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

    return al, cl


def ar_cr_ref(al, q, cl, mask_cross):
    """mask_cross: (..., B, M_key, M_query) bool, True = key-block visible."""
    l_hat = (al @ q.transpose(-1, -2)).to(torch.float)
    l_hat = l_hat - cl[..., :, None]
    l_hat = l_hat + torch.where(mask_cross, 0.0, -float("inf"))
    l = F.softmax(l_hat, dim=-2)

    cr = torch.sum(l, dim=-1).transpose(-1, -2)
    ar = (l.to(q.dtype) @ q).transpose(-2, -3)

    return ar, cr


def al_y_cl_ref(ar, k, v, cr, sm_scale, mask, eps=1e-12):
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
    y = (r.to(v.dtype) @ v).transpose(-2, -3)

    return al, y, cl


def z_ref(al, q, cl, y, mask_final=None):
    l_hat = (q @ al.transpose(-1, -2)).to(torch.float)
    l_hat = l_hat - cl[..., None, :]
    if mask_final is not None:
        l_hat = l_hat + torch.where(mask_final, 0.0, -float("inf"))
    l = F.softmax(l_hat, dim=-1)

    z = (l.to(y.dtype) @ y).transpose(-2, -3).contiguous()

    return z


def monarch_attention_causal_torch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None,
    T: int,
    B: int,
    pre_pad: bool,
    causal: bool = False,
) -> Tensor:
    E, H, N, D = q.shape
    _, _, _, Dv = v.shape
    M = (N + B - 1) // B
    N_padded = M * B

    if causal and pre_pad:
        raise NotImplementedError(
            "causal=True only supports pad_type='post' (pre_pad interacts "
            "with block-position semantics that haven't been verified for "
            "the causal cut)."
        )

    sm_scale = 1 / sqrt(D)

    pad_t = (N_padded - N, 0) if pre_pad else (0, N_padded - N)
    pad_t_2d = (0, 0) + pad_t

    q = F.pad(q, pad_t_2d).view(E, H, M, B, D)
    k = F.pad(k, pad_t_2d).view(E, H, M, B, D)
    v = F.pad(v, pad_t_2d).view(E, H, M, B, Dv)

    ar = q
    cr = torch.ones(E, H, M, B, device=q.device, dtype=torch.float)
    q = q.transpose(-2, -3)

    pad_offset = N_padded - N if pre_pad else 0
    range_n = torch.arange(M * B).view(M, B).to(q.device)
    valid = range_n >= pad_offset if pre_pad else range_n < N  # (M, B)

    if attn_mask is not None:
        attn_mask = F.pad(attn_mask, pad_t).view(E, 1, M, B)
        valid = torch.logical_and(valid, attn_mask)  # (E, M, B) or (1,M,B)

    # --- build masks ---
    # Row-block (al_cl_ref): (M, Bq, Bk) -> query b >= key c, same block.
    bq = torch.arange(B, device=q.device).view(1, B, 1)
    bk = torch.arange(B, device=q.device).view(1, 1, B)
    if causal:
        row_causal = bq >= bk  # (1, B, B), broadcasts over M
    else:
        row_causal = torch.ones(1, B, B, dtype=torch.bool, device=q.device)

    # key validity (padding) broadcast over query axis: (M,1,B) or (E,M,1,B)
    key_valid = valid[..., None, :]
    row_mask = torch.logical_and(row_causal, key_valid)  # broadcasts to (...,M,B,B)
    # al_cl_ref expects mask shaped to broadcast against r_hat (...,M,Bq,Bk)
    row_mask = row_mask.unsqueeze(-4) if row_mask.dim() == 3 else row_mask.unsqueeze(1)
    # normalize to (E_or_1, 1, M, Bq, Bk)
    if row_mask.dim() == 4:  # (M,B,B) case with no attn_mask batch dim
        row_mask = row_mask.unsqueeze(0)

    # Column-block (ar_cr_ref): (M_key, M_query) -> key block <= query block.
    mq = torch.arange(M, device=q.device).view(1, 1, M)
    mk = torch.arange(M, device=q.device).view(1, M, 1)
    if causal:
        col_causal = mk <= mq  # (1, M, M): True where key-block <= query-block
    else:
        col_causal = torch.ones(1, M, M, dtype=torch.bool, device=q.device)
    # broadcasts fine against l_hat (...,B,M_key,M_query); no extra padding
    # term needed here -- padding-invalid rows already get -inf upstream via
    # al/cl coming out of a fully-masked row (their softmax degenerates but
    # contributes ~0 weight since cr accumulates near machine-eps mass).
    col_mask = col_causal

    # Final step (z_ref): query n = m*B+b attends over key-block m' via al;
    # causal cut is again block-level: m' <= m_of_query. Reuse col_causal
    # shape (1, M_key, M_query) but z_ref's l_hat is (...,Bq,M_key) per query
    # block m_q, batched over m_q outside this function's tensor shape --
    # actually z operates with q still shaped (E,H,B,M,D) i.e. batched over
    # b, contracting the M axis of al against the M axis intrinsic to q's
    # position -- so reuse the same (1,M_key,M_query)-style logic, applied
    # per query block below via a gather over M_query from q's own block
    # index, which for z_ref's l_hat shape (...,M_query,M_key) is exactly
    # col_causal transposed.
    final_mask = col_causal.transpose(-1, -2) if causal else None  # (1,Mq,Mk)

    mask = row_mask
    mask_t = col_mask

    for _ in range(T - 1):
        al, cl = al_cl_ref(ar, k, cr, sm_scale, mask)
        ar, cr = ar_cr_ref(al, q, cl, mask_t)

    al, y, cl = al_y_cl_ref(ar, k, v, cr, sm_scale, mask)
    z = z_ref(al, q, cl, y, mask_final=final_mask)
    z = z.view(E, H, N_padded, Dv)

    return z[..., N_padded - N :, :] if pre_pad else z[..., :N, :]
