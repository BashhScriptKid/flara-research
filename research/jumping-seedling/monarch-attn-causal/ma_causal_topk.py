"""Version C: local-exact-window + exact top-k retrieval over full past
history, as the off-diagonal mechanism. Correctness-only probe -- no
optimization, no ANN/LSH, no clustering. O(N) brute-force search per
query block is fine and expected here.

Why: all three prior off-diagonal attempts (naive additive combine,
count-normalized, log-domain joint-softmax with a mean-pooled pseudo-key)
compress the off-diagonal history into ONE fixed-size summary vector per
query, and every one of them lost a needle buried in background keys --
averaging (mean-pooling) is exactly the operation that destroys a single
outlier's signature, no matter how well the *scale* of that summary is
calibrated against the local branch. This version doesn't compress at
all: it keeps every past key/value and, for each query, retrieves the
exact top-k most similar past keys (real dot-product similarity,
brute-force over the full causal history) and attends over just those k
with real softmax, real values.

This also sidesteps the calibration problem that took three attempts to
partially fix in ma_causal_linear_hybrid.py: since the top-k logits are
literal sm_scale * (q . k) dot products -- the exact same units as the
local diagonal block's scores -- they fold into ONE joint softmax with
no rescaling needed at all. No summary statistic, no mismatch.

Causality: for a query in block m, ALL of blocks 0..m-1 are entirely in
the past by construction (block-level), so the top-k candidate pool for
that query is simply every key in blocks 0..m-1 -- no per-key causal
mask needed beyond the block boundary itself. The current block's own
causal cut (query b sees local key c only if c <= b) is handled by the
unchanged diagonal term, exactly as in ma_causal_linear_hybrid.py.

Open question this file is built to answer (not assume): pick a
concrete k. Too-small k is a real, DIFFERENT failure mode from
compression-dilution -- a retrieval MISS (the needle exists in the
candidate pool with a real, undiluted score, but doesn't rank in the
top-k) rather than a SIGNAL LOSS (the needle's contribution is averaged
away before it can even be scored). eval_topk.py tests both k=16 and
k=8 explicitly to distinguish "top-k retrieval works" from "this k is
too small."
"""

import torch
import torch.nn.functional as F
from math import sqrt

Tensor = torch.Tensor


def monarch_causal_topk(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    B: int,
    topk: int,
    eps: float = 1e-6,
) -> Tensor:
    E, H, N, D = q.shape
    _, _, _, Dv = v.shape
    M = (N + B - 1) // B
    N_padded = M * B
    sm_scale = 1 / sqrt(D)

    pad = N_padded - N
    q = F.pad(q, (0, 0, 0, pad)).view(E, H, M, B, D)
    k = F.pad(k, (0, 0, 0, pad)).view(E, H, M, B, D)
    v = F.pad(v, (0, 0, 0, pad)).view(E, H, M, B, Dv)

    range_n = torch.arange(N_padded).view(M, B)
    valid = (range_n < N)  # (M,B) bool

    causal_local = torch.tril(torch.ones(B, B, dtype=torch.bool, device=q.device))

    outputs = []
    for m in range(M):
        q_m, k_m, v_m = q[:, :, m], k[:, :, m], v[:, :, m]  # (E,H,B,*)
        valid_m = valid[m].view(1, 1, 1, B)

        # --- local causal scores (block m's own keys), same as before ---
        local_scores = sm_scale * (q_m @ k_m.transpose(-1, -2))  # (E,H,B,B)
        local_mask = causal_local.view(1, 1, B, B) & valid_m
        local_scores = local_scores.masked_fill(~local_mask, -float("inf"))

        local_v_exp = v_m.unsqueeze(2).expand(-1, -1, B, -1, -1)  # (E,H,B_q,B_k,Dv)

        if m == 0:
            combined = local_scores
            cand_v = local_v_exp
        else:
            # --- exact top-k retrieval over ALL past keys (blocks 0..m-1) ---
            past_k = k[:, :, :m].reshape(E, H, m * B, D)
            past_v = v[:, :, :m].reshape(E, H, m * B, Dv)
            off_scores = sm_scale * (q_m @ past_k.transpose(-1, -2))  # (E,H,B,m*B)

            kk = min(topk, m * B)
            top_vals, top_idx = torch.topk(off_scores, k=kk, dim=-1)  # (E,H,B,kk)

            idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, -1, Dv)  # (E,H,B,kk,Dv)
            past_v_exp = past_v.unsqueeze(2).expand(-1, -1, B, -1, -1)  # (E,H,B,m*B,Dv)
            gathered_v = torch.gather(past_v_exp, 3, idx_exp)  # (E,H,B,kk,Dv)

            combined = torch.cat([local_scores, top_vals], dim=-1)  # (E,H,B,B+kk)
            cand_v = torch.cat([local_v_exp, gathered_v], dim=-2)  # (E,H,B,B+kk,Dv)

        row_max = torch.clamp(combined.max(dim=-1, keepdim=True).values, min=-1e30)
        row_max = torch.nan_to_num(row_max, neginf=0.0)
        exp_combined = torch.exp(combined - row_max)
        exp_combined = torch.nan_to_num(exp_combined, nan=0.0)
        denom = exp_combined.sum(dim=-1, keepdim=True) + eps

        out_m = (exp_combined.unsqueeze(-1) * cand_v).sum(dim=-2) / denom
        outputs.append(out_m)

    z = torch.stack(outputs, dim=2).view(E, H, N_padded, Dv)
    return z[..., :N, :]
