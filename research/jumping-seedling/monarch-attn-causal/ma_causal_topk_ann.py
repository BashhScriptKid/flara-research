"""Version C-ANN: approximate top-k off-diagonal retrieval via coarse
clustering, as a fourth ablation point alongside compression (Version B),
exact top-k (Version C), and dual-rep Monarch (dual_opt).

Simplest reasonable approach, not state-of-the-art by request: cluster
the causal past-key pool into ~sqrt(P) buckets (P = pool size) via a
few Lloyd iterations of dot-product k-means (consistent with the
inner-product scoring used everywhere else in this codebase, not L2),
recomputed FROM SCRATCH per block (no incremental cluster maintenance --
a real implementation would update clusters incrementally rather than
re-run k-means on the whole growing pool every block; this is a known,
deliberate simplification for a data-collection probe, not a
production design). For each query, probe the `n_probe` nearest
centroids by the same dot-product score, restrict top-k selection to
keys assigned to those clusters only.

IMPORTANT CAVEAT ON COST NUMBERS: this reference implementation still
computes the full (B, P) exact dot-product score matrix and then masks
it down to the probed clusters, rather than skipping the score
computation for non-probed keys entirely. That means the wall-clock
numbers here reflect "clustering overhead ADDED ON TOP OF exact top-k's
own scoring cost," not the speed benefit a real ANN index (which
*skips* scoring unprobed points) would deliver. Cost numbers from this
file characterize whether the recall/accuracy tradeoff exists, not
what a well-engineered ANN would actually cost.

Causality: clustering is built only from the causal past pool (blocks
0..m-1) at the time each block is processed, so no future information
leaks into the cluster structure or assignments.
"""

import torch
import torch.nn.functional as F
from math import sqrt

Tensor = torch.Tensor


def _kmeans(keys: Tensor, n_clusters: int, n_iters: int, sm_scale: float):
    """keys: (E,H,P,D). Dot-product (not L2) k-means, few Lloyd iters."""
    E, H, P, D = keys.shape
    n_clusters = max(1, min(n_clusters, P))
    idx = torch.linspace(0, P - 1, n_clusters, device=keys.device).long()
    centroids = keys[:, :, idx, :].clone()  # (E,H,n_clusters,D)

    assign = torch.zeros(E, H, P, dtype=torch.long, device=keys.device)
    for _ in range(n_iters):
        sims = sm_scale * (keys @ centroids.transpose(-1, -2))  # (E,H,P,n_clusters)
        assign = sims.argmax(dim=-1)  # (E,H,P)
        onehot = F.one_hot(assign, n_clusters).to(keys.dtype)  # (E,H,P,n_clusters)
        sums = torch.einsum("ehpc,ehpd->ehcd", onehot, keys)  # (E,H,n_clusters,D)
        counts = onehot.sum(dim=2).unsqueeze(-1)  # (E,H,n_clusters,1)
        new_centroids = sums / counts.clamp(min=1)
        empty = counts == 0
        centroids = torch.where(empty, centroids, new_centroids)
    return centroids, assign, n_clusters


def monarch_causal_topk_ann(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    B: int,
    topk: int,
    n_probe: int = 3,
    kmeans_iters: int = 3,
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
    valid = range_n < N

    causal_local = torch.tril(torch.ones(B, B, dtype=torch.bool, device=q.device))

    outputs = []
    for m in range(M):
        q_m, k_m, v_m = q[:, :, m], k[:, :, m], v[:, :, m]
        valid_m = valid[m].view(1, 1, 1, B)

        local_scores = sm_scale * (q_m @ k_m.transpose(-1, -2))
        local_mask = causal_local.view(1, 1, B, B) & valid_m
        local_scores = local_scores.masked_fill(~local_mask, -float("inf"))
        local_v_exp = v_m.unsqueeze(2).expand(-1, -1, B, -1, -1)

        if m == 0:
            combined = local_scores
            cand_v = local_v_exp
        else:
            past_k = k[:, :, :m].reshape(E, H, m * B, D)
            past_v = v[:, :, :m].reshape(E, H, m * B, v.shape[-1])
            P = m * B

            n_clusters = max(1, round(sqrt(P)))
            centroids, assign, n_clusters = _kmeans(past_k, n_clusters, kmeans_iters, sm_scale)

            centroid_scores = sm_scale * (q_m @ centroids.transpose(-1, -2))  # (E,H,B,n_clusters)
            n_probe_eff = min(n_probe, n_clusters)
            top_cluster_idx = torch.topk(centroid_scores, k=n_probe_eff, dim=-1).indices  # (E,H,B,n_probe_eff)

            assign_exp = assign.unsqueeze(2).unsqueeze(-1)  # (E,H,1,P,1)
            probe_exp = top_cluster_idx.unsqueeze(3)  # (E,H,B,1,n_probe_eff)
            cand_mask = (assign_exp == probe_exp).any(dim=-1)  # (E,H,B,P)

            # NOTE: full (B,P) score computed then masked -- see module
            # docstring caveat on why this doesn't reflect real ANN speed.
            off_scores = sm_scale * (q_m @ past_k.transpose(-1, -2))  # (E,H,B,P)
            off_scores = off_scores.masked_fill(~cand_mask, -float("inf"))

            kk = min(topk, P)
            top_vals, top_idx = torch.topk(off_scores, k=kk, dim=-1)  # (E,H,B,kk)

            idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, -1, past_v.shape[-1])
            past_v_exp = past_v.unsqueeze(2).expand(-1, -1, B, -1, -1)
            gathered_v = torch.gather(past_v_exp, 3, idx_exp)

            combined = torch.cat([local_scores, top_vals], dim=-1)
            cand_v = torch.cat([local_v_exp, gathered_v], dim=-2)

        row_max = torch.clamp(combined.max(dim=-1, keepdim=True).values, min=-1e30)
        row_max = torch.nan_to_num(row_max, neginf=0.0)
        exp_combined = torch.exp(combined - row_max)
        exp_combined = torch.nan_to_num(exp_combined, nan=0.0)
        denom = exp_combined.sum(dim=-1, keepdim=True) + eps

        out_m = (exp_combined.unsqueeze(-1) * cand_v).sum(dim=-2) / denom
        outputs.append(out_m)

    z = torch.stack(outputs, dim=2).view(E, H, N_padded, v.shape[-1])
    return z[..., :N, :]
