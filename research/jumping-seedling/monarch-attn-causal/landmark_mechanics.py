"""Five landmark/representative-construction mechanics, pluggable into
the same downstream framework (each produces R "pseudo-query" vectors
per block, which then attend over the block's own B_l keys to produce
al_R/y_R -- isolating the SELECTION strategy as the only variable,
consistent with tonight's ablation discipline of changing one thing at
a time). All operate on (...,Bl,D)/(...,Bl,Dv) tensors with arbitrary
leading batch dims.
"""

import torch
import torch.nn.functional as F


def _attend(landmark_q, k_block, v_block, sm_scale):
    scores = sm_scale * (landmark_q @ k_block.transpose(-1, -2))  # (...,R,Bl)
    weights = F.softmax(scores, dim=-1)
    al_R = weights @ k_block
    y_R = weights @ v_block
    return al_R, y_R


def landmarks_random_reuse(k_block, v_block, R, sm_scale):
    """Tonight's baseline: R evenly-spaced real keys as pseudo-queries."""
    Bl = k_block.shape[-2]
    idx = torch.linspace(0, Bl - 1, R, device=k_block.device).round().long().clamp(max=Bl - 1)
    landmark_q = k_block.index_select(-2, idx)
    return _attend(landmark_q, k_block, v_block, sm_scale)


def landmarks_top_magnitude(k_block, v_block, R, sm_scale):
    """R keys with largest L2 norm as pseudo-queries -- a cheap proxy for
    'salient' content, no attention/clustering needed to find them."""
    Bl = k_block.shape[-2]
    R = min(R, Bl)
    norms = k_block.norm(dim=-1)  # (...,Bl)
    idx = norms.topk(R, dim=-1).indices  # (...,R)
    landmark_q = torch.gather(k_block, -2, idx.unsqueeze(-1).expand(*idx.shape, k_block.shape[-1]))
    return _attend(landmark_q, k_block, v_block, sm_scale)


def landmarks_kmeans(k_block, v_block, R, sm_scale, iters=3):
    """R k-means centroids (dot-product k-means, few Lloyd iterations,
    consistent with the k-means already used in ma_causal_topk_ann.py)
    as pseudo-queries."""
    *batch, Bl, D = k_block.shape
    R = min(R, Bl)
    idx0 = torch.linspace(0, Bl - 1, R, device=k_block.device).round().long().clamp(max=Bl - 1)
    centroids = k_block.index_select(-2, idx0).clone()  # (...,R,D)
    for _ in range(iters):
        sims = sm_scale * (k_block @ centroids.transpose(-1, -2))  # (...,Bl,R)
        assign = sims.argmax(dim=-1)  # (...,Bl)
        onehot = F.one_hot(assign, R).to(k_block.dtype)  # (...,Bl,R)
        sums = onehot.transpose(-1, -2) @ k_block  # (...,R,D)
        counts = onehot.sum(dim=-2).unsqueeze(-1)  # (...,R,1)
        new_centroids = sums / counts.clamp(min=1)
        empty = counts == 0
        centroids = torch.where(empty, centroids, new_centroids)
    return _attend(centroids, k_block, v_block, sm_scale)


def landmarks_fps(k_block, v_block, R, sm_scale):
    """Farthest-point sampling: greedily pick R keys maximizing minimum
    pairwise distance to already-picked keys -- covers diverse directions
    instead of clustering around a mean or picking by raw magnitude."""
    *batch, Bl, D = k_block.shape
    flat_batch = int(torch.tensor(batch).prod().item()) if batch else 1
    k_flat = k_block.reshape(flat_batch, Bl, D)
    R = min(R, Bl)

    picked = torch.zeros(flat_batch, R, dtype=torch.long, device=k_block.device)
    picked[:, 0] = 0  # deterministic start: first position in the block
    min_dist = torch.full((flat_batch, Bl), float("inf"), device=k_block.device)
    for r in range(1, R):
        last = k_flat[torch.arange(flat_batch), picked[:, r - 1]]  # (flat_batch,D)
        d = (k_flat - last.unsqueeze(1)).norm(dim=-1)  # (flat_batch,Bl)
        min_dist = torch.minimum(min_dist, d)
        picked[:, r] = min_dist.argmax(dim=-1)

    idx = picked.view(*batch, R) if batch else picked.view(R)
    landmark_q = torch.gather(k_block, -2, idx.unsqueeze(-1).expand(*idx.shape, D))
    return _attend(landmark_q, k_block, v_block, sm_scale)


def landmarks_maxpool(k_block, v_block, R, sm_scale):
    """Split the block into R contiguous chunks, max-pool each chunk
    per-feature-dimension (both K and V) -- preserves per-dimension
    outliers instead of averaging them away. Structurally different from
    the other four: no attention step, no query/key correspondence
    preserved for V (each output dim may come from a different real
    position within the chunk)."""
    *batch, Bl, D = k_block.shape
    Dv = v_block.shape[-1]
    R = min(R, Bl)
    chunk_bounds = torch.linspace(0, Bl, R + 1).round().long()
    al_chunks, y_chunks = [], []
    for r in range(R):
        lo, hi = chunk_bounds[r].item(), max(chunk_bounds[r + 1].item(), chunk_bounds[r].item() + 1)
        al_chunks.append(k_block[..., lo:hi, :].amax(dim=-2))
        y_chunks.append(v_block[..., lo:hi, :].amax(dim=-2))
    al_R = torch.stack(al_chunks, dim=-2)  # (...,R,D)
    y_R = torch.stack(y_chunks, dim=-2)  # (...,R,Dv)
    return al_R, y_R


MECHANICS = {
    "random_reuse": landmarks_random_reuse,
    "top_magnitude": landmarks_top_magnitude,
    "kmeans": landmarks_kmeans,
    "fps": landmarks_fps,
    "maxpool": landmarks_maxpool,
}
