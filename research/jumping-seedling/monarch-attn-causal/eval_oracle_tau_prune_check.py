"""Diagnostic-only oracle-tau prune-rate check, per Fable: isolate
whether the 0% prune rate found with local-window-seeded tau_seed
(ma_meta_threshold_ball_prune.py) reflects genuinely-too-loose bounding
balls, or was an artifact of tau_seed being a small (16-key), possibly
biased sample relative to the tiers it's tested against.

Oracle tau here = a single per-query quantile computed from the FULL
pooled real scores across every active tier's block (cheating with
knowledge no real algorithm has before scoring -- same "oracle" role
this session has used before, e.g. eval_tau_inflation.py's oracle_tau).
This is diagnostic only: it still requires scoring every tier first to
compute the pool, defeating the actual point of pruning -- its only
job here is to answer "if tau were perfectly estimated, would the
bounding balls ever be tight enough to prove a block prunable?"

D=16 caveat (per Fable): this harness uses the same toy key dimension
as the rest of this session's fast-iteration experiments. A null result
here does not necessarily predict production head-dim behavior in
either direction -- noted, not chased unless this result is ambiguous.
"""

import sys, math
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

D, Dv = 16, 16
B = 16
W_blocks = 1
needle_pos = 18
NEEDLE_SCALE = 6.0
QUANTILE = 0.90


def make_base(seed, N):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    v_needle = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    return bq, bk, bv, e, v_needle


def oracle_prune_stats(q, k, v, B, W_blocks, quantile=QUANTILE):
    E, H, N, Dh = q.shape
    sm_scale = 1 / math.sqrt(Dh)

    M_base_needed = (N + B - 1) // B
    L = max(1, math.ceil(math.log2(max(M_base_needed, 2))))
    N_padded = B * (1 << L)
    pad = N_padded - N

    qb = F.pad(q, (0, 0, 0, pad)).view(E, H, -1, B, Dh)
    M = qb.shape[2]
    k_flat = F.pad(k, (0, 0, 0, pad))

    centers, radii = {}, {}
    for l in range(L):
        Bl = B * (1 << l)
        Ml = N_padded // Bl
        k_block_l = k_flat.view(E, H, Ml, Bl, Dh)
        center_l = k_block_l.mean(dim=-2)
        radius_l = (k_block_l - center_l.unsqueeze(-2)).norm(dim=-1).max(dim=-1).values
        centers[l] = center_l
        radii[l] = radius_l

    n_tested = 0
    n_pruned = 0

    for m0 in range(M):
        q_m = qb[:, :, m0]
        n = m0 - W_blocks + 1
        candidates = []
        if n > 0:
            for l in range(L):
                if (n >> l) & 1:
                    candidates.append((l, (n >> (l + 1)) << 1))
        if not candidates:
            continue

        # pool REAL scores across every active tier's block (oracle: full knowledge)
        pooled_scores = []
        per_tier_info = []
        for l, block_idx in candidates:
            Bl = B * (1 << l)
            Ml = N_padded // Bl
            if block_idx >= Ml:
                continue
            k_block = k_flat.view(E, H, Ml, Bl, Dh)[:, :, block_idx]
            scores = sm_scale * (q_m @ k_block.transpose(-1, -2))
            pooled_scores.append(scores)
            per_tier_info.append((l, block_idx))

        if not pooled_scores:
            continue
        pooled = torch.cat(pooled_scores, dim=-1)
        tau_oracle = torch.quantile(pooled, quantile, dim=-1, keepdim=True)  # (E,H,B,1)

        q_norm = q_m.norm(dim=-1, keepdim=True)

        for l, block_idx in per_tier_info:
            center = centers[l][:, :, block_idx]
            radius = radii[l][:, :, block_idx]
            bound = sm_scale * (
                (q_m * center.unsqueeze(2)).sum(-1, keepdim=True)
                + q_norm * radius.view(E, H, 1, 1)
            )
            prunable = bound < tau_oracle
            n_tested += prunable.numel()
            n_pruned += int(prunable.sum().item())

    return {"n_tested": n_tested, "n_pruned": n_pruned, "prune_rate": n_pruned / n_tested if n_tested else 0.0}


print("=== Oracle-tau prune-rate sweep, N=256 -> 8192 ===")
print("(tau = quantile over the FULL pooled real scores across all active tiers,")
print(" diagnostic-only -- isolates 'are the balls too loose' from 'was tau_seed biased')")
print()
configs = [
    (256, 240),
    (512, 496),
    (1024, 1008),
    (2048, 2032),
    (4096, 4080),
    (8192, 8176),
]
print(f"{'N':>6} {'query_pos':>10} | {'prune_rate':>10} {'n_tested':>10} {'n_pruned':>10}")
for Nval, qpv in configs:
    bq, bk, bv, e, v_needle = make_base(2000, N=Nval)
    k_full = bk.clone(); k_full[0, 0, needle_pos] = e * NEEDLE_SCALE
    v_full = bv.clone(); v_full[0, 0, needle_pos] = v_needle
    q_full = bq.clone(); q_full[0, 0, qpv] = e * NEEDLE_SCALE
    stats = oracle_prune_stats(q_full, k_full, v_full, B=B, W_blocks=W_blocks)
    print(f"{Nval:>6} {qpv:>10} | {stats['prune_rate']:>10.2%} {stats['n_tested']:>10} {stats['n_pruned']:>10}")
