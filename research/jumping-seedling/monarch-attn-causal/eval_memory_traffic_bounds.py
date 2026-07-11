"""Memory-traffic axis for the roofline model, scoped as bounds rather
than a full cache simulation (per the cost discussion: a real LRU/
eviction simulator would carry the same false-precision risk this
session already ruled out for PyTorch wall-clock timing -- not
trustworthy until there's a real Rust/AVX2 kernel to profile with
actual hardware counters on the actual 5500U).

Key reframing that changes the shape of this question: threshold
selection reads REAL keys/values directly (k_flat.view(...) is a
reshaped VIEW over the same underlying K,V tensors, not a separate
compressed copy per tier) -- so the DRAM-resident DATA VOLUME is
identical between dense attention and threshold selection: N*D*4 bytes
(K) + N*Dv*4 bytes (V), per head. The real question is ACCESS PATTERN
/ reuse, not total footprint.

Two honest bounds, no simulation:
- FLOOR: every distinct K/V byte read from DRAM exactly once (best
  case, perfect reuse/caching). Identical for dense and threshold
  selection, since both need the same underlying data at least once.
- NAIVE (measured, not estimated): total bytes touched by the actual
  query-major loop with zero reuse credit -- i.e. every (query, active
  key) touch counted as a fresh read. This reuses the exact qk_terms/
  av_terms counters already validated in eval_flop_accounting.py, just
  scaled by element size instead of FLOP count.

Also reports total K+V footprint vs the 5500U's 8MB L3 at
PRODUCTION-scale parameters (not the toy D=16 used elsewhere in this
session for fast iteration) -- at toy scale everything trivially fits
in L3 and the axis is uninformative; the point of this script is to
check whether that's still true at realistic model dimensions.
"""

import sys, math
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

L3_BYTES = 8 * 1024 * 1024
QUANTILE = 0.90


def instrumented_run(q, k, v, B, W_blocks, quantile=QUANTILE):
    E, H, N, Dh = q.shape
    _, _, _, Dvh = v.shape
    sm_scale = 1 / math.sqrt(Dh)

    M_base_needed = (N + B - 1) // B
    L = max(1, math.ceil(math.log2(max(M_base_needed, 2))))
    N_padded = B * (1 << L)
    pad = N_padded - N

    qb = F.pad(q, (0, 0, 0, pad)).view(E, H, -1, B, Dh)
    kb = F.pad(k, (0, 0, 0, pad)).view(E, H, -1, B, Dh)
    M = qb.shape[2]
    valid_mb = (torch.arange(N_padded).view(M, B) < N)
    k_flat = F.pad(k, (0, 0, 0, pad))
    v_flat = F.pad(v, (0, 0, 0, pad))

    qk_key_touches = 0   # count of individual key-vector touches (not query-key pairs) for QK stage
    av_value_touches = 0  # count of individual value-vector touches for AV stage

    for m0 in range(M):
        q_m = qb[:, :, m0]
        w_start = max(0, m0 - W_blocks + 1)
        win_k = kb[:, :, w_start : m0 + 1].reshape(E, H, -1, Dh)
        n_win_blocks = m0 - w_start + 1
        win_valid = valid_mb[w_start : m0 + 1].reshape(-1)

        # local window: every valid key/value in the window is touched once
        # per query-group visit (this loop iteration), not once per query row
        # -- a single batched read serves all B queries in this group.
        n_win_keys = int(win_valid.sum().item())
        qk_key_touches += n_win_keys * E * H
        av_value_touches += n_win_keys * E * H

        n = m0 - W_blocks + 1
        candidates = []
        if n > 0:
            for l in range(L):
                if (n >> l) & 1:
                    candidates.append((l, (n >> (l + 1)) << 1))

        per_tier_scores, per_tier_kv = [], []
        for l, block_idx in candidates:
            Bl = B * (1 << l)
            Ml = N_padded // Bl
            if block_idx >= Ml:
                continue
            k_block = k_flat.view(E, H, Ml, Bl, Dh)[:, :, block_idx]
            scores = sm_scale * (q_m @ k_block.transpose(-1, -2))
            per_tier_scores.append(scores)
            per_tier_kv.append((l, block_idx, Bl))
            # QK stage: this tier's Bl keys are touched once per query-group visit
            qk_key_touches += Bl * E * H

        if per_tier_scores:
            pooled = torch.cat(per_tier_scores, dim=-1)
            shared_tau = torch.quantile(pooled, quantile, dim=-1, keepdim=True)
            for scores, (l, block_idx, Bl) in zip(per_tier_scores, per_tier_kv):
                survivors = scores >= shared_tau
                # AV stage: only surviving VALUES are gathered (real implementation
                # would gather just these; this counts distinct survivor slots per
                # query-group, upper-bounding at Bl per (E,H) since survivors vary
                # per query row within the group)
                any_survivor_col = survivors.any(dim=2)  # (E,H,Bl): does this key survive for ANY query in the group
                av_value_touches += int(any_survivor_col.sum().item())
                has_non_surv = (~survivors).sum(dim=-1) > 0
                av_value_touches += int(has_non_surv.sum().item())  # +1 residual-centroid value per (E,H,row)

    return qk_key_touches, av_value_touches, N, Dh, Dvh, H


print("=== Memory-traffic bounds: dense vs threshold-selection, production-scale D ===")
print(f"L3 budget: {L3_BYTES/1e6:.1f} MB")
print()

D, Dv = 64, 64   # production-ish head dim, not the toy D=16 used elsewhere
H = 8
B = 64
W_blocks = 1

print(f"{'N':>7} | {'K+V floor MB':>13} {'fits L3?':>9} | {'naive DRAM MB (dense)':>21} | {'naive DRAM MB (thresh)':>22} | {'naive ratio':>11}")
for N in (1024, 2048, 4096, 8192, 16384):
    g = torch.Generator().manual_seed(42)
    q = torch.randn(1, H, N, D, generator=g) * 0.5
    k = torch.randn(1, H, N, D, generator=g) * 0.5
    v = torch.randn(1, H, N, Dv, generator=g) * 0.5

    qk_touches, av_touches, N_, Dh, Dvh, H_ = instrumented_run(q, k, v, B=B, W_blocks=W_blocks)

    floor_bytes = N * D * 4 * H + N * Dv * 4 * H  # K + V, once each, per head
    fits = "yes" if floor_bytes <= L3_BYTES else "no"

    naive_thresh_bytes = qk_touches * D * 4 + av_touches * Dv * 4

    # dense naive bound: SAME block-tiled convention as threshold-selection --
    # a block of B queries shares one read of its causal-prefix keys/values
    # (not re-read per individual query row). Using the raw O(N^2) per-pair
    # count here would be an apples-to-oranges comparison, since threshold-
    # selection's count already credits free reuse within each query-group.
    M_dense = (N + B - 1) // B
    dense_key_touches = sum(min((m0 + 1) * B, N) for m0 in range(M_dense))
    naive_dense_bytes = dense_key_touches * D * 4 * H + dense_key_touches * Dv * 4 * H

    ratio = naive_thresh_bytes / naive_dense_bytes

    print(f"{N:>7} | {floor_bytes/1e6:>13.2f} {fits:>9} | {naive_dense_bytes/1e6:>21.1f} | {naive_thresh_bytes/1e6:>22.1f} | {ratio:>11.3f}")

print()
print("FLOOR = best-case DRAM traffic if K,V are read exactly once each (identical")
print("for dense and threshold-selection, since both need the same underlying data).")
print("NAIVE = actual measured traffic in the current query-major loop with ZERO reuse")
print("credit (every touch treated as a fresh DRAM read) -- an upper bound, not a")
print("cache-aware measurement. Real traffic sits somewhere between FLOOR and NAIVE,")
print("depending on implementation-specific loop ordering and cache-blocking, which")
print("has not been decided yet -- flagging this range rather than asserting a point")
print("estimate, consistent with the PyTorch-timing-is-not-a-cost-verdict discipline.")
