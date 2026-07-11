"""Same-norm-controlled needle probe for LSH-style hashing as a
prefilter mechanism: bucket KEYS individually by sign-of-dot-product
against a fixed set of random hyperplanes (SimHash-style), not by
block-pooling -- technically distinct from every other cheap-signature
family already tried and killed in this arc (k-means centroids, robust
centroids, magnitude, FPS, bounding balls). The query hashes into the
same hyperplane space; only keys sharing the query's hash bucket (or
within Hamming distance <= H of it) survive to real scoring.

Prediction from the design-space closure consult: this should fail the
same same-norm control every other untrained cheap-proxy family failed,
because it's still a signature computed WITHOUT seeing the real query-key
relevance -- structurally the same exclusion-cliff risk Louver-style
threshold selection was invented to eliminate at the start of this arc.
Testing directly rather than assuming.
"""
import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

D, Dv = 16, 16
N = 512
QUERY_POS = 256
BACKGROUND_NORM = 0.5 * (D ** 0.5)
N_TRIALS = 30
N_HYPERPLANES = 8   # 2^8 = 256 buckets, reasonable for N=512 background


def make_scene(seed, needle_pos=64):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    val = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    k_full, v_full = bk.clone(), bv.clone()
    k_full[0, 0, needle_pos] = e * BACKGROUND_NORM  # same-norm control
    v_full[0, 0, needle_pos] = val
    hyperplanes = F.normalize(torch.randn(N_HYPERPLANES, D, generator=g), dim=1)
    return bq, k_full, v_full, e, val, hyperplanes


def lsh_hash(vecs, hyperplanes):
    # vecs: (..., D), hyperplanes: (N_HYPERPLANES, D) -> bucket code per vec
    signs = (vecs @ hyperplanes.T) >= 0  # (..., N_HYPERPLANES) bool
    return signs


def hamming(a, b):
    return (a != b).sum(dim=-1)


def stderr(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return (var / n) ** 0.5


print("=== Same-norm-controlled needle probe: LSH-style per-key hashing prefilter ===")
print(f"({N_HYPERPLANES} random hyperplanes, needle at pos 64, causal query at pos {QUERY_POS},")
print(f" {N_TRIALS} independent seeds; reports: does the needle SURVIVE the hash-bucket filter,")
print(f" and does the query correctly retrieve it via exact attention restricted to survivors)")
print()

for max_hamming in (0, 1, 2):
    recalls = []
    survival_flags = []
    for trial in range(N_TRIALS):
        seed = 20_000 + trial
        bq, k_full, v_full, e, val, hyperplanes = make_scene(seed)
        q_full = bq.clone()
        q_full[0, 0, QUERY_POS] = e * 6.0

        query_vec = q_full[0, 0, QUERY_POS]
        query_hash = lsh_hash(query_vec, hyperplanes)  # (N_HYPERPLANES,)

        # causal candidate keys: positions 0..QUERY_POS
        cand_keys = k_full[0, 0, :QUERY_POS]  # (QUERY_POS, D)
        cand_hashes = lsh_hash(cand_keys, hyperplanes)  # (QUERY_POS, N_HYPERPLANES)
        dists = hamming(cand_hashes, query_hash.unsqueeze(0))  # (QUERY_POS,)
        survivors_mask = dists <= max_hamming

        needle_survives = bool(survivors_mask[64].item())
        survival_flags.append(needle_survives)

        if survivors_mask.sum() == 0:
            recalls.append(0.0)  # no survivors at all -> degenerate, treat as failure
            continue

        surv_idx = survivors_mask.nonzero(as_tuple=True)[0]
        surv_k = cand_keys[surv_idx]
        surv_v = v_full[0, 0, :QUERY_POS][surv_idx]
        scores = (query_vec @ surv_k.T) / (D ** 0.5)
        weights = torch.softmax(scores, dim=0)
        z = weights @ surv_v
        recalls.append(F.cosine_similarity(z, val, dim=0).item())

    mean_r = sum(recalls) / len(recalls)
    se = stderr(recalls)
    frac_good = sum(1 for r in recalls if r > 0.5) / len(recalls)
    needle_survival_rate = sum(survival_flags) / len(survival_flags)
    avg_n_survivors = "n/a"
    print(f"max_hamming={max_hamming} | needle_survival_rate={needle_survival_rate:.2%} | "
          f"mean cos={mean_r:.4f} +-{se:.4f} | frac>0.5={frac_good:.2f}")

print()
print("If needle_survival_rate is low even at max_hamming=2 (i.e. the needle's own")
print("hash bucket often doesn't match the query's), that's the exclusion-cliff")
print("failure mode this whole arc's threshold-selection design was built to avoid --")
print("confirming LSH-style per-key hashing inherits the same problem as every other")
print("untrained cheap-proxy prefilter tried in this arc.")
