"""Adversarial follow-up to eval_lsh_prefilter_samenorm.py: the natural-case
result (100% survival, mean cos ~0.96-1.0) used a query maximally aligned
with the needle's exact direction -- the easiest possible case for LSH,
since angular-similarity preservation is exactly what LSH guarantees. Per
this arc's established discipline (bucket routing passed its natural case
at 96.4% before failing 83% under adversarial construction), a favorable
natural-case result alone proves nothing -- testing two harder constructions
before drawing any conclusion:

1. Boundary case: needle direction perturbed slightly off the query
   direction (not perfectly parallel), so it may sit near a hyperplane
   decision boundary -- does a small perturbation cause hash-bucket
   exclusion?
2. Adversarial decoy pressure: many decoys constructed to hash into the
   SAME bucket as the needle (same causal region), competing in the
   post-filter softmax -- does LSH have the same decoy-pressure cliff
   that killed bucket routing (83% fail rate) and threshold selection
   under crafted adversarial pressure (matches dense's 90% ceiling)?
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
N_HYPERPLANES = 8
NEEDLE_POS = 64


def lsh_hash(vecs, hyperplanes):
    return (vecs @ hyperplanes.T) >= 0


def hamming(a, b):
    return (a != b).sum(dim=-1)


def stderr(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return (var / n) ** 0.5


def run_trial(q_vec, k_full, v_full, val, hyperplanes, max_hamming):
    query_hash = lsh_hash(q_vec, hyperplanes)
    cand_keys = k_full[0, 0, :QUERY_POS]
    cand_hashes = lsh_hash(cand_keys, hyperplanes)
    dists = hamming(cand_hashes, query_hash.unsqueeze(0))
    survivors_mask = dists <= max_hamming
    needle_survives = bool(survivors_mask[NEEDLE_POS].item())
    if survivors_mask.sum() == 0:
        return 0.0, needle_survives
    surv_idx = survivors_mask.nonzero(as_tuple=True)[0]
    surv_k = cand_keys[surv_idx]
    surv_v = v_full[0, 0, :QUERY_POS][surv_idx]
    scores = (q_vec @ surv_k.T) / (D ** 0.5)
    weights = torch.softmax(scores, dim=0)
    z = weights @ surv_v
    return F.cosine_similarity(z, val, dim=0).item(), needle_survives


print("=== 1: boundary case -- needle direction perturbed off the query direction ===")
print(f"{'perturb':>8} | {'needle_survival':>16} | {'mean cos':>9} {'+-1 SE':>8} {'frac>0.5':>9}")
for perturb in (0.0, 0.1, 0.2, 0.3, 0.5):
    recalls, survs = [], []
    for trial in range(N_TRIALS):
        seed = 20_000 + trial
        g = torch.Generator().manual_seed(seed)
        bq = torch.randn(1, 1, N, D, generator=g) * 0.5
        bk = torch.randn(1, 1, N, D, generator=g) * 0.5
        bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
        e = F.normalize(torch.randn(D, generator=g), dim=0)
        val = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
        needle_dir = F.normalize(e + perturb * torch.randn(D, generator=g), dim=0)
        k_full, v_full = bk.clone(), bv.clone()
        k_full[0, 0, NEEDLE_POS] = needle_dir * BACKGROUND_NORM
        v_full[0, 0, NEEDLE_POS] = val
        hyperplanes = F.normalize(torch.randn(N_HYPERPLANES, D, generator=g), dim=1)
        q_vec = e * 6.0  # query still points at the TRUE direction e, needle is perturbed off it
        cos, surv = run_trial(q_vec, k_full, v_full, val, hyperplanes, max_hamming=1)
        recalls.append(cos)
        survs.append(surv)
    mean_r = sum(recalls) / len(recalls)
    se = stderr(recalls)
    frac_good = sum(1 for r in recalls if r > 0.5) / len(recalls)
    surv_rate = sum(survs) / len(survs)
    print(f"{perturb:>8} | {surv_rate:>16.2%} | {mean_r:>9.4f} {se:>8.4f} {frac_good:>9.2f}")

print()
print("=== 2: adversarial decoy pressure -- decoys hashed into the needle's own bucket ===")
print(f"{'n_decoys':>9} | {'needle_survival':>16} | {'mean cos':>9} {'+-1 SE':>8} {'frac>0.5':>9}")
for n_decoys in (0, 5, 20, 50):
    recalls, survs = [], []
    for trial in range(N_TRIALS):
        seed = 20_000 + trial
        g = torch.Generator().manual_seed(seed)
        bq = torch.randn(1, 1, N, D, generator=g) * 0.5
        bk = torch.randn(1, 1, N, D, generator=g) * 0.5
        bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
        e = F.normalize(torch.randn(D, generator=g), dim=0)
        val = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
        k_full, v_full = bk.clone(), bv.clone()
        k_full[0, 0, NEEDLE_POS] = e * BACKGROUND_NORM
        v_full[0, 0, NEEDLE_POS] = val
        hyperplanes = F.normalize(torch.randn(N_HYPERPLANES, D, generator=g), dim=1)

        # decoys: same-norm, direction close to the needle's (so they land in
        # the SAME hash bucket and compete in the post-filter softmax), but
        # not identical -- distinct decoy values, unrelated to the true val.
        if n_decoys > 0:
            decoy_positions = torch.randperm(QUERY_POS - 1, generator=g)[:n_decoys]
            decoy_positions = decoy_positions[decoy_positions != NEEDLE_POS]
            for pos in decoy_positions.tolist():
                decoy_dir = F.normalize(e + 0.05 * torch.randn(D, generator=g), dim=0)
                k_full[0, 0, pos] = decoy_dir * BACKGROUND_NORM

        q_vec = e * 6.0
        cos, surv = run_trial(q_vec, k_full, v_full, val, hyperplanes, max_hamming=1)
        recalls.append(cos)
        survs.append(surv)
    mean_r = sum(recalls) / len(recalls)
    se = stderr(recalls)
    frac_good = sum(1 for r in recalls if r > 0.5) / len(recalls)
    surv_rate = sum(survs) / len(survs)
    print(f"{n_decoys:>9} | {surv_rate:>16.2%} | {mean_r:>9.4f} {se:>8.4f} {frac_good:>9.2f}")
