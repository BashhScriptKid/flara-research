"""Real tau-threshold rebuild, per Fable: the prior "threshold+residual"
script used a FIXED QUOTA (topk, always exactly round(admit_frac*M)
blocks admitted) -- a finer-grained top-k hedge, not genuine Louver-
style graceful degradation. A real threshold admits a VARIABLE number
of blocks per query, decided by score magnitude (fewer on easy/
unambiguous queries, more on hard/ambiguous ones) -- Louver's actual
principle. The "84.5% of Meta's quality at ~3x cheaper" economics claim
is provisional until measured under real thresholding, since a fixed
quota pays the same cost regardless of query difficulty while a real
threshold's average admitted-count is an OUTPUT of the experiment, not
an input choice.

tau is set per-query as a quantile of that query's own block_scores
(mirroring Meta's own shared-tau mechanism: a query-specific, score-
distribution-derived threshold, not a fixed rank cutoff). Reports the
RESULTING average admitted-block-count (not assumed) alongside
recovery quality, so the FLOP accounting can be redone on measured
numbers.
"""
import sys
sys.path.insert(0, "repo")
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 16
B = 8
BACKGROUND_NORM = 0.5 * (D ** 0.5)
N_TRAIN_STEPS = 3000
BATCH_SIZE = 32
LR = 0.01
N_EVAL_TRIALS = 300
H_HEADS = 4
HIDDEN = 32

TRAIN_M = 16
TRAIN_DECOYS = 2
SEED = 0


class RichLandmark(nn.Module):
    def __init__(self, dim, hidden, n_heads):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.pseudo_queries = nn.Parameter(torch.randn(n_heads, dim) * 0.1)
        self.out_proj = nn.Linear(n_heads * dim, dim)

    def forward(self, block_keys):
        transformed = self.mlp(block_keys)
        scale = 1.0 / (transformed.shape[-1] ** 0.5)
        scores = torch.einsum("...bd,hd->...bh", transformed, self.pseudo_queries) * scale
        weights = F.softmax(scores, dim=-2)
        pooled = torch.einsum("...bh,...bd->...hd", weights, transformed)
        pooled_flat = pooled.reshape(*pooled.shape[:-2], -1)
        return self.out_proj(pooled_flat)


def make_batch(batch_size, seed, m, n_decoys):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(batch_size, m, B, D, generator=g) * 0.5
    values = torch.randn(batch_size, m, B, D, generator=g) * 0.5

    needle_block = torch.randint(0, m, (batch_size,), generator=g)
    needle_slot = torch.randint(0, B, (batch_size,), generator=g)
    needle_dir = F.normalize(torch.randn(batch_size, D, generator=g), dim=-1)
    needle_val = F.normalize(torch.randn(batch_size, D, generator=g), dim=-1) * 5.0

    for i in range(batch_size):
        keys[i, needle_block[i], needle_slot[i]] = needle_dir[i] * BACKGROUND_NORM
        values[i, needle_block[i], needle_slot[i]] = needle_val[i]
        used_slots = {needle_slot[i].item()}
        for _ in range(min(n_decoys, B - 1)):
            slot = torch.randint(0, B, (1,), generator=g).item()
            while slot in used_slots:
                slot = torch.randint(0, B, (1,), generator=g).item()
            used_slots.add(slot)
            decoy_dir = F.normalize(torch.randn(D, generator=g), dim=-1)
            keys[i, needle_block[i], slot] = decoy_dir * BACKGROUND_NORM

    query = needle_dir * 6.0
    return keys, values, query, needle_block, needle_val


def block_scores_fn(model, keys, query):
    landmark_k = model(keys)
    scale = 1.0 / (D ** 0.5)
    return torch.einsum("nd,nmd->nm", query, landmark_k) * scale


print("=== Training selection-objective model (seed 0, identical to prior runs) ===")
torch.manual_seed(SEED)
model = RichLandmark(D, HIDDEN, H_HEADS)
opt = torch.optim.Adam(model.parameters(), lr=LR)
for step in range(N_TRAIN_STEPS):
    keys, values, query, needle_block, needle_val = make_batch(BATCH_SIZE, step + SEED * 100_000, TRAIN_M, TRAIN_DECOYS)
    scores = block_scores_fn(model, keys, query)
    loss = F.cross_entropy(scores, needle_block)
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f"training done, final CE loss={loss.item():.4f}\n")


def tau_threshold_retrieval(model, keys, values, query, tau):
    """tau: a FIXED, GLOBAL scalar (calibrated once, query-independent) --
    NOT a per-query quantile of that same query's own scores. A same-
    distribution quantile always returns ~the same RANK regardless of
    how confident/peaked the distribution is (confirmed empirically:
    the first attempt at this produced std_admit_frac=0.000 across all
    trials -- a rank-based cutoff in disguise, not a real threshold).
    A fixed absolute tau lets admitted-block-count genuinely vary per
    query based on how many of THAT query's scores clear the same bar."""
    n, m, b, d = keys.shape
    scale = 1.0 / (D ** 0.5)
    block_scores = block_scores_fn(model, keys, query)  # (n, M)
    admit_mask = block_scores >= tau  # (n, M), variable count per query, real threshold
    # guard: every query must admit at least 1 block (never zero real reads)
    no_admit = ~admit_mask.any(dim=-1)
    if no_admit.any():
        fallback_idx = block_scores.argmax(dim=-1)
        admit_mask[no_admit, fallback_idx[no_admit]] = True

    z = torch.zeros(n, D)
    n_admit_per_query = torch.zeros(n)
    for i in range(n):
        adm = admit_mask[i]
        n_admit_per_query[i] = adm.sum().item()
        adm_keys = keys[i, adm].reshape(-1, D)
        adm_values = values[i, adm].reshape(-1, D)
        real_scores = (query[i] @ adm_keys.T) * scale
        real_weights_unnorm = real_scores.exp()

        non_adm = ~adm
        if non_adm.any():
            resid_keys = keys[i, non_adm].reshape(-1, D)
            resid_values = values[i, non_adm].reshape(-1, D)
            resid_mean_k = resid_keys.mean(dim=0)
            resid_mean_v = resid_values.mean(dim=0)
            resid_score = (query[i] @ resid_mean_k) * scale
            resid_weight_unnorm = resid_score.exp()
        else:
            resid_weight_unnorm = torch.tensor(0.0)
            resid_mean_v = torch.zeros(D)

        denom = real_weights_unnorm.sum() + resid_weight_unnorm + 1e-6
        z[i] = (real_weights_unnorm @ adm_values + resid_weight_unnorm * resid_mean_v) / denom
    return z, admit_mask, n_admit_per_query


def eval_tau(model, n_trials, seed_offset, m, n_decoys, tau):
    with torch.no_grad():
        keys, values, query, needle_block, needle_val = make_batch(n_trials, seed_offset, m, n_decoys)
        z, admit_mask, n_admit_per_query = tau_threshold_retrieval(model, keys, values, query, tau)
        cos = F.cosine_similarity(z, needle_val, dim=-1)
        needle_admitted = admit_mask.gather(1, needle_block.unsqueeze(-1)).squeeze(-1)
        mean_cos = cos.mean().item()
        frac_good = (cos > 0.5).float().mean().item()
        needle_admit_rate = needle_admitted.float().mean().item()
        avg_admit_frac = (n_admit_per_query / m).mean().item()
        std_admit_frac = (n_admit_per_query / m).std().item()
    return mean_cos, frac_good, needle_admit_rate, avg_admit_frac, std_admit_frac


META_MEAN_COS_M16 = 0.9296

# calibrate fixed tau candidates from the POOLED score distribution over many
# queries (query-independent statistic), not from any single query's own scores
with torch.no_grad():
    cal_keys, cal_values, cal_query, cal_needle_block, cal_needle_val = make_batch(500, 555_555, TRAIN_M, TRAIN_DECOYS)
    cal_scores = block_scores_fn(model, cal_keys, cal_query)  # (500, M)
    pooled_scores = cal_scores.flatten()
    tau_candidates = torch.quantile(pooled_scores, torch.tensor([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97])).tolist()

print(f"=== Real tau-threshold sweep at M={TRAIN_M} (FIXED global tau, admitted count is an OUTPUT not an input) ===")
print(f"{'tau':>8} | {'mean_cos':>9} {'frac>0.5':>9} {'needle_admit':>13} | {'avg_admit_frac':>15} {'std_admit_frac':>15}")
for tau in tau_candidates:
    mean_cos, frac_good, needle_admit_rate, avg_admit_frac, std_admit_frac = eval_tau(
        model, N_EVAL_TRIALS, 999_999, TRAIN_M, TRAIN_DECOYS, tau
    )
    print(f"{tau:>8.3f} | {mean_cos:>9.4f} {frac_good:>9.2%} {needle_admit_rate:>13.2%} | {avg_admit_frac:>15.2%} {std_admit_frac:>15.3f}")

print()
print(f"Meta's mean_cos reference: {META_MEAN_COS_M16:.4f}")
print("std_admit_frac > 0 confirms genuine variable admission (unlike the first, accidentally")
print("rank-based attempt). Compare avg_admit_frac needed to hit ~80% of Meta's mean_cos against")
print("the fixed-quota result (admit_frac=0.3 -> mean_cos=0.7857) to see whether real thresholding")
print("needs a similar, smaller, or larger average admit rate for the same quality.")
