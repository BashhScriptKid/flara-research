"""Step 2 of Fable's sequencing: threshold+residual read mechanism,
replacing hard top-k selection. Per Fable: top-1 select_acc demanding
the needle beat M-1 competitors is a much harder bar than what
tau-threshold gating requires -- a generous threshold admitting the
top-K% of blocks by landmark score into "active" (real per-key scoring)
status, with the REST folded into ONE residual aggregate (Meta-style,
never silently dropped), only needs the needle's score to clear a loose
bar, not win an argmax outright. A landmark with near-zero-but-negative
mean margin may still push the needle above a lenient threshold most of
the time even though it rarely wins outright -- untested until now.

KILL CRITERION, set BEFORE running, not decided post-hoc (per Fable's
explicit instruction): if recovering >=80% of Meta's mean_cos requires
admitting >50% of blocks into the active tier at M=16, this mechanism
has no economic case over Meta (defeats the whole point of a sub-O(N)
scoring win) and the trained-landmark axis is closed.

Uses the SAME selection-trained model/config as
train_landmark_selection_objective.py (seed 0, listwise CE on
block_scores, hard task: 2 same-norm decoys, M=16, oracle query) --
retrained here identically for a self-contained script.
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

# kill criterion, pinned BEFORE running
KILL_TARGET_FRAC_OF_META = 0.80
KILL_MAX_ADMIT_FRAC = 0.50


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


print("=== Training selection-objective model (seed 0, same as prior run) ===")
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


def threshold_residual_retrieval(model, keys, values, query, admit_frac):
    """admit_frac: fraction of M blocks (by rank) admitted to real per-key
    attention. Rest folded into ONE residual aggregate (mean value across
    all non-admitted blocks), matching Meta's never-silently-drop
    principle."""
    n, m, b, d = keys.shape
    scale = 1.0 / (D ** 0.5)
    block_scores = block_scores_fn(model, keys, query)  # (n, M)

    n_admit = max(1, int(round(admit_frac * m)))
    admit_idx = block_scores.topk(n_admit, dim=-1).indices  # (n, n_admit)
    admit_mask = torch.zeros(n, m, dtype=torch.bool)
    admit_mask.scatter_(1, admit_idx, True)

    z = torch.zeros(n, D)
    needle_admitted = torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        adm = admit_mask[i]
        adm_keys = keys[i, adm].reshape(-1, D)      # (n_admit*B, D)
        adm_values = values[i, adm].reshape(-1, D)
        real_scores = (query[i] @ adm_keys.T) * scale
        real_weights_unnorm = real_scores.exp()

        # residual: one aggregate candidate for ALL non-admitted blocks
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
    return z, admit_mask


def eval_threshold(model, n_trials, seed_offset, m, n_decoys, admit_frac):
    with torch.no_grad():
        keys, values, query, needle_block, needle_val = make_batch(n_trials, seed_offset, m, n_decoys)
        z, admit_mask = threshold_residual_retrieval(model, keys, values, query, admit_frac)
        cos = F.cosine_similarity(z, needle_val, dim=-1)
        needle_admitted = admit_mask.gather(1, needle_block.unsqueeze(-1)).squeeze(-1)
        mean_cos = cos.mean().item()
        frac_good = (cos > 0.5).float().mean().item()
        needle_admit_rate = needle_admitted.float().mean().item()
    return mean_cos, frac_good, needle_admit_rate


META_MEAN_COS_M16 = 0.9296  # from eval_causal_monarch_samenorm.py's diagonal-needle battery, same construction family
kill_target = KILL_TARGET_FRAC_OF_META * META_MEAN_COS_M16

print(f"=== Threshold+residual sweep at M={TRAIN_M} (training config) ===")
print(f"Kill criterion (set before running): recovering >={KILL_TARGET_FRAC_OF_META:.0%} of Meta's mean_cos")
print(f"({kill_target:.4f}) must NOT require admitting >{KILL_MAX_ADMIT_FRAC:.0%} of blocks.")
print()
print(f"{'admit_frac':>10} {'n_admit':>8} | {'mean_cos':>9} {'frac>0.5':>9} {'needle_admit_rate':>18}")
results = []
for admit_frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
    mean_cos, frac_good, needle_admit_rate = eval_threshold(model, N_EVAL_TRIALS, 999_999, TRAIN_M, TRAIN_DECOYS, admit_frac)
    n_admit = max(1, int(round(admit_frac * TRAIN_M)))
    results.append((admit_frac, mean_cos))
    print(f"{admit_frac:>10.1f} {n_admit:>8} | {mean_cos:>9.4f} {frac_good:>9.2%} {needle_admit_rate:>18.2%}")

print()
# find minimum admit_frac clearing the kill target
clearing = [af for af, mc in results if mc >= kill_target]
if clearing:
    min_clear = min(clearing)
    verdict = "PASSES" if min_clear <= KILL_MAX_ADMIT_FRAC else "FAILS"
    print(f"Minimum admit_frac clearing {KILL_TARGET_FRAC_OF_META:.0%} of Meta's mean_cos: {min_clear:.0%}")
    print(f"KILL CRITERION: {verdict} (threshold was <= {KILL_MAX_ADMIT_FRAC:.0%} admit_frac)")
else:
    print(f"NEVER clears {KILL_TARGET_FRAC_OF_META:.0%} of Meta's mean_cos even at admit_frac=1.0")
    print("KILL CRITERION: FAILS")
