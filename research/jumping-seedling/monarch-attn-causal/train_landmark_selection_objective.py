"""Step 1 of Fable's design-scrutiny sequencing: resolve objective-
mismatch vs. architectural-ceiling for the trained-landmark axis.

IMPORTANT -- this is NOT a repeat of v1's null result. v1 tested a weak
single-pseudo-query architecture with CE loss on the EASY task (no
decoys), later shown to be a pooling-magnitude artifact (38.8% accuracy
even at zero parameters / no training). This script tests the RichLandmark
architecture (MLP + 4-head pooling, from v2) with a real listwise CE
selection objective directly on block_scores, on the HARD task (2
same-norm decoys in the needle's own block, full cross-block
competition) -- that specific combination has never been tested. The
existing "34.67% select_acc at M=16" number came from a model trained
on the SOFT-POOLED RETRIEVAL objective (-cosine similarity), then
evaluated post-hoc via argmax -- nothing in that training ever
optimized ranking margin. This isolates the objective as the only
changed variable: same architecture, same hard task, same oracle query,
same M=16/decoys=2 training config as the existing v2/hard-select runs.

Reports margin distribution (needle_score - max_other_score), not just
accuracy -- the real diagnostic for ceiling vs. mismatch. Multi-seed
(3 training seeds) since this is the decisive experiment in the arc so
far and a single positive seed shouldn't be trusted per this arc's own
track record of catching single-run artifacts (see the original K=1
Sliding same-norm result).
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
TRAIN_SEEDS = (0, 1, 2)  # multi-seed per Fable's refinement 4


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

    query = needle_dir * 6.0  # oracle query -- held identical to existing v2/hard-select runs
    return keys, values, query, needle_block, needle_val


def block_scores_fn(model, keys, query):
    landmark_k = model(keys)
    scale = 1.0 / (D ** 0.5)
    return torch.einsum("nd,nmd->nm", query, landmark_k) * scale


def eval_select_and_margin(model, n_trials, seed_offset, m, n_decoys):
    model.eval()
    with torch.no_grad():
        keys, values, query, needle_block, needle_val = make_batch(n_trials, seed_offset, m, n_decoys)
        scores = block_scores_fn(model, keys, query)  # (n, M)
        pred = scores.argmax(dim=-1)
        select_acc = (pred == needle_block).float().mean().item()

        # margin = needle's own score - max score among all OTHER blocks
        needle_score = scores.gather(1, needle_block.unsqueeze(-1)).squeeze(-1)
        masked = scores.clone()
        masked.scatter_(1, needle_block.unsqueeze(-1), float("-inf"))
        max_other = masked.max(dim=-1).values
        margin = (needle_score - max_other)
    model.train()
    return select_acc, margin


def train_one_seed(seed):
    torch.manual_seed(seed)
    model = RichLandmark(D, HIDDEN, H_HEADS)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for step in range(N_TRAIN_STEPS):
        keys, values, query, needle_block, needle_val = make_batch(BATCH_SIZE, step + seed * 100_000, TRAIN_M, TRAIN_DECOYS)
        scores = block_scores_fn(model, keys, query)
        loss = F.cross_entropy(scores, needle_block)  # listwise CE directly on block_scores
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, loss.item()


print("=== Step 1: train-for-selection objective (listwise CE on block_scores) ===")
print("(RichLandmark arch from v2, HARD task -- 2 same-norm decoys, oracle query --")
print(" only the OBJECTIVE changes vs. the existing 34.67% select_acc number)")
print()

models = []
for seed in TRAIN_SEEDS:
    model, final_loss = train_one_seed(seed)
    models.append(model)
    print(f"seed {seed}: training done, final train-batch CE loss={final_loss:.4f}")
print()

print("=== Comparison table at M=16 (training config), per Fable's refinement 6 ===")
accs_new = []
margins_new_all = []
for i, seed in enumerate(TRAIN_SEEDS):
    acc, margin = eval_select_and_margin(models[i], N_EVAL_TRIALS, 999_999, TRAIN_M, TRAIN_DECOYS)
    accs_new.append(acc)
    margins_new_all.append(margin)
    print(f"  (c) selection-trained seed={seed}: select_acc={acc:.2%}, "
          f"margin mean={margin.mean().item():.4f} median={margin.median().item():.4f} "
          f"frac_margin>0={((margin>0).float().mean().item()):.2%}")

acc_mean = sum(accs_new) / len(accs_new)
acc_spread = max(accs_new) - min(accs_new)
print(f"\n  (c) selection-trained, mean over {len(TRAIN_SEEDS)} seeds: "
      f"select_acc={acc_mean:.2%} (spread={acc_spread:.2%})")

print(f"  (a) EXISTING soft-pooled-trained + hard-select-eval (from prior run): select_acc=34.67%")
print(f"  (b) EXISTING untrained control (from prior run): select_acc=7.00%")
print(f"  chance baseline (1/{TRAIN_M}): {1/TRAIN_M:.2%}")

print()
print("=== M-sweep with the selection-trained model (seed 0), per Fable's refinement 5 ===")
print(f"{'M':>5} | {'top-1 select_acc':>17} {'margin mean':>12} {'margin median':>14}")
model0 = models[0]
for m in (8, 16, 32, 64, 128):
    acc, margin = eval_select_and_margin(model0, N_EVAL_TRIALS, 700_000 + m, m, TRAIN_DECOYS)
    print(f"{m:>5} | {acc:>17.2%} {margin.mean().item():>12.4f} {margin.median().item():>14.4f}")

print()
print("Reference (existing soft-pooled-trained model's M-sweep, top-1, for comparison):")
print("  M=8: 49.00%, M=16: 34.67%, M=32: 22.67%, M=64: 17.67%, M=128: 11.33%")
print()
print("If selection-training shifts accuracy up at every M while preserving the SHAPE")
print("of the degradation curve, that confirms the order-statistic explanation (fixed")
print("margin distribution, growing noise ceiling ~sqrt(2 ln M)) -- meaning threshold/")
print("log(M)-scaled-k is the right read mechanism regardless of objective, not top-1.")
print("If margins still show heavy overlap near zero despite CE training, that's")
print("evidence of a soft architectural ceiling, not a fixable objective mismatch.")
