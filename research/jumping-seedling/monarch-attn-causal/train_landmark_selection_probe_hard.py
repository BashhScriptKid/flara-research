"""Hard version of train_landmark_selection_probe.py, per the confound
found in the first attempt: plain uniform mean-pooling ALONE got 38.8%
accuracy on the easy version (M=16 clean, independent block
alternatives) with zero training -- the needle's fixed-magnitude
contribution survives naive averaging of 7 near-cancelling random
background keys, a pure architectural artifact unrelated to learning.

Two fixes, matching what actually broke Sliding (not the easy ranking
task the first probe accidentally tested):

1. SAME-NORM DECOYS within the needle's own block: 2-3 other keys in
   the needle's block are ALSO same-norm, high-magnitude, non-cancelling
   contributions (directions uncorrelated with the needle) -- removes
   the "needle is the only large-magnitude survivor of averaging" free
   pass that let plain mean-pooling win the easy version.
2. FULL CROSS-BLOCK JOINT COMPETITION: instead of the needle's block
   only needing to outrank clean independent alternatives, ALL blocks'
   landmarks compete in one softmax against the real query, and the
   loss/metric is VALUE retrieval (cosine similarity to the true needle
   value via softmax-weighted landmark values), not just block-ranking
   accuracy -- matching the actual mechanism (and actual failure mode)
   measured for Sliding, not an easier proxy for it.

Baseline: untrained (random-init) pseudo_query, same architecture, same
data. If trained beats this baseline decisively and untrained stays
near the same magnitude-artifact-driven floor as before, that's real
signal. If both fail (or both succeed) at the same rate, training isn't
adding anything on this harder, more realistic task either.
"""
import sys
sys.path.insert(0, "repo")
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 16
B = 8
M = 16
N_DECOYS_PER_BLOCK = 2   # same-norm decoys IN the needle's own block
BACKGROUND_NORM = 0.5 * (D ** 0.5)
N_TRAIN_STEPS = 3000
BATCH_SIZE = 32
LR = 0.05
N_EVAL_TRIALS = 300


class LearnedLandmark(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pseudo_query = nn.Parameter(torch.randn(dim) * 0.1)

    def forward(self, block_keys):
        scale = 1.0 / (block_keys.shape[-1] ** 0.5)
        scores = (block_keys @ self.pseudo_query) * scale
        weights = F.softmax(scores, dim=-1)
        landmark_k = (weights.unsqueeze(-1) * block_keys).sum(dim=-2)
        return landmark_k, weights


def make_batch(batch_size, seed):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(batch_size, M, B, D, generator=g) * 0.5
    values = torch.randn(batch_size, M, B, D, generator=g) * 0.5

    needle_block = torch.randint(0, M, (batch_size,), generator=g)
    needle_slot = torch.randint(0, B, (batch_size,), generator=g)
    needle_dir = F.normalize(torch.randn(batch_size, D, generator=g), dim=-1)
    needle_val = F.normalize(torch.randn(batch_size, D, generator=g), dim=-1) * 5.0

    for i in range(batch_size):
        keys[i, needle_block[i], needle_slot[i]] = needle_dir[i] * BACKGROUND_NORM
        values[i, needle_block[i], needle_slot[i]] = needle_val[i]

        # same-norm decoys: OTHER slots in the SAME block, also high-magnitude,
        # uncorrelated random directions -- removes the free "only large
        # contributor" pass the easy version accidentally relied on.
        used_slots = {needle_slot[i].item()}
        for _ in range(N_DECOYS_PER_BLOCK):
            slot = torch.randint(0, B, (1,), generator=g).item()
            while slot in used_slots:
                slot = torch.randint(0, B, (1,), generator=g).item()
            used_slots.add(slot)
            decoy_dir = F.normalize(torch.randn(D, generator=g), dim=-1)
            keys[i, needle_block[i], slot] = decoy_dir * BACKGROUND_NORM
            # decoy values are irrelevant random content, not the needle's value

    query = needle_dir * 6.0
    return keys, values, query, needle_block, needle_val


def joint_retrieval(model, keys, values, query, use_pseudo_query=True):
    if use_pseudo_query:
        landmark_k, _ = model(keys)  # (n, M, D)
    else:
        landmark_k = keys.mean(dim=-2)  # untrained baseline: plain mean pooling
    landmark_v = values.mean(dim=-2)  # value side always plain-pooled (landmark is for SELECTION only)

    scale = 1.0 / (D ** 0.5)
    block_scores = torch.einsum("nd,nmd->nm", query, landmark_k) * scale  # (n, M)
    block_weights = F.softmax(block_scores, dim=-1)  # (n, M)
    z = torch.einsum("nm,nmd->nd", block_weights, landmark_v)  # (n, D) -- retrieved value
    return z, block_scores


def eval_metrics(model, n_trials, seed_offset, use_pseudo_query=True):
    model.eval()
    with torch.no_grad():
        keys, values, query, needle_block, needle_val = make_batch(n_trials, seed_offset)
        z, block_scores = joint_retrieval(model, keys, values, query, use_pseudo_query)
        cos = F.cosine_similarity(z, needle_val, dim=-1)
        pred_block = block_scores.argmax(dim=-1)
        block_acc = (pred_block == needle_block).float().mean().item()
        mean_cos = cos.mean().item()
        frac_good = (cos > 0.5).float().mean().item()
    model.train()
    return block_acc, mean_cos, frac_good


print("=== Hard trained-landmark probe: same-norm decoys + full cross-block joint retrieval ===")
print(f"(M={M} blocks, B={B}, {N_DECOYS_PER_BLOCK} same-norm decoys in the needle's own block,")
print(f" metric = VALUE retrieval cosine similarity via joint softmax, not block-ranking accuracy)")
print()

torch.manual_seed(0)
model = LearnedLandmark(D)
opt = torch.optim.Adam(model.parameters(), lr=LR)

block_acc0, cos0, frac0 = eval_metrics(model, N_EVAL_TRIALS, 100_000, use_pseudo_query=False)
print(f"UNTRAINED baseline (plain mean-pooling, no pseudo_query): block_acc={block_acc0:.2%} mean_cos={cos0:.4f} frac>0.5={frac0:.2%}")
block_acc1, cos1, frac1 = eval_metrics(model, N_EVAL_TRIALS, 100_000, use_pseudo_query=True)
print(f"RANDOM-INIT pseudo_query (pre-training):              block_acc={block_acc1:.2%} mean_cos={cos1:.4f} frac>0.5={frac1:.2%}")
print()

for step in range(N_TRAIN_STEPS):
    keys, values, query, needle_block, needle_val = make_batch(BATCH_SIZE, seed=step)
    z, block_scores = joint_retrieval(model, keys, values, query, use_pseudo_query=True)
    loss = F.cross_entropy(block_scores, needle_block)  # still train the SELECTION signal

    opt.zero_grad()
    loss.backward()
    opt.step()

    if (step + 1) % 300 == 0:
        block_acc, cos, frac = eval_metrics(model, N_EVAL_TRIALS, 100_000 + step, use_pseudo_query=True)
        print(f"step {step+1:>5} | loss={loss.item():.4f} | block_acc={block_acc:.2%} mean_cos={cos:.4f} frac>0.5={frac:.2%}")

print()
block_acc_f, cos_f, frac_f = eval_metrics(model, N_EVAL_TRIALS, 999_999, use_pseudo_query=True)
print(f"FINAL trained (fresh seed):     block_acc={block_acc_f:.2%} mean_cos={cos_f:.4f} frac>0.5={frac_f:.2%}")
block_acc_u, cos_u, frac_u = eval_metrics(model, N_EVAL_TRIALS, 999_999, use_pseudo_query=False)
print(f"FINAL untrained (same fresh seed): block_acc={block_acc_u:.2%} mean_cos={cos_u:.4f} frac>0.5={frac_u:.2%}")
print()
print(f"chance block-ranking baseline (1/M): {1/M:.2%}")
print()
print("If trained decisively beats untrained on mean_cos/frac>0.5 (not just block_acc,")
print("which the value-averaging step can still dilute even with correct selection),")
print("that's real signal the trained-landmark axis might work. If trained ~= untrained,")
print("or both collapse near the untrained-baseline floor, training isn't adding value")
print("on this harder, more realistic construction either.")
