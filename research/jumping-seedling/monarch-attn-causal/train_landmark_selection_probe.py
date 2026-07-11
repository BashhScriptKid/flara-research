"""Cheap synthetic pretraining probe for a TRAINED landmark/summarizer,
per the trained-representative-construction axis flagged as genuinely
open (never tested) in the design-space closure consult. Every untrained
representative-construction heuristic in this arc (k-means, magnitude,
FPS, maxpool, LSH, Sinkhorn-refined/Sliding) failed a same-norm
adversarial control -- this tests whether a TRAINED version can succeed
where those failed, before committing to real model integration.

Architecture (Option A from the scoping: landmark for SELECTION only,
never read as a value -- the one principle every surviving mechanism in
this arc, Meta included, shares): a single learned "pseudo-query" vector
p, used to attention-pool over a block's real keys to build that block's
landmark. p is NOT the final query -- it's a learned prior over what
kind of content is salient, trained across many random scenes. At
selection time, the REAL incoming query scores landmarks via dot
product; the landmark that wins gets read via real per-key attention
(not simulated here -- this probe only tests SELECTION accuracy, the
cheap first-stage question).

Training task: M blocks of B random background keys each, ONE block
contains a same-norm-controlled needle (direction varies per scene).
Query is built aligned with the needle's direction (matching the
construction used throughout this arc's other same-norm probes).
Loss: cross-entropy over which block's landmark best matches the query.
"""
import sys
sys.path.insert(0, "repo")
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 16
B = 8            # keys per block
M = 16           # number of blocks
BACKGROUND_NORM = 0.5 * (D ** 0.5)
N_TRAIN_STEPS = 2000
BATCH_SIZE = 32
LR = 0.05
N_EVAL_TRIALS = 200


class LearnedLandmark(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pseudo_query = nn.Parameter(torch.randn(dim) * 0.1)

    def forward(self, block_keys):
        # block_keys: (..., M, B, D) -> landmarks: (..., M, D)
        scale = 1.0 / (block_keys.shape[-1] ** 0.5)
        scores = (block_keys @ self.pseudo_query) * scale  # (..., M, B)
        weights = F.softmax(scores, dim=-1)
        landmark = (weights.unsqueeze(-1) * block_keys).sum(dim=-2)  # (..., M, D)
        return landmark


def make_batch(batch_size, seed):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(batch_size, M, B, D, generator=g) * 0.5
    needle_block = torch.randint(0, M, (batch_size,), generator=g)
    needle_slot = torch.randint(0, B, (batch_size,), generator=g)
    needle_dir = F.normalize(torch.randn(batch_size, D, generator=g), dim=-1)

    for i in range(batch_size):
        keys[i, needle_block[i], needle_slot[i]] = needle_dir[i] * BACKGROUND_NORM

    query = needle_dir * 6.0  # query aligned with the true needle direction
    return keys, query, needle_block


def eval_accuracy(model, n_trials, seed_offset=100_000):
    model.eval()
    correct = 0
    with torch.no_grad():
        keys, query, needle_block = make_batch(n_trials, seed_offset)
        landmarks = model(keys)  # (n_trials, M, D)
        scale = 1.0 / (D ** 0.5)
        scores = torch.einsum("nd,nmd->nm", query, landmarks) * scale  # (n_trials, M)
        pred = scores.argmax(dim=-1)
        correct = (pred == needle_block).sum().item()
    model.train()
    return correct / n_trials


print("=== Trained landmark selection probe: can a learned pseudo-query beat every untrained heuristic? ===")
print(f"(M={M} blocks, B={B} keys/block, D={D}, same-norm-controlled needle, {N_TRAIN_STEPS} train steps)")
print()

torch.manual_seed(0)
model = LearnedLandmark(D)
opt = torch.optim.Adam(model.parameters(), lr=LR)

print(f"pre-training eval accuracy (random init): {eval_accuracy(model, N_EVAL_TRIALS):.2%}")
print(f"chance baseline (1/M): {1/M:.2%}")
print()

for step in range(N_TRAIN_STEPS):
    keys, query, needle_block = make_batch(BATCH_SIZE, seed=step)
    landmarks = model(keys)
    scale = 1.0 / (D ** 0.5)
    scores = torch.einsum("nd,nmd->nm", query, landmarks) * scale
    loss = F.cross_entropy(scores, needle_block)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if (step + 1) % 200 == 0:
        acc = eval_accuracy(model, N_EVAL_TRIALS, seed_offset=100_000 + step)
        print(f"step {step+1:>5} | loss={loss.item():.4f} | held-out selection accuracy={acc:.2%}")

print()
final_acc = eval_accuracy(model, N_EVAL_TRIALS, seed_offset=999_999)
print(f"FINAL held-out selection accuracy (fresh seed): {final_acc:.2%}  (chance: {1/M:.2%})")
print()
print("If final accuracy is well above chance (1/M) and stable, the trained landmark")
print("learned a general salience prior that generalizes across random needle")
print("directions -- a real signal this axis might work where every untrained")
print("heuristic failed. If it stays near chance, training didn't help and the")
print("axis should be filed as also-doesn't-work, not just untested.")
