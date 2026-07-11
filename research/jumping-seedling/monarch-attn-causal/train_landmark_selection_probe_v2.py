"""Richer architecture + more directly-aligned objective, per the v1
result: a single global learned pseudo-query with proxy classification
loss showed ZERO learning signal (loss flat, trained ~= untrained) on
the hard construction (same-norm decoys + full cross-block joint
retrieval). Two changes:

1. Per-key MLP transform before pooling (2-layer, small hidden dim) --
   gives the model a nonlinear reshaping of key-space to work with,
   instead of a single fixed linear pseudo-query direction.
2. Multi-head pooling (H independent pseudo-queries, concatenated then
   projected) -- more expressive capacity than one attention head.
3. Direct retrieval loss: minimize -cosine_similarity(retrieved, true
   needle value), the ACTUAL downstream objective, instead of a proxy
   block-classification cross-entropy that may not align well with it.

Same hard construction as v1 (same-norm decoys in the needle's own
block, full cross-block joint softmax competition, value retrieval
metric) -- only the model and loss change, so this is a clean
apples-to-apples architecture/objective comparison against v1's null
result, not a new task.
"""
import sys
sys.path.insert(0, "repo")
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 16
B = 8
M = 16
N_DECOYS_PER_BLOCK = 2
BACKGROUND_NORM = 0.5 * (D ** 0.5)
N_TRAIN_STEPS = 3000
BATCH_SIZE = 32
LR = 0.01
N_EVAL_TRIALS = 300
H_HEADS = 4
HIDDEN = 32


class RichLandmark(nn.Module):
    def __init__(self, dim, hidden, n_heads):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.pseudo_queries = nn.Parameter(torch.randn(n_heads, dim) * 0.1)
        self.out_proj = nn.Linear(n_heads * dim, dim)

    def forward(self, block_keys):
        # block_keys: (..., M, B, D)
        transformed = self.mlp(block_keys)  # (..., M, B, D)
        scale = 1.0 / (transformed.shape[-1] ** 0.5)
        # scores per head: (..., M, B, H)
        scores = torch.einsum("...bd,hd->...bh", transformed, self.pseudo_queries) * scale
        weights = F.softmax(scores, dim=-2)  # softmax over B (keys), per head
        # pooled per head: (..., M, H, D)
        pooled = torch.einsum("...bh,...bd->...hd", weights, transformed)
        pooled_flat = pooled.reshape(*pooled.shape[:-2], -1)  # (..., M, H*D)
        landmark = self.out_proj(pooled_flat)  # (..., M, D)
        return landmark


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
        used_slots = {needle_slot[i].item()}
        for _ in range(N_DECOYS_PER_BLOCK):
            slot = torch.randint(0, B, (1,), generator=g).item()
            while slot in used_slots:
                slot = torch.randint(0, B, (1,), generator=g).item()
            used_slots.add(slot)
            decoy_dir = F.normalize(torch.randn(D, generator=g), dim=-1)
            keys[i, needle_block[i], slot] = decoy_dir * BACKGROUND_NORM

    query = needle_dir * 6.0
    return keys, values, query, needle_block, needle_val


def joint_retrieval(model, keys, values, query):
    landmark_k = model(keys)  # (n, M, D)
    landmark_v = values.mean(dim=-2)  # (n, M, D) -- value side stays plain-pooled
    scale = 1.0 / (D ** 0.5)
    block_scores = torch.einsum("nd,nmd->nm", query, landmark_k) * scale
    block_weights = F.softmax(block_scores, dim=-1)
    z = torch.einsum("nm,nmd->nd", block_weights, landmark_v)
    return z, block_scores


def eval_metrics(model, n_trials, seed_offset):
    model.eval()
    with torch.no_grad():
        keys, values, query, needle_block, needle_val = make_batch(n_trials, seed_offset)
        z, block_scores = joint_retrieval(model, keys, values, query)
        cos = F.cosine_similarity(z, needle_val, dim=-1)
        pred_block = block_scores.argmax(dim=-1)
        block_acc = (pred_block == needle_block).float().mean().item()
        mean_cos = cos.mean().item()
        frac_good = (cos > 0.5).float().mean().item()
    model.train()
    return block_acc, mean_cos, frac_good


print("=== v2: richer architecture (MLP + multi-head pooling), direct retrieval-loss training ===")
print(f"(same hard construction as v1: {N_DECOYS_PER_BLOCK} same-norm decoys/block, joint retrieval metric)")
print()

torch.manual_seed(0)
model = RichLandmark(D, HIDDEN, H_HEADS)
opt = torch.optim.Adam(model.parameters(), lr=LR)

block_acc0, cos0, frac0 = eval_metrics(model, N_EVAL_TRIALS, 100_000)
print(f"RANDOM-INIT (pre-training): block_acc={block_acc0:.2%} mean_cos={cos0:.4f} frac>0.5={frac0:.2%}")
print()

for step in range(N_TRAIN_STEPS):
    keys, values, query, needle_block, needle_val = make_batch(BATCH_SIZE, seed=step)
    z, block_scores = joint_retrieval(model, keys, values, query)
    # direct retrieval objective: maximize cosine similarity to the true needle value
    loss = -F.cosine_similarity(z, needle_val, dim=-1).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    if (step + 1) % 300 == 0:
        block_acc, cos, frac = eval_metrics(model, N_EVAL_TRIALS, 100_000 + step)
        print(f"step {step+1:>5} | loss={loss.item():.4f} | block_acc={block_acc:.2%} mean_cos={cos:.4f} frac>0.5={frac:.2%}")

print()
block_acc_f, cos_f, frac_f = eval_metrics(model, N_EVAL_TRIALS, 999_999)
print(f"FINAL trained (fresh seed): block_acc={block_acc_f:.2%} mean_cos={cos_f:.4f} frac>0.5={frac_f:.2%}")

# untrained baseline with the SAME architecture (random init, never trained) for fair comparison
torch.manual_seed(1)
model_untrained = RichLandmark(D, HIDDEN, H_HEADS)
block_acc_u, cos_u, frac_u = eval_metrics(model_untrained, N_EVAL_TRIALS, 999_999)
print(f"UNTRAINED same-architecture baseline (fresh seed): block_acc={block_acc_u:.2%} mean_cos={cos_u:.4f} frac>0.5={frac_u:.2%}")
print()
print(f"chance block-ranking baseline (1/M): {1/M:.2%}")
