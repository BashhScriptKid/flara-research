"""Robustness sweep for the v2 trained landmark (MLP + multi-head
pooling, direct retrieval loss), per this arc's own established
discipline: a single fixed configuration's win proves nothing on its
own (Sliding wasn't disqualified until swept across decoy count and
far-region length). Trains ONCE at the base config (M=16, 2 decoys/
block, matching the v2 result already found), then evaluates the FROZEN
trained model across:

1. Decoy-pressure sweep (0/1/2/4/6 decoys in the needle's own block,
   holding M=16) -- tests whether the win survives harder within-block
   competition than what it was trained on.
2. Block-count sweep (M=8/16/32/64, holding 2 decoys/block) -- tests
   whether the win survives more cross-block competition / a bigger
   discrimination problem than what it was trained on.

Both swept against the SAME frozen model (no retraining per config) to
test generalization, not just re-optimization for each new setting --
and against a matched untrained baseline at every point.
"""
import sys
sys.path.insert(0, "repo")
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 16
B = 8
BACKGROUND_NORM = 0.5 * (D ** 0.5)
N_TRAIN_STEPS = 1500  # v2 showed the plateau starts ~600-1500; train to the plateau, not further
BATCH_SIZE = 32
LR = 0.01
N_EVAL_TRIALS = 300
H_HEADS = 4
HIDDEN = 32

TRAIN_M = 16
TRAIN_DECOYS = 2


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


def joint_retrieval(model, keys, values, query):
    landmark_k = model(keys)
    landmark_v = values.mean(dim=-2)
    scale = 1.0 / (D ** 0.5)
    block_scores = torch.einsum("nd,nmd->nm", query, landmark_k) * scale
    block_weights = F.softmax(block_scores, dim=-1)
    z = torch.einsum("nm,nmd->nd", block_weights, landmark_v)
    return z, block_scores


def eval_metrics(model, n_trials, seed_offset, m, n_decoys):
    model.eval()
    with torch.no_grad():
        keys, values, query, needle_block, needle_val = make_batch(n_trials, seed_offset, m, n_decoys)
        z, block_scores = joint_retrieval(model, keys, values, query)
        cos = F.cosine_similarity(z, needle_val, dim=-1)
        pred_block = block_scores.argmax(dim=-1)
        block_acc = (pred_block == needle_block).float().mean().item()
        mean_cos = cos.mean().item()
        frac_good = (cos > 0.5).float().mean().item()
    model.train()
    return block_acc, mean_cos, frac_good


print(f"=== Training the v2 model once at base config (M={TRAIN_M}, {TRAIN_DECOYS} decoys/block) ===")
torch.manual_seed(0)
model = RichLandmark(D, HIDDEN, H_HEADS)
opt = torch.optim.Adam(model.parameters(), lr=LR)
for step in range(N_TRAIN_STEPS):
    keys, values, query, needle_block, needle_val = make_batch(BATCH_SIZE, step, TRAIN_M, TRAIN_DECOYS)
    z, block_scores = joint_retrieval(model, keys, values, query)
    loss = -F.cosine_similarity(z, needle_val, dim=-1).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f"training done ({N_TRAIN_STEPS} steps), final train-batch loss={loss.item():.4f}")
print()

torch.manual_seed(1)
model_untrained = RichLandmark(D, HIDDEN, H_HEADS)

print("=== Sweep 1: decoy pressure within the needle's block, M held at training value (16) ===")
print(f"{'n_decoys':>9} | {'trained mean_cos':>17} {'trained frac>0.5':>17} | {'untrained mean_cos':>18} {'untrained frac>0.5':>18}")
for n_decoys in (0, 1, 2, 4, 6):
    _, cos_t, frac_t = eval_metrics(model, N_EVAL_TRIALS, 500_000 + n_decoys, TRAIN_M, n_decoys)
    _, cos_u, frac_u = eval_metrics(model_untrained, N_EVAL_TRIALS, 500_000 + n_decoys, TRAIN_M, n_decoys)
    print(f"{n_decoys:>9} | {cos_t:>17.4f} {frac_t:>17.2%} | {cos_u:>18.4f} {frac_u:>18.2%}")

print()
print("=== Sweep 2: block count (cross-block competition), decoys held at training value (2) ===")
print(f"{'M':>5} | {'trained mean_cos':>17} {'trained frac>0.5':>17} | {'untrained mean_cos':>18} {'untrained frac>0.5':>18}")
for m in (8, 16, 32, 64):
    _, cos_t, frac_t = eval_metrics(model, N_EVAL_TRIALS, 600_000 + m, m, TRAIN_DECOYS)
    _, cos_u, frac_u = eval_metrics(model_untrained, N_EVAL_TRIALS, 600_000 + m, m, TRAIN_DECOYS)
    print(f"{m:>5} | {cos_t:>17.4f} {frac_t:>17.2%} | {cos_u:>18.4f} {frac_u:>18.2%}")
