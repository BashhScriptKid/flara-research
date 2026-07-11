"""Adjacent architecture to counter the identified weakness: the v2
sweeps showed the trained landmark is reasonably robust at DISTINGUISHING
the needle's block from within-block decoys (flat across decoy pressure),
but degrades badly at scale because it reads a POOLED landmark VALUE via
a joint softmax over all M blocks -- dilution happens twice (within-block
value-averaging, then cross-block softmax-averaging).

This variant uses the trained landmark PURELY for hard top-1 block
SELECTION (argmax over landmark scores against the real query), then
does REAL per-key attention within just the selected block -- no pooled
value is ever read. This is exactly Option (a) from the original
scoping, and the one principle every surviving mechanism in this arc
(Meta included) shares: a compressed proxy decides WHAT to look at,
never carries the answer itself.

Tests whether the landmark is "good enough" as a discrete selector even
though it wasn't good enough as a value-carrying summarizer -- and
whether this sidesteps the cross-block dilution that broke both the
untrained baselines and the v2 soft-pooling approach.
"""
import sys
sys.path.insert(0, "repo")
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 16
B = 8
BACKGROUND_NORM = 0.5 * (D ** 0.5)
N_TRAIN_STEPS = 1500
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


def soft_joint_retrieval(model, keys, values, query):
    # training objective still uses the SOFT pooled path (differentiable,
    # same as v2) -- only EVALUATION switches to hard-select + real attention
    landmark_k = model(keys)
    landmark_v = values.mean(dim=-2)
    scale = 1.0 / (D ** 0.5)
    block_scores = torch.einsum("nd,nmd->nm", query, landmark_k) * scale
    block_weights = F.softmax(block_scores, dim=-1)
    z = torch.einsum("nm,nmd->nd", block_weights, landmark_v)
    return z, block_scores


def hard_select_then_real_attention(model, keys, values, query, top_k=1):
    landmark_k = model(keys)  # (n, M, D)
    scale = 1.0 / (D ** 0.5)
    block_scores = torch.einsum("nd,nmd->nm", query, landmark_k) * scale  # (n, M)

    n, m, b, d = keys.shape
    topk_idx = block_scores.topk(top_k, dim=-1).indices  # (n, top_k)

    z = torch.zeros(n, D)
    for i in range(n):
        sel_keys = keys[i, topk_idx[i]].reshape(-1, D)      # (top_k*B, D)
        sel_values = values[i, topk_idx[i]].reshape(-1, D)  # (top_k*B, D)
        real_scores = (query[i] @ sel_keys.T) * scale       # REAL per-key scores, no pooling
        real_weights = F.softmax(real_scores, dim=-1)
        z[i] = real_weights @ sel_values
    return z, topk_idx


def eval_hard(model, n_trials, seed_offset, m, n_decoys, top_k=1):
    model.eval()
    with torch.no_grad():
        keys, values, query, needle_block, needle_val = make_batch(n_trials, seed_offset, m, n_decoys)
        z, topk_idx = hard_select_then_real_attention(model, keys, values, query, top_k)
        cos = F.cosine_similarity(z, needle_val, dim=-1)
        selected = (topk_idx == needle_block.unsqueeze(-1)).any(dim=-1)
        select_acc = selected.float().mean().item()
        mean_cos = cos.mean().item()
        frac_good = (cos > 0.5).float().mean().item()
    model.train()
    return select_acc, mean_cos, frac_good


print(f"=== Training v2 model (soft pooling, direct retrieval loss) at base config ===")
torch.manual_seed(0)
model = RichLandmark(D, HIDDEN, H_HEADS)
opt = torch.optim.Adam(model.parameters(), lr=LR)
for step in range(N_TRAIN_STEPS):
    keys, values, query, needle_block, needle_val = make_batch(BATCH_SIZE, step, TRAIN_M, TRAIN_DECOYS)
    z, block_scores = soft_joint_retrieval(model, keys, values, query)
    loss = -F.cosine_similarity(z, needle_val, dim=-1).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f"training done, final train-batch loss={loss.item():.4f}\n")

torch.manual_seed(1)
model_untrained = RichLandmark(D, HIDDEN, H_HEADS)

print("=== Hard-select (top-1) + REAL per-key attention within selected block ===")
print("(same trained/untrained models as before -- only the READ mechanism changes)")
print()
print(f"{'M':>5} | {'trained select_acc':>18} {'trained mean_cos':>17} {'trained frac>0.5':>17} | "
      f"{'untr select_acc':>16} {'untr mean_cos':>14} {'untr frac>0.5':>14}")
for m in (8, 16, 32, 64, 128):
    sel_t, cos_t, frac_t = eval_hard(model, N_EVAL_TRIALS, 700_000 + m, m, TRAIN_DECOYS, top_k=1)
    sel_u, cos_u, frac_u = eval_hard(model_untrained, N_EVAL_TRIALS, 700_000 + m, m, TRAIN_DECOYS, top_k=1)
    print(f"{m:>5} | {sel_t:>18.2%} {cos_t:>17.4f} {frac_t:>17.2%} | {sel_u:>16.2%} {cos_u:>14.4f} {frac_u:>14.2%}")

print()
print("=== Same sweep, top_k=2 (hedge -- read real attention over the top 2 candidate blocks) ===")
print(f"{'M':>5} | {'trained select_acc':>18} {'trained mean_cos':>17} {'trained frac>0.5':>17} | "
      f"{'untr select_acc':>16} {'untr mean_cos':>14} {'untr frac>0.5':>14}")
for m in (8, 16, 32, 64, 128):
    sel_t, cos_t, frac_t = eval_hard(model, N_EVAL_TRIALS, 800_000 + m, m, TRAIN_DECOYS, top_k=2)
    sel_u, cos_u, frac_u = eval_hard(model_untrained, N_EVAL_TRIALS, 800_000 + m, m, TRAIN_DECOYS, top_k=2)
    print(f"{m:>5} | {sel_t:>18.2%} {cos_t:>17.4f} {frac_t:>17.2%} | {sel_u:>16.2%} {cos_u:>14.4f} {frac_u:>14.2%}")
