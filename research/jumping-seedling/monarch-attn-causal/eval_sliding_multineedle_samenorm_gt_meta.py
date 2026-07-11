"""Extends eval_sliding_multineedle_samenorm_multiseed.py with GT (exact
dense causal attention) and Meta (threshold selection) baselines on the
IDENTICAL seeds/scenes, per Fable: a weak absolute cosine number (~0.2
mean for Sliding) is uninterpretable without knowing what's achievable
at that signal strength -- this arc's own established practice (see
JOURNAL.md line 412-419, the original Sliding validation's "even ground
truth degrades hard at low signal, converges toward its own noise floor"
caveat). Resolves whether Sliding's ~0.2 mean cos reflects real signal
loss (GT scores meaningfully higher) or an intrinsically hard scene
(GT also lands near ~0.2), and whether Meta's real-per-key-score design
actually avoids this weakness as the structural argument predicted.
"""
import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_sliding_monarch import sliding_monarch_causal
from ma_meta_threshold_fast_residual import monarch_meta_threshold_fast_residual

D, Dv = 16, 16
B = 8
W_blocks = 1
T = 3
N = 512
FAR_BLOCK_START = 64
QUERY_POS = 256
BACKGROUND_NORM = 0.5 * (D ** 0.5)
N_TRIALS = 30


def make_scene(seed, K):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    needle_positions = FAR_BLOCK_START + torch.randperm(B, generator=g)[:K].sort().values
    directions = [F.normalize(torch.randn(D, generator=g), dim=0) for _ in range(K)]
    values = [F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0 for _ in range(K)]
    k_full, v_full = bk.clone(), bv.clone()
    for pos, e, val in zip(needle_positions.tolist(), directions, values):
        k_full[0, 0, pos] = e * BACKGROUND_NORM
        v_full[0, 0, pos] = val
    return bq, k_full, v_full, directions, values


def stderr(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return (var / n) ** 0.5


print("=== GT vs Meta vs Sliding, IDENTICAL same-norm-controlled scenes ===")
print(f"({N_TRIALS} independent seeds per K, same generator/needle-placement/norm-control as the Sliding-only run)")
print()
print(f"{'K':>3} {'mech':>10} | {'n':>5} {'mean cos':>9} {'+-1 SE':>8} {'min':>8} {'max':>8} {'frac>0.5':>9}")

for K in (1, 2, 4, 8):
    gt_recalls, meta_recalls, sliding_recalls = [], [], []
    for trial in range(N_TRIALS):
        seed = 10_000 + trial
        bq, k_full, v_full, directions, values = make_scene(seed=seed, K=K)
        for e, val in zip(directions, values):
            q_full = bq.clone()
            q_full[0, 0, QUERY_POS] = e * 6.0

            z_gt = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)[0, 0, QUERY_POS]
            gt_recalls.append(F.cosine_similarity(z_gt, val, dim=0).item())

            z_meta, _ = monarch_meta_threshold_fast_residual(q_full, k_full, v_full, B=B, W_blocks=W_blocks, quantile=0.90)
            meta_recalls.append(F.cosine_similarity(z_meta[0, 0, QUERY_POS], val, dim=0).item())

            z_sl = sliding_monarch_causal(q_full, k_full, v_full, B=B, W_blocks=W_blocks, T=T)[0, 0, QUERY_POS]
            sliding_recalls.append(F.cosine_similarity(z_sl, val, dim=0).item())

    for name, recalls in (("GT", gt_recalls), ("Meta", meta_recalls), ("Sliding", sliding_recalls)):
        mean_r = sum(recalls) / len(recalls)
        se = stderr(recalls)
        min_r, max_r = min(recalls), max(recalls)
        frac_good = sum(1 for r in recalls if r > 0.5) / len(recalls)
        print(f"{K:>3} {name:>10} | {len(recalls):>5} {mean_r:>9.4f} {se:>8.4f} {min_r:>8.4f} {max_r:>8.4f} {frac_good:>9.2f}")
    print()

print("If GT scores meaningfully higher than Sliding at the same K, that confirms")
print("real signal loss (Sliding's representative genuinely worse than achievable).")
print("If GT also lands near ~0.2, the scene itself is intrinsically hard even for")
print("exact attention, and the finding is about the harness, not about Sliding.")
print("Meta's numbers directly test whether real-per-key-score selection avoids")
print("this weakness the way the structural argument (no compressed salience")
print("decision) predicted.")
