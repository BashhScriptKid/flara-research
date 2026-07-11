"""Multi-seed rerun of eval_sliding_multineedle_samenorm.py, per Fable's
methodological catch: the original probe used a SINGLE seed per K value
(one trial for K=1, effectively a point estimate for every row), and
K values weren't even independently comparable to each other (directions
and values are drawn as separate list comprehensions, so the RNG stream
diverges after the first draw -- K=1's needle value isn't the same draw
as K=2's first needle value despite sharing a seed). The apparent
K=1->K=2 non-monotonicity in the original run could be single-draw noise,
not a real mechanism crossover -- this reruns with independent seeds per
K, 30 trials each, reporting mean/min/CI instead of point estimates.
"""
import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_sliding_monarch import sliding_monarch_causal

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


print("=== Multi-seed same-norm-controlled probe: SlidingMonarchAttention ===")
print(f"({N_TRIALS} independent seeds per K, each seed contributing all K needle recalls from that scene)")
print()
print(f"{'K needles':>10} | {'n_recalls':>9} {'mean cos':>9} {'+-1 SE':>8} {'min cos':>8} {'max cos':>8} {'frac>0.5':>9}")
for K in (1, 2, 4, 8):
    all_recalls = []
    for trial in range(N_TRIALS):
        seed = 10_000 + trial  # independent seed stream per K (base offset shared, trial index varies)
        bq, k_full, v_full, directions, values = make_scene(seed=seed, K=K)
        for e, val in zip(directions, values):
            q_full = bq.clone()
            q_full[0, 0, QUERY_POS] = e * 6.0
            z = sliding_monarch_causal(q_full, k_full, v_full, B=B, W_blocks=W_blocks, T=T)[0, 0, QUERY_POS]
            all_recalls.append(F.cosine_similarity(z, val, dim=0).item())

    mean_r = sum(all_recalls) / len(all_recalls)
    se = stderr(all_recalls)
    min_r = min(all_recalls)
    max_r = max(all_recalls)
    frac_good = sum(1 for r in all_recalls if r > 0.5) / len(all_recalls)
    print(f"{K:>10} | {len(all_recalls):>9} {mean_r:>9.4f} {se:>8.4f} {min_r:>8.4f} {max_r:>8.4f} {frac_good:>9.2f}")

print()
print("If K=1's mean cos is consistently negative/near-zero across many independent")
print("trials (tight SE, not just one unlucky draw), that CONFIRMS the detection")
print("failure -- the magnitude-artifact hypothesis holds. If K=1 shows wide variance")
print("straddling zero (large SE relative to mean), that's weak/unreliable signal,")
print("a materially different and less alarming finding than systematic failure.")
