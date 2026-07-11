"""Two cheap, targeted sweeps to test Fable's cross-block-dilution
hypothesis for Sliding's disqualifying weakness (see JOURNAL.md): the
far region isn't just one block's representative vs. the local window --
EVERY Monarch block strictly before the window contributes its own
single representative, all combined in ONE joint softmax. A needle's
signal gets diluted twice: once within its own block (Sinkhorn-averaged
with 7 background keys, pre-score), then again competing against every
other purely-background far-block representative (post-score, but
already-diluted).

Sweep 1 (far-region length): vary QUERY_POS, holding the needle's block
fixed -- more far blocks strictly before the window = more competing
representatives. If cos rises as the far region shrinks, dilution-by-
competition is confirmed as the dominant effect.

Sweep 2 (T-iteration budget): vary T at fixed QUERY_POS. Distinguishes
"needs more refinement iterations" (fixable) from "R=1 collapse is a
structural ceiling regardless of budget" (not fixable) -- a plateau
under more T closes the door on the former permanently.

K=1 (single needle, simplest/clearest signal) throughout, 30 independent
seeds per config, same generator/norm-control as the prior probes.
"""
import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_sliding_monarch import sliding_monarch_causal

D, Dv = 16, 16
B = 8
W_blocks = 1
N = 512
FAR_BLOCK_START = 64
BACKGROUND_NORM = 0.5 * (D ** 0.5)
N_TRIALS = 30


def make_scene(seed):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    val = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    needle_pos = FAR_BLOCK_START  # first slot of the far block, K=1
    k_full, v_full = bk.clone(), bv.clone()
    k_full[0, 0, needle_pos] = e * BACKGROUND_NORM
    v_full[0, 0, needle_pos] = val
    return bq, k_full, v_full, e, val


def stderr(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return (var / n) ** 0.5


def run(query_pos, t, n_trials=N_TRIALS):
    recalls = []
    for trial in range(n_trials):
        seed = 10_000 + trial
        bq, k_full, v_full, e, val = make_scene(seed)
        q_full = bq.clone()
        q_full[0, 0, query_pos] = e * 6.0
        z = sliding_monarch_causal(q_full, k_full, v_full, B=B, W_blocks=W_blocks, T=t)[0, 0, query_pos]
        recalls.append(F.cosine_similarity(z, val, dim=0).item())
    mean_r = sum(recalls) / len(recalls)
    se = stderr(recalls)
    frac_good = sum(1 for r in recalls if r > 0.5) / len(recalls)
    return mean_r, se, frac_good


print("=== Sweep 1: far-region length (QUERY_POS), T=3 fixed ===")
print("(needle block fixed at [64,72); more far blocks compete as QUERY_POS grows)")
print(f"{'QUERY_POS':>10} {'n_far_blocks':>13} | {'mean cos':>9} {'+-1 SE':>8} {'frac>0.5':>9}")
for qp in (80, 96, 128, 192, 256, 384, 480):
    n_far_blocks = qp // B - FAR_BLOCK_START // B  # rough count of far-block competitors strictly before window
    mean_r, se, frac_good = run(query_pos=qp, t=3)
    print(f"{qp:>10} {n_far_blocks:>13} | {mean_r:>9.4f} {se:>8.4f} {frac_good:>9.2f}")

print()
print("=== Sweep 2: T-iteration budget, QUERY_POS=256 fixed ===")
print(f"{'T':>5} | {'mean cos':>9} {'+-1 SE':>8} {'frac>0.5':>9}")
for t in (3, 5, 10, 20):
    mean_r, se, frac_good = run(query_pos=256, t=t)
    print(f"{t:>5} | {mean_r:>9.4f} {se:>8.4f} {frac_good:>9.2f}")

print()
print("Sweep 1: rising cos as QUERY_POS shrinks (fewer far blocks) confirms")
print("cross-block dilution as the dominant mechanism.")
print("Sweep 2: a PLATEAU across T (not converging toward GT's ~0.87) means this")
print("is a structural ceiling, not an iteration-budget problem -- 'more T-iteration'")
print("would not have fixed Sliding regardless of how much compute was spent on it.")
