"""Oracle-tau vs reservoir-tau correctness experiment (Fable's proposed
falsifiable test): threshold-based (Louver-style) selection replacing
bucket routing's read step, tested on the SAME harnesses used for the
natural mis-routing and adversarial cliff findings. No new kernels --
this tests the SELECTION mechanism in isolation, exact scores over the
whole stressed block, joint softmax with the local window exactly like
ma_meta_bucket_route.py's structure, just swapping the read step.

Prediction being tested: oracle tau should drive the natural 3.60%
mis-routing floor to ~0 (fixes the exclusion failure). Oracle tau
should NOT materially improve the 83% adversarial failure rate (cannot
fix the in-scope scoring competition once the decoy also exceeds tau).
If either half fails, the mental model from the Fable synthesis is
wrong.
"""

import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

D, Dv = 16, 16
B = 4
N = 256
QUERY_POS = 128
BLOCK_SPAN = 128
W_blocks = 1
BACKGROUND_NORM = 0.5 * (D ** 0.5)
sm_scale = 1 / (D ** 0.5)


def threshold_attention(q_full, k_full, v_full, tau_mode, needle_score=None):
    """q_full/k_full/v_full: (1,1,N,D)/(1,1,N,D)/(1,1,N,Dv). Local window
    is just the query's own tiny block (W_blocks=1, matches the bucket-
    routing tests); the stressed block [0,BLOCK_SPAN) is read via
    threshold selection instead of bucket routing."""
    k_block = k_full[0, 0, :BLOCK_SPAN]  # (BLOCK_SPAN, D)
    v_block = v_full[0, 0, :BLOCK_SPAN]  # (BLOCK_SPAN, Dv)
    query = q_full[0, 0, QUERY_POS]  # (D,)

    scores = sm_scale * (k_block @ query)  # (BLOCK_SPAN,)

    if tau_mode == "oracle":
        assert needle_score is not None
        tau = needle_score - 1e-6  # guarantees the needle's own score clears tau
    elif tau_mode == "reservoir":
        # Louver-style: estimate tau from a sample of the block's own
        # scores at a target selectivity (top ~10% -> roughly R=13 survivors
        # out of 128, a believable operating point), NOT from the needle's
        # own score -- this is the realistic, needle-blind estimator.
        sample = scores  # whole block available here; a real system would subsample
        tau = torch.quantile(sample, 0.90).item()
    else:
        raise ValueError(tau_mode)

    survivors = scores >= tau
    masked_scores = scores.masked_fill(~survivors, -float("inf"))

    # local window: query's own tiny block (post-stressed-block, causally
    # separate) -- for this isolated test just use the stressed block's
    # threshold-selected survivors directly as the sole context, matching
    # how the earlier bucket-routing tests measured the far-branch alone.
    row_max = torch.clamp(masked_scores.max(), min=-1e30)
    exp_s = torch.nan_to_num(torch.exp(masked_scores - row_max), nan=0.0)
    denom = exp_s.sum() + 1e-6
    weights = exp_s / denom
    z = weights @ v_block
    return z, survivors.sum().item()


def make_needle_scene(seed, needle_pos, needle_scale):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    val = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    k_full = bk.clone(); k_full[0, 0, needle_pos] = e * needle_scale
    v_full = bv.clone(); v_full[0, 0, needle_pos] = val
    q_full = bq.clone(); q_full[0, 0, QUERY_POS] = e * 6.0
    needle_score = sm_scale * (e * needle_scale) @ (e * 6.0)
    return q_full, k_full, v_full, val, needle_score.item()


print("=== Part 1: natural mis-routing floor, oracle vs reservoir tau, n=200 ===")
print("(prediction: oracle should drive fail rate toward ~0)")
print()
N_TRIALS = 200
for tau_mode in ("oracle", "reservoir"):
    coses = []
    fails = 0
    survivor_counts = []
    g = torch.Generator().manual_seed(7)
    for trial in range(N_TRIALS):
        needle_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=g).item()
        q_full, k_full, v_full, val, needle_score = make_needle_scene(2000 + trial, needle_pos, BACKGROUND_NORM)
        z, n_surv = threshold_attention(q_full, k_full, v_full, tau_mode, needle_score)
        cos = F.cosine_similarity(z, val, dim=0).item()
        coses.append(cos)
        survivor_counts.append(n_surv)
        if cos < 0.5:
            fails += 1
    mean_c = sum(coses) / len(coses)
    mean_surv = sum(survivor_counts) / len(survivor_counts)
    print(f"{tau_mode:>10}: mean cos={mean_c:.4f}, fail rate (cos<0.5)={fails/N_TRIALS:.2%}, "
          f"avg survivors={mean_surv:.1f}/{BLOCK_SPAN}")

print()
print("=== Part 2: adversarial construction, oracle vs reservoir tau, n=30 ===")
print("(prediction: oracle should NOT materially improve on the 83.33% baseline)")
print()
N_ADV_TRIALS = 30
for tau_mode in ("oracle", "reservoir"):
    coses = []
    fails = 0
    for trial in range(N_ADV_TRIALS):
        seed = 3000 + trial
        gg = torch.Generator().manual_seed(seed)
        bq = torch.randn(1, 1, N, D, generator=gg) * 0.5
        bk = torch.randn(1, 1, N, D, generator=gg) * 0.5
        bv = torch.randn(1, 1, N, Dv, generator=gg) * 0.5
        e = F.normalize(torch.randn(D, generator=gg), dim=0)
        val = F.normalize(torch.randn(Dv, generator=gg), dim=0) * 5.0
        needle_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=gg).item()
        decoy_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=gg).item()
        while decoy_pos == needle_pos:
            decoy_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=gg).item()
        decoy_dir = F.normalize(e + 0.3 * torch.randn(D, generator=gg), dim=0)

        k_full, v_full = bk.clone(), bv.clone()
        k_full[0, 0, needle_pos] = e * BACKGROUND_NORM
        v_full[0, 0, needle_pos] = val
        k_full[0, 0, decoy_pos] = decoy_dir * (BACKGROUND_NORM * 3.0)

        q_full = bq.clone()
        q_full[0, 0, QUERY_POS] = e * 6.0
        needle_score = (sm_scale * (e * BACKGROUND_NORM) @ (e * 6.0)).item()

        z, n_surv = threshold_attention(q_full, k_full, v_full, tau_mode, needle_score)
        cos = F.cosine_similarity(z, val, dim=0).item()
        coses.append(cos)
        if cos < 0.5:
            fails += 1
    mean_c = sum(coses) / len(coses)
    print(f"{tau_mode:>10}: mean cos={mean_c:.4f}, fail rate (cos<0.5)={fails/N_ADV_TRIALS:.2%}")

print()
print("Reference: bucket routing (arithmetic-mean) on this exact adversarial")
print("construction gave 83.33% fail rate, mean cos=0.3006.")
