"""Same-norm-controlled needle probe for CausalMonarchAttention
(ma_causal_dual_opt.py), the mechanism the user actually meant by
"Causal" -- testing the suspicion raised when reading its algorithm:
unlike SlidingMonarchAttention (which has a REAL exact local window and
only compresses the FAR region), CausalMonarchAttention compresses
EVERYTHING, including the query's own/diagonal block (via the causal-
masked representative al_c) -- there is no real per-key exact region
anywhere in this mechanism. If Sliding already failed this same-norm
control despite having a real window to fall back on, CausalMonarch
has strictly less real-data access and should be expected to fail at
least as badly, quite possibly worse -- verifying this directly rather
than assuming it.

Two needle placements tested:
1. FAR block (same construction as the Sliding probe): needle in an
   earlier block, retrieved via the far/reuse representative al_f.
2. DIAGONAL block (the query's OWN block): needle in the SAME block as
   the query. For Sliding this would be a trivial win (exact window
   covers it). For CausalMonarchAttention this is STILL compressed via
   al_c, not read directly -- testing whether even this "easiest
   possible" placement survives.

30 independent seeds per config from the start (multi-seed learned as
mandatory after the single-trial Sliding result turned out to be an
unlucky draw, not representative).
"""
import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_causal_dual_opt import monarch_attention_causal_dual_opt_torch
from ma_meta_threshold_fast_residual import monarch_meta_threshold_fast_residual

D, Dv = 16, 16
B = 8
T = 3
N = 512
BACKGROUND_NORM = 0.5 * (D ** 0.5)
N_TRIALS = 30
QUERY_POS = 261            # intra-block index 5 within block 32 (261//8=32) -- leaves room
                           # for a causally-earlier needle in the SAME block
FAR_BLOCK_START = 64       # earlier block (block 8) -> retrieved via al_f (far/reuse representative)
DIAGONAL_NEEDLE_POS = 258  # same block as QUERY_POS (258//8=32), intra-block index 2, causally before it


def make_scene(seed, needle_pos):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    val = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    k_full, v_full = bk.clone(), bv.clone()
    k_full[0, 0, needle_pos] = e * BACKGROUND_NORM
    v_full[0, 0, needle_pos] = val
    q_full = bq.clone()
    q_full[0, 0, QUERY_POS] = e * 6.0
    return q_full, k_full, v_full, val


def stderr(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return (var / n) ** 0.5


def run_battery(needle_pos, label):
    print(f"=== {label} (needle at pos {needle_pos}, query at {QUERY_POS}) ===")
    gt_recalls, meta_recalls, cm_recalls = [], [], []
    for trial in range(N_TRIALS):
        seed = 30_000 + trial
        q_full, k_full, v_full, val = make_scene(seed, needle_pos)

        z_gt = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)[0, 0, QUERY_POS]
        gt_recalls.append(F.cosine_similarity(z_gt, val, dim=0).item())

        z_meta, _ = monarch_meta_threshold_fast_residual(q_full, k_full, v_full, B=B, W_blocks=1, quantile=0.90)
        meta_recalls.append(F.cosine_similarity(z_meta[0, 0, QUERY_POS], val, dim=0).item())

        z_cm = monarch_attention_causal_dual_opt_torch(q_full, k_full, v_full, attn_mask=None, T=T, B=B, pre_pad=False)[0, 0, QUERY_POS]
        cm_recalls.append(F.cosine_similarity(z_cm, val, dim=0).item())

    for name, recalls in (("GT", gt_recalls), ("Meta", meta_recalls), ("CausalMonarch", cm_recalls)):
        mean_r = sum(recalls) / len(recalls)
        se = stderr(recalls)
        min_r, max_r = min(recalls), max(recalls)
        frac_good = sum(1 for r in recalls if r > 0.5) / len(recalls)
        print(f"  {name:>14} | n={len(recalls):>3} mean={mean_r:>8.4f} +-{se:.4f} min={min_r:>8.4f} max={max_r:>8.4f} frac>0.5={frac_good:.2%}")
    print()


run_battery(FAR_BLOCK_START, "Far-block needle (retrieved via al_f)")
run_battery(DIAGONAL_NEEDLE_POS, "Diagonal/own-block needle (retrieved via al_c)")
