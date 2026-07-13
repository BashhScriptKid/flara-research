"""v2 of the stress sweep (eval_gated_stress.py), using the richer
per-tier ResidualGateV2 (ma_meta_threshold_gated_v2.py) instead of v1's
query-only global bias. Same scene construction, same curriculum-training
approach, same sweep -- only the gate architecture changes, so results
are directly comparable to the v1 table (which showed zero measurable
effect at every needle strength).
"""
import torch
import torch.nn.functional as F

from ma_meta_threshold_fast_residual import monarch_meta_threshold_fast_residual
from ma_meta_threshold_gated_v2 import ResidualGateV2, monarch_meta_threshold_gated_v2

D, Dv = 16, 16
B = 8
N = 512
W_BLOCKS = 1
QUANTILE = 0.90
QUERY_POS = 261
Q_ALIGN = 6.0
MAX_TIERS = 6  # L for N=512, B=8: ceil(log2(64)) = 6

N_CALIB_TRIALS = 20
NORM_LEVELS = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 3.0, 4.0]

N_TRAIN_STEPS = 600
BATCH_SIZE = 16
LR = 0.02


def make_scene(seed, needle_pos, needle_norm):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    val = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    k_full, v_full = bk.clone(), bv.clone()
    k_full[0, 0, needle_pos] = e * needle_norm
    v_full[0, 0, needle_pos] = val
    q_full = bq.clone()
    q_full[0, 0, QUERY_POS] = e * Q_ALIGN
    return q_full, k_full, v_full, val


def make_batch(batch_size, seed, norm_lo, norm_hi):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(batch_size, 1, N, D, generator=g) * 0.5
    bk = torch.randn(batch_size, 1, N, D, generator=g) * 0.5
    bv = torch.randn(batch_size, 1, N, Dv, generator=g) * 0.5
    needle_pos = torch.randint(0, QUERY_POS, (batch_size,), generator=g)
    needle_norm = norm_lo + torch.rand(batch_size, generator=g) * (norm_hi - norm_lo)
    k_full, v_full, q_full = bk.clone(), bv.clone(), bq.clone()
    vals = torch.zeros(batch_size, Dv)
    for i in range(batch_size):
        e = F.normalize(torch.randn(D, generator=g), dim=0)
        val = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
        k_full[i, 0, needle_pos[i]] = e * needle_norm[i]
        v_full[i, 0, needle_pos[i]] = val
        q_full[i, 0, QUERY_POS] = e * Q_ALIGN
        vals[i] = val
    return q_full, k_full, v_full, vals


def sweep(label, needle_pos, gate):
    print(f"--- {label} ---")
    print(f"{'norm':>6} {'GT':>10} {'Meta-frozen':>12} {'Meta+gateV2':>12}")
    for norm in NORM_LEVELS:
        gt_r, meta_r, gated_r = [], [], []
        with torch.no_grad():
            for trial in range(N_CALIB_TRIALS):
                seed = 50_000 + trial
                q, k, v, val = make_scene(seed, needle_pos, norm)
                z_gt = F.scaled_dot_product_attention(q, k, v, is_causal=True)[0, 0, QUERY_POS]
                gt_r.append(F.cosine_similarity(z_gt, val, dim=0).item())
                z_meta, _ = monarch_meta_threshold_fast_residual(q, k, v, B=B, W_blocks=W_BLOCKS, quantile=QUANTILE)
                meta_r.append(F.cosine_similarity(z_meta[0, 0, QUERY_POS], val, dim=0).item())
                z_gated = monarch_meta_threshold_gated_v2(q, k, v, B=B, W_blocks=W_BLOCKS, gate=gate, quantile=QUANTILE)
                gated_r.append(F.cosine_similarity(z_gated[0, 0, QUERY_POS], val, dim=0).item())
        print(f"{norm:>6.2f} {sum(gt_r)/len(gt_r):>10.4f} {sum(meta_r)/len(meta_r):>12.4f} {sum(gated_r)/len(gated_r):>12.4f}")
    print()


def train_gate_curriculum(norm_lo, norm_hi):
    gate = ResidualGateV2(D, max_tiers=MAX_TIERS)
    opt = torch.optim.Adam(gate.parameters(), lr=LR)
    for step in range(N_TRAIN_STEPS):
        q, k, v, vals = make_batch(BATCH_SIZE, seed=step, norm_lo=norm_lo, norm_hi=norm_hi)
        z = monarch_meta_threshold_gated_v2(q, k, v, B=B, W_blocks=W_BLOCKS, gate=gate, quantile=QUANTILE)
        out = z[:, 0, QUERY_POS]
        loss = (1.0 - F.cosine_similarity(out, vals, dim=-1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == N_TRAIN_STEPS - 1:
            print(f"  step {step:4d}  loss={loss.item():.4f}")
    return gate


if __name__ == "__main__":
    print("=== Training ResidualGateV2 (per-tier context) over the full sweep range ===")
    gate = train_gate_curriculum(0.0, max(NORM_LEVELS))
    print()

    print("=== Sweep with curriculum-trained ResidualGateV2 ===")
    sweep("Far-block needle, needle_norm sweep", needle_pos=64, gate=gate)
    sweep("Diagonal/own-block needle, needle_norm sweep", needle_pos=258, gate=gate)
