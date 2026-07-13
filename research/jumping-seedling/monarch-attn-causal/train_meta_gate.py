"""Train the ResidualGate (ma_meta_threshold_gated.py) end-to-end against
ground-truth causal attention, then compare against the frozen
ma_meta_threshold_fast_residual baseline on the SAME same-norm
needle-retrieval battery used throughout this arc
(eval_causal_monarch_samenorm.py's construction), so results are directly
comparable to the existing table. Gate params only (~17 scalars for D=16):
everything else stays frozen, matching the "don't touch validated
selection, only the residual mixing weight" scoping.
"""
import torch
import torch.nn.functional as F

from ma_meta_threshold_fast_residual import monarch_meta_threshold_fast_residual
from ma_meta_threshold_gated import ResidualGate, monarch_meta_threshold_gated

D, Dv = 16, 16
B = 8
N = 512
W_BLOCKS = 1
QUANTILE = 0.90
BACKGROUND_NORM = 0.5 * (D ** 0.5)
QUERY_POS = 261  # matches eval_causal_monarch_samenorm.py

N_TRAIN_STEPS = 400
BATCH_SIZE = 16
LR = 0.02
N_TRIALS = 30


def make_batch(batch_size, seed):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(batch_size, 1, N, D, generator=g) * 0.5
    bk = torch.randn(batch_size, 1, N, D, generator=g) * 0.5
    bv = torch.randn(batch_size, 1, N, Dv, generator=g) * 0.5
    needle_pos = torch.randint(0, QUERY_POS, (batch_size,), generator=g)  # mixes far + diagonal
    k_full, v_full, q_full = bk.clone(), bv.clone(), bq.clone()
    vals = torch.zeros(batch_size, Dv)
    for i in range(batch_size):
        e = F.normalize(torch.randn(D, generator=g), dim=0)
        val = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
        k_full[i, 0, needle_pos[i]] = e * BACKGROUND_NORM
        v_full[i, 0, needle_pos[i]] = val
        q_full[i, 0, QUERY_POS] = e * 6.0
        vals[i] = val
    return q_full, k_full, v_full, vals


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


def train_gate():
    gate = ResidualGate(D)
    opt = torch.optim.Adam(gate.parameters(), lr=LR)
    for step in range(N_TRAIN_STEPS):
        q, k, v, vals = make_batch(BATCH_SIZE, seed=step)
        z = monarch_meta_threshold_gated(q, k, v, B=B, W_blocks=W_BLOCKS, gate=gate, quantile=QUANTILE)
        out = z[:, 0, QUERY_POS]  # (batch, Dv)
        loss = (1.0 - F.cosine_similarity(out, vals, dim=-1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == N_TRAIN_STEPS - 1:
            print(f"  step {step:4d}  loss={loss.item():.4f}  gate.w.norm={gate.w.weight.norm().item():.4f} gate.bias={gate.w.bias.item():.4f}")
    return gate


def run_battery(gate, needle_pos, label):
    print(f"=== {label} (needle at pos {needle_pos}, query at {QUERY_POS}) ===")
    gt_recalls, meta_recalls, gated_recalls = [], [], []
    with torch.no_grad():
        for trial in range(N_TRIALS):
            seed = 30_000 + trial
            q_full, k_full, v_full, val = make_scene(seed, needle_pos)

            z_gt = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)[0, 0, QUERY_POS]
            gt_recalls.append(F.cosine_similarity(z_gt, val, dim=0).item())

            z_meta, _ = monarch_meta_threshold_fast_residual(q_full, k_full, v_full, B=B, W_blocks=W_BLOCKS, quantile=QUANTILE)
            meta_recalls.append(F.cosine_similarity(z_meta[0, 0, QUERY_POS], val, dim=0).item())

            z_gated = monarch_meta_threshold_gated(q_full, k_full, v_full, B=B, W_blocks=W_BLOCKS, gate=gate, quantile=QUANTILE)
            gated_recalls.append(F.cosine_similarity(z_gated[0, 0, QUERY_POS], val, dim=0).item())

    for name, recalls in (("GT", gt_recalls), ("Meta (frozen)", meta_recalls), ("Meta+gate (trained)", gated_recalls)):
        mean_r = sum(recalls) / len(recalls)
        se = stderr(recalls)
        min_r, max_r = min(recalls), max(recalls)
        frac_good = sum(1 for r in recalls if r > 0.5) / len(recalls)
        print(f"  {name:>20} | n={len(recalls):>3} mean={mean_r:>8.4f} +-{se:.4f} min={min_r:>8.4f} max={max_r:>8.4f} frac>0.5={frac_good:.2%}")
    print()


if __name__ == "__main__":
    print("=== training ResidualGate ===")
    gate = train_gate()
    print()
    run_battery(gate, 64, "Far-block needle")
    run_battery(gate, 258, "Diagonal/own-block needle")
