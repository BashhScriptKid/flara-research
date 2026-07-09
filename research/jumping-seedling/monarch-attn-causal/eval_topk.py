import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_causal_dual_opt import monarch_attention_causal_dual_opt_torch as causal_dual_opt
from ma_causal_topk import monarch_causal_topk as topk_hybrid

torch.manual_seed(0)
B = 16

print("=== causal validity (identity-V trick) ===")
N = 64
q = torch.randn(1, 1, N, 8)
k = torch.randn(1, 1, N, 8)
eye = torch.eye(N).expand(1, 1, N, N)
for kk in (16, 8):
    A = topk_hybrid(q, k, eye, B=B, topk=kk)
    leak = torch.triu(A[0, 0], diagonal=1).abs().max().item()
    row_sums = A[0, 0].sum(-1)
    print(f"k={kk}: max future weight (want ~0): {leak:.6e}, row sums min/max: {row_sums.min().item():.6f}/{row_sums.max().item():.6f}")

print()
print("=== aggregate approximation quality vs exact causal softmax (random Q/K/V) ===")
print(f"{'N':>5} | {'topk16 MSE':>11} {'topk16 cos':>10} | {'topk8 MSE':>11} {'topk8 cos':>10} | {'dual_opt MSE':>12} {'dual_opt cos':>12}")
for N in (64, 256, 1024):
    E, H, D = 1, 1, 8
    q = torch.randn(E, H, N, D)
    k = torch.randn(E, H, N, D)
    v = torch.randn(E, H, N, D)
    z_gt = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    z_16 = topk_hybrid(q, k, v, B=B, topk=16)
    mse_16 = (z_16 - z_gt).pow(2).mean().item()
    cos_16 = F.cosine_similarity(z_16.flatten(), z_gt.flatten(), dim=0).item()

    z_8 = topk_hybrid(q, k, v, B=B, topk=8)
    mse_8 = (z_8 - z_gt).pow(2).mean().item()
    cos_8 = F.cosine_similarity(z_8.flatten(), z_gt.flatten(), dim=0).item()

    z_d = causal_dual_opt(q, k, v, None, T=3, B=B, pre_pad=False)
    mse_d = (z_d - z_gt).pow(2).mean().item()
    cos_d = F.cosine_similarity(z_d.flatten(), z_gt.flatten(), dim=0).item()

    print(f"{N:>5} | {mse_16:>11.4e} {cos_16:>10.4f} | {mse_8:>11.4e} {cos_8:>10.4f} | {mse_d:>12.4e} {cos_d:>12.4f}")

print()
print("=== needle-in-haystack: exact top-k retrieval ===")
torch.manual_seed(1)
N, D, Dv = 256, 16, 16
Bb = 16
needle_pos = 18  # block 1
query_positions = [20, 48, 96, 160, 240]

background_q = torch.randn(1, 1, N, D) * 0.5
background_k = torch.randn(1, 1, N, D) * 0.5
background_v = torch.randn(1, 1, N, Dv) * 0.5

e = F.normalize(torch.randn(D), dim=0)
v_needle = F.normalize(torch.randn(Dv), dim=0) * 5.0
signal_scale = 6.0

k_full = background_k.clone()
k_full[0, 0, needle_pos] = e * signal_scale
v_full = background_v.clone()
v_full[0, 0, needle_pos] = v_needle

mean_v_other = torch.cat(
    [background_v[0, 0, :needle_pos], background_v[0, 0, needle_pos + 1:]], dim=0
).mean(dim=0)

print(f"{'query_pos':>9} {'dist(blocks)':>12} | {'GT cos':>8} | {'topk16 cos':>10} | {'topk8 cos':>9} | {'dual_opt cos':>12} | {'mean-V (ctrl)':>13}")
for qp in query_positions:
    q_full = background_q.clone()
    q_full[0, 0, qp] = e * signal_scale

    z_gt = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)[0, 0, qp]
    z_16 = topk_hybrid(q_full, k_full, v_full, B=Bb, topk=16)[0, 0, qp]
    z_8 = topk_hybrid(q_full, k_full, v_full, B=Bb, topk=8)[0, 0, qp]
    z_d = causal_dual_opt(q_full, k_full, v_full, None, T=3, B=Bb, pre_pad=False)[0, 0, qp]

    cos_gt = F.cosine_similarity(z_gt, v_needle, dim=0).item()
    cos_16 = F.cosine_similarity(z_16, v_needle, dim=0).item()
    cos_8 = F.cosine_similarity(z_8, v_needle, dim=0).item()
    cos_d = F.cosine_similarity(z_d, v_needle, dim=0).item()
    cos_ctrl = F.cosine_similarity(mean_v_other, v_needle, dim=0).item()

    dist_blocks = (qp // Bb) - (needle_pos // Bb)
    # how many background candidates does the needle have to beat at this distance?
    pool_size = (qp // Bb) * Bb
    print(f"{qp:>9} {dist_blocks:>12} | {cos_gt:>8.4f} | {cos_16:>10.4f} | {cos_8:>9.4f} | {cos_d:>12.4f} | {cos_ctrl:>13.4f}  (pool={pool_size})")
