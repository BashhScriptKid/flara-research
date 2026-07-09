import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma.ma_torch import monarch_attention_torch as orig_noncausal
from ma_causal import monarch_attention_causal_torch as causal_single
from ma_causal_dual import monarch_attention_causal_dual_torch as causal_dual

torch.manual_seed(0)
B, T = 4, 3

print(f"{'N':>4} | {'single MSE':>11} {'single cos':>10} | {'dual MSE':>11} {'dual cos':>10} | {'noncausal MSE':>13} {'noncausal cos':>13}")
for N in (16, 64, 128):
    E, H, D = 1, 1, 8
    q = torch.randn(E, H, N, D)
    k = torch.randn(E, H, N, D)
    v = torch.randn(E, H, N, D)

    z_gt_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    z_gt_full = F.scaled_dot_product_attention(q, k, v, is_causal=False)

    z_single = causal_single(q, k, v, None, T, B, pre_pad=False, causal=True)
    z_dual = causal_dual(q, k, v, None, T, B, pre_pad=False)
    z_nc = orig_noncausal(q, k, v, None, T, B, False)

    mse_s = (z_single - z_gt_causal).pow(2).mean().item()
    cos_s = F.cosine_similarity(z_single.flatten(), z_gt_causal.flatten(), dim=0).item()
    mse_d = (z_dual - z_gt_causal).pow(2).mean().item()
    cos_d = F.cosine_similarity(z_dual.flatten(), z_gt_causal.flatten(), dim=0).item()
    mse_nc = (z_nc - z_gt_full).pow(2).mean().item()
    cos_nc = F.cosine_similarity(z_nc.flatten(), z_gt_full.flatten(), dim=0).item()

    print(f"{N:>4} | {mse_s:>11.4e} {cos_s:>10.4f} | {mse_d:>11.4e} {cos_d:>10.4f} | {mse_nc:>13.4e} {cos_nc:>13.4f}")

# causal validity check for dual
print()
print("=== dual causal validity ===")
N = 32
q = torch.randn(1, 1, N, 8)
k = torch.randn(1, 1, N, 8)
eye = torch.eye(N).expand(1, 1, N, N)
A = causal_dual(q, k, eye, None, T, B, pre_pad=False)
leak = torch.triu(A[0, 0], diagonal=1).abs().max().item()
print(f"max future weight (want ~0): {leak:.6e}")
print(f"row sums (want ~1): min={A[0,0].sum(-1).min().item():.6f} max={A[0,0].sum(-1).max().item():.6f}")
