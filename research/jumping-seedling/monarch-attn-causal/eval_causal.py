import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_causal import monarch_attention_causal_torch

torch.manual_seed(0)

E, H, N, D = 1, 1, 16, 8
B, T = 4, 3

q = torch.randn(E, H, N, D)
k = torch.randn(E, H, N, D)
v = torch.randn(E, H, N, D)

print("=== causal validity (identity-V trick, like get_matrix) ===")
eye = torch.eye(N).expand(E, H, N, N)
A = monarch_attention_causal_torch(q, k, eye, None, T, B, pre_pad=False, causal=True)
leak = torch.triu(A[0, 0], diagonal=1).abs().max().item()
print(f"max future weight (want ~0): {leak:.6e}")
print(f"row sums (want ~1): {A[0,0].sum(-1)[:5]}")

print()
print("=== output vs ground-truth causal softmax attention ===")
z = monarch_attention_causal_torch(q, k, v, None, T, B, pre_pad=False, causal=True)
z_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
mse = (z - z_ref).pow(2).mean().item()
cos = F.cosine_similarity(z.flatten(), z_ref.flatten(), dim=0).item()
print(f"MSE: {mse:.6e}  cosine sim: {cos:.6f}")
