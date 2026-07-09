import sys
sys.path.insert(0, "repo")
import torch
from ma.monarch_attention import MonarchAttention

torch.manual_seed(0)
N, D, B, T = 16, 8, 4, 3
E, H = 1, 1

q = torch.randn(E, H, N, D)
k = torch.randn(E, H, N, D)

ma = MonarchAttention(block_size=B, num_steps=T, pad_type="post", impl="torch")

# Try a real (N, N) pairwise causal mask, as a decoder would need.
causal_2d = torch.tril(torch.ones(N, N, dtype=torch.bool))
causal_mask_nn = causal_2d.view(1, 1, N, N).expand(E, H, N, N)

print("=== Attempt 1: pass a real (N,N) causal mask ===")
try:
    A = ma.get_matrix(q, k, attention_mask=causal_mask_nn)
    print("Accepted shape (N,N) mask without error. A.shape =", A.shape)
    leak = torch.triu(A[0, 0], diagonal=1).abs().max().item()
    print(f"max |A[i,j]| for j>i (should be 0 if causal-safe): {leak:.6e}")
except Exception as e:
    print(f"REJECTED: {type(e).__name__}: {e}")

print()
print("=== Attempt 2: pass the (N,)-shaped mask it actually expects (padding-style) ===")
# what the code's .view(E,1,M,B) implies: a per-key mask, shape (E, N) or (E,1,N)
padding_mask = torch.ones(E, N, dtype=torch.bool)  # all valid, no padding
A2 = ma.get_matrix(q, k, attention_mask=padding_mask)
print("Accepted 1D-per-key mask. A.shape =", A2.shape)
# This mask is identical for every query row -- confirm it cannot express causality
# by checking if row 0 (which should only see key 0 if causal) sees later keys.
print("A2[0,0] row 0 (query 0) attention over all 16 keys:")
print(A2[0, 0, 0].detach().numpy())
print(f"-> nonzero weight on keys > 0 for query 0: {A2[0,0,0,1:].abs().max().item():.6e}")
print("(if this is nonzero, query 0 attends to future keys -- mask is NOT causal, it's padding-only)")
