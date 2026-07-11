"""Exports fixed-seed q/k/v + PyTorch CausalMonarchAttention reference
output as raw f32 binary. Imports ma_causal_dual_opt.py directly (the
original Monarch-family causal-masking building block, predating
SlidingMonarchAttention) -- not reimplemented here.
"""
import struct
import sys
sys.path.insert(0, "../monarch-attn-causal")
import torch

from ma_causal_dual_opt import monarch_attention_causal_dual_opt_torch

HEAD_DIM = 16
N_HEADS = 2
SEQ_LEN = 37  # same non-block-aligned length used for the other references
B = 8
T = 3


def write_f32(path, tensor):
    flat = tensor.contiguous().view(-1).tolist()
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(flat)}f", *flat))


g = torch.Generator().manual_seed(2026)
q = torch.randn(1, N_HEADS, SEQ_LEN, HEAD_DIM, generator=g) * 0.5
k = torch.randn(1, N_HEADS, SEQ_LEN, HEAD_DIM, generator=g) * 0.5
v = torch.randn(1, N_HEADS, SEQ_LEN, HEAD_DIM, generator=g) * 0.5

out = monarch_attention_causal_dual_opt_torch(q, k, v, attn_mask=None, T=T, B=B, pre_pad=False)

write_f32("testdata/causal_monarch_q.bin", q.squeeze(0))
write_f32("testdata/causal_monarch_k.bin", k.squeeze(0))
write_f32("testdata/causal_monarch_v.bin", v.squeeze(0))
write_f32("testdata/causal_monarch_out_ref.bin", out.squeeze(0))

print(f"Exported: seq_len={SEQ_LEN}, head_dim={HEAD_DIM}, n_heads={N_HEADS}, B={B}, T={T}")
print(f"q shape {tuple(q.shape)}, out shape {tuple(out.shape)}")
