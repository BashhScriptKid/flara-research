"""Exports fixed-seed q/k/v + PyTorch SlidingMonarchAttention reference
output as raw f32 binary, for the Rust crate's cross-validation test.
Uses ma_sliding_monarch.py directly (the validated reference this
session built and tested extensively) -- not reimplemented, imported.
"""
import struct
import sys
sys.path.insert(0, "../monarch-attn-causal")
import torch

from ma_sliding_monarch import sliding_monarch_causal

HEAD_DIM = 16
N_HEADS = 2
SEQ_LEN = 37  # same non-block-aligned length used for the causal reference
B = 8
W_BLOCKS = 2
T = 3
W_REFINE = None  # resolves to W_BLOCKS, the validated default


def write_f32(path, tensor):
    flat = tensor.contiguous().view(-1).tolist()
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(flat)}f", *flat))


g = torch.Generator().manual_seed(2026)
q = torch.randn(1, N_HEADS, SEQ_LEN, HEAD_DIM, generator=g) * 0.5
k = torch.randn(1, N_HEADS, SEQ_LEN, HEAD_DIM, generator=g) * 0.5
v = torch.randn(1, N_HEADS, SEQ_LEN, HEAD_DIM, generator=g) * 0.5

out = sliding_monarch_causal(q, k, v, B=B, W_blocks=W_BLOCKS, T=T, W_refine=W_REFINE)

write_f32("testdata/sliding_q.bin", q.squeeze(0))
write_f32("testdata/sliding_k.bin", k.squeeze(0))
write_f32("testdata/sliding_v.bin", v.squeeze(0))
write_f32("testdata/sliding_out_ref.bin", out.squeeze(0))

print(f"Exported: seq_len={SEQ_LEN}, head_dim={HEAD_DIM}, n_heads={N_HEADS}, B={B}, W_blocks={W_BLOCKS}, T={T}")
print(f"q shape {tuple(q.shape)}, out shape {tuple(out.shape)}")
