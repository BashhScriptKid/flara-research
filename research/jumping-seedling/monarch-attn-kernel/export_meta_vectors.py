"""Exports fixed-seed q/k/v + PyTorch MetaMonarchAttention (fast-residual,
final confirmed design) reference output as raw f32 binary. Imports
ma_meta_threshold_fast_residual.py directly -- the validated reference
this session built and numerically verified against the masked-reduction
implementation (max abs diff ~3-6e-8) -- not reimplemented here.
"""
import struct
import sys
sys.path.insert(0, "../monarch-attn-causal")
import torch

from ma_meta_threshold_fast_residual import monarch_meta_threshold_fast_residual

HEAD_DIM = 16
N_HEADS = 2
SEQ_LEN = 37  # same non-block-aligned length used for the other two references
B = 8
W_BLOCKS = 1
QUANTILE = 0.90


def write_f32(path, tensor):
    flat = tensor.contiguous().view(-1).tolist()
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(flat)}f", *flat))


g = torch.Generator().manual_seed(2026)
q = torch.randn(1, N_HEADS, SEQ_LEN, HEAD_DIM, generator=g) * 0.5
k = torch.randn(1, N_HEADS, SEQ_LEN, HEAD_DIM, generator=g) * 0.5
v = torch.randn(1, N_HEADS, SEQ_LEN, HEAD_DIM, generator=g) * 0.5

out, _n_survivor_gather_ops = monarch_meta_threshold_fast_residual(
    q, k, v, B=B, W_blocks=W_BLOCKS, quantile=QUANTILE
)

write_f32("testdata/meta_q.bin", q.squeeze(0))
write_f32("testdata/meta_k.bin", k.squeeze(0))
write_f32("testdata/meta_v.bin", v.squeeze(0))
write_f32("testdata/meta_out_ref.bin", out.squeeze(0))

print(f"Exported: seq_len={SEQ_LEN}, head_dim={HEAD_DIM}, n_heads={N_HEADS}, B={B}, W_blocks={W_BLOCKS}, quantile={QUANTILE}")
print(f"q shape {tuple(q.shape)}, out shape {tuple(out.shape)}")
