"""Exports fixed-seed q/k/v + PyTorch dense-causal-GQA reference output as
raw f32 binary, for the Rust crate's cross-validation test to load and
diff against. Uses the same production config (head_dim=64, n_q_heads=14,
n_kv_heads=2) as the roofline analysis, at a small seq_len for a fast test.
"""
import struct
import torch
import torch.nn.functional as F

HEAD_DIM = 64
N_Q_HEADS = 14
N_KV_HEADS = 2
SEQ_LEN = 37  # deliberately not a power of 2 / block-aligned, to stress edge handling


def dense_causal_gqa_reference(q, k, v):
    # q: (n_q_heads, seq_len, head_dim), k/v: (n_kv_heads, seq_len, head_dim)
    group = N_Q_HEADS // N_KV_HEADS
    k_rep = k.repeat_interleave(group, dim=0)  # (n_q_heads, seq_len, head_dim)
    v_rep = v.repeat_interleave(group, dim=0)
    q_b = q.unsqueeze(0)
    k_b = k_rep.unsqueeze(0)
    v_b = v_rep.unsqueeze(0)
    out = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True)
    return out.squeeze(0)  # (n_q_heads, seq_len, head_dim)


def write_f32(path, tensor):
    flat = tensor.contiguous().view(-1).tolist()
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(flat)}f", *flat))


g = torch.Generator().manual_seed(2026)
q = torch.randn(N_Q_HEADS, SEQ_LEN, HEAD_DIM, generator=g) * 0.5
k = torch.randn(N_KV_HEADS, SEQ_LEN, HEAD_DIM, generator=g) * 0.5
v = torch.randn(N_KV_HEADS, SEQ_LEN, HEAD_DIM, generator=g) * 0.5

out = dense_causal_gqa_reference(q, k, v)

write_f32("testdata/causal_q.bin", q)
write_f32("testdata/causal_k.bin", k)
write_f32("testdata/causal_v.bin", v)
write_f32("testdata/causal_out_ref.bin", out)

print(f"Exported: seq_len={SEQ_LEN}, head_dim={HEAD_DIM}, n_q_heads={N_Q_HEADS}, n_kv_heads={N_KV_HEADS}")
print(f"q shape {tuple(q.shape)}, k/v shape {tuple(k.shape)}, out shape {tuple(out.shape)}")
