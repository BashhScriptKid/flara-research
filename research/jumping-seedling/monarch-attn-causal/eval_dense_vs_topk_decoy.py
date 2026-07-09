import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

D, Dv = 16, 16
B = 16
N = 256
needle_pos = 18
NEEDLE_SCALE = 3.0
qp_fixed = 240

print("=== Fable's targeted follow-up: dense-attention control against exact top-k's")
print("    decoy cliff specifically (hard-exclusion mechanism, not soft dilution) ===")
print()
print("Exact same construction as tonight's original top-k decoy stress test.")
print()


def make_base(seed):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    v_needle = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    return bq, bk, bv, e, v_needle


print(f"{'num_decoys':>10} | {'dense cos':>10} | {'topk8 cos':>10} {'topk16 cos':>11}")
print("(topk8/topk16 numbers are tonight's original results, reproduced here for reference)")
topk8_ref = {0: 0.9539, 5: 0.7085, 20: -0.0073, 50: -0.1063}
topk16_ref = {0: 0.9421, 5: 0.7068, 20: 0.3134, 50: -0.0996}

dense_results = {}
for num_decoys in (0, 5, 20, 50):
    cos_vals = []
    for trial in range(10):
        g = torch.Generator().manual_seed(1000 + trial)
        bq, bk, bv, e, v_needle = make_base(1)
        k_full = bk.clone(); k_full[0, 0, needle_pos] = e * NEEDLE_SCALE
        v_full = bv.clone(); v_full[0, 0, needle_pos] = v_needle
        if num_decoys > 0:
            decoy_positions = torch.randperm(qp_fixed - 1, generator=g)[:num_decoys]
            decoy_positions = decoy_positions[decoy_positions != needle_pos]
            decoy_scales = 0.9 + 0.4 * torch.rand(len(decoy_positions), generator=g)
            for pos, dscale in zip(decoy_positions.tolist(), decoy_scales.tolist()):
                k_full[0, 0, pos] = e * (NEEDLE_SCALE * dscale)
        q_full = bq.clone(); q_full[0, 0, qp_fixed] = e * NEEDLE_SCALE

        z_full = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)
        z = z_full[0, 0, qp_fixed]
        cos_vals.append(F.cosine_similarity(z, v_needle, dim=0).item())
    mean_c = sum(cos_vals) / len(cos_vals)
    dense_results[num_decoys] = mean_c
    print(f"{num_decoys:>10} | {mean_c:>10.4f} | {topk8_ref[num_decoys]:>10.4f} {topk16_ref[num_decoys]:>11.4f}")

print()
print("Interpretation: if dense stays flat/graceful while topk crashes to negative,")
print("that confirms top-k's hard-exclusion cutoff is a genuinely WORSE failure mode")
print("than what dense softmax competition alone produces -- unlike the bucket-routing")
print("case, where dense attention collapsed just as much (or more) than the approximation.")
