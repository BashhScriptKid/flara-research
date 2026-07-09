import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

D, Dv = 16, 16
N = 256
QUERY_POS = 128
BLOCK_SPAN = 128
BACKGROUND_NORM = 0.5 * (D ** 0.5)
n_trials = 30

print("=== Fable's decisive control: does PLAIN DENSE causal softmax attention")
print("    (no Monarch, no bucketing) also collapse on this exact adversarial")
print("    needle+decoy construction? ===")
print()
print("If dense attention ALSO collapses: bucket routing is no worse than the ground")
print("truth it approximates -- the cliff is inherited from softmax scoring itself.")
print("If dense attention does NOT collapse: small-candidate-pool dynamics specifically")
print("amplify domination -- a genuinely bucket-routing-specific problem.")
print()

recalls = []
fails = 0
true_misses = 0
for trial in range(n_trials):
    seed = 3000 + trial
    gg = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=gg) * 0.5
    bk = torch.randn(1, 1, N, D, generator=gg) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=gg) * 0.5
    e = F.normalize(torch.randn(D, generator=gg), dim=0)
    val = F.normalize(torch.randn(Dv, generator=gg), dim=0) * 5.0
    needle_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=gg).item()
    decoy_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=gg).item()
    while decoy_pos == needle_pos:
        decoy_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=gg).item()
    decoy_dir = F.normalize(e + 0.3 * torch.randn(D, generator=gg), dim=0)

    k_full, v_full = bk.clone(), bv.clone()
    k_full[0, 0, needle_pos] = e * BACKGROUND_NORM
    v_full[0, 0, needle_pos] = val
    k_full[0, 0, decoy_pos] = decoy_dir * BACKGROUND_NORM * 3.0

    q_full = bq.clone()
    q_full[0, 0, QUERY_POS] = e * 6.0

    # plain dense causal softmax attention, full context, no approximation at all
    z_full = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)
    z = z_full[0, 0, QUERY_POS]
    cos = F.cosine_similarity(z, val, dim=0).item()
    recalls.append(cos)
    if cos < 0.5:
        fails += 1
    if cos < 0.0:
        true_misses += 1

mean_r, min_r = sum(recalls) / len(recalls), min(recalls)
print(f"dense causal softmax attention: mean cos={mean_r:.4f}, min cos={min_r:.4f}, "
      f"fail rate (cos<0.5)={fails/n_trials:.2%}, true-miss rate (cos<0.0)={true_misses/n_trials:.2%}")
print()
print("For reference (same harness, same seeds):")
print("  arithmetic-mean bucket routing: mean=0.3006, fail=83.33%, true-miss=23.33%")
print("  geometric-median bucket routing: mean=0.3533, fail=80.00%, true-miss=20.00%")
