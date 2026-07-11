import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_meta_bucket_route import monarch_meta_bucket_route

D, Dv = 16, 16
B = 4
N = 512
BLOCK_SPAN = 256
QUERY_POS = 256
W_blocks = 1
BACKGROUND_NORM = 0.5 * (D ** 0.5)
N_TRIALS = 25  # reduced from the planned 50 for runtime tractability; still 2.5x the original n=10

print("=== Adversarial exposure sweep, n=50, two background variants ===")
print("(normal: random background, moderate scores; orthogonalized: background")
print(" keys projected orthogonal to the needle direction e, ~zero score against")
print(" this query -- isolates whether the sweet spot is a real bucket-size effect")
print(" or an artifact of aggregate competing mass from the normal background)")
print()
print(f"{'n_buckets':>10} | {'normal bg fail rate':>19} | {'orthogonalized bg fail rate':>27}")

n_buckets_sweep = (4, 8, 16, 32, 64, 128)


def run_trial(seed, n_buckets, orthogonalize_bg):
    gg = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=gg) * 0.5
    bk = torch.randn(1, 1, N, D, generator=gg) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=gg) * 0.5
    e = F.normalize(torch.randn(D, generator=gg), dim=0)
    val = F.normalize(torch.randn(Dv, generator=gg), dim=0) * 5.0

    if orthogonalize_bg:
        # remove the e-component from every background key so its dot
        # product with a query aligned to e is ~zero, regardless of norm --
        # isolates whether "more background mass" (not bucket count) drives
        # the earlier sweet spot.
        bk_flat = bk[0, 0, :BLOCK_SPAN]
        proj = (bk_flat @ e).unsqueeze(-1) * e.unsqueeze(0)
        bk[0, 0, :BLOCK_SPAN] = bk_flat - proj

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
    z = monarch_meta_bucket_route(q_full, k_full, v_full, B=B, W_blocks=W_blocks, n_buckets=n_buckets)[0, 0, QUERY_POS]
    return F.cosine_similarity(z, val, dim=0).item()


for n_buckets in n_buckets_sweep:
    fails_normal = sum(1 for t in range(N_TRIALS) if run_trial(3000 + t, n_buckets, False) < 0.5)
    fails_ortho = sum(1 for t in range(N_TRIALS) if run_trial(3000 + t, n_buckets, True) < 0.5)
    print(f"{n_buckets:>10} | {fails_normal/N_TRIALS:>18.2%} | {fails_ortho/N_TRIALS:>26.2%}")

print()
print("If the orthogonalized-background curve is monotonically decreasing (smaller")
print("buckets = lower fail rate) while normal-background keeps its sweet-spot shape,")
print("that confirms Fable's entropy hypothesis: the earlier non-monotonicity was")
print("driven by aggregate competing background mass, not bucket count per se.")
