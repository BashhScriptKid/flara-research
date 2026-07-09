import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_meta_monarch import monarch_meta
from landmark_mechanics import MECHANICS

D, Dv = 16, 16
B = 4
N = 256
QUERY_POS = 128
BLOCK_SPAN = 128
W_blocks = 1

# Background keys are randn*0.5, D=16 -> expected norm ~= 0.5*sqrt(16) = 2.0.
# Match the needle's key norm to that instead of the usual 6.0x scale-up --
# only DIRECTION distinguishes the needle now, not magnitude.
BACKGROUND_NORM = 0.5 * (D ** 0.5)


def make_same_norm_needle(seed, needle_pos):
    g = torch.Generator().manual_seed(seed)
    bq = torch.randn(1, 1, N, D, generator=g) * 0.5
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    val = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    k_full = bk.clone(); k_full[0, 0, needle_pos] = e * BACKGROUND_NORM  # SAME norm as background
    v_full = bv.clone(); v_full[0, 0, needle_pos] = val
    q_full = bq.clone(); q_full[0, 0, QUERY_POS] = e * 6.0  # query itself still strong, unrelated to key norm
    return q_full, k_full, v_full, val


print(f"=== Same-norm needle check (key norm matched to background ~{BACKGROUND_NORM:.2f}) ===")
print("Only direction distinguishes the needle now -- tests whether top_magnitude/fps")
print("genuinely detect salience or were exploiting magnitude-scaling in the earlier test.")
print()
for name in ("top_magnitude", "fps", "random_reuse"):
    fn = MECHANICS[name]
    print(f"-- {name} --")
    print(f"{'R':>4} | {'mean cos':>9} {'min cos':>8}")
    for R in (2, 4, 8, 16, 32, 64):
        coses = []
        g = torch.Generator().manual_seed(42)
        for trial in range(5):
            needle_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=g).item()
            q_full, k_full, v_full, val = make_same_norm_needle(seed=100 + trial, needle_pos=needle_pos)
            z = monarch_meta(q_full, k_full, v_full, B=B, W_blocks=W_blocks, R=R, landmark_fn=fn)[0, 0, QUERY_POS]
            coses.append(F.cosine_similarity(z, val, dim=0).item())
        mean_c, min_c = sum(coses) / len(coses), min(coses)
        print(f"{R:>4} | {mean_c:>9.4f} {min_c:>8.4f}")
    print()
