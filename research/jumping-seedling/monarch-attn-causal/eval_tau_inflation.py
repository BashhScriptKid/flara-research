"""Tau-inflation attack test (Fable's proposed follow-up): does a
population of many MODERATE decoys drag the RESERVOIR tau estimate
upward until it excludes the needle, even though no single decoy
dominates the needle at scoring time (the mechanism behind the earlier
83%/90% adversarial cliff)? This is a different attack shape -- the
earlier test had one high-magnitude, direction-correlated decoy that
both oracle and reservoir tau admitted equally (0.90/0.90 fail rate,
identical). This test specifically targets tau ESTIMATION itself.

Oracle tau here = the quantile computed from a decoy-free REFERENCE
sample of the same background distribution (an idealized, uncorrupted
threshold, held fixed as a control). Reservoir tau = the quantile
computed from the ACTUAL block including the injected decoys -- the
one that can legitimately be dragged.
"""

import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

D, Dv = 16, 16
N = 256
QUERY_POS = 128
BLOCK_SPAN = 128
BACKGROUND_NORM = 0.5 * (D ** 0.5)
sm_scale = 1 / (D ** 0.5)
QUANTILE = 0.90


def run_trial(seed, num_decoys, decoy_strength_ratio):
    g = torch.Generator().manual_seed(seed)
    bk = torch.randn(1, 1, N, D, generator=g) * 0.5
    bv = torch.randn(1, 1, N, Dv, generator=g) * 0.5
    e = F.normalize(torch.randn(D, generator=g), dim=0)
    val = F.normalize(torch.randn(Dv, generator=g), dim=0) * 5.0
    needle_pos = torch.randint(0, BLOCK_SPAN, (1,), generator=g).item()

    query = e * 6.0

    # oracle tau: quantile from a DECOY-FREE reference sample of the SAME
    # background (same seed's own clean background scores, before any
    # needle/decoy injection) -- the idealized, uncorrupted threshold.
    clean_k_block = bk[0, 0, :BLOCK_SPAN]
    clean_scores = sm_scale * (clean_k_block @ query)
    oracle_tau = torch.quantile(clean_scores, QUANTILE).item()

    # now build the actual (corrupted) scene: needle + k moderate decoys
    k_full = bk.clone()
    v_full = bv.clone()
    k_full[0, 0, needle_pos] = e * BACKGROUND_NORM
    v_full[0, 0, needle_pos] = val
    needle_score = (sm_scale * (e * BACKGROUND_NORM) @ query).item()

    if num_decoys > 0:
        decoy_positions = torch.randperm(BLOCK_SPAN, generator=g)[:num_decoys + 1]
        decoy_positions = decoy_positions[decoy_positions != needle_pos][:num_decoys]
        for pos in decoy_positions.tolist():
            decoy_dir = F.normalize(e + 0.5 * torch.randn(D, generator=g), dim=0)
            k_full[0, 0, pos] = decoy_dir * (BACKGROUND_NORM * decoy_strength_ratio)

    actual_k_block = k_full[0, 0, :BLOCK_SPAN]
    actual_scores = sm_scale * (actual_k_block @ query)
    reservoir_tau = torch.quantile(actual_scores, QUANTILE).item()

    def outcome(tau):
        survivors = actual_scores >= tau
        needle_included = bool(survivors[needle_pos].item())
        masked = actual_scores.masked_fill(~survivors, -float("inf"))
        row_max = torch.clamp(masked.max(), min=-1e30)
        exp_s = torch.nan_to_num(torch.exp(masked - row_max), nan=0.0)
        weights = exp_s / (exp_s.sum() + 1e-6)
        z = weights @ v_full[0, 0, :BLOCK_SPAN]
        cos = F.cosine_similarity(z, val, dim=0).item()
        return needle_included, cos

    oracle_included, oracle_cos = outcome(oracle_tau)
    reservoir_included, reservoir_cos = outcome(reservoir_tau)
    return oracle_included, oracle_cos, reservoir_included, reservoir_cos, oracle_tau, reservoir_tau, needle_score


print("=== Tau-inflation attack: many MODERATE decoys, does reservoir tau drift")
print("    above the needle's honest score while oracle tau stays fixed? ===")
print()
N_TRIALS = 100

for decoy_strength_ratio in (0.8, 1.0, 1.2, 1.5):
    print(f"-- decoy strength = {decoy_strength_ratio}x needle's own norm --")
    print(f"{'num_decoys':>10} | {'oracle excl rate':>16} {'oracle mean cos':>16} | "
          f"{'reservoir excl rate':>19} {'reservoir mean cos':>19} | {'avg tau gap (res-oracle)':>24}")
    for num_decoys in (0, 5, 10, 20, 40):
        oracle_excl, reservoir_excl = 0, 0
        oracle_coses, reservoir_coses = [], []
        tau_gaps = []
        for trial in range(N_TRIALS):
            oi, oc, ri, rc, otau, rtau, nscore = run_trial(5000 + trial, num_decoys, decoy_strength_ratio)
            if not oi:
                oracle_excl += 1
            if not ri:
                reservoir_excl += 1
            oracle_coses.append(oc)
            reservoir_coses.append(rc)
            tau_gaps.append(rtau - otau)
        print(f"{num_decoys:>10} | {oracle_excl/N_TRIALS:>15.1%} {sum(oracle_coses)/N_TRIALS:>16.4f} | "
              f"{reservoir_excl/N_TRIALS:>18.1%} {sum(reservoir_coses)/N_TRIALS:>19.4f} | "
              f"{sum(tau_gaps)/N_TRIALS:>24.4f}")
    print()

print("If reservoir exclusion rate climbs with num_decoys while oracle stays near 0,")
print("that's the relocated tau-inflation failure, quantified. If both track together,")
print("the reservoir estimator is robust in this regime.")
