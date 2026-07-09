"""Version B: local-exact-window + linear-recurrent-global hybrid.

Replaces Monarch's off-diagonal (cross-block) term with a fixed-size
kernelized/linear-attention running state, updated block-by-block left
to right. The diagonal block keeps exact causal softmax attention (no
approximation needed there -- B is small, O(B^2 D) per block is cheap,
and there's no reason to Monarch-approximate something this small).

This is NOT Monarch attention anymore for the off-diagonal part -- it's
the same shape as Infini-attention / gated-linear-attention hybrids
(compressed recurrent global summary + exact local window). Documented
as a known, not novel, risk class: linear/kernelized attention is known
to underperform softmax attention on recall-heavy, copying-style tasks,
because a fixed-size state can't preserve arbitrary distant tokens with
softmax's sharp selectivity. That's the thing the needle-in-haystack
probe in eval_linear_hybrid.py is specifically designed to catch --
aggregate MSE/cosine similarity won't surface this failure mode on
random Gaussian Q/K/V, since there's nothing "discriminative" to lose.

Feature map: phi(x) = elu(x) + 1 (Katharopoulos et al., "Transformers
are RNNs" -- the standard positive-kernel choice for linear attention).

Combination history (two prior attempts, both superseded):
1. Naive: summed unnormalized numerator/denominator directly
   (`(num_diag+num_off)/(denom_diag+denom_off)`). Bug: `denom_off` was a
   raw running sum, so it grew with sequence length while `denom_diag`
   stays capped near O(1) per key (max-subtracted exp scores). Let an
   uninformative background sum swamp a correct, sharp local answer even
   at zero distance.
2. Count-normalized: divided S/Z by running key-count before reading, so
   the global branch reads as *one averaged pseudo-key* instead of a sum
   that grows with history length. Measurably better (needle-test
   same-block retrieval cosine 0.17 -> 0.57) but still wrong: exp_scores
   are bounded to (0,1] by construction, while `phi(q)*phi(k)` is an
   unbounded raw dot product with no such cap -- no constant rescaling
   fixes an apples-to-oranges saturating-vs-unbounded mismatch.

CURRENT (log-domain pseudo-key, joint softmax): give the global branch
an actual logit in the *same units* as local scores, and fold it into
ONE softmax alongside the local block's causal-masked scores, instead of
combining two separately-normalized quantities after the fact.

  - Logit: maintain a running mean key `k_bar = (sum of valid past K) /
    count` (plain D-dim average, NOT phi-space) and score it exactly like
    a local key: `global_logit = sm_scale * (q_t . k_bar)`. Masked to
    -inf when count == 0 (no history yet, e.g. block 0). This makes it
    directly comparable to local scores under the same max-subtraction.
  - Value: still read via the phi-weighted linear-kernel state,
    `v_bar = (phi(q_t) @ S) / (phi(q_t) @ Z)` -- this ratio is invariant
    to any constant rescaling of S/Z (numerator and denominator scale
    together), so no count-normalization is needed for this part; the
    calibration problem was specific to using `phi(q)@Z` as an
    unnormalized *weight*, not to using it as a normalized ratio.
  - Combine: `[local causal-masked scores | global_logit]` all go through
    ONE softmax (single max-subtraction, single normalizer). Local
    weights multiply `v_m` as usual; the single global weight multiplies
    `v_bar`.

This makes "how much to trust the global summary vs. a specific local
match" a real, unit-consistent competition instead of an ad hoc sum --
the thing the naive and count-normalized versions both got wrong. It
does NOT restore softmax's fine-grained selectivity *within* the
compressed state (v_bar is still one fixed-size summary, not a lookup
over individual past tokens) -- that's the genuine, expected linear-
attention limitation the needle test is trying to isolate, now with the
calibration confound removed.

Complexity: off-diagonal is O(M * D * Dv) total (state update + query
read, both O(1) per block) vs. the dual-rep Monarch version's
O(B * M^2 * D) -- linear in M instead of quadratic. No T-iteration
refinement here; this is a single left-to-right pass.
"""

import torch
import torch.nn.functional as F
from math import sqrt

Tensor = torch.Tensor


def phi(x: Tensor) -> Tensor:
    return F.elu(x) + 1.0


def monarch_causal_linear_hybrid(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    B: int,
    eps: float = 1e-6,
) -> Tensor:
    E, H, N, D = q.shape
    _, _, _, Dv = v.shape
    M = (N + B - 1) // B
    N_padded = M * B
    sm_scale = 1 / sqrt(D)

    pad = N_padded - N
    q = F.pad(q, (0, 0, 0, pad)).view(E, H, M, B, D)
    k = F.pad(k, (0, 0, 0, pad)).view(E, H, M, B, D)
    v = F.pad(v, (0, 0, 0, pad)).view(E, H, M, B, Dv)

    range_n = torch.arange(N_padded).view(M, B)
    valid = (range_n < N).to(q.dtype)  # (M,B), 1.0 valid / 0.0 pad

    causal_local = torch.tril(torch.ones(B, B, dtype=torch.bool, device=q.device))

    S = torch.zeros(E, H, D, Dv, device=q.device, dtype=q.dtype)
    Z = torch.zeros(E, H, D, device=q.device, dtype=q.dtype)
    K_sum = torch.zeros(E, H, D, device=q.device, dtype=q.dtype)
    count = torch.zeros(E, H, 1, 1, device=q.device, dtype=q.dtype)

    outputs = []
    for m in range(M):
        q_m, k_m, v_m = q[:, :, m], k[:, :, m], v[:, :, m]  # (E,H,B,*)
        valid_m = valid[m].view(1, 1, 1, B)  # key validity within this block

        # --- local causal scores (block m's own keys) ---
        scores = sm_scale * (q_m @ k_m.transpose(-1, -2))  # (E,H,B,B)
        local_mask = causal_local.view(1, 1, B, B) & valid_m.bool()
        scores = scores.masked_fill(~local_mask, -float("inf"))

        # --- global pseudo-key logit: mean of past raw keys, same units
        # as local scores (sm_scale * dot product), so it competes fairly
        # in one shared softmax instead of a separately-normalized sum.
        has_history = (count > 0).view(E, H, 1, 1).expand(E, H, B, 1)
        safe_count = torch.clamp(count, min=1.0)
        k_bar = K_sum / safe_count.squeeze(-1)  # (E,H,D)
        global_logit = sm_scale * (q_m @ k_bar.unsqueeze(-1))  # (E,H,B,1)
        global_logit = global_logit.masked_fill(~has_history, -float("inf"))

        combined = torch.cat([scores, global_logit], dim=-1)  # (E,H,B,B+1)
        row_max = torch.clamp(combined.max(dim=-1, keepdim=True).values, min=-1e30)
        row_max = torch.nan_to_num(row_max, neginf=0.0)
        exp_combined = torch.exp(combined - row_max)
        exp_combined = torch.nan_to_num(exp_combined, nan=0.0)
        denom = exp_combined.sum(dim=-1, keepdim=True) + eps

        local_w = exp_combined[..., :B]  # (E,H,B,B)
        global_w = exp_combined[..., B:]  # (E,H,B,1)

        num_local = local_w @ v_m  # (E,H,B,Dv)

        # --- global value: phi-weighted read of the linear-kernel state.
        # Invariant to any constant rescaling of S/Z (ratio), so no
        # count-normalization needed here -- that confound was specific to
        # using phi(q)@Z as an unnormalized weight, not as a ratio.
        phi_q = phi(q_m)  # (E,H,B,D)
        v_bar = (phi_q @ S) / (phi_q @ Z.unsqueeze(-1) + eps)  # (E,H,B,Dv)

        out_m = (num_local + global_w * v_bar) / denom
        outputs.append(out_m)

        # --- update running state with this block's (masked) keys/values ---
        valid_col = valid_m.transpose(-1, -2)  # (1,1,B,1), broadcasts over D/Dv
        phi_k = phi(k_m) * valid_col  # zero padded keys
        k_m_masked = k_m * valid_col
        v_m_masked = v_m * valid_col
        S = S + phi_k.transpose(-1, -2) @ v_m_masked  # (E,H,D,Dv)
        Z = Z + phi_k.sum(dim=-2)  # (E,H,D)
        K_sum = K_sum + k_m_masked.sum(dim=-2)  # (E,H,D)
        count = count + valid_m.sum(dim=-1, keepdim=True).transpose(-1, -2)

    z = torch.stack(outputs, dim=2).view(E, H, N_padded, Dv)
    return z[..., :N, :]
