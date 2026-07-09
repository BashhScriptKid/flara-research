"""Version B-2: multi-slot compression, the identified fix for Version
B's capacity limit (see JOURNAL.md's Version B closing read: "a single
averaged summary vector cannot simultaneously represent 'nothing
notable happened' and 'there was one specific important thing here.'
A real fix needs a summary that preserves outliers instead of averaging
them away: multiple summary slots with routing/clustering.").

Same joint-softmax combination as Version B's attempt 3 (real
dot-product logits, one shared softmax with the local diagonal, no
scale-mismatch), but now with `n_slots` independent running (K_sum,
V_sum, count) accumulators instead of one. Each new causal key gets
routed, online, to its NEAREST existing slot centroid (dot-product
similarity, argmax) -- an outlier (needle) competing against many
similar background keys for slot space is very different from an
outlier being averaged directly into ONE global mean: with enough
slots, a distinctive key can end up sharing a slot with only a handful
of similar keys instead of the entire history, so it isn't washed out
the way Version B's single mean-key was.

Causality / seeding: NO pre-seeding from a fixed set of early keys --
that would bake future-relative-to-early-queries content into the slot
state before block 0's own causal window has even been processed
(a real leak, caught and avoided during design, not by accident). Slots
start genuinely empty; a query in block 0 sees zero global content
(consistent with every other mechanism in this investigation -- there's
no history yet). Cold-start consequence: since an empty slot's
centroid is exactly zero, the FIRST slot (index 0, via argmax tie-
breaking) absorbs all early keys until its centroid moves away from
zero and later keys start routing elsewhere -- an accepted, explicit
tradeoff for causal safety, not fixed here.

Each query block's global branch reads slot state from BEFORE that
block's own keys are folded in (query in block m never sees its own
block's keys via the global branch -- that's the local diagonal
term's job, avoiding double-counting and avoiding same-block leakage).
"""

import torch
import torch.nn.functional as F
from math import sqrt

Tensor = torch.Tensor


def monarch_causal_multislot(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    B: int,
    n_slots: int,
    eps: float = 1e-6,
) -> Tensor:
    E, H, N, D = q.shape
    _, _, _, Dv = v.shape
    M = (N + B - 1) // B
    N_padded = M * B
    sm_scale = 1 / sqrt(D)

    pad = N_padded - N
    qb = F.pad(q, (0, 0, 0, pad)).view(E, H, M, B, D)
    kb = F.pad(k, (0, 0, 0, pad)).view(E, H, M, B, D)
    vb = F.pad(v, (0, 0, 0, pad)).view(E, H, M, B, Dv)

    range_n = torch.arange(N_padded).view(M, B)
    valid = range_n < N

    causal_local = torch.tril(torch.ones(B, B, dtype=torch.bool, device=q.device))

    K_sum = torch.zeros(E, H, n_slots, D, device=q.device, dtype=q.dtype)
    V_sum = torch.zeros(E, H, n_slots, Dv, device=q.device, dtype=q.dtype)
    count = torch.zeros(E, H, n_slots, 1, device=q.device, dtype=q.dtype)

    outputs = []
    for m in range(M):
        q_m, k_m, v_m = qb[:, :, m], kb[:, :, m], vb[:, :, m]
        valid_m = valid[m].view(1, 1, 1, B)

        local_scores = sm_scale * (q_m @ k_m.transpose(-1, -2))
        local_mask = causal_local.view(1, 1, B, B) & valid_m
        local_scores = local_scores.masked_fill(~local_mask, -float("inf"))

        safe_count = count.clamp(min=1.0)
        k_bar = K_sum / safe_count  # (E,H,n_slots,D)
        v_bar = V_sum / safe_count  # (E,H,n_slots,Dv)
        has_content = (count > 0).squeeze(-1).view(E, H, 1, n_slots)

        slot_logits = sm_scale * (q_m @ k_bar.transpose(-1, -2))  # (E,H,B,n_slots)
        slot_logits = slot_logits.masked_fill(~has_content.expand(E, H, B, n_slots), -float("inf"))

        combined = torch.cat([local_scores, slot_logits], dim=-1)
        row_max = torch.clamp(combined.max(dim=-1, keepdim=True).values, min=-1e30)
        row_max = torch.nan_to_num(row_max, neginf=0.0)
        exp_c = torch.nan_to_num(torch.exp(combined - row_max), nan=0.0)
        denom = exp_c.sum(dim=-1, keepdim=True) + eps

        local_w = exp_c[..., :B]
        slot_w = exp_c[..., B:]  # (E,H,B,n_slots)

        num_local = local_w @ v_m
        num_slot = slot_w @ v_bar  # (E,H,B,n_slots)@(E,H,n_slots,Dv) -> (E,H,B,Dv)
        out_m = (num_local + num_slot) / denom
        outputs.append(out_m)

        # --- route this block's valid keys to nearest slot, update state ---
        valid_col = valid_m.transpose(-1, -2)  # (1,1,B,1)
        sims = sm_scale * (k_m @ k_bar.transpose(-1, -2))  # (E,H,B,n_slots)
        assign = sims.argmax(dim=-1)  # (E,H,B)
        onehot = F.one_hot(assign, n_slots).to(k_m.dtype) * valid_col  # (E,H,B,n_slots)

        add_K = torch.einsum("ehbc,ehbd->ehcd", onehot, k_m)
        add_V = torch.einsum("ehbc,ehbd->ehcd", onehot, v_m)
        add_count = onehot.sum(dim=2).unsqueeze(-1)  # (E,H,n_slots,1)

        K_sum = K_sum + add_K
        V_sum = V_sum + add_V
        count = count + add_count

    z = torch.stack(outputs, dim=2).view(E, H, N_padded, Dv)
    return z[..., :N, :]
