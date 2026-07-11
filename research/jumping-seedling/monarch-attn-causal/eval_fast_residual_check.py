"""Verify the exact-subtraction residual centroid produces numerically
identical output to the current masked-reduction implementation
(ma_meta_threshold_shared_tau.py) -- per Fable, this must be an exact
identity, not an approximation, and the one real risk is floating-point
cancellation when subtracting two similar-magnitude sums (large Bl,
small survivor fraction). Checking max-diff directly rather than
assuming the algebra survives floating point.
"""

import sys
sys.path.insert(0, "repo")
import torch
import torch.nn.functional as F

from ma_meta_threshold_shared_tau import monarch_meta_threshold_shared_tau
from ma_meta_threshold_fast_residual import monarch_meta_threshold_fast_residual

D, Dv = 16, 16
B = 16
W_blocks = 1

print("=== Numerical exactness check: fast-residual vs masked-reduction ===")
print(f"{'N':>6} | {'max abs diff':>14} | {'max rel diff':>14}")
for N in (256, 512, 1024, 2048, 4096, 8192):
    g = torch.Generator().manual_seed(123)
    q = torch.randn(1, 2, N, D, generator=g) * 0.5
    k = torch.randn(1, 2, N, D, generator=g) * 0.5
    v = torch.randn(1, 2, N, Dv, generator=g) * 0.5

    z_old = monarch_meta_threshold_shared_tau(q, k, v, B=B, W_blocks=W_blocks)
    z_new, n_gather_ops = monarch_meta_threshold_fast_residual(q, k, v, B=B, W_blocks=W_blocks)

    diff = (z_old - z_new).abs()
    max_abs = diff.max().item()
    denom = z_old.abs().clamp(min=1e-6)
    max_rel = (diff / denom).max().item()

    print(f"{N:>6} | {max_abs:>14.2e} | {max_rel:>14.2e}")

print()
print("=== Causal validity check on the fast-residual variant ===")
g = torch.Generator().manual_seed(7)
q = torch.randn(1, 2, 256, D, generator=g)
k = torch.randn(1, 2, 256, D, generator=g)
v = torch.randn(1, 2, 256, Dv, generator=g)
z, _ = monarch_meta_threshold_fast_residual(q, k, v, B=B, W_blocks=W_blocks)
k2 = k.clone(); k2[0, 0, 200] += 100.0
z2, _ = monarch_meta_threshold_fast_residual(q, k2, v, B=B, W_blocks=W_blocks)
leak = (z[0, 0, :190] - z2[0, 0, :190]).abs().max().item()
print(f"leak (positions <190 after perturbing pos 200): {leak:.8f}")
print(f"all outputs finite: {torch.isfinite(z).all().item()}")
