"""
Re-verification of §6.5 (Per-Type Weight Gating) against the current N=1,152 dataset.

No generator script was preserved from the original session (RESEARCH_LOG.md Session 22
describes the approach in prose only: blend 4 specialized regex weight sets -- hex-opt,
bsv-opt, kw-opt, default -- by per-sample "type affinity" computed from the same regex
features). Reconstructed here using affinities derived from the hex/backslash/keywords
sub-features and three single-feature-dominant weight sets, compared against the single
DE-optimized weight set already used by the production ensemble (ensemble_full.py).
"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score

from ensemble_full import (
    feature_matrix, FEATURE_NAMES, n_obf, n_ben, y_true, tpr_minus_fpr,
)

cache = json.load(open("data/ensemble_full_cache.json"))
default_weights_dict = cache["norm_full"]["regex_sub_weights"]
default_w = np.array([default_weights_dict[p] for p in FEATURE_NAMES])

hex_idx = FEATURE_NAMES.index('hex')
bs_idx = FEATURE_NAMES.index('backslash')
bsd_idx = FEATURE_NAMES.index('backslash_density')
kw_idx = FEATURE_NAMES.index('keywords')

# Single-feature-dominant weight sets (90% on the named feature, 10% spread on the rest)
def dominant_weights(idx):
    w = np.full(len(FEATURE_NAMES), 0.1 / (len(FEATURE_NAMES) - 1))
    w[idx] = 0.9
    return w

hex_opt = dominant_weights(hex_idx)
bsv_opt = dominant_weights(bsd_idx)
kw_opt = dominant_weights(kw_idx)

# Per-sample type affinity from raw regex feature values (normalized to sum to 1 across
# the 3 "channels" + a residual "default" channel)
hex_aff = feature_matrix[:, hex_idx]
bsv_aff = feature_matrix[:, bsd_idx]
kw_aff = feature_matrix[:, kw_idx]
aff_sum = hex_aff + bsv_aff + kw_aff
default_aff = np.clip(1.0 - aff_sum, 0.0, None)
total = hex_aff + bsv_aff + kw_aff + default_aff + 1e-10
hex_aff, bsv_aff, kw_aff, default_aff = (a / total for a in (hex_aff, bsv_aff, kw_aff, default_aff))

# Blended weights per sample, blended regex score per sample
blended_raw = (
    hex_aff[:, None] * hex_opt[None, :] +
    bsv_aff[:, None] * bsv_opt[None, :] +
    kw_aff[:, None] * kw_opt[None, :] +
    default_aff[:, None] * default_w[None, :]
)
gated_score = np.sum(blended_raw * feature_matrix, axis=1)

# Baseline: single default weight set
default_score = feature_matrix @ default_w

auc_gated = roc_auc_score(y_true, gated_score)
auc_default = roc_auc_score(y_true, default_score)
print(f"Single default weight set AUC: {auc_default:.4f}")
print(f"Per-type gated (blended) AUC:  {auc_gated:.4f}")

# Per-type detection rate comparison at 99th percentile threshold, for hex/backslash_visual/misc
import re
def classify_encoding(text):
    t = text.lower()
    if 'decode this hex' in t or '\\x' in text: return 'hex'
    if 'base64' in t: return 'base64'
    if re.search(r'rot\d+', t): return 'rot'
    if text.count('\\') > 5: return 'backslash_visual'
    if re.search(r'[A-Z]\.[A-Z]\.[A-Z]', text): return 'dot_spacing'
    if '<|im_start|>' in text: return 'xml_tag'
    return 'unstructured'

obf_trigger = json.load(open("data/obf_trigger.json"))
obf_types = [classify_encoding(t) for t in obf_trigger]

ben_default = default_score[n_obf:]
ben_gated = gated_score[n_obf:]
thr_default = np.percentile(ben_default, 99)
thr_gated = np.percentile(ben_gated, 99)

print(f"\n{'Type':<20} {'N':>5} {'Default det@1%':>16} {'Gated det@1%':>14}")
for t in ['hex', 'backslash_visual', 'unstructured']:
    idx = [i for i, tt in enumerate(obf_types) if tt == t]
    if not idx:
        continue
    d_default = np.mean(default_score[idx] >= thr_default)
    d_gated = np.mean(gated_score[idx] >= thr_gated)
    print(f"{t:<20} {len(idx):>5} {100*d_default:>15.1f}% {100*d_gated:>13.1f}%")
