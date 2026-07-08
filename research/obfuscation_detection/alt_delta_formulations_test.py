"""
Test alternative aggregations of the per-chunk-pair delta sequence
(max, variance, positional) against the existing mean-delta baseline,
both raw AUC and OLS-residual AUC after controlling for chunk_count + word_count.
"""
import numpy as np
import json
import re
from sklearn.metrics import roc_auc_score

data = json.load(open("data/per_chunk_deltas_e5.json"))
per_sample = data["per_sample_deltas"]
labels = np.array([1 if l == "obfuscation" else 0 for l in data["labels"]])

obf_trigger = json.load(open("data/obf_trigger.json"))
obf_benign = json.load(open("data/obf_benign.json"))
obf_samples = obf_trigger if isinstance(obf_trigger[0], str) else [s['text'] for s in obf_trigger]
ben_samples = obf_benign if isinstance(obf_benign[0], str) else [s['text'] for s in obf_benign]
texts = obf_samples + ben_samples
assert len(texts) == len(labels)


def chunk_text(text, merge_threshold=8):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    for s in sentences:
        if not s.strip(): continue
        if len(s) > 50:
            for sub in re.split(r'(?<=[,;])\s+', s):
                if sub.strip(): chunks.append(sub.strip())
        else:
            chunks.append(s.strip())
    if len(chunks) < 2 and len(text.split()) > 10:
        words = text.split()
        mid = len(words) // 2
        chunks = [' '.join(words[:mid]), ' '.join(words[mid:])]
    if not chunks: chunks = [text]
    merged = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        if len(current.split()) < merge_threshold:
            if i + 1 < len(chunks):
                chunks[i + 1] = current + ' ' + chunks[i + 1]
            elif merged:
                merged[-1] = merged[-1] + ' ' + current
            else:
                merged.append(current)
        else:
            merged.append(current)
        i += 1
    return merged if merged else [text]


nc_all = np.array([len(chunk_text(t)) for t in texts])
word_all = np.array([len(t.split()) for t in texts])


def residual_auc(feature, label, nc, wc, name):
    X = np.column_stack([np.ones_like(nc, dtype=float), nc, wc])
    beta, *_ = np.linalg.lstsq(X, feature, rcond=None)
    resid = feature - X @ beta
    r2 = 1 - np.sum(resid**2) / np.sum((feature - feature.mean())**2)
    auc_raw = roc_auc_score(label, feature)
    auc_raw_signed = max(auc_raw, 1 - auc_raw)
    auc_resid = roc_auc_score(label, resid)
    auc_resid_signed = max(auc_resid, 1 - auc_resid)
    print(f"{name}: n={len(feature)} R2={r2:.4f} AUC_raw={auc_raw_signed:.4f} AUC_resid={auc_resid_signed:.4f} (delta={auc_raw_signed-auc_resid_signed:+.4f})")
    return auc_raw_signed, auc_resid_signed


# 1. MAX delta
max_delta = np.array([max(d) if d else 0.0 for d in per_sample])
residual_auc(max_delta, labels, nc_all, word_all, "MAX delta (n=full)")

# 2. VARIANCE / std of deltas (only well-defined for >=2 deltas, i.e. >=3 chunks)
mask_var = np.array([len(d) >= 2 for d in per_sample])
var_delta_full = np.array([np.var(d) if len(d) >= 2 else 0.0 for d in per_sample])
print(f"\nVariance feature well-defined for n={mask_var.sum()}/{len(mask_var)} samples (>=3 chunks); using 0.0 fill for the rest, matching MAX/MEAN convention.")
residual_auc(var_delta_full, labels, nc_all, word_all, "VAR delta (0-filled, n=full)")
# Also restricted to the well-defined subset only, for honesty
residual_auc(var_delta_full[mask_var], labels[mask_var], nc_all[mask_var], word_all[mask_var], "VAR delta (subset n>=3 chunks ONLY)")

# 3. Positional: delta[0] vs delta[1], restricted to samples with >=3 chunks (>=2 deltas)
mask_pos = np.array([len(d) >= 2 for d in per_sample])
d0 = np.array([d[0] if len(d) >= 2 else np.nan for d in per_sample])
d1 = np.array([d[1] if len(d) >= 2 else np.nan for d in per_sample])
print(f"\nPositional subset: n={mask_pos.sum()}/{len(mask_pos)} samples have >=3 chunks (delta[0] and delta[1] both defined).")
residual_auc(d0[mask_pos], labels[mask_pos], nc_all[mask_pos], word_all[mask_pos], "delta[0] (first transition), subset")
residual_auc(d1[mask_pos], labels[mask_pos], nc_all[mask_pos], word_all[mask_pos], "delta[1] (second transition), subset")

# Baseline reference (mean delta, full sample) for comparison -- recompute here for consistency
mean_delta = np.array([np.mean(d) if d else 0.0 for d in per_sample])
print()
residual_auc(mean_delta, labels, nc_all, word_all, "MEAN delta (baseline, recomputed here)")
