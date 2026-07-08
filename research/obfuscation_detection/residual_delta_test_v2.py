"""
Residual-AUC test v2: extends residual_delta_test.py's structural-confound
control set (chunk_count + word_count) with two more cheap surface features
(char entropy, special_char_ratio) to see whether delta still collapses to
near-chance once a richer set of structural/surface confounds is removed.
"""
import numpy as np
import json
import re
from collections import Counter
from sklearn.metrics import roc_auc_score

with open("data/sentence_chunk_cache.json") as f:
    sent_cache = json.load(f)

obf_trigger = json.load(open("data/obf_trigger.json"))
obf_benign = json.load(open("data/obf_benign.json"))
obf_samples = obf_trigger if isinstance(obf_trigger[0], str) else [s['text'] for s in obf_trigger]
ben_samples = obf_benign if isinstance(obf_benign[0], str) else [s['text'] for s in obf_benign]

labels = np.array([1] * len(obf_samples) + [0] * len(ben_samples))  # 1=obf, 0=ben

e5_sent = sent_cache["models"]["nvidia/nv-embedqa-e5-v5"]
d_e5 = np.array(e5_sent["deltas"])


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


def char_entropy(text):
    freq = Counter(text)
    n = max(len(text), 1)
    return -sum((cnt / n) * np.log2(cnt / n) for cnt in freq.values() if cnt > 0)


def special_char_ratio(text):
    if not text:
        return 0.0
    special = sum(1 for c in text if not c.isalpha() and not c.isspace())
    return special / len(text)


obf_chunks_list = [chunk_text(t) for t in obf_samples]
ben_chunks_list = [chunk_text(t) for t in ben_samples]
nc_all = np.array([len(c) for c in obf_chunks_list] + [len(c) for c in ben_chunks_list])
word_all = np.array([len(t.split()) for t in obf_samples] + [len(t.split()) for t in ben_samples])
entropy_all = np.array([char_entropy(t) for t in obf_samples] + [char_entropy(t) for t in ben_samples])
special_all = np.array([special_char_ratio(t) for t in obf_samples] + [special_char_ratio(t) for t in ben_samples])

assert len(d_e5) == len(labels) == len(nc_all) == len(word_all) == len(entropy_all) == len(special_all)

# --- 2-predictor baseline (chunk_count + word_count), for direct comparison ---
X2 = np.column_stack([np.ones_like(nc_all, dtype=float), nc_all, word_all])
beta2, *_ = np.linalg.lstsq(X2, d_e5, rcond=None)
resid2 = d_e5 - X2 @ beta2
r2_2 = 1 - np.sum(resid2**2) / np.sum((d_e5 - d_e5.mean())**2)
auc_resid2 = roc_auc_score(labels, resid2)
auc_resid2_signed = max(auc_resid2, 1 - auc_resid2)

# --- 4-predictor (chunk_count + word_count + entropy + special_char_ratio) ---
X4 = np.column_stack([np.ones_like(nc_all, dtype=float), nc_all, word_all, entropy_all, special_all])
beta4, *_ = np.linalg.lstsq(X4, d_e5, rcond=None)
pred4 = X4 @ beta4
resid4 = d_e5 - pred4
r2_4 = 1 - np.sum(resid4**2) / np.sum((d_e5 - d_e5.mean())**2)

auc_raw = roc_auc_score(labels, d_e5)
auc_resid4 = roc_auc_score(labels, resid4)
auc_resid4_signed = max(auc_resid4, 1 - auc_resid4)

print(f"N = {len(d_e5)}")
print()
print("--- Baseline: delta ~ chunk_count + word_count (2 predictors) ---")
print(f"R^2 = {r2_2:.4f}, AUC(residual, sign-corrected) = {auc_resid2_signed:.4f}")
print()
print("--- Extended: delta ~ chunk_count + word_count + entropy + special_char_ratio (4 predictors) ---")
print(f"R^2 = {r2_4:.4f}")
print(f"  coefficients: intercept={beta4[0]:.5f}, chunk_count={beta4[1]:.5f}, "
      f"word_count={beta4[2]:.6f}, entropy={beta4[3]:.5f}, special_char_ratio={beta4[4]:.5f}")
print(f"AUC(raw delta)       = {auc_raw:.4f}")
print(f"AUC(residual delta)  = {auc_resid4:.4f}  (sign-corrected: {auc_resid4_signed:.4f})")
print(f"Delta in AUC from removing structure+surface: {auc_raw - auc_resid4_signed:+.4f}")
print()
print(f"Comparison: residual AUC went from {auc_resid2_signed:.4f} (2-predictor) "
      f"to {auc_resid4_signed:.4f} (4-predictor); R^2 went from {r2_2:.4f} to {r2_4:.4f}.")
