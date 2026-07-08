"""
Residual-AUC test: how much of delta angle's discriminative power survives
once chunk count and word count (the established structural confounds,
r=0.68 / r=0.47) are regressed out.

Method: OLS delta ~ chunk_count + word_count (fit on the full sample, since
this is a diagnostic of the existing production feature, not a held-out
generalization claim), then AUC of the residual against the obf/benign label.
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


obf_chunks_list = [chunk_text(t) for t in obf_samples]
ben_chunks_list = [chunk_text(t) for t in ben_samples]
nc_all = np.array([len(c) for c in obf_chunks_list] + [len(c) for c in ben_chunks_list])
word_all = np.array([len(t.split()) for t in obf_samples] + [len(t.split()) for t in ben_samples])

assert len(d_e5) == len(labels) == len(nc_all) == len(word_all)

# OLS: delta ~ 1 + chunk_count + word_count
X = np.column_stack([np.ones_like(nc_all, dtype=float), nc_all, word_all])
beta, *_ = np.linalg.lstsq(X, d_e5, rcond=None)
pred = X @ beta
resid = d_e5 - pred

r2 = 1 - np.sum(resid**2) / np.sum((d_e5 - d_e5.mean())**2)

auc_raw = roc_auc_score(labels, d_e5)
auc_resid = roc_auc_score(labels, resid)
auc_resid_signed = max(auc_resid, 1 - auc_resid)

print(f"N = {len(d_e5)}")
print(f"OLS delta ~ chunk_count + word_count: R^2 = {r2:.4f}")
print(f"  coefficients: intercept={beta[0]:.5f}, chunk_count={beta[1]:.5f}, word_count={beta[2]:.6f}")
print(f"AUC(raw delta)      = {auc_raw:.4f}")
print(f"AUC(residual delta) = {auc_resid:.4f}  (sign-corrected: {auc_resid_signed:.4f})")
print(f"Delta in AUC from removing structure: {auc_raw - auc_resid_signed:+.4f}")
