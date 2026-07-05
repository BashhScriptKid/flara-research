"""
Cross-model residual-AUC test: for EVERY cached embedding model, regress its
sentence delta on chunk_count + word_count (same structural confounds as
residual_delta_test.py) and check whether the residual still discriminates
obf/benign, or collapses to chance like e5's does.
"""
import numpy as np
import json
import re
from sklearn.metrics import roc_auc_score

with open("data/sentence_chunk_cache.json") as f:
    sent_cache = json.load(f)

obf_trigger = json.load(open("data/obf_trigger.json"))
obf_benign = json.load(open("data/obf_benign.json"))
obf_samples = obf_trigger if isinstance(obf_trigger[0], str) else [s['text'] for s in obf_trigger]
ben_samples = obf_benign if isinstance(obf_benign[0], str) else [s['text'] for s in obf_benign]

labels = np.array([1] * len(obf_samples) + [0] * len(ben_samples))  # 1=obf, 0=ben


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

X = np.column_stack([np.ones_like(nc_all, dtype=float), nc_all, word_all])

print(f"N = {len(labels)}")
print(f"{'model':<35} {'raw AUC':>8} {'resid AUC':>10} {'R2(structure)':>14}")

rows = []
for model_name in sent_cache["models"]:
    d = np.array(sent_cache["models"][model_name]["deltas"])
    assert len(d) == len(labels) == len(nc_all)

    beta, *_ = np.linalg.lstsq(X, d, rcond=None)
    pred = X @ beta
    resid = d - pred
    r2 = 1 - np.sum(resid**2) / np.sum((d - d.mean())**2)

    auc_raw = roc_auc_score(labels, d)
    auc_raw_signed = max(auc_raw, 1 - auc_raw)
    auc_resid = roc_auc_score(labels, resid)
    auc_resid_signed = max(auc_resid, 1 - auc_resid)

    rows.append((model_name, auc_raw_signed, auc_resid_signed, r2))
    print(f"{model_name:<35} {auc_raw_signed:>8.4f} {auc_resid_signed:>10.4f} {r2:>14.4f}")

print()
print("Delta (raw - residual) per model:")
for name, raw, resid, r2 in rows:
    print(f"  {name:<35} {raw - resid:+.4f}")
