"""
Cross-model residual-AUC test, richer-confound version (Session 27 follow-up).

The original cross_model_residual_test.py used only chunk_count + word_count
as OLS controls and found a large model-dependent gap (e5 residual AUC=0.54,
bge-m3=0.63, nemotron=0.82). residual_delta_test_v2.py showed that adding
char_entropy + special_char_ratio as controls for e5 alone raised residual
AUC from 0.54 to 0.72 (suppressor-variable effect). This script re-runs the
cross-model comparison with the same 4-confound control set, to check whether
the model gap is a real property of the embeddings or an artifact of the
weaker 2-confound model used originally.
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
all_samples = obf_samples + ben_samples

labels = np.array([1] * len(obf_samples) + [0] * len(ben_samples))


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


chunks_list = [chunk_text(t) for t in all_samples]
nc_all = np.array([len(c) for c in chunks_list])
word_all = np.array([len(t.split()) for t in all_samples])
entropy_all = np.array([char_entropy(t) for t in all_samples])
special_all = np.array([special_char_ratio(t) for t in all_samples])

X = np.column_stack([np.ones_like(nc_all, dtype=float), nc_all, word_all, entropy_all, special_all])

print(f"{'Model':30s} {'AUC(raw)':>10s} {'AUC(resid,2cf)':>15s} {'AUC(resid,4cf)':>15s} {'R2(4cf)':>10s}")
for model_name, model_data in sent_cache["models"].items():
    d = np.array(model_data["deltas"])
    assert len(d) == len(labels)

    # 2-confound (original methodology, for comparison)
    X2 = X[:, :3]
    beta2, *_ = np.linalg.lstsq(X2, d, rcond=None)
    resid2 = d - X2 @ beta2
    auc_resid2 = roc_auc_score(labels, resid2)
    auc_resid2 = max(auc_resid2, 1 - auc_resid2)

    # 4-confound (richer methodology)
    beta4, *_ = np.linalg.lstsq(X, d, rcond=None)
    pred4 = X @ beta4
    resid4 = d - pred4
    r2_4 = 1 - np.sum(resid4**2) / np.sum((d - d.mean())**2)
    auc_resid4 = roc_auc_score(labels, resid4)
    auc_resid4 = max(auc_resid4, 1 - auc_resid4)

    auc_raw = roc_auc_score(labels, d)
    auc_raw = max(auc_raw, 1 - auc_raw)

    print(f"{model_name:30s} {auc_raw:10.4f} {auc_resid2:15.4f} {auc_resid4:15.4f} {r2_4:10.4f}")
