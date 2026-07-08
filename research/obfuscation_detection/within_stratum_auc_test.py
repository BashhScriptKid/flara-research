"""
Within-stratum AUC: non-parametric alternative to the linear-OLS residual
test (residual_delta_test.py). Group samples by exact chunk count, compute
AUC of raw delta restricted to each stratum, to check whether delta angle
discriminates obf/benign once chunk count is held fixed exactly (no
linearity assumption).
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

labels = np.array([1] * len(obf_samples) + [0] * len(ben_samples))  # obf-first, then ben

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

assert len(d_e5) == len(labels) == len(nc_all)

# Unconditional distribution by class
print("=== Chunk count distribution (obf vs ben) ===")
obf_nc_counts = Counter(nc_all[labels == 1].tolist())
ben_nc_counts = Counter(nc_all[labels == 0].tolist())
all_ccs = sorted(set(obf_nc_counts) | set(ben_nc_counts))
for cc in all_ccs:
    print(f"  chunk_count={cc:>3}: n_obf={obf_nc_counts.get(cc,0):>4}  n_ben={ben_nc_counts.get(cc,0):>4}")
print(f"  TOTAL: n_obf={len(obf_samples)}  n_ben={len(ben_samples)}")

# Within-stratum AUC for bins with >=5 of EACH class
print("\n=== Within-stratum AUC (bins with >=5 obf AND >=5 ben) ===")
MIN_N = 5
qualifying = []
for cc in all_ccs:
    n_obf = obf_nc_counts.get(cc, 0)
    n_ben = ben_nc_counts.get(cc, 0)
    if n_obf >= MIN_N and n_ben >= MIN_N:
        mask = nc_all == cc
        y = labels[mask]
        s = d_e5[mask]
        auc = roc_auc_score(y, s)
        auc_signed = max(auc, 1 - auc)
        qualifying.append((cc, n_obf, n_ben, auc, auc_signed))
        print(f"  chunk_count={cc:>3}: n_obf={n_obf:>4} n_ben={n_ben:>4}  AUC={auc:.4f} (sign-corrected {auc_signed:.4f})")

if not qualifying:
    print("  NO qualifying strata (>=5 of each class) exist.")
else:
    weighted_n = sum(n_obf + n_ben for _, n_obf, n_ben, _, _ in qualifying)
    weighted_auc = sum(auc_signed * (n_obf + n_ben) for _, n_obf, n_ben, _, auc_signed in qualifying) / weighted_n
    print(f"\n  {len(qualifying)} qualifying strata covering {weighted_n}/{len(labels)} samples ({100*weighted_n/len(labels):.1f}%)")
    print(f"  N-weighted mean sign-corrected AUC across qualifying strata: {weighted_auc:.4f}")
