"""
Real, reproducible baseline AUC computation against the current N=1,100 dataset.

Replaces hardcoded literals in flara_paper_template.py (Table 3 / abstract / conclusion)
that could not be traced to any script. Uses standard sklearn ROC-AUC throughout, with
automatic sign-flip for baselines that may be naturally inverted on the current dataset
(if roc_auc_score < 0.5, the feature points the wrong way -- flip it and report the
natural-orientation AUC, consistent with ensemble_full.py's threshold methodology).
"""
import json
import re
import math
import numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score

obf_samples = json.load(open("data/obf_trigger.json"))
ben_samples = json.load(open("data/obf_benign.json"))
n_obf, n_ben = len(obf_samples), len(ben_samples)
all_samples = obf_samples + ben_samples
y_true = np.array([1] * n_obf + [0] * n_ben)

cache = json.load(open("data/sentence_chunk_cache.json"))
d_sent_e5 = np.array(cache["models"]["nvidia/nv-embedqa-e5-v5"]["deltas"])
para_deltas = np.load("data/para_deltas_full.npy")


def char_entropy(text):
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def special_char_ratio(text):
    if not text:
        return 0.0
    special = sum(1 for c in text if not c.isalpha() and not c.isspace())
    return special / len(text)


def regex_hex_score(text):
    hex_matches = re.findall(r'\\x[0-9a-fA-F]{2}', text)
    return min(sum(len(m) for m in hex_matches) / max(len(text), 1), 1.0)


from ensemble_full import compute_regex_features, FEATURE_NAMES, feature_matrix

ent_all = np.array([char_entropy(t) for t in all_samples])
scr_all = np.array([special_char_ratio(t) for t in all_samples])
hex_all = np.array([regex_hex_score(t) for t in all_samples])
kw_idx = FEATURE_NAMES.index('keywords')
kw_all = feature_matrix[:, kw_idx]

full_idx = list(range(len(FEATURE_NAMES)))
full_w = json.load(open("data/ensemble_full_cache.json"))["norm_full"]["regex_sub_weights"]
full_regex_all = feature_matrix @ np.array([full_w[n] for n in FEATURE_NAMES])


def auc_with_sign(scores, label):
    raw_auc = roc_auc_score(y_true, scores)
    if raw_auc < 0.5:
        return 1 - raw_auc, True
    return raw_auc, False


def det_at_fpr(scores, fpr_target):
    ben = scores[n_obf:]
    obf = scores[:n_obf]
    thr = np.percentile(ben, 100 * (1 - fpr_target))
    return float(np.mean(obf > thr))


methods = {
    "Character entropy": ent_all,
    "Special char ratio": scr_all,
    "Regex (hex only)": hex_all,
    "Regex (keywords only)": kw_all,
    "Regex (9 patterns, weighted)": full_regex_all,
    "Sentence delta only": d_sent_e5,
    "Paragraph delta only": para_deltas,
}

results = {}
print(f"{'Method':<32s} {'AUC':>7s} {'Flipped':>8s} {'Det@1%':>8s} {'Det@5%':>8s}")
for name, scores in methods.items():
    auc, flipped = auc_with_sign(scores, name)
    s = (-scores if flipped else scores)
    d1 = det_at_fpr(s, 0.01)
    d5 = det_at_fpr(s, 0.05)
    results[name] = {"auc": float(auc), "flipped": flipped, "det1": d1, "det5": d5}
    print(f"{name:<32s} {auc:>7.3f} {str(flipped):>8s} {d1*100:>7.1f}% {d5*100:>7.1f}%")

json.dump(results, open("data/baseline_auc_cache.json", "w"), indent=2)
print("\nSaved data/baseline_auc_cache.json")
