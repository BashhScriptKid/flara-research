"""
Full four-feature ensemble detector: sentence delta + paragraph delta +
regex (9-pattern, weighted) + weird-unicode single-chunk bonus.

Reconstructs the ensemble described in RESEARCH_LOG.md Sessions 23-26,
which was originally run interactively and never saved as a script.
Caches per-sample scores to data/ensemble_full_cache.json for reuse by
graphs/final/obf_analysis.py.
"""
import numpy as np
import json
import re
import os
from collections import Counter
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import differential_evolution

os.chdir(os.path.dirname(os.path.abspath(__file__)))

obf_samples = json.load(open("data/obf_trigger.json"))
ben_samples = json.load(open("data/obf_benign.json"))
n_obf, n_ben = len(obf_samples), len(ben_samples)
all_samples = obf_samples + ben_samples
y_true = np.array([1] * n_obf + [0] * n_ben)

cache = json.load(open("data/sentence_chunk_cache.json"))
d_sent_e5 = np.array(cache["models"]["nvidia/nv-embedqa-e5-v5"]["deltas"])
para_deltas = np.load("data/para_deltas_full.npy")

# ── Regex sub-features (Session 26 table: 9 patterns, re-optimized weights) ─
KEYWORDS = re.compile(
    r'decode|translate|execute|convert|interpret|print|output|follow|'
    r'compromise|breach|exfiltrate|acknowledge|pwned|status|confirm|'
    r'generate|respond|payload|encrypted|decipher|decrypt|hidden|reveal|'
    r'template|concatenate|expand|acronym|paraphrase|steganograph|'
    r'and (?:translate|decode|output|return|follow|respond|clean|provide|generate)|'
    r"but (?:if|when|unless|always|never)|don't (?:fix|change|modify|translate|alter|edit|correct)|"
    r'only (?:translate|output|return|respond|provide|give)|'
    r'then (?:translate|decode|output|return|follow|respond)|'
    r'now (?:translate|decode|output|return|follow|respond|reverse)',
    re.IGNORECASE
)

FEATURE_NAMES = ['hex', 'backslash', 'keywords', 'repetition', 'special_chars',
                 'backslash_density', 'digit_escape', 'consecutive_backslash', 'no_alpha']
PRUNED_FEATURES = ['hex', 'backslash', 'keywords', 'repetition', 'backslash_density', 'no_alpha']


def compute_regex_features(text):
    scores = {}
    t_len = max(len(text), 1)
    hex_matches = re.findall(r'\\x[0-9a-fA-F]{2}', text)
    scores['hex'] = min(sum(len(m) for m in hex_matches) / t_len, 1.0)
    bs_matches = re.findall(r'\\[^x0-9\n\r]', text)
    scores['backslash'] = min(sum(len(m) for m in bs_matches) / t_len, 1.0)
    kw_matches = KEYWORDS.finditer(text)
    scores['keywords'] = min(sum(len(m.group()) for m in kw_matches) / t_len, 1.0)
    lines = text.split('\n')
    scores['repetition'] = max(Counter(lines).values()) / max(len(lines), 1) if len(lines) > 1 else 0.0
    alnum = sum(c.isalnum() for c in text)
    scores['special_chars'] = 1.0 - (alnum / t_len)
    scores['backslash_density'] = min(text.count('\\') / t_len, 1.0)
    digit_esc = re.findall(r'\\[0-9]{1,3}', text)
    scores['digit_escape'] = min(sum(len(m) for m in digit_esc) / t_len, 1.0)
    consec = re.findall(r'(\\\\){3,}', text)
    scores['consecutive_backslash'] = min(sum(len(m) for m in consec) / t_len, 1.0)
    alpha_chars = sum(c.isalpha() for c in text)
    scores['no_alpha'] = 1.0 - (alpha_chars / t_len)
    return scores


feature_matrix = np.array([
    [f[p] for p in FEATURE_NAMES] for f in (compute_regex_features(t) for t in all_samples)
])

# ── Weird unicode (single-chunk only bonus, Session 25) ─────────────────
WEIRD_UNICODE_RE = re.compile(
    r'[\U0001D400-\U0001D7FF！-～①-⓿²-₟'
    r'​-⁤﻿̀-ͯ←-⇿⬀-⯿]'
)
is_single_chunk = d_sent_e5 == 0
weird_raw = np.array([1.0 if WEIRD_UNICODE_RE.search(t) else 0.0 for t in all_samples])
weird_raw = np.where(is_single_chunk, weird_raw, 0.0)

sent_norm = (d_sent_e5 - d_sent_e5.min()) / (d_sent_e5.max() - d_sent_e5.min() + 1e-10)
para_norm = (para_deltas - para_deltas.min()) / (para_deltas.max() - para_deltas.min() + 1e-10)


def tpr_minus_fpr(scores):
    ben = scores[n_obf:]
    thr = np.percentile(ben, 99)
    preds = (scores >= thr).astype(int)
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))
    tn = np.sum((preds == 0) & (y_true == 0))
    fpr = fp / (fp + tn) if (fp + tn) else 0
    tpr = tp / (tp + fn) if (tp + fn) else 0
    return tpr - fpr


def optimize_regex_weights(feature_names):
    idx = [FEATURE_NAMES.index(p) for p in feature_names]

    def objective(w):
        w = np.abs(w)
        w = w / (w.sum() + 1e-10)
        raw = feature_matrix[:, idx] @ w
        return -tpr_minus_fpr(raw)

    result = differential_evolution(objective, [(0.0, 1.0)] * len(idx),
                                     maxiter=200, seed=42, tol=1e-6, polish=True)
    w = np.abs(result.x)
    w = w / (w.sum() + 1e-10)
    return dict(zip(feature_names, w))


def optimize_ensemble_weights(regex_norm):
    components = np.stack([sent_norm, regex_norm, para_norm, weird_raw], axis=1)

    def objective(w):
        w = np.abs(w)
        w = w / (w.sum() + 1e-10)
        return -tpr_minus_fpr(components @ w)

    result = differential_evolution(objective, [(0.0, 1.0)] * 4,
                                     maxiter=200, seed=42, tol=1e-6, polish=True)
    w = np.abs(result.x)
    w = w / (w.sum() + 1e-10)
    return w  # [w_sent, w_regex, w_para, w_weird]


def build_ensemble(regex_weights):
    idx = [FEATURE_NAMES.index(p) for p in regex_weights]
    w = np.array([regex_weights[p] for p in regex_weights])
    regex_raw = feature_matrix[:, idx] @ w
    regex_raw_min, regex_raw_max = regex_raw.min(), regex_raw.max()
    regex_norm = (regex_raw - regex_raw_min) / (regex_raw_max - regex_raw_min + 1e-10)
    w_sent, w_regex, w_para, w_weird = optimize_ensemble_weights(regex_norm)
    ensemble = w_sent * sent_norm + w_regex * regex_norm + w_para * para_norm + w_weird * weird_raw
    print(f"  ensemble weights: sent={w_sent:.4f} regex={w_regex:.4f} "
          f"para={w_para:.4f} weird={w_weird:.4f}")
    norm_params = {
        "regex_feature_names": list(regex_weights.keys()),
        "regex_sub_weights": {k: float(v) for k, v in regex_weights.items()},
        "regex_raw_min": float(regex_raw_min), "regex_raw_max": float(regex_raw_max),
        "sent_min": float(d_sent_e5.min()), "sent_max": float(d_sent_e5.max()),
        "para_min": float(para_deltas.min()), "para_max": float(para_deltas.max()),
        "w_sent": float(w_sent), "w_regex": float(w_regex),
        "w_para": float(w_para), "w_weird": float(w_weird),
    }
    return ensemble, regex_raw, norm_params


def evaluate(ensemble, label):
    auc = roc_auc_score(y_true, ensemble)
    ben = ensemble[n_obf:]
    obf = ensemble[:n_obf]
    thr = np.percentile(ben, 99)
    preds = (ensemble >= thr).astype(int)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fpr_arr, tpr_arr, _ = roc_curve(y_true, ensemble)
    det1 = tpr_arr[np.searchsorted(fpr_arr, 0.01)]
    det5 = tpr_arr[np.searchsorted(fpr_arr, 0.05)]
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    print(f"{label:20s} AUC={auc:.4f} Det@1%={det1:.3f} Det@5%={det5:.3f} F1={f1:.3f} "
          f"TP={tp} FP={fp} FN={fn} TN={tn}")
    return dict(auc=auc, det1=det1, det5=det5, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn, threshold=thr)


if __name__ == "__main__":
    print("Optimizing regex sub-weights (full, 9 features)...")
    regex_weights_full = optimize_regex_weights(FEATURE_NAMES)
    for p, w in regex_weights_full.items():
        print(f"  {p:25s}: {w:.4f}")

    print("\nOptimizing regex sub-weights (pruned, 6 features)...")
    regex_weights_pruned = optimize_regex_weights(PRUNED_FEATURES)
    for p, w in regex_weights_pruned.items():
        print(f"  {p:25s}: {w:.4f}")

    print("\nBuilding full ensemble...")
    ens_full, regex_raw_full, norm_full = build_ensemble(regex_weights_full)
    stats_full = evaluate(ens_full, "Full (9 regex)")

    print("\nBuilding pruned ensemble...")
    ens_pruned, regex_raw_pruned, norm_pruned = build_ensemble(regex_weights_pruned)
    stats_pruned = evaluate(ens_pruned, "Pruned (6 regex)")

    print("\nLogged reference (Sessions 25-26, prior dataset version):")
    print("  Full AUC=0.9872 Det@1%=0.972 F1=0.971 TP=280 FP=10 FN=7")
    print("  Pruned AUC=0.9855 Det@1%=0.972 F1=0.972 TP=279 FP=8 FN=8")

    out = {
        "n_obf": n_obf, "n_ben": n_ben,
        "ensemble_full": ens_full.tolist(),
        "ensemble_pruned": ens_pruned.tolist(),
        "stats_full": stats_full,
        "stats_pruned": stats_pruned,
        "norm_full": norm_full,
        "norm_pruned": norm_pruned,
    }
    with open("data/ensemble_full_cache.json", "w") as f:
        json.dump(out, f)
    print("\nSaved data/ensemble_full_cache.json")
