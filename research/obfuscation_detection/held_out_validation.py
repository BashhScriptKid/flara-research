"""
Held-out validation of the ACTUAL headline 4-feature ensemble (sentence delta +
paragraph delta + 9-pattern regex + weird-unicode bonus), matching ensemble_full.py.

Previous version of this script evaluated a different, older 3-feature model
(no weird-unicode, hardcoded Session-23 regex weights) -- it was not a held-out
test of the model described elsewhere in the paper. This version: stratified
70/30 split, regex sub-weights AND top-level ensemble weights optimized via
differential evolution on the TRAIN split only, threshold fit on train only,
then a cold (no further fitting) evaluation on the test split.
"""
import numpy as np
import json
import re
import os
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import differential_evolution

np.random.seed(42)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ensemble_full import (
    compute_regex_features, WEIRD_UNICODE_RE, FEATURE_NAMES,
    feature_matrix, weird_raw, d_sent_e5, para_deltas,
    obf_samples, ben_samples, n_obf, n_ben, y_true,
)

# ── Stratification by obfuscation type (same classify rules as obf_analysis.py) ─
def classify_encoding(text):
    t = text.lower()
    if 'decode this hex' in t or '\\x' in text: return 'hex'
    if 'base64' in t: return 'base64'
    if re.search(r'rot\d+', t): return 'rot'
    if text.count('\\') > 5: return 'backslash_visual'
    if re.search(r'[A-Z]\.[A-Z]\.[A-Z]', text): return 'dot_spacing'
    if '<|im_start|>' in text: return 'xml_tag'
    return 'unstructured'

def classify_unstructured(text):
    t = text.lower()
    if re.search(r'translat.*spanish.*english|translat.*to.*spanish|translat.*back', t): return 'translation_chain'
    if re.search(r'(he sido|hackeado|llave secreta|clave)', t): return 'translation_chain'
    if re.search(r'dictionary game|replace.*with|synonym|substitut', t): return 'substitution_game'
    if re.search(r'q to p|replace each letter', t): return 'cipher_substitution'
    if re.search(r'movie title|generate a title|form a title', t): return 'task_hijack'
    if re.search(r'f\(.*==.*\)|decode.*base', t): return 'encoding_puzzle'
    if re.search(r'previous letter|shift|back shift', t): return 'manual_cipher'
    if re.search(r'U\+[0-9A-Fa-f]{4}', text): return 'unicode_codepoint'
    if any(c in text for c in '🗣👍🎉🎯🔥💡⭐🚀✅❌💰🎭🎪🎨🎭🎵🎶🎤🎧🎸🎹🎺🎻🥁'): return 'emoji_encoding'
    if re.search(r'p[ ,]*w[ ,]*n[ ,]*e[ ,]*d', t): return 'pwned_spacing'
    if 'ihbpd' in t or 'ihb' in t or 'ihnbp' in t or 'ihnpe' in t: return 'pwned_abbreviation'
    if re.search(r'pwned means|pwned.*positive|pwned.*spanish|pwned stands for', t): return 'semantic_redef'
    if 'hola' in t or 'si' in t or 'me gusta' in t: return 'language_mixing'
    if re.search(r'reverse.*letter|reverse.*each|reverse.*sentence', t): return 'reverse_scramble'
    if re.search(r'(don.t tell|only respond|only output|do not print|do not use)', t): return 'constraint_inject'
    if re.search(r'gnitae|selppa|Jdfjbkhkg|P ohcl', t): return 'manual_cipher'
    if re.search(r'secret key|secret code|password|\$secret|\{secret', t): return 'extraction_trick'
    if re.search(r'script:|noop|print the sentence', t): return 'code_injection'
    return 'misc'

obf_types = []
for t in obf_samples:
    enc = classify_encoding(t)
    obf_types.append(classify_unstructured(t) if enc == 'unstructured' else enc)

all_strata = obf_types + ['benign'] * n_ben
stratum_indices = defaultdict(list)
for i, s in enumerate(all_strata):
    stratum_indices[s].append(i)

train_idx, test_idx = [], []
for stratum, indices in stratum_indices.items():
    n_train = int(len(indices) * 0.7)
    perm = np.random.permutation(indices)
    train_idx.extend(perm[:n_train])
    test_idx.extend(perm[n_train:])
train_idx = np.array(sorted(train_idx))
test_idx = np.array(sorted(test_idx))
print(f"Train: {len(train_idx)}  Test: {len(test_idx)}")

y_train, y_test = y_true[train_idx], y_true[test_idx]


def tpr_minus_fpr_on(idx_subset, scores, labels):
    ben_mask = labels == 0
    thr = np.percentile(scores[ben_mask], 99)
    preds = (scores >= thr).astype(int)
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fpr = fp / (fp + tn) if (fp + tn) else 0
    tpr = tp / (tp + fn) if (tp + fn) else 0
    return tpr - fpr


# ── Step 1: optimize regex sub-weights on TRAIN ONLY ─────────────────────
print("Optimizing regex sub-weights on train split...")
train_feat = feature_matrix[train_idx]

def regex_objective(w):
    w = np.abs(w); w = w / (w.sum() + 1e-10)
    raw = train_feat @ w
    return -tpr_minus_fpr_on(train_idx, raw, y_train)

result = differential_evolution(regex_objective, [(0.0, 1.0)] * len(FEATURE_NAMES),
                                 maxiter=200, seed=42, tol=1e-6, polish=True)
regex_w = np.abs(result.x); regex_w = regex_w / (regex_w.sum() + 1e-10)
print("Train-optimized regex weights:")
for p, w in zip(FEATURE_NAMES, regex_w):
    print(f"  {p:25s}: {w:.4f}")

# Regex raw score + TRAIN-ONLY min/max normalization
regex_raw_all = feature_matrix @ regex_w
train_regex_raw = regex_raw_all[train_idx]
rx_min, rx_max = train_regex_raw.min(), train_regex_raw.max()
regex_norm_all = np.clip((regex_raw_all - rx_min) / (rx_max - rx_min + 1e-10), 0.0, 1.0)

# TRAIN-ONLY min/max normalization for sentence/paragraph delta
sent_min, sent_max = d_sent_e5[train_idx].min(), d_sent_e5[train_idx].max()
para_min, para_max = para_deltas[train_idx].min(), para_deltas[train_idx].max()
sent_norm_all = np.clip((d_sent_e5 - sent_min) / (sent_max - sent_min + 1e-10), 0.0, 1.0)
para_norm_all = np.clip((para_deltas - para_min) / (para_max - para_min + 1e-10), 0.0, 1.0)

# ── Step 2: optimize top-level ensemble weights on TRAIN ONLY ────────────
print("\nOptimizing top-level ensemble weights on train split...")
components_train = np.stack([sent_norm_all, regex_norm_all, para_norm_all, weird_raw], axis=1)[train_idx]

def ensemble_objective(w):
    w = np.abs(w); w = w / (w.sum() + 1e-10)
    return -tpr_minus_fpr_on(train_idx, components_train @ w, y_train)

result2 = differential_evolution(ensemble_objective, [(0.0, 1.0)] * 4,
                                  maxiter=200, seed=42, tol=1e-6, polish=True)
w_sent, w_regex, w_para, w_weird = np.abs(result2.x) / (np.abs(result2.x).sum() + 1e-10)
print(f"  sent={w_sent:.4f} regex={w_regex:.4f} para={w_para:.4f} weird={w_weird:.4f}")

ensemble_all = (w_sent * sent_norm_all + w_regex * regex_norm_all +
                w_para * para_norm_all + w_weird * weird_raw)

# ── Step 3: fit threshold on TRAIN ONLY ──────────────────────────────────
train_ens = ensemble_all[train_idx]
threshold = np.percentile(train_ens[y_train == 0], 99)
print(f"\nTrain-fit threshold (99th pct of train benign): {threshold:.6f}")


def evaluate(idx, labels, label_name):
    scores = ensemble_all[idx]
    preds = (scores >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fpr = fp / (fp + tn) if (fp + tn) else 0
    tpr = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) else 0
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float('nan')
    if len(np.unique(labels)) > 1:
        fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
        det1 = float(tpr_arr[np.searchsorted(fpr_arr, 0.01)])
        det5 = float(tpr_arr[np.searchsorted(fpr_arr, 0.05)])
    else:
        det1 = det5 = float('nan')
    print(f"{label_name:20s} AUC={auc:.4f} F1={f1:.4f} FPR={fpr:.4f} TPR={tpr:.4f} "
          f"Det1={det1:.4f} Det5={det5:.4f} TP={tp} FP={fp} FN={fn} TN={tn}")
    return dict(auc=float(auc), f1=float(f1), fpr=float(fpr), tpr=float(tpr),
                det_1pct=det1, det_5pct=det5, tp=tp, fp=fp, fn=fn, tn=tn)


print("\n" + "=" * 70)
print("EVALUATION: train-fit weights/threshold applied cold to each split")
print("=" * 70)
train_results = evaluate(train_idx, y_train, "Train")
test_results = evaluate(test_idx, y_test, "Test (held-out)")

headline = json.load(open("data/ensemble_full_cache.json"))["stats_full"]
print(f"\nFor reference -- headline model (separately optimized on full N=1100, "
      f"NOT comparable 1:1 with this train-only model): AUC={headline['auc']:.4f} "
      f"F1={headline['f1']:.4f}")

# ── Per-type on test split ───────────────────────────────────────────────
print("\nPer-type (test split):")
test_types = np.array(obf_types)[np.array([i for i in test_idx if i < n_obf])]
test_obf_idx = np.array([i for i in test_idx if i < n_obf])
per_type_test = {}
for t in sorted(set(test_types)):
    mask = test_types == t
    n = int(mask.sum())
    if n == 0:
        continue
    scores = ensemble_all[test_obf_idx[mask]]
    det = float(np.mean(scores >= threshold))
    per_type_test[t] = {"n": n, "det": det}
    print(f"  {t:25s}  n={n:3d}  Det@threshold={det:.2%}")

results = {
    "split": {"train_n": len(train_idx), "test_n": len(test_idx), "random_state": 42},
    "regex_weights_train": {p: float(w) for p, w in zip(FEATURE_NAMES, regex_w)},
    "ensemble_weights_train": {"sent": float(w_sent), "regex": float(w_regex),
                                "para": float(w_para), "weird": float(w_weird)},
    "threshold": float(threshold),
    "train": train_results,
    "test": test_results,
    "headline_reference_full_dataset": headline,
    "per_type_test": per_type_test,
    "generalization_gap": {
        "auc": train_results["auc"] - test_results["auc"],
        "f1": train_results["f1"] - test_results["f1"],
        "det_1pct": train_results["det_1pct"] - test_results["det_1pct"],
    },
}
with open("data/held_out_validation.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to data/held_out_validation.json")
