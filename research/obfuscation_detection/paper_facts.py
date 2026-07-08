"""
Single source of truth for every number quoted in flara_paper_template.py.

Ground-up reconstruction (see /home/bashh/.claude/plans/imperative-soaring-penguin.md):
instead of copying numbers forward from prior, repeatedly-patched prose, this
script re-derives every contested/recent figure directly from the data caches,
and passes through every stable, previously-verified figure from its existing
cache file with an explicit source label. The paper's prose should quote this
script's output, never the reverse.

Run: .venv/bin/python paper_facts.py
"""
import numpy as np
import json
import re
import os
from collections import Counter
from sklearn.metrics import roc_auc_score
from scipy.optimize import differential_evolution

os.chdir(os.path.dirname(os.path.abspath(__file__)))
FACTS = {}

# ─────────────────────────────────────────────────────────────────────────
# Shared loaders / reconstructions (reused verbatim across sections)
# ─────────────────────────────────────────────────────────────────────────
obf_samples = json.load(open("data/obf_trigger.json"))
ben_samples = json.load(open("data/obf_benign.json"))
n_obf, n_ben = len(obf_samples), len(ben_samples)
all_samples = obf_samples + ben_samples
y = np.array([1] * n_obf + [0] * n_ben)

sent_cache = json.load(open("data/sentence_chunk_cache.json"))
d_sent_e5 = np.array(sent_cache["models"]["nvidia/nv-embedqa-e5-v5"]["deltas"])
para_deltas = np.load("data/para_deltas_full.npy")
norm_full = json.load(open("data/ensemble_full_cache.json"))["norm_full"]
stats_full = json.load(open("data/ensemble_full_cache.json"))["stats_full"]


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
    if not text: return 0.0
    special = sum(1 for c in text if not c.isalpha() and not c.isspace())
    return special / len(text)


chunks_list = [chunk_text(t) for t in all_samples]
nc_all = np.array([len(c) for c in chunks_list])
word_all = np.array([len(t.split()) for t in all_samples])
entropy_all = np.array([char_entropy(t) for t in all_samples])
special_all = np.array([special_char_ratio(t) for t in all_samples])

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


feature_matrix = np.array([[f[p] for p in FEATURE_NAMES] for f in (compute_regex_features(t) for t in all_samples)])
WEIRD_UNICODE_RE = re.compile(r'[\U0001D400-\U0001D7FF！-～①-⓿²-₟​-⁤﻿̀-ͯ←-⇿⬀-⯿]')
is_single_chunk = d_sent_e5 == 0
weird_raw_base = np.array([1.0 if WEIRD_UNICODE_RE.search(t) else 0.0 for t in all_samples])
weird_raw = np.where(is_single_chunk, weird_raw_base, 0.0)

idx = [FEATURE_NAMES.index(p) for p in norm_full["regex_feature_names"]]
w_regex_sub = np.array([norm_full["regex_sub_weights"][p] for p in norm_full["regex_feature_names"]])
regex_raw = feature_matrix[:, idx] @ w_regex_sub
regex_norm = (regex_raw - norm_full["regex_raw_min"]) / (norm_full["regex_raw_max"] - norm_full["regex_raw_min"] + 1e-10)
sent_norm = (d_sent_e5 - d_sent_e5.min()) / (d_sent_e5.max() - d_sent_e5.min() + 1e-10)
para_norm = (para_deltas - para_deltas.min()) / (para_deltas.max() - para_deltas.min() + 1e-10)


def tpr_minus_fpr(scores):
    ben = scores[n_obf:]
    thr = np.percentile(ben, 99)
    preds = (scores >= thr).astype(int)
    tp = np.sum((preds == 1) & (y == 1)); fp = np.sum((preds == 1) & (y == 0))
    fn = np.sum((preds == 0) & (y == 1)); tn = np.sum((preds == 0) & (y == 0))
    fpr = fp / (fp + tn) if (fp + tn) else 0
    tpr = tp / (tp + fn) if (tp + fn) else 0
    return tpr - fpr


def optimize(components, n):
    def obj(wv):
        wv = np.abs(wv); wv = wv / (wv.sum() + 1e-10)
        return -tpr_minus_fpr(components @ wv)
    r = differential_evolution(obj, [(0.0, 1.0)] * n, maxiter=300, seed=42, tol=1e-10, polish=True)
    wv = np.abs(r.x); wv = wv / (wv.sum() + 1e-10)
    return wv


def auc_signed(labels, scores):
    a = roc_auc_score(labels, scores)
    return max(a, 1 - a)


# ─────────────────────────────────────────────────────────────────────────
# §4 Dataset
# ─────────────────────────────────────────────────────────────────────────
single_chunk_obf = int((d_sent_e5[:n_obf] == 0).sum())
single_chunk_ben = int((d_sent_e5[n_obf:] == 0).sum())
FACTS["dataset"] = {
    "n_obf": n_obf, "n_ben": n_ben, "n_total": n_obf + n_ben,
    "single_chunk_rate_obf": single_chunk_obf / n_obf,
    "single_chunk_rate_ben": single_chunk_ben / n_ben,
    "source": "data/obf_trigger.json, data/obf_benign.json, data/sentence_chunk_cache.json",
}

multi_para_mask_all = para_deltas != 0
FACTS["sentence_paragraph_correlation"] = {
    "corr_full_dataset": float(np.corrcoef(d_sent_e5, para_deltas)[0, 1]),
    "corr_multi_paragraph_subset_only": float(np.corrcoef(d_sent_e5[multi_para_mask_all], para_deltas[multi_para_mask_all])[0, 1]),
    "n_multi_paragraph_subset": int(multi_para_mask_all.sum()),
    "source": "recomputed in paper_facts.py from data/sentence_chunk_cache.json + data/para_deltas_full.npy",
}

# ─────────────────────────────────────────────────────────────────────────
# §5 Headline ensemble (pass-through, unchanged, already verified repeatedly)
# ─────────────────────────────────────────────────────────────────────────
FACTS["headline_ensemble"] = {**stats_full, "weights": norm_full, "source": "data/ensemble_full_cache.json"}

# ─────────────────────────────────────────────────────────────────────────
# §5.1 Chunk-count substitution test (delta vs trivial structural feature)
# ─────────────────────────────────────────────────────────────────────────
w_sent, w_regex, w_para, w_weird = norm_full["w_sent"], norm_full["w_regex"], norm_full["w_para"], norm_full["w_weird"]
ens_baseline = w_sent * sent_norm + w_regex * regex_norm + w_para * para_norm + w_weird * weird_raw
nc_norm = (nc_all - nc_all.min()) / (nc_all.max() - nc_all.min() + 1e-10)
ens_chunkcount_swap = w_sent * nc_norm + w_regex * regex_norm + w_para * para_norm + w_weird * weird_raw
comp_no_sent = np.stack([regex_norm, para_norm, weird_raw], axis=1)
w_no_sent = optimize(comp_no_sent, 3)
ens_no_sent = comp_no_sent @ w_no_sent
FACTS["chunk_count_substitution"] = {
    "auc_baseline": roc_auc_score(y, ens_baseline),
    "auc_chunkcount_swap_same_weights": roc_auc_score(y, ens_chunkcount_swap),
    "auc_sentence_delta_removed_reoptimized": roc_auc_score(y, ens_no_sent),
    "source": "recomputed in paper_facts.py from data/ensemble_full_cache.json + data/sentence_chunk_cache.json",
}

# ─────────────────────────────────────────────────────────────────────────
# §5 Paragraph delta: standalone, subset, flip-count, removal cost
# ─────────────────────────────────────────────────────────────────────────
multi_para = para_deltas != 0
n_multi_para_obf = int(multi_para[:n_obf].sum())
n_multi_para_ben = int(multi_para[n_obf:].sum())

comp4 = np.stack([sent_norm, regex_norm, para_norm, weird_raw], axis=1)
w4 = optimize(comp4, 4)
ens4 = comp4 @ w4
thr4 = np.percentile(ens4[n_obf:], 99)
flagged4 = ens4 >= thr4

comp3_nopara = np.stack([sent_norm, regex_norm, weird_raw], axis=1)
w3_nopara = optimize(comp3_nopara, 3)
ens3_nopara = comp3_nopara @ w3_nopara
thr3 = np.percentile(ens3_nopara[n_obf:], 99)
flagged3 = ens3_nopara >= thr3

multi_para_obf_idx = np.where(multi_para[:n_obf])[0]
n_caught_with = int(flagged4[multi_para_obf_idx].sum())
n_caught_without = int(flagged3[multi_para_obf_idx].sum())
n_flipped = int((flagged4[multi_para_obf_idx] & ~flagged3[multi_para_obf_idx]).sum())

FACTS["paragraph_delta"] = {
    "standalone_auc_full_dataset": auc_signed(y, para_deltas),
    "n_multi_paragraph_obf": n_multi_para_obf, "n_multi_paragraph_ben": n_multi_para_ben,
    "standalone_auc_multi_paragraph_subset_only": (
        auc_signed(y[multi_para], para_deltas[multi_para]) if n_multi_para_ben > 0 else None
    ),
    "caveat_subset_auc": f"subset AUC computed against only {n_multi_para_ben} benign sample(s) -- illustrative, not headline evidence",
    "ensemble_weight_full_dataset": w_para,
    "ensemble_auc_with_para": roc_auc_score(y, ens4),
    "ensemble_auc_without_para": roc_auc_score(y, ens3_nopara),
    "removal_cost": roc_auc_score(y, ens4) - roc_auc_score(y, ens3_nopara),
    "multi_paragraph_obf_caught_with_para": f"{n_caught_with}/{n_multi_para_obf}",
    "multi_paragraph_obf_caught_without_para": f"{n_caught_without}/{n_multi_para_obf}",
    "multi_paragraph_obf_flipped_by_removal": n_flipped,
    "source": "recomputed in paper_facts.py from data/para_deltas_full.npy + data/ensemble_full_cache.json",
}

# ─────────────────────────────────────────────────────────────────────────
# Held-out validation weight comparison (the discrepancy the last review flagged)
# ─────────────────────────────────────────────────────────────────────────
held_out = json.load(open("data/held_out_validation.json"))
FACTS["held_out_validation"] = {
    **held_out,
    "para_weight_train_vs_full": {
        "train_70pct": held_out["ensemble_weights_train"]["para"],
        "full_dataset": norm_full["w_para"],
        "delta": held_out["ensemble_weights_train"]["para"] - norm_full["w_para"],
    },
    "source": "data/held_out_validation.json",
}

# ─────────────────────────────────────────────────────────────────────────
# Cross-model residual AUC (e5 / bge-m3 / nemotron, 2-confound and 4-confound)
# ─────────────────────────────────────────────────────────────────────────
X2 = np.column_stack([np.ones_like(nc_all, dtype=float), nc_all, word_all])
X4 = np.column_stack([np.ones_like(nc_all, dtype=float), nc_all, word_all, entropy_all, special_all])
cross_model = {}
for model_name, model_data in sent_cache["models"].items():
    d = np.array(model_data["deltas"])
    beta2, *_ = np.linalg.lstsq(X2, d, rcond=None)
    resid2 = d - X2 @ beta2
    beta4, *_ = np.linalg.lstsq(X4, d, rcond=None)
    resid4 = d - X4 @ beta4
    r2_4 = 1 - np.sum(resid4 ** 2) / np.sum((d - d.mean()) ** 2)
    cross_model[model_name] = {
        "auc_raw": auc_signed(y, d),
        "auc_residual_2confound": auc_signed(y, resid2),
        "auc_residual_4confound": auc_signed(y, resid4),
        "r2_4confound": r2_4,
    }
FACTS["cross_model_residual_auc"] = {**cross_model, "source": "recomputed in paper_facts.py from data/sentence_chunk_cache.json"}

# ─────────────────────────────────────────────────────────────────────────
# MAX vs MEAN aggregation + redundancy/ceiling check
# ─────────────────────────────────────────────────────────────────────────
per_chunk = json.load(open("data/per_chunk_deltas_e5.json"))
max_delta = np.array([max(d) if d else 0.0 for d in per_chunk["per_sample_deltas"]])
beta_max, *_ = np.linalg.lstsq(X2, max_delta, rcond=None)
resid_max = max_delta - X2 @ beta_max
max_norm = (max_delta - max_delta.min()) / (max_delta.max() - max_delta.min() + 1e-10)

comp4_max = np.stack([max_norm, regex_norm, para_norm, weird_raw], axis=1)
w4_max = optimize(comp4_max, 4)
ens4_max = comp4_max @ w4_max

regex_weird_score = 0.45 * regex_norm + 0.32 * weird_raw
thr_low = np.percentile(regex_weird_score[~(y == 1)], 99)
weakly_flagged_obf = (y == 1) & (regex_weird_score < thr_low)
hard_mask = weakly_flagged_obf | (y == 0)
n_hard_obf = int(weakly_flagged_obf.sum())
n_obf_covered_by_regex_weird = n_obf - n_hard_obf

FACTS["max_vs_mean_aggregation"] = {
    "mean_residual_auc_2confound": FACTS["cross_model_residual_auc"]["nvidia/nv-embedqa-e5-v5"]["auc_residual_2confound"],
    "max_residual_auc_2confound": auc_signed(y, resid_max),
    "ensemble_auc_mean_based": roc_auc_score(y, ens4),
    "ensemble_auc_max_based": roc_auc_score(y, ens4_max),
    "ensemble_weights_mean_based": dict(zip(["sent", "regex", "para", "weird"], w4.tolist())),
    "ensemble_weights_max_based": dict(zip(["sent_as_max", "regex", "para", "weird"], w4_max.tolist())),
    "corr_resid_max_vs_regex_norm": float(np.corrcoef(resid_max, regex_norm)[0, 1]),
    "corr_resid_max_vs_weird_raw": float(np.corrcoef(resid_max, weird_raw)[0, 1]),
    "obf_covered_by_regex_weird_alone": f"{n_obf_covered_by_regex_weird}/{n_obf}",
    "obf_covered_by_regex_weird_alone_pct": n_obf_covered_by_regex_weird / n_obf,
    "n_hard_obf_cases": n_hard_obf,
    "auc_resid_max_on_hard_cases": auc_signed(y[hard_mask], resid_max[hard_mask]) if n_hard_obf > 5 else None,
    "source": "recomputed in paper_facts.py from data/per_chunk_deltas_e5.json (redundancy_check.py methodology)",
}

# ─────────────────────────────────────────────────────────────────────────
# Single-chunk encoding ceiling (§7.2 in old draft)
# ─────────────────────────────────────────────────────────────────────────
single_chunk_obf_mask = (d_sent_e5[:n_obf] == 0)
n_single_chunk_obf = int(single_chunk_obf_mask.sum())
kw_scores_obf = feature_matrix[:n_obf, FEATURE_NAMES.index('keywords')]
kw_catch = int(((kw_scores_obf > 0) & single_chunk_obf_mask).sum())
unicode_catch = int(((weird_raw_base[:n_obf] > 0) & single_chunk_obf_mask).sum())
union_catch = int((((kw_scores_obf > 0) | (weird_raw_base[:n_obf] > 0)) & single_chunk_obf_mask).sum())
ens_with_weird = w_sent * sent_norm + w_regex * regex_norm + w_para * para_norm + w_weird * weird_raw
ens_without_weird = w_sent * sent_norm + w_regex * regex_norm + w_para * para_norm
FACTS["weird_unicode_contribution"] = {
    "single_chunk_subset_auc_with_weird_unicode": roc_auc_score(y[is_single_chunk], ens_with_weird[is_single_chunk]),
    "single_chunk_subset_auc_without_weird_unicode": roc_auc_score(y[is_single_chunk], ens_without_weird[is_single_chunk]),
    "source": "recomputed in paper_facts.py from data/ensemble_full_cache.json (production weights, fixed, no re-optimisation)",
}

FACTS["single_chunk_ceiling"] = {
    "n_single_chunk_obf": n_single_chunk_obf,
    "single_chunk_rate_obf": n_single_chunk_obf / n_obf,
    "keyword_catch": f"{kw_catch}/{n_single_chunk_obf}",
    "unicode_catch": f"{int(unicode_catch)}/{n_single_chunk_obf}",
    "union_catch": f"{union_catch}/{n_single_chunk_obf}",
    "union_catch_pct": union_catch / n_single_chunk_obf,
    "source": "recomputed in paper_facts.py from data/obf_trigger.json + regex/weird-unicode logic",
}

# ─────────────────────────────────────────────────────────────────────────
# Stable, previously-verified ablation results -- passed through, not recomputed
# ─────────────────────────────────────────────────────────────────────────
FACTS["ablations_passthrough"] = {
    "distribution_distance": {**json.load(open("data/distribution_distance_results.json")), "source": "data/distribution_distance_results.json"},
    "length_scaled_softmax": {**json.load(open("data/length_scaled_results.json")), "source": "data/length_scaled_results.json"},
    "paragraph_combination_early": {**json.load(open("data/paragraph_results.json")), "source": "data/paragraph_results.json"},
    "neuralchemy_crossdataset": {**json.load(open("data/neuralchemy_results.json")), "source": "data/neuralchemy_results.json"},
    "baseline_comparison": {**json.load(open("data/baseline_auc_cache.json")), "source": "data/baseline_auc_cache.json"},
    "adaptive_attack_eval": {**json.load(open("data/adaptive_attack_results.json")), "source": "data/adaptive_attack_results.json"},
}

# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    def clean(o):
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list): return [clean(v) for v in o]
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        return o
    out = clean(FACTS)
    with open("data/paper_facts.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
