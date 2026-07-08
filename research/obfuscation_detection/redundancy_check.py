"""
Tests whether MAX-delta's failure to improve the ensemble is actually due to
redundancy with regex/weird-unicode (overlap in which samples get flagged),
a ceiling effect (ensemble already near-saturated), or an objective-mismatch
artifact (tpr_minus_fpr at fixed threshold vs AUC).
"""
import numpy as np
import json
import re
import os
from collections import Counter
from sklearn.metrics import roc_auc_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))

obf_samples = json.load(open("data/obf_trigger.json"))
ben_samples = json.load(open("data/obf_benign.json"))
n_obf, n_ben = len(obf_samples), len(ben_samples)
all_samples = obf_samples + ben_samples
y_true = np.array([1] * n_obf + [0] * n_ben)

cache = json.load(open("data/sentence_chunk_cache.json"))
d_sent_e5 = np.array(cache["models"]["nvidia/nv-embedqa-e5-v5"]["deltas"])
para_deltas = np.load("data/para_deltas_full.npy")

per_chunk = json.load(open("data/per_chunk_deltas_e5.json"))
max_delta = np.array([max(d) if d else 0.0 for d in per_chunk["per_sample_deltas"]])

# structural confounds, same chunk_text() logic as residual_delta_test.py
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

chunks_list = [chunk_text(t) for t in all_samples]
nc_all = np.array([len(c) for c in chunks_list])
word_all = np.array([len(t.split()) for t in all_samples])

X = np.column_stack([np.ones_like(nc_all, dtype=float), nc_all, word_all])
beta, *_ = np.linalg.lstsq(X, max_delta, rcond=None)
resid_max = max_delta - X @ beta

# regex + weird, exact same logic as ensemble_full.py
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

feature_matrix = np.array([
    [f[p] for p in FEATURE_NAMES] for f in (compute_regex_features(t) for t in all_samples)
])

WEIRD_UNICODE_RE = re.compile(
    r'[\U0001D400-\U0001D7FF！-～①-⓿²-₟'
    r'​-⁤﻿̀-ͯ←-⇿⬀-⯿]'
)
is_single_chunk = d_sent_e5 == 0
weird_raw = np.array([1.0 if WEIRD_UNICODE_RE.search(t) else 0.0 for t in all_samples])
weird_raw = np.where(is_single_chunk, weird_raw, 0.0)

norm_full = json.load(open("data/ensemble_full_cache.json"))["norm_full"]
idx = [FEATURE_NAMES.index(p) for p in norm_full["regex_feature_names"]]
w = np.array([norm_full["regex_sub_weights"][p] for p in norm_full["regex_feature_names"]])
regex_raw = feature_matrix[:, idx] @ w
regex_norm = (regex_raw - norm_full["regex_raw_min"]) / (norm_full["regex_raw_max"] - norm_full["regex_raw_min"] + 1e-10)

print("=== Check 1: redundancy (correlation of MAX's residual signal with regex/weird) ===")
print(f"corr(resid_max, regex_norm) = {np.corrcoef(resid_max, regex_norm)[0,1]:.4f}")
print(f"corr(resid_max, weird_raw)  = {np.corrcoef(resid_max, weird_raw)[0,1]:.4f}")
print(f"corr(resid_max, para_norm-equivalent para_deltas) = {np.corrcoef(resid_max, para_deltas)[0,1]:.4f}")

print("\n=== Check 2: ceiling effect (overlap of who regex/weird already catch vs who resid_max would catch) ===")
# samples regex+weird alone already separate well (high score on obf, low on ben)
regex_weird_score = 0.45 * regex_norm + 0.32 * weird_raw  # rough production-like combo, sent/para excluded
# among obf samples NOT already strongly flagged by regex/weird, does resid_max distinguish them?
obf_mask = y_true == 1
thr_low = np.percentile(regex_weird_score[~obf_mask], 99)  # benign 99th pct as cutoff
weakly_flagged_obf = obf_mask & (regex_weird_score < thr_low)
print(f"obf samples NOT caught by regex+weird alone (score < benign 99th pct): {weakly_flagged_obf.sum()} / {obf_mask.sum()}")
if weakly_flagged_obf.sum() > 5:
    auc_resid_on_hard = roc_auc_score(
        y_true[weakly_flagged_obf | (~obf_mask)],
        resid_max[weakly_flagged_obf | (~obf_mask)]
    )
    print(f"AUC(resid_max) restricted to (hard obf + all benign): {auc_resid_on_hard:.4f}")

print("\n=== Check 3: objective mismatch (full ensemble standalone AUC, sent=mean vs sent=max, NOT just tpr-fpr) ===")
sent_norm_mean = (d_sent_e5 - d_sent_e5.min()) / (d_sent_e5.max() - d_sent_e5.min() + 1e-10)
max_norm = (max_delta - max_delta.min()) / (max_delta.max() - max_delta.min() + 1e-10)
para_norm = (para_deltas - para_deltas.min()) / (para_deltas.max() - para_deltas.min() + 1e-10)

w_sent, w_regex, w_para, w_weird = norm_full["w_sent"], norm_full["w_regex"], norm_full["w_para"], norm_full["w_weird"]
ens_mean = w_sent * sent_norm_mean + w_regex * regex_norm + w_para * para_norm + w_weird * weird_raw
ens_max_sameweights = w_sent * max_norm + w_regex * regex_norm + w_para * para_norm + w_weird * weird_raw
print(f"AUC ensemble (production weights, sent=MEAN): {roc_auc_score(y_true, ens_mean):.5f}")
print(f"AUC ensemble (production weights, sent=MAX):  {roc_auc_score(y_true, ens_max_sameweights):.5f}")
