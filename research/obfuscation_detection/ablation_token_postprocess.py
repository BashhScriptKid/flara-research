"""
Post-process token_delta_v2.py's output for §6.2 (Intra-Chunk Token Angles):
filter to single-chunk samples (sentence delta == 0) and compute per-type AUC
for backslash_visual and the broad 'unstructured' bucket, matching the
paper's specific claims.
"""
import json
import re
import numpy as np
from sklearn.metrics import roc_auc_score

tok = json.load(open("data/delta_cache_token.json"))
model_key = list(tok.keys())[0]
rows = tok[model_key]
text_to_delta = {r['text']: r['delta'] for r in rows}
text_to_label = {r['text']: r['label'] for r in rows}

cache = json.load(open("data/sentence_chunk_cache.json"))
sent_delta = {t: d for t, d in zip(cache['texts'], cache['models']['nvidia/nv-embedqa-e5-v5']['deltas'])}

obf_trigger = json.load(open("data/obf_trigger.json"))

def classify_encoding(text):
    t = text.lower()
    if 'decode this hex' in t or '\\x' in text: return 'hex'
    if 'base64' in t: return 'base64'
    if re.search(r'rot\d+', t): return 'rot'
    if text.count('\\') > 5: return 'backslash_visual'
    if re.search(r'[A-Z]\.[A-Z]\.[A-Z]', text): return 'dot_spacing'
    if '<|im_start|>' in text: return 'xml_tag'
    return 'unstructured'

single_chunk_texts = [t for t, d in sent_delta.items() if d == 0.0]
sc_set = set(single_chunk_texts)

sc_obf_bsv = [t for t in obf_trigger if t in sc_set and classify_encoding(t) == 'backslash_visual']
sc_obf_unstructured = [t for t in obf_trigger if t in sc_set and classify_encoding(t) == 'unstructured']
sc_ben = [t for t, l in text_to_label.items() if l == 'benign' and t in sc_set]

print(f"Single-chunk backslash_visual: {len(sc_obf_bsv)}")
print(f"Single-chunk unstructured: {len(sc_obf_unstructured)}")
print(f"Single-chunk benign: {len(sc_ben)}")

ben_deltas = np.array([text_to_delta[t] for t in sc_ben])

for name, group in [("backslash_visual", sc_obf_bsv), ("unstructured", sc_obf_unstructured)]:
    if not group:
        continue
    obf_deltas = np.array([text_to_delta[t] for t in group])
    y = np.array([1] * len(obf_deltas) + [0] * len(ben_deltas))
    scores = np.concatenate([obf_deltas, ben_deltas])
    auc = roc_auc_score(y, scores)
    auc_best = max(auc, 1 - auc)
    print(f"{name}: n={len(group)}  AUC(raw)={auc:.4f}  AUC(best-direction)={auc_best:.4f}")
