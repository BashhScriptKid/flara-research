"""
Multi-Hop (Skip-Gram) Delta test.
Re-embed all chunks, compute adjacent, skip-1, skip-2 deltas.
Compare detection power for each hop distance.
"""
import numpy as np
import json
import re
import requests
import time
import os
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score, roc_curve

NIM_API_KEY = os.getenv('NIM_API_KEY', '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')
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nv-embedqa-e5-v5"
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def nim_embed(texts, model=MODEL, batch_size=64):
    all_emb = [None] * len(texts)
    headers = {"Authorization": f"Bearer {NIM_API_KEY}", "Content-Type": "application/json"}
    def clean(t):
        t = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', t)
        t = re.sub(r'[\U0001D400-\U0001D7FF]', '', t)
        t = re.sub(r'\s{2,}', ' ', t)
        return t[:1600].strip() or "empty"
    for i in range(0, len(texts), batch_size):
        batch = [clean(t) for t in texts[i:i + batch_size]]
        try:
            r = requests.post(f"{NIM_BASE_URL}/embeddings", headers=headers,
                            json={"model": model, "input": batch, "input_type": "passage"}, timeout=60)
            r.raise_for_status()
            for j, item in enumerate(sorted(r.json()['data'], key=lambda x: x['index'])):
                all_emb[i + j] = np.array(item['embedding'], dtype=np.float32)
        except Exception as e:
            for j, t in enumerate(batch):
                idx = i + j
                if all_emb[idx] is not None: continue
                try:
                    r2 = requests.post(f"{NIM_BASE_URL}/embeddings", headers=headers,
                                     json={"model": model, "input": [t], "input_type": "passage"}, timeout=30)
                    r2.raise_for_status()
                    all_emb[idx] = np.array(r2.json()['data'][0]['embedding'], dtype=np.float32)
                except:
                    all_emb[idx] = np.zeros(4096, dtype=np.float32)
                time.sleep(0.1)
        if i % 320 == 0:
            print(f"  Embedded {min(i+len(batch), len(texts))}/{len(texts)} chunks...")
    return all_emb


def angle(e1, e2):
    n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
    if n1 < 1e-10 or n2 < 1e-10: return 0.0
    return float(np.arccos(np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)))


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


# ═══════════════════════════════════════════════════════════════════
# Load data and chunk
# ═══════════════════════════════════════════════════════════════════
print("Loading and chunking...")
cache = json.load(open("data/sentence_chunk_cache.json"))
texts = cache['texts']
labels = cache['labels']

obf_trigger = json.load(open("data/obf_trigger.json"))
obf_samples = obf_trigger if isinstance(obf_trigger[0], str) else [s['text'] for s in obf_trigger]
obf_types = []
for t in obf_samples:
    enc = classify_encoding(t)
    if enc == 'unstructured':
        obf_types.append(classify_unstructured(t))
    else:
        obf_types.append(enc)

# Rebuild chunk lists
all_chunks = []
chunk_ranges = []
for text in texts:
    chunks = chunk_text(text)
    start = len(all_chunks)
    all_chunks.extend(chunks)
    chunk_ranges.append((start, len(all_chunks)))

n_chunks = len(all_chunks)
print(f"Total chunks: {n_chunks}")
print(f"Chunk counts: {Counter(r[1]-r[0] for r in chunk_ranges).most_common(10)}")

# ═══════════════════════════════════════════════════════════════════
# Re-embed all chunks
# ═══════════════════════════════════════════════════════════════════
print(f"\nEmbedding {n_chunks} chunks...")
t0 = time.time()
all_emb = nim_embed(all_chunks)
print(f"Embedded in {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════
# Compute skip-gram deltas
# ═══════════════════════════════════════════════════════════════════
print("\nComputing skip-gram deltas...")

adjacent_deltas = []
skip1_deltas = []
skip2_deltas = []
n_chunk_counts = []

for si, ei in chunk_ranges:
    emb = all_emb[si:ei]
    n = len(emb)
    n_chunk_counts.append(n)
    
    # Adjacent (standard)
    if n >= 2:
        angles = [angle(emb[i], emb[i+1]) for i in range(n-1)]
        angles = [a for a in angles if a > 0]
        adjacent_deltas.append(float(np.mean(angles)) if angles else 0.0)
    else:
        adjacent_deltas.append(0.0)
    
    # Skip-1 (i to i+2)
    if n >= 3:
        angles = [angle(emb[i], emb[i+2]) for i in range(n-2)]
        angles = [a for a in angles if a > 0]
        skip1_deltas.append(float(np.mean(angles)) if angles else 0.0)
    else:
        skip1_deltas.append(0.0)
    
    # Skip-2 (i to i+3)
    if n >= 4:
        angles = [angle(emb[i], emb[i+3]) for i in range(n-3)]
        angles = [a for a in angles if a > 0]
        skip2_deltas.append(float(np.mean(angles)) if angles else 0.0)
    else:
        skip2_deltas.append(0.0)

adjacent_deltas = np.array(adjacent_deltas)
skip1_deltas = np.array(skip1_deltas)
skip2_deltas = np.array(skip2_deltas)
n_chunk_counts = np.array(n_chunk_counts)
labels_arr = np.array(labels)
obf_mask = labels_arr == 'obfuscation'
ben_mask = labels_arr == 'benign'

# ═══════════════════════════════════════════════════════════════════
# Overall Results
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("OVERALL: ADJACENT vs SKIP-1 vs SKIP-2")
print(f"{'='*70}")

for name, deltas in [('Adjacent (std)', adjacent_deltas),
                      ('Skip-1 (i→i+2)', skip1_deltas),
                      ('Skip-2 (i→i+3)', skip2_deltas)]:
    obf = deltas[obf_mask]
    ben = deltas[ben_mask]
    y = np.array([1]*len(obf) + [0]*len(ben))
    sc = np.concatenate([obf, ben])
    auc = max(roc_auc_score(y, sc), roc_auc_score(y, -sc))
    fpr_arr, tpr_arr, _ = roc_curve(y, sc)
    det_1 = tpr_arr[np.searchsorted(fpr_arr, 0.01)]
    det_5 = tpr_arr[np.searchsorted(fpr_arr, 0.05)]
    print(f"\n  {name}:")
    print(f"    Obf: μ={np.mean(obf):.4f} σ={np.std(obf):.4f}")
    print(f"    Ben: μ={np.mean(ben):.4f} σ={np.std(ben):.4f}")
    print(f"    AUC={auc:.4f}  Det@1%={det_1:.4f}  Det@5%={det_5:.4f}")

# ═══════════════════════════════════════════════════════════════════
# Multi-chunk only
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("MULTI-CHUNK ONLY (n >= 3 chunks)")
print(f"{'='*70}")

for min_chunks in [3, 4]:
    multi_mask = n_chunk_counts >= min_chunks
    print(f"\n  min_chunks={min_chunks}: {multi_mask.sum()} samples")
    for name, deltas in [('Adjacent', adjacent_deltas), ('Skip-1', skip1_deltas), ('Skip-2', skip2_deltas)]:
        obf = deltas[obf_mask & multi_mask]
        ben = deltas[ben_mask & multi_mask]
        if len(obf) == 0 or len(ben) == 0: continue
        y = np.array([1]*len(obf) + [0]*len(ben))
        sc = np.concatenate([obf, ben])
        try:
            auc = max(roc_auc_score(y, sc), roc_auc_score(y, -sc))
            fpr_arr, tpr_arr, _ = roc_curve(y, sc)
            det_1 = tpr_arr[np.searchsorted(fpr_arr, 0.01)]
            print(f"    {name:12s}: AUC={auc:.4f} Det@1%={det_1:.4f}")
        except:
            print(f"    {name:12s}: N/A")

# ═══════════════════════════════════════════════════════════════════
# Per-type
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PER-TYPE AUC")
print(f"{'='*70}")

print(f"\n{'Type':<25} {'N':>4} {'Chunks':>7} {'Adj':>8} {'Skip1':>8} {'Skip2':>8} {'Best':>6}")
print("-" * 70)

for t in sorted(set(obf_types)):
    mask = np.array(obf_types) == t
    n = mask.sum()
    if n < 3: continue
    
    avg_chunks = np.mean(n_chunk_counts[obf_mask][mask])
    obf_adj = adjacent_deltas[obf_mask][mask]
    obf_s1 = skip1_deltas[obf_mask][mask]
    obf_s2 = skip2_deltas[obf_mask][mask]
    
    ben_adj = adjacent_deltas[ben_mask]
    ben_s1 = skip1_deltas[ben_mask]
    ben_s2 = skip2_deltas[ben_mask]
    
    y = np.array([1]*n + [0]*len(ben_adj))
    
    try:
        auc_adj = max(roc_auc_score(y, np.concatenate([obf_adj, ben_adj])),
                       roc_auc_score(y, -np.concatenate([obf_adj, ben_adj])))
    except: auc_adj = float('nan')
    try:
        auc_s1 = max(roc_auc_score(y, np.concatenate([obf_s1, ben_s1])),
                       roc_auc_score(y, -np.concatenate([obf_s1, ben_s1])))
    except: auc_s1 = float('nan')
    try:
        auc_s2 = max(roc_auc_score(y, np.concatenate([obf_s2, ben_s2])),
                       roc_auc_score(y, -np.concatenate([obf_s2, ben_s2])))
    except: auc_s2 = float('nan')
    
    vals = [('adj', auc_adj), ('s1', auc_s1), ('s2', auc_s2)]
    vals_valid = [(n, v) for n, v in vals if not np.isnan(v)]
    best = max(vals_valid, key=lambda x: x[1])[0] if vals_valid else '—'
    
    print(f"{t:<25} {n:>4} {avg_chunks:>7.1f} {auc_adj:>8.4f} {auc_s1:>8.4f} {auc_s2:>8.4f} {best:>6}")

# ═══════════════════════════════════════════════════════════════════
# Unique catches
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("UNIQUE CATCHES")
print(f"{'='*70}")

adj_thr = np.percentile(adjacent_deltas[ben_mask], 99)
s1_thr = np.percentile(skip1_deltas[ben_mask], 99)
s2_thr = np.percentile(skip2_deltas[ben_mask], 99)

adj_caught = adjacent_deltas[obf_mask] >= adj_thr
s1_caught = skip1_deltas[obf_mask] >= s1_thr
s2_caught = skip2_deltas[obf_mask] >= s2_thr

print(f"Adjacent catches: {adj_caught.sum()}/{obf_mask.sum()}")
print(f"Skip-1 catches:   {s1_caught.sum()}/{obf_mask.sum()}")
print(f"Skip-2 catches:   {s2_caught.sum()}/{obf_mask.sum()}")
print(f"Union:            {(adj_caught | s1_caught | s2_caught).sum()}/{obf_mask.sum()}")
print(f"Skip-1 unique (adj misses): {(s1_caught & ~adj_caught).sum()}")
print(f"Skip-2 unique (adj misses): {(s2_caught & ~adj_caught).sum()}")

# Show unique catches
unique_s1_idx = np.where(obf_mask)[0][s1_caught & ~adj_caught]
if len(unique_s1_idx) > 0:
    print(f"\nSkip-1 unique catches:")
    for idx in unique_s1_idx[:10]:
        print(f"  [{idx}] type={obf_types[idx]} adj={adjacent_deltas[idx]:.4f} s1={skip1_deltas[idx]:.4f} chunks={n_chunk_counts[idx]}")
        print(f"       {texts[idx][:80]}...")

print("\nDone.")
