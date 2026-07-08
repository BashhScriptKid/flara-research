import numpy as np
import json
import re
import requests
import time
import os
from collections import Counter
from sklearn.metrics import roc_auc_score, roc_curve

NIM_API_KEY = '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'
NIM_BASE_URL = 'https://integrate.api.nvidia.com/v1'
MODEL = 'nvidia/nv-embedqa-e5-v5'

def nim_embed(texts, batch_size=32):
    all_emb = [None] * len(texts)
    headers = {'Authorization': f'Bearer {NIM_API_KEY}', 'Content-Type': 'application/json'}
    def clean(t):
        t = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', t)
        t = re.sub(r'\s{2,}', ' ', t)
        return t[:1600].strip() or 'empty'
    for i in range(0, len(texts), batch_size):
        batch = [clean(t) for t in texts[i:i + batch_size]]
        try:
            r = requests.post(f'{NIM_BASE_URL}/embeddings', headers=headers,
                            json={'model': MODEL, 'input': batch, 'input_type': 'passage'}, timeout=60)
            r.raise_for_status()
            for j, item in enumerate(sorted(r.json()['data'], key=lambda x: x['index'])):
                all_emb[i + j] = np.array(item['embedding'], dtype=np.float32)
        except:
            for j, t in enumerate(batch):
                idx = i + j
                if all_emb[idx] is not None: continue
                try:
                    r2 = requests.post(f'{NIM_BASE_URL}/embeddings', headers=headers,
                                     json={'model': MODEL, 'input': [t], 'input_type': 'passage'}, timeout=30)
                    r2.raise_for_status()
                    all_emb[idx] = np.array(r2.json()['data'][0]['embedding'], dtype=np.float32)
                except:
                    all_emb[idx] = np.zeros(4096, dtype=np.float32)
                time.sleep(0.1)
    return all_emb

def angle(e1, e2):
    n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
    if n1 < 1e-10 or n2 < 1e-10: return 0.0
    return float(np.arccos(np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)))

cache = json.load(open('data/sentence_chunk_cache.json'))
e5_deltas = np.array(cache['models']['nvidia/nv-embedqa-e5-v5']['deltas'])
texts = cache['texts']
labels = cache['labels']

sc_obf = [i for i in range(len(labels)) if e5_deltas[i] == 0.0 and labels[i] == 'obfuscation']
sc_ben = [i for i in range(len(labels)) if e5_deltas[i] == 0.0 and labels[i] == 'benign']
print(f'Single-chunk: {len(sc_obf)} obf, {len(sc_ben)} ben')

def smart_split(text):
    parts = re.split(r'\b(but|however|although|and|then|now|instead|while|whereas|so|yet|only|just|even)\b', text, flags=re.IGNORECASE)
    if len(parts) > 1:
        merged = []
        for i, part in enumerate(parts):
            if i > 0 and len(part.strip()) <= 5 and merged:
                merged[-1] = merged[-1] + part
            else:
                merged.append(part)
        chunks = [c.strip() for c in merged if c.strip()]
        if len(chunks) >= 2:
            return chunks, 'connective'
    parts = re.split(r'(?<=[,;:])\s+', text)
    if len(parts) >= 2:
        return [p.strip() for p in parts if p.strip()], 'punctuation'
    words = text.split()
    if len(words) >= 4:
        mid = len(words) // 2
        return [' '.join(words[:mid]), ' '.join(words[mid:])], 'midpoint'
    return [text], 'none'

split_types = {}
text_to_chunks = {}
for i in sc_obf + sc_ben:
    chunks, stype = smart_split(texts[i])
    text_to_chunks[i] = chunks
    split_types[i] = stype

print(f'Split types: {Counter(split_types.values())}')

for i in sc_obf[:10]:
    chunks = text_to_chunks[i]
    if len(chunks) > 1:
        print(f'  [{i}] ({split_types[i]}) {len(chunks)} chunks:')
        for c in chunks:
            print(f'    "{c[:70]}"')

all_texts = []
for i in sc_obf + sc_ben:
    all_texts.extend(text_to_chunks[i])
print(f'\nEmbedding {len(all_texts)} chunks...')
all_emb = nim_embed(all_texts)

deltas = {}
offset = 0
for i in sc_obf + sc_ben:
    chunks = text_to_chunks[i]
    chunk_embs = all_emb[offset:offset + len(chunks)]
    offset += len(chunks)
    if len(chunks) >= 2:
        angles = [angle(chunk_embs[j], chunk_embs[j+1]) for j in range(len(chunks)-1)]
        deltas[i] = float(np.mean(angles))
    else:
        deltas[i] = 0.0

obf_d = [deltas[i] for i in sc_obf]
ben_d = [deltas[i] for i in sc_ben]
print(f'\nObf: mu={np.mean(obf_d):.4f} sig={np.std(obf_d):.4f}')
print(f'Ben: mu={np.mean(ben_d):.4f} sig={np.std(ben_d):.4f}')

y = np.array([1]*len(obf_d) + [0]*len(ben_d))
sc_arr = np.array(obf_d + ben_d)
auc = max(roc_auc_score(y, sc_arr), roc_auc_score(y, -sc_arr))
fpr_arr, tpr_arr, _ = roc_curve(y, sc_arr)
det_1 = tpr_arr[np.searchsorted(fpr_arr, 0.01)]
det_5 = tpr_arr[np.searchsorted(fpr_arr, 0.05)]
print(f'AUC: {auc:.4f}  Det@1%: {det_1:.4f}  Det@5%: {det_5:.4f}')

print('\n--- Split-type breakdown ---')
for stype in ['connective', 'punctuation', 'midpoint', 'none']:
    obf_idx = [i for i in sc_obf if split_types[i] == stype]
    ben_idx = [i for i in sc_ben if split_types[i] == stype]
    if obf_idx:
        obf_vals = [deltas[i] for i in obf_idx]
        ben_vals = [deltas[i] for i in ben_idx] if ben_idx else [0.0]
        print(f'  {stype:15s}: obf mu={np.mean(obf_vals):.4f} (n={len(obf_idx)}) | ben mu={np.mean(ben_vals):.4f} (n={len(ben_idx)})')

# Compare with existing multi-granularity
print('\n--- Comparison: original sentence delta vs connective-split delta ---')
print(f'Original single-chunk delta: AUC=0.500 (all zeros)')
print(f'Connective-split delta:      AUC={auc:.4f}')

# Now test: how does connective-split compare to original multi-chunk delta?
multi_obf = [i for i in range(len(labels)) if e5_deltas[i] > 0.0 and labels[i] == 'obfuscation']
multi_ben = [i for i in range(len(labels)) if e5_deltas[i] > 0.0 and labels[i] == 'benign']
print(f'\nMulti-chunk samples: {len(multi_obf)} obf, {len(multi_ben)} ben')
print(f'Connective split recovers: {sum(1 for i in sc_obf if deltas[i] > 0)}/{len(sc_obf)} single-chunk obf samples')

print('\nDone.')
