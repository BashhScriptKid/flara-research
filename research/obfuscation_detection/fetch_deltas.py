#!/usr/bin/env python3
"""
Fetch all deltas once, save to disk. All other scripts load from here.
"""

import json
import re
import math
import time
import requests
import numpy as np
from typing import List

NIM_API_URL = "https://integrate.api.nvidia.com/v1"
NIM_API_KEY = "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"

def chunk_input(text):
    # Strip (example XXXX) patterns
    text = re.sub(r'\s*\(example\s+\d+\)\.*\s*$', '', text)
    text = re.sub(r'\s*\(example\s+\d+\)\.*\s*', ' ', text).strip()
    
    if len(text) < 5:
        return [text] if text else []
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    for s in sentences:
        if not s.strip(): continue
        if len(s) > 50:
            for sub in re.split(r'(?<=[,;])\s+', s):
                if sub.strip(): chunks.append(sub.strip())
        else:
            chunks.append(s.strip())
    
    # Improved fallback: only split if we have enough words and no sentence boundaries
    if len(chunks) < 2 and len(text) > 10:
        words = text.split()
        if len(words) >= 10:  # Increased from 6 to 10
            mid = len(words) // 2
            chunks = [' '.join(words[:mid]), ' '.join(words[mid:])]
        else:
            # Too short to split meaningfully - keep as single chunk
            chunks = [text]
    
    return chunks if chunks else [text]

def get_embeddings(texts, model):
    headers = {"Authorization": f"Bearer {NIM_API_KEY}", "Content-Type": "application/json"}
    payload = {"input": texts, "model": model, "input_type": "passage"}
    for attempt in range(3):
        try:
            r = requests.post(f"{NIM_API_URL}/embeddings", json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                return [item["embedding"] for item in r.json()["data"]]
            elif r.status_code == 429:
                time.sleep(2 ** attempt)
        except:
            time.sleep(1)
    return []

def compute_angle(v1, v2):
    dot = sum(a*b for a,b in zip(v1,v2))
    m1 = math.sqrt(sum(a*a for a in v1))
    m2 = math.sqrt(sum(b*b for b in v2))
    if m1 > 0 and m2 > 0:
        cos_angle = max(-1, min(1, dot/(m1*m2)))
        angle = math.acos(cos_angle)
        
        # Compute signed angle using Gram-Schmidt
        # Normalize vectors
        v1_norm = [a/m1 for a in v1]
        v2_norm = [b/m2 for b in v2]
        
        # Project v2 onto v1 to get orthogonal component
        proj = sum(a*b for a,b in zip(v1_norm, v2_norm))
        e2_orth = [v2_norm[i] - proj*v1_norm[i] for i in range(len(v2_norm))]
        e2_m = math.sqrt(sum(a*a for a in e2_orth))
        
        if e2_m > 1e-10:
            # Sign determined by first non-zero component of orthogonal direction
            for val in e2_orth:
                if abs(val) > 1e-10:
                    return math.copysign(angle, val)
        return angle
    return 0

def compute_delta_batch(texts, model, batch_size=20):
    all_deltas = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        all_chunks = []
        chunk_map = []
        for t_idx, text in enumerate(batch):
            chunks = chunk_input(text)
            for c_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_map.append((t_idx, c_idx))

        embeddings = get_embeddings(all_chunks, model)
        emb_map = {}
        for idx, (t_idx, c_idx) in enumerate(chunk_map):
            if idx < len(embeddings):
                emb_map[(t_idx, c_idx)] = embeddings[idx]

        for t_idx in range(len(batch)):
            chunks = chunk_input(batch[t_idx])
            text_embs = [emb_map.get((t_idx, c)) for c in range(len(chunks))]
            text_embs = [e for e in text_embs if e is not None]
            if len(text_embs) < 2:
                all_deltas.append(0); continue
            angles = [compute_angle(text_embs[j-1], text_embs[j]) for j in range(1, len(text_embs))]
            if not angles:
                all_deltas.append(0); continue
            temp = 0.5
            # Favor lower angles: negate before softmax
            min_a = min(angles)
            exp_vals = [math.exp((min_a - a) / temp) for a in angles]
            sum_exp = sum(exp_vals)
            weights = [e / sum_exp for e in exp_vals]
            all_deltas.append(sum(w * a for w, a in zip(weights, angles)))
        time.sleep(0.5)
    return all_deltas

def main():
    np.random.seed(42)

    with open('data/obf_trigger.json') as f:
        obf_all = json.load(f)
    with open('data/obf_benign.json') as f:
        ben_all = json.load(f)

    obf_idx = np.random.choice(len(obf_all), size=100, replace=False)
    ben_idx = np.random.choice(len(ben_all), size=200, replace=False)
    obf_samples = [obf_all[i] for i in obf_idx]
    ben_samples = [ben_all[i] for i in ben_idx]
    all_texts = obf_samples + ben_samples
    labels = ['obfuscation'] * 100 + ['benign'] * 200

    models = ["nvidia/nv-embedqa-e5-v5", "nvidia/llama-nemotron-embed-1b-v2"]

    cache = {
        'n_obf': 100,
        'n_ben': 200,
        'texts': all_texts,
        'labels': labels,
        'models': {}
    }

    for model in models:
        short = model.split('/')[-1]
        print(f"Fetching {short}...")
        t0 = time.perf_counter()
        deltas = compute_delta_batch(all_texts, model)
        elapsed = time.perf_counter() - t0
        print(f"  {len(deltas)} deltas in {elapsed:.1f}s")

        cache['models'][short] = {
            'deltas': deltas,
            'mean': float(np.mean(deltas)),
            'std': float(np.std(deltas)),
            'obf_mean': float(np.mean(deltas[:100])),
            'obf_std': float(np.std(deltas[:100])),
            'ben_mean': float(np.mean(deltas[100:])),
            'ben_std': float(np.std(deltas[100:])),
        }

    with open('data/delta_cache.json', 'w') as f:
        json.dump(cache, f)
    print(f"\nSaved: data/delta_cache.json ({len(all_texts)} texts × {len(models)} models)")

if __name__ == '__main__':
    main()
