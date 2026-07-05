#!/usr/bin/env python3
"""
Fetch deltas for ALL embedding models on NIM API.
"""

import json
import re
import math
import time
import requests
import numpy as np

NIM_API_URL = "https://integrate.api.nvidia.com/v1"
NIM_API_KEY = "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"

EMBEDDING_MODELS = [
    "nvidia/nv-embedqa-e5-v5",
    "nvidia/llama-nemotron-embed-1b-v2",
    "baai/bge-m3",
    "nvidia/nv-embed-v1",
    "nvidia/embed-qa-4",
    "snowflake/arctic-embed-l",
    "nvidia/llama-3.2-nv-embedqa-1b-v1",
    "nvidia/nv-embedcode-7b-v1",
    "nvidia/nv-embedqa-mistral-7b-v2",
    "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
    "nvidia/llama-nemotron-embed-vl-1b-v2",
]

def chunk_input(text):
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
    if len(chunks) < 2 and len(text) > 10:
        words = text.split()
        if len(words) >= 10:
            mid = len(words) // 2
            chunks = [' '.join(words[:mid]), ' '.join(words[mid:])]
        else:
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
        v1_norm = [a/m1 for a in v1]
        v2_norm = [b/m2 for b in v2]
        proj = sum(a*b for a,b in zip(v1_norm, v2_norm))
        e2_orth = [v2_norm[i] - proj*v1_norm[i] for i in range(len(v2_norm))]
        e2_m = math.sqrt(sum(a*a for a in e2_orth))
        if e2_m > 1e-10:
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
    
    cache = {
        'n_obf': 100,
        'n_ben': 200,
        'texts': all_texts,
        'labels': labels,
        'models': {}
    }
    
    results = []
    
    for model in EMBEDDING_MODELS:
        short = model.split('/')[-1]
        print(f"\n{'='*60}")
        print(f"Fetching {model}...")
        
        t0 = time.perf_counter()
        deltas = compute_delta_batch(all_texts, model)
        elapsed = time.perf_counter() - t0
        
        if len(deltas) != 300:
            print(f"  FAILED: Only got {len(deltas)} deltas")
            continue
        
        obf_d = deltas[:100]
        ben_d = deltas[100:]
        obf_mean = sum(obf_d)/len(obf_d)
        ben_mean = sum(ben_d)/len(ben_d)
        obf_std = float(np.std(obf_d))
        ben_std = float(np.std(ben_d))
        neg_ben = sum(1 for d in ben_d if d < 0)
        separation = obf_mean - ben_mean
        
        print(f"  {len(deltas)} deltas in {elapsed:.1f}s")
        print(f"  Obfuscation: μ={obf_mean:.4f}, σ={obf_std:.4f}")
        print(f"  Benign:      μ={ben_mean:.4f}, σ={ben_std:.4f}")
        print(f"  Negative benign: {neg_ben}/200")
        print(f"  Separation: {separation:.4f}")
        
        cache['models'][short] = {
            'deltas': deltas,
            'mean': float(sum(deltas)/len(deltas)),
            'std': float(np.std(deltas)),
            'obf_mean': obf_mean,
            'obf_std': obf_std,
            'ben_mean': ben_mean,
            'ben_std': ben_std,
        }
        
        results.append({
            'model': short,
            'time': elapsed,
            'obf_mean': obf_mean,
            'obf_std': obf_std,
            'ben_mean': ben_mean,
            'ben_std': ben_std,
            'separation': separation,
            'neg_benign': neg_ben,
        })
    
    with open('data/delta_cache_all_models.json', 'w') as f:
        json.dump(cache, f)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Time':<8} {'Obf μ':<10} {'Ben μ':<10} {'Sep':<10} {'Neg Ben':<10}")
    print("-" * 78)
    for r in sorted(results, key=lambda x: x['separation'], reverse=True):
        print(f"{r['model']:<30} {r['time']:<8.1f} {r['obf_mean']:<10.4f} {r['ben_mean']:<10.4f} {r['separation']:<10.4f} {r['neg_benign']:<10}")
    
    print(f"\nSaved: data/delta_cache_all_models.json ({len(results)} models × 300 texts)")

if __name__ == '__main__':
    main()
