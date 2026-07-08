#!/usr/bin/env python3
"""
Larger benchmark with statistical significance.
Samples 100 obfuscation + 200 benign (larger than 50/50 but manageable).
"""

import json
import re
import math
import requests
import numpy as np
import pandas as pd
from scipy import stats
from typing import List
import time
from collections import Counter
from sklearn.metrics import roc_auc_score

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

def get_embeddings_batch(texts, model):
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
    """Compute delta for multiple texts using batched API calls."""
    deltas = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # For each text, get chunks and flatten
        all_chunks = []
        chunk_map = []  # (text_idx, chunk_idx)
        for t_idx, text in enumerate(batch):
            chunks = chunk_input(text)
            for c_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_map.append((t_idx, c_idx))
        
        if not all_chunks:
            deltas.extend([0] * len(batch))
            continue
        
        # Batch embed all chunks
        embeddings = get_embeddings_batch(all_chunks, model)
        
        if not embeddings:
            deltas.extend([0] * len(batch))
            continue
        
        # Build embedding map
        emb_map = {}
        for idx, (t_idx, c_idx) in enumerate(chunk_map):
            if idx < len(embeddings):
                emb_map[(t_idx, c_idx)] = embeddings[idx]
        
        # Compute delta for each text
        for t_idx in range(len(batch)):
            chunks = chunk_input(batch[t_idx])
            text_embs = [emb_map.get((t_idx, c)) for c in range(len(chunks))]
            text_embs = [e for e in text_embs if e is not None]
            
            if len(text_embs) < 2:
                deltas.append(0)
                continue
            
            angles = [compute_angle(text_embs[j-1], text_embs[j]) for j in range(1, len(text_embs))]
            if not angles:
                deltas.append(0)
                continue
            
            temp = 0.5
            # Favor lower angles: negate before softmax
            min_a = min(angles)
            exp_vals = [math.exp((min_a - a) / temp) for a in angles]
            sum_exp = sum(exp_vals)
            weights = [e / sum_exp for e in exp_vals]
            deltas.append(sum(w * a for w, a in zip(weights, angles)))
        
        time.sleep(0.5)
    
    return deltas

def char_entropy(text):
    if not text: return 0
    freq = Counter(text)
    total = len(text)
    return -sum((c/total) * math.log2(c/total) for c in freq.values())

def special_char_ratio(text):
    if not text: return 0
    return sum(1 for c in text if not c.isalpha() and not c.isspace()) / len(text)

def regex_score(text):
    score = 0
    if re.search(r'[0-9a-fA-F]{20,}', text): score += 1
    if re.search(r'[A-Za-z0-9+/]{20,}={0,2}', text): score += 1
    if re.search(r'(decode|eval|atob|btoa|fromCharCode|hex|base64)', text, re.IGNORECASE): score += 0.5
    if re.search(r'(\\u[0-9a-fA-F]{4}){3,}', text): score += 1
    return min(score, 2.0) / 2.0

def bootstrap_auc(obf_deltas, ben_deltas, n_bootstrap=500):
    """Bootstrap AUC with confidence intervals."""
    aucs = []
    for _ in range(n_bootstrap):
        obf_sample = np.random.choice(obf_deltas, size=len(obf_deltas), replace=True)
        ben_sample = np.random.choice(ben_deltas, size=len(ben_deltas), replace=True)
        y = [1]*len(obf_sample) + [0]*len(ben_sample)
        scores = list(-obf_sample) + list(-ben_sample)
        try:
            aucs.append(roc_auc_score(y, scores))
        except:
            pass
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def bootstrap_fpr_fnr(obf_deltas, ben_deltas, threshold, n_bootstrap=500):
    """Bootstrap FPR and FNR with confidence intervals."""
    fprs, fnrs = [], []
    for _ in range(n_bootstrap):
        obf_s = np.random.choice(obf_deltas, size=len(obf_deltas), replace=True)
        ben_s = np.random.choice(ben_deltas, size=len(ben_deltas), replace=True)
        tp = np.sum(obf_s < threshold)
        fp = np.sum(ben_s < threshold)
        fn = np.sum(obf_s >= threshold)
        tn = np.sum(ben_s >= threshold)
        fprs.append(fp/(fp+tn) if (fp+tn)>0 else 0)
        fnrs.append(fn/(fn+tp) if (fn+tp)>0 else 0)
    return (np.mean(fprs), np.percentile(fprs, 2.5), np.percentile(fprs, 97.5)), \
           (np.mean(fnrs), np.percentile(fnrs, 2.5), np.percentile(fnrs, 97.5))

def main():
    np.random.seed(42)
    
    with open('data/obf_trigger.json') as f:
        obf_all = json.load(f)
    with open('data/obf_benign.json') as f:
        ben_all = json.load(f)
    
    # Sample 100 obfuscation + 200 benign
    obf_idx = np.random.choice(len(obf_all), size=100, replace=False)
    ben_idx = np.random.choice(len(ben_all), size=200, replace=False)
    obf_samples = [obf_all[i] for i in obf_idx]
    ben_samples = [ben_all[i] for i in ben_idx]
    
    print(f"Dataset: {len(obf_samples)} obfuscation, {len(ben_samples)} benign")
    
    models = ["nvidia/nv-embedqa-e5-v5", "nvidia/llama-nemotron-embed-1b-v2"]
    
    summary_rows = []
    all_results = []
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")
        
        # Batch compute deltas
        print("Computing deltas for obfuscation...")
        obf_deltas = compute_delta_batch(obf_samples, model)
        print("Computing deltas for benign...")
        ben_deltas = compute_delta_batch(ben_samples, model)
        
        obf_arr = np.array(obf_deltas)
        ben_arr = np.array(ben_deltas)
        
        # Collect per-sample results
        for text, delta in zip(obf_samples, obf_deltas):
            all_results.append({'model': model, 'label': 'obfuscation', 'delta': delta,
                              'entropy': char_entropy(str(text)), 'special_ratio': special_char_ratio(str(text)),
                              'regex_score': regex_score(str(text))})
        for text, delta in zip(ben_samples, ben_deltas):
            all_results.append({'model': model, 'label': 'benign', 'delta': delta,
                              'entropy': char_entropy(str(text)), 'special_ratio': special_char_ratio(str(text)),
                              'regex_score': regex_score(str(text))})
        
        # AUC with CI
        auc_mean, auc_lo, auc_hi = bootstrap_auc(obf_arr, ben_arr)
        
        # Find threshold at 5% FPR
        all_d = np.concatenate([obf_arr, ben_arr])
        best_t, best_diff = 0, 999
        for t in np.linspace(all_d.min(), all_d.max(), 500):
            fp = np.sum(ben_arr < t)
            fpr = fp / len(ben_arr)
            if abs(fpr - 0.05) < best_diff:
                best_diff = abs(fpr - 0.05)
                best_t = t
        
        # Compute metrics at this threshold
        tp = np.sum(obf_arr < best_t)
        fp = np.sum(ben_arr < best_t)
        fn = np.sum(obf_arr >= best_t)
        tn = np.sum(ben_arr >= best_t)
        
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0
        recall = 1 - fnr_val
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Bootstrap CI for FPR/FNR
        (fpr_m, fpr_lo, fpr_hi), (fnr_m, fnr_lo, fnr_hi) = bootstrap_fpr_fnr(obf_arr, ben_arr, best_t)
        
        # Mann-Whitney U
        u_stat, p_val = stats.mannwhitneyu(obf_arr, ben_arr, alternative='less')
        
        # Cohen's d
        pooled_std = np.sqrt((np.std(obf_arr)**2 + np.std(ben_arr)**2) / 2)
        cohens_d = (np.mean(ben_arr) - np.mean(obf_arr)) / pooled_std
        
        print(f"\nResults:")
        print(f"  AUC-ROC: {auc_mean:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]")
        print(f"  F1: {f1:.4f}")
        print(f"  Recall @ 5% FPR: {recall:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  FPR: {fpr_m:.4f} [{fpr_lo:.4f}, {fpr_hi:.4f}]")
        print(f"  FNR: {fnr_m:.4f} [{fnr_lo:.4f}, {fnr_hi:.4f}]")
        print(f"  Threshold: {best_t:.4f}")
        print(f"  Obfuscation: {np.mean(obf_arr):.4f} ± {np.std(obf_arr):.4f}")
        print(f"  Benign: {np.mean(ben_arr):.4f} ± {np.std(ben_arr):.4f}")
        print(f"  Mann-Whitney U: p={p_val:.2e}")
        print(f"  Cohen's d: {cohens_d:.3f}")
        
        summary_rows.append({
            'model': model.split('/')[-1][:25],
            'auc': f"{auc_mean:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]",
            'f1': f"{f1:.3f}",
            'recall_5pct_fpr': f"{recall:.3f}",
            'precision': f"{precision:.3f}",
            'p_value': f"{p_val:.1e}",
            "cohens_d": f"{cohens_d:.3f}",
            'threshold': f"{best_t:.3f}"
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv('data/full_benchmark_summary.csv', index=False)
    
    combined = pd.DataFrame(all_results)
    combined.to_csv('data/full_dataset_results.csv', index=False)
    print(f"Saved per-sample results: data/full_dataset_results.csv ({len(combined)} rows)")

if __name__ == '__main__':
    main()
