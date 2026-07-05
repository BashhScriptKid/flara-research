#!/usr/bin/env python3
"""
Latency benchmark + z-score threshold portability validation.

Measures per-input time for each detection method and tests whether
z-score normalization makes thresholds portable across embedding models.
"""

import json
import re
import math
import time
import requests
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple

NIM_API_URL = "https://integrate.api.nvidia.com/v1"
NIM_API_KEY = "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"

# ─── Core functions (copied from full_benchmark.py for self-containment) ───

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

# ─── Latency measurement ───

def measure_latency(texts: List[str], model: str, n_runs: int = 3) -> dict:
    """Measure per-input latency for each method.

    Returns dict with mean±std for each component.
    """
    n = len(texts)

    # --- Delta angle (includes API call) ---
    delta_chunk_times = []
    delta_api_times = []
    delta_compute_times = []
    delta_total_times = []

    for _ in range(n_runs):
        for text in texts:
            t0 = time.perf_counter()
            chunks = chunk_input(text)
            t1 = time.perf_counter()
            embeddings = get_embeddings(chunks, model)
            t2 = time.perf_counter()

            if len(embeddings) >= 2:
                angles = [compute_angle(embeddings[j-1], embeddings[j]) for j in range(1, len(embeddings))]
                temp = 0.5
                # Favor lower angles: negate before softmax
                min_a = min(angles)
                exp_vals = [math.exp((min_a - a) / temp) for a in angles]
                sum_exp = sum(exp_vals)
                weights = [e / sum_exp for e in exp_vals]
                _ = sum(w * a for w, a in zip(weights, angles))
            t3 = time.perf_counter()

            delta_chunk_times.append(t1 - t0)
            delta_api_times.append(t2 - t1)
            delta_compute_times.append(t3 - t2)
            delta_total_times.append(t3 - t0)

    # --- CPU-only methods (no API call) ---
    entropy_times = []
    special_times = []
    regex_times = []

    for _ in range(n_runs):
        for text in texts:
            t0 = time.perf_counter()
            _ = char_entropy(text)
            t1 = time.perf_counter()
            entropy_times.append(t1 - t0)

            t0 = time.perf_counter()
            _ = special_char_ratio(text)
            t1 = time.perf_counter()
            special_times.append(t1 - t0)

            t0 = time.perf_counter()
            _ = regex_score(text)
            t1 = time.perf_counter()
            regex_times.append(t1 - t0)

    def stats(arr):
        arr = np.array(arr) * 1000  # convert to ms
        return f"{np.mean(arr):.2f} ± {np.std(arr):.2f}"

    return {
        'delta_chunk_ms': stats(delta_chunk_times),
        'delta_api_ms': stats(delta_api_times),
        'delta_compute_ms': stats(delta_compute_times),
        'delta_total_ms': stats(delta_total_times),
        'entropy_ms': stats(entropy_times),
        'special_ms': stats(special_times),
        'regex_ms': stats(regex_times),
        'n_inputs': n,
        'n_runs': n_runs,
    }

# ─── Z-score portability ───

def compute_zscore_stats(deltas: np.ndarray) -> Tuple[float, float]:
    """Compute mean and std of a delta distribution."""
    return float(np.mean(deltas)), float(np.std(deltas))

def zscore_classify(theta: float, mu_benign: float, sigma_benign: float, k: float = 2.0) -> bool:
    """Return True if input should be flagged (z-score < -k, i.e., below benign mean)."""
    if sigma_benign == 0:
        return False
    z = (theta - mu_benign) / sigma_benign
    return z < -k

def validate_portability(obf_deltas: np.ndarray, ben_deltas: np.ndarray,
                          mu_benign: float, sigma_benign: float, k: float = 2.0) -> dict:
    """Validate z-score threshold on a dataset.

    Classification: flag if z < -k (delta is far below benign mean).
    """
    tp = sum(1 for d in obf_deltas if zscore_classify(d, mu_benign, sigma_benign, k))
    fp = sum(1 for d in ben_deltas if zscore_classify(d, mu_benign, sigma_benign, k))
    fn = sum(1 for d in obf_deltas if not zscore_classify(d, mu_benign, sigma_benign, k))
    tn = sum(1 for d in ben_deltas if not zscore_classify(d, mu_benign, sigma_benign, k))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        'k': k,
        'mu_benign': mu_benign,
        'sigma_benign': sigma_benign,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
    }

def main():
    np.random.seed(42)

    # Load data
    with open('data/obf_trigger.json') as f:
        obf_all = json.load(f)
    with open('data/obf_benign.json') as f:
        ben_all = json.load(f)

    # Sample for latency test (20 inputs, enough to get stable timing)
    lat_n = 20
    obf_lat = obf_all[:lat_n]
    ben_lat = ben_all[:lat_n]
    lat_texts = obf_lat + ben_lat

    # Full dataset for z-score validation (N=300: 100 obf + 200 ben)
    obf_idx = np.random.choice(len(obf_all), size=100, replace=False)
    ben_idx = np.random.choice(len(ben_all), size=200, replace=False)
    obf_full = [obf_all[i] for i in obf_idx]
    ben_full = [ben_all[i] for i in ben_idx]

    models = ["nvidia/nv-embedqa-e5-v5", "nvidia/llama-nemotron-embed-1b-v2"]

    all_latency = []
    all_portability = []

    for model in models:
        short = model.split('/')[-1]
        print(f"\n{'='*60}")
        print(f"Model: {short}")
        print(f"{'='*60}")

        # ── Latency ──
        print(f"\nMeasuring latency ({lat_n} inputs × 3 runs)...")
        lat = measure_latency(lat_texts, model, n_runs=3)
        lat['model'] = short
        all_latency.append(lat)

        print(f"  Delta chunk:   {lat['delta_chunk_ms']}")
        print(f"  Delta API:     {lat['delta_api_ms']}")
        print(f"  Delta compute: {lat['delta_compute_ms']}")
        print(f"  Delta TOTAL:   {lat['delta_total_ms']}")
        print(f"  Entropy:       {lat['entropy_ms']}")
        print(f"  Special char:  {lat['special_ms']}")
        print(f"  Regex:         {lat['regex_ms']}")

        # ── Z-score portability ──
        print(f"\nComputing deltas for full dataset (N=300)...")

        # Compute deltas for full dataset
        all_texts = obf_full + ben_full
        all_deltas = []
        batch_size = 20
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i+batch_size]
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
                    all_deltas.append(0)
                    continue
                angles = [compute_angle(text_embs[j-1], text_embs[j]) for j in range(1, len(text_embs))]
                if not angles:
                    all_deltas.append(0)
                    continue
                temp = 0.5
                # Favor lower angles: negate before softmax
                min_a = min(angles)
                exp_vals = [math.exp((min_a - a) / temp) for a in angles]
                sum_exp = sum(exp_vals)
                weights = [e / sum_exp for e in exp_vals]
                all_deltas.append(sum(w * a for w, a in zip(weights, angles)))
            time.sleep(0.5)

        all_deltas = np.array(all_deltas)
        obf_deltas = all_deltas[:100]
        ben_deltas = all_deltas[100:]

        mu_b, sig_b = compute_zscore_stats(ben_deltas)
        mu_o, sig_o = compute_zscore_stats(obf_deltas)

        print(f"\n  Benign distribution:  μ={mu_b:.4f}, σ={sig_b:.4f}")
        print(f"  Obfuscation distrib:  μ={mu_o:.4f}, σ={sig_o:.4f}")

        # Test different k values
        print(f"\n  Z-score portability (trained on THIS model's benign distribution):")
        for k in [1.0, 1.5, 2.0, 2.5, 3.0]:
            res = validate_portability(obf_deltas, ben_deltas, mu_b, sig_b, k)
            print(f"    k={k}: recall={res['recall']:.3f}, FPR={res['fpr']:.3f}, F1={res['f1']:.3f}")

        # Cross-model test: train on E5, test on Nemotron (and vice versa)
        # This requires both models' deltas for the same inputs
        # Store for later
        port_row = {
            'model': short,
            'mu_benign': mu_b,
            'sigma_benign': sig_b,
            'mu_obf': mu_o,
            'sigma_obf': sig_o,
            'benign_deltas': ben_deltas.tolist(),
            'obf_deltas': obf_deltas.tolist(),
        }
        all_portability.append(port_row)

    # ── Cross-model portability test ──
    if len(all_portability) == 2:
        print(f"\n{'='*60}")
        print(f"CROSS-MODEL PORTABILITY TEST")
        print(f"{'='*60}")

        p0 = all_portability[0]  # E5
        p1 = all_portability[1]  # Nemotron

        # Train threshold on E5, apply to Nemotron
        print(f"\nTrain on {p0['model']}, test on {p1['model']}:")
        for k in [1.0, 1.5, 2.0, 2.5, 3.0]:
            res = validate_portability(
                np.array(p1['obf_deltas']),
                np.array(p1['benign_deltas']),
                p0['mu_benign'], p0['sigma_benign'], k
            )
            print(f"    k={k}: recall={res['recall']:.3f}, FPR={res['fpr']:.3f}, F1={res['f1']:.3f}")

        # Train threshold on Nemotron, apply to E5
        print(f"\nTrain on {p1['model']}, test on {p0['model']}:")
        for k in [1.0, 1.5, 2.0, 2.5, 3.0]:
            res = validate_portability(
                np.array(p0['obf_deltas']),
                np.array(p0['benign_deltas']),
                p1['mu_benign'], p1['sigma_benign'], k
            )
            print(f"    k={k}: recall={res['recall']:.3f}, FPR={res['fpr']:.3f}, F1={res['f1']:.3f}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("LATENCY SUMMARY")
    print(f"{'='*60}")
    lat_df = pd.DataFrame(all_latency)
    print(lat_df.to_string(index=False))
    lat_df.to_csv('data/latency_results.csv', index=False)
    print(f"\nSaved: data/latency_results.csv")

    print(f"\n{'='*60}")
    print("PORTABILITY SUMMARY")
    print(f"{'='*60}")
    for p in all_portability:
        print(f"{p['model']}: μ_benign={p['mu_benign']:.4f}, σ={p['sigma_benign']:.4f}")

if __name__ == '__main__':
    main()
