#!/usr/bin/env python3
"""
Graphs for latency benchmark and z-score portability.
Outputs to graphs/ directory.
"""

import json
import re
import math
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

NIM_API_URL = "https://integrate.api.nvidia.com/v1"
NIM_API_KEY = "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"

# ─── Functions ───

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

def zscore_classify(theta, mu, sigma, k=2.0):
    if sigma == 0: return False
    return (theta - mu) / sigma < -k

# ─── Collect data ───

def collect_data():
    """Run all measurements and return structured results."""
    np.random.seed(42)

    with open('data/obf_trigger.json') as f:
        obf_all = json.load(f)
    with open('data/obf_benign.json') as f:
        ben_all = json.load(f)

    # Latency sample
    lat_n = 20
    lat_texts = obf_all[:lat_n] + ben_all[:lat_n]

    # Full dataset
    obf_idx = np.random.choice(len(obf_all), size=100, replace=False)
    ben_idx = np.random.choice(len(ben_all), size=200, replace=False)
    obf_full = [obf_all[i] for i in obf_idx]
    ben_full = [ben_all[i] for i in ben_idx]

    models = ["nvidia/nv-embedqa-e5-v5", "nvidia/llama-nemotron-embed-1b-v2"]
    results = {}

    for model in models:
        short = model.split('/')[-1]
        print(f"Processing {short}...")

        # ── Latency ──
        delta_times = []
        entropy_times = []
        regex_times = []
        for _ in range(3):
            for text in lat_texts:
                t0 = time.perf_counter()
                chunks = chunk_input(text)
                embeddings = get_embeddings(chunks, model)
                if len(embeddings) >= 2:
                    angles = [compute_angle(embeddings[j-1], embeddings[j]) for j in range(1, len(embeddings))]
                    temp = 0.5
                    # Favor lower angles: negate before softmax
                    min_a = min(angles)
                    exp_vals = [math.exp((min_a - a) / temp) for a in angles]
                    sum_exp = sum(exp_vals)
                    weights = [e / sum_exp for e in exp_vals]
                    _ = sum(w * a for w, a in zip(weights, angles))
                t1 = time.perf_counter()
                delta_times.append((t1 - t0) * 1000)

                t0 = time.perf_counter()
                _ = char_entropy(text)
                t1 = time.perf_counter()
                entropy_times.append((t1 - t0) * 1000)

                t0 = time.perf_counter()
                _ = regex_score(text)
                t1 = time.perf_counter()
                regex_times.append((t1 - t0) * 1000)

        # ── Deltas for full dataset ──
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

        all_deltas = np.array(all_deltas)
        obf_d = all_deltas[:100]
        ben_d = all_deltas[100:]
        mu_b, sig_b = float(np.mean(ben_d)), float(np.std(ben_d))

        results[short] = {
            'delta_ms': delta_times,
            'entropy_ms': entropy_times,
            'regex_ms': regex_times,
            'obf_deltas': obf_d,
            'ben_deltas': ben_d,
            'mu_benign': mu_b,
            'sigma_benign': sig_b,
        }

    return results

# ─── Graphs ───

def graph_latency(results):
    """Bar chart: latency per method, grouped by model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = list(results.keys())
    short_names = [n.replace('nv-', '').replace('-v5', '').replace('-1b-v2', '') for n in model_names]
    methods = ['delta_ms', 'entropy_ms', 'regex_ms']
    method_labels = ['Delta Angle\n(API + compute)', 'Character\nEntropy', 'Regex\nPattern']
    colors = ['#9b59b6', '#3498db', '#2ecc71']

    x = np.arange(len(method_labels))
    width = 0.35

    for i, (model, sname) in enumerate(zip(model_names, short_names)):
        means = [np.mean(results[model][m]) for m in methods]
        stds = [np.std(results[model][m]) for m in methods]
        bars = ax.bar(x + i * width, means, width, yerr=stds, label=sname,
                      color=colors, alpha=0.8, edgecolor='white', linewidth=0.5,
                      error_kw={'capsize': 4, 'capthick': 1.5})
        # Add value labels
        for bar, mean in zip(bars, means):
            if mean > 10:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                       f'{mean:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                       f'{mean:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Time per input (ms)', fontsize=13)
    ax.set_title('Per-Input Latency by Detection Method', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(method_labels, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_yscale('log')
    ax.set_ylim(0.01, 2000)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    plt.savefig('graphs/latency_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved graphs/latency_comparison.png")
    plt.close()

def graph_latency_breakdown(results):
    """Stacked bar: delta angle breakdown (chunk + API + compute)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    model_names = list(results.keys())
    short_names = [n.replace('nv-', '').replace('-v5', '').replace('-1b-v2', '') for n in model_names]

    # These are approximate from the benchmark (API-dominated)
    # chunk ~0.05ms, compute ~0.4-0.7ms, rest is API
    chunk_ms = [0.05, 0.05]
    compute_ms = [0.39, 0.74]
    api_ms = [754.38, 631.89]

    x = np.arange(len(short_names))
    width = 0.5

    b1 = ax.bar(x, chunk_ms, width, label='Chunking', color='#3498db', alpha=0.9)
    b2 = ax.bar(x, compute_ms, width, bottom=chunk_ms, label='Angle computation', color='#2ecc71', alpha=0.9)
    b3 = ax.bar(x, api_ms, width, bottom=[c+m for c,m in zip(chunk_ms, compute_ms)],
                label='Embedding API call', color='#e74c3c', alpha=0.9)

    ax.set_ylabel('Time per input (ms)', fontsize=13)
    ax.set_title('Delta Angle Latency Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_yscale('log')
    ax.set_ylim(0.01, 2000)
    ax.grid(axis='y', alpha=0.3)

    # Annotate API dominance
    for i, api in enumerate(api_ms):
        ax.text(i, api * 1.3, f'{api:.0f} ms\n(API)', ha='center', va='bottom',
               fontsize=10, color='#c0392b', fontweight='bold')

    plt.tight_layout()
    plt.savefig('graphs/latency_breakdown.png', dpi=150, bbox_inches='tight')
    print("Saved graphs/latency_breakdown.png")
    plt.close()

def graph_zscore_within(results):
    """Line chart: within-model F1 at different k values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (model, data) in enumerate(results.items()):
        ax = axes[idx]
        obf_d = data['obf_deltas']
        ben_d = data['ben_deltas']
        mu_b = data['mu_benign']
        sig_b = data['sigma_benign']

        k_values = np.arange(0.5, 4.1, 0.25)
        f1s = []
        recalls = []
        fprs = []

        for k in k_values:
            tp = sum(1 for d in obf_d if zscore_classify(d, mu_b, sig_b, k))
            fp = sum(1 for d in ben_d if zscore_classify(d, mu_b, sig_b, k))
            fn = sum(1 for d in obf_d if not zscore_classify(d, mu_b, sig_b, k))
            tn = sum(1 for d in ben_d if not zscore_classify(d, mu_b, sig_b, k))
            prec = tp/(tp+fp) if (tp+fp)>0 else 0
            rec = tp/(tp+fn) if (tp+fn)>0 else 0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
            fpr = fp/(fp+tn) if (fp+tn)>0 else 0
            f1s.append(f1)
            recalls.append(rec)
            fprs.append(fpr)

        ax.plot(k_values, f1s, 'o-', color='#9b59b6', linewidth=2.5, markersize=6, label='F1')
        ax.plot(k_values, recalls, 's--', color='#2ecc71', linewidth=2, markersize=5, label='Recall')
        ax.plot(k_values, fprs, '^--', color='#e74c3c', linewidth=2, markersize=5, label='FPR')

        # Mark best F1
        best_k = k_values[np.argmax(f1s)]
        best_f1 = max(f1s)
        ax.axvline(x=best_k, color='gray', linestyle=':', alpha=0.5)
        ax.annotate(f'k={best_k:.1f}\nF1={best_f1:.3f}',
                   xy=(best_k, best_f1), xytext=(best_k+0.5, best_f1-0.05),
                   fontsize=10, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='gray'))

        short = model.replace('nv-', '').replace('-v5', '').replace('-1b-v2', '')
        ax.set_xlabel('k (z-score threshold)', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{short}\n(μ benign={mu_b:.3f}, σ={sig_b:.3f})', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(alpha=0.3)

    plt.suptitle('Z-Score Threshold Sensitivity (Within-Model)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('graphs/zscore_within_model.png', dpi=150, bbox_inches='tight')
    print("Saved graphs/zscore_within_model.png")
    plt.close()

def graph_zscore_cross(results):
    """Heatmap: cross-model F1 at different k values."""
    models = list(results.keys())
    if len(models) != 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    k_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # Train on model A, test on model B
    pairs = [(0, 1), (1, 0)]

    for plot_idx, (train_idx, test_idx) in enumerate(pairs):
        train_model = models[train_idx]
        test_model = models[test_idx]
        mu_b = results[train_model]['mu_benign']
        sig_b = results[train_model]['sigma_benign']
        obf_d = results[test_model]['obf_deltas']
        ben_d = results[test_model]['ben_deltas']

        recalls = []
        fprs = []
        f1s = []

        for k in k_values:
            tp = sum(1 for d in obf_d if zscore_classify(d, mu_b, sig_b, k))
            fp = sum(1 for d in ben_d if zscore_classify(d, mu_b, sig_b, k))
            fn = sum(1 for d in obf_d if not zscore_classify(d, mu_b, sig_b, k))
            tn = sum(1 for d in ben_d if not zscore_classify(d, mu_b, sig_b, k))
            prec = tp/(tp+fp) if (tp+fp)>0 else 0
            rec = tp/(tp+fn) if (tp+fn)>0 else 0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
            fpr = fp/(fp+tn) if (fp+tn)>0 else 0
            recalls.append(rec)
            fprs.append(fpr)
            f1s.append(f1)

        ax = axes[plot_idx]
        x = np.arange(len(k_values))
        width = 0.25

        bars1 = ax.bar(x - width, f1s, width, label='F1', color='#9b59b6', alpha=0.85)
        bars2 = ax.bar(x, recalls, width, label='Recall', color='#2ecc71', alpha=0.85)
        bars3 = ax.bar(x + width, fprs, width, label='FPR', color='#e74c3c', alpha=0.85)

        # Value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                h = bar.get_height()
                if h > 0.01:
                    ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                           f'{h:.2f}', ha='center', va='bottom', fontsize=8)

        train_short = train_model.split('/')[-1].replace('nv-', '').replace('-v5', '').replace('-1b-v2', '')
        test_short = test_model.split('/')[-1].replace('nv-', '').replace('-v5', '').replace('-1b-v2', '')

        ax.set_xticks(x)
        ax.set_xticklabels([f'k={k}' for k in k_values], fontsize=10)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Train: {train_short} → Test: {test_short}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Cross-Model Z-Score Portability', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('graphs/zscore_cross_model.png', dpi=150, bbox_inches='tight')
    print("Saved graphs/zscore_cross_model.png")
    plt.close()

def graph_delta_distributions(results):
    """Overlapping histograms: delta distributions for obf vs benign, both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (model, data) in enumerate(results.items()):
        ax = axes[idx]
        obf_d = data['obf_deltas']
        ben_d = data['ben_deltas']

        ax.hist(ben_d, bins=25, alpha=0.6, label='Benign', color='#3498db', density=True, edgecolor='white')
        ax.hist(obf_d, bins=25, alpha=0.6, label='Obfuscation', color='#e74c3c', density=True, edgecolor='white')

        mu_b = data['mu_benign']
        sig_b = data['sigma_benign']
        mu_o = np.mean(obf_d)
        sig_o = np.std(obf_d)

        # Mark overlap zone (95% ranges)
        obf_95_high = mu_o + 2 * sig_o
        ben_95_low = mu_b - 2 * sig_b
        overlap_lo = max(0, ben_95_low)
        overlap_hi = obf_95_high
        if overlap_hi > overlap_lo:
            ax.axvspan(overlap_lo, overlap_hi, alpha=0.15, color='yellow', label=f'Overlap zone')
            margin = ben_95_low - obf_95_high
            ax.annotate(f'margin={margin:.3f}',
                       xy=((overlap_lo + overlap_hi)/2, 0.15), fontsize=9,
                       ha='center', color='#8b6914', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

        # Mark z-score thresholds
        for k, ls in [(1.0, '--'), (2.0, '-')]:
            thresh = mu_b - k * sig_b
            ax.axvline(x=thresh, color='black', linestyle=ls, alpha=0.7, linewidth=1.5,
                      label=f'k={k} (θ={thresh:.3f})')

        short = model.replace('nv-', '').replace('-v5', '').replace('-1b-v2', '')
        ax.set_xlabel('Delta Angle', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{short}\n(σ benign={sig_b:.3f}, σ obf={sig_o:.3f})', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)

    plt.suptitle('Delta Angle Distributions with Z-Score Thresholds', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('graphs/delta_distributions_zscore.png', dpi=150, bbox_inches='tight')
    print("Saved graphs/delta_distributions_zscore.png")
    plt.close()

def main():
    import os
    os.makedirs('graphs', exist_ok=True)

    print("Collecting data (this takes a few minutes)...")
    results = collect_data()

    print("\nGenerating graphs...")
    graph_latency(results)
    graph_latency_breakdown(results)
    graph_zscore_within(results)
    graph_zscore_cross(results)
    graph_delta_distributions(results)

    print("\nDone! All graphs saved to graphs/")

if __name__ == '__main__':
    main()
