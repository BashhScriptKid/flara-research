#!/usr/bin/env python3
"""
Graphs from cached delta data. No API calls.
Usage: python graphs_cached.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

os.makedirs('graphs', exist_ok=True)

with open('data/delta_cache.json') as f:
    cache = json.load(f)

models = list(cache['models'].keys())
results = {}
for m in models:
    d = cache['models'][m]
    all_d = np.array(d['deltas'])
    results[m] = {
        'obf_deltas': all_d[:cache['n_obf']],
        'ben_deltas': all_d[cache['n_obf']:],
        'mu_benign': d['ben_mean'],
        'sigma_benign': d['ben_std'],
        'mu_obf': d['obf_mean'],
        'sigma_obf': d['obf_std'],
    }

def zscore_classify(theta, mu, sigma, k=2.0):
    if sigma == 0: return False
    return (theta - mu) / sigma < -k

# ─── 1. Latency bar (hardcoded from benchmark — no API needed) ───

def graph_latency():
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ['Delta Angle\n(API + compute)', 'Character\nEntropy', 'Regex\nPattern']
    e5 = [754.82, 0.03, 0.04]
    nem = [632.68, 0.03, 0.04]
    x = np.arange(len(methods))
    w = 0.35
    b1 = ax.bar(x - w/2, e5, w, label='embedqa-e5', color='#9b59b6', alpha=0.85, yerr=[184.37, 0.03, 0.05], capsize=4)
    b2 = ax.bar(x + w/2, nem, w, label='llama-nemotron-embed', color='#8e44ad', alpha=0.65, yerr=[142.95, 0.04, 0.04], capsize=4)
    for bars, vals in [(b1, e5), (b2, nem)]:
        for bar, v in zip(bars, vals):
            label = f'{v:.0f}' if v > 10 else f'{v:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.4,
                   label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Time per input (ms)', fontsize=13)
    ax.set_title('Per-Input Latency by Detection Method', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_yscale('log')
    ax.set_ylim(0.01, 2000)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphs/latency_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved graphs/latency_comparison.png")
    plt.close()

# ─── 2. Latency breakdown ───

def graph_latency_breakdown():
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ['embedqa-e5', 'llama-nemotron-embed']
    chunk = [0.05, 0.05]
    compute = [0.39, 0.74]
    api = [754.38, 631.89]
    x = np.arange(len(names))
    w = 0.5
    ax.bar(x, chunk, w, label='Chunking', color='#3498db', alpha=0.9)
    ax.bar(x, compute, w, bottom=chunk, label='Angle computation', color='#2ecc71', alpha=0.9)
    ax.bar(x, api, w, bottom=[c+m for c,m in zip(chunk, compute)], label='Embedding API call', color='#e74c3c', alpha=0.9)
    for i, a in enumerate(api):
        ax.text(i, a * 1.3, f'{a:.0f} ms\n(API)', ha='center', va='bottom', fontsize=10, color='#c0392b', fontweight='bold')
    ax.set_ylabel('Time per input (ms)', fontsize=13)
    ax.set_title('Delta Angle Latency Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_yscale('log')
    ax.set_ylim(0.01, 2000)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphs/latency_breakdown.png', dpi=150, bbox_inches='tight')
    print("Saved graphs/latency_breakdown.png")
    plt.close()

# ─── 3. Z-score within-model ───

def graph_zscore_within():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (model, data) in enumerate(results.items()):
        ax = axes[idx]
        obf_d = data['obf_deltas']
        ben_d = data['ben_deltas']
        mu_b = data['mu_benign']
        sig_b = data['sigma_benign']

        k_values = np.arange(0.5, 4.1, 0.25)
        f1s, recalls, fprs = [], [], []
        for k in k_values:
            tp = sum(1 for d in obf_d if zscore_classify(d, mu_b, sig_b, k))
            fp = sum(1 for d in ben_d if zscore_classify(d, mu_b, sig_b, k))
            fn = sum(1 for d in obf_d if not zscore_classify(d, mu_b, sig_b, k))
            tn = sum(1 for d in ben_d if not zscore_classify(d, mu_b, sig_b, k))
            prec = tp/(tp+fp) if (tp+fp)>0 else 0
            rec = tp/(tp+fn) if (tp+fn)>0 else 0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
            fpr = fp/(fp+tn) if (fp+tn)>0 else 0
            f1s.append(f1); recalls.append(rec); fprs.append(fpr)

        ax.plot(k_values, f1s, 'o-', color='#9b59b6', linewidth=2.5, markersize=6, label='F1')
        ax.plot(k_values, recalls, 's--', color='#2ecc71', linewidth=2, markersize=5, label='Recall')
        ax.plot(k_values, fprs, '^--', color='#e74c3c', linewidth=2, markersize=5, label='FPR')

        best_k = k_values[np.argmax(f1s)]
        best_f1 = max(f1s)
        ax.axvline(x=best_k, color='gray', linestyle=':', alpha=0.5)
        ax.annotate(f'k={best_k:.1f}\nF1={best_f1:.3f}', xy=(best_k, best_f1),
                   xytext=(best_k+0.5, best_f1-0.05), fontsize=10, fontweight='bold',
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

# ─── 4. Cross-model ───

def graph_zscore_cross():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    k_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    pairs = [(0, 1), (1, 0)]

    for plot_idx, (train_idx, test_idx) in enumerate(pairs):
        train_m = models[train_idx]
        test_m = models[test_idx]
        mu_b = results[train_m]['mu_benign']
        sig_b = results[train_m]['sigma_benign']
        obf_d = results[test_m]['obf_deltas']
        ben_d = results[test_m]['ben_deltas']

        f1s, recalls, fprs = [], [], []
        for k in k_values:
            tp = sum(1 for d in obf_d if zscore_classify(d, mu_b, sig_b, k))
            fp = sum(1 for d in ben_d if zscore_classify(d, mu_b, sig_b, k))
            fn = sum(1 for d in obf_d if not zscore_classify(d, mu_b, sig_b, k))
            tn = sum(1 for d in ben_d if not zscore_classify(d, mu_b, sig_b, k))
            prec = tp/(tp+fp) if (tp+fp)>0 else 0
            rec = tp/(tp+fn) if (tp+fn)>0 else 0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
            fpr = fp/(fp+tn) if (fp+tn)>0 else 0
            f1s.append(f1); recalls.append(rec); fprs.append(fpr)

        ax = axes[plot_idx]
        x = np.arange(len(k_values))
        w = 0.25
        b1 = ax.bar(x - w, f1s, w, label='F1', color='#9b59b6', alpha=0.85)
        b2 = ax.bar(x, recalls, w, label='Recall', color='#2ecc71', alpha=0.85)
        b3 = ax.bar(x + w, fprs, w, label='FPR', color='#e74c3c', alpha=0.85)
        for bars in [b1, b2, b3]:
            for bar in bars:
                h = bar.get_height()
                if h > 0.01:
                    ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                           f'{h:.2f}', ha='center', va='bottom', fontsize=8)

        t_short = train_m.split('/')[-1].replace('nv-', '').replace('-v5', '').replace('-1b-v2', '')
        te_short = test_m.split('/')[-1].replace('nv-', '').replace('-v5', '').replace('-1b-v2', '')
        ax.set_xticks(x)
        ax.set_xticklabels([f'k={k}' for k in k_values], fontsize=10)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Train: {t_short} → Test: {te_short}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Cross-Model Z-Score Portability', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('graphs/zscore_cross_model.png', dpi=150, bbox_inches='tight')
    print("Saved graphs/zscore_cross_model.png")
    plt.close()

# ─── 5. Distributions with overlap ───

def graph_delta_distributions():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (model, data) in enumerate(results.items()):
        ax = axes[idx]
        obf_d = data['obf_deltas']
        ben_d = data['ben_deltas']

        ax.hist(ben_d, bins=25, alpha=0.6, label='Benign', color='#3498db', density=True, edgecolor='white')
        ax.hist(obf_d, bins=25, alpha=0.6, label='Obfuscation', color='#e74c3c', density=True, edgecolor='white')

        mu_b = data['mu_benign']
        sig_b = data['sigma_benign']
        mu_o = data['mu_obf']
        sig_o = data['sigma_obf']

        obf_95_high = mu_o + 2 * sig_o
        ben_95_low = mu_b - 2 * sig_b
        overlap_lo = max(0, ben_95_low)
        overlap_hi = obf_95_high
        if overlap_hi > overlap_lo:
            ax.axvspan(overlap_lo, overlap_hi, alpha=0.15, color='yellow', label='Overlap zone')
            margin = ben_95_low - obf_95_high
            ax.annotate(f'margin={margin:.3f}', xy=((overlap_lo+overlap_hi)/2, 0.15),
                       fontsize=9, ha='center', color='#8b6914', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

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

# ─── Run all ───

print("Generating graphs from cache (no API calls)...")
graph_latency()
graph_latency_breakdown()
graph_zscore_within()
graph_zscore_cross()
graph_delta_distributions()
print("Done!")
