#!/usr/bin/env python3
"""
Compare threshold techniques: z-score, percentile, robust z-score (MAD), quantile transform.
All from cache — no API calls.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import rankdata

os.makedirs('graphs', exist_ok=True)

with open('data/delta_cache.json') as f:
    cache = json.load(f)

models = list(cache['models'].keys())
data = {}
for m in models:
    d = cache['models'][m]
    all_d = np.array(d['deltas'])
    data[m] = {
        'obf': all_d[:cache['n_obf']],
        'ben': all_d[cache['n_obf']:],
        'mu_b': d['ben_mean'],
        'sig_b': d['ben_std'],
    }

# ─── Threshold functions ───

def zscore_flag(deltas, mu, sigma, k):
    """Classic z-score: flag if (θ - μ) / σ < -k"""
    if sigma == 0: return np.zeros(len(deltas), dtype=bool)
    return ((deltas - mu) / sigma) < -k

def percentile_flag(deltas, ben_deltas, p):
    """Percentile: flag if below p-th percentile of benign."""
    threshold = np.percentile(ben_deltas, p)
    return deltas < threshold

def mad_flag(deltas, ben_deltas, k):
    """Robust z-score (MAD): flag if (θ - median) / (1.4826 * MAD) < -k"""
    median = np.median(ben_deltas)
    mad = np.median(np.abs(ben_deltas - median))
    mad_scaled = 1.4826 * mad  # scale to match std for normal distributions
    if mad_scaled == 0: return np.zeros(len(deltas), dtype=bool)
    return ((deltas - median) / mad_scaled) < -k

def quantile_flag(deltas, ben_deltas, threshold_uniform):
    """Quantile transform: map to [0,1] using benign empirical CDF, flag if < threshold."""
    # For each delta, compute its rank among benign
    # P(flag) = fraction of benign below this value
    all_combined = np.concatenate([ben_deltas, deltas])
    ranks = rankdata(all_combined, method='average')
    # benign ranks are first len(ben_deltas)
    # delta ranks are the rest
    ben_ranks = ranks[:len(ben_deltas)]
    delta_ranks = ranks[len(ben_deltas):]
    # Normalize to [0,1] using ben_ranks as reference
    # For each delta, compute fraction of benign <= it
    ben_sorted = np.sort(ben_deltas)
    probs = np.searchsorted(ben_sorted, deltas, side='left') / len(ben_sorted)
    return probs < threshold_uniform

# ─── Evaluate all methods ───

def evaluate(obf, ben, flag_fn):
    tp = np.sum(flag_fn(obf))
    fp = np.sum(flag_fn(ben))
    fn = np.sum(~flag_fn(obf))
    tn = np.sum(~flag_fn(ben))
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0
    return {'recall': rec, 'fpr': fpr, 'f1': f1, 'precision': prec, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}

# ─── Run comparison ───

all_results = {}

for m in models:
    obf = data[m]['obf']
    ben = data[m]['ben']
    mu_b = data[m]['mu_b']
    sig_b = data[m]['sig_b']
    short = m.replace('nv-', '').replace('-v5', '').replace('-1b-v2', '')

    print(f"\n{'='*60}")
    print(f"{short}")
    print(f"{'='*60}")

    results = []

    # 1. Z-score (baseline)
    for k in [1.0, 1.5, 2.0, 2.5]:
        res = evaluate(obf, ben, lambda d, k=k: zscore_flag(d, mu_b, sig_b, k))
        res['method'] = f'z-score'
        res['param'] = f'k={k}'
        results.append(res)

    # 2. Percentile
    for p in [1, 2, 5, 10]:
        res = evaluate(obf, ben, lambda d, p=p: percentile_flag(d, ben, p))
        res['method'] = 'percentile'
        res['param'] = f'p={p}%'
        results.append(res)

    # 3. Robust z-score (MAD)
    for k in [1.0, 1.5, 2.0, 2.5]:
        res = evaluate(obf, ben, lambda d, k=k: mad_flag(d, ben, k))
        res['method'] = 'MAD'
        res['param'] = f'k={k}'
        results.append(res)

    # 4. Quantile transform
    for t in [0.01, 0.02, 0.05, 0.10]:
        res = evaluate(obf, ben, lambda d, t=t: quantile_flag(d, ben, t))
        res['method'] = 'quantile'
        res['param'] = f't={t}'
        results.append(res)

    # Print table
    print(f"{'Method':<12} {'Param':<8} {'Recall':<8} {'FPR':<8} {'F1':<8}")
    print("-" * 50)
    for r in results:
        print(f"{r['method']:<12} {r['param']:<8} {r['recall']:<8.3f} {r['fpr']:<8.3f} {r['f1']:<8.3f}")

    all_results[m] = results

# ─── Graph: F1 comparison across methods ───

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, m in enumerate(models):
    ax = axes[idx]
    results = all_results[m]
    short = m.replace('nv-', '').replace('-v5', '').replace('-1b-v2', '')

    methods = ['z-score', 'percentile', 'MAD', 'quantile']
    colors = {'z-score': '#9b59b6', 'percentile': '#3498db', 'MAD': '#2ecc71', 'quantile': '#e67e22'}

    for method in methods:
        method_results = [r for r in results if r['method'] == method]
        params = [r['param'] for r in method_results]
        f1s = [r['f1'] for r in method_results]
        recalls = [r['recall'] for r in method_results]
        fprs = [r['fpr'] for r in method_results]

        x = np.arange(len(params))
        ax.plot(x, f1s, 'o-', color=colors[method], linewidth=2, markersize=8, label=f'{method} (F1)')
        ax.plot(x, recalls, 's--', color=colors[method], linewidth=1.5, markersize=5, alpha=0.6, label=f'{method} (Recall)')
        ax.plot(x, fprs, '^:', color=colors[method], linewidth=1.5, markersize=5, alpha=0.4, label=f'{method} (FPR)')

        # Annotate best F1 per method
        best_i = np.argmax(f1s)
        ax.annotate(f'{f1s[best_i]:.3f}', xy=(best_i, f1s[best_i]),
                   xytext=(best_i, f1s[best_i] + 0.05), fontsize=9, fontweight='bold',
                   ha='center', color=colors[method])

    ax.set_xticks(np.arange(len([r['param'] for r in results])))
    ax.set_xticklabels([r['param'] for r in results], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{short}', fontsize=13, fontweight='bold')
    ax.set_ylim(-0.05, 1.15)
    ax.grid(alpha=0.3)

    # Custom legend — only show method names
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[m], linewidth=2, label=m) for m in methods]
    ax.legend(handles=legend_elements, fontsize=10, loc='lower left')

plt.suptitle('Threshold Technique Comparison: F1 / Recall / FPR', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graphs/threshold_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nSaved graphs/threshold_comparison.png")
plt.close()

# ─── Graph: Best operating point per method ───

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, m in enumerate(models):
    ax = axes[idx]
    results = all_results[m]
    short = m.replace('nv-', '').replace('-v5', '').replace('-1b-v2', '')

    methods = ['z-score', 'percentile', 'MAD', 'quantile']
    colors = {'z-score': '#9b59b6', 'percentile': '#3498db', 'MAD': '#2ecc71', 'quantile': '#e67e22'}

    # For each method, pick the operating point with best F1
    best_per_method = []
    for method in methods:
        method_results = [r for r in results if r['method'] == method]
        best = max(method_results, key=lambda r: r['f1'])
        best['method'] = method
        best_per_method.append(best)

    names = [r['method'] for r in best_per_method]
    f1s = [r['f1'] for r in best_per_method]
    recalls = [r['recall'] for r in best_per_method]
    fprs = [r['fpr'] for r in best_per_method]

    x = np.arange(len(names))
    w = 0.25

    b1 = ax.bar(x - w, f1s, w, label='F1', color=[colors[n] for n in names], alpha=0.9)
    b2 = ax.bar(x, recalls, w, label='Recall', color=[colors[n] for n in names], alpha=0.6)
    b3 = ax.bar(x + w, fprs, w, label='FPR', color=[colors[n] for n in names], alpha=0.35, edgecolor=[colors[n] for n in names], linewidth=1.5)

    for bars, vals in [(b1, f1s), (b2, recalls), (b3, fprs)]:
        for bar, v in zip(bars, vals):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., v + 0.015,
                       f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{short} — Best operating point per method', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Best F1 Operating Point by Threshold Technique', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graphs/threshold_best_points.png', dpi=150, bbox_inches='tight')
print(f"Saved graphs/threshold_best_points.png")
plt.close()

# ─── Summary: which method wins? ───

print(f"\n{'='*60}")
print("SUMMARY: Best F1 per method")
print(f"{'='*60}")
for m in models:
    short = m.replace('nv-', '').replace('-v5', '').replace('-1b-v2', '')
    results = all_results[m]
    methods = ['z-score', 'percentile', 'MAD', 'quantile']
    print(f"\n{short}:")
    for method in methods:
        method_results = [r for r in results if r['method'] == method]
        best = max(method_results, key=lambda r: r['f1'])
        print(f"  {method:<12} best F1={best['f1']:.3f}  ({best['param']}, recall={best['recall']:.3f}, FPR={best['fpr']:.3f})")
