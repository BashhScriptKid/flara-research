#!/usr/bin/env python3
"""
Final graph suite: distributions, risk vs delta scatter, correlation matrix, latency, ROC/FPR-FNR.
All from cache + existing CSVs. No API calls.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re, math, os

os.makedirs('graphs', exist_ok=True)

# ─── Load data ───

with open('data/delta_cache.json') as f:
    cache = json.load(f)

models = list(cache['models'].keys())

# Compute risk score from available features
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

texts = cache['texts']
labels = cache['labels']

# Build per-sample feature frame
rows = []
for i, text in enumerate(texts):
    ent = char_entropy(text)
    scr = special_char_ratio(text)
    rs = regex_score(text)
    for m in models:
        delta = cache['models'][m]['deltas'][i]
        # Risk score: normalized average of available features
        # Normalize each to [0,1] using observed range
        rows.append({
            'label': labels[i],
            'model': m,
            'delta': delta,
            'entropy': ent,
            'special_ratio': scr,
            'regex': rs,
        })

df = pd.DataFrame(rows)

# Compute risk score per model (delta is model-specific, others are shared)
for m in models:
    mask = df['model'] == m
    sub = df[mask]
    for col in ['entropy', 'special_ratio', 'regex']:
        lo, hi = sub[col].min(), sub[col].max()
        df.loc[mask, f'{col}_norm'] = (sub[col] - lo) / (hi - lo) if hi > lo else 0
    lo, hi = sub['delta'].min(), sub['delta'].max()
    df.loc[mask, 'delta_norm'] = (sub['delta'] - lo) / (hi - lo) if hi > lo else 0
    df.loc[mask, 'risk'] = (df.loc[mask, 'entropy_norm'] + df.loc[mask, 'special_ratio_norm'] +
                            df.loc[mask, 'regex_norm'] + df.loc[mask, 'delta_norm']) / 4

# ─── 1. Distributions ───

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, m in enumerate(models):
    ax = axes[idx]
    obf = df[(df['model'] == m) & (df['label'] == 'obfuscation')]['delta']
    ben = df[(df['model'] == m) & (df['label'] == 'benign')]['delta']
    short = m.replace('nv-', '').replace('-v5', '').replace('-1b-v2', '')

    ax.hist(ben, bins=25, alpha=0.6, label='Benign', color='#3498db', density=True, edgecolor='white')
    ax.hist(obf, bins=25, alpha=0.6, label='Obfuscation', color='#e74c3c', density=True, edgecolor='white')

    mu_b, sig_b = ben.mean(), ben.std()
    mu_o, sig_o = obf.mean(), obf.std()
    obf_95 = mu_o + 2*sig_o
    ben_95 = mu_b - 2*sig_b
    if obf_95 > ben_95:
        ax.axvspan(ben_95, obf_95, alpha=0.15, color='yellow', label='Overlap')
        ax.annotate(f'margin={ben_95 - obf_95:.3f}', xy=((ben_95+obf_95)/2, 0.15),
                   fontsize=9, ha='center', color='#8b6914', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax.set_xlabel('Signed Delta Angle', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{short}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('Delta Angle Distributions', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graphs/distributions.png', dpi=150, bbox_inches='tight')
print("Saved graphs/distributions.png")
plt.close()

# ─── 2. Scatter: risk vs delta ───

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for idx, m in enumerate(models):
    ax = axes[idx]
    sub = df[df['model'] == m]
    short = m.replace('nv-', '').replace('-v5', '').replace('-1b-v2', '')

    ben = sub[sub['label'] == 'benign']
    obf = sub[sub['label'] == 'obfuscation']

    ax.scatter(ben['delta'], ben['risk'], alpha=0.5, s=30, c='#3498db', label='Benign', edgecolors='white', linewidth=0.3)
    ax.scatter(obf['delta'], obf['risk'], alpha=0.5, s=30, c='#e74c3c', label='Obfuscation', edgecolors='white', linewidth=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    # Marginal means
    for label, color, yoff in [('benign', '#3498db', 0.03), ('obfuscation', '#e74c3c', -0.04)]:
        s = sub[sub['label'] == label]
        mx, my = s['delta'].mean(), s['risk'].mean()
        ax.plot(mx, my, 'D', color=color, markersize=12, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        ax.annotate(f'μ=({mx:.3f}, {my:.3f})', xy=(mx, my), xytext=(mx + 0.05, my + yoff),
                   fontsize=9, fontweight='bold', color=color,
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    ax.set_xlabel('Signed Delta Angle', fontsize=12)
    ax.set_ylabel('Risk Score', fontsize=12)
    ax.set_title(f'{short}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.suptitle('Risk Score vs Delta Angle', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graphs/scatter_risk_delta.png', dpi=150, bbox_inches='tight')
print("Saved graphs/scatter_risk_delta.png")
plt.close()

# ─── 3. Correlation matrix ───

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
features = ['delta', 'entropy', 'special_ratio', 'regex']
display = ['Delta', 'Entropy', 'Special %', 'Regex']

for idx, m in enumerate(models):
    ax = axes[idx]
    sub = df[df['model'] == m]
    corr = sub[features].corr()
    short = m.replace('nv-', '').replace('-v5', '').replace('-1b-v2', '')

    im = ax.imshow(corr.values, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(display)))
    ax.set_xticklabels(display, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(range(len(display)))
    ax.set_yticklabels(display, fontsize=10)

    for i in range(len(display)):
        for j in range(len(display)):
            v = corr.values[i, j]
            color = 'white' if abs(v) > 0.5 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=11, fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
    ax.set_title(f'{short}', fontsize=13, fontweight='bold')

plt.suptitle('Feature Correlation Matrix', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graphs/correlation_matrix.png', dpi=150, bbox_inches='tight')
print("Saved graphs/correlation_matrix.png")
plt.close()

# ─── Final list ───

print("\nGraphs folder:")
for f in sorted(os.listdir('graphs')):
    if f.endswith('.png'):
        print(f"  {f}")
