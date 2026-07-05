#!/usr/bin/env python3
"""
Scatter plot + correlation matrix from cached data. No API calls.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
import re, math, os

os.makedirs('graphs', exist_ok=True)

with open('data/delta_cache.json') as f:
    cache = json.load(f)

# ─── Compute features ───

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
models = list(cache['models'].keys())

rows = []
for i, text in enumerate(texts):
    for m in models:
        rows.append({
            'label': labels[i],
            'text_len': len(text),
            'entropy': char_entropy(text),
            'special_ratio': special_char_ratio(text),
            'regex': regex_score(text),
            'delta': cache['models'][m]['deltas'][i],
            'model': m,
        })

df = pd.DataFrame(rows)

# ─── 1. Scatter: delta vs entropy ───

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, m in enumerate(models):
    ax = axes[idx]
    sub = df[df['model'] == m]
    short = m.replace('nv-', '').replace('-v5', '').replace('-1b-v2', '')

    ben = sub[sub['label'] == 'benign']
    obf = sub[sub['label'] == 'obfuscation']

    ax.scatter(ben['entropy'], ben['delta'], alpha=0.5, s=30, c='#3498db', label='Benign', edgecolors='white', linewidth=0.3)
    ax.scatter(obf['entropy'], obf['delta'], alpha=0.5, s=30, c='#e74c3c', label='Obfuscation', edgecolors='white', linewidth=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    # Add marginal means
    for label, color, offset in [('benign', '#3498db', 0.03), ('obfuscation', '#e74c3c', -0.06)]:
        sub_l = sub[sub['label'] == label]
        mx, my = sub_l['entropy'].mean(), sub_l['delta'].mean()
        ax.plot(mx, my, 'D', color=color, markersize=12, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        ax.annotate(f'μ=({mx:.1f}, {my:.3f})', xy=(mx, my), xytext=(mx + 0.3, my + offset),
                   fontsize=9, fontweight='bold', color=color,
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    ax.set_xlabel('Character Entropy', fontsize=12)
    ax.set_ylabel('Signed Delta Angle', fontsize=12)
    ax.set_title(f'{short}\n(Δentropy={ben["entropy"].mean()-obf["entropy"].mean():.2f}, '
                 f'Δdelta={ben["delta"].mean()-obf["delta"].mean():.3f})',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.suptitle('Delta Angle vs Character Entropy', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graphs/scatter_delta_entropy.png', dpi=150, bbox_inches='tight')
print("Saved graphs/scatter_delta_entropy.png")
plt.close()

# ─── 2. Scatter: delta vs text_len ───

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, m in enumerate(models):
    ax = axes[idx]
    sub = df[df['model'] == m]
    short = m.replace('nv-', '').replace('-v5', '').replace('-1b-v2', '')

    ben = sub[sub['label'] == 'benign']
    obf = sub[sub['label'] == 'obfuscation']

    ax.scatter(ben['text_len'], ben['delta'], alpha=0.5, s=30, c='#3498db', label='Benign', edgecolors='white', linewidth=0.3)
    ax.scatter(obf['text_len'], obf['delta'], alpha=0.5, s=30, c='#e74c3c', label='Obfuscation', edgecolors='white', linewidth=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    ax.set_xlabel('Text Length (chars)', fontsize=12)
    ax.set_ylabel('Signed Delta Angle', fontsize=12)
    ax.set_title(f'{short}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.suptitle('Delta Angle vs Text Length', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graphs/scatter_delta_textlen.png', dpi=150, bbox_inches='tight')
print("Saved graphs/scatter_delta_textlen.png")
plt.close()

# ─── 3. Correlation matrix ───

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

features = ['delta', 'entropy', 'special_ratio', 'regex', 'text_len']
labels_display = ['Delta', 'Entropy', 'Special %', 'Regex', 'Text Len']

for idx, m in enumerate(models):
    ax = axes[idx]
    sub = df[df['model'] == m]
    corr = sub[features].corr()
    short = m.replace('nv-', '').replace('-v5', '').replace('-1b-v2', '')

    im = ax.imshow(corr.values, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels_display)))
    ax.set_xticklabels(labels_display, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(range(len(labels_display)))
    ax.set_yticklabels(labels_display, fontsize=10)

    for i in range(len(labels_display)):
        for j in range(len(labels_display)):
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

# ─── Print correlation summary ───

print("\nDelta angle correlations with other features:")
for m in models:
    short = m.replace('nv-', '').replace('-v5', '').replace('-1b-v2', '')
    sub = df[df['model'] == m]
    corr = sub[features].corr()['delta'].drop('delta')
    print(f"  {short}: " + ", ".join(f"{f}={v:.3f}" for f, v in corr.items()))
