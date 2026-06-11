#!/usr/bin/env python3
"""
Clean benchmark graphs for delta angle analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

os.makedirs("benchmark_data/graphs", exist_ok=True)

# Load data
df = pd.read_csv("benchmark_data/combined_results.csv")
df['model_short'] = df['model'].map({
    'nv-embedqa-e5-v5': 'E5-v5 (130M)',
    'llama-nemotron-1b-v2': 'Nemotron (1B)'
})

# --- Graph 1: Simple box plot ---
fig, ax = plt.subplots(figsize=(8, 5))

# Separate data
triggers = df[df['label'] == 'trigger']
benigns = df[df['label'] == 'benign']
models = ['E5-v5 (130M)', 'Nemotron (1B)']

x = np.arange(len(models))
width = 0.35

t_means = [triggers[triggers['model_short'] == m]['delta'].mean() for m in models]
t_stds = [triggers[triggers['model_short'] == m]['delta'].std() for m in models]
b_means = [benigns[benigns['model_short'] == m]['delta'].mean() for m in models]
b_stds = [benigns[benigns['model_short'] == m]['delta'].std() for m in models]

bars1 = ax.bar(x - width/2, t_means, width, label='Injection', color='#ff6b6b', yerr=t_stds, capsize=5, alpha=0.8)
bars2 = ax.bar(x + width/2, b_means, width, label='Normal', color='#4ecdc4', yerr=b_stds, capsize=5, alpha=0.8)

ax.set_ylabel('Average Delta Angle')
ax.set_title('Delta Angle by Model and Class')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 1.3)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.3f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("benchmark_data/graphs/01_comparison.png")
plt.close()
print("Saved: 01_comparison.png")

# --- Graph 2: Scatter plot with clear labels ---
fig, ax = plt.subplots(figsize=(8, 8))

for model, color in [('E5-v5 (130M)', '#e74c3c'), ('Nemotron (1B)', '#3498db')]:
    subset = df[df['model_short'] == model]
    t = subset[subset['label'] == 'trigger']
    b = subset[subset['label'] == 'benign']
    ax.scatter(t['delta'], t['risk'], c=color, marker='o', s=80, alpha=0.6, label=f'{model} - Injection', edgecolors='black', linewidth=0.5)
    ax.scatter(b['delta'], b['risk'], c=color, marker='s', s=80, alpha=0.6, label=f'{model} - Normal', edgecolors='black', linewidth=0.5)

ax.set_xlabel('Delta Angle')
ax.set_ylabel('Risk Score')
ax.set_title('Delta Angle vs Risk Score')
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("benchmark_data/graphs/02_scatter.png")
plt.close()
print("Saved: 02_scatter.png")

# --- Graph 3: Histogram overlay ---
fig, ax = plt.subplots(figsize=(8, 5))

for model, color in [('E5-v5 (130M)', '#e74c3c'), ('Nemotron (1B)', '#3498db')]:
    t = df[(df['model_short'] == model) & (df['label'] == 'trigger')]['delta']
    b = df[(df['model_short'] == model) & (df['label'] == 'benign')]['delta']
    ax.hist(t, bins=15, alpha=0.5, color=color, label=f'{model} Injection', density=True)
    ax.hist(b, bins=15, alpha=0.3, color=color, label=f'{model} Normal', density=True, hatch='//')

ax.set_xlabel('Delta Angle')
ax.set_ylabel('Density')
ax.set_title('Delta Angle Distribution')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("benchmark_data/graphs/03_histogram.png")
plt.close()
print("Saved: 03_histogram.png")

print("\nDone. Upload to imgur.")
