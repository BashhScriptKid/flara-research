"""
Token-based model comparison graphs.
"""
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# Load results
results = json.load(open('data/token_based_all_models.json'))

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# Color scheme
COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

os.makedirs('graphs', exist_ok=True)

# ============================================================================
# Graph 1: AUC Comparison Bar Chart
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

models = sorted(results.keys(), key=lambda x: -results[x]['auc'])
short_names = [m.split('/')[-1] for m in models]
aucs = [results[m]['auc'] for m in models]

bars = ax.barh(range(len(models)), aucs, color=COLORS[:len(models)], edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(short_names, fontsize=11)
ax.set_xlabel('AUC-ROC', fontsize=12)
ax.set_title('Token-Based Delta Angle — Model Comparison (AUC)', fontsize=14, fontweight='bold')
ax.set_xlim(0.4, 1.0)

# Add value labels
for i, (bar, auc) in enumerate(zip(bars, aucs)):
    ax.text(auc + 0.01, i, f'{auc:.3f}', va='center', fontsize=11, fontweight='bold')

# Add vertical line for E5 (baseline)
e5_idx = models.index('nvidia/nv-embedqa-e5-v5')
ax.axvline(aucs[e5_idx], color='gray', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('graphs/token_model_comparison_auc.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved graphs/token_model_comparison_auc.png')

# ============================================================================
# Graph 2: Separation Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

seps = [results[m]['separation'] for m in models]
bars = ax.barh(range(len(models)), seps, color=COLORS[:len(models)], edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(short_names, fontsize=11)
ax.set_xlabel('Separation (degrees)', fontsize=12)
ax.set_title('Token-Based Delta Angle — Model Comparison (Separation)', fontsize=14, fontweight='bold')

for i, (bar, sep) in enumerate(zip(bars, seps)):
    ax.text(sep + 0.1, i, f'{sep:.3f}°', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('graphs/token_model_comparison_sep.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved graphs/token_model_comparison_sep.png')

# ============================================================================
# Graph 3: FPR vs TPR Scatter
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

for i, model in enumerate(models):
    r = results[model]
    ax.scatter(r['optimal_fpr'], r['optimal_tpr'], s=200, color=COLORS[i], 
              label=short_names[i], edgecolors='black', linewidth=0.5, zorder=5)
    ax.annotate(short_names[i], (r['optimal_fpr'], r['optimal_tpr']), 
               xytext=(10, 5), textcoords='offset points', fontsize=10)

# Add ideal point
ax.scatter(0, 1, s=150, color='gold', marker='*', edgecolors='black', linewidth=0.5, zorder=5)
ax.annotate('Ideal', (0, 1), xytext=(10, -5), textcoords='offset points', fontsize=10)

ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax.set_ylabel('True Positive Rate (TPR/Recall)', fontsize=12)
ax.set_title('Token-Based Delta — FPR vs TPR Operating Points', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('graphs/token_model_fpr_tpr.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved graphs/token_model_fpr_tpr.png')

# ============================================================================
# Graph 4: F1 Score Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

f1s = [results[m]['best_f1'] for m in models]
bars = ax.barh(range(len(models)), f1s, color=COLORS[:len(models)], edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(short_names, fontsize=11)
ax.set_xlabel('Best F1 Score', fontsize=12)
ax.set_title('Token-Based Delta Angle — Model Comparison (F1)', fontsize=14, fontweight='bold')

for i, (bar, f1) in enumerate(zip(bars, f1s)):
    ax.text(f1 + 0.01, i, f'{f1:.3f}', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('graphs/token_model_comparison_f1.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved graphs/token_model_comparison_f1.png')

# ============================================================================
# Graph 5: Summary Table
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

# Table data
headers = ['Model', 'AUC', 'Separation', 'F1', 'FPR', 'TPR', 'Time']
cell_data = []
for m in models:
    r = results[m]
    cell_data.append([
        m.split('/')[-1],
        f"{r['auc']:.3f}",
        f"{r['separation']:.3f}°",
        f"{r['best_f1']:.3f}",
        f"{r['optimal_fpr']:.3f}",
        f"{r['optimal_tpr']:.3f}",
        f"{r['time']:.0f}s"
    ])

table = ax.table(cellText=cell_data, colLabels=headers, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Color header
for j in range(len(headers)):
    table[0, j].set_facecolor('#34495e')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Highlight best rows
best_auc_idx = models.index(max(results.keys(), key=lambda x: results[x]['auc']))
best_f1_idx = models.index(max(results.keys(), key=lambda x: results[x]['best_f1']))
for j in range(len(headers)):
    table[best_auc_idx + 1, j].set_facecolor('#d5f5e3')
    table[best_f1_idx + 1, j].set_facecolor('#d6eaf8')

ax.set_title('Token-Based Delta Angle — Complete Model Comparison', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('graphs/token_model_summary_table.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved graphs/token_model_summary_table.png')

print('\nAll graphs saved to graphs/')
