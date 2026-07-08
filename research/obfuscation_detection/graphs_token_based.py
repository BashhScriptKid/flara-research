"""
Token-based delta angle analysis graphs.
Comparing sentence chunking vs token-based approach.
"""
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# Load token-based results
token_results = json.load(open('data/delta_cache_token_full.json'))

# Extract deltas by label
deltas_by_label = defaultdict(list)
for key, r in token_results.items():
    deltas_by_label[r['label']].append(r['delta'])

ben = np.array(deltas_by_label['benign'])
obf = np.array(deltas_by_label['obfuscation'])

print(f'BENIGN: μ={np.mean(ben):.3f}° σ={np.std(ben):.3f}° n={len(ben)}')
print(f'OBFUSCATION: μ={np.mean(obf):.3f}° σ={np.std(obf):.3f}° n={len(obf)}')
print(f'Separation: {abs(np.mean(ben) - np.mean(obf)):.3f}°')

# Create output directory
os.makedirs('graphs', exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# Color scheme
COLOR_OBF = '#e74c3c'
COLOR_BEN = '#2ecc71'
COLOR_SEP = '#3498db'

# ============================================================================
# Graph 1: Distribution comparison (token-based)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax = axes[0]
bins = np.linspace(0, 45, 40)
ax.hist(ben, bins=bins, alpha=0.7, color=COLOR_BEN, label=f'Benign (μ={np.mean(ben):.2f}°)', density=True, edgecolor='white', linewidth=0.5)
ax.hist(obf, bins=bins, alpha=0.7, color=COLOR_OBF, label=f'Obfuscation (μ={np.mean(obf):.2f}°)', density=True, edgecolor='white', linewidth=0.5)

# Threshold line (95th percentile of benign)
threshold = np.percentile(ben, 95)
ax.axvline(threshold, color=COLOR_SEP, linestyle='--', linewidth=2, label=f'Threshold (95th pct): {threshold:.1f}°')
ax.axvline(np.mean(ben), color=COLOR_BEN, linestyle=':', linewidth=2, alpha=0.7)
ax.axvline(np.mean(obf), color=COLOR_OBF, linestyle=':', linewidth=2, alpha=0.7)

ax.set_xlabel('Token Delta Angle (degrees)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Token-Based Delta Distribution', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim(0, 45)

# Cumulative distribution
ax = axes[1]
sorted_ben = np.sort(ben)
sorted_obf = np.sort(obf)
cdf_ben = np.arange(1, len(sorted_ben) + 1) / len(sorted_ben)
cdf_obf = np.arange(1, len(sorted_obf) + 1) / len(sorted_obf)

ax.plot(sorted_ben, cdf_ben, color=COLOR_BEN, linewidth=2.5, label='Benign')
ax.plot(sorted_obf, cdf_obf, color=COLOR_OBF, linewidth=2.5, label='Obfuscation')
ax.axvline(threshold, color=COLOR_SEP, linestyle='--', linewidth=2, alpha=0.7, label=f'Threshold: {threshold:.1f}°')
ax.axhline(0.95, color='gray', linestyle=':', linewidth=1, alpha=0.5)

ax.set_xlabel('Token Delta Angle (degrees)', fontsize=12)
ax.set_ylabel('Cumulative Probability', fontsize=12)
ax.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(0, 45)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('graphs/token_based_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved graphs/token_based_distribution.png')

# ============================================================================
# Graph 2: ROC Curve
# ============================================================================
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

y_true = [0]*len(ben) + [1]*len(obf)
# Negated because obf < ben
y_scores = [-d for d in list(ben) + list(obf)]

fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
auc = roc_auc_score(y_true, y_scores)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr, tpr, color=COLOR_SEP, linewidth=3, label=f'ROC (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
ax.fill_between(fpr, tpr, alpha=0.1, color=COLOR_SEP)

# Find optimal threshold (Youden's J)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold_roc = thresholds_roc[optimal_idx]
ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'o', markersize=12, color=COLOR_OBF, 
        label=f'Optimal: FPR={fpr[optimal_idx]:.3f}, TPR={tpr[optimal_idx]:.3f}')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
ax.set_title('ROC Curve — Token-Based Delta Angle', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('graphs/token_based_roc.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved graphs/token_based_roc.png')

# ============================================================================
# Graph 3: Precision-Recall Curve
# ============================================================================
precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
ap = average_precision_score(y_true, y_scores)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(recall, precision, color=COLOR_SEP, linewidth=3, label=f'PR (AP = {ap:.3f})')
ax.fill_between(recall, precision, alpha=0.1, color=COLOR_SEP)

# Find F1-optimal threshold
f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
optimal_f1_idx = np.argmax(f1_scores)
ax.plot(recall[optimal_f1_idx], precision[optimal_f1_idx], 'o', markersize=12, color=COLOR_OBF,
        label=f'Best F1={f1_scores[optimal_f1_idx]:.3f}')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve — Token-Based Delta', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)

plt.tight_layout()
plt.savefig('graphs/token_based_pr.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved graphs/token_based_pr.png')

# ============================================================================
# Graph 4: FPR/FNR vs Threshold
# ============================================================================
thresholds = np.linspace(5, 40, 200)
fprs = []
fnrs = []
f1s = []

for t in thresholds:
    pred_obf = obf < t
    tp = np.sum(pred_obf)
    fn = np.sum(~pred_obf)
    fp = np.sum(ben < t)
    tn = np.sum(ben >= t)
    
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
    
    fprs.append(fpr_val)
    fnrs.append(fnr_val)
    f1s.append(f1_val)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds, fprs, color=COLOR_OBF, linewidth=2.5, label='FPR')
ax.plot(thresholds, fnrs, color=COLOR_BEN, linewidth=2.5, label='FNR')
ax.plot(thresholds, f1s, color=COLOR_SEP, linewidth=2.5, label='F1', linestyle='--')

# Mark optimal F1
optimal_idx = np.argmax(f1s)
ax.axvline(thresholds[optimal_idx], color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax.plot(thresholds[optimal_idx], f1s[optimal_idx], 'o', markersize=10, color=COLOR_SEP, 
        label=f'Best F1={f1s[optimal_idx]:.3f} @ {thresholds[optimal_idx]:.1f}°')

# Mark 95th percentile
ax.axvline(threshold, color='purple', linestyle='--', linewidth=1.5, alpha=0.7, 
           label=f'95th pct: {threshold:.1f}°')

ax.set_xlabel('Threshold (degrees)', fontsize=12)
ax.set_ylabel('Rate', fontsize=12)
ax.set_title('FPR/FNR/F1 vs Threshold — Token-Based Delta', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim(5, 40)
ax.set_ylim(-0.02, 1.02)

plt.tight_layout()
plt.savefig('graphs/token_based_threshold.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved graphs/token_based_threshold.png')

# ============================================================================
# Graph 5: Token count distribution
# ============================================================================
token_counts_ben = [token_results[k]['token_count'] for k, r in token_results.items() if r['label'] == 'benign']
token_counts_obf = [token_results[k]['token_count'] for k, r in token_results.items() if r['label'] == 'obfuscation']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
bins = np.linspace(0, max(max(token_counts_ben), max(token_counts_obf)) + 5, 40)
ax.hist(token_counts_ben, bins=bins, alpha=0.7, color=COLOR_BEN, label='Benign', density=True, edgecolor='white')
ax.hist(token_counts_obf, bins=bins, alpha=0.7, color=COLOR_OBF, label='Obfuscation', density=True, edgecolor='white')
ax.set_xlabel('Token Count', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Input Length Distribution', fontsize=14, fontweight='bold')
ax.legend()

ax = axes[1]
# Scatter: token count vs delta
ben_tc = [token_results[k]['token_count'] for k, r in token_results.items() if r['label'] == 'benign']
ben_d = [r['delta'] for k, r in token_results.items() if r['label'] == 'benign']
obf_tc = [token_results[k]['token_count'] for k, r in token_results.items() if r['label'] == 'obfuscation']
obf_d = [r['delta'] for k, r in token_results.items() if r['label'] == 'obfuscation']

ax.scatter(ben_tc, ben_d, alpha=0.4, s=15, color=COLOR_BEN, label='Benign', edgecolors='none')
ax.scatter(obf_tc, obf_d, alpha=0.4, s=15, color=COLOR_OBF, label='Obfuscation', edgecolors='none')
ax.axhline(threshold, color=COLOR_SEP, linestyle='--', linewidth=1.5, alpha=0.7, label=f'Threshold: {threshold:.1f}°')
ax.set_xlabel('Token Count', fontsize=12)
ax.set_ylabel('Token Delta Angle (degrees)', fontsize=12)
ax.set_title('Delta vs Input Length', fontsize=14, fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig('graphs/token_based_token_count.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved graphs/token_based_token_count.png')

print()
print('All graphs saved to graphs/')
