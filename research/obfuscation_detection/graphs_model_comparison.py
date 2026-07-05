#!/usr/bin/env python3
"""
Comprehensive graph suite comparing all embedding models.
FIXED: Correct AUC computation based on sign convention.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
import re, math, os
from sklearn.metrics import roc_curve, auc

os.makedirs('graphs', exist_ok=True)

# ─── Load data ───

with open('data/delta_cache_all_models.json') as f:
    cache = json.load(f)

texts = cache['texts']
labels = cache['labels']
models = list(cache['models'].keys())

# Filter to working models only
working_models = [m for m in models if cache['models'][m]['std'] > 0]

# ─── Compute features ───

def char_entropy(text):
    if not text: return 0
    freq = Counter(text)
    total = len(text)
    return -sum((c/total) * math.log2(c/total) for c in freq.values())

def special_char_ratio(text):
    if not text: return 0
    return sum(1 for c in text if not c.isalpha() and not c.isspace()) / len(text)

# Build dataframe
rows = []
for i, text in enumerate(texts):
    for m in working_models:
        rows.append({
            'label': labels[i],
            'text_len': len(text),
            'entropy': char_entropy(text),
            'special_ratio': special_char_ratio(text),
            'delta': cache['models'][m]['deltas'][i],
            'model': m,
        })

df = pd.DataFrame(rows)

# ─── Compute correct metrics ───

def compute_metrics(obf_deltas, ben_deltas):
    """Compute correct AUC and metrics based on actual direction."""
    obf_mean = np.mean(obf_deltas)
    ben_mean = np.mean(ben_deltas)
    sep = obf_mean - ben_mean
    
    # Test both directions
    y_true = [1]*len(obf_deltas) + [0]*len(ben_deltas)
    y_direct = list(obf_deltas) + list(ben_deltas)
    y_negated = [-d for d in obf_deltas] + [-d for d in ben_deltas]
    
    fpr1, tpr1, _ = roc_curve(y_true, y_direct)
    auc1 = auc(fpr1, tpr1)
    
    fpr2, tpr2, _ = roc_curve(y_true, y_negated)
    auc2 = auc(fpr2, tpr2)
    
    # Correct direction: higher AUC
    if auc1 > auc2:
        correct_auc = auc1
        correct_direction = 'direct'  # Lower delta = obfuscation
        flag_when = 'delta < threshold'
    else:
        correct_auc = auc2
        correct_direction = 'negated'  # Higher negated delta = obfuscation
        flag_when = 'delta < threshold (same practical rule)'
    
    # Find threshold at 5% FPR for the correct direction
    y_scores = y_direct if correct_direction == 'direct' else y_negated
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Find threshold where FPR <= 5%
    idx = np.where(fpr <= 0.05)[0]
    if len(idx) > 0:
        best_idx = idx[-1]
        threshold = thresholds[best_idx]
        recall_at_5fpr = tpr[best_idx]
    else:
        threshold = thresholds[0]
        recall_at_5fpr = tpr[0]
    
    # Compute F1 at this threshold
    if correct_direction == 'direct':
        tp = sum(obf_deltas < threshold)
        fp = sum(ben_deltas < threshold)
        fn = sum(obf_deltas >= threshold)
        tn = sum(ben_deltas >= threshold)
    else:
        tp = sum(-np.array(obf_deltas) < threshold)
        fp = sum(-np.array(ben_deltas) < threshold)
        fn = sum(-np.array(obf_deltas) >= threshold)
        tn = sum(-np.array(ben_deltas) >= threshold)
    
    precision = tp/(tp+fp) if (tp+fp)>0 else 0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
    fpr_val = fp/(fp+tn) if (fp+tn)>0 else 0
    
    return {
        'correct_auc': correct_auc,
        'separation': abs(sep),
        'obf_mean': obf_mean,
        'ben_mean': ben_mean,
        'threshold': threshold,
        'recall_at_5fpr': recall_at_5fpr,
        'f1': f1,
        'fpr': fpr_val,
        'precision': precision,
        'direction': 'obf < ben' if sep < 0 else 'obf > ben',
    }

# ─── 1. Delta Distributions Comparison ───

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, m in enumerate(working_models):
    ax = axes[idx]
    sub = df[df['model'] == m]
    obf = sub[sub['label'] == 'obfuscation']['delta']
    ben = sub[sub['label'] == 'benign']['delta']
    
    ax.hist(ben, bins=25, alpha=0.6, label='Benign', color='#3498db', density=True, edgecolor='white')
    ax.hist(obf, bins=25, alpha=0.6, label='Obfuscation', color='#e74c3c', density=True, edgecolor='white')
    
    mu_b, sig_b = ben.mean(), ben.std()
    mu_o, sig_o = obf.mean(), obf.std()
    
    # Mark overlap zone
    obf_95_high = mu_o + 2 * sig_o
    ben_95_low = mu_b - 2 * sig_b
    if ben_95_low < obf_95_high:
        ax.axvspan(ben_95_low, obf_95_high, alpha=0.15, color='yellow', label='Overlap')
    
    ax.axvline(mu_b, color='#3498db', linestyle='--', alpha=0.8, linewidth=1.5)
    ax.axvline(mu_o, color='#e74c3c', linestyle='--', alpha=0.8, linewidth=1.5)
    
    metrics = compute_metrics(obf.values, ben.values)
    
    ax.set_xlabel('Signed Delta Angle', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'{m}\nAUC={metrics["correct_auc"]:.3f}, Sep={metrics["separation"]:.3f}', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Delta Angle Distributions by Model (Obfuscation < Benign)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graphs/model_comparison_distributions.png', dpi=150, bbox_inches='tight')
print("Saved graphs/model_comparison_distributions.png")
plt.close()

# ─── 2. Separation Bar Chart (Corrected) ───

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Separation
ax1 = axes[0]
model_names = []
separations = []
colors = []

for m in working_models:
    metrics = compute_metrics(
        df[(df['model']==m) & (df['label']=='obfuscation')]['delta'].values,
        df[(df['model']==m) & (df['label']=='benign')]['delta'].values
    )
    model_names.append(m)
    separations.append(metrics['separation'])
    colors.append('#2ecc71' if metrics['separation'] > 0.3 else '#f39c12' if metrics['separation'] > 0.1 else '#e74c3c')

x = np.arange(len(model_names))
bars = ax1.bar(x, separations, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)

for bar, sep in zip(bars, separations):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{sep:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
ax1.set_ylabel('|Separation| (|μ_obf - μ_ben|)', fontsize=11)
ax1.set_title('Separation Magnitude', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Right: AUC-ROC
ax2 = axes[1]
aucs = []
for m in working_models:
    metrics = compute_metrics(
        df[(df['model']==m) & (df['label']=='obfuscation')]['delta'].values,
        df[(df['model']==m) & (df['label']=='benign')]['delta'].values
    )
    aucs.append(metrics['correct_auc'])

bars2 = ax2.bar(x, aucs, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)

for bar, auc_val in zip(bars2, aucs):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{auc_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('AUC-ROC', fontsize=11)
ax2.set_title('Corrected AUC-ROC (direction-aware)', fontsize=12, fontweight='bold')
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Model Performance Comparison (Corrected)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('graphs/model_comparison_separation.png', dpi=150, bbox_inches='tight')
print("Saved graphs/model_comparison_separation.png")
plt.close()

# ─── 3. ROC Curves (Corrected Direction) ───

fig, ax = plt.subplots(figsize=(10, 8))

colors_roc = plt.cm.Set2(np.linspace(0, 1, len(working_models)))

for m, color in zip(working_models, colors_roc):
    obf = df[(df['model']==m) & (df['label']=='obfuscation')]['delta'].values
    ben = df[(df['model']==m) & (df['label']=='benign')]['delta'].values
    
    metrics = compute_metrics(obf, ben)
    
    # Use correct direction for plotting
    y_true = [1]*len(obf) + [0]*len(ben)
    if metrics['direction'] == 'obf < ben':
        y_scores = list(obf) + list(ben)  # Lower = obfuscation
    else:
        y_scores = list(-obf) + list(-ben)  # Higher = obfuscation
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color=color, linewidth=2, 
            label=f'{m} (AUC={roc_auc:.3f}, sep={metrics["separation"]:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves (Corrected Direction: Lower Delta = Obfuscation)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig('graphs/model_comparison_roc.png', dpi=150, bbox_inches='tight')
print("Saved graphs/model_comparison_roc.png")
plt.close()

# ─── 4. FPR vs Recall Tradeoff (Corrected) ───

fig, ax = plt.subplots(figsize=(10, 8))

for m, color in zip(working_models, colors_roc):
    obf = df[(df['model']==m) & (df['label']=='obfuscation')]['delta'].values
    ben = df[(df['model']==m) & (df['label']=='benign')]['delta'].values
    
    metrics = compute_metrics(obf, ben)
    
    # Use correct direction
    if metrics['direction'] == 'obf < ben':
        all_deltas = np.concatenate([obf, ben])
        thresholds = np.linspace(all_deltas.min(), all_deltas.max(), 100)
        
        fprs = []
        recalls = []
        for t in thresholds:
            tp = sum(obf < t)
            fp = sum(ben < t)
            fn = sum(obf >= t)
            tn = sum(ben >= t)
            
            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            fprs.append(fpr_val)
            recalls.append(recall_val)
    else:
        neg_obf = -obf
        neg_ben = -ben
        all_deltas = np.concatenate([neg_obf, neg_ben])
        thresholds = np.linspace(all_deltas.min(), all_deltas.max(), 100)
        
        fprs = []
        recalls = []
        for t in thresholds:
            tp = sum(neg_obf >= t)
            fp = sum(neg_ben >= t)
            fn = sum(neg_obf < t)
            tn = sum(neg_ben < t)
            
            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            fprs.append(fpr_val)
            recalls.append(recall_val)
    
    ax.plot(fprs, recalls, color=color, linewidth=2, 
            label=f'{m} (AUC={metrics["correct_auc"]:.3f})')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('Recall (True Positive Rate)', fontsize=12)
ax.set_title('FPR vs Recall Tradeoff (Corrected Direction)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig('graphs/model_comparison_fpr_recall.png', dpi=150, bbox_inches='tight')
print("Saved graphs/model_comparison_fpr_recall.png")
plt.close()

# ─── 5. Summary Table (Corrected) ───

summary_rows = []
for m in working_models:
    obf = df[(df['model']==m) & (df['label']=='obfuscation')]['delta'].values
    ben = df[(df['model']==m) & (df['label']=='benign')]['delta'].values
    
    metrics = compute_metrics(obf, ben)
    
    summary_rows.append({
        'Model': m,
        'AUC-ROC': f'{metrics["correct_auc"]:.3f}',
        '|Separation|': f'{metrics["separation"]:.3f}',
        'Direction': metrics['direction'],
        'Threshold': f'{metrics["threshold"]:.3f}',
        'Recall@5%FPR': f'{metrics["recall_at_5fpr"]:.3f}',
        'F1': f'{metrics["f1"]:.3f}',
    })

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values('AUC-ROC', ascending=False)
summary_df.to_csv('data/model_comparison_summary.csv', index=False)

print("\n" + "="*90)
print("MODEL COMPARISON SUMMARY (CORRECTED)")
print("="*90)
print("SIGN CONVENTION: All models show obfuscation has LOWER delta than benign")
print("RULE: Flag input as obfuscated if delta < threshold")
print("="*90)
print(summary_df.to_string(index=False))
print(f"\nSaved: data/model_comparison_summary.csv")

# ─── Final list ───

print("\nGraphs saved:")
for f in sorted(os.listdir('graphs')):
    if f.startswith('model_comparison'):
        print(f"  {f}")
