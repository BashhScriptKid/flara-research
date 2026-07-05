#!/usr/bin/env python3
"""
Baseline comparison: FPR/FNR against existing detection methods.
Methods compared:
1. Regex (hex/base64 pattern detection)
2. Character entropy threshold
3. Special character ratio
4. Delta angle (our method)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math
from collections import Counter
from scipy.integrate import trapezoid

def char_entropy(text):
    """Compute character entropy."""
    if not text:
        return 0
    freq = Counter(text)
    total = len(text)
    entropy = -sum((c/total) * math.log2(c/total) for c in freq.values())
    return entropy

def special_char_ratio(text):
    """Ratio of non-alphabetic, non-space characters."""
    if not text:
        return 0
    special = sum(1 for c in text if not c.isalpha() and not c.isspace())
    return special / len(text)

def regex_score(text):
    """Score based on encoding pattern detection."""
    score = 0
    # Hex pattern: 20+ hex chars
    if re.search(r'[0-9a-fA-F]{20,}', text):
        score += 1
    # Base64 pattern: 20+ base64 chars with optional padding
    if re.search(r'[A-Za-z0-9+/]{20,}={0,2}', text):
        score += 1
    # Multiple encoding references
    if re.search(r'(decode|eval|atob|btoa|fromCharCode|hex|base64)', text, re.IGNORECASE):
        score += 0.5
    # Unicode escapes
    if re.search(r'(\\u[0-9a-fA-F]{4}){3,}', text):
        score += 1
    # Null bytes or control chars
    if re.search(r'[\x00-\x08\x0e-\x1f]', text):
        score += 0.5
    return min(score, 2.0) / 2.0  # Normalize to 0-1

def compute_rates(labels, scores, threshold, positive_direction='higher'):
    """Compute FPR and FNR at a given threshold."""
    if positive_direction == 'higher':
        predicted_positive = np.array(scores) >= threshold
    else:
        predicted_positive = np.array(scores) <= threshold
    
    labels = np.array(labels)
    
    tp = np.sum(predicted_positive & (labels == 1))
    fp = np.sum(predicted_positive & (labels == 0))
    fn = np.sum(~predicted_positive & (labels == 1))
    tn = np.sum(~predicted_positive & (labels == 0))
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return fpr, fnr

def main():
    # Load data
    df = pd.read_csv('data/standalone_results.csv')
    
    # Compute baseline features for each sample
    results = []
    for _, row in df.iterrows():
        text = row['text']
        results.append({
            'model': row['model'],
            'label': row['label'],
            'delta': row['delta'],
            'entropy': char_entropy(str(text)),
            'special_ratio': special_char_ratio(str(text)),
            'regex_score': regex_score(str(text))
        })
    
    results_df = pd.DataFrame(results)
    
    # --- Graph 1: ROC-style FPR vs FNR tradeoff ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = results_df['model'].unique()
    colors = {'entropy': '#e74c3c', 'special_char': '#3498db', 
              'regex': '#2ecc71', 'delta': '#9b59b6'}
    
    for idx, model in enumerate(models):
        model_df = results_df[results_df['model'] == model]
        
        labels = (model_df['label'] == 'obfuscation').astype(int).values
        
        # Method scores
        methods = {
            'entropy': ('entropy', 'higher'),
            'special_char': ('special_ratio', 'higher'),
            'regex': ('regex_score', 'higher'),
            'delta': ('delta', 'lower')  # Lower delta = obfuscation
        }
        
        ax = axes[idx]
        
        for method_name, (col, direction) in methods.items():
            scores = model_df[col].values
            thresholds = np.linspace(scores.min(), scores.max(), 100)
            
            fprs = []
            fnrs = []
            for t in thresholds:
                fpr, fnr = compute_rates(labels, scores, t, direction)
                fprs.append(fpr)
                fnrs.append(fnr)
            
            # Compute AUC (lower is better for FPR-FNR curve)
            fnr_arr = np.array(fnrs)
            fpr_arr = np.array(fprs)
            # Sort by FPR for proper curve
            sort_idx = np.argsort(fpr_arr)
            auc = trapezoid(fnr_arr[sort_idx], fpr_arr[sort_idx])
            
            short_name = model.split('/')[-1][:20]
            ax.plot(fpr_arr, fnrs, color=colors[method_name], linewidth=2,
                   label=f'{method_name} (AUC={auc:.3f})')
        
        ax.plot([0, 1], [1, 0], 'k--', linewidth=1, alpha=0.3, label='Perfect')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('False Negative Rate', fontsize=12)
        ax.set_title(f'FPR-FNR Tradeoff: {model.split("/")[-1][:25]}', fontsize=13)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.invert_xaxis()  # Lower FPR is better (left side)
    
    plt.suptitle('Detection Method Comparison: FPR vs FNR', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('graphs/fpr_fnr_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved graphs/fpr_fnr_comparison.png")
    
    # --- Graph 2: Operating point comparison (fixed FPR = 5%) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    fixed_fpr = 0.05
    
    for idx, model in enumerate(models):
        model_df = results_df[results_df['model'] == model]
        labels = (model_df['label'] == 'obfuscation').astype(int).values
        
        method_names = []
        fnr_at_fpr = []
        f1_scores = []
        
        for method_name, (col, direction) in methods.items():
            scores = model_df[col].values
            
            # Find threshold that gives FPR closest to fixed_fpr
            thresholds = np.linspace(scores.min(), scores.max(), 200)
            best_threshold = thresholds[0]
            best_diff = float('inf')
            
            for t in thresholds:
                fpr, fnr = compute_rates(labels, scores, t, direction)
                diff = abs(fpr - fixed_fpr)
                if diff < best_diff:
                    best_diff = diff
                    best_threshold = t
                    best_fpr = fpr
                    best_fnr = fnr
            
            # Compute F1 at this threshold
            if direction == 'higher':
                predicted = scores >= best_threshold
            else:
                predicted = scores <= best_threshold
            
            tp = np.sum(predicted & (labels == 1))
            fp = np.sum(predicted & (labels == 0))
            fn = np.sum(~predicted & (labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            method_names.append(method_name)
            fnr_at_fpr.append(best_fnr)
            f1_scores.append(f1)
        
        ax = axes[idx]
        x = np.arange(len(method_names))
        
        bars = ax.bar(x, [1 - f for f in fnr_at_fpr], 0.6, 
                     color=[colors[m] for m in method_names], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, fontsize=11)
        ax.set_ylabel('Recall (1 - FNR)', fontsize=12)
        ax.set_title(f'Recall @ FPR={fixed_fpr:.0%}: {model.split("/")[-1][:25]}', fontsize=13)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add F1 labels on bars
        for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                   f'F1={f1:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(f'Recall at Fixed FPR={fixed_fpr:.0%}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('graphs/recall_at_fpr5.png', dpi=150, bbox_inches='tight')
    print("Saved graphs/recall_at_fpr5.png")
    
    # --- Graph 3: Feature correlation heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    corr_cols = ['delta', 'entropy', 'special_ratio', 'regex_score']
    corr_labels = ['Delta Angle', 'Entropy', 'Special Char %', 'Regex']
    
    for model in models:
        model_df = results_df[results_df['model'] == model]
        corr = model_df[corr_cols].corr()
        
        print(f"\n{model.split('/')[-1]} correlations:")
        print(corr.round(3))
    
    # Average correlation across models
    all_corr = []
    for model in models:
        model_df = results_df[results_df['model'] == model]
        all_corr.append(model_df[corr_cols].corr())
    
    avg_corr = sum(all_corr) / len(all_corr)
    
    im = ax.imshow(avg_corr.values, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_labels)))
    ax.set_xticklabels(corr_labels, fontsize=11)
    ax.set_yticks(range(len(corr_labels)))
    ax.set_yticklabels(corr_labels, fontsize=11)
    
    # Add correlation values
    for i in range(len(corr_labels)):
        for j in range(len(corr_labels)):
            text = ax.text(j, i, f'{avg_corr.values[i, j]:.2f}',
                          ha='center', va='center', fontsize=12,
                          color='white' if abs(avg_corr.values[i, j]) > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title('Feature Correlation (Averaged Across Models)', fontsize=13)
    plt.tight_layout()
    plt.savefig('graphs/feature_correlation.png', dpi=150, bbox_inches='tight')
    print("Saved graphs/feature_correlation.png")
    
    # --- Print summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY: Operating Point Comparison (FPR ≈ 5%)")
    print("=" * 70)
    print(f"{'Method':<15} {'E5-v5 Recall':<15} {'Nemotron Recall':<15} {'Avg F1':<10}")
    print("-" * 70)
    
    for i, method in enumerate(method_names):
        recalls = []
        f1s = []
        for idx, model in enumerate(models):
            model_df = results_df[results_df['model'] == model]
            labels = (model_df['label'] == 'obfuscation').astype(int).values
            col, direction = methods[method]
            scores = model_df[col].values
            
            # Find threshold
            thresholds = np.linspace(scores.min(), scores.max(), 200)
            best_t, best_diff = thresholds[0], float('inf')
            for t in thresholds:
                fpr, fnr = compute_rates(labels, scores, t, direction)
                if abs(fpr - fixed_fpr) < best_diff:
                    best_diff = abs(fpr - fixed_fpr)
                    best_t = t
            
            if direction == 'higher':
                predicted = scores >= best_t
            else:
                predicted = scores <= best_t
            
            tp = np.sum(predicted & (labels == 1))
            fp = np.sum(predicted & (labels == 0))
            fn = np.sum(~predicted & (labels == 1))
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            recalls.append(recall)
            f1s.append(f1)
        
        avg_recall = np.mean(recalls)
        avg_f1 = np.mean(f1s)
        print(f"{method:<15} {recalls[0]:<15.3f} {recalls[1]:<15.3f} {avg_f1:<10.3f}")

if __name__ == '__main__':
    main()
