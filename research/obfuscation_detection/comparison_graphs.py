#!/usr/bin/env python3
"""Generate ROC and PR curves for obfuscation detection."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os

def plot_roc_curves(df):
    """Plot ROC curves for both models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {'nvidia/nv-embedqa-e5-v5': '#e74c3c', 'nvidia/llama-nemotron-embed-1b-v2': '#3498db'}
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        obf = model_df[model_df['label'] == 'obfuscation']['delta']
        ben = model_df[model_df['label'] == 'benign']['delta']
        
        y_true = [1] * len(obf) + [0] * len(ben)
        y_scores = list(-obf.values) + list(-ben.values)
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        short_name = model.split('/')[-1][:25]
        ax.plot(fpr, tpr, color=colors[model], linewidth=2, 
                label=f'{short_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve: Obfuscation Detection', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig('graphs/roc_curves.png', dpi=150, bbox_inches='tight')
    print("Saved graphs/roc_curves.png")

def plot_pr_curves(df):
    """Plot Precision-Recall curves for both models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {'nvidia/nv-embedqa-e5-v5': '#e74c3c', 'nvidia/llama-nemotron-embed-1b-v2': '#3498db'}
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        obf = model_df[model_df['label'] == 'obfuscation']['delta']
        ben = model_df[model_df['label'] == 'benign']['delta']
        
        y_true = [1] * len(obf) + [0] * len(ben)
        y_scores = list(-obf.values) + list(-ben.values)
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        short_name = model.split('/')[-1][:25]
        ax.plot(recall, precision, color=colors[model], linewidth=2, 
                label=f'{short_name} (AP = {ap:.3f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve: Obfuscation Detection', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig('graphs/pr_curves.png', dpi=150, bbox_inches='tight')
    print("Saved graphs/pr_curves.png")

def plot_threshold_analysis(df):
    """Plot F1, precision, recall vs threshold for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'nvidia/nv-embedqa-e5-v5': '#e74c3c', 'nvidia/llama-nemotron-embed-1b-v2': '#3498db'}
    
    for idx, model in enumerate(df['model'].unique()):
        model_df = df[df['model'] == model]
        obf = model_df[model_df['label'] == 'obfuscation']['delta'].values
        ben = model_df[model_df['label'] == 'benign']['delta'].values
        
        thresholds = np.linspace(0, 1.3, 100)
        f1s, precs, recs = [], [], []
        
        for t in thresholds:
            tp = np.sum(obf < t)
            fp = np.sum(ben < t)
            fn = np.sum(obf >= t)
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            
            f1s.append(f1)
            precs.append(p)
            recs.append(r)
        
        ax = axes[idx]
        ax.plot(thresholds, f1s, 'k-', linewidth=2, label='F1')
        ax.plot(thresholds, precs, '--', color='#2ecc71', linewidth=1.5, label='Precision')
        ax.plot(thresholds, recs, ':', color='#e67e22', linewidth=1.5, label='Recall')
        
        short_name = model.split('/')[-1][:25]
        ax.set_title(short_name, fontsize=12)
        ax.set_xlabel('Threshold', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1.3])
        ax.set_ylim([0, 1.05])
    
    plt.suptitle('Threshold Analysis: F1, Precision, Recall', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('graphs/threshold_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved graphs/threshold_analysis.png")

def main():
    df = pd.read_csv('data/standalone_results.csv')
    
    os.makedirs('graphs', exist_ok=True)
    
    plot_roc_curves(df)
    plot_pr_curves(df)
    plot_threshold_analysis(df)
    
    print("\nDone. Upload to imgur for paper embedding.")

if __name__ == '__main__':
    main()
