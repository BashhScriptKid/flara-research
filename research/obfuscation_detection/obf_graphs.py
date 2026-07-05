#!/usr/bin/env python3
"""Obfuscation detection benchmark analysis and graph generation."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
import json
import sys

def load_data():
    """Load combined obfuscation results."""
    df = pd.read_csv('data/obf_combined_results.csv')
    return df

def compute_stats(df, model):
    """Compute statistics for a specific model."""
    model_df = df[df['model'] == model]
    obf = model_df[model_df['label'] == 'obfuscation']['delta']
    ben = model_df[model_df['label'] == 'benign']['delta']
    
    stats_dict = {
        'model': model,
        'obf_mean': obf.mean(),
        'obf_std': obf.std(),
        'obf_median': obf.median(),
        'ben_mean': ben.mean(),
        'ben_std': ben.std(),
        'ben_median': ben.median(),
        'gap': abs(ben.mean() - obf.mean()),
        'overlap_pct': calculate_overlap(obf, ben),
    }
    
    # Find optimal threshold (lower delta = obfuscation)
    all_deltas = np.concatenate([obf.values, ben.values])
    all_deltas.sort()
    best_f1 = 0
    best_threshold = 0
    for threshold in all_deltas:
        # Obfuscation has LOWER delta, so classify as obfuscation if delta < threshold
        tp = np.sum(obf < threshold)  # True positive: obfuscation correctly identified
        fp = np.sum(ben < threshold)  # False positive: benign misclassified as obfuscation
        fn = np.sum(obf >= threshold) # False negative: obfuscation missed
        tn = np.sum(ben >= threshold) # True negative: benign correctly identified
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            stats_dict['precision'] = precision
            stats_dict['recall'] = recall
            stats_dict['f1'] = f1
            stats_dict['threshold'] = threshold
            stats_dict['tp'] = int(tp)
            stats_dict['fp'] = int(fp)
            stats_dict['fn'] = int(fn)
            stats_dict['tn'] = int(tn)
    
    return stats_dict

def calculate_overlap(obf, ben):
    """Calculate percentage of overlapping distributions."""
    min_val = min(obf.min(), ben.min())
    max_val = max(obf.max(), ben.max())
    
    bins = np.linspace(min_val, max_val, 50)
    obf_hist, _ = np.histogram(obf, bins=bins, density=True)
    ben_hist, _ = np.histogram(ben, bins=bins, density=True)
    
    overlap = np.minimum(obf_hist, ben_hist)
    overlap_pct = np.sum(overlap) / np.sum(obf_hist) * 100
    return overlap_pct

def plot_bar_chart(stats_list):
    """Plot comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = [s['model'].split('/')[-1][:20] for s in stats_list]
    obf_means = [s['obf_mean'] for s in stats_list]
    ben_means = [s['ben_mean'] for s in stats_list]
    gaps = [s['gap'] for s in stats_list]
    
    x = np.arange(len(models))
    width = 0.35
    
    # Delta means
    axes[0].bar(x - width/2, obf_means, width, label='Obfuscation', color='#e74c3c', alpha=0.8)
    axes[0].bar(x + width/2, ben_means, width, label='Benign', color='#3498db', alpha=0.8)
    axes[0].set_ylabel('Average Delta Angle')
    axes[0].set_title('Delta Angle by Category')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Gap
    axes[1].bar(x, gaps, 0.5, color='#2ecc71', alpha=0.8)
    axes[1].set_ylabel('Delta Gap (|Benign - Obfuscation|)')
    axes[1].set_title('Separation Gap')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    # F1
    f1s = [s['f1'] for s in stats_list]
    axes[2].bar(x, f1s, 0.5, color='#9b59b6', alpha=0.8)
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Best F1 Score')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45, ha='right')
    axes[2].set_ylim(0, 1)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('graphs/obf_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved obf_comparison.png")

def plot_distribution(df, model):
    """Plot distribution comparison for a single model."""
    model_df = df[df['model'] == model]
    obf = model_df[model_df['label'] == 'obfuscation']['delta']
    ben = model_df[model_df['label'] == 'benign']['delta']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Histograms
    ax.hist(obf, bins=25, alpha=0.5, label=f'Obfuscation (n={len(obf)})', color='#e74c3c', density=True)
    ax.hist(ben, bins=25, alpha=0.5, label=f'Benign (n={len(ben)})', color='#3498db', density=True)
    
    # KDE
    if len(obf) > 1:
        obf_kde = stats.gaussian_kde(obf)
        x_range = np.linspace(min(obf.min(), ben.min()), max(obf.max(), ben.max()), 100)
        ax.plot(x_range, obf_kde(x_range), color='#e74c3c', linewidth=2)
    
    if len(ben) > 1:
        ben_kde = stats.gaussian_kde(ben)
        x_range = np.linspace(min(obf.min(), ben.min()), max(obf.max(), ben.max()), 100)
        ax.plot(x_range, ben_kde(x_range), color='#3498db', linewidth=2)
    
    ax.set_xlabel('Delta Angle')
    ax.set_ylabel('Density')
    ax.set_title(f'Delta Angle Distribution: {model}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    safe_name = model.split('/')[-1][:30]
    plt.savefig(f'graphs/obf_dist_{safe_name}.png', dpi=150, bbox_inches='tight')
    print(f"Saved obf_dist_{safe_name}.png")

def plot_scatter(df, model):
    """Plot delta vs risk score."""
    model_df = df[df['model'] == model]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {'obfuscation': '#e74c3c', 'benign': '#3498db'}
    for label in ['obfuscation', 'benign']:
        subset = model_df[model_df['label'] == label]
        ax.scatter(subset['delta'], subset['risk'], 
                  alpha=0.6, label=label, c=colors[label], s=30)
    
    ax.set_xlabel('Delta Angle')
    ax.set_ylabel('Risk Score')
    ax.set_title(f'Delta vs Risk: {model}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    safe_name = model.split('/')[-1][:30]
    plt.savefig(f'graphs/obf_scatter_{safe_name}.png', dpi=150, bbox_inches='tight')
    print(f"Saved obf_scatter_{safe_name}.png")

def upload_to_imgur(image_path):
    """Upload image to imgur."""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        response = requests.post(
            'https://api.imgur.com/3/image',
            headers={'Authorization': 'Client-ID 546c25a59c58ad7'},
            files={'image': image_data}
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['data']['link']
        else:
            print(f"  Upload failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"  Upload error: {e}")
        return None

def main():
    print("Loading obfuscation benchmark data...")
    df = load_data()
    
    models = df['model'].unique()
    print(f"Models: {list(models)}")
    
    stats_list = []
    for model in models:
        stats = compute_stats(df, model)
        stats_list.append(stats)
        
        print(f"\n--- {model} ---")
        print(f"  Obfuscation: {stats['obf_mean']:.4f} ± {stats['obf_std']:.4f}")
        print(f"  Benign:      {stats['ben_mean']:.4f} ± {stats['ben_std']:.4f}")
        print(f"  Gap:         {stats['gap']:.4f}")
        print(f"  F1:          {stats['f1']:.4f} @ threshold {stats['threshold']:.4f}")
        print(f"  TP={stats['tp']} FP={stats['fp']} FN={stats['fn']} TN={stats['tn']}")
        print(f"  Overlap:     {stats['overlap_pct']:.1f}%")
    
    # Generate graphs
    print("\nGenerating graphs...")
    plot_bar_chart(stats_list)
    
    for model in models:
        plot_distribution(df, model)
        plot_scatter(df, model)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for stats in stats_list:
        model_short = stats['model'].split('/')[-1][:30]
        print(f"{model_short}: gap={stats['gap']:.3f}, f1={stats['f1']:.3f}, overlap={stats['overlap_pct']:.1f}%")
    
    avg_gap = np.mean([s['gap'] for s in stats_list])
    avg_f1 = np.mean([s['f1'] for s in stats_list])
    print(f"\nAverage gap: {avg_gap:.3f}")
    print(f"Average F1: {avg_f1:.3f}")
    
    # Save stats
    with open('data/obf_stats.json', 'w') as f:
        json.dump(stats_list, f, indent=2)
    print("\nSaved stats to obf_stats.json")

if __name__ == '__main__':
    main()
