#!/usr/bin/env python3
"""
Standalone obfuscation detection benchmark.
No AMDON dependency - uses NIM API directly.
"""

import json
import re
import math
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple

# NIM API config
NIM_API_URL = "https://integrate.api.nvidia.com/v1"
NIM_API_KEY = "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"

def chunk_input(text: str) -> List[str]:
    """Split text into semantic chunks."""
    # Split by sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        
        # Further split long sentences by commas or semicolons
        if len(sentence) > 50:
            sub_chunks = re.split(r'(?<=[,;])\s+', sentence)
            for sub_chunk in sub_chunks:
                if sub_chunk.strip():
                    chunks.append(sub_chunk.strip())
        else:
            chunks.append(sentence.strip())
    
    # Fallback: split by word count if < 2 chunks
    if len(chunks) < 2 and len(text) > 10:
        words = text.split()
        mid = len(words) // 2
        chunks = [' '.join(words[:mid]), ' '.join(words[mid:])]
    
    return chunks

def get_embeddings(texts: List[str], model: str) -> List[List[float]]:
    """Get embeddings from NIM API."""
    headers = {
        "Authorization": f"Bearer {NIM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Determine input_type based on model
    input_type = "passage"
    if "e5" in model or "nemotron" in model:
        input_type = "passage"
    
    payload = {
        "input": texts,
        "model": model,
        "input_type": input_type
    }
    
    response = requests.post(f"{NIM_API_URL}/embeddings", json=payload, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    data = response.json()
    return [item["embedding"] for item in data["data"]]

def compute_angle(v1: List[float], v2: List[float]) -> float:
    """Compute angle between two vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    
    if mag1 > 0 and mag2 > 0:
        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        return math.acos(cos_angle)
    return 0

def compute_delta_angle(text: str, model: str) -> float:
    """Compute delta angle for a text sample."""
    chunks = chunk_input(text)
    
    if len(chunks) < 2:
        return 0
    
    # Get embeddings for all chunks
    embeddings = get_embeddings(chunks, model)
    
    # Compute angles between consecutive chunks
    angles = []
    for i in range(1, len(embeddings)):
        angle = compute_angle(embeddings[i-1], embeddings[i])
        angles.append(angle)
    
    if not angles:
        return 0
    
    # Softmax weighting
    temperature = 0.5
    max_angle = max(angles)
    exp_values = [math.exp((a - max_angle) / temperature) for a in angles]
    sum_exp = sum(exp_values)
    weights = [e / sum_exp for e in exp_values]
    
    # Weighted mean
    return sum(w * a for w, a in zip(weights, angles))

def load_data() -> Tuple[List[str], List[str]]:
    """Load obfuscation and benign samples."""
    with open('data/obf_trigger.json') as f:
        obf = json.load(f)
    with open('data/obf_benign.json') as f:
        benign = json.load(f)
    return obf, benign

def run_benchmark(model: str, max_samples: int = 50) -> pd.DataFrame:
    """Run benchmark for a single model."""
    obf_samples, benign_samples = load_data()
    
    # Limit samples
    obf_samples = obf_samples[:max_samples]
    benign_samples = benign_samples[:max_samples]
    
    results = []
    
    print(f"Testing {model}...")
    print(f"  {len(obf_samples)} obfuscation, {len(benign_samples)} benign")
    
    # Process obfuscation samples
    for i, text in enumerate(obf_samples):
        try:
            delta = compute_delta_angle(text, model)
            results.append({
                'model': model,
                'text': text[:100],
                'label': 'obfuscation',
                'delta': delta
            })
            if (i + 1) % 10 == 0:
                print(f"    {i + 1}/{len(obf_samples)} obfuscation")
        except Exception as e:
            print(f"    Error: {e}")
    
    # Process benign samples
    for i, text in enumerate(benign_samples):
        try:
            delta = compute_delta_angle(text, model)
            results.append({
                'model': model,
                'text': text[:100],
                'label': 'benign',
                'delta': delta
            })
            if (i + 1) % 10 == 0:
                print(f"    {i + 1}/{len(benign_samples)} benign")
        except Exception as e:
            print(f"    Error: {e}")
    
    return pd.DataFrame(results)

def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute performance metrics."""
    obf = df[df['label'] == 'obfuscation']['delta']
    ben = df[df['label'] == 'benign']['delta']
    
    # Find optimal threshold
    all_deltas = np.concatenate([obf.values, ben.values])
    all_deltas.sort()
    
    best_f1 = 0
    best_threshold = 0
    best_metrics = {}
    
    for threshold in all_deltas:
        tp = np.sum(obf < threshold)
        fp = np.sum(ben < threshold)
        fn = np.sum(obf >= threshold)
        tn = np.sum(ben >= threshold)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'threshold': threshold,
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn)
            }
    
    return {
        'obf_mean': obf.mean(),
        'obf_std': obf.std(),
        'ben_mean': ben.mean(),
        'ben_std': ben.std(),
        'gap': abs(ben.mean() - obf.mean()),
        **best_metrics
    }

def plot_results(df: pd.DataFrame, metrics: dict, model: str):
    """Generate visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    obf = df[df['label'] == 'obfuscation']['delta']
    ben = df[df['label'] == 'benign']['delta']
    
    # Distribution plot
    axes[0].hist(obf, bins=20, alpha=0.5, label='Obfuscation', color='#e74c3c', density=True)
    axes[0].hist(ben, bins=20, alpha=0.5, label='Benign', color='#3498db', density=True)
    axes[0].axvline(metrics['threshold'], color='black', linestyle='--', label=f"Threshold={metrics['threshold']:.3f}")
    axes[0].set_xlabel('Delta Angle')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Delta Angle Distribution: {model}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Metrics bar
    names = ['F1', 'Precision', 'Recall']
    values = [metrics['f1'], metrics['precision'], metrics['recall']]
    colors = ['#9b59b6', '#3498db', '#2ecc71']
    
    axes[1].bar(names, values, color=colors, alpha=0.8)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Score')
    axes[1].set_title(f'Performance Metrics')
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(values):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    safe_name = model.split('/')[-1][:30]
    plt.savefig(f'graphs/standalone_{safe_name}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved graphs/standalone_{safe_name}.png")

def main():
    models = [
        "nvidia/nv-embedqa-e5-v5",
        "nvidia/llama-nemotron-embed-1b-v2"
    ]
    
    all_results = []
    all_metrics = []
    
    for model in models:
        df = run_benchmark(model, max_samples=50)
        metrics = compute_metrics(df)
        
        all_results.append(df)
        all_metrics.append(metrics)
        
        print(f"\n  Results for {model}:")
        print(f"    Obfuscation: {metrics['obf_mean']:.4f} ± {metrics['obf_std']:.4f}")
        print(f"    Benign:      {metrics['ben_mean']:.4f} ± {metrics['ben_std']:.4f}")
        print(f"    Gap:         {metrics['gap']:.4f}")
        print(f"    F1:          {metrics['f1']:.4f}")
        print(f"    Threshold:   {metrics['threshold']:.4f}")
        
        plot_results(df, metrics, model)
    
    # Combined CSV
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv('data/standalone_results.csv', index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for model, metrics in zip(models, all_metrics):
        name = model.split('/')[-1][:30]
        print(f"{name}: gap={metrics['gap']:.3f}, f1={metrics['f1']:.3f}")
    
    avg_gap = np.mean([m['gap'] for m in all_metrics])
    avg_f1 = np.mean([m['f1'] for m in all_metrics])
    print(f"\nAverage gap: {avg_gap:.3f}")
    print(f"Average F1: {avg_f1:.3f}")

if __name__ == '__main__':
    main()
