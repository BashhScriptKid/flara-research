#!/usr/bin/env python3
"""
Download and analyze prompt injection benchmark datasets.
"""

import json
import os
import sys
from pathlib import Path

# Try to import datasets
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' library not installed. Install with: pip install datasets")

def download_owasp_benchmark():
    """Download OWASP LLM01 Prompt Injection Benchmark."""
    print("Downloading OWASP LLM01 Benchmark...")
    
    if not HAS_DATASETS:
        print("Error: 'datasets' library not installed")
        return None
    
    try:
        ds = load_dataset("PointGuardAI/Prompt-Injection-OWASP-Benchmark-V2")
        print(f"  Loaded {len(ds['train'])} samples")
        return ds
    except Exception as e:
        print(f"  Error: {e}")
        return None

def analyze_dataset(ds, name):
    """Analyze dataset and extract statistics."""
    print(f"\nAnalyzing {name}...")
    
    trigger_samples = []
    benign_samples = []
    
    for sample in ds['train']:
        text = sample['text']
        should_trigger = sample['should_trigger']
        
        if should_trigger == 1:
            trigger_samples.append(text)
        else:
            benign_samples.append(text)
    
    print(f"  Trigger samples: {len(trigger_samples)}")
    print(f"  Benign samples: {len(benign_samples)}")
    
    return trigger_samples, benign_samples

def compute_basic_stats(texts, label):
    """Compute basic text statistics."""
    if not texts:
        return {}
    
    lengths = [len(t) for t in texts]
    word_counts = [len(t.split()) for t in texts]
    
    return {
        "label": label,
        "count": len(texts),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "avg_words": sum(word_counts) / len(word_counts),
        "min_words": min(word_counts),
        "max_words": max(word_counts)
    }

if __name__ == "__main__":
    print("=" * 60)
    print("PROMPT INJECTION BENCHMARK ANALYSIS")
    print("=" * 60)
    
    # Download OWASP Benchmark
    ds = download_owasp_benchmark()
    
    if ds is not None:
        # Analyze dataset
        trigger_samples, benign_samples = analyze_dataset(ds, "OWASP LLM01 Benchmark")
        
        # Compute statistics
        trigger_stats = compute_basic_stats(trigger_samples, "trigger")
        benign_stats = compute_basic_stats(benign_samples, "benign")
        
        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)
        
        print(f"\nTrigger samples:")
        for k, v in trigger_stats.items():
            if k != "label":
                print(f"  {k}: {v}")
        
        print(f"\nBenign samples:")
        for k, v in benign_stats.items():
            if k != "label":
                print(f"  {k}: {v}")
        
        # Save samples to file for delta angle analysis
        output_dir = Path("benchmark_data")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "owasp_trigger.json", "w") as f:
            json.dump(trigger_samples[:100], f, indent=2)  # Save first 100 for testing
        
        with open(output_dir / "owasp_benign.json", "w") as f:
            json.dump(benign_samples[:100], f, indent=2)  # Save first 100 for testing
        
        print(f"\nSaved 100 samples per class to {output_dir}/")
        print("Run delta angle analysis on these samples to compute detection rates.")