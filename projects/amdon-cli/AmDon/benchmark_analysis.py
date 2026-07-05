#!/usr/bin/env python3
"""
Delta Angle Benchmark: Test the delta angle measure against real prompt injection datasets.
"""

import json
import csv
import os
import sys
from pathlib import Path

# Dataset URLs
DATASETS = {
    "owasp_benchmark": {
        "url": "https://huggingface.co/datasets/PointGuardAI/Prompt-Injection-OWASP-Benchmark-V2",
        "size": 600,
        "description": "OWASP LLM01 aligned benchmark"
    },
    "injecagent": {
        "url": "https://github.com/uiuc-kang-lab/InjecAgent",
        "size": 1054,
        "description": "Indirect prompt injection in tool-integrated agents"
    },
    "hackaprompt": {
        "url": "https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset",
        "size": 600000,
        "description": "Global prompt hacking competition dataset"
    }
}

def print_datasets():
    """Print available datasets."""
    print("=" * 60)
    print("PROMPT INJECTION BENCHMARK DATASETS")
    print("=" * 60)
    print()
    
    for name, info in DATASETS.items():
        print(f"  {name}")
        print(f"    URL: {info['url']}")
        print(f"    Size: {info['size']:,} samples")
        print(f"    Description: {info['description']}")
        print()
    
    print("=" * 60)
    print("ANALYSIS PLAN")
    print("=" * 60)
    print()
    print("1. Download datasets")
    print("2. Extract normal vs injection prompts")
    print("3. Run delta angle analysis on each")
    print("4. Compute detection rates and false positive rates")
    print("5. Compare with baselines (entropy, embedding clustering)")
    print("6. Update paper with results")
    print()

if __name__ == "__main__":
    print_datasets()