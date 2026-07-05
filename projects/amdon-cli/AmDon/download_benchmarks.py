#!/usr/bin/env python3
"""
Download prompt injection benchmark datasets for delta angle analysis.
Uses neuralchemy/prompt-injection-Threat-Matrix (32K samples, 7 categories).
"""

import json
import os
from pathlib import Path

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Error: 'datasets' library not installed. Install with: pip install datasets")


def download_threat_matrix():
    """Download neuralchemy/prompt-injection-Threat-Matrix."""
    if not HAS_DATASETS:
        return None

    print("Downloading neuralchemy/prompt-injection-Threat-Matrix (multiclass)...")
    try:
        ds = load_dataset("neuralchemy/prompt-injection-Threat-Matrix", "multiclass")
        print(f"  Train: {len(ds['train'])}, Val: {len(ds['validation'])}, Test: {len(ds['test'])}")

        test = ds['test']

        # Group by intent category
        categories = {}
        for sample in test:
            intent = sample['intent']
            label = sample['binary_label']
            text = sample['text']
            severity = sample['severity']

            if intent not in categories:
                categories[intent] = {"trigger": [], "benign": [], "severities": []}

            if label == 1:
                categories[intent]["trigger"].append(text)
                categories[intent]["severities"].append(severity)
            else:
                categories[intent]["benign"].append(text)

        print("\n  Category breakdown:")
        for cat, data in sorted(categories.items()):
            sev_info = f", severity range: {min(data['severities'])}-{max(data['severities'])}" if data['severities'] else ""
            print(f"    {cat}: {len(data['trigger'])} trigger, {len(data['benign'])} benign{sev_info}")

        all_trigger = [s['text'] for s in test if s['binary_label'] == 1]
        all_benign = [s['text'] for s in test if s['binary_label'] == 0]
        print(f"\n  Total: {len(all_trigger)} trigger, {len(all_benign)} benign")

        return {
            "categories": categories,
            "all_trigger": all_trigger,
            "all_benign": all_benign
        }
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 60)
    print("PROMPT INJECTION BENCHMARK DOWNLOAD")
    print("=" * 60)

    data = download_threat_matrix()
    if data is None:
        print("Failed to download dataset.")
        return

    out = Path("benchmark_data")
    out.mkdir(exist_ok=True)

    # Save flat files for overall analysis
    with open(out / "test_trigger.json", "w") as f:
        json.dump(data["all_trigger"], f, indent=2)
    with open(out / "test_benign.json", "w") as f:
        json.dump(data["all_benign"], f, indent=2)

    # Save per-category files
    cat_dir = out / "categories"
    cat_dir.mkdir(exist_ok=True)
    for cat, cat_data in data["categories"].items():
        with open(cat_dir / f"{cat}_trigger.json", "w") as f:
            json.dump(cat_data["trigger"], f, indent=2)
        with open(cat_dir / f"{cat}_benign.json", "w") as f:
            json.dump(cat_data["benign"], f, indent=2)

    print(f"\nSaved to {out}/")
    print("Run '/benchmark' in AMDON to analyze.")


if __name__ == "__main__":
    main()
