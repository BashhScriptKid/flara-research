"""
Token-based delta angle computation using NVIDIA NIM API.
Optimized for batch processing.
"""
import numpy as np
import requests
import json
import time
import os
import re
from collections import defaultdict

NIM_API_KEY = os.getenv('NIM_API_KEY', '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')
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

def nim_embed(texts, model="nvidia/nv-embedqa-e5-v5", batch_size=256):
    """Embed texts using NIM API with batching."""
    all_embeddings = []
    headers = {
        "Authorization": f"Bearer {NIM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {
            "model": model,
            "input": batch,
            "input_type": "passage",
        }
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{NIM_BASE_URL}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                data = response.json()
                sorted_data = sorted(data['data'], key=lambda x: x['index'])
                batch_embeddings = [np.array(item['embedding'], dtype=np.float32) for item in sorted_data]
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"FATAL: {e}")
                    return None
                time.sleep(1.5 ** attempt)
    
    return all_embeddings


def tokenize_text(text):
    """Tokenize text into words."""
    return re.findall(r'\b\w+\b', text.lower())


def compute_signed_angle(e1, e2):
    """Compute signed angle between two vectors using Gram-Schmidt."""
    norm_e1 = np.linalg.norm(e1)
    if norm_e1 < 1e-10:
        return 0.0
    e1_norm = e1 / norm_e1
    
    proj = np.dot(e2, e1_norm) * e1_norm
    orth = e2 - proj
    norm_e2 = np.linalg.norm(e2)
    
    if norm_e2 < 1e-10:
        return 0.0
    
    sign = 1 if np.dot(orth, np.ones_like(orth)) >= 0 else -1
    angle = sign * np.arccos(np.clip(
        np.dot(e1, e2) / (norm_e1 * norm_e2), -1.0, 1.0
    ))
    return np.degrees(angle)


def compute_token_deltas_batch(texts, model="nvidia/nv-embedqa-e5-v5"):
    """
    Compute token deltas for multiple texts efficiently.
    Batches all tokens from all texts in a single API call.
    """
    results = []
    
    # Collect all tokens from all texts
    all_tokens = []
    text_token_ranges = []
    
    for text in texts:
        tokens = tokenize_text(text)
        start = len(all_tokens)
        all_tokens.extend(tokens)
        end = len(all_tokens)
        text_token_ranges.append((start, end))
    
    if not all_tokens:
        return [(0.0, []) for _ in texts]
    
    print(f"  Total tokens: {len(all_tokens)}")
    
    # Embed all tokens in a single large batch
    start_time = time.time()
    all_embeddings = nim_embed(all_tokens, model=model, batch_size=256)
    embed_time = time.time() - start_time
    print(f"  Embedding time: {embed_time:.1f}s")
    
    if all_embeddings is None:
        return [(0.0, []) for _ in texts]
    
    # Compute deltas for each text
    for text, (start, end) in zip(texts, text_token_ranges):
        tokens = all_tokens[start:end]
        embeddings = all_embeddings[start:end]
        
        if len(embeddings) < 2:
            results.append((0.0, tokens))
            continue
        
        angles = []
        for i in range(len(embeddings) - 1):
            angle = compute_signed_angle(embeddings[i], embeddings[i + 1])
            if angle != 0.0:
                angles.append(angle)
        
        avg_delta = np.mean(angles) if angles else 0.0
        results.append((avg_delta, tokens))
    
    return results


def fetch_and_cache_token_deltas(output_file="data/delta_cache_token.json"):
    """Fetch token-based deltas for all samples and cache."""
    print("Loading samples...")
    obf_trigger = json.load(open("data/obf_trigger.json"))
    obf_benign = json.load(open("data/obf_benign.json"))
    
    # Handle both string lists and dict lists
    if isinstance(obf_trigger[0], str):
        obf_samples = [{'text': t, 'label': 'obfuscation'} for t in obf_trigger]
    else:
        obf_samples = obf_trigger
    
    if isinstance(obf_benign[0], str):
        ben_samples = [{'text': t, 'label': 'benign'} for t in obf_benign]
    else:
        ben_samples = obf_benign
    
    # Combine and deduplicate
    combined = {}
    for sample in obf_samples:
        combined[sample['text']] = sample
    for sample in ben_samples:
        combined[sample['text']] = sample
    
    texts = list(combined.keys())
    labels = [combined[text]['label'] for text in texts]
    
    print(f"Total unique texts: {len(texts)}")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Process in chunks to avoid memory issues
    chunk_size = 100
    all_results = {}
    
    model = "nvidia/nv-embedqa-e5-v5"
    print(f"\nModel: {model}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for i in range(0, len(texts), chunk_size):
        chunk_texts = texts[i:i + chunk_size]
        chunk_labels = labels[i:i + chunk_size]
        
        print(f"\nChunk {i//chunk_size + 1}/{(len(texts) + chunk_size - 1)//chunk_size}")
        results = compute_token_deltas_batch(chunk_texts, model=model)
        
        for text, label, (delta, tokens) in zip(chunk_texts, chunk_labels, results):
            if model not in all_results:
                all_results[model] = []
            all_results[model].append({
                'text': text,
                'label': label,
                'delta': float(delta),
                'token_count': len(tokens)
            })
    
    elapsed = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    
    for model, results in all_results.items():
        deltas_by_label = defaultdict(list)
        for r in results:
            deltas_by_label[r['label']].append(r['delta'])
        
        print(f"\nModel: {model}")
        print(f"Time: {elapsed:.1f}s")
        
        for label in sorted(deltas_by_label.keys()):
            vals = deltas_by_label[label]
            print(f"  {label}: μ={np.mean(vals):.3f}° σ={np.std(vals):.3f}° n={len(vals)}")
        
        # Compute metrics
        if 'benign' in deltas_by_label and 'obfuscation' in deltas_by_label:
            ben = np.array(deltas_by_label['benign'])
            obf = np.array(deltas_by_label['obfuscation'])
            
            # Percentile threshold
            threshold = np.percentile(ben, 95)
            pred_obf = obf < threshold
            
            tp = np.sum(pred_obf)
            fn = np.sum(~pred_obf)
            fp = np.sum(ben < threshold)
            tn = np.sum(ben >= threshold)
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            print(f"\n  Threshold (95th pct ben): {threshold:.3f}°")
            print(f"  Recall: {recall:.1%} ({tp}/{tp+fn})")
            print(f"  FPR: {fpr:.1%} ({fp}/{fp+tn})")
    
    # Save cache
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nCached to {output_file}")
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Token-based delta angle computation")
    parser.add_argument("--fetch", action="store_true", help="Fetch and cache token deltas")
    parser.add_argument("--output", default="data/delta_cache_token.json", help="Output cache file")
    
    args = parser.parse_args()
    
    if args.fetch:
        fetch_and_cache_token_deltas(args.output)
    else:
        # Default: fetch token deltas
        fetch_and_cache_token_deltas(args.output)
