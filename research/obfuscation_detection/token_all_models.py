"""
Token-based delta angle computation for ALL NIM models.
Tests 6 working models with token-based approach.
"""
import numpy as np
import json
import time
import requests
import os
import re
from collections import defaultdict

NIM_API_KEY = os.getenv('NIM_API_KEY', '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')
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

MODELS = [
    "nvidia/nv-embedqa-e5-v5",
    "nvidia/llama-nemotron-embed-1b-v2",
    "baai/bge-m3",
    "nvidia/nv-embed-v1",
    "nvidia/nv-embedcode-7b-v1",
    "nvidia/llama-nemotron-embed-vl-1b-v2",
]

def nim_embed(texts, model, batch_size=256):
    """Embed texts using NIM API."""
    all_embeddings = []
    headers = {"Authorization": f"Bearer {NIM_API_KEY}", "Content-Type": "application/json"}
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {"model": model, "input": batch, "input_type": "passage"}
        for attempt in range(3):
            try:
                response = requests.post(f"{NIM_BASE_URL}/embeddings", headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                sorted_data = sorted(data['data'], key=lambda x: x['index'])
                all_embeddings.extend([np.array(item['embedding'], dtype=np.float32) for item in sorted_data])
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  FATAL: {e}")
                    return None
                time.sleep(1.5 ** attempt)
    return all_embeddings


def compute_signed_angle(e1, e2):
    """Compute signed angle between two vectors."""
    n1 = np.linalg.norm(e1)
    if n1 < 1e-10: return 0.0
    e1n = e1 / n1
    n2 = np.linalg.norm(e2)
    if n2 < 1e-10: return 0.0
    orth = e2 - np.dot(e2, e1n) * e1n
    sign = 1 if np.dot(orth, np.ones_like(orth)) >= 0 else -1
    angle = sign * np.arccos(np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1))
    return np.degrees(angle)


def tokenize(text):
    """Tokenize text into words."""
    return re.findall(r'\b\w+\b', text.lower())


def main():
    # Load samples
    print("Loading samples...")
    obf_trigger = json.load(open("data/obf_trigger.json"))
    obf_benign = json.load(open("data/obf_benign.json"))
    
    if isinstance(obf_trigger[0], str):
        obf_samples = [{'text': t, 'label': 'obfuscation'} for t in obf_trigger]
    else:
        obf_samples = obf_trigger
    if isinstance(obf_benign[0], str):
        ben_samples = [{'text': t, 'label': 'benign'} for t in obf_benign]
    else:
        ben_samples = obf_benign
    
    combined = {}
    for s in obf_samples + ben_samples:
        combined[s['text']] = s
    
    texts = list(combined.keys())
    labels = [combined[t]['label'] for t in texts]
    print(f"Total: {len(texts)} | obf={labels.count('obfuscation')} ben={labels.count('benign')}")
    
    # Tokenize all texts
    all_tokens = []
    text_token_ranges = []
    for text in texts:
        tokens = tokenize(text)
        start = len(all_tokens)
        all_tokens.extend(tokens)
        end = len(all_tokens)
        text_token_ranges.append((start, end))
    
    print(f"Total tokens: {len(all_tokens)}")
    
    # Test each model
    all_model_results = {}
    
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")
        
        # Embed all tokens
        start_time = time.time()
        all_embeddings = []
        chunk_size = 10000
        
        for i in range(0, len(all_tokens), chunk_size):
            chunk = all_tokens[i:i + chunk_size]
            print(f"  Embedding tokens {i}-{i+len(chunk)}...")
            chunk_emb = nim_embed(chunk, model=model, batch_size=256)
            if chunk_emb is None:
                print(f"  FAILED - skipping model")
                break
            all_embeddings.extend(chunk_emb)
        
        if len(all_embeddings) != len(all_tokens):
            print(f"  Expected {len(all_tokens)} embeddings, got {len(all_embeddings)} - skipping")
            continue
        
        elapsed = time.time() - start_time
        print(f"  Embedding time: {elapsed:.1f}s")
        
        # Compute deltas
        deltas = []
        for start, end in text_token_ranges:
            tok_emb = all_embeddings[start:end]
            if len(tok_emb) < 2:
                deltas.append(0.0)
                continue
            
            angles = []
            for i in range(len(tok_emb) - 1):
                angle = compute_signed_angle(tok_emb[i], tok_emb[i+1])
                if angle != 0.0:
                    angles.append(angle)
            
            deltas.append(float(np.mean(angles)) if angles else 0.0)
        
        # Compute metrics
        deltas_by_label = defaultdict(list)
        for d, l in zip(deltas, labels):
            deltas_by_label[l].append(d)
        
        ben = np.array(deltas_by_label['benign'])
        obf = np.array(deltas_by_label['obfuscation'])
        
        sep = abs(np.mean(ben) - np.mean(obf))
        
        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            y = [0]*len(ben) + [1]*len(obf)
            scores = list(ben) + list(obf)
            auc_direct = roc_auc_score(y, scores)
            auc_neg = roc_auc_score(y, [-s for s in scores])
            auc = max(auc_direct, auc_neg)
        except:
            auc = 0.0
        
        # Best F1
        best_f1 = 0
        best_threshold = 0
        for pct in range(50, 100):
            t = np.percentile(ben, pct)
            pred = obf < t
            tp = np.sum(pred)
            fn = np.sum(~pred)
            fp = np.sum(ben < t)
            tn = np.sum(ben >= t)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        
        # Optimal FPR/TPR
        try:
            from sklearn.metrics import roc_curve
            y = [0]*len(ben) + [1]*len(obf)
            scores = [-d for d in list(ben) + list(obf)]  # negated
            fpr, tpr, _ = roc_curve(y, scores)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_fpr = fpr[optimal_idx]
            optimal_tpr = tpr[optimal_idx]
        except:
            optimal_fpr = 0
            optimal_tpr = 0
        
        # Zero deltas
        ben_zeros = np.sum(np.abs(ben) < 0.01)
        obf_zeros = np.sum(np.abs(obf) < 0.01)
        
        result = {
            'model': model,
            'time': elapsed,
            'ben_mean': float(np.mean(ben)),
            'ben_std': float(np.std(ben)),
            'obf_mean': float(np.mean(obf)),
            'obf_std': float(np.std(obf)),
            'separation': float(sep),
            'auc': float(auc),
            'best_f1': float(best_f1),
            'best_threshold': float(best_threshold),
            'optimal_fpr': float(optimal_fpr),
            'optimal_tpr': float(optimal_tpr),
            'ben_zeros': int(ben_zeros),
            'obf_zeros': int(obf_zeros),
            'ben_n': len(ben),
            'obf_n': len(obf),
        }
        
        all_model_results[model] = result
        
        # Print summary
        print(f"\n  BENIGN:    μ={np.mean(ben):.3f}° σ={np.std(ben):.3f}° (zeros: {ben_zeros}/{len(ben)})")
        print(f"  OBFUSCATION: μ={np.mean(obf):.3f}° σ={np.std(obf):.3f}° (zeros: {obf_zeros}/{len(obf)})")
        print(f"  Separation: {sep:.3f}°")
        print(f"  AUC: {auc:.3f}")
        print(f"  Best F1: {best_f1:.3f} @ {best_threshold:.2f}°")
        print(f"  Optimal: FPR={optimal_fpr:.3f} TPR={optimal_tpr:.3f}")
    
    # Save results
    os.makedirs("data", exist_ok=True)
    with open("data/token_based_all_models.json", "w") as f:
        json.dump(all_model_results, f, indent=2)
    
    # Print comparison table
    print(f"\n\n{'='*100}")
    print("MODEL COMPARISON - TOKEN-BASED DELTA ANGLE")
    print(f"{'='*100}")
    print(f"{'Model':<45} {'AUC':>6} {'Sep':>8} {'F1':>6} {'FPR':>6} {'TPR':>6} {'Time':>8}")
    print(f"{'-'*100}")
    
    for model, r in sorted(all_model_results.items(), key=lambda x: -x[1]['auc']):
        short_name = model.split('/')[-1][:44]
        print(f"{short_name:<45} {r['auc']:>6.3f} {r['separation']:>7.3f}° {r['best_f1']:>6.3f} {r['optimal_fpr']:>6.3f} {r['optimal_tpr']:>6.3f} {r['time']:>7.1f}s")
    
    print(f"\nCached to data/token_based_all_models.json")


if __name__ == "__main__":
    main()
