"""
Length-scaled delta angle computation.
Sentence-boundary chunking + length-scaled softmax + merge threshold.
"""
import numpy as np
import json
import time
import requests
import os
import re
from collections import defaultdict
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, roc_curve

NIM_API_KEY = os.getenv('NIM_API_KEY', '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')
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


def nim_embed(texts, model="nvidia/nv-embedqa-e5-v5", batch_size=64):
    """Embed texts using NIM API with robust error handling."""
    all_embeddings = [None] * len(texts)
    headers = {"Authorization": f"Bearer {NIM_API_KEY}", "Content-Type": "application/json"}
    
    def clean_text(t):
        t = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', t)
        t = re.sub(r'[\U0001D400-\U0001D7FF\U00002100-\U0000214F\U0000FE00-\U0000FE0F\U0000200B-\U0000200F\U00002028-\U0000202F]', '', t)
        t = re.sub(r'(\\[.\-/\\]){3,}', ' ', t)
        t = re.sub(r'\\{2,}', ' ', t)
        t = re.sub(r'\s{2,}', ' ', t)
        t = t[:1600].strip()
        return t if t else "empty chunk"
    
    # Batch embed
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        clean_batch = [clean_text(t) for t in batch]
        
        payload = {"model": model, "input": clean_batch, "input_type": "passage"}
        try:
            response = requests.post(f"{NIM_BASE_URL}/embeddings", headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            sorted_data = sorted(data['data'], key=lambda x: x['index'])
            for j, item in enumerate(sorted_data):
                all_embeddings[i + j] = np.array(item['embedding'], dtype=np.float32)
        except Exception as e:
            # Retry failed items individually
            for j, t in enumerate(clean_batch):
                idx = i + j
                if all_embeddings[idx] is not None:
                    continue
                try:
                    r = requests.post(f"{NIM_BASE_URL}/embeddings", 
                                    headers=headers, 
                                    json={"model": model, "input": [t], "input_type": "passage"}, 
                                    timeout=30)
                    r.raise_for_status()
                    all_embeddings[idx] = np.array(r.json()['data'][0]['embedding'], dtype=np.float32)
                except:
                    # Last resort: use zero embedding
                    all_embeddings[idx] = np.zeros(4096, dtype=np.float32)
                    print(f"  WARN: chunk {idx} failed permanently")
                time.sleep(0.1)
    
    return all_embeddings


def chunk_text(text, merge_threshold=8):
    """
    Chunk text using sentence boundaries, clause boundaries, and merge threshold.
    
    1. Split at sentence boundaries: (?<=[.!?])\\s+
    2. Split long sentences at commas/semicolons
    3. Merge chunks < merge_threshold words with neighbor
    """
    # Step 1: Split at sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Step 2: Split long sentences at commas/semicolons
    chunks = []
    for s in sentences:
        if not s.strip():
            continue
        if len(s) > 50:
            for sub in re.split(r'(?<=[,;])\s+', s):
                if sub.strip():
                    chunks.append(sub.strip())
        else:
            chunks.append(s.strip())
    
    # Step 3: Fallback if still single chunk
    if len(chunks) < 2 and len(text.split()) > 10:
        words = text.split()
        mid = len(words) // 2
        chunks = [' '.join(words[:mid]), ' '.join(words[mid:])]
    
    if not chunks:
        chunks = [text]
    
    # Step 4: Merge chunks below threshold with neighbor
    merged = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        current_words = len(current.split())
        
        # If below threshold, merge with next (or previous if at end)
        if current_words < merge_threshold:
            if i + 1 < len(chunks):
                # Merge with next
                chunks[i + 1] = current + ' ' + chunks[i + 1]
            elif merged:
                # Merge with previous
                merged[-1] = merged[-1] + ' ' + current
            else:
                merged.append(current)
        else:
            merged.append(current)
        
        i += 1
    
    return merged if merged else [text]


def compute_signed_angle(e1, e2):
    """Compute signed angle between two vectors using Gram-Schmidt."""
    n1 = np.linalg.norm(e1)
    if n1 < 1e-10:
        return 0.0
    e1n = e1 / n1
    
    n2 = np.linalg.norm(e2)
    if n2 < 1e-10:
        return 0.0
    
    orth = e2 - np.dot(e2, e1n) * e1n
    sign = 1 if np.dot(orth, np.ones_like(orth)) >= 0 else -1
    angle = sign * np.arccos(np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1))
    return np.degrees(angle)


def length_scaled_delta(chunks, embeddings, alpha=1.0, tau=0.5):
    """
    Compute length-scaled delta angle.
    
    θ(x) = (Σᵢ wᵢ × θᵢ) / Σⱼ len(cⱼ)
    
    Where wᵢ = (len(cᵢ)^α × exp(θᵢ/τ)) / Σⱼ (len(cⱼ)^α × exp(θⱼ/τ))
    """
    if len(embeddings) < 2:
        return 0.0, []
    
    # Compute angles between consecutive chunks
    angles = []
    chunk_lengths = []
    
    for i in range(len(embeddings) - 1):
        angle = compute_signed_angle(embeddings[i], embeddings[i + 1])
        angles.append(angle)
        chunk_lengths.append(len(chunks[i].split()))
    
    # Add last chunk length
    chunk_lengths.append(len(chunks[-1].split()))
    
    if not angles:
        return 0.0, []
    
    angles = np.array(angles)
    chunk_lengths = np.array(chunk_lengths[:-1])  # Match angles length
    
    # Length-scaled softmax weights
    # wᵢ = softmax(len(cᵢ)^α × θᵢ/τ)
    length_factor = np.power(chunk_lengths, alpha)
    angle_logits = angles / tau
    # Combine length and angle into a single logit
    logits = length_factor * angle_logits
    weights = softmax(logits)
    
    # Weighted average angle (no division by total length)
    delta = np.sum(weights * angles)
    
    return float(delta), angles.tolist()


def compute_length_scaled_deltas(texts, model="nvidia/nv-embedqa-e5-v5", alpha=1.0, tau=0.5, merge_threshold=8):
    """
    Compute length-scaled deltas for multiple texts.
    """
    # Chunk all texts
    all_chunks = []
    text_chunk_ranges = []
    
    for text in texts:
        chunks = chunk_text(text, merge_threshold=merge_threshold)
        start = len(all_chunks)
        all_chunks.extend(chunks)
        end = len(all_chunks)
        text_chunk_ranges.append((start, end))
    
    print(f"  Total chunks: {len(all_chunks)}")
    
    # Embed all chunks
    all_embeddings = nim_embed(all_chunks, model=model)
    if all_embeddings is None or len(all_embeddings) != len(all_chunks):
        print(f"  Embedding mismatch: got {len(all_embeddings) if all_embeddings else 0}/{len(all_chunks)}")
        return [0.0] * len(texts), [[] for _ in texts]
    
    # Compute deltas
    deltas = []
    all_angles = []
    
    for text, (start, end) in zip(texts, text_chunk_ranges):
        chunks = all_chunks[start:end]
        embeddings = all_embeddings[start:end]
        
        delta, angles = length_scaled_delta(chunks, embeddings, alpha=alpha, tau=tau)
        deltas.append(delta)
        all_angles.append(angles)
    
    return deltas, all_angles


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Length-scaled delta angle computation")
    parser.add_argument("--alpha", type=float, default=1.0, help="Length scaling exponent")
    parser.add_argument("--tau", type=float, default=0.5, help="Softmax temperature")
    parser.add_argument("--merge", type=int, default=8, help="Merge threshold (words)")
    parser.add_argument("--model", default="nvidia/nv-embedqa-e5-v5", help="NIM model")
    parser.add_argument("--compare", action="store_true", help="Compare multiple alpha values")
    
    args = parser.parse_args()
    
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
    
    # Combine and deduplicate
    combined = {}
    for s in obf_samples + ben_samples:
        combined[s['text']] = s
    
    texts = list(combined.keys())
    labels = [combined[t]['label'] for t in texts]
    
    print(f"Total: {len(texts)} | obf={labels.count('obfuscation')} ben={labels.count('benign')}")
    
    if args.compare:
        # Compare multiple alpha values
        alphas = [0.5, 1.0, 1.5, 2.0]
        results = {}
        
        for alpha in alphas:
            print(f"\n{'='*60}")
            print(f"Alpha = {alpha}")
            print(f"{'='*60}")
            
            deltas, angles = compute_length_scaled_deltas(
                texts, model=args.model, alpha=alpha, tau=args.tau, merge_threshold=args.merge
            )
            
            # Compute metrics
            deltas_by_label = defaultdict(list)
            for d, l in zip(deltas, labels):
                deltas_by_label[l].append(d)
            
            ben = np.array(deltas_by_label['benign'])
            obf = np.array(deltas_by_label['obfuscation'])
            
            sep = abs(np.mean(ben) - np.mean(obf))
            
            try:
                from sklearn.metrics import roc_auc_score, roc_curve as _roc_curve
                y = [0]*len(ben) + [1]*len(obf)
                sc = list(ben) + list(obf)
                auc = max(roc_auc_score(y, sc), roc_auc_score(y, [-s for s in sc]))
            except:
                auc = 0.0
            
            # Determine direction
            higher_is_obf = np.mean(obf) > np.mean(ben)
            
            bf1 = 0
            bt = 0
            if higher_is_obf:
                # Higher delta = obfuscation
                for pct in range(1, 51):
                    t = np.percentile(obf, pct)
                    pred = np.array(deltas)[np.array(labels) == 'obfuscation'] > t
                    tp = np.sum(pred)
                    fn = np.sum(~pred)
                    fp = np.sum(np.array(deltas)[np.array(labels) == 'benign'] > t)
                    p = tp / (tp + fp) if (tp + fp) > 0 else 0
                    r = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                    if f1 > bf1:
                        bf1 = f1
                        bt = t
                direction = ">"
            else:
                # Lower delta = obfuscation
                for pct in range(50, 100):
                    t = np.percentile(ben, pct)
                    pred = np.array(deltas)[np.array(labels) == 'obfuscation'] < t
                    tp = np.sum(pred)
                    fn = np.sum(~pred)
                    fp = np.sum(np.array(deltas)[np.array(labels) == 'benign'] < t)
                    p = tp / (tp + fp) if (tp + fp) > 0 else 0
                    r = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                    if f1 > bf1:
                        bf1 = f1
                        bt = t
                direction = "<"
            
            # Optimal FPR/TPR
            y = [0]*len(ben) + [1]*len(obf)
            if higher_is_obf:
                sc = list(ben) + list(obf)
            else:
                sc = [-d for d in list(ben) + list(obf)]
            fpr_a, tpr_a, _ = roc_curve(y, sc)
            j = tpr_a - fpr_a
            idx = np.argmax(j)
            
            results[alpha] = {
                'auc': auc,
                'f1': bf1,
                'threshold': bt,
                'direction': direction,
                'sep': sep,
                'ben_mean': float(np.mean(ben)),
                'obf_mean': float(np.mean(obf)),
                'ben_std': float(np.std(ben)),
                'obf_std': float(np.std(obf)),
                'optimal_fpr': float(fpr_a[idx]),
                'optimal_tpr': float(tpr_a[idx]),
            }
            
            print(f"  BENIGN:  μ={np.mean(ben):.6f} σ={np.std(ben):.6f}")
            print(f"  OBF:     μ={np.mean(obf):.6f} σ={np.std(obf):.6f}")
            print(f"  Sep={sep:.6f} AUC={auc:.3f} F1={bf1:.3f} @ {direction} {bt:.6f}")
            print(f"  Optimal: FPR={fpr_a[idx]:.3f} TPR={tpr_a[idx]:.3f}")
        
        # Print comparison table
        print(f"\n\n{'='*80}")
        print("ALPHA COMPARISON")
        print(f"{'='*80}")
        print(f"{'Alpha':>6} {'AUC':>6} {'F1':>6} {'Sep':>10} {'Ben μ':>10} {'Obf μ':>10}")
        print(f"{'-'*80}")
        for alpha, r in sorted(results.items()):
            print(f"{alpha:>6.1f} {r['auc']:>6.3f} {r['f1']:>6.3f} {r['sep']:>10.6f} {r['ben_mean']:>10.6f} {r['obf_mean']:>10.6f}")
        
        # Save results
        with open("data/length_scaled_comparison.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to data/length_scaled_comparison.json")
    
    else:
        # Single run
        print(f"\nRunning with alpha={args.alpha}, tau={args.tau}, merge={args.merge}")
        
        deltas, angles = compute_length_scaled_deltas(
            texts, model=args.model, alpha=args.alpha, tau=args.tau, merge_threshold=args.merge
        )
        
        # Compute metrics
        deltas_by_label = defaultdict(list)
        for d, l in zip(deltas, labels):
            deltas_by_label[l].append(d)
        
        ben = np.array(deltas_by_label['benign'])
        obf = np.array(deltas_by_label['obfuscation'])
        
        sep = abs(np.mean(ben) - np.mean(obf))
        
        try:
            from sklearn.metrics import roc_auc_score
            y = [0]*len(ben) + [1]*len(obf)
            sc = list(ben) + list(obf)
            auc = max(roc_auc_score(y, sc), roc_auc_score(y, [-s for s in sc]))
        except:
            auc = 0.0
        
        bf1 = 0
        bt = 0
        for pct in range(50, 100):
            t = np.percentile(ben, pct)
            pred = obf < t
            tp = np.sum(pred)
            fn = np.sum(~pred)
            fp = np.sum(ben < t)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > bf1:
                bf1 = f1
                bt = t
        
        print(f"\nBENIGN:  μ={np.mean(ben):.6f} σ={np.std(ben):.6f}")
        print(f"OBF:     μ={np.mean(obf):.6f} σ={np.std(obf):.6f}")
        print(f"Sep={sep:.6f} AUC={auc:.3f} F1={bf1:.3f} @ {bt:.6f}")
        
        # Save results
        results = {
            'model': args.model,
            'alpha': args.alpha,
            'tau': args.tau,
            'merge_threshold': args.merge,
            'benign_mean': float(np.mean(ben)),
            'benign_std': float(np.std(ben)),
            'obf_mean': float(np.mean(obf)),
            'obf_std': float(np.std(obf)),
            'separation': float(sep),
            'auc': float(auc),
            'best_f1': float(bf1),
            'best_threshold': float(bt),
        }
        
        with open("data/length_scaled_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to data/length_scaled_results.json")


if __name__ == "__main__":
    main()
