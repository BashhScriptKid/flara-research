"""
Delta angle computation: plain sentence chunking + merge + average.
No softmax, no length scaling. Just chunk → embed → angle → average.
"""
import numpy as np
import json
import re
import requests
import time
import os
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve

NIM_API_KEY = os.getenv('NIM_API_KEY', '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')
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


def chunk_text(text, merge_threshold=8):
    """Sentence boundaries → clause boundaries → merge short chunks."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
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
    
    if len(chunks) < 2 and len(text.split()) > 10:
        words = text.split()
        mid = len(words) // 2
        chunks = [' '.join(words[:mid]), ' '.join(words[mid:])]
    
    if not chunks:
        chunks = [text]
    
    # Merge chunks below threshold with next neighbor
    merged = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        if len(current.split()) < merge_threshold:
            if i + 1 < len(chunks):
                chunks[i + 1] = current + ' ' + chunks[i + 1]
            elif merged:
                merged[-1] = merged[-1] + ' ' + current
            else:
                merged.append(current)
        else:
            merged.append(current)
        i += 1
    
    return merged if merged else [text]


def compute_angle(e1, e2):
    """Unsigned angle between two embedding vectors."""
    n1 = np.linalg.norm(e1)
    if n1 < 1e-10:
        return 0.0
    n2 = np.linalg.norm(e2)
    if n2 < 1e-10:
        return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)))


def nim_embed(texts, model="nvidia/nv-embedqa-e5-v5", batch_size=64):
    """Embed texts with retry and cleaning."""
    all_emb = [None] * len(texts)
    headers = {"Authorization": f"Bearer {NIM_API_KEY}", "Content-Type": "application/json"}
    
    def clean(t):
        t = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', t)
        t = re.sub(r'[\U0001D400-\U0001D7FF]', '', t)
        t = re.sub(r'(\\[.\-/\\]){3,}', ' ', t)
        t = re.sub(r'\\{2,}', ' ', t)
        t = re.sub(r'\s{2,}', ' ', t)
        return t[:1600].strip() or "empty"
    
    for i in range(0, len(texts), batch_size):
        batch = [clean(t) for t in texts[i:i + batch_size]]
        try:
            r = requests.post(f"{NIM_BASE_URL}/embeddings", headers=headers,
                            json={"model": model, "input": batch, "input_type": "passage"}, timeout=60)
            r.raise_for_status()
            for j, item in enumerate(sorted(r.json()['data'], key=lambda x: x['index'])):
                all_emb[i + j] = np.array(item['embedding'], dtype=np.float32)
        except:
            for j, t in enumerate(batch):
                idx = i + j
                if all_emb[idx] is not None:
                    continue
                try:
                    r2 = requests.post(f"{NIM_BASE_URL}/embeddings", headers=headers,
                                     json={"model": model, "input": [t], "input_type": "passage"}, timeout=30)
                    r2.raise_for_status()
                    all_emb[idx] = np.array(r2.json()['data'][0]['embedding'], dtype=np.float32)
                except:
                    all_emb[idx] = np.zeros(4096, dtype=np.float32)
                    print(f"  WARN: chunk {idx} failed")
                time.sleep(0.1)
    
    return all_emb


def compute_deltas(texts, model="nvidia/nv-embedqa-e5-v5", merge_threshold=8):
    """Compute delta angles for texts using sentence chunking."""
    # Chunk all texts
    all_chunks = []
    text_ranges = []
    for text in texts:
        chunks = chunk_text(text, merge_threshold)
        start = len(all_chunks)
        all_chunks.extend(chunks)
        text_ranges.append((start, len(all_chunks)))
    
    # Embed
    all_emb = nim_embed(all_chunks, model=model)
    
    # Compute average angle per text
    deltas = []
    for si, ei in text_ranges:
        emb = all_emb[si:ei]
        if len(emb) < 2:
            deltas.append(0.0)
            continue
        angles = [compute_angle(emb[i], emb[i + 1]) for i in range(len(emb) - 1)]
        angles = [a for a in angles if a != 0.0]
        deltas.append(float(np.mean(angles)) if angles else 0.0)
    
    return deltas, len(all_chunks)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/nv-embedqa-e5-v5")
    parser.add_argument("--merge", type=int, default=8)
    parser.add_argument("--compare-models", action="store_true")
    args = parser.parse_args()
    
    # Load data
    print("Loading samples...")
    obf_trigger = json.load(open("data/obf_trigger.json"))
    obf_benign = json.load(open("data/obf_benign.json"))
    
    if isinstance(obf_trigger[0], str):
        obf_samples = obf_trigger
    else:
        obf_samples = [s['text'] for s in obf_trigger]
    if isinstance(obf_benign[0], str):
        ben_samples = obf_benign
    else:
        ben_samples = [s['text'] for s in obf_benign]
    
    combined = {}
    for t in obf_samples: combined[t] = "obfuscation"
    for t in ben_samples: combined[t] = "benign"
    
    texts = list(combined.keys())
    labels = [combined[t] for t in texts]
    
    print(f"Total: {len(texts)} | obf={labels.count('obfuscation')} ben={labels.count('benign')}")
    
    if args.compare_models:
        models = [
            "nvidia/nv-embedqa-e5-v5",
            "nvidia/llama-nemotron-embed-1b-v2",
            "baai/bge-m3",
        ]
        results = {}
        for model in models:
            print(f"\n{'='*60}")
            print(f"Model: {model}")
            deltas, n_chunks = compute_deltas(texts, model=model, merge_threshold=args.merge)
            print(f"Chunks: {n_chunks}")
            
            deltas = np.array(deltas)
            labels_arr = np.array(labels)
            ben = deltas[labels_arr == 'benign']
            obf = deltas[labels_arr == 'obfuscation']
            
            sep = abs(np.mean(ben) - np.mean(obf))
            y = [0] * len(ben) + [1] * len(obf)
            sc = list(ben) + list(obf)
            auc = max(roc_auc_score(y, sc), roc_auc_score(y, [-s for s in sc]))
            
            # F1
            bf1 = 0; bt = 0
            for pct in range(1, 51):
                t = np.percentile(obf, pct)
                pred = deltas[labels_arr == 'obfuscation'] > t
                tp = np.sum(pred); fn = np.sum(~pred)
                fp = np.sum(deltas[labels_arr == 'benign'] > t)
                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                if f1 > bf1: bf1 = f1; bt = t
            
            fpr_a, tpr_a, _ = roc_curve(y, sc)
            j = tpr_a - fpr_a; idx = np.argmax(j)
            
            results[model] = {
                'auc': auc, 'f1': bf1, 'threshold': bt, 'sep': sep,
                'ben_mean': float(np.mean(ben)), 'obf_mean': float(np.mean(obf)),
                'ben_std': float(np.std(ben)), 'obf_std': float(np.std(obf)),
                'fpr': float(fpr_a[idx]), 'tpr': float(tpr_a[idx]),
            }
            
            print(f"  BENIGN:  μ={np.mean(ben):.6f} σ={np.std(ben):.6f}")
            print(f"  OBF:     μ={np.mean(obf):.6f} σ={np.std(obf):.6f}")
            print(f"  Sep={sep:.6f} AUC={auc:.3f} F1={bf1:.3f} @ > {bt:.6f}")
            print(f"  Optimal: FPR={fpr_a[idx]:.3f} TPR={tpr_a[idx]:.3f}")
        
        # Comparison table
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}")
        print(f"{'Model':<40} {'AUC':>6} {'F1':>6} {'FPR':>6} {'TPR':>6}")
        print(f"{'-'*80}")
        for m, r in sorted(results.items(), key=lambda x: -x[1]['auc']):
            print(f"{m.split('/')[-1]:<40} {r['auc']:>6.3f} {r['f1']:>6.3f} {r['fpr']:>6.3f} {r['tpr']:>6.3f}")
        
        with open("data/sentence_chunk_models.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to data/sentence_chunk_models.json")
    
    else:
        print(f"\nModel: {args.model} | Merge: {args.merge}")
        deltas, n_chunks = compute_deltas(texts, model=args.model, merge_threshold=args.merge)
        print(f"Chunks: {n_chunks}")
        
        deltas = np.array(deltas)
        labels_arr = np.array(labels)
        ben = deltas[labels_arr == 'benign']
        obf = deltas[labels_arr == 'obfuscation']
        
        sep = abs(np.mean(ben) - np.mean(obf))
        y = [0] * len(ben) + [1] * len(obf)
        sc = list(ben) + list(obf)
        auc = max(roc_auc_score(y, sc), roc_auc_score(y, [-s for s in sc]))
        
        bf1 = 0; bt = 0
        for pct in range(1, 51):
            t = np.percentile(obf, pct)
            pred = deltas[labels_arr == 'obfuscation'] > t
            tp = np.sum(pred); fn = np.sum(~pred)
            fp = np.sum(deltas[labels_arr == 'benign'] > t)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > bf1: bf1 = f1; bt = t
        
        fpr_a, tpr_a, _ = roc_curve(y, sc)
        j = tpr_a - fpr_a; idx = np.argmax(j)
        
        print(f"\nBENIGN:  μ={np.mean(ben):.6f} σ={np.std(ben):.6f}")
        print(f"OBF:     μ={np.mean(obf):.6f} σ={np.std(obf):.6f}")
        print(f"Sep={sep:.6f} AUC={auc:.3f} F1={bf1:.3f} @ > {bt:.6f}")
        print(f"Optimal: FPR={fpr_a[idx]:.3f} TPR={tpr_a[idx]:.3f}")
        
        results = {
            'model': args.model, 'merge_threshold': args.merge,
            'n_chunks': n_chunks, 'auc': float(auc), 'f1': float(bf1),
            'threshold': float(bt), 'sep': float(sep),
            'benign_mean': float(np.mean(ben)), 'obf_mean': float(np.mean(obf)),
            'fpr': float(fpr_a[idx]), 'tpr': float(tpr_a[idx]),
        }
        with open("data/sentence_chunk_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to data/sentence_chunk_results.json")


if __name__ == "__main__":
    main()
