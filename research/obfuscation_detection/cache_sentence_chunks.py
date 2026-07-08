"""
Cache sentence chunking deltas for all 3 NIM models.
Saves to data/sentence_chunk_cache.json
"""
import numpy as np
import json
import re
import requests
import time
import os

NIM_API_KEY = os.getenv('NIM_API_KEY', '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')
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


def chunk_text(text, merge_threshold=8):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    for s in sentences:
        if not s.strip(): continue
        if len(s) > 50:
            for sub in re.split(r'(?<=[,;])\s+', s):
                if sub.strip(): chunks.append(sub.strip())
        else:
            chunks.append(s.strip())
    if len(chunks) < 2 and len(text.split()) > 10:
        words = text.split()
        mid = len(words) // 2
        chunks = [' '.join(words[:mid]), ' '.join(words[mid:])]
    if not chunks: chunks = [text]
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
    if n1 < 1e-10: return 0.0
    n2 = np.linalg.norm(e2)
    if n2 < 1e-10: return 0.0
    return float(np.arccos(np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)))


def nim_embed(texts, model, batch_size=64):
    all_emb = [None] * len(texts)
    headers = {"Authorization": f"Bearer {NIM_API_KEY}", "Content-Type": "application/json"}
    def clean(t):
        t = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', t)
        t = re.sub(r'[\U0001D400-\U0001D7FF]', '', t)
        t = re.sub(r'(\\[.\-/\\]){3,}', ' ', t)
        t = re.sub(r'\\{2,}', ' ', t)
        t = re.sub(r'\s{2,}', ' ', t)
        # Digit-escape/backslash-digit stress-test text (e.g. "\8\4\ \1\0\4\...")
        # tokenizes at ~1 token/char and isn't touched by the regexes above
        # (those only target letter-backslash patterns), so even 700 chars can
        # exceed the API's 512-token limit. 400 chars stays safely under it
        # even at this worst-case tokenization density.
        return t[:400].strip() or "empty"
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
                if all_emb[idx] is not None: continue
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


def main():
    os.makedirs("data", exist_ok=True)
    
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
    
    print(f"Total: {len(texts)} samples")
    
    # Chunk all texts
    all_chunks = []
    text_ranges = []
    for text in texts:
        chunks = chunk_text(text)
        start = len(all_chunks)
        all_chunks.extend(chunks)
        text_ranges.append((start, len(all_chunks)))
    print(f"Total chunks: {len(all_chunks)}")
    
    models = [
        "nvidia/nv-embedqa-e5-v5",
        "nvidia/llama-nemotron-embed-1b-v2",
        "baai/bge-m3",
    ]
    
    results = {
        "n_obf": labels.count("obfuscation"),
        "n_ben": labels.count("benign"),
        "texts": texts,
        "labels": labels,
        "text_ranges": text_ranges,
        "models": {}
    }
    
    for model in models:
        name = model.split("/")[-1] if "/" in model else model
        print(f"\n{model}...")
        
        t0 = time.time()
        all_emb = nim_embed(all_chunks, model)
        embed_time = time.time() - t0
        
        deltas = []
        for si, ei in text_ranges:
            emb = all_emb[si:ei]
            if len(emb) < 2:
                deltas.append(0.0)
                continue
            angles = [compute_angle(emb[i], emb[i + 1]) for i in range(len(emb) - 1)]
            angles = [a for a in angles if a != 0.0]
            deltas.append(float(np.mean(angles)) if angles else 0.0)
        
        deltas = np.array(deltas)
        labels_arr = np.array(labels)
        ben = deltas[labels_arr == 'benign']
        obf = deltas[labels_arr == 'obfuscation']
        
        print(f"  Embed: {embed_time:.1f}s | Chunks: {len(all_chunks)}")
        print(f"  Benign: μ={np.mean(ben):.4f} σ={np.std(ben):.4f}")
        print(f"  Obf:    μ={np.mean(obf):.4f} σ={np.std(obf):.4f}")
        print(f"  Sep:    {abs(np.mean(ben)-np.mean(obf)):.4f}")
        
        results["models"][model] = {
            "deltas": [float(d) for d in deltas],
            "mean": float(np.mean(deltas)),
            "std": float(np.std(deltas)),
            "ben_mean": float(np.mean(ben)),
            "ben_std": float(np.std(ben)),
            "obf_mean": float(np.mean(obf)),
            "obf_std": float(np.std(obf)),
            "embed_time": embed_time,
            "n_chunks": len(all_chunks),
        }
    
    with open("data/sentence_chunk_cache.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/sentence_chunk_cache.json")


if __name__ == "__main__":
    main()
