"""
Fetch per-chunk consecutive-delta sequences (not just the mean) for
nv-embedqa-e5-v5 only, to test alternative aggregations (max, variance,
position-specific) against the existing mean-delta baseline.
Saves to data/per_chunk_deltas_e5.json, obf-first-then-benign sample order
(matching sentence_chunk_cache.json's convention).
"""
import numpy as np
import json
import re
import requests
import time
import os

from cache_sentence_chunks import NIM_API_KEY
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nv-embedqa-e5-v5"


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
        return t[:400].strip() or "empty"
    for i in range(0, len(texts), batch_size):
        batch = [clean(t) for t in texts[i:i + batch_size]]
        try:
            r = requests.post(f"{NIM_BASE_URL}/embeddings", headers=headers,
                            json={"model": model, "input": batch, "input_type": "passage"}, timeout=60)
            r.raise_for_status()
            for j, item in enumerate(sorted(r.json()['data'], key=lambda x: x['index'])):
                all_emb[i + j] = np.array(item['embedding'], dtype=np.float32)
        except Exception as e:
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
        print(f"  embedded {min(i+batch_size,len(texts))}/{len(texts)}")
    return all_emb


def main():
    os.makedirs("data", exist_ok=True)
    obf_trigger = json.load(open("data/obf_trigger.json"))
    obf_benign = json.load(open("data/obf_benign.json"))
    obf_samples = obf_trigger if isinstance(obf_trigger[0], str) else [s['text'] for s in obf_trigger]
    ben_samples = obf_benign if isinstance(obf_benign[0], str) else [s['text'] for s in obf_benign]

    texts = obf_samples + ben_samples
    labels = ["obfuscation"] * len(obf_samples) + ["benign"] * len(ben_samples)
    print(f"Total samples: {len(texts)}")

    all_chunks = []
    text_ranges = []
    for text in texts:
        chunks = chunk_text(text)
        start = len(all_chunks)
        all_chunks.extend(chunks)
        text_ranges.append((start, len(all_chunks)))
    print(f"Total chunks: {len(all_chunks)}")

    t0 = time.time()
    all_emb = nim_embed(all_chunks, MODEL)
    embed_time = time.time() - t0
    print(f"Embed time: {embed_time:.1f}s")

    per_sample_deltas = []
    for si, ei in text_ranges:
        emb = all_emb[si:ei]
        if len(emb) < 2:
            per_sample_deltas.append([])
            continue
        angles = [compute_angle(emb[i], emb[i + 1]) for i in range(len(emb) - 1)]
        per_sample_deltas.append(angles)

    out = {
        "model": MODEL,
        "labels": labels,
        "n_obf": len(obf_samples),
        "n_ben": len(ben_samples),
        "per_sample_deltas": per_sample_deltas,
        "embed_time": embed_time,
        "n_chunks": len(all_chunks),
    }
    with open("data/per_chunk_deltas_e5.json", "w") as f:
        json.dump(out, f)
    print("Saved data/per_chunk_deltas_e5.json")


if __name__ == "__main__":
    main()
