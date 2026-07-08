"""
Token-based delta: all 6 NIM models. Run incrementally.
Each model saves progress to avoid losing work on timeout.
"""
import numpy as np
import json
import time
import requests
import os
import re
from collections import defaultdict

NIM_API_KEY = "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"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

MODELS = [
    "nvidia/nv-embedqa-e5-v5",
    "nvidia/llama-nemotron-embed-1b-v2",
    "baai/bge-m3",
    "nvidia/nv-embed-v1",
    "nvidia/nv-embedcode-7b-v1",
    "nvidia/llama-nemotron-embed-vl-1b-v2",
]

OUTPUT = "data/token_based_all_models.json"


def nim_embed(texts, model, batch_size=256):
    all_embeddings = []
    headers = {"Authorization": f"Bearer {NIM_API_KEY}", "Content-Type": "application/json"}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {"model": model, "input": batch, "input_type": "passage"}
        for attempt in range(3):
            try:
                r = requests.post(f"{NIM_BASE_URL}/embeddings", headers=headers, json=payload, timeout=60)
                r.raise_for_status()
                data = r.json()
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                all_embeddings.extend([np.array(item["embedding"], dtype=np.float32) for item in sorted_data])
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  FATAL: {e}", flush=True)
                    return None
                time.sleep(1.5 ** attempt)
    return all_embeddings


def compute_signed_angle(e1, e2):
    n1 = np.linalg.norm(e1)
    if n1 < 1e-10: return 0.0
    e1n = e1 / n1
    n2 = np.linalg.norm(e2)
    if n2 < 1e-10: return 0.0
    orth = e2 - np.dot(e2, e1n) * e1n
    sign = 1 if np.dot(orth, np.ones_like(orth)) >= 0 else -1
    angle = sign * np.arccos(np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1))
    return np.degrees(angle)


def main():
    # Load samples
    obf_trigger = json.load(open("data/obf_trigger.json"))
    obf_benign = json.load(open("data/obf_benign.json"))
    if isinstance(obf_trigger[0], str):
        obf_samples = [{"text": t, "label": "obfuscation"} for t in obf_trigger]
    else:
        obf_samples = obf_trigger
    if isinstance(obf_benign[0], str):
        ben_samples = [{"text": t, "label": "benign"} for t in obf_benign]
    else:
        ben_samples = obf_benign

    combined = {}
    for s in obf_samples + ben_samples:
        combined[s["text"]] = s

    texts = list(combined.keys())
    labels = [combined[t]["label"] for t in texts]

    # Tokenize
    all_tokens = []
    text_token_ranges = []
    for text in texts:
        tokens = re.findall(r"\b\w+\b", text.lower())
        start = len(all_tokens)
        all_tokens.extend(tokens)
        end = len(all_tokens)
        text_token_ranges.append((start, end))

    print(f"Total: {len(texts)} | obf={labels.count('obfuscation')} ben={labels.count('benign')} tokens={len(all_tokens)}", flush=True)

    # Load existing progress
    if os.path.exists(OUTPUT):
        all_results = json.load(open(OUTPUT))
    else:
        all_results = {}

    for model in MODELS:
        if model in all_results:
            print(f"\n{model} already done - skipping", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"Model: {model}", flush=True)

        # Embed
        start_time = time.time()
        all_embeddings = []
        chunk_size = 5000
        failed = False

        for i in range(0, len(all_tokens), chunk_size):
            chunk = all_tokens[i:i + chunk_size]
            print(f"  Tokens {i}-{i+len(chunk)}...", flush=True, end=" ")
            chunk_emb = nim_embed(chunk, model=model, batch_size=256)
            if chunk_emb is None:
                print("FAILED", flush=True)
                failed = True
                break
            all_embeddings.extend(chunk_emb)
            print(f"OK ({len(all_embeddings)})", flush=True)

        if failed or len(all_embeddings) != len(all_tokens):
            print(f"  Skipping model (got {len(all_embeddings)}/{len(all_tokens)})", flush=True)
            continue

        elapsed = time.time() - start_time
        print(f"  Embedding: {elapsed:.0f}s", flush=True)

        # Compute deltas
        deltas = []
        for start, end in text_token_ranges:
            tok_emb = all_embeddings[start:end]
            if len(tok_emb) < 2:
                deltas.append(0.0)
                continue
            angles = []
            for i in range(len(tok_emb) - 1):
                angle = compute_signed_angle(tok_emb[i], tok_emb[i + 1])
                if angle != 0.0:
                    angles.append(angle)
            deltas.append(float(np.mean(angles)) if angles else 0.0)

        # Compute metrics
        deltas_by_label = defaultdict(list)
        for d, l in zip(deltas, labels):
            deltas_by_label[l].append(d)

        ben = np.array(deltas_by_label["benign"])
        obf = np.array(deltas_by_label["obfuscation"])
        sep = abs(np.mean(ben) - np.mean(obf))

        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            y = [0] * len(ben) + [1] * len(obf)
            scores = list(ben) + list(obf)
            auc = max(roc_auc_score(y, scores), roc_auc_score(y, [-s for s in scores]))
        except:
            auc = 0.0

        # Best F1
        best_f1 = 0
        best_t = 0
        for pct in range(50, 100):
            t = np.percentile(ben, pct)
            pred = obf < t
            tp = np.sum(pred)
            fn = np.sum(~pred)
            fp = np.sum(ben < t)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        # Optimal FPR/TPR
        try:
            from sklearn.metrics import roc_curve
            y = [0] * len(ben) + [1] * len(obf)
            scores = [-d for d in list(ben) + list(obf)]
            fpr_arr, tpr_arr, _ = roc_curve(y, scores)
            j = tpr_arr - fpr_arr
            idx = np.argmax(j)
            opt_fpr = float(fpr_arr[idx])
            opt_tpr = float(tpr_arr[idx])
        except:
            opt_fpr = opt_tpr = 0.0

        result = {
            "model": model,
            "time": elapsed,
            "ben_mean": float(np.mean(ben)),
            "ben_std": float(np.std(ben)),
            "obf_mean": float(np.mean(obf)),
            "obf_std": float(np.std(obf)),
            "separation": float(sep),
            "auc": float(auc),
            "best_f1": float(best_f1),
            "best_threshold": float(best_t),
            "optimal_fpr": opt_fpr,
            "optimal_tpr": opt_tpr,
            "ben_zeros": int(np.sum(np.abs(ben) < 0.01)),
            "obf_zeros": int(np.sum(np.abs(obf) < 0.01)),
        }

        all_results[model] = result

        # Save after each model
        with open(OUTPUT, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"  BENIGN:  μ={np.mean(ben):.3f}° σ={np.std(ben):.3f}°", flush=True)
        print(f"  OBF:     μ={np.mean(obf):.3f}° σ={np.std(obf):.3f}°", flush=True)
        print(f"  Sep={sep:.3f}° AUC={auc:.3f} F1={best_f1:.3f} FPR={opt_fpr:.3f} TPR={opt_tpr:.3f}", flush=True)

    # Final table
    print(f"\n\n{'='*100}", flush=True)
    print("TOKEN-BASED MODEL COMPARISON", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"{'Model':<45} {'AUC':>6} {'Sep':>8} {'F1':>6} {'FPR':>6} {'TPR':>6}", flush=True)
    print(f"{'-'*100}", flush=True)
    for m, r in sorted(all_results.items(), key=lambda x: -x[1]["auc"]):
        print(f"{m.split('/')[-1]:<45} {r['auc']:>6.3f} {r['separation']:>7.3f}° {r['best_f1']:>6.3f} {r['optimal_fpr']:>6.3f} {r['optimal_tpr']:>6.3f}", flush=True)


if __name__ == "__main__":
    main()
