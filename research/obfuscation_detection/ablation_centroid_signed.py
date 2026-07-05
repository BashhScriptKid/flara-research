"""
Re-verification of §6.1 (Distribution Distance / centroid cosine distance) and
§6.3 (Signed Angles / Gram-Schmidt) against the current N=1,152 dataset.

Neither original script was preserved (only output artifacts: data/dist_distances.npy,
data/benign_centroid.npy, data/distribution_distance_results.json -- all dated from the
pre-expansion N=1,100 run, with no generator script in the repo). Reconstructed here.
"""
import numpy as np
import json
import re
import time
import os
from sklearn.metrics import roc_auc_score

from cache_sentence_chunks import nim_embed, chunk_text, compute_angle

os.chdir(os.path.dirname(os.path.abspath(__file__)))

obf_trigger = json.load(open("data/obf_trigger.json"))
obf_benign = json.load(open("data/obf_benign.json"))
obf_samples = obf_trigger if isinstance(obf_trigger[0], str) else [s['text'] for s in obf_trigger]
ben_samples = obf_benign if isinstance(obf_benign[0], str) else [s['text'] for s in obf_benign]
all_samples = obf_samples + ben_samples
n_obf, n_ben = len(obf_samples), len(ben_samples)
y_true = np.array([1] * n_obf + [0] * n_ben)
print(f"N={len(all_samples)} (obf={n_obf}, ben={n_ben})")

# ═══════════════════════════════════════════════════════════════════
# §6.1 Distribution Distance: whole-text embedding, cosine distance from
# benign centroid.
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6.1 DISTRIBUTION DISTANCE (whole-text E5 embeddings)")
print("=" * 70)
print("Embedding whole texts (no chunking)...")
t0 = time.time()
whole_emb = nim_embed(all_samples, model="nvidia/nv-embedqa-e5-v5")
whole_emb = np.array(whole_emb)
print(f"Embedded {len(whole_emb)} texts in {time.time()-t0:.1f}s")

ben_emb = whole_emb[n_obf:]
benign_centroid = ben_emb.mean(axis=0)
benign_centroid_norm = benign_centroid / np.linalg.norm(benign_centroid)


def cosine_dist(emb, centroid_norm):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1e-10
    emb_norm = emb / norms
    cos_sim = emb_norm @ centroid_norm
    return 1.0 - cos_sim


dist_all = cosine_dist(whole_emb, benign_centroid_norm)
auc_dist = roc_auc_score(y_true, dist_all)
auc_dist = max(auc_dist, 1 - auc_dist)
print(f"\nCentroid-distance AUC (best direction): {auc_dist:.4f}")
print(f"Obf dist: mean={dist_all[:n_obf].mean():.4f}  Ben dist: mean={dist_all[n_obf:].mean():.4f}")

# Combine with sentence + paragraph delta (matches paper's "combining with delta" test)
cache = json.load(open("data/sentence_chunk_cache.json"))
d_sent = np.array(cache["models"]["nvidia/nv-embedqa-e5-v5"]["deltas"])
para = np.load("data/para_deltas_full.npy")

d_sent_norm = (d_sent - d_sent.min()) / (d_sent.max() - d_sent.min() + 1e-10)
para_norm = (para - para.min()) / (para.max() - para.min() + 1e-10)
dist_norm = (dist_all - dist_all.min()) / (dist_all.max() - dist_all.min() + 1e-10)

combined_sent_para = (d_sent_norm + para_norm) / 2
auc_sent_para_alone = roc_auc_score(y_true, combined_sent_para)
combined_all3 = (d_sent_norm + para_norm + dist_norm) / 3
auc_all3 = roc_auc_score(y_true, combined_all3)
print(f"Sentence+paragraph delta alone AUC: {auc_sent_para_alone:.4f}")
print(f"Sentence+paragraph+centroid-distance AUC: {auc_all3:.4f}")

json.dump({
    "auc_dist_only": float(auc_dist),
    "auc_sent_para_alone": float(auc_sent_para_alone),
    "auc_all_three": float(auc_all3),
    "n_obf": n_obf, "n_ben": n_ben,
}, open("data/distribution_distance_results.json", "w"), indent=2)
print("Saved data/distribution_distance_results.json")

# ═══════════════════════════════════════════════════════════════════
# §6.3 Signed Angles: Gram-Schmidt directional sign vs unsigned, per model.
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6.3 SIGNED ANGLES (Gram-Schmidt) -- E5 and Nemotron")
print("=" * 70)


def compute_signed_angle_deg(e1, e2):
    n1 = np.linalg.norm(e1)
    if n1 < 1e-10:
        return 0.0
    e1n = e1 / n1
    n2 = np.linalg.norm(e2)
    if n2 < 1e-10:
        return 0.0
    orth = e2 - np.dot(e2, e1n) * e1n
    sign = 1 if np.dot(orth, np.ones_like(orth)) >= 0 else -1
    raw = sign * np.arccos(np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1))
    return np.degrees(raw)


all_chunks = []
chunk_ranges = []
for text in all_samples:
    chunks = chunk_text(text)
    start = len(all_chunks)
    all_chunks.extend(chunks)
    chunk_ranges.append((start, len(all_chunks)))
print(f"Total chunks: {len(all_chunks)}")

neg_angle_count_total = 0
for model in ["nvidia/nv-embedqa-e5-v5", "nvidia/llama-nemotron-embed-1b-v2"]:
    print(f"\n--- {model} ---")
    emb = nim_embed(all_chunks, model=model)
    emb = [np.array(e) for e in emb]

    unsigned_deltas = []
    signed_deltas = []
    neg_count = 0
    total_pairs = 0
    for si, ei in chunk_ranges:
        chunk_emb = emb[si:ei]
        if len(chunk_emb) < 2:
            unsigned_deltas.append(0.0)
            signed_deltas.append(0.0)
            continue
        unsigned_angles = [compute_angle(chunk_emb[i], chunk_emb[i + 1]) for i in range(len(chunk_emb) - 1)]
        signed_angles_deg = [compute_signed_angle_deg(chunk_emb[i], chunk_emb[i + 1]) for i in range(len(chunk_emb) - 1)]
        total_pairs += len(signed_angles_deg)
        neg_count += sum(1 for a in signed_angles_deg if a < 0)
        unsigned_angles = [a for a in unsigned_angles if a != 0.0]
        unsigned_deltas.append(float(np.mean(unsigned_angles)) if unsigned_angles else 0.0)
        signed_deltas.append(float(np.mean(signed_angles_deg)))

    neg_angle_count_total += neg_count
    unsigned_deltas = np.array(unsigned_deltas)
    signed_deltas = np.array(signed_deltas)

    auc_unsigned = roc_auc_score(y_true, unsigned_deltas)
    auc_unsigned = max(auc_unsigned, 1 - auc_unsigned)
    # Raw signed score, NOT abs(): the point of this test is whether inconsistent
    # sign assignment hurts separability. Taking abs() would silently undo any such
    # degradation and defeat the test.
    auc_signed_raw = roc_auc_score(y_true, signed_deltas)
    auc_signed_raw = max(auc_signed_raw, 1 - auc_signed_raw)
    print(f"  Negative angles: {neg_count}/{total_pairs} ({100*neg_count/max(total_pairs,1):.2f}%)")
    print(f"  AUC unsigned: {auc_unsigned:.4f}")
    print(f"  AUC signed (raw, not abs): {auc_signed_raw:.4f}")

print("\nDone.")
