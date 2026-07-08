#!/usr/bin/env python3
"""
Chunk Length Distribution Analysis
Finds the "natural trough" for merge threshold in sentence-boundary chunking.
"""

import json
import re
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Chunking logic (exact copy from full_benchmark.py) ──────────────────────
def chunk_input(text):
    text = re.sub(r'\s*\(example\s+\d+\)\.*\s*$', '', text)
    text = re.sub(r'\s*\(example\s+\d+\)\.*\s*', ' ', text).strip()

    if len(text) < 5:
        return [text] if text else []

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

    if len(chunks) < 2 and len(text) > 10:
        words = text.split()
        if len(words) >= 10:
            mid = len(words) // 2
            chunks = [' '.join(words[:mid]), ' '.join(words[mid:])]
        else:
            chunks = [text]

    return chunks if chunks else [text]


# ── Load datasets ────────────────────────────────────────────────────────────
DATA_DIR = "/home/bashh/Documents/! Codes/Flara-workspace/research/obfuscation_detection/data"

with open(f"{DATA_DIR}/obf_trigger.json") as f:
    obf_texts = json.load(f)
with open(f"{DATA_DIR}/obf_benign.json") as f:
    benign_texts = json.load(f)

print(f"Loaded: {len(obf_texts)} obfuscation, {len(benign_texts)} benign samples")


# ── Chunk everything ─────────────────────────────────────────────────────────
def analyze_dataset(texts, label):
    records = []
    all_chunk_lengths = []
    for text in texts:
        chunks = chunk_input(text)
        n_chunks = len(chunks)
        lengths = [len(c.split()) for c in chunks]
        all_chunk_lengths.extend(lengths)
        records.append({
            "n_chunks": n_chunks,
            "lengths": lengths,
            "is_single": n_chunks == 1,
            "text_words": len(text.split()),
        })
    return records, all_chunk_lengths


obf_records, obf_chunk_lens = analyze_dataset(obf_texts, "obf")
benign_records, benign_chunk_lens = analyze_dataset(benign_texts, "benign")
all_records = obf_records + benign_records
all_chunk_lens = obf_chunk_lens + benign_chunk_lens

print(f"\nTotal texts: {len(all_records)}")
print(f"Total chunks: {len(all_chunk_lens)}")


# ── 1. Chunk count distribution ──────────────────────────────────────────────
obf_nc = Counter(r["n_chunks"] for r in obf_records)
benign_nc = Counter(r["n_chunks"] for r in benign_records)

print("\n" + "=" * 60)
print("CHUNK COUNT DISTRIBUTION (per text)")
print("=" * 60)
max_nc = max(max(obf_nc.keys(), default=0), max(benign_nc.keys(), default=0))
print(f"{'Chunks':>8} {'Obf':>8} {'%':>7} {'Benign':>8} {'%':>7}")
for n in range(1, max_nc + 1):
    o = obf_nc.get(n, 0)
    b = benign_nc.get(n, 0)
    print(f"{n:>8} {o:>8} {o/len(obf_records)*100:>6.1f}% {b:>8} {b/len(benign_records)*100:>6.1f}%")


# ── 2. Single-chunk stats ───────────────────────────────────────────────────
obf_single = sum(1 for r in obf_records if r["is_single"])
benign_single = sum(1 for r in benign_records if r["is_single"])
print(f"\nSingle-chunk texts: obf {obf_single}/{len(obf_records)} ({obf_single/len(obf_records)*100:.1f}%), "
      f"benign {benign_single}/{len(benign_records)} ({benign_single/len(benign_records)*100:.1f}%)")


# ── 3. Chunk length distribution (all chunks) ──────────────────────────────
print("\n" + "=" * 60)
print("CHUNK LENGTH DISTRIBUTION (word count, all chunks)")
print("=" * 60)
lens_arr = np.array(all_chunk_lens)
print(f"  Count:   {len(lens_arr)}")
print(f"  Mean:    {lens_arr.mean():.1f}")
print(f"  Median:  {np.median(lens_arr):.1f}")
print(f"  Std:     {lens_arr.std():.1f}")
print(f"  Min:     {lens_arr.min()}")
print(f"  Max:     {lens_arr.max()}")
print(f"  P10:     {np.percentile(lens_arr, 10):.0f}")
print(f"  P25:     {np.percentile(lens_arr, 25):.0f}")
print(f"  P50:     {np.percentile(lens_arr, 50):.0f}")
print(f"  P75:     {np.percentile(lens_arr, 75):.0f}")
print(f"  P90:     {np.percentile(lens_arr, 90):.0f}")

obf_lens_arr = np.array(obf_chunk_lens)
benign_lens_arr = np.array(benign_chunk_lens)
print(f"\n  Obf chunks:   n={len(obf_lens_arr)}, mean={obf_lens_arr.mean():.1f}, median={np.median(obf_lens_arr):.1f}")
print(f"  Benign chunks: n={len(benign_lens_arr)}, mean={benign_lens_arr.mean():.1f}, median={np.median(benign_lens_arr):.1f}")


# ── 4. Histogram: chunk lengths ─────────────────────────────────────────────
# Bin by 1-word increments up to 40, then a catch-all
max_bin = 45
bins = list(range(0, max_bin)) + [max_bin + 1000]
bin_labels = [str(i) for i in range(0, max_bin)] + ["45+"]

obf_hist, _ = np.histogram(obf_lens_arr, bins=bins)
benign_hist, _ = np.histogram(benign_lens_arr, bins=bins)
total_hist = obf_hist + benign_hist

print("\n" + "=" * 60)
print("CHUNK LENGTH HISTOGRAM (per word count)")
print("=" * 60)
print(f"{'Words':>6} {'Obf':>6} {'Benign':>8} {'Total':>7} {'Bar'}")
for i, (o, b, t) in enumerate(zip(obf_hist, benign_hist, total_hist)):
    if t == 0:
        continue
    bar = "█" * min(t // 3, 50)
    label = bin_labels[i]
    print(f"{label:>6} {o:>6} {b:>8} {t:>7}  {bar}")


# ── 5. Find natural trough ──────────────────────────────────────────────────
# Smooth the total histogram to find local minima (troughs)
from scipy.ndimage import uniform_filter1d

smoothed = uniform_filter1d(total_hist.astype(float), size=3)

# Find local minima in the range 3-25 words (practical merge range)
trough_candidates = []
for i in range(2, min(25, len(smoothed) - 1)):
    if smoothed[i] <= smoothed[i - 1] and smoothed[i] <= smoothed[i + 1]:
        trough_candidates.append((i, smoothed[i]))

print("\n" + "=" * 60)
print("LOCAL MINIMA (troughs) in smoothed chunk-length histogram")
print("=" * 60)
if trough_candidates:
    for word_count, val in trough_candidates:
        print(f"  Trough at {word_count} words (smoothed count: {val:.1f})")
    best_trough = min(trough_candidates, key=lambda x: x[1])
    print(f"\n  >>> Deepest trough: {best_trough[0]} words")
else:
    print("  No clear trough found. Looking for flattest region...")
    # Fallback: find the word count with lowest density in 3-20 range
    region = total_hist[3:20]
    min_idx = np.argmin(region) + 3
    print(f"  Flat region around: {min_idx} words")
    best_trough = (min_idx, total_hist[min_idx])


# ── 6. Merge threshold impact analysis ──────────────────────────────────────
print("\n" + "=" * 60)
print("MERGE THRESHOLD IMPACT")
print("=" * 60)
print("If chunks shorter than N words are merged with their neighbor:")
print()

thresholds = [3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
print(f"{'Threshold':>10} {'Texts Affected':>15} {'% Affected':>12} {'Chunks Merged':>15} {'% Merged':>10}")
print("-" * 65)

for thresh in thresholds:
    affected_texts = 0
    total_merged = 0
    total_chunks_before = 0
    total_chunks_after = 0

    for rec in all_records:
        before = rec["n_chunks"]
        total_chunks_before += before
        if before <= 1:
            continue

        # Simulate merge: merge any chunk < thresh words with its neighbor
        merged_chunks = []
        skip = False
        lengths = rec["lengths"]
        for i, length in enumerate(lengths):
            if skip:
                skip = False
                continue
            if length < thresh and i < len(lengths) - 1:
                # Merge with next
                merged_chunks.append(length + lengths[i + 1])
                skip = True
            else:
                merged_chunks.append(length)

        after = len(merged_chunks)
        total_chunks_after += after
        merged_count = before - after
        if merged_count > 0:
            affected_texts += 1
            total_merged += merged_count

    pct_texts = affected_texts / len(all_records) * 100
    pct_chunks = total_merged / total_chunks_before * 100 if total_chunks_before else 0
    print(f"{thresh:>10} {affected_texts:>15} {pct_texts:>11.1f}% {total_merged:>15} {pct_chunks:>9.1f}%")


# ── 7. Detailed merge simulation at recommended threshold ───────────────────
rec_thresh = best_trough[0]
print(f"\n{'=' * 60}")
print(f"DETAILED MERGE AT THRESHOLD = {rec_thresh} words")
print(f"{'=' * 60}")

for label, records in [("Obfuscation", obf_records), ("Benign", benign_records), ("ALL", all_records)]:
    affected = 0
    total_before = 0
    total_after = 0
    for rec in records:
        before = rec["n_chunks"]
        total_before += before
        if before <= 1:
            continue
        merged_chunks = []
        skip = False
        for i, length in enumerate(rec["lengths"]):
            if skip:
                skip = False
                continue
            if length < rec_thresh and i < len(rec["lengths"]) - 1:
                merged_chunks.append(length + rec["lengths"][i + 1])
                skip = True
            else:
                merged_chunks.append(length)
        after = len(merged_chunks)
        total_after += after
        if before != after:
            affected += 1

    print(f"  {label}: {affected}/{len(records)} texts affected "
          f"({affected/len(records)*100:.1f}%), "
          f"chunks: {total_before} -> {total_after} "
          f"({(total_before-total_after)/total_before*100:.1f}% reduction)")


# ── 8. Distribution of chunk lengths BELOW the threshold ────────────────────
print("\n" + "=" * 60)
print(f"CHUNKS SHORTER THAN THRESHOLD ({rec_thresh} words)")
print("=" * 60)
below = [l for l in all_chunk_lens if l < rec_thresh]
above = [l for l in all_chunk_lens if l >= rec_thresh]
print(f"  Below threshold: {len(below)} chunks ({len(below)/len(all_chunk_lens)*100:.1f}%)")
print(f"  At/above:        {len(above)} chunks ({len(above)/len(all_chunk_lens)*100:.1f}%)")
if below:
    below_arr = np.array(below)
    print(f"  Below distribution: mean={below_arr.mean():.1f}, "
          f"median={np.median(below_arr):.0f}, "
          f"min={below_arr.min()}, max={below_arr.max()}")
    bc = Counter(below)
    print(f"  Counts: {dict(sorted(bc.items()))}")


# ── 9. Plots ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Chunk length histogram (all)
ax = axes[0, 0]
x = range(len(obf_hist))
ax.bar(x, benign_hist, alpha=0.7, label="Benign", color="steelblue")
ax.bar(x, obf_hist, alpha=0.7, label="Obfuscation", color="coral", bottom=benign_hist)
ax.set_xlabel("Chunk length (words)")
ax.set_ylabel("Count")
ax.set_title("Chunk Length Distribution (All Chunks)")
ax.legend()
ax.set_xlim(0, 30)
# Mark the trough
if trough_candidates:
    ax.axvline(x=best_trough[0], color='red', linestyle='--', alpha=0.8,
               label=f"Trough @ {best_trough[0]}w")
ax.legend()

# Plot 2: Smoothed histogram with trough highlighted
ax = axes[0, 1]
x_smooth = range(len(smoothed))
ax.plot(x_smooth, smoothed, color="darkblue", linewidth=2)
ax.fill_between(x_smooth, smoothed, alpha=0.3)
for tc in trough_candidates:
    ax.plot(tc[0], tc[1], 'rv', markersize=12)
ax.axvline(x=best_trough[0], color='red', linestyle='--', alpha=0.8)
ax.set_xlabel("Chunk length (words)")
ax.set_ylabel("Smoothed count")
ax.set_title(f"Smoothed Chunk-Length Density\n(Trough = {best_trough[0]} words)")
ax.set_xlim(0, 30)

# Plot 3: Merge threshold impact curve
ax = axes[1, 0]
thresh_range = range(2, 25)
pct_affected = []
for thresh in thresh_range:
    affected = 0
    for rec in all_records:
        if rec["n_chunks"] <= 1:
            continue
        merged_chunks = []
        skip = False
        for i, length in enumerate(rec["lengths"]):
            if skip:
                skip = False
                continue
            if length < thresh and i < len(rec["lengths"]) - 1:
                merged_chunks.append(length + rec["lengths"][i + 1])
                skip = True
            else:
                merged_chunks.append(length)
        if len(merged_chunks) != rec["n_chunks"]:
            affected += 1
    pct_affected.append(affected / len(all_records) * 100)

ax.plot(list(thresh_range), pct_affected, 'o-', color="darkgreen", linewidth=2)
ax.axvline(x=best_trough[0], color='red', linestyle='--', alpha=0.8,
           label=f"Trough @ {best_trough[0]}w")
ax.set_xlabel("Merge threshold (words)")
ax.set_ylabel("% texts affected")
ax.set_title("Impact of Merge Threshold on Texts")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Chunk count distribution
ax = axes[1, 1]
max_nc_plot = min(max(max(obf_nc.keys(), default=1), max(benign_nc.keys(), default=1)), 10)
nc_range = range(1, max_nc_plot + 1)
obf_vals = [obf_nc.get(n, 0) for n in nc_range]
benign_vals = [benign_nc.get(n, 0) for n in nc_range]
width = 0.35
ax.bar([n - width / 2 for n in nc_range], obf_vals, width, label="Obfuscation", color="coral", alpha=0.8)
ax.bar([n + width / 2 for n in nc_range], benign_vals, width, label="Benign", color="steelblue", alpha=0.8)
ax.set_xlabel("Number of chunks")
ax.set_ylabel("Number of texts")
ax.set_title("Chunks-per-Text Distribution")
ax.set_xticks(list(nc_range))
ax.legend()

plt.tight_layout()
out_path = "/home/bashh/Documents/! Codes/Flara-workspace/research/obfuscation_detection/graphs/chunk_length_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {out_path}")


# ── 10. Recommendation ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)
print(f"""
Natural trough in chunk-length distribution: {best_trough[0]} words

This is the boundary between "short fragments" (likely split mid-sentence
at commas/semicolons) and "normal chunks" (full sentences or well-formed
clauses). Chunks below this length are artifacts of aggressive splitting
and should be merged with their neighbor.

Recommended merge threshold: {best_trough[0]} words
  - Merge any chunk shorter than {best_trough[0]} words with its adjacent chunk
  - This eliminates artificial short fragments while preserving natural sentence boundaries
  - Expected impact: see table above for exact percentages
""")
