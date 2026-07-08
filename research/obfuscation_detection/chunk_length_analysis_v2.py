#!/usr/bin/env python3
"""
Chunk Length Distribution Analysis v2
Finds the practical merge threshold by analyzing the density cliff.
"""

import json
import re
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
def analyze_dataset(texts):
    records, all_lengths = [], []
    for text in texts:
        chunks = chunk_input(text)
        lengths = [len(c.split()) for c in chunks]
        all_lengths.extend(lengths)
        records.append({"n_chunks": len(chunks), "lengths": lengths, "is_single": len(chunks) == 1})
    return records, all_lengths

obf_records, obf_lens = analyze_dataset(obf_texts)
benign_records, benign_lens = analyze_dataset(benign_texts)
all_records = obf_records + benign_records
all_lens = obf_lens + benign_lens


# ── Raw histogram (no smoothing bias) ───────────────────────────────────────
max_bin = 25
bins = list(range(1, max_bin + 2))
bin_labels = list(range(1, max_bin + 1))

obf_hist, _ = np.histogram(obf_lens, bins=bins)
benign_hist, _ = np.histogram(benign_lens, bins=bins)
total_hist = obf_hist + benign_hist

print("\n" + "=" * 60)
print("CHUNK LENGTH HISTOGRAM (raw counts)")
print("=" * 60)
print(f"{'Words':>6} {'Obf':>6} {'Benign':>8} {'Total':>7}  {'Ratio O/B':>9}  Bar")
for i, (o, b, t) in enumerate(zip(obf_hist, benign_hist, total_hist)):
    ratio = f"{o/max(b,1):.1f}x" if b > 0 else "N/A"
    bar = "█" * min(t // 5, 40)
    print(f"{bin_labels[i]:>6} {o:>6} {b:>8} {t:>7}  {ratio:>9}  {bar}")


# ── Key insight: ratio of obf/benign per bin ────────────────────────────────
print("\n" + "=" * 60)
print("OBF/BENIGN RATIO PER CHUNK LENGTH")
print("(High ratio = obfuscation-specific chunk size)")
print("=" * 60)
for i in range(len(bin_labels)):
    b = max(benign_hist[i], 1)
    ratio = obf_hist[i] / b
    marker = " <-- dominant in obf" if ratio > 3 and obf_hist[i] > 20 else ""
    print(f"  {bin_labels[i]:>2} words: {ratio:.2f}x{marker}")


# ── Find the density cliff ──────────────────────────────────────────────────
# The "cliff" is where the count drops by >50% from the previous bin
print("\n" + "=" * 60)
print("DENSITY CLIFF DETECTION")
print("=" * 60)
for i in range(1, len(total_hist)):
    if total_hist[i - 1] > 0:
        drop = 1 - total_hist[i] / total_hist[i - 1]
        if drop > 0.3:
            print(f"  {bin_labels[i-1]}w -> {bin_labels[i]}w: "
                  f"{total_hist[i-1]} -> {total_hist[i]} ({drop*100:.0f}% drop)")
    # Also check ratio changes
    if i >= 2:
        ratio_prev = obf_hist[i-1] / max(benign_hist[i-1], 1)
        ratio_curr = obf_hist[i] / max(benign_hist[i], 1)
        if ratio_prev > 2 and ratio_curr < 1 and total_hist[i-1] > 50:
            print(f"  ** Ratio flip at {bin_labels[i]}w: obf/benign {ratio_prev:.1f}x -> {ratio_curr:.1f}x")


# ── Detailed merge simulation ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("MERGE THRESHOLD IMPACT (granular)")
print("=" * 70)
print(f"{'Thresh':>7} {'Affected':>9} {'%Texts':>7} {'Merged':>8} {'%Chunks':>8} {'Δ%Texts':>8}")
print("-" * 70)

prev_pct_texts = 0
for thresh in range(2, 16):
    affected = 0
    total_merged = 0
    total_before = 0
    for rec in all_records:
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
            if length < thresh and i < len(rec["lengths"]) - 1:
                merged_chunks.append(length + rec["lengths"][i + 1])
                skip = True
            else:
                merged_chunks.append(length)
        after = len(merged_chunks)
        if after != before:
            affected += 1
            total_merged += before - after

    pct_texts = affected / len(all_records) * 100
    pct_chunks = total_merged / total_before * 100
    delta = pct_texts - prev_pct_texts
    marker = " <-- CLIFF" if delta > 5 else ""
    print(f"{thresh:>7} {affected:>9} {pct_texts:>6.1f}% {total_merged:>8} {pct_chunks:>7.1f}% {delta:>+7.1f}%{marker}")
    prev_pct_texts = pct_texts


# ── Per-dataset merge at candidate thresholds ────────────────────────────────
print("\n" + "=" * 70)
print("PER-DATASET MERGE IMPACT AT KEY THRESHOLDS")
print("=" * 70)
for thresh in [5, 7, 8, 10]:
    print(f"\n  --- Threshold = {thresh} words ---")
    for label, records in [("Obf", obf_records), ("Benign", benign_records), ("ALL", all_records)]:
        affected = 0
        tb, ta = 0, 0
        for rec in records:
            tb += rec["n_chunks"]
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
            ta += len(merged_chunks)
            if len(merged_chunks) != rec["n_chunks"]:
                affected += 1
        reduction = (tb - ta) / tb * 100 if tb else 0
        print(f"    {label:>7}: {affected:>4}/{len(records)} texts ({affected/len(records)*100:>5.1f}%), "
              f"chunks {tb}->{ta} (-{reduction:.1f}%)")


# ── Distribution of chunk sizes at the cliff ─────────────────────────────────
print("\n" + "=" * 60)
print("WHAT HAPPENS TO 7-WORD CHUNKS?")
print("=" * 60)
seven_word_chunks = []
for rec in all_records:
    for i, length in enumerate(rec["lengths"]):
        if length == 7:
            # Get context: what's the neighbor?
            prev_len = rec["lengths"][i-1] if i > 0 else None
            next_len = rec["lengths"][i+1] if i < len(rec["lengths"]) - 1 else None
            seven_word_chunks.append({
                "prev": prev_len, "next": next_len,
                "total_chunks": rec["n_chunks"],
                "is_obf": rec in obf_records
            })

print(f"  Total 7-word chunks: {len(seven_word_chunks)}")
print(f"  Of which in obf texts: {sum(1 for c in seven_word_chunks if c['is_obf'])}")
print(f"  Of which in benign texts: {sum(1 for c in seven_word_chunks if not c['is_obf'])}")

# What are they usually next to?
next_counts = Counter(c["next"] for c in seven_word_chunks if c["next"] is not None)
prev_counts = Counter(c["prev"] for c in seven_word_chunks if c["prev"] is not None)
print(f"\n  What follows a 7-word chunk: {dict(sorted(next_counts.items()))}")
print(f"  What precedes a 7-word chunk: {dict(sorted(prev_counts.items()))}")

# If merged with next, what's the resulting size?
resulting_sizes = []
for c in seven_word_chunks:
    if c["next"] is not None:
        resulting_sizes.append(7 + c["next"])
if resulting_sizes:
    rs = np.array(resulting_sizes)
    print(f"\n  If merged with NEXT chunk: sizes {rs.min()}-{rs.max()}, mean={rs.mean():.1f}, median={np.median(rs):.0f}")
    rc = Counter(resulting_sizes)
    print(f"  Resulting sizes: {dict(sorted(rc.items()))}")


# ── Plots ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Histogram with cliff annotation
ax = axes[0, 0]
x = np.arange(len(obf_hist))
ax.bar(x - 0.15, benign_hist, 0.3, alpha=0.8, label="Benign", color="steelblue")
ax.bar(x + 0.15, obf_hist, 0.3, alpha=0.8, label="Obfuscation", color="coral")
# Mark cliff
cliff_idx = 6  # index for 7 words
ax.annotate('DENSITY CLIFF\n7w→8w: 907→116\n(-87%)',
            xy=(cliff_idx, total_hist[cliff_idx]),
            xytext=(cliff_idx + 3, total_hist[cliff_idx] * 0.85),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='red'))
ax.axvline(x=cliff_idx + 0.5, color='red', linestyle='--', alpha=0.5)
ax.axvline(x=cliff_idx + 1.5, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel("Chunk length (words)")
ax.set_ylabel("Count")
ax.set_title("Chunk Length Distribution\n(Cliff at 7→8 words)")
ax.set_xticks(x)
ax.set_xticklabels(bin_labels)
ax.set_xlim(-0.5, 16.5)
ax.legend()

# Plot 2: Impact curve with step detection
ax = axes[0, 1]
thresh_range = list(range(2, 16))
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

ax.plot(thresh_range, pct_affected, 'o-', color="darkgreen", linewidth=2.5, markersize=8)
# Annotate the step
for i, (t, p) in enumerate(zip(thresh_range, pct_affected)):
    if i > 0:
        delta = p - pct_affected[i-1]
        if delta > 3:
            ax.annotate(f'{delta:.1f}% jump',
                       xy=(t, p), xytext=(t - 1.5, p + 2),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, color='red', fontweight='bold')
ax.set_xlabel("Merge threshold (words)")
ax.set_ylabel("% texts affected")
ax.set_title("Merge Threshold Impact\n(Step = natural boundary)")
ax.grid(True, alpha=0.3)
ax.set_xticks(thresh_range)

# Plot 3: Obf vs Benign chunk length side-by-side
ax = axes[1, 0]
obf_c = Counter(obf_lens)
benign_c = Counter(benign_lens)
all_vals = sorted(set(list(obf_c.keys()) + list(benign_c.keys())))
all_vals = [v for v in all_vals if v <= 20]
x = np.arange(len(all_vals))
obf_vals = [obf_c.get(v, 0) for v in all_vals]
benign_vals = [benign_c.get(v, 0) for v in all_vals]
ax.bar(x - 0.2, obf_vals, 0.4, alpha=0.8, color="coral", label="Obfuscation")
ax.bar(x + 0.2, benign_vals, 0.4, alpha=0.8, color="steelblue", label="Benign")
ax.set_xlabel("Chunk length (words)")
ax.set_ylabel("Count")
ax.set_title("Obfuscation vs Benign Chunk Sizes\n(7w is obfuscation-dominated)")
ax.set_xticks(x)
ax.set_xticklabels(all_vals)
ax.legend()

# Plot 4: Percentile chart
ax = axes[1, 1]
lens_arr = np.array(all_lens)
percentiles = [10, 25, 50, 75, 90, 95, 99]
pvals = [np.percentile(lens_arr, p) for p in percentiles]
ax.barh(range(len(percentiles)), pvals, color="teal", alpha=0.7)
ax.set_yticks(range(len(percentiles)))
ax.set_yticklabels([f"P{p}" for p in percentiles])
ax.set_xlabel("Chunk length (words)")
ax.set_title("Chunk Length Percentiles")
for i, v in enumerate(pvals):
    ax.text(v + 0.2, i, f"{v:.0f}", va='center', fontweight='bold')

plt.tight_layout()
out_path = "/home/bashh/Documents/! Codes/Flara-workspace/research/obfuscation_detection/graphs/chunk_length_analysis_v2.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {out_path}")


# ── FINAL RECOMMENDATION ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL RECOMMENDATION")
print("=" * 70)
print("""
KEY FINDING: The chunk-length distribution has a sharp cliff at 7→8 words.
  - 907 chunks (40.8%) are exactly 7 words (median)
  - Only 116 chunks (5.2%) are 8 words
  - This cliff is the boundary between "comma/semicolon splits" and "real sentences"

The merge threshold should be set AT the cliff boundary:

  RECOMMENDED THRESHOLD: 8 words
  ─────────────────────────────────────────────
  Merge any chunk shorter than 8 words with its neighbor.

  WHY 8:
  - Catches all the 7-word comma-split fragments (the biggest cluster)
  - Minimal impact on legitimate 8+ word sentence chunks
  - The impact curve shows a sharp step at 8 (9.5% → 24.4%)
  - Below 8: only 9.5% of texts affected (too conservative)
  - At 8: 24.4% of texts affected (catches the comma-split artifacts)
  - Above 8: diminishing returns (26.6% at 10, barely more)

  ALTERNATIVE: 7 words (conservative)
  - Only merges chunks < 7 words (the 1-6 word fragments)
  - Affects only 9.5% of texts
  - Leaves the 7-word cluster intact

  ALTERNATIVE: 10 words (aggressive)
  - Also merges 8-9 word chunks
  - Affects 26.6% of texts
  - Slightly more aggressive, but still reasonable
""")
