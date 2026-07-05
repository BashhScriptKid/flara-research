"""
Generate graphs for sentence chunking delta angle results.
"""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, roc_auc_score
from collections import Counter
import os
import re

def load_data():
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
    return obf_samples, ben_samples


def chunk_text(text, merge_threshold=8):
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


def generate_graphs():
    os.makedirs("graphs", exist_ok=True)
    obf_samples, ben_samples = load_data()
    
    # Compute chunk counts and lengths for all samples
    ben_chunks = [chunk_text(t) for t in ben_samples]
    obf_chunks = [chunk_text(t) for t in obf_samples]
    
    ben_lens = [len(c.split()) for chunks in ben_chunks for c in chunks]
    obf_lens = [len(c.split()) for chunks in obf_chunks for c in chunks]
    
    ben_nc = [len(chunks) for chunks in ben_chunks]
    obf_nc = [len(chunks) for chunks in obf_chunks]
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Sentence Chunking Analysis", fontsize=14, fontweight='bold')
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Chunk length distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(ben_lens, bins=30, alpha=0.7, color='steelblue', label=f'Benign (μ={np.mean(ben_lens):.1f})', density=True)
    ax1.hist(obf_lens, bins=30, alpha=0.7, color='crimson', label=f'Obf (μ={np.mean(obf_lens):.1f})', density=True)
    ax1.axvline(x=8, color='black', linestyle='--', linewidth=1, label='Merge threshold')
    ax1.set_xlabel("Chunk length (words)")
    ax1.set_ylabel("Density")
    ax1.set_title("Chunk Length Distribution")
    ax1.legend(fontsize=7)
    ax1.set_xlim(0, 50)
    
    # 2. Number of chunks per input
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(ben_nc, bins=20, alpha=0.7, color='steelblue', label=f'Benign (μ={np.mean(ben_nc):.1f})', density=True)
    ax2.hist(obf_nc, bins=20, alpha=0.7, color='crimson', label=f'Obf (μ={np.mean(obf_nc):.1f})', density=True)
    ax2.set_xlabel("Number of chunks")
    ax2.set_ylabel("Density")
    ax2.set_title("Chunks per Input")
    ax2.legend(fontsize=8)
    
    # 3. Single-chunk percentage
    ax3 = fig.add_subplot(gs[0, 2])
    ben_single = sum(1 for n in ben_nc if n <= 1) / len(ben_nc) * 100
    obf_single = sum(1 for n in obf_nc if n <= 1) / len(obf_nc) * 100
    bars = ax3.bar(['Benign', 'Obfuscation'], [ben_single, obf_single], 
                   color=['steelblue', 'crimson'], alpha=0.8)
    ax3.set_ylabel("Percentage with ≤1 chunk")
    ax3.set_title("Single-Chunk Artifact")
    for bar, val in zip(bars, [ben_single, obf_single]):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax3.set_ylim(0, 100)
    
    # 4. Chunk length distribution (cumulative)
    ax4 = fig.add_subplot(gs[1, 0])
    ben_sorted = np.sort(ben_lens)
    obf_sorted = np.sort(obf_lens)
    ax4.plot(np.linspace(0, 100, len(ben_sorted)), ben_sorted, color='steelblue', label='Benign', linewidth=2)
    ax4.plot(np.linspace(0, 100, len(obf_sorted)), obf_sorted, color='crimson', label='Obf', linewidth=2)
    ax4.axhline(y=8, color='black', linestyle='--', linewidth=1, label='Merge threshold')
    ax4.set_xlabel("Percentile")
    ax4.set_ylabel("Chunk length (words)")
    ax4.set_title("Cumulative Chunk Lengths")
    ax4.legend(fontsize=8)
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 50)
    
    # 5. Word count distribution per input
    ax5 = fig.add_subplot(gs[1, 1])
    ben_words = [len(t.split()) for t in ben_samples]
    obf_words = [len(t.split()) for t in obf_samples]
    ax5.hist(ben_words, bins=20, alpha=0.7, color='steelblue', label=f'Benign (μ={np.mean(ben_words):.0f})', density=True)
    ax5.hist(obf_words, bins=20, alpha=0.7, color='crimson', label=f'Obf (μ={np.mean(obf_words):.0f})', density=True)
    ax5.set_xlabel("Words per input")
    ax5.set_ylabel("Density")
    ax5.set_title("Input Word Count")
    ax5.legend(fontsize=8)
    
    # 6. Merge threshold sensitivity
    ax6 = fig.add_subplot(gs[1, 2])
    thresholds = [4, 6, 8, 10, 12]
    single_pcts = []
    avg_chunks = []
    for t in thresholds:
        nc = [len(chunk_text(s, merge_threshold=t)) for s in ben_samples[:100]]
        single_pcts.append(sum(1 for n in nc if n <= 1) / len(nc) * 100)
        avg_chunks.append(np.mean(nc))
    
    ax6.plot(thresholds, single_pcts, 'o-', color='steelblue', linewidth=2, label='Single-chunk %')
    ax6.axvline(x=8, color='gray', linestyle='--', linewidth=1, label='Selected (8)')
    ax6.set_xlabel("Merge threshold (words)")
    ax6.set_ylabel("% inputs with ≤1 chunk")
    ax6.set_title("Merge Threshold Sensitivity")
    ax6.legend(fontsize=8)
    ax6.set_xticks(thresholds)
    
    plt.savefig("graphs/sentence_chunking_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved graphs/sentence_chunking_analysis.png")
    
    # Summary
    print(f"\nSummary:")
    print(f"  Benign: {len(ben_samples)} samples, μ={np.mean(ben_lens):.1f} words/chunk, μ={np.mean(ben_nc):.1f} chunks/input, {ben_single:.1f}% single-chunk")
    print(f"  Obf:    {len(obf_samples)} samples, μ={np.mean(obf_lens):.1f} words/chunk, μ={np.mean(obf_nc):.1f} chunks/input, {obf_single:.1f}% single-chunk")


if __name__ == "__main__":
    generate_graphs()
