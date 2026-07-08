"""
Full graph suite for Delta Angle paper — sentence chunking method.
Generates publication-quality figures in graphs/sentence_chunking/
"""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, roc_auc_score
import os
import re

OUT = "graphs/sentence_chunking"
os.makedirs(OUT, exist_ok=True)

# ── Load cached data ───────────────────────────────────────────────
with open("data/sentence_chunk_cache.json") as f:
    cache = json.load(f)

texts = cache["texts"]
labels = np.array(cache["labels"])
ben_mask = labels == "benign"
obf_mask = labels == "obfuscation"

# ── Helper ──────────────────────────────────────────────────────────
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


# ── Compute chunk-level stats ───────────────────────────────────────
ben_texts = [t for t, l in zip(texts, labels) if l == "benign"]
obf_texts = [t for t, l in zip(texts, labels) if l == "obfuscation"]
ben_chunks_list = [chunk_text(t) for t in ben_texts]
obf_chunks_list = [chunk_text(t) for t in obf_texts]

ben_lens = [len(c.split()) for chunks in ben_chunks_list for c in chunks]
obf_lens = [len(c.split()) for chunks in obf_chunks_list for c in chunks]
ben_nc = [len(chunks) for chunks in ben_chunks_list]
obf_nc = [len(chunks) for chunks in obf_chunks_list]
ben_words = [len(t.split()) for t in ben_texts]
obf_words = [len(t.split()) for t in obf_texts]

# ── E5 deltas (degrees) ───────────────────────────────────────────
e5 = cache["models"]["nvidia/nv-embedqa-e5-v5"]
d_e5 = np.array(e5["deltas"])
d_e5_ben = d_e5[ben_mask]
d_e5_obf = d_e5[obf_mask]

# ── Figure style ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})
C_BEN = "#4C72B0"
C_OBF = "#C44E52"


# ═══════════════════════════════════════════════════════════════════
# FIG 1 — Delta distributions (E5) — 2 panels
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.hist(d_e5_ben, bins=40, alpha=0.75, color=C_BEN, label=f"Benign (μ={np.mean(d_e5_ben):.4f})", density=True, edgecolor="white", linewidth=0.5)
ax.hist(d_e5_obf, bins=40, alpha=0.75, color=C_OBF, label=f"Obfuscation (μ={np.mean(d_e5_obf):.4f})", density=True, edgecolor="white", linewidth=0.5)
ax.axvline(0, color="gray", ls="--", lw=0.8, label="Zero (single-chunk)")
ax.set_xlabel("Average Delta Angle (radians)")
ax.set_ylabel("Density")
ax.set_title("(a) Delta Distribution")
ax.legend(framealpha=0.9)

ax = axes[1]
parts = ax.violinplot([d_e5_ben, d_e5_obf], positions=[0, 1], showmeans=True, showmedians=True)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor([C_BEN, C_OBF][i])
    pc.set_alpha(0.6)
parts["cmeans"].set_color("black")
parts["cmedians"].set_color("gray")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Benign", "Obfuscation"])
ax.set_ylabel("Average Delta Angle (radians)")
ax.set_title("(b) Violin Plot")
ax.set_yscale("symlog", linthresh=0.01)

fig.suptitle("Figure 1 — Delta Angle Distributions (E5, N=1100)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_distributions.png", bbox_inches="tight")
plt.close()
print("✓ fig1_distributions.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 2 — ROC curves (all 3 models)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))

for model_key, color, ls in [
    ("nvidia/nv-embedqa-e5-v5", C_BEN, "-"),
    ("nvidia/llama-nemotron-embed-1b-v2", "#55A868", "--"),
    ("baai/bge-m3", "#DD8452", ":"),
]:
    m = cache["models"][model_key]
    d = np.array(m["deltas"])
    name = model_key.split("/")[-1]

    # figure out sign: if obf_mean > ben_mean, high = obf; else flip
    sign = 1 if m["obf_mean"] > m["ben_mean"] else -1
    sc = sign * d
    y = (labels == "obfuscation").astype(int)
    fpr, tpr, _ = roc_curve(y, sc)
    auc = roc_auc_score(y, sc)
    ax.plot(fpr, tpr, color=color, ls=ls, lw=2, label=f"{name} (AUC={auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Figure 2 — ROC Curves (Sentence Chunking)")
ax.legend(loc="lower right", framealpha=0.9)
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_roc.png", bbox_inches="tight")
plt.close()
print("✓ fig2_roc.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 3 — FPR / TPR at various thresholds (E5)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))

y_true = (labels == "obfuscation").astype(int)
sign = 1 if e5["obf_mean"] > e5["ben_mean"] else -1
sc = sign * d_e5
fpr_arr, tpr_arr, thresholds = roc_curve(y_true, sc)

fpr_a, tpr_a, thresh_arr = roc_curve(y_true, sc)
ax.plot(thresh_arr, tpr_a, color=C_OBF, lw=2, label="TPR (Recall)")
ax.plot(thresh_arr, fpr_a, color=C_BEN, lw=2, label="FPR")
ax.plot(thresh_arr, tpr_a - fpr_a, color="gray", lw=1.5, ls="--", label="TPR − FPR")

# mark optimal
j = tpr_a - fpr_a
idx = np.argmax(j)
ax.axvline(thresh_arr[idx], color="black", ls=":", lw=1, label=f"Optimal (t={thresh_arr[idx]:.3f})")

ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Rate")
ax.set_title("Figure 3 — Threshold Analysis (E5)")
ax.legend(framealpha=0.9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_threshold.png", bbox_inches="tight")
plt.close()
print("✓ fig3_threshold.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 4 — Chunk length distribution + chunks-per-input
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.hist(ben_lens, bins=35, alpha=0.7, color=C_BEN, label=f"Benign (μ={np.mean(ben_lens):.1f})", density=True, edgecolor="white", linewidth=0.5)
ax.hist(obf_lens, bins=35, alpha=0.7, color=C_OBF, label=f"Obf (μ={np.mean(obf_lens):.1f})", density=True, edgecolor="white", linewidth=0.5)
ax.axvline(8, color="black", ls="--", lw=1, label="Merge threshold (8)")
ax.set_xlabel("Chunk Length (words)")
ax.set_ylabel("Density")
ax.set_title("(a) Chunk Length Distribution")
ax.legend(framealpha=0.9)
ax.set_xlim(0, 55)

ax = axes[1]
ax.hist(ben_nc, bins=range(0, max(max(ben_nc), max(obf_nc)) + 2), alpha=0.7, color=C_BEN, label=f"Benign (μ={np.mean(ben_nc):.1f})", density=True, edgecolor="white", linewidth=0.5)
ax.hist(obf_nc, bins=range(0, max(max(ben_nc), max(obf_nc)) + 2), alpha=0.7, color=C_OBF, label=f"Obf (μ={np.mean(obf_nc):.1f})", density=True, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Number of Chunks per Input")
ax.set_ylabel("Density")
ax.set_title("(b) Chunks per Input")
ax.legend(framealpha=0.9)

fig.suptitle("Figure 4 — Chunking Statistics", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_chunk_stats.png", bbox_inches="tight")
plt.close()
print("✓ fig4_chunk_stats.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 5 — Model comparison bars
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

model_names = []
aucs = []
f1s = []
fprs = []
for model_key in ["nvidia/nv-embedqa-e5-v5", "nvidia/llama-nemotron-embed-1b-v2", "baai/bge-m3"]:
    m = cache["models"][model_key]
    name = model_key.split("/")[-1]
    if name == "nv-embedqa-e5-v5": name = "E5"
    elif name == "llama-nemotron-embed-1b-v2": name = "Nemotron"
    elif name == "bge-m3": name = "BGE-M3"
    model_names.append(name)

    d = np.array(m["deltas"])
    sign = 1 if m["obf_mean"] > m["ben_mean"] else -1
    sc = sign * d
    y = (labels == "obfuscation").astype(int)
    auc = roc_auc_score(y, sc)
    aucs.append(auc)

    # F1
    bf1 = 0; bt = 0
    obf_d = sign * d[obf_mask]
    ben_d = sign * d[ben_mask]
    for pct in range(1, 51):
        t = np.percentile(obf_d, pct)
        pred = obf_d > t
        tp = np.sum(pred); fn = np.sum(~pred)
        fp = np.sum(ben_d > t)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > bf1: bf1 = f1; bt = t
    f1s.append(bf1)

    fpr_a, tpr_a, _ = roc_curve(y, sc)
    j = tpr_a - fpr_a; idx = np.argmax(j)
    fprs.append(fpr_a[idx])

colors = [C_BEN, "#55A868", "#DD8452"]

axes[0].bar(model_names, aucs, color=colors, edgecolor="white", linewidth=1.5)
axes[0].set_ylabel("AUC-ROC")
axes[0].set_title("(a) AUC-ROC")
axes[0].set_ylim(0.5, 1.0)
for i, v in enumerate(aucs):
    axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold", fontsize=10)

axes[1].bar(model_names, f1s, color=colors, edgecolor="white", linewidth=1.5)
axes[1].set_ylabel("Best F1")
axes[1].set_title("(b) F1 Score")
axes[1].set_ylim(0, 1.0)
for i, v in enumerate(f1s):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold", fontsize=10)

axes[2].bar(model_names, fprs, color=colors, edgecolor="white", linewidth=1.5)
axes[2].set_ylabel("Optimal FPR")
axes[2].set_title("(c) False Positive Rate")
axes[2].set_ylim(0, 0.05)
for i, v in enumerate(fprs):
    axes[2].text(i, v + 0.0005, f"{v:.3f}", ha="center", fontweight="bold", fontsize=10)

fig.suptitle("Figure 5 — Model Comparison (Sentence Chunking)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_model_comparison.png", bbox_inches="tight")
plt.close()
print("✓ fig5_model_comparison.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 6 — Single-chunk bar chart
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 4.5))

ben_single = sum(1 for n in ben_nc if n <= 1) / len(ben_nc) * 100
obf_single = sum(1 for n in obf_nc if n <= 1) / len(obf_nc) * 100

bars = ax.bar(["Benign", "Obfuscation"], [ben_single, obf_single],
              color=[C_BEN, C_OBF], edgecolor="white", linewidth=1.5)
ax.set_ylabel("% Inputs with ≤ 1 Chunk")
ax.set_title("Figure 6 — Single-Chunk Rate")
for bar, val in zip(bars, [ben_single, obf_single]):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)
ax.set_ylim(0, 110)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig6_single_chunk.png", bbox_inches="tight")
plt.close()
print("✓ fig6_single_chunk.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 7 — Risk score scatter (delta vs entropy proxy)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))

def char_entropy(text):
    from collections import Counter
    freq = Counter(text)
    n = max(len(text), 1)
    return -sum((cnt / n) * np.log2(cnt / n) for cnt in freq.values() if cnt > 0)

ben_entropy = [char_entropy(t) for t in ben_texts]
obf_entropy = [char_entropy(t) for t in obf_texts]

ax.scatter(d_e5_ben, ben_entropy, alpha=0.4, s=20, c=C_BEN, label="Benign", edgecolors="none")
ax.scatter(d_e5_obf, obf_entropy, alpha=0.4, s=20, c=C_OBF, label="Obfuscation", edgecolors="none")
ax.set_xlabel("Average Delta Angle (radians)")
ax.set_ylabel("Character Entropy (bits)")
ax.set_title("Figure 7 — Delta vs Entropy")
ax.legend(framealpha=0.9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig7_scatter.png", bbox_inches="tight")
plt.close()
print("✓ fig7_scatter.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 8 — Correlation matrix
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 5))

deltas_all = d_e5
nc_all = np.array(ben_nc + obf_nc)
word_all = np.array(ben_words + obf_words)
avg_chunk_len = np.array([np.mean([len(c.split()) for c in chunks]) if chunks else 0 for chunks in ben_chunks_list + obf_chunks_list])
entropy_all = np.array(ben_entropy + obf_entropy)

matrix = np.corrcoef([deltas_all, nc_all, word_all, avg_chunk_len, entropy_all])
labels_corr = ["Delta", "Chunks", "Words", "Avg Chunk Len", "Entropy"]

im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(labels_corr, rotation=45, ha="right")
ax.set_yticklabels(labels_corr)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(matrix[i, j]) > 0.5 else "black")
ax.set_title("Figure 8 — Correlation Matrix")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig8_correlation.png", bbox_inches="tight")
plt.close()
print("✓ fig8_correlation.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 9 — Summary table (text)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")

headers = ["Model", "AUC", "F1", "FPR", "TPR", "Direction", "Sep (rad)"]
rows = []
for model_key in ["nvidia/nv-embedqa-e5-v5", "nvidia/llama-nemotron-embed-1b-v2", "baai/bge-m3"]:
    m = cache["models"][model_key]
    name = model_key.split("/")[-1]
    d = np.array(m["deltas"])
    sign = 1 if m["obf_mean"] > m["ben_mean"] else -1
    sc = sign * d
    y = (labels == "obfuscation").astype(int)
    auc = roc_auc_score(y, sc)
    fpr_a, tpr_a, _ = roc_curve(y, sc)
    j = tpr_a - fpr_a; idx = np.argmax(j)
    direction = "HIGH=obf" if m["obf_mean"] > m["ben_mean"] else "LOW=obf"
    sep = abs(m["obf_mean"] - m["ben_mean"])
    rows.append([name, f"{auc:.3f}", "—", f"{fpr_a[idx]:.3f}", f"{tpr_a[idx]:.3f}", direction, f"{sep:.4f}"])

table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#E8E8E8")
        cell.set_text_props(fontweight="bold")
ax.set_title("Figure 9 — Summary Results (Sentence Chunking, N=1100)", fontsize=13, fontweight="bold", pad=20)
fig.tight_layout()
fig.savefig(f"{OUT}/fig9_summary_table.png", bbox_inches="tight")
plt.close()
print("✓ fig9_summary_table.png")


print(f"\nAll 9 figures saved to {OUT}/")
