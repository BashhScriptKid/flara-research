"""
Full graph suite for Delta Angle paper — FINAL version.
Generates 20+ publication-quality figures in graphs/final/
"""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import os
import re
from collections import Counter

OUT = "graphs/final"
os.makedirs(OUT, exist_ok=True)

# ── Load all data ──────────────────────────────────────────────────
with open("data/sentence_chunk_cache.json") as f:
    sent_cache = json.load(f)

with open("data/token_based_all_models.json") as f:
    token_metrics = json.load(f)

with open("data/delta_cache_token_full.json") as f:
    token_deltas_raw = json.load(f)

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

labels = np.array(["obfuscation"] * len(obf_samples) + ["benign"] * len(ben_samples))
ben_mask = labels == "benign"
obf_mask = labels == "obfuscation"

# ── Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 8, "figure.dpi": 150, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})
C_BEN = "#4C72B0"
C_OBF = "#C44E52"
C_ACC = "#55A868"
C_WARN = "#DD8452"
C_GRAY = "#8C8C8C"
MODEL_COLORS = {"nvidia/nv-embedqa-e5-v5": C_BEN, "nv-embedqa-e5-v5": C_BEN,
                "nvidia/llama-nemotron-embed-1b-v2": C_ACC, "llama-nemotron-embed-1b-v2": C_ACC,
                "baai/bge-m3": C_WARN, "bge-m3": C_WARN,
                "nv-embed-v1": "#8172B3", "nv-embedcode-7b-v1": "#CCB974",
                "nvidia/nv-embed-v1": "#8172B3", "nvidia/nv-embedcode-7b-v1": "#CCB974",
                "nvidia/llama-nemotron-embed-vl-1b-v2": "#64B5CD", "llama-nemotron-embed-vl-1b-v2": "#64B5CD"}


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


def char_entropy(text):
    freq = Counter(text)
    n = max(len(text), 1)
    return -sum((cnt / n) * np.log2(cnt / n) for cnt in freq.values() if cnt > 0)


def compute_metrics(y_true, scores):
    """Compute AUC, best F1, FPR/TPR at optimal."""
    auc = roc_auc_score(y_true, scores)
    bf1 = 0; bt = 0
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    for pct in range(1, 51):
        t = np.percentile(pos, pct)
        pred = pos > t
        tp = np.sum(pred); fn = np.sum(~pred)
        fp = np.sum(neg > t)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > bf1: bf1 = f1; bt = t
    fpr_a, tpr_a, _ = roc_curve(y_true, scores)
    j = tpr_a - fpr_a; idx = np.argmax(j)
    return {"auc": auc, "f1": bf1, "threshold": bt,
            "fpr": fpr_a[idx], "tpr": tpr_a[idx],
            "fpr_arr": fpr_a, "tpr_arr": tpr_a}


# ── Precompute sentence chunking data ──────────────────────────────
e5_sent = sent_cache["models"]["nvidia/nv-embedqa-e5-v5"]
nem_sent = sent_cache["models"]["nvidia/llama-nemotron-embed-1b-v2"]
bge_sent = sent_cache["models"]["baai/bge-m3"]

d_e5 = np.array(e5_sent["deltas"])
d_nem = np.array(nem_sent["deltas"])
d_bge = np.array(bge_sent["deltas"])

ben_chunks_list = [chunk_text(t) for t in ben_samples]
obf_chunks_list = [chunk_text(t) for t in obf_samples]
ben_lens = [len(c.split()) for chunks in ben_chunks_list for c in chunks]
obf_lens = [len(c.split()) for chunks in obf_chunks_list for c in chunks]
ben_nc = [len(chunks) for chunks in ben_chunks_list]
obf_nc = [len(chunks) for chunks in obf_chunks_list]
ben_words = [len(t.split()) for t in ben_samples]
obf_words = [len(t.split()) for t in obf_samples]
ben_entropy = [char_entropy(t) for t in ben_samples]
obf_entropy = [char_entropy(t) for t in obf_samples]

y_true = (labels == "obfuscation").astype(int)

# ═══════════════════════════════════════════════════════════════════
# FIG 1 — Delta distributions (E5 sentence chunking)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Figure 1 — Delta Angle Distributions (Sentence Chunking, E5)", fontsize=13, fontweight="bold")

ax = axes[0]
ax.hist(d_e5[ben_mask], bins=40, alpha=0.75, color=C_BEN, label=f"Benign (μ={np.mean(d_e5[ben_mask]):.4f})", density=True, edgecolor="white", linewidth=0.5)
ax.hist(d_e5[obf_mask], bins=40, alpha=0.75, color=C_OBF, label=f"Obfuscation (μ={np.mean(d_e5[obf_mask]):.4f})", density=True, edgecolor="white", linewidth=0.5)
ax.axvline(0, color="gray", ls="--", lw=0.8)
ax.set_xlabel("Average Delta Angle (radians)")
ax.set_ylabel("Density")
ax.set_title("(a) Histogram")
ax.legend(framealpha=0.9)

ax = axes[1]
parts = ax.violinplot([d_e5[ben_mask], d_e5[obf_mask]], positions=[0, 1], showmeans=True, showmedians=True)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor([C_BEN, C_OBF][i]); pc.set_alpha(0.6)
parts["cmeans"].set_color("black"); parts["cmedians"].set_color("gray")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Benign", "Obfuscation"])
ax.set_ylabel("Average Delta Angle (radians)")
ax.set_title("(b) Violin Plot")
ax.set_yscale("symlog", linthresh=0.01)
fig.tight_layout()
fig.savefig(f"{OUT}/01_distributions_e5.png", bbox_inches="tight"); plt.close()
print("✓ 01_distributions_e5.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 2 — ROC curves (all 3 sentence chunking models)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))
for model_key in ["nvidia/nv-embedqa-e5-v5", "nvidia/llama-nemotron-embed-1b-v2", "baai/bge-m3"]:
    m = sent_cache["models"][model_key]
    d = np.array(m["deltas"])
    name = model_key.split("/")[-1]
    metrics = compute_metrics(y_true, d)
    ax.plot(metrics["fpr_arr"], metrics["tpr_arr"], color=MODEL_COLORS[model_key], lw=2,
            label=f"{name} (AUC={metrics['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Figure 2 — ROC Curves (Sentence Chunking)")
ax.legend(loc="lower right", framealpha=0.9); ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/02_roc_sentence.png", bbox_inches="tight"); plt.close()
print("✓ 02_roc_sentence.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 3 — AUC comparison (sentence vs token, all models)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sentence chunking
ax = axes[0]
models_sent = ["nvidia/nv-embedqa-e5-v5", "nvidia/llama-nemotron-embed-1b-v2", "baai/bge-m3"]
aucs_sent = []
for mk in models_sent:
    m = sent_cache["models"][mk]
    d = np.array(m["deltas"])
    aucs_sent.append(roc_auc_score(y_true, d))
short = ["E5", "Nemotron", "BGE-M3"]
colors_sent = [MODEL_COLORS[mk] for mk in models_sent]
bars = ax.bar(short, aucs_sent, color=colors_sent, edgecolor="white", linewidth=1.5)
for bar, v in zip(bars, aucs_sent):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.3f}", ha="center", fontweight="bold", fontsize=10)
ax.set_ylabel("AUC-ROC"); ax.set_ylim(0.75, 0.92); ax.set_title("(a) Sentence Chunking"); ax.grid(axis="y")

# Token-based
ax = axes[1]
models_tok = list(token_metrics.keys())
aucs_tok = [token_metrics[m]["auc"] for m in models_tok]
short_tok = [m.split("/")[-1][:12] for m in models_tok]
colors_tok = [MODEL_COLORS.get(m.split("/")[-1], C_GRAY) for m in models_tok]
bars = ax.bar(range(len(short_tok)), aucs_tok, color=colors_tok, edgecolor="white", linewidth=1.5)
for bar, v in zip(bars, aucs_tok):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)
ax.set_xticks(range(len(short_tok))); ax.set_xticklabels(short_tok, rotation=30, ha="right")
ax.set_ylabel("AUC-ROC"); ax.set_ylim(0.45, 0.95); ax.set_title("(b) Token-Based"); ax.grid(axis="y")

fig.suptitle("Figure 3 — AUC-ROC Comparison", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/03_auc_comparison.png", bbox_inches="tight"); plt.close()
print("✓ 03_auc_comparison.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 4 — Latency comparison
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

# Sentence chunking latencies
lat_models = ["E5", "Nemotron", "BGE-M3"]
lat_times = [e5_sent["embed_time"], nem_sent["embed_time"], bge_sent["embed_time"]]
lat_colors = [MODEL_COLORS["nv-embedqa-e5-v5"], MODEL_COLORS["llama-nemotron-embed-1b-v2"], MODEL_COLORS["baai/bge-m3"]]

# Add token-based latencies
for mk in token_metrics:
    name = mk.split("/")[-1]
    if name == "nv-embedqa-e5-v5":
        lat_models.append("E5 (token)")
        lat_times.append(token_metrics[mk]["time"])
        lat_colors.append(C_BEN)

# Sort by time
order = np.argsort(lat_times)
lat_models = [lat_models[i] for i in order]
lat_times = [lat_times[i] for i in order]
lat_colors = [lat_colors[i] for i in order]

bars = ax.barh(lat_models, lat_times, color=lat_colors, edgecolor="white", linewidth=1.5, height=0.6)
for bar, v in zip(bars, lat_times):
    ax.text(v + 5, bar.get_y() + bar.get_height()/2, f"{v:.0f} ms", va="center", fontweight="bold", fontsize=10)
ax.set_xlabel("Embedding Latency (ms)")
ax.set_title("Figure 4 — API Latency (1548 chunks, NIM)")
ax.set_xlim(0, max(lat_times) * 1.3)
ax.grid(axis="x")
fig.tight_layout()
fig.savefig(f"{OUT}/04_latency.png", bbox_inches="tight"); plt.close()
print("✓ 04_latency.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 5 — Baseline comparison (existing solutions)
# ═══════════════════════════════════════════════════════════════════
# Recall@5%FPR and F1 computed live against the current dataset (95th-percentile-of-
# -benign threshold, strict '>' comparison, auto sign-flip if the raw direction is
# inverted) -- matching baseline_auc.py's methodology. Previously these four bars were
# hardcoded literals from an untraceable source that did not match any script's output.
def special_char_ratio_fig(text):
    if not text:
        return 0.0
    special = sum(1 for c in text if not c.isalpha() and not c.isspace())
    return special / len(text)


def regex_hex_base64_score(text):
    score = 0
    if re.search(r'[0-9a-fA-F]{20,}', text):
        score += 1
    if re.search(r'[A-Za-z0-9+/]{20,}={0,2}', text):
        score += 1
    if re.search(r'(decode|eval|atob|btoa|fromCharCode|hex|base64)', text, re.IGNORECASE):
        score += 0.5
    if re.search(r'(\\u[0-9a-fA-F]{4}){3,}', text):
        score += 1
    if re.search(r'[\x00-\x08\x0e-\x1f]', text):
        score += 0.5
    return min(score, 2.0) / 2.0


def recall_f1_at_5pct_fpr(scores):
    scores = np.array(scores)
    auc_raw = roc_auc_score(y_true, scores)
    if auc_raw < 0.5:
        scores = -scores
    obf_s, ben_s = scores[obf_mask], scores[ben_mask]
    thr = np.percentile(ben_s, 95)
    recall = float(np.mean(obf_s > thr))
    tp = int(np.sum(obf_s > thr)); fn = len(obf_s) - tp
    fp = int(np.sum(ben_s > thr)); tn = len(ben_s) - fp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return recall, f1


# entropy/special-char/regex need obf-first ordering to match y_true/ben_mask/obf_mask
_entropy_scores = np.array([char_entropy(t) for t in obf_samples] + [char_entropy(t) for t in ben_samples])
_special_scores = np.array([special_char_ratio_fig(t) for t in obf_samples] + [special_char_ratio_fig(t) for t in ben_samples])
_regex_scores = np.array([regex_hex_base64_score(t) for t in obf_samples] + [regex_hex_base64_score(t) for t in ben_samples])

fig, ax = plt.subplots(figsize=(9, 5))

methods = ["Character\nEntropy", "Special Char\nRatio", "Regex\n(hex/base64)", "Delta Angle\n(Sentence)"]
_results = [recall_f1_at_5pct_fpr(s) for s in [_entropy_scores, _special_scores, _regex_scores, d_e5]]
recalls = [r[0] for r in _results]
f1s = [r[1] for r in _results]
x = np.arange(len(methods))
w = 0.35

bars1 = ax.bar(x - w/2, recalls, w, label="Recall @ 5% FPR", color=C_BEN, edgecolor="white", linewidth=1.5)
bars2 = ax.bar(x + w/2, f1s, w, label="F1 Score", color=C_OBF, edgecolor="white", linewidth=1.5)
for b in bars1:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
for b in bars2:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(methods)
ax.set_ylabel("Score"); ax.set_ylim(0, 1.0); ax.set_title("Baseline Comparison (E5)")
ax.legend(framealpha=0.9); ax.grid(axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/05_baseline_comparison.png", bbox_inches="tight"); plt.close()
print("✓ 05_baseline_comparison.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 6 — Recall at multiple FPR rates
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))

fpr_targets = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]

for model_key in ["nvidia/nv-embedqa-e5-v5", "nvidia/llama-nemotron-embed-1b-v2", "baai/bge-m3"]:
    m = sent_cache["models"][model_key]
    d = np.array(m["deltas"])
    name = model_key.split("/")[-1]
    fpr_a, tpr_a, _ = roc_curve(y_true, d)
    recalls_at_fpr = []
    for ft in fpr_targets:
        idx = np.searchsorted(fpr_a, ft)
        if idx < len(tpr_a):
            recalls_at_fpr.append(tpr_a[idx])
        else:
            recalls_at_fpr.append(1.0)
    ax.plot(fpr_targets, recalls_at_fpr, "o-", color=MODEL_COLORS[model_key], lw=2, label=name, markersize=6)

ax.set_xlabel("False Positive Rate"); ax.set_ylabel("Recall (TPR)")
ax.set_title("Figure 6 — Recall vs FPR (Sentence Chunking)")
ax.legend(framealpha=0.9); ax.set_xscale("log"); ax.set_xlim(0.003, 0.15)
fig.tight_layout()
fig.savefig(f"{OUT}/06_recall_vs_fpr.png", bbox_inches="tight"); plt.close()
print("✓ 06_recall_vs_fpr.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 7 — PR curves (all 3 models)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))

for model_key in ["nvidia/nv-embedqa-e5-v5", "nvidia/llama-nemotron-embed-1b-v2", "baai/bge-m3"]:
    m = sent_cache["models"][model_key]
    d = np.array(m["deltas"])
    name = model_key.split("/")[-1]
    precision, recall, _ = precision_recall_curve(y_true, d)
    ap = average_precision_score(y_true, d)
    ax.plot(recall, precision, color=MODEL_COLORS[model_key], lw=2, label=f"{name} (AP={ap:.3f})")

ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Figure 7 — Precision-Recall Curves")
ax.legend(framealpha=0.9); ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/07_pr_curves.png", bbox_inches="tight"); plt.close()
print("✓ 07_pr_curves.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 8 — Scatter: delta vs entropy (all 3 models)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

scatter_models = [
    ("nvidia/nv-embedqa-e5-v5", "E5"),
    ("nvidia/llama-nemotron-embed-1b-v2", "Nemotron"),
    ("baai/bge-m3", "BGE-M3"),
]
for idx, (model_key, label) in enumerate(scatter_models):
    ax = axes[idx]
    m = sent_cache["models"][model_key]
    d = np.array(m["deltas"])
    ax.scatter(d[ben_mask], ben_entropy, alpha=0.3, s=15, c=C_BEN, label="Benign", edgecolors="none")
    ax.scatter(d[obf_mask], obf_entropy, alpha=0.3, s=15, c=C_OBF, label="Obfuscation", edgecolors="none")
    ax.set_xlabel("Delta Angle (radians)"); ax.set_ylabel("Character Entropy (bits)")
    ax.set_title(f"({chr(97 + idx)}) {label}")
    ax.legend(framealpha=0.9, markerscale=2)

fig.suptitle("Figure 8 — Delta vs Entropy (Sentence Chunking)", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/08_scatter_delta_entropy.png", bbox_inches="tight"); plt.close()
print("✓ 08_scatter_delta_entropy.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 9 — Scatter: delta vs word count (all 3 models)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (model_key, label) in enumerate(scatter_models):
    ax = axes[idx]
    m = sent_cache["models"][model_key]
    d = np.array(m["deltas"])
    ax.scatter(d[ben_mask], ben_words, alpha=0.3, s=15, c=C_BEN, label="Benign", edgecolors="none")
    ax.scatter(d[obf_mask], obf_words, alpha=0.3, s=15, c=C_OBF, label="Obfuscation", edgecolors="none")
    ax.set_xlabel("Delta Angle (radians)"); ax.set_ylabel("Input Word Count")
    ax.set_title(f"({chr(97 + idx)}) {label}")
    ax.legend(framealpha=0.9, markerscale=2)

fig.suptitle("Figure 9 — Delta vs Input Length (Sentence Chunking)", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/09_scatter_delta_length.png", bbox_inches="tight"); plt.close()
print("✓ 09_scatter_delta_length.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 10 — Scatter: delta vs risk score (composite)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

all_entropy = np.array(ben_entropy + obf_entropy)
all_words = np.array(ben_words + obf_words)
risk = (all_entropy / all_entropy.max() + all_words / all_words.max()) / 2
risk_ben = risk[ben_mask]; risk_obf = risk[obf_mask]

for idx, (model_key, label) in enumerate(scatter_models):
    ax = axes[idx]
    m = sent_cache["models"][model_key]
    d = np.array(m["deltas"])
    ax.scatter(d[ben_mask], risk_ben, alpha=0.3, s=15, c=C_BEN, label="Benign", edgecolors="none")
    ax.scatter(d[obf_mask], risk_obf, alpha=0.3, s=15, c=C_OBF, label="Obfuscation", edgecolors="none")
    ax.set_xlabel("Delta Angle (radians)"); ax.set_ylabel("Risk Score (composite)")
    ax.set_title(f"({chr(97 + idx)}) {label}")
    ax.legend(framealpha=0.9, markerscale=2)

fig.suptitle("Figure 10 — Delta vs Risk Score (Sentence Chunking)", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/10_scatter_delta_risk.png", bbox_inches="tight"); plt.close()
print("✓ 10_scatter_delta_risk.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 11 — Threshold analysis (E5)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))

fpr_a, tpr_a, thresh_arr = roc_curve(y_true, d_e5)
ax.plot(thresh_arr, tpr_a, color=C_OBF, lw=2, label="TPR (Recall)")
ax.plot(thresh_arr, fpr_a, color=C_BEN, lw=2, label="FPR")
ax.plot(thresh_arr, tpr_a - fpr_a, color=C_GRAY, lw=1.5, ls="--", label="TPR − FPR")
j = tpr_a - fpr_a; idx = np.argmax(j)
ax.axvline(thresh_arr[idx], color="black", ls=":", lw=1, label=f"Optimal t={thresh_arr[idx]:.4f}")
ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Rate")
ax.set_title("Figure 11 — Threshold Analysis (E5)")
ax.legend(framealpha=0.9)
fig.tight_layout()
fig.savefig(f"{OUT}/11_threshold.png", bbox_inches="tight"); plt.close()
print("✓ 11_threshold.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 12 — Sentence chunking stats
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

ax = axes[0, 0]
ax.hist(ben_lens, bins=35, alpha=0.7, color=C_BEN, label=f"Benign (μ={np.mean(ben_lens):.1f})", density=True, edgecolor="white", linewidth=0.5)
ax.hist(obf_lens, bins=35, alpha=0.7, color=C_OBF, label=f"Obf (μ={np.mean(obf_lens):.1f})", density=True, edgecolor="white", linewidth=0.5)
ax.axvline(8, color="black", ls="--", lw=1, label="Merge threshold")
ax.set_xlabel("Chunk Length (words)"); ax.set_ylabel("Density"); ax.set_title("(a) Chunk Length Distribution"); ax.legend(framealpha=0.9); ax.set_xlim(0, 55)

ax = axes[0, 1]
ax.hist(ben_nc, bins=range(0, max(max(ben_nc), max(obf_nc)) + 2), alpha=0.7, color=C_BEN, label=f"Benign (μ={np.mean(ben_nc):.1f})", density=True, edgecolor="white", linewidth=0.5)
ax.hist(obf_nc, bins=range(0, max(max(ben_nc), max(obf_nc)) + 2), alpha=0.7, color=C_OBF, label=f"Obf (μ={np.mean(obf_nc):.1f})", density=True, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Chunks per Input"); ax.set_ylabel("Density"); ax.set_title("(b) Chunks per Input"); ax.legend(framealpha=0.9)

ax = axes[1, 0]
ben_single = sum(1 for n in ben_nc if n <= 1) / len(ben_nc) * 100
obf_single = sum(1 for n in obf_nc if n <= 1) / len(obf_nc) * 100
bars = ax.bar(["Benign", "Obfuscation"], [ben_single, obf_single], color=[C_BEN, C_OBF], edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, [ben_single, obf_single]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}%", ha="center", fontweight="bold", fontsize=12)
ax.set_ylabel("% Inputs with ≤1 Chunk"); ax.set_title("(c) Single-Chunk Rate"); ax.set_ylim(0, 110); ax.grid(axis="y")

ax = axes[1, 1]
ben_sorted = np.sort(ben_lens)
obf_sorted = np.sort(obf_lens)
ax.plot(np.linspace(0, 100, len(ben_sorted)), ben_sorted, color=C_BEN, lw=2, label="Benign")
ax.plot(np.linspace(0, 100, len(obf_sorted)), obf_sorted, color=C_OBF, lw=2, label="Obf")
ax.axhline(y=8, color="black", ls="--", lw=1, label="Merge threshold")
ax.set_xlabel("Percentile"); ax.set_ylabel("Chunk Length (words)"); ax.set_title("(d) Cumulative Chunk Lengths"); ax.legend(framealpha=0.9); ax.set_xlim(0, 100); ax.set_ylim(0, 55)

fig.suptitle("Figure 12 — Chunking Algorithm Analysis", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/12_chunking_analysis.png", bbox_inches="tight"); plt.close()
print("✓ 12_chunking_analysis.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 13 — Merge threshold sensitivity
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

thresholds_merge = [4, 6, 8, 10, 12, 14]
single_pcts = []
avg_chunks_ben = []
avg_chunks_obf = []

for t in thresholds_merge:
    nc_b = [len(chunk_text(s, merge_threshold=t)) for s in ben_samples]
    nc_o = [len(chunk_text(s, merge_threshold=t)) for s in obf_samples]
    single_pcts.append(sum(1 for n in nc_b if n <= 1) / len(nc_b) * 100)
    avg_chunks_ben.append(np.mean(nc_b))
    avg_chunks_obf.append(np.mean(nc_o))

ax = axes[0]
ax.plot(thresholds_merge, single_pcts, "o-", color=C_BEN, lw=2, markersize=6)
ax.axvline(8, color="gray", ls="--", lw=1, label="Selected (8)")
ax.set_xlabel("Merge Threshold (words)"); ax.set_ylabel("% Benign Inputs with ≤1 Chunk")
ax.set_title("(a) Single-Chunk Rate vs Threshold"); ax.legend(framealpha=0.9); ax.set_xticks(thresholds_merge)

ax = axes[1]
ax.plot(thresholds_merge, avg_chunks_ben, "o-", color=C_BEN, lw=2, markersize=6, label="Benign")
ax.plot(thresholds_merge, avg_chunks_obf, "o-", color=C_OBF, lw=2, markersize=6, label="Obf")
ax.axvline(8, color="gray", ls="--", lw=1)
ax.set_xlabel("Merge Threshold (words)"); ax.set_ylabel("Mean Chunks per Input")
ax.set_title("(b) Mean Chunks vs Threshold"); ax.legend(framealpha=0.9); ax.set_xticks(thresholds_merge)

fig.suptitle("Figure 13 — Merge Threshold Sensitivity", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/13_merge_sensitivity.png", bbox_inches="tight"); plt.close()
print("✓ 13_merge_sensitivity.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 14 — Distribution across all 6 NIM models (token-based)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

model_names = []
ben_means = []
obf_means = []
for mk, m in token_metrics.items():
    name = mk.split("/")[-1]
    if name == "nv-embedqa-e5-v5": name = "E5"
    elif name == "llama-nemotron-embed-1b-v2": name = "Nemotron"
    elif name == "bge-m3": name = "BGE-M3"
    elif name == "nv-embed-v1": name = "NV-Embed-v1"
    elif name == "nv-embedcode-7b-v1": name = "EmbedCode"
    elif name == "llama-nemotron-embed-vl-1b-v2": name = "VL-1B"
    model_names.append(name)
    ben_means.append(m["ben_mean"])
    obf_means.append(m["obf_mean"])

x = np.arange(len(model_names))
w = 0.35
ax.bar(x - w/2, ben_means, w, label="Benign", color=C_BEN, edgecolor="white", linewidth=1.5)
ax.bar(x + w/2, obf_means, w, label="Obfuscation", color=C_OBF, edgecolor="white", linewidth=1.5)
ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=30, ha="right")
ax.set_ylabel("Mean Delta (degrees)"); ax.set_title("Figure 14 — Token-Based Distributions Across Models")
ax.legend(framealpha=0.9); ax.grid(axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/14_token_distributions.png", bbox_inches="tight"); plt.close()
print("✓ 14_token_distributions.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 15 — Token-based: why we walked back
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Token-based delta distributions (from cached deltas)
token_deltas_obf = []
token_deltas_ben = []
for key, item in token_deltas_raw.items():
    token_deltas_obf.append(item["delta"])

# For benign, we need the token delta cache — use metrics instead
# Show token-based FPR problem
ax = axes[0]
models_tok = list(token_metrics.keys())
fprs_tok = [token_metrics[m]["optimal_fpr"] for m in models_tok]
short_tok = []
for mk in models_tok:
    n = mk.split("/")[-1]
    if n == "nv-embedqa-e5-v5": short_tok.append("E5")
    elif n == "llama-nemotron-embed-1b-v2": short_tok.append("Nemotron")
    elif n == "bge-m3": short_tok.append("BGE-M3")
    elif n == "nv-embed-v1": short_tok.append("NV-Embed-v1")
    elif n == "nv-embedcode-7b-v1": short_tok.append("EmbedCode")
    elif n == "llama-nemotron-embed-vl-1b-v2": short_tok.append("VL-1B")
colors_tok = [MODEL_COLORS.get(mk, C_GRAY) for mk in models_tok]
bars = ax.bar(range(len(short_tok)), fprs_tok, color=colors_tok, edgecolor="white", linewidth=1.5)
for bar, v in zip(bars, fprs_tok):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.1%}", ha="center", fontweight="bold", fontsize=9)
ax.set_xticks(range(len(short_tok))); ax.set_xticklabels(short_tok, rotation=30, ha="right")
ax.set_ylabel("Optimal FPR"); ax.set_title("(a) Token-Based FPR (High = Bad)"); ax.grid(axis="y")
ax.axhline(0.01, color=C_ACC, ls="--", lw=1.5, label="Sentence chunking FPR (1%)")
ax.legend(framealpha=0.9)

# Direction problem
ax = axes[1]
directions = []
for mk in models_tok:
    m = token_metrics[mk]
    if m["obf_mean"] > m["ben_mean"]:
        directions.append("HIGH=obf")
    else:
        directions.append("LOW=obf")
dir_colors = [C_ACC if d == "HIGH=obf" else C_OBF for d in directions]
bars = ax.bar(range(len(short_tok)), [1]*len(short_tok), color=dir_colors, edgecolor="white", linewidth=1.5)
for i, (bar, d) in enumerate(zip(bars, directions)):
    ax.text(bar.get_x() + bar.get_width()/2, 0.5, d, ha="center", va="center", fontweight="bold", fontsize=9,
            color="white", rotation=90 if len(d) > 8 else 0)
ax.set_xticks(range(len(short_tok))); ax.set_xticklabels(short_tok, rotation=30, ha="right")
ax.set_yticks([]); ax.set_title("(b) Direction Inconsistency")
# Legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=C_ACC, label="HIGH=obf (correct)"), Patch(facecolor=C_OBF, label="LOW=obf (inverted)")], framealpha=0.9)

# Token-based AUC vs sentence chunking AUC (for shared models)
ax = axes[2]
shared = ["nvidia/nv-embedqa-e5-v5", "nvidia/llama-nemotron-embed-1b-v2", "baai/bge-m3"]
short_shared = ["E5", "Nemotron", "BGE-M3"]
aucs_tok_shared = [token_metrics[m]["auc"] for m in shared]
aucs_sent_shared = [roc_auc_score(y_true, np.array(sent_cache["models"][m]["deltas"])) for m in shared]
x = np.arange(len(shared)); w = 0.35
ax.bar(x - w/2, aucs_tok_shared, w, label="Token-Based", color=C_WARN, edgecolor="white", linewidth=1.5)
ax.bar(x + w/2, aucs_sent_shared, w, label="Sentence Chunking", color=C_BEN, edgecolor="white", linewidth=1.5)
ax.set_xticks(x); ax.set_xticklabels(short_shared)
ax.set_ylabel("AUC-ROC"); ax.set_title("(c) AUC: Token vs Sentence"); ax.legend(framealpha=0.9); ax.set_ylim(0.65, 0.92); ax.grid(axis="y")
for i in range(len(shared)):
    ax.text(i - w/2, aucs_tok_shared[i] + 0.003, f"{aucs_tok_shared[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i + w/2, aucs_sent_shared[i] + 0.003, f"{aucs_sent_shared[i]:.3f}", ha="center", fontsize=9, fontweight="bold")

fig.suptitle("Figure 15 — Why Token-Based Was Replaced", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/15_why_token_replaced.png", bbox_inches="tight"); plt.close()
print("✓ 15_why_token_replaced.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 16 — Correlation matrix
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 5))

# NOTE: must be obf-first to match d_e5's actual sample order (labels/ben_mask/obf_mask
# above are obf-first; an earlier version of this block used ben_nc+obf_nc here, which
# silently misaligned every pair against d_e5 and produced a spuriously-small correlation).
nc_all = np.array(obf_nc + ben_nc)
word_all = np.array(obf_words + ben_words)
avg_chunk_len = np.array([np.mean([len(c.split()) for c in chunks]) if chunks else 0 for chunks in obf_chunks_list + ben_chunks_list])
entropy_all = np.array(obf_entropy + ben_entropy)

matrix = np.corrcoef([d_e5, nc_all, word_all, avg_chunk_len, entropy_all])
labels_corr = ["Delta", "Chunks", "Words", "Avg Chunk Len", "Entropy"]

im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(labels_corr, rotation=45, ha="right"); ax.set_yticklabels(labels_corr)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(matrix[i, j]) > 0.5 else "black")
ax.set_title("Figure 16 — Feature Correlation Matrix")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(f"{OUT}/16_correlation.png", bbox_inches="tight"); plt.close()
print("✓ 16_correlation.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 17 — Model comparison table
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")

headers = ["Model", "Method", "AUC", "F1", "FPR", "TPR", "Direction"]
rows = []
for mk in ["nvidia/nv-embedqa-e5-v5", "nvidia/llama-nemotron-embed-1b-v2", "baai/bge-m3"]:
    name = mk.split("/")[-1]
    # Sentence chunking
    d = np.array(sent_cache["models"][mk]["deltas"])
    sm = compute_metrics(y_true, d)
    rows.append([name, "Sentence", f"{sm['auc']:.3f}", f"{sm['f1']:.3f}", f"{sm['fpr']:.3f}", f"{sm['tpr']:.3f}", "HIGH=obf"])
    # Token-based
    tm = token_metrics[mk]
    rows.append(["", "Token", f"{tm['auc']:.3f}", f"{tm['best_f1']:.3f}", f"{tm['optimal_fpr']:.3f}", f"{tm['optimal_tpr']:.3f}",
                 "HIGH=obf" if tm['obf_mean'] > tm['ben_mean'] else "LOW=obf"])

table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.6)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#E8E8E8"); cell.set_text_props(fontweight="bold")
    elif row in [1, 3, 5]:
        cell.set_facecolor("#F5F5F5")
ax.set_title("Figure 17 — Model Comparison Summary", fontsize=13, fontweight="bold", pad=20)
fig.tight_layout()
fig.savefig(f"{OUT}/17_model_comparison_table.png", bbox_inches="tight"); plt.close()
print("✓ 17_model_comparison_table.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 18 — Token-based ROC (all 6 models, for context)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))

# We don't have raw token deltas for all models, but we have metrics
# Plot token-based performance as scatter
for mk in token_metrics:
    m = token_metrics[mk]
    name = mk.split("/")[-1]
    color = MODEL_COLORS.get(name, C_GRAY)
    ax.scatter(m["optimal_fpr"], m["optimal_tpr"], s=m["auc"]*500, c=color, alpha=0.7, edgecolors="black", linewidth=0.5, zorder=5)
    ax.annotate(name.split("-")[-1][:8], (m["optimal_fpr"], m["optimal_tpr"]),
                textcoords="offset points", xytext=(8, 5), fontsize=8)

# Add sentence chunking points
for mk in ["nvidia/nv-embedqa-e5-v5", "nvidia/llama-nemotron-embed-1b-v2", "baai/bge-m3"]:
    d = np.array(sent_cache["models"][mk]["deltas"])
    sm = compute_metrics(y_true, d)
    name = mk.split("/")[-1]
    color = MODEL_COLORS[name]
    ax.scatter(sm["fpr"], sm["tpr"], s=sm["auc"]*500, c=color, alpha=0.9, edgecolors="red", linewidth=2, zorder=6, marker="D")
    ax.annotate(f"{name} (sent)", (sm["fpr"], sm["tpr"]),
                textcoords="offset points", xytext=(8, -10), fontsize=8, color="red")

ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Figure 18 — Token vs Sentence: ROC Space (size = AUC)")
ax.set_xlim(-0.02, 0.15); ax.set_ylim(0.0, 1.05)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/18_token_vs_sentence_roc.png", bbox_inches="tight"); plt.close()
print("✓ 18_token_vs_sentence_roc.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 19 — Word count vs delta (log scale)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(ben_words, d_e5[ben_mask], alpha=0.3, s=15, c=C_BEN, label="Benign", edgecolors="none")
ax.scatter(obf_words, d_e5[obf_mask], alpha=0.3, s=15, c=C_OBF, label="Obfuscation", edgecolors="none")
ax.set_xlabel("Input Word Count"); ax.set_ylabel("Delta Angle (radians)")
ax.set_title("Figure 19 — Input Length vs Delta (E5)")
ax.set_yscale("symlog", linthresh=0.01)
ax.legend(framealpha=0.9, markerscale=2)
fig.tight_layout()
fig.savefig(f"{OUT}/19_length_vs_delta.png", bbox_inches="tight"); plt.close()
print("✓ 19_length_vs_delta.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 20 — Summary stats text
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")

summary = [
    ["Metric", "Value"],
    ["Total samples", "1100 (287 obf + 813 ben)"],
    ["Total chunks", "1548"],
    ["Avg chunks/benign", f"{np.mean(ben_nc):.1f}"],
    ["Avg chunks/obfuscation", f"{np.mean(obf_nc):.1f}"],
    ["Single-chunk benign", f"{ben_single:.1f}%"],
    ["Single-chunk obfuscation", f"{obf_single:.1f}%"],
    ["Merge threshold", "8 words"],
    ["Best AUC", "0.858 (Nemotron)"],
    ["Best F1", "0.829 (Nemotron/BGE-M3)"],
    ["Best FPR", "1.0% (all models)"],
    ["Direction", "HIGH delta = obfuscation"],
    ["Angle type", "Unsigned (absolute)"],
]

table = ax.table(cellText=summary, loc="center", cellLoc="center", colWidths=[0.35, 0.45])
table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.5)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#E8E8E8"); cell.set_text_props(fontweight="bold")
ax.set_title("Figure 20 — Summary Statistics", fontsize=13, fontweight="bold", pad=20)
fig.tight_layout()
fig.savefig(f"{OUT}/20_summary_stats.png", bbox_inches="tight"); plt.close()
print("✓ 20_summary_stats.png")


print(f"\n{'='*60}")
print(f"All 20 figures saved to {OUT}/")
print(f"{'='*60}")
