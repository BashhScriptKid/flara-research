"""
Obfuscation-focused analysis — per-type breakdown (v3).
Includes multi-granularity (sentence + paragraph) and baseline methods.
Generates graphs in graphs/final/obf_analysis/
"""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score, roc_curve
import os
import re
from collections import Counter, defaultdict

OUT = "graphs/final/obf_analysis"
os.makedirs(OUT, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────
with open("data/sentence_chunk_cache.json") as f:
    cache = json.load(f)

para_deltas = np.load("data/para_deltas_full.npy")

obf_samples = json.load(open("data/obf_trigger.json"))
ben_samples = json.load(open("data/obf_benign.json"))
n_obf = len(obf_samples)
n_ben = len(ben_samples)

labels_all = np.array(["obfuscation"] * n_obf + ["benign"] * n_ben)
y_true = (labels_all == "obfuscation").astype(int)

# Sentence deltas (E5, averaged across models for robustness)
d_sent_e5 = np.array(cache["models"]["nvidia/nv-embedqa-e5-v5"]["deltas"])
d_sent_nem = np.array(cache["models"]["nvidia/llama-nemotron-embed-1b-v2"]["deltas"])
d_sent_bge = np.array(cache["models"]["baai/bge-m3"]["deltas"])
d_sent_avg = (d_sent_e5 + d_sent_nem + d_sent_bge) / 3

# Combined sentence + paragraph
d_combined = 0.5 * d_sent_avg + 0.5 * para_deltas

# ── Baseline methods ───────────────────────────────────────────────
from collections import Counter as Ctr

def char_entropy(text):
    freq = Ctr(text)
    n = max(len(text), 1)
    return -sum((cnt / n) * np.log2(cnt / n) for cnt in freq.values() if cnt > 0)

def special_char_ratio(text):
    special = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return special / max(len(text), 1)

def regex_hex_score(text):
    """Score based on hex pattern density."""
    hex_matches = re.findall(r'\\x[0-9a-fA-F]{2}', text)
    return len(hex_matches) / max(len(text.split()), 1)

def regex_backslash_score(text):
    """Score based on backslash density."""
    bs = text.count('\\')
    return bs / max(len(text), 1)

all_samples = obf_samples + ben_samples
ent_all = np.array([char_entropy(t) for t in all_samples])
scr_all = np.array([special_char_ratio(t) for t in all_samples])
hex_all = np.array([regex_hex_score(t) for t in all_samples])
bs_all = np.array([regex_backslash_score(t) for t in all_samples])

# ── Full ensemble (sentence delta + paragraph delta + regex + weird unicode) ─
# Loaded from data/ensemble_full_cache.json, built by ensemble_full.py, which
# re-optimizes regex sub-weights and top-level ensemble weights via
# differential evolution against the current dataset (see that script for
# why the old RESEARCH_LOG weights couldn't be reused directly: dataset drift).
_ens_cache = json.load(open("data/ensemble_full_cache.json"))
assert _ens_cache["n_obf"] == n_obf and _ens_cache["n_ben"] == n_ben
ensemble_all = np.array(_ens_cache["ensemble_full"])

# ── Classification ─────────────────────────────────────────────────
def classify_encoding(text):
    t = text.lower()
    if 'decode this hex' in t or '\\x' in text: return 'hex'
    if 'base64' in t: return 'base64'
    if re.search(r'rot\d+', t): return 'rot'
    if text.count('\\') > 5: return 'backslash_visual'
    if re.search(r'[A-Z]\.[A-Z]\.[A-Z]', text): return 'dot_spacing'
    if '<|im_start|>' in text: return 'xml_tag'
    return 'unstructured'

def classify_unstructured(text):
    t = text.lower()
    if re.search(r'translat.*spanish.*english|translat.*to.*spanish|translat.*back', t): return 'translation_chain'
    if re.search(r'(he sido|hackeado|llave secreta|clave)', t): return 'translation_chain'
    if re.search(r'dictionary game|replace.*with|synonym|substitut', t): return 'substitution_game'
    if re.search(r'q to p|replace each letter', t): return 'cipher_substitution'
    if re.search(r'movie title|generate a title|form a title', t): return 'task_hijack'
    if re.search(r'f\(.*==.*\)|decode.*base', t): return 'encoding_puzzle'
    if re.search(r'previous letter|shift|back shift', t): return 'manual_cipher'
    if re.search(r'U\+[0-9A-Fa-f]{4}', text): return 'unicode_codepoint'
    if any(c in text for c in '🗣👍🎉🎯🔥💡⭐🚀✅❌💰🎭🎪🎨🎭🎵🎶🎤🎧🎸🎹🎺🎻🥁'): return 'emoji_encoding'
    if re.search(r'p[ ,]*w[ ,]*n[ ,]*e[ ,]*d', t): return 'pwned_spacing'
    if 'ihbpd' in t or 'ihb' in t or 'ihnbp' in t or 'ihnpe' in t: return 'pwned_abbreviation'
    if re.search(r'pwned means|pwned.*positive|pwned.*spanish|pwned stands for', t): return 'semantic_redef'
    if 'hola' in t or 'si' in t or 'me gusta' in t: return 'language_mixing'
    if re.search(r'reverse.*letter|reverse.*each|reverse.*sentence', t): return 'reverse_scramble'
    if re.search(r'(don.t tell|only respond|only output|do not print|do not use)', t): return 'constraint_inject'
    if re.search(r'gnitae|selppa|Jdfjbkhkg|P ohcl', t): return 'manual_cipher'
    if re.search(r'secret key|secret code|password|\$secret|\{secret', t): return 'extraction_trick'
    if re.search(r'script:|noop|print the sentence', t): return 'code_injection'
    return 'misc'

enc_types = [classify_encoding(t) for t in obf_samples]
full_types = []
for i, et in enumerate(enc_types):
    if et == 'unstructured':
        full_types.append(classify_unstructured(obf_samples[i]))
    else:
        full_types.append(et)

# ── Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 7, "figure.dpi": 150, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})
C_BEN = "#4C72B0"
C_OBF = "#C44E52"
C_WARN = "#DD8452"
PALETTE = ["#C44E52", "#4C72B0", "#55A868", "#DD8452", "#8172B3", "#CCB974",
           "#64B5CD", "#937860", "#E5AE38", "#6D904F", "#8B8B8B", "#A457A3",
           "#D6A756", "#76B7B2", "#FF9DA7", "#BAB0AC", "#5B5EA6", "#9B2335",
           "#DFCFBE", "#55B555", "#9E0142", "#FD420F", "#3B6CB4", "#7FC97F"]

# Major types (>= 3 samples)
type_counts = Counter(full_types)
sorted_types = [t for t, _ in type_counts.most_common()]
major_types = [t for t in sorted_types if type_counts[t] >= 3]


# ═══════════════════════════════════════════════════════════════════
# FIG 1 — Full type distribution
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
counts = [type_counts[t] for t in sorted_types]
bars = ax.barh(sorted_types[::-1], counts[::-1], color=PALETTE[:len(sorted_types)], edgecolor="white", linewidth=1.5)
for bar, v in zip(bars, counts[::-1]):
    ax.text(v + 1, bar.get_y() + bar.get_height()/2, str(v), va="center", fontweight="bold", fontsize=10)
ax.set_xlabel("Count"); ax.set_title("Figure 1 — Obfuscation Type Distribution (N=287)")
ax.grid(axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/01_type_distribution.png", bbox_inches="tight"); plt.close()
print("✓ 01_type_distribution.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 2 — Delta by type (sentence, box + strip)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))
d_obf_sent = d_sent_avg[:n_obf]
d_ben_sent = d_sent_avg[n_obf:]

type_groups = defaultdict(list)
for i, ft in enumerate(full_types):
    type_groups[ft].append(d_obf_sent[i])

sorted_by_mean = sorted(type_groups.keys(), key=lambda x: -np.mean(type_groups[x]))
data = [type_groups[t] for t in sorted_by_mean]

bp = ax.boxplot(data, tick_labels=sorted_by_mean, patch_artist=True, showfliers=False,
                medianprops=dict(color='black', linewidth=1.5), widths=0.6)
for patch, color in zip(bp['boxes'], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.5)
for i, (t, dvals) in enumerate(zip(sorted_by_mean, data)):
    jitter = np.random.normal(0, 0.04, len(dvals))
    ax.scatter([i + 1 + j for j in jitter], dvals, alpha=0.4, s=10, c=PALETTE[i], edgecolors="none")
ax.axhline(np.mean(d_ben_sent), color=C_BEN, ls="--", lw=1.5, label=f"Benign mean ({np.mean(d_ben_sent):.4f})")
ax.set_ylabel("Delta Angle (radians)"); ax.set_title("Figure 2 — Sentence Delta by Type (E5 avg)")
ax.tick_params(axis='x', rotation=40); ax.legend(framealpha=0.9)
fig.tight_layout()
fig.savefig(f"{OUT}/02_delta_by_type.png", bbox_inches="tight"); plt.close()
print("✓ 02_delta_by_type.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 3 — Detection heatmap (type × model) — sentence only
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))
models_list = ["nvidia/nv-embedqa-e5-v5", "nvidia/llama-nemotron-embed-1b-v2", "baai/bge-m3"]
model_names = ["E5", "Nemotron", "BGE-M3"]

heatmap = np.zeros((len(major_types), len(models_list)))
for j, mk in enumerate(models_list):
    d_all = np.array(cache["models"][mk]["deltas"])
    d_ben = d_all[n_obf:]
    threshold = np.percentile(d_ben, 99)
    for i, et in enumerate(major_types):
        indices = [k for k, t in enumerate(full_types) if t == et]
        heatmap[i, j] = np.mean(d_all[indices] > threshold) * 100

im = ax.imshow(heatmap, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(model_names))); ax.set_yticks(range(len(major_types)))
ax.set_xticklabels(model_names); ax.set_yticklabels(major_types)
for i in range(len(major_types)):
    for j in range(len(models_list)):
        ax.text(j, i, f"{heatmap[i, j]:.0f}%", ha="center", va="center", fontsize=10,
                color="white" if heatmap[i, j] < 50 else "black", fontweight="bold")
ax.set_title("Figure 3 — Detection Rate by Type × Model (Sentence, FPR=1%)")
fig.colorbar(im, ax=ax, shrink=0.8, label="Detection Rate (%)")
fig.tight_layout()
fig.savefig(f"{OUT}/03_detection_heatmap_model.png", bbox_inches="tight"); plt.close()
print("✓ 03_detection_heatmap_model.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 4 — Detection heatmap (type × METHOD) — the money graph
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 8))

methods = {
    "Delta (sentence)": d_sent_avg,
    "Delta (paragraph)": para_deltas,
    "Delta (combined)": d_combined,
    "Regex (hex)": hex_all,
    "Regex (backslash)": bs_all,
    "Char entropy": ent_all,
    "Special char %": scr_all,
    "Ensemble": ensemble_all,
}
method_names = list(methods.keys())

heatmap = np.zeros((len(major_types), len(method_names)))
for j, (mname, scores) in enumerate(methods.items()):
    s_ben = scores[n_obf:]
    threshold = np.percentile(s_ben, 99)
    for i, et in enumerate(major_types):
        indices = [k for k, t in enumerate(full_types) if t == et]
        heatmap[i, j] = np.mean(scores[indices] > threshold) * 100

im = ax.imshow(heatmap, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(method_names))); ax.set_yticks(range(len(major_types)))
ax.set_xticklabels(method_names, rotation=35, ha="right"); ax.set_yticklabels(major_types)
for i in range(len(major_types)):
    for j in range(len(method_names)):
        ax.text(j, i, f"{heatmap[i, j]:.0f}%", ha="center", va="center", fontsize=10,
                color="white" if heatmap[i, j] < 50 else "black", fontweight="bold")
ax.set_title("Figure 4 — Detection Rate by Type × Method (FPR=1%)")
fig.colorbar(im, ax=ax, shrink=0.8, label="Detection Rate (%)")
fig.tight_layout()
fig.savefig(f"{OUT}/04_detection_heatmap_method.png", bbox_inches="tight"); plt.close()
print("✓ 04_detection_heatmap_method.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 4b — FNR heatmap (Type × Method), companion to Fig 4 (FPR=1% threshold)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 8))

fnr_heatmap = 100 - heatmap  # heatmap here is still Fig 4's detection-rate matrix

im = ax.imshow(fnr_heatmap, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(method_names))); ax.set_yticks(range(len(major_types)))
ax.set_xticklabels(method_names, rotation=35, ha="right"); ax.set_yticklabels(major_types)
for i in range(len(major_types)):
    for j in range(len(method_names)):
        ax.text(j, i, f"{fnr_heatmap[i, j]:.0f}%", ha="center", va="center", fontsize=10,
                color="white" if fnr_heatmap[i, j] > 50 else "black", fontweight="bold")
ax.set_title("Figure 4b — Miss Rate (FNR) by Type × Method (FPR=1%)")
fig.colorbar(im, ax=ax, shrink=0.8, label="FNR (%)")
fig.tight_layout()
fig.savefig(f"{OUT}/04b_fnr_heatmap_method.png", bbox_inches="tight"); plt.close()
print("✓ 04b_fnr_heatmap_method.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 4c — FPR strip by method (no type axis: benign samples are untyped)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 1.8))

fpr_row = np.zeros((1, len(method_names)))
for j, (mname, scores) in enumerate(methods.items()):
    s_ben = scores[n_obf:]
    threshold = np.percentile(s_ben, 99)
    fpr_row[0, j] = np.mean(s_ben > threshold) * 100

im = ax.imshow(fpr_row, cmap="RdYlGn_r", vmin=0, vmax=20, aspect="auto")
ax.set_xticks(range(len(method_names))); ax.set_yticks([0])
ax.set_xticklabels(method_names, rotation=35, ha="right"); ax.set_yticklabels(["Overall"])
for j in range(len(method_names)):
    ax.text(j, 0, f"{fpr_row[0, j]:.1f}%", ha="center", va="center", fontsize=10,
            color="white" if fpr_row[0, j] > 10 else "black", fontweight="bold")
ax.set_title("Figure 4c — FPR by Method (overall, benign samples are untyped)")
fig.colorbar(im, ax=ax, shrink=0.8, label="FPR (%)")
fig.tight_layout()
fig.savefig(f"{OUT}/04c_fpr_strip_method.png", bbox_inches="tight"); plt.close()
print("✓ 04c_fpr_strip_method.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 5 — Difficulty ranking (combined)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))
s_ben_c = d_combined[n_obf:]
threshold_c = np.percentile(s_ben_c, 99)

type_metrics = []
for et in sorted_types:
    indices = [i for i, t in enumerate(full_types) if t == et]
    vals = d_combined[indices]
    det_rate = np.mean(vals > threshold_c) * 100
    type_metrics.append({"type": et, "count": len(indices), "detection_rate": det_rate})

types_sorted = [m["type"] for m in type_metrics]
det_rates = [m["detection_rate"] for m in type_metrics]
counts_vals = [m["count"] for m in type_metrics]

colors = []
for dr in det_rates:
    if dr >= 80: colors.append("#28a745")
    elif dr >= 50: colors.append("#ffc107")
    elif dr >= 20: colors.append("#fd7e14")
    else: colors.append("#dc3545")

bars = ax.barh(range(len(types_sorted)), det_rates, color=colors, edgecolor="white", linewidth=1.5, height=0.7)
for bar, v, c in zip(bars, det_rates, counts_vals):
    ax.text(v + 1, bar.get_y() + bar.get_height()/2, f"{v:.0f}% (n={c})", va="center", fontweight="bold", fontsize=9)
ax.set_yticks(range(len(types_sorted))); ax.set_yticklabels(types_sorted)
ax.set_xlabel("Detection Rate (%)")
ax.set_title("Figure 5 — Detection Difficulty Ranking (Combined, FPR=1%)")
ax.set_xlim(0, 115); ax.grid(axis="x")
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#28a745", label="Easy (≥80%)"), Patch(facecolor="#ffc107", label="Medium (50-80%)"),
                   Patch(facecolor="#fd7e14", label="Hard (20-50%)"), Patch(facecolor="#dc3545", label="Very Hard (<20%)")]
ax.legend(handles=legend_elements, framealpha=0.9, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/05_difficulty_ranking.png", bbox_inches="tight"); plt.close()
print("✓ 05_difficulty_ranking.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 6 — Sentence vs Paragraph vs Combined (grouped bar)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))

s_ben_s = d_sent_avg[n_obf:]
s_ben_p = para_deltas[n_obf:]
s_ben_c = d_combined[n_obf:]
th_s = np.percentile(s_ben_s, 99)
th_p = np.percentile(s_ben_p, 99)
th_c = np.percentile(s_ben_c, 99)

x = np.arange(len(major_types))
w = 0.25
method_colors = [C_BEN, C_WARN, "#55A868"]
method_labels = ["Sentence", "Paragraph", "Combined"]
thresholds = [th_s, th_p, th_c]
deltas_list = [d_sent_avg[:n_obf], para_deltas[:n_obf], d_combined[:n_obf]]

for j in range(3):
    det_rates_m = []
    for et in major_types:
        indices = [i for i, t in enumerate(full_types) if t == et]
        det = np.mean(deltas_list[j][indices] > thresholds[j]) * 100
        det_rates_m.append(det)
    ax.bar(x + j * w, det_rates_m, w, label=method_labels[j], color=method_colors[j], edgecolor="white", linewidth=1.5)

ax.set_xticks(x + w); ax.set_xticklabels(major_types, rotation=35, ha="right")
ax.set_ylabel("Detection Rate (%)")
ax.set_title("Figure 6 — Sentence vs Paragraph vs Combined (FPR=1%)")
ax.legend(framealpha=0.9); ax.set_ylim(0, 115); ax.grid(axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/06_method_comparison.png", bbox_inches="tight"); plt.close()
print("✓ 06_method_comparison.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 7 — ROC per major type (combined)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 7))
for i, et in enumerate(major_types[:8]):
    indices = [k for k, t in enumerate(full_types) if t == et]
    d_type = d_combined[indices]
    y = np.array([1] * len(d_type) + [0] * n_ben)
    sc = np.concatenate([d_type, s_ben_c])
    fpr, tpr, _ = roc_curve(y, sc)
    auc = roc_auc_score(y, sc)
    ax.plot(fpr, tpr, lw=2, color=PALETTE[i], label=f"{et} (AUC={auc:.3f}, n={len(indices)})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Figure 7 — ROC per Type (Combined Delta)")
ax.legend(loc="lower right", framealpha=0.9, fontsize=8); ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/07_roc_per_type.png", bbox_inches="tight"); plt.close()
print("✓ 07_roc_per_type.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 8 — Cross-model consistency
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Figure 8 — Cross-Model Correlation", fontsize=14, fontweight="bold", y=1.02)
model_pairs = [
    ("nvidia/nv-embedqa-e5-v5", "nvidia/llama-nemotron-embed-1b-v2", "E5", "Nemotron"),
    ("nvidia/nv-embedqa-e5-v5", "baai/bge-m3", "E5", "BGE-M3"),
    ("nvidia/llama-nemotron-embed-1b-v2", "baai/bge-m3", "Nemotron", "BGE-M3"),
]
for idx, (mk1, mk2, n1, n2) in enumerate(model_pairs):
    ax = axes[idx]
    d1 = np.array(cache["models"][mk1]["deltas"])[:n_obf]
    d2 = np.array(cache["models"][mk2]["deltas"])[:n_obf]
    for i, et in enumerate(major_types):
        mask = [t == et for t in full_types]
        ax.scatter(d1[mask], d2[mask], alpha=0.5, s=25, label=et, edgecolors="none", color=PALETTE[i])
    corr = np.corrcoef(d1, d2)[0, 1]
    lim = max(d1.max(), d2.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel(f"{n1} Delta"); ax.set_ylabel(f"{n2} Delta")
    ax.set_title(f"({chr(97+idx)}) {n1} vs {n2} (r={corr:.3f})")
    ax.legend(framealpha=0.8, fontsize=6, markerscale=1.5, ncol=2)
fig.tight_layout()
fig.savefig(f"{OUT}/08_cross_model.png", bbox_inches="tight"); plt.close()
print("✓ 08_cross_model.png")


# ═══════════════════════════════════════════════════════════════════
# FIG 9 — Summary table
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 8))
ax.axis("off")

headers = ["Type", "N", "Sent Δ", "Para Δ", "Comb Δ", "Sent%", "Para%", "Comb%", "Regex%", "Entropy%", "Best"]
rows = []
for et in sorted_types:
    indices = [i for i, t in enumerate(full_types) if t == et]
    n = len(indices)
    if n == 0: continue

    mean_s = np.mean(d_sent_avg[:n_obf][indices]) if et in full_types else 0
    mean_p = np.mean(para_deltas[:n_obf][indices]) if et in full_types else 0
    mean_c = np.mean(d_combined[:n_obf][indices]) if et in full_types else 0

    det_s = np.mean(d_sent_avg[:n_obf][indices] > th_s) * 100
    det_p = np.mean(para_deltas[:n_obf][indices] > th_p) * 100
    det_c = np.mean(d_combined[:n_obf][indices] > th_c) * 100

    # Regex detection
    det_hex = np.mean(hex_all[indices] > np.percentile(hex_all[n_obf:], 99)) * 100
    det_ent = np.mean(ent_all[indices] > np.percentile(ent_all[n_obf:], 99)) * 100
    det_regex = max(det_hex, det_ent)

    best_det = max(det_s, det_p, det_c, det_regex)
    best_method = "combined" if det_c == best_det else "sentence" if det_s == best_det else "regex" if det_regex == best_det else "paragraph"

    rows.append([et, str(n), f"{mean_s:.3f}", f"{mean_p:.3f}", f"{mean_c:.3f}",
                 f"{det_s:.0f}%", f"{det_p:.0f}%", f"{det_c:.0f}%", f"{det_regex:.0f}%", f"{det_ent:.0f}%", best_method])

table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center",
                 colWidths=[0.16, 0.04, 0.07, 0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06, 0.08])
table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.4)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#E8E8E8"); cell.set_text_props(fontweight="bold")
ax.set_title("Figure 9 — Full Obfuscation Analysis Summary", fontsize=13, fontweight="bold", pad=20)
fig.tight_layout()
fig.savefig(f"{OUT}/09_summary_table.png", bbox_inches="tight"); plt.close()
print("✓ 09_summary_table.png")


print(f"\n{'='*60}")
print(f"All 9 figures saved to {OUT}/")
print(f"{'='*60}")
