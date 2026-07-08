#!/usr/bin/env python3
"""
Figure generation for "A Servant And A Guard: Why A Model Can't Be Both".
Reads the final results + bootstrap CI data and produces the PDF's figures.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHS = os.path.join(BASE_DIR, "graphs")
os.makedirs(GRAPHS, exist_ok=True)

with open("/tmp/dg_results/results_final.json") as f:
    RESULTS = json.load(f)
with open("/tmp/dg_results/bootstrap_ci.json") as f:
    CI = json.load(f)

CONDITIONS = ["monolithic", "decoupled", "cot_ablation", "directed_cot",
              "safeguard", "safeguard_ours", "third_party", "peer_consensus"]
LABELS = ["monolithic", "decoupled", "cot_ablation", "directed_cot",
          "safeguard", "safeguard_ours", "third_party", "peer_consensus"]

COLORS = {
    "monolithic":     "#888888",
    "decoupled":      "#1a1a2e",
    "cot_ablation":   "#c0392b",
    "directed_cot":   "#e67e22",
    "safeguard":      "#7f8c8d",
    "safeguard_ours": "#2980b9",
    "third_party":    "#27ae60",
    "peer_consensus": "#16a085",
}

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.6,
    "figure.facecolor": "white",
})

# ── Figure 1: Detection / FPR / F1 with bootstrap 95% CIs ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
metrics = [("det", "Detection Rate", "%"), ("fpr", "False Positive Rate", "%"), ("f1", "F1", "")]

for ax, (key, title, unit) in zip(axes, metrics):
    vals, los, his = [], [], []
    for c in CONDITIONS:
        point, lo, hi = CI[c][key]
        scale = 100 if unit == "%" else 1
        vals.append(point * scale)
        los.append((point - lo) * scale)
        his.append((hi - point) * scale)
    x = np.arange(len(CONDITIONS))
    bars = ax.bar(x, vals, color=[COLORS[c] for c in CONDITIONS], width=0.62)
    ax.errorbar(x, vals, yerr=[los, his], fmt="none", ecolor="black", elinewidth=0.8, capsize=2.5)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=55, ha="right", fontsize=7.5)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.set_ylabel(unit if unit else "score")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Detection / FPR / F1 across all eight conditions, with bootstrap 95% CIs (N=5,000 resamples)",
             fontsize=10, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(os.path.join(GRAPHS, "01_metrics_with_ci.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# ── Figure 2: Self-referential framing trade-off (decoupled -> third_party -> peer_consensus) ──
fig, ax = plt.subplots(figsize=(5.2, 4.6))
framing_conditions = ["decoupled", "third_party", "peer_consensus"]
det_vals = [CI[c]["det"][0] * 100 for c in framing_conditions]
fpr_vals = [CI[c]["fpr"][0] * 100 for c in framing_conditions]
det_err = [[(CI[c]["det"][0] - CI[c]["det"][1]) * 100 for c in framing_conditions],
           [(CI[c]["det"][2] - CI[c]["det"][0]) * 100 for c in framing_conditions]]
fpr_err = [[(CI[c]["fpr"][0] - CI[c]["fpr"][1]) * 100 for c in framing_conditions],
           [(CI[c]["fpr"][2] - CI[c]["fpr"][0]) * 100 for c in framing_conditions]]

ax.errorbar(fpr_vals, det_vals, xerr=fpr_err, yerr=det_err, fmt="none", ecolor="#999999", elinewidth=0.8, capsize=3, zorder=1)
for i, c in enumerate(framing_conditions):
    ax.scatter(fpr_vals[i], det_vals[i], s=110, color=COLORS[c], zorder=3, edgecolor="white", linewidth=1.2)
    ax.annotate(c, (fpr_vals[i], det_vals[i]), textcoords="offset points", xytext=(8, 6), fontsize=8.5)

ax.plot(fpr_vals, det_vals, color="#aaaaaa", linewidth=1.0, linestyle="--", zorder=0)
ax.set_xlabel("False Positive Rate (%)")
ax.set_ylabel("Detection Rate (%)")
ax.set_title("Self-distancing framing axis:\nmonotonic sensitivity gain, FPR cost scales with distance",
             fontsize=9.5, fontweight="bold")
ax.grid(alpha=0.25, linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(GRAPHS, "02_framing_tradeoff.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# ── Figure 3: monolithic vs decoupled-family overview (F1 with quality overlay) ──
fig, ax1 = plt.subplots(figsize=(8.5, 4))
x = np.arange(len(CONDITIONS))
f1_vals = [CI[c]["f1"][0] for c in CONDITIONS]
f1_lo = [CI[c]["f1"][0] - CI[c]["f1"][1] for c in CONDITIONS]
f1_hi = [CI[c]["f1"][2] - CI[c]["f1"][0] for c in CONDITIONS]

bars = ax1.bar(x, f1_vals, color=[COLORS[c] for c in CONDITIONS], width=0.6, zorder=2)
ax1.errorbar(x, f1_vals, yerr=[f1_lo, f1_hi], fmt="none", ecolor="black", elinewidth=0.8, capsize=3, zorder=3)
ax1.set_xticks(x)
ax1.set_xticklabels(LABELS, rotation=40, ha="right", fontsize=8)
ax1.set_ylabel("F1 (with 95% bootstrap CI)")
ax1.set_ylim(0, 1.0)
ax1.grid(axis="y", alpha=0.25, linewidth=0.5)
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_title("F1 across all conditions — overlapping CIs flag comparisons needing paired tests (§5.4)",
              fontsize=9.5, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(GRAPHS, "03_f1_overview.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

print("Figures written to", GRAPHS)
for f in sorted(os.listdir(GRAPHS)):
    if f.endswith(".png"):
        print(" -", f)
