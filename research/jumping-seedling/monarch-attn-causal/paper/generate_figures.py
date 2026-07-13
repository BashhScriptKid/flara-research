"""Generates all figures for the TauMonarchAttention paper from the
established results already measured throughout the investigation
(JOURNAL.md) -- no new experiments, just plotting recorded numbers.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT = "figures"

# ---------------------------------------------------------------------------
# Fig 1: same-norm needle-retrieval comparison, GT vs TauMonarch vs Sliding vs CausalMonarch
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.2))
labels = ["Dense\n(GT)", "TauMonarch\n(Meta)", "Sliding\n(far-block)", "CausalMonarch\n(diagonal)"]
means = [0.883, 0.930, 0.20, 0.229]
errs = [0.009, 0.006, 0.03, 0.048]
colors = ["#4C72B0", "#55A868", "#C44E52", "#C44E52"]
bars = ax.bar(labels, means, yerr=errs, capsize=4, color=colors, width=0.6)
ax.set_ylabel("mean cosine similarity to true value")
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(3.6, 0.52, "retrieval-quality floor", fontsize=7, color="gray", ha="right")
ax.set_title("Same-norm-controlled needle retrieval, N=30+ trials/mechanism")
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_samenorm_comparison.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 2: Sliding far-region-length sweep -- cross-block dilution
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.2))
n_far_blocks = [2, 4, 8, 16, 24, 40, 52]
mean_cos = [0.4307, 0.4126, 0.3702, 0.2958, 0.2661, 0.2242, 0.1872]
ax.plot(n_far_blocks, mean_cos, marker="o", color="#C44E52")
ax.set_xlabel("number of competing far-block representatives")
ax.set_ylabel("mean cosine similarity")
ax.set_title("SlidingMonarchAttention: cross-block dilution\n(T=3, needle block fixed)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_sliding_dilution_sweep.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 3: Sliding T-iteration flat-line -- structural ceiling, not budget problem
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.2))
t_vals = [3, 5, 10, 20]
mean_cos_t = [0.2661, 0.2661, 0.2661, 0.2661]
ax.plot(t_vals, mean_cos_t, marker="s", color="#C44E52", markersize=8)
ax.set_ylim(0, 1.0)
ax.axhline(0.883, color="#4C72B0", linestyle="--", linewidth=1, label="Dense (GT) reference")
ax.set_xlabel("T-iteration refinement rounds")
ax.set_ylabel("mean cosine similarity")
ax.set_title("SlidingMonarchAttention: identical to 4 decimal places\nregardless of refinement budget")
ax.legend(loc="center right", fontsize=8)
ax.set_xticks(t_vals)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_sliding_t_sweep.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 4: TauMonarch FLOP-stage accounting vs N
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.2))
Ns = [256, 512, 1024, 2048, 4096]
qk_ratio = [1.000, 1.000, 1.000, 1.000, 1.000]
av_ratio = [0.178, 0.141, 0.122, 0.111, 0.106]
blended = [0.589, 0.571, 0.561, 0.556, 0.553]
ax.plot(Ns, qk_ratio, marker="o", label="QK scoring stage (1.0x, zero savings)", color="#C44E52")
ax.plot(Ns, av_ratio, marker="s", label="AV/softmax stage", color="#55A868")
ax.plot(Ns, blended, marker="^", label="Blended (naive, pre-fix)", color="#4C72B0")
ax.set_xscale("log", base=2)
ax.set_xlabel("sequence length N")
ax.set_ylabel("FLOP ratio vs. dense attention")
ax.set_title("TauMonarchAttention: stage-separated FLOP accounting")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_flop_stages.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 5: Real hardware wall-clock, N=8192
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.2))
labels5 = ["Dense", "Sliding", "TauMonarch\n(sort-based tau)", "TauMonarch\n(quickselect)"]
times = [10.9, 1.42, 16.5, 8.9]
colors5 = ["#4C72B0", "#C44E52", "#DD8452", "#55A868"]
bars = ax.bar(labels5, times, color=colors5, width=0.6)
ax.set_ylabel("wall-clock time (s), N=8192")
ax.set_title("Real hardware measurement, AMD Ryzen 5 5500U\n(criterion + perf stat)")
for b, t in zip(bars, times):
    ax.text(b.get_x() + b.get_width() / 2, t + 0.3, f"{t}s", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_hardware_wallclock.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 6: trained-landmark kill criterion -- fixed-quota vs real-threshold
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.2))
quota_frac = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
quota_cos = [0.5442, 0.6483, 0.7857, 0.8123, 0.8761, 0.9174, 0.9288, 0.9297, 0.9340]
thresh_frac = [0.4544, 0.5085, 0.5625, 0.6173, 0.6656, 0.7646, 0.8871, 0.9629]
thresh_cos = [0.6184, 0.6529, 0.6878, 0.7494, 0.7777, 0.8194, 0.8863, 0.9188]
ax.plot(quota_frac, quota_cos, marker="o", label="Fixed quota (top-k, artifact)", color="#DD8452")
ax.plot(thresh_frac, thresh_cos, marker="s", label="Real threshold (genuine)", color="#C44E52")
ax.axhline(0.7437, color="gray", linestyle="--", linewidth=0.8, label="kill-criterion target (80% of Meta)")
ax.axvline(0.50, color="gray", linestyle=":", linewidth=0.8, label="kill-criterion cap (50% admission)")
ax.set_xlabel("fraction of blocks admitted to real scoring")
ax.set_ylabel("mean cosine similarity")
ax.set_title("Trained-landmark selection: quota artifact vs. genuine threshold")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/fig6_landmark_kill_criterion.png")
plt.close(fig)

print("All figures generated in", OUT)
