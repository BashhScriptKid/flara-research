"""Second batch of figures, adding visual coverage for sweeps/comparisons
currently only described in prose in the paper.
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
# Fig 7: Bucket routing natural vs adversarial gap
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 3.2))
labels = ["Natural case\n(random, n=1000)", "Adversarial\n(Voronoi-boundary decoy)"]
success = [96.4, 17.0]  # 96.4% recall, 100-83% fail = 17% success
colors = ["#55A868", "#C44E52"]
bars = ax.bar(labels, success, color=colors, width=0.55)
ax.set_ylabel("recall / success rate (%)")
ax.set_ylim(0, 105)
for b, v in zip(bars, success):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}%", ha="center", fontsize=9)
ax.set_title("Bucket routing: natural case masks a real adversarial defect")
fig.tight_layout()
fig.savefig(f"{OUT}/fig7_bucket_routing_gap.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 8: Five untrained R-landmark mechanics, same-norm control result.
# Categorical pass/fail grid (not fabricated continuous values) --
# uncorrected outcomes are qualitative (pass/fail) in the source record,
# and same-norm-corrected results are uniformly poor (mean cos ~0.00-0.18
# across all five, per JOURNAL.md) but not individually distinguished
# per mechanic in the record, so a shared "fail" cell is the honest
# representation rather than invented per-mechanic precision.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.0))
mechs = ["Random reuse", "Top-magnitude", "k-means", "FPS", "Max-pool"]
uncorrected = ["FAIL", "passes (R=2)", "FAIL", "passes (R=2)", "FAIL"]
corrected = ["FAIL", "FAIL", "FAIL", "FAIL", "FAIL"]

cell_color = {"FAIL": "#C44E52", "passes (R=2)": "#DD8452"}
for row_i, (mech, unc, corr) in enumerate(zip(mechs, uncorrected, corrected)):
    y = len(mechs) - row_i
    ax.barh(y, 1, left=0, height=0.7, color=cell_color[unc], edgecolor="white")
    ax.barh(y, 1, left=1.1, height=0.7, color=cell_color[corr], edgecolor="white")
    ax.text(0.5, y, unc, ha="center", va="center", fontsize=8, color="white")
    ax.text(1.6, y, corr, ha="center", va="center", fontsize=8, color="white")
    ax.text(-0.15, y, mech, ha="right", va="center", fontsize=9)
ax.text(0.5, len(mechs) + 0.9, "Uncorrected\n(large-magnitude needle)", ha="center", fontsize=8.5, fontweight="bold")
ax.text(1.6, len(mechs) + 0.9, "Same-norm-\ncontrolled", ha="center", fontsize=8.5, fontweight="bold")
ax.set_xlim(-1.3, 2.2)
ax.set_ylim(0.3, len(mechs) + 1.7)
ax.axis("off")
ax.set_title("Five untrained representative-construction heuristics:\nthe same-norm control reverses both apparent \"passes\"", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig8_five_landmark_mechanics.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 9: LSH exclusion cliff vs perturbation
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.2))
perturb = [0.0, 0.1, 0.2, 0.3, 0.5]
survival = [1.00, 0.7333, 0.40, 0.2667, 0.20]
mean_cos_lsh = [0.9844, 0.7332, 0.4102, 0.2575, 0.2094]
ax.plot(perturb, survival, marker="o", label="needle survival rate", color="#C44E52")
ax.plot(perturb, mean_cos_lsh, marker="s", label="mean cosine similarity", color="#4C72B0")
ax.set_xlabel("needle direction perturbation off exact query alignment")
ax.set_ylabel("rate / similarity")
ax.set_title("LSH-hashing prefilter: exclusion cliff under\nrealistic (imperfect) query-needle alignment")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig9_lsh_exclusion_cliff.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 10: Roofline arithmetic intensity vs ridge point
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.2))
Ns = [512, 1024, 2048, 4096, 8192]
dense_ai = [199.5, 211.0, 217.3, 220.6, 222.3]
thresh_ai = [112.6, 119.1, 124.5, 128.4, 130.9]
ax.plot(Ns, dense_ai, marker="o", label="Dense attention", color="#4C72B0")
ax.plot(Ns, thresh_ai, marker="s", label="TauMonarchAttention", color="#55A868")
ax.axhspan(4.62, 7.88, color="gray", alpha=0.25, label="ridge point range")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("sequence length N")
ax.set_ylabel("arithmetic intensity (FLOPs/byte)")
ax.set_title("Roofline placement, real production GQA config\n(both mechanisms comfortably compute-bound)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig10_roofline_ridge_point.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 11: Residual-centroid FLOP correction story.
# Real reported RANGES (not fabricated single points): naive AV-only
# estimate ~0.55x; with the omitted residual-computation cost included,
# 1.05-1.23x; exact-algebraic fix, 0.55-0.65x. Shown as min/max bars via
# asymmetric error bars anchored at each range's midpoint, so the plot
# does not imply a precision the underlying measurement doesn't have.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.2))
stages = ["Naive AV-only\nestimate", "+ omitted\nresidual cost", "Exact-algebraic\nfix (final)"]
lo = [0.55, 1.05, 0.55]
hi = [0.55, 1.23, 0.65]
mid = [(a + b) / 2 for a, b in zip(lo, hi)]
err_lo = [m - a for m, a in zip(mid, lo)]
err_hi = [b - m for b, m in zip(hi, mid)]
colors11 = ["#DD8452", "#C44E52", "#55A868"]
x = np.arange(len(stages))
bars = ax.bar(x, mid, yerr=[err_lo, err_hi], capsize=5, color=colors11, width=0.55)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="parity with dense (1.0x)")
ax.set_xticks(x)
ax.set_xticklabels(stages)
ax.set_ylabel("blended FLOP ratio vs. dense attention")
for xi, a, b in zip(x, lo, hi):
    label = f"{a}x" if a == b else f"{a}–{b}x"
    ax.text(xi, b + 0.05, label, ha="center", fontsize=8.5)
ax.legend(fontsize=8)
ax.set_title("The residual-centroid cost correction:\nfrom a false ~1.8x win to a real ~1.5–1.7x ceiling")
fig.tight_layout()
fig.savefig(f"{OUT}/fig11_residual_correction_story.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 12: Trained-landmark selection margin vs M (order-statistic pattern)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.2))
Ms = [8, 16, 32, 64, 128]
margin_mean = [-0.0522, -0.7012, -1.1795, -1.5948, -2.0027]
sqrt_2lnM = [-np.sqrt(2 * np.log(m)) for m in Ms]
# scale the theoretical curve to overlay for shape comparison
scale = margin_mean[-1] / sqrt_2lnM[-1]
theory = [s * scale for s in sqrt_2lnM]
ax.plot(Ms, margin_mean, marker="o", label="measured mean margin", color="#C44E52")
ax.plot(Ms, theory, linestyle="--", label=r"$-\sqrt{2\ln M}$ (scaled, order-statistic prediction)", color="#4C72B0")
ax.set_xscale("log", base=2)
ax.set_xlabel("number of candidate blocks M")
ax.set_ylabel("mean selection margin\n(needle score − max competitor score)")
ax.axhline(0, color="gray", linewidth=0.6)
ax.legend(fontsize=7.5)
ax.set_title("Trained-landmark selection margin: order-statistic\ndegradation, independent of training objective")
fig.tight_layout()
fig.savefig(f"{OUT}/fig12_landmark_margin_sweep.png")
plt.close(fig)

print("Second batch of figures generated in", OUT)
