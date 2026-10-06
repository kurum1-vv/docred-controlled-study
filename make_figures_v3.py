import os
import sys
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

SEEDS = [42, 123, 456, 789, 2024]

DATA = {
    "backbone": [66.40, 66.34, 66.25, 66.42, 66.70],
    "+MS-ECC": [66.23, 66.52, 66.38, 66.22, 66.46],
    "+GR-GR": [66.01, 65.93, 66.27, 66.43, 66.16],
    "+CB-FL": [66.30, 66.24, 66.82, 66.20, 66.78],
    "+MS-ECC+GR-GR": [65.94, 65.96, 66.13, 66.18, 66.24],
    "+MS-ECC+CB-FL": [65.86, 66.74, 66.10, 66.51, 66.12],
    "+GR-GR+CB-FL": [66.45, 66.23, 66.34, 66.23, 66.03],
    "full (all three)": [65.72, 65.79, 66.09, 66.23, 65.71],
}

labels = list(DATA.keys())
means = [statistics.mean(DATA[k]) for k in labels]
stds = [statistics.stdev(DATA[k]) for k in labels]
base_mean = means[0]
deltas = [m - base_mean for m in means]

colors = ["#4c72b0"] + ["#dd8452"] * 3 + ["#55a868"] * 3 + ["#c44e52"]

fig, ax = plt.subplots(figsize=(10, 4.2))
x = range(len(labels))
ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
ax.axhline(base_mean, color="gray", linestyle="--", linewidth=1, label=f"backbone = {base_mean:.2f}")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("DocRED dev F1 (%)")
ax.set_ylim(min(means) - 1.0, max(means) + 1.0)
ax.set_title("Figure 2. DocRED dev F1 by configuration (5-seed mean ± std)")
for i, m in enumerate(means):
    ax.text(i, m + 0.12, f"{m:.2f}", ha="center", fontsize=8)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "figure2_experiment_results.png"), dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.bar(x, deltas, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("ΔF1 vs. backbone (5-seed mean)")
ax.set_title("Figure 3. Ablation: ΔF1 versus the reproduced backbone")
for i, d in enumerate(deltas):
    ax.text(i, d + (0.02 if d >= 0 else -0.08), f"{d:+.2f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "figure3_ablation_study.png"), dpi=200)
plt.close(fig)

print("wrote figure2_experiment_results.png and figure3_ablation_study.png to", OUT)
