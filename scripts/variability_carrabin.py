#!/usr/bin/env python3
"""
Intertrial variability figure -- carrabin task.

On first run (SAMPLE_PIDS = None), prints a pid/std table so you can choose
representative participants, then exits. Set SAMPLE_PIDS at the top of this
file and rerun to generate the figure.

Usage:
    python scripts/variability_carrabin.py
"""

import sys
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from itertools import combinations
from pathlib import Path
from scipy.stats import gaussian_kde
from statannotations.Annotator import Annotator

# -- path setup ----------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import data_path, FIGURES_DIR
from utils.plot_style import apply_style

# -- configuration (edit here) -------------------------------------------------
RUN_FOLDER = "wasserstein"

# Set to None on first run to print pid/std table, then fill in and rerun.
SAMPLE_PIDS = {"narrow": 20, "medium": 18, "broad": 4}

MODEL_ORDER = ["Bayes", "RL", "NoisyCounting"]
LINESTYLES = ["solid", "dashed", "dotted"]  # narrow / medium / broad

# -- style ---------------------------------------------------------------------
apply_style()
palette = sns.color_palette("colorblind")
PALETTE = {
    "Human": "0.3",  # neutral grey for human data
    "Bayes": palette[0],
    "RL": palette[1],
    "NoisyCounting": palette[2],
}

# -- load data -----------------------------------------------------------------
run_dir = data_path("runs") / RUN_FOLDER

human = pd.read_pickle(data_path("carrabin.pkl"))

models = {}
for mt in MODEL_ORDER:
    f = run_dir / f"{mt}_carrabin_responses.pkl"
    assert f.exists(), f"Missing: {f}"
    models[mt] = pd.read_pickle(f)

# Wasserstein performance files (for violin panel)
perf_dfs = []
for mt in MODEL_ORDER:
    f = run_dir / f"{mt}_carrabin_performance.pkl"
    if f.exists():
        perf_dfs.append(pd.read_pickle(f))
    else:
        print(f"Warning: missing {f.name}")
assert perf_dfs, f"No performance files found in {run_dir}"
perf = pd.concat(perf_dfs, ignore_index=True)

# -- per-participant std -------------------------------------------------------
pid_std = (
    human.groupby("pid")["response"]
    .std()
    .rename("response_std")
    .reset_index()
    .sort_values("response_std")
    .reset_index(drop=True)
)

if SAMPLE_PIDS is None:
    print("pid / response_std table (sorted narrow -> broad):")
    print(pid_std.to_string(index=False))
    print("\nSet SAMPLE_PIDS at the top of this script and rerun.")
    sys.exit(0)

sample_labels = ["narrow", "medium", "broad"]
sample_pids = [SAMPLE_PIDS[l] for l in sample_labels]

# -- figure layout -------------------------------------------------------------
fig = plt.figure(constrained_layout=True)
gs = gridspec.GridSpec(
    2,
    4,
    figure=fig,
    height_ratios=[1, 1.2],
)

# Row 1: 4 KDE panels
ax_kde = []
for i in range(4):
    sharey = ax_kde[0] if i > 0 else None
    ax_kde.append(fig.add_subplot(gs[0, i], sharey=sharey))

# Row 2: 2 panels each spanning 2 columns
ax_std = fig.add_subplot(gs[1, :2])
ax_viol = fig.add_subplot(gs[1, 2:])

# -- row 1: KDE panels ---------------------------------------------------------
sources = [("Human", human)] + [(mt, models[mt]) for mt in MODEL_ORDER]

# Compute shared x/y limits across all panels and all pids
all_responses = pd.concat(
    [src[["response"]] for _, src in sources], ignore_index=True
)
x_min, x_max = all_responses["response"].min(), all_responses["response"].max()
x_pad = (x_max - x_min) * 0.05
x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 400)

max_density = 0.0
kde_cache = {}  # (label, pid) -> density on x_grid

for label, src in sources:
    for pid in sample_pids:
        vals = src[src["pid"] == pid]["response"].values
        kde = gaussian_kde(vals)
        density = kde(x_grid)
        kde_cache[(label, pid)] = density
        max_density = max(max_density, density.max())

y_max = max_density * 1.1

for ax, (label, _) in zip(ax_kde, sources):
    color = PALETTE[label]
    for pid, ls in zip(sample_pids, LINESTYLES):
        density = kde_cache[(label, pid)]
        ax.fill_between(x_grid, density, alpha=0.3, color=color)
        ax.plot(x_grid, density, color=color, linestyle=ls, linewidth=1.5)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(0, y_max)
    ax.set_title(label)
    ax.set_xlabel("Response")
    ax.set_ylabel("Density" if label == "Human" else "")
    if label != "Human":
        plt.setp(ax.get_yticklabels(), visible=False)
    sns.despine(ax=ax, top=True, right=True)

# Direct curve labels on Human panel
ax_human = ax_kde[0]
for pid, ls, lbl in zip(sample_pids, LINESTYLES, sample_labels):
    density = kde_cache[("Human", pid)]
    # Place label at the x position of peak density
    peak_idx = np.argmax(density)
    ax_human.text(
        x_grid[peak_idx],
        density[peak_idx] + y_max * 0.02,
        lbl,
        ha="center",
        va="bottom",
        fontsize=7,
        color=PALETTE["Human"],
    )

# -- row 2a: population std distribution ---------------------------------------
std_vals = pid_std["response_std"].values
kde_pop = gaussian_kde(std_vals)
x_std = np.linspace(std_vals.min() * 0.9, std_vals.max() * 1.1, 400)
ax_std.fill_between(x_std, kde_pop(x_std), alpha=0.3, color="0.5")
ax_std.plot(x_std, kde_pop(x_std), color="0.3", linewidth=1.5)

# Mark sample participants
for pid, ls in zip(sample_pids, LINESTYLES):
    std_val = pid_std.loc[pid_std["pid"] == pid, "response_std"].values[0]
    ax_std.axvline(std_val, color="0.3", linestyle=ls, linewidth=1.5)

ax_std.set_xlabel("Response std")
ax_std.set_ylabel("Density")
ax_std.set_title("Population variability")
sns.despine(ax=ax_std, top=True, right=True)

# -- row 2b: Wasserstein violin plots ------------------------------------------
sns.violinplot(
    data=perf,
    x="model_type",
    y="cv_loss_mean",
    order=MODEL_ORDER,
    hue="model_type",
    palette=PALETTE,
    inner="point",
    legend=False,
    cut=0,
    ax=ax_viol,
)
np.random.seed(42)
sns.stripplot(
    data=perf,
    x="model_type",
    y="cv_loss_mean",
    order=MODEL_ORDER,
    color="0.2",
    alpha=0.5,
    jitter=0.2,
    size=4,
    ax=ax_viol,
)
ax_viol.set_title("Wasserstein loss")
ax_viol.set_ylabel("Wasserstein")
ax_viol.set_xlabel("")
sns.despine(ax=ax_viol, top=True, right=True)

pairs = list(combinations(MODEL_ORDER, 2))
annotator = Annotator(
    ax_viol,
    pairs,
    data=perf,
    x="model_type",
    y="cv_loss_mean",
    order=MODEL_ORDER,
)
annotator.configure(test="Wilcoxon", text_format="star", loc="inside")
with open(os.devnull, "w") as devnull:
    old_stdout = sys.stdout
    sys.stdout = devnull
    annotator.apply_and_annotate()
    sys.stdout = old_stdout

# -- save ----------------------------------------------------------------------
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "variability_carrabin.png", dpi=300)
plt.savefig(FIGURES_DIR / "variability_carrabin.pdf")
print(f"Saved figures/variability_carrabin.{{png,pdf}}")
