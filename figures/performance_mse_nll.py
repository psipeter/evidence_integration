#!/usr/bin/env python3
"""
Model performance under MSE/NLL loss.
Distribution of cross-validated loss across participants for each model and task.
One panel per task, 1 row x 3 columns.

Usage:
    python figures/performance_mse_nll.py [run_folder]

Default run_folder: MSE
Edit configuration variables at the top of this file.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations
from statannotations.Annotator import Annotator

# -- path setup ----------------------------------------------------------------
PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from utils.paths import data_path
from utils.plot_style import apply_style

# -- configuration (edit here) -------------------------------------------------
RUN_FOLDER = sys.argv[1] if len(sys.argv) > 1 else "MSE"

MODEL_ORDER = {
    "carrabin": ["Bayes", "RL", "NoisyCounting"],
    "jiang": ["Bayes", "RL", "DeGroot"],
    "yoo": ["Mean", "RL", "ADM"],
}
YLABELS = {
    "carrabin": "MSE",
    "jiang": "NLL",
    "yoo": "MSE",
}
TITLES = {
    "carrabin": "Ratio Estimation",
    "jiang": "Social Learning",
    "yoo": "Value Comparison",
}

# -- style ---------------------------------------------------------------------
apply_style()
palette = sns.color_palette("colorblind")
PALETTE = {
    "Bayes": palette[0],
    "Mean": palette[0],
    "RL": palette[1],
    "NoisyCounting": palette[2],
    "DeGroot": palette[2],
    "ADM": palette[2],
}

# -- load data -----------------------------------------------------------------
run_dir = data_path("runs") / RUN_FOLDER

dfs = []
for dataset, models in MODEL_ORDER.items():
    for model_type in models:
        f = run_dir / f"{model_type}_{dataset}_performance.pkl"
        if f.exists():
            dfs.append(pd.read_pickle(f))
        else:
            print(f"Missing: {f.name}")

assert dfs, f"No performance files found in {run_dir}"
perf = pd.concat(dfs, ignore_index=True)

# -- plot ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, constrained_layout=True)

for ax, dataset in zip(axes, ["carrabin", "jiang", "yoo"]):
    subset = perf[perf["dataset"] == dataset]
    order = MODEL_ORDER[dataset]

    if subset.empty:
        ax.set_title(TITLES[dataset])
        ax.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        sns.despine(ax=ax, top=True, right=True)
        continue

    sns.violinplot(
        data=subset,
        x="model_type",
        y="cv_loss_mean",
        order=order,
        hue="model_type",
        palette=PALETTE,
        inner="point",
        legend=False,
        cut=0,
        ax=ax,
    )
    np.random.seed(42)
    sns.stripplot(
        data=subset,
        x="model_type",
        y="cv_loss_mean",
        order=order,
        color="0.2",
        alpha=0.5,
        jitter=0.2,
        size=4,
        ax=ax,
    )
    ax.set_title(TITLES[dataset])
    ax.set_ylabel(YLABELS[dataset])
    ax.set_xlabel("")
    sns.despine(ax=ax, top=True, right=True)

    pairs = list(combinations(order, 2))
    annotator = Annotator(
        ax, pairs, data=subset, x="model_type", y="cv_loss_mean", order=order
    )
    annotator.configure(test="Wilcoxon", text_format="star", loc="inside")
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        annotator.apply_and_annotate()
        sys.stdout = old_stdout

fig_dir = PROJ / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "performance_mse_nll.png", dpi=300)
plt.savefig(fig_dir / "performance_mse_nll.pdf")
print(f"Saved figures/performance_mse_nll.{{png,pdf}}")
