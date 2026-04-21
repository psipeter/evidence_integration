#!/usr/bin/env python3
"""
Model performance under default ``response`` loss (MSE or jiang NLL).
Distribution of cross-validated loss across participants for each model and task.
One panel per task, 1 row x 3 columns.

Usage:
    python scripts/model_performance.py [run_folder]

Default run_folder: MSE
Edit configuration variables at the top of this file.
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -- path setup ----------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import data_path, FIGURES_DIR
from utils.plot_style import annotate_violins, apply_style, get_palette, FIGURE_SIZE

# -- configuration (edit here) -------------------------------------------------
parser = argparse.ArgumentParser(description="Plot model performance across tasks.")
parser.add_argument("run_folder", nargs="?", default="MSE")
parser.add_argument(
    "--nef_type",
    default="NEF_recurrent",
    choices=("NEF_recurrent", "NEF_synaptic"),
)
args = parser.parse_args()

RUN_FOLDER = args.run_folder
nef_type = args.nef_type

MODEL_ORDER = {
    "carrabin": ["Bayes", "RL", "NoisyCounting", "NEF"],
    "jiang": ["Bayes", "RL", "DeGroot"],
    "yoo": ["Mean", "RL", "ADM", "NEF"],
}
YLABELS = {
    "carrabin": "Mean Squared Error",
    "jiang":    "Negative Log-Likelihood",
    "yoo":      "Mean Squared Error",
}
TITLES = {
    "carrabin": "Ratio Estimation",
    "jiang": "Social Learning",
    "yoo": "Value Comparison",
}

# -- style ---------------------------------------------------------------------
apply_style()
PALETTE = get_palette()

# -- load data -----------------------------------------------------------------
run_dir = data_path("runs") / RUN_FOLDER

dfs = []
for dataset, models in MODEL_ORDER.items():
    for model_type in models:
        load_model_type = nef_type if model_type == "NEF" else model_type
        f = run_dir / f"{load_model_type}_{dataset}_performance.pkl"
        if f.exists():
            dfs.append(pd.read_pickle(f))
        else:
            print(f"Missing: {f.name}")

assert dfs, f"No performance files found in {run_dir}"
perf = pd.concat(dfs, ignore_index=True)
perf["model_type"] = perf["model_type"].replace(
    {
        "NEF_recurrent": "NEF",
        "NEF_synaptic": "NEF",
    }
)

# -- plot ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE, constrained_layout=True)

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
        inner=None,
        legend=False,
        cut=0,
        ax=ax,
    )
    ax.set_ylabel(YLABELS[dataset])
    ax.set_xlabel("")
    sns.despine(ax=ax, top=True, right=True)

    annotate_violins(ax, subset, "model_type", "cv_loss_mean", order)
    ax.set_title(TITLES[dataset])

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "model_performance.png", dpi=300)
plt.savefig(FIGURES_DIR / "model_performance.pdf")
print("Saved figures/model_performance.{png,pdf}")
