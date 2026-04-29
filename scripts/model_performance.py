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

# -- CLI -----------------------------------------------------------------------
_parser = argparse.ArgumentParser()
_parser.add_argument("--run_folder", type=str, default="joint")
_parser.add_argument("--include_rl_lambda", action="store_true", default=False)
_args, _ = _parser.parse_known_args()
RUN_FOLDER = _args.run_folder

# -- configuration (edit here) -------------------------------------------------
MODEL_ORDER = {
    "carrabin": ["Bayes", "RL", "NoisyCounting", "NEF_recurrent"],
    "jiang": ["Bayes", "RL", "DeGroot", "NEF_recurrent"],
    "yoo": ["Mean", "RL", "ADM", "NEF_recurrent"],
}
if _args.include_rl_lambda:
    MODEL_ORDER["carrabin"].append("RL_lambda")
    MODEL_ORDER["jiang"].extend(["RL_lambda", "RL_lambda_rd"])
    MODEL_ORDER["yoo"].append("RL_lambda")
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


def _get_loss(perf_df: pd.DataFrame) -> pd.Series:
    """Return response_component if available, else cv_loss_mean."""
    if "response_component" in perf_df.columns:
        rc = perf_df["response_component"]
        if rc.notna().all():
            return rc
    return perf_df["cv_loss_mean"]


dfs = []
warned_missing: set[str] = set()
for dataset, models in MODEL_ORDER.items():
    for model_type in models:
        f = run_dir / f"{model_type}_{dataset}_performance.pkl"
        if f.exists():
            perf_df = pd.read_pickle(f)
            perf_df["plot_loss"] = _get_loss(perf_df)
            perf_df["uses_response_component"] = (
                "response_component" in perf_df.columns
                and perf_df["response_component"].notna().all()
            )
            dfs.append(perf_df)
        else:
            key = f"{model_type}_{dataset}"
            if key not in warned_missing:
                print(f"Warning: missing {f.name}, skipping {model_type} ({dataset})")
                warned_missing.add(key)

if dfs:
    perf = pd.concat(dfs, ignore_index=True)
else:
    perf = pd.DataFrame(
        columns=["dataset", "model_type", "plot_loss", "uses_response_component"]
    )
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
    available_models = set(subset["model_type"].unique())
    order = [
        "NEF" if m.startswith("NEF") else m
        for m in MODEL_ORDER[dataset]
        if ("NEF" if m.startswith("NEF") else m) in available_models
    ]

    if subset.empty or not order:
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
        y="plot_loss",
        order=order,
        hue="model_type",
        palette=PALETTE,
        inner=None,
        legend=False,
        cut=0,
        ax=ax,
    )
    if subset["uses_response_component"].all():
        ax.set_ylabel("Response loss (MSE / NLL)")
    else:
        ax.set_ylabel("CV loss")
    ax.set_xlabel("")
    sns.despine(ax=ax, top=True, right=True)

    if len(order) >= 2:
        annotate_violins(ax, subset, "model_type", "plot_loss", order)
    ax.set_title(TITLES[dataset])

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "model_performance.png", dpi=300)
plt.savefig(FIGURES_DIR / "model_performance.pdf")
print("Saved figures/model_performance.{png,pdf}")
