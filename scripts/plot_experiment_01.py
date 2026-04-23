#!/usr/bin/env python3
"""
Plot results from experiment_01_error_activity.

Regplot of mean error population activity vs prediction error
across all pids, trials, and observations.

Usage:
    python scripts/plot_experiment_01.py
    python scripts/plot_experiment_01.py --dataset yoo --pe_type decoded
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, data_path
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette

DATASET = "carrabin"
OBS_MIN = 3
OBS_MAX = 5

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=DATASET)
parser.add_argument(
    "--pe_type",
    type=str,
    default="raw",
    choices=("raw", "decoded"),
    help="Which prediction error to plot",
)
args = parser.parse_args()

data_file = (
    data_path("experiments") / "experiment_01" / f"experiment_01_{args.dataset}.pkl"
)
if not data_file.exists():
    raise FileNotFoundError(
        f"Combined output not found: {data_file}\n"
        f"Run: python experiments/experiment_01_error_activity.py "
        f"--collect --dataset {args.dataset}"
    )

df = pd.read_pickle(data_file)
obs_col = "stage" if args.dataset == "jiang" else "observation"
plot_df = df[(df[obs_col] >= OBS_MIN) & (df[obs_col] <= OBS_MAX)].copy()

if plot_df.empty:
    raise ValueError(f"No data for {obs_col} {OBS_MIN}-{OBS_MAX} in {args.dataset}")

plot_df["delta_response"] = plot_df["response_after"] - plot_df["response_before"]

pe_col = f"prediction_error_{args.pe_type}"

apply_style()
PALETTE = get_palette()
color_on = PALETTE.get("NEF_recurrent", PALETTE.get("NEF", "C0"))
color_off = PALETTE.get("RL", "C1")

fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, constrained_layout=True)
ax_activity = axes[0]
ax_update = axes[1]
sns.regplot(
    data=plot_df,
    x=pe_col,
    y="mean_activity_on",
    scatter=True,
    scatter_kws={"alpha": 0.15, "s": 6, "color": color_on},
    line_kws={"color": color_on, "linewidth": 2, "label": "on neurons"},
    ax=ax_activity,
)
sns.regplot(
    data=plot_df,
    x=pe_col,
    y="mean_activity_off",
    scatter=True,
    scatter_kws={"alpha": 0.15, "s": 6, "color": color_off},
    line_kws={"color": color_off, "linewidth": 2, "label": "off neurons"},
    ax=ax_activity,
)
ax_activity.legend()
ax_activity.set_xlabel(f"Prediction error ({args.pe_type})")
ax_activity.set_ylabel("Mean neuron activity (Hz)")
ax_activity.set_title(f"{args.dataset} — {obs_col}s {OBS_MIN}–{OBS_MAX}")
sns.despine(ax=ax_activity, top=True, right=True)

sns.regplot(
    data=plot_df,
    x="mean_activity_on",
    y="delta_response",
    scatter=True,
    scatter_kws={"alpha": 0.15, "s": 6, "color": color_on},
    line_kws={"color": color_on, "linewidth": 2, "label": "on neurons"},
    ax=ax_update,
)
sns.regplot(
    data=plot_df,
    x="mean_activity_off",
    y="delta_response",
    scatter=True,
    scatter_kws={"alpha": 0.15, "s": 6, "color": color_off},
    line_kws={"color": color_off, "linewidth": 2, "label": "off neurons"},
    ax=ax_update,
)
ax_update.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax_update.legend()
ax_update.set_xlabel("Mean neuron activity (Hz)")
ax_update.set_ylabel("Δ response (after − before observation)")
ax_update.set_title(f"{args.dataset} — {obs_col}s {OBS_MIN}–{OBS_MAX}")
sns.despine(ax=ax_update, top=True, right=True)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fname = f"experiment_01_{args.dataset}"
plt.savefig(FIGURES_DIR / f"{fname}.png", dpi=300)
plt.savefig(FIGURES_DIR / f"{fname}.pdf")
print(f"Saved figures/{fname}.{{png,pdf}}")
