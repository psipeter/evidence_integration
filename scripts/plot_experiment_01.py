#!/usr/bin/env python3
"""
Plot results from experiment_01_error_activity (combined carrabin + jiang).

Loads combined experiment pickles when present and builds a 2×3 figure
(carrabin | jiang | yoo blank). Yoo column is reserved for future data.

Usage:
    python scripts/plot_experiment_01.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, data_path
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette

OBS_MIN = 3
OBS_MAX = 5
pe_col = "prediction_error_raw"

data: dict[str, pd.DataFrame | None] = {}
for ds in ("carrabin", "jiang"):
    pkl_path = data_path("experiments") / "experiment_01" / f"experiment_01_{ds}.pkl"
    if pkl_path.exists():
        data[ds] = pd.read_pickle(pkl_path)
    else:
        print(f"Warning: missing combined output: {pkl_path}")
        data[ds] = None

plot_df_carrabin: pd.DataFrame | None = None
if data["carrabin"] is not None:
    df_c = data["carrabin"]
    obs_col_c = "observation"
    tmp = df_c[(df_c[obs_col_c] >= OBS_MIN) & (df_c[obs_col_c] <= OBS_MAX)].copy()
    if not tmp.empty:
        tmp["delta_response"] = tmp["response_after"] - tmp["response_before"]
        plot_df_carrabin = tmp

plot_df_jiang: pd.DataFrame | None = None
if data["jiang"] is not None:
    tmp_j = data["jiang"].copy()
    tmp_j["delta_response"] = tmp_j["response_after"] - tmp_j["response_before"]
    plot_df_jiang = tmp_j

apply_style()
PALETTE = get_palette()
color_on = PALETTE.get("NEF_recurrent", PALETTE.get("NEF", "C0"))
color_off = PALETTE.get("RL", "C1")

stage_palette = None
if plot_df_jiang is not None and not plot_df_jiang.empty and "stage" in plot_df_jiang.columns:
    n_stages = int(plot_df_jiang["stage"].nunique())
    if n_stages > 0:
        stage_palette = sns.color_palette("flare", n_colors=n_stages)

fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE, constrained_layout=True)
for ax in (axes[0, 2], axes[1, 2]):
    ax.set_visible(False)

obs_col_c = "observation"
carrabin_title = f"carrabin — {obs_col_c}s {OBS_MIN}–{OBS_MAX}"

# --- Column 0: carrabin ---
if plot_df_carrabin is not None:
    sns.regplot(
        data=plot_df_carrabin,
        x=pe_col,
        y="mean_activity_on",
        scatter=True,
        scatter_kws={"alpha": 0.15, "s": 6, "color": color_on},
        line_kws={"color": color_on, "linewidth": 2, "label": "on neurons"},
        ax=axes[0, 0],
    )
    sns.regplot(
        data=plot_df_carrabin,
        x=pe_col,
        y="mean_activity_off",
        scatter=True,
        scatter_kws={"alpha": 0.15, "s": 6, "color": color_off},
        line_kws={"color": color_off, "linewidth": 2, "label": "off neurons"},
        ax=axes[0, 0],
    )
    axes[0, 0].legend()
    axes[0, 0].set_xlabel("Prediction error (raw)")
    axes[0, 0].set_ylabel("Mean neuron activity (Hz)")
    axes[0, 0].set_title(carrabin_title)
    sns.despine(ax=axes[0, 0], top=True, right=True)

    sns.regplot(
        data=plot_df_carrabin,
        x="mean_activity_on",
        y="delta_response",
        scatter=True,
        scatter_kws={"alpha": 0.15, "s": 6, "color": color_on},
        line_kws={"color": color_on, "linewidth": 2, "label": "on neurons"},
        ax=axes[1, 0],
    )
    sns.regplot(
        data=plot_df_carrabin,
        x="mean_activity_off",
        y="delta_response",
        scatter=True,
        scatter_kws={"alpha": 0.15, "s": 6, "color": color_off},
        line_kws={"color": color_off, "linewidth": 2, "label": "off neurons"},
        ax=axes[1, 0],
    )
    axes[1, 0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1, 0].legend()
    axes[1, 0].set_xlabel("Mean neuron activity (Hz)")
    axes[1, 0].set_ylabel("Δ response (after − before observation)")
    axes[1, 0].set_title(carrabin_title)
    sns.despine(ax=axes[1, 0], top=True, right=True)
else:
    axes[0, 0].set_visible(False)
    axes[1, 0].set_visible(False)

# --- Column 1: jiang ---
if plot_df_jiang is not None and stage_palette is not None:
    for i, stage_val in enumerate(sorted(plot_df_jiang["stage"].unique())):
        stage_df = plot_df_jiang[plot_df_jiang["stage"] == stage_val]
        sns.regplot(
            data=stage_df,
            x="prediction_error_raw",
            y="mean_activity_on",
            scatter=True,
            scatter_kws={"alpha": 0.15, "s": 6, "color": stage_palette[i]},
            line_kws={
                "color": stage_palette[i],
                "linewidth": 2,
                "label": f"stage {stage_val}",
            },
            ax=axes[0, 1],
        )
    axes[0, 1].legend()
    axes[0, 1].set_xlabel("Prediction error (raw)")
    axes[0, 1].set_ylabel("Mean neuron activity (Hz)")
    axes[0, 1].set_title("jiang — all stages")
    sns.despine(ax=axes[0, 1], top=True, right=True)

    for i, stage_val in enumerate(sorted(plot_df_jiang["stage"].unique())):
        stage_df = plot_df_jiang[plot_df_jiang["stage"] == stage_val]
        sns.regplot(
            data=stage_df,
            x="rd",
            y="delta_response",
            scatter=True,
            scatter_kws={"alpha": 0.2, "s": 6, "color": stage_palette[i]},
            line_kws={
                "color": stage_palette[i],
                "linewidth": 2,
                "label": f"stage {stage_val}",
            },
            ax=axes[1, 1],
        )
    axes[1, 1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1, 1].legend()
    axes[1, 1].set_xlabel("Neighbor degree (rd)")
    axes[1, 1].set_ylabel("Δ response (after − before observation)")
    axes[1, 1].set_title("jiang — all stages")
    sns.despine(ax=axes[1, 1], top=True, right=True)
else:
    axes[0, 1].set_visible(False)
    axes[1, 1].set_visible(False)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fname = "experiment_01_combined"
plt.savefig(FIGURES_DIR / f"{fname}.png", dpi=300)
plt.savefig(FIGURES_DIR / f"{fname}.pdf")
print(f"Saved figures/{fname}.{{png,pdf}}")
