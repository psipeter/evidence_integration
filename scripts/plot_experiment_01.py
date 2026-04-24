#!/usr/bin/env python3
"""
Plot results from experiment_01_error_activity (combined carrabin + jiang).

Loads combined experiment pickles when present and builds a 2×3 figure
(carrabin | jiang | yoo). Bottom row, third column remains unused.

Usage:
    python scripts/plot_experiment_01.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, data_path
from utils.plot_style import FIGURE_SIZE, apply_style

OBS_MIN = 3
OBS_MAX = 5
pe_col = "prediction_error_raw"

data: dict[str, pd.DataFrame | None] = {}
for ds in ("carrabin", "jiang", "yoo"):
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
    if "stage" in tmp_j.columns:
        plot_df_jiang = tmp_j[tmp_j["stage"].isin([1, 2, 3])]
    else:
        plot_df_jiang = None

plot_df_yoo: pd.DataFrame | None = None
if data["yoo"] is not None:
    tmp_y = data["yoo"].copy()
    tmp_y["delta_response"] = tmp_y["response_after"] - tmp_y["response_before"]
    plot_df_yoo = tmp_y

lambda_palette: list | None = None
if plot_df_yoo is not None and not plot_df_yoo.empty:
    lambda_bin_edges = np.quantile(
        plot_df_yoo["lambda_"].dropna(), np.linspace(0, 1, 5)
    )
    plot_df_yoo["lambda_bin"] = pd.cut(
        plot_df_yoo["lambda_"],
        bins=lambda_bin_edges,
        include_lowest=True,
        duplicates="drop",
    )
    n_lambda_bins = plot_df_yoo["lambda_bin"].nunique()
    lambda_palette = sns.color_palette("colorblind", n_colors=n_lambda_bins)

if plot_df_jiang is not None and not plot_df_jiang.empty:

    def add_row_idx(df):
        df = df.copy()
        df["row_idx"] = df.groupby(["pid", "trial", "stage"]).cumcount()
        return df

    jiang_raw = pd.read_pickle(data_path("jiang.pkl"))
    who_rd = jiang_raw[["pid", "trial", "stage", "who", "rd"]].copy()

    plot_df_jiang = add_row_idx(plot_df_jiang)
    who_rd = add_row_idx(who_rd)
    plot_df_jiang = plot_df_jiang.merge(
        who_rd[["pid", "trial", "stage", "row_idx", "who", "rd"]].rename(
            columns={"rd": "rd_true"}
        ),
        on=["pid", "trial", "stage", "row_idx"],
        how="left",
    ).drop(columns=["row_idx"])

    stage2_rd = (
        plot_df_jiang[plot_df_jiang["stage"] == 2][["pid", "trial", "who", "rd_true"]]
        .drop_duplicates()
        .rename(columns={"rd_true": "rd_stage2"})
    )
    plot_df_jiang = plot_df_jiang.merge(stage2_rd, on=["pid", "trial", "who"], how="left")
    plot_df_jiang.loc[plot_df_jiang["stage"] == 1, "rd_true"] = (
        plot_df_jiang.loc[plot_df_jiang["stage"] == 1, "rd_stage2"]
    )
    plot_df_jiang = plot_df_jiang.drop(columns=["rd_stage2"])

    rd_bin_edges = np.quantile(
        plot_df_jiang["rd_true"].dropna(), np.linspace(0, 1, 11)
    )

apply_style()
cb_palette = sns.color_palette("colorblind")

fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE, constrained_layout=True)
axes[1, 2].set_visible(False)

obs_col_c = "observation"
carrabin_title = f"carrabin — {obs_col_c}s {OBS_MIN}–{OBS_MAX}"

# --- Column 0: carrabin ---
if plot_df_carrabin is not None:
    sns.regplot(
        data=plot_df_carrabin,
        x=pe_col,
        y="mean_activity_on",
        scatter=True,
        scatter_kws={"alpha": 0.15, "s": 6, "color": cb_palette[0]},
        line_kws={"color": cb_palette[0], "linewidth": 2, "label": "on neurons"},
        ax=axes[0, 0],
    )
    sns.regplot(
        data=plot_df_carrabin,
        x=pe_col,
        y="mean_activity_off",
        scatter=True,
        scatter_kws={"alpha": 0.15, "s": 6, "color": cb_palette[1]},
        line_kws={"color": cb_palette[1], "linewidth": 2, "label": "off neurons"},
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
        scatter_kws={"alpha": 0.15, "s": 6, "color": cb_palette[0]},
        line_kws={"color": cb_palette[0], "linewidth": 2, "label": "on neurons"},
        ax=axes[1, 0],
    )
    sns.regplot(
        data=plot_df_carrabin,
        x="mean_activity_off",
        y="delta_response",
        scatter=True,
        scatter_kws={"alpha": 0.15, "s": 6, "color": cb_palette[1]},
        line_kws={"color": cb_palette[1], "linewidth": 2, "label": "off neurons"},
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
if plot_df_jiang is not None and not plot_df_jiang.empty:
    for i, stage_val in enumerate([1, 2, 3]):
        stage_df = plot_df_jiang[plot_df_jiang["stage"] == stage_val]
        if stage_df.empty:
            continue
        sns.regplot(
            data=stage_df,
            x="prediction_error_raw",
            y="mean_activity_on",
            x_bins=np.arange(-2, 2, 0.2),
            scatter_kws={"alpha": 0.6, "s": 20, "color": cb_palette[i]},
            line_kws={
                "color": cb_palette[i],
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

    for i, stage_val in enumerate([1, 2, 3]):
        stage_df = plot_df_jiang[plot_df_jiang["stage"] == stage_val]
        if stage_df.empty:
            continue
        sns.regplot(
            data=stage_df,
            x="rd_true",
            y="mean_activity_on",
            x_bins=rd_bin_edges,
            scatter_kws={"alpha": 0.6, "s": 20, "color": cb_palette[i]},
            line_kws={
                "color": cb_palette[i],
                "linewidth": 2,
                "label": f"stage {stage_val}",
            },
            ax=axes[1, 1],
        )
    axes[1, 1].legend()
    axes[1, 1].set_xlabel("Neighbor degree (rd)")
    axes[1, 1].set_ylabel("Mean on-neuron activity (Hz)")
    axes[1, 1].set_title("jiang — all stages")
    sns.despine(ax=axes[1, 1], top=True, right=True)
else:
    axes[0, 1].set_visible(False)
    axes[1, 1].set_visible(False)

# --- Column 2: yoo ---
if (
    plot_df_yoo is not None
    and not plot_df_yoo.empty
    and lambda_palette is not None
):
    for i, (bin_label, bin_df) in enumerate(plot_df_yoo.groupby("lambda_bin")):
        sns.regplot(
            data=bin_df,
            x="observation",
            y="mean_activity_on",
            x_bins=int(bin_df["observation"].nunique()),
            scatter_kws={"alpha": 0.6, "s": 20, "color": lambda_palette[i]},
            line_kws={
                "color": lambda_palette[i],
                "linewidth": 2,
                "label": f"λ {bin_label}",
            },
            ax=axes[0, 2],
        )
    axes[0, 2].legend()
    axes[0, 2].set_xlabel("Observation number")
    axes[0, 2].set_ylabel("Mean on-neuron activity (Hz)")
    axes[0, 2].set_title("yoo — λ bins")
    sns.despine(ax=axes[0, 2], top=True, right=True)
else:
    axes[0, 2].set_visible(False)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fname = "experiment_01_combined"
plt.savefig(FIGURES_DIR / f"{fname}.png", dpi=300)
plt.savefig(FIGURES_DIR / f"{fname}.pdf")
print(f"Saved figures/{fname}.{{png,pdf}}")
