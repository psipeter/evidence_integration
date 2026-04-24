#!/usr/bin/env python3
"""
Plot error ensemble activity against reconstructed prediction error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, RUNS_DIR, data_path
from utils.plot_style import FIGURE_SIZE, apply_style

ENCODER_THRESHOLD = 0.5
MODEL_TYPE = "NEF_recurrent"
DATASETS = ("carrabin", "jiang", "yoo")
pe_col = "prediction_error_raw"
OBS_RANGE = {
    "carrabin": (2, 5),
    "jiang": (1, 3),  # stages 1-3, skip stage 0
    "yoo": (2, 30),  # adjust as needed
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses_folder", type=str, default="joint_loss")
    parser.add_argument("--activity_folder", type=str, default="save_activities")
    args = parser.parse_args()

    activities_data: dict[str, pd.DataFrame | None] = {}
    encoders_data: dict[str, pd.DataFrame | None] = {}
    responses_data: dict[str, pd.DataFrame | None] = {}
    raw_data: dict[str, pd.DataFrame | None] = {}

    available_datasets: list[str] = []
    for dataset in DATASETS:
        activities_path = (
            data_path("experiments")
            / args.activity_folder
            / f"activities_error_{dataset}.pkl"
        )
        encoders_path = (
            data_path("experiments")
            / args.activity_folder
            / f"encoders_error_{dataset}.pkl"
        )
        responses_path = (
            RUNS_DIR / args.responses_folder / f"{MODEL_TYPE}_{dataset}_responses.pkl"
        )
        raw_path = data_path(f"{dataset}.pkl")

        required = [activities_path, encoders_path, responses_path, raw_path]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            print(f"Warning: missing required files for {dataset}:")
            for p in missing:
                print(f"  - {p}")
            activities_data[dataset] = None
            encoders_data[dataset] = None
            responses_data[dataset] = None
            raw_data[dataset] = None
            continue

        activities_data[dataset] = pd.read_pickle(activities_path)
        encoders_data[dataset] = pd.read_pickle(encoders_path)
        responses_data[dataset] = pd.read_pickle(responses_path)
        raw_data[dataset] = pd.read_pickle(raw_path)
        available_datasets.append(dataset)

    if not available_datasets:
        print("No datasets available with all required files.")
        return

    # Reconstruct raw prediction error for carrabin.
    if "carrabin" in available_datasets:
        responses_df = responses_data["carrabin"]
        raw_df = raw_data["carrabin"]
        activities_df = activities_data["carrabin"]
        assert responses_df is not None and raw_df is not None and activities_df is not None

        carrabin_merged = responses_df.merge(
            raw_df[["pid", "trial", "observation", "value"]],
            on=["pid", "trial", "observation"],
            how="left",
        )
        carrabin_merged = carrabin_merged.sort_values(["pid", "trial", "observation"])
        carrabin_merged["prev_response"] = (
            carrabin_merged.groupby(["pid", "trial"])["response"].shift(1).fillna(0.0)
        )
        carrabin_merged[pe_col] = carrabin_merged["value"] - carrabin_merged["prev_response"]

        activities_data["carrabin"] = activities_df.merge(
            carrabin_merged[["pid", "trial", "observation", pe_col]],
            on=["pid", "trial", "observation"],
            how="left",
        )

    if "jiang" in available_datasets:
        responses_df = responses_data["jiang"]
        raw_df = raw_data["jiang"]
        activities_df = activities_data["jiang"]
        assert responses_df is not None and raw_df is not None and activities_df is not None

        jiang_merged = responses_df.merge(
            raw_df[["pid", "trial", "stage", "value"]].drop_duplicates(),
            on=["pid", "trial", "stage"],
            how="left",
        )
        jiang_merged = jiang_merged.sort_values(["pid", "trial", "stage"])
        jiang_merged["prev_response"] = (
            jiang_merged.groupby(["pid", "trial"])["response"].shift(1).fillna(0.0)
        )
        jiang_merged[pe_col] = jiang_merged["value"] - jiang_merged["prev_response"]
        jiang_merged = jiang_merged[jiang_merged["stage"] > 0]

        activities_data["jiang"] = activities_df.merge(
            jiang_merged[["pid", "trial", "stage", pe_col]],
            on=["pid", "trial", "stage"],
            how="left",
        )
    if "jiang" in available_datasets and activities_data["jiang"] is not None:
        activities_data["jiang"] = (
            activities_data["jiang"]
            .sort_values(["pid", "trial", "stage"])
            .groupby(["pid", "trial", "stage"], sort=False)
            .first()
            .reset_index()
        )

    if "yoo" in available_datasets:
        responses_df = responses_data["yoo"]
        raw_df = raw_data["yoo"]
        activities_df = activities_data["yoo"]
        assert responses_df is not None and raw_df is not None and activities_df is not None

        yoo_merged = responses_df.merge(
            raw_df[["pid", "trial", "observation", "value"]],
            on=["pid", "trial", "observation"],
            how="left",
        )
        yoo_merged = yoo_merged.sort_values(["pid", "trial", "observation"])
        yoo_merged["prev_response"] = (
            yoo_merged.groupby(["pid", "trial"])["response"].shift(1).fillna(0.0)
        )
        yoo_merged[pe_col] = yoo_merged["value"] - yoo_merged["prev_response"]

        activities_data["yoo"] = activities_df.merge(
            yoo_merged[["pid", "trial", "observation", pe_col]],
            on=["pid", "trial", "observation"],
            how="left",
        )

    # Compute on/off activity means from encoder metadata for all loaded datasets.
    for dataset in available_datasets:
        activities_df = activities_data[dataset]
        encoders_df = encoders_data[dataset]
        if activities_df is None or encoders_df is None:
            continue

        neuron_cols = [c for c in activities_df.columns if c.startswith("n")]
        for pid, pid_enc in encoders_df.groupby("pid"):
            on_idx = pid_enc[pid_enc["enc_dim_1"] > ENCODER_THRESHOLD]["neuron_idx"].values
            off_idx = pid_enc[pid_enc["enc_dim_1"] < -ENCODER_THRESHOLD]["neuron_idx"].values
            mask = activities_df["pid"] == pid

            on_cols = [f"n{i}" for i in on_idx if f"n{i}" in neuron_cols]
            off_cols = [f"n{i}" for i in off_idx if f"n{i}" in neuron_cols]

            activities_df.loc[mask, "mean_activity_on"] = activities_df.loc[
                mask, on_cols
            ].mean(axis=1)
            activities_df.loc[mask, "mean_activity_off"] = activities_df.loc[
                mask, off_cols
            ].mean(axis=1)

        activities_data[dataset] = activities_df

    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE, constrained_layout=True)

    cb_palette = sns.color_palette("colorblind")
    color_on = cb_palette[0]
    color_off = cb_palette[1]
    dataset_styles = {
        "carrabin": {"marker": "o", "linestyle": "-"},
        "jiang": {"marker": "s", "linestyle": "--"},
        "yoo": {"marker": "^", "linestyle": ":"},
    }

    for dataset in available_datasets:
        plot_df = activities_data[dataset]
        if plot_df is None:
            continue
        obs_col = "stage" if dataset == "jiang" else "observation"
        obs_min, obs_max = OBS_RANGE[dataset]
        plot_df = plot_df[(plot_df[obs_col] >= obs_min) & (plot_df[obs_col] <= obs_max)].copy()
        if plot_df.empty:
            continue

        style = dataset_styles[dataset]
        for y_col, color, label in [
            ("mean_activity_on", color_on, "on neurons"),
            ("mean_activity_off", color_off, "off neurons"),
        ]:
            ax.scatter(
                plot_df[pe_col],
                plot_df[y_col],
                alpha=0.15,
                s=6,
                color=color,
                marker=style["marker"],
            )
            sns.regplot(
                data=plot_df,
                x=pe_col,
                y=y_col,
                scatter=False,
                line_kws={
                    "color": color,
                    "linewidth": 2,
                    "linestyle": style["linestyle"],
                },
                ax=ax,
            )

    dataset_handles = []
    for dataset in available_datasets:
        style = dataset_styles[dataset]
        handle = mlines.Line2D(
            [],
            [],
            color="gray",
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=6,
            label=dataset,
        )
        dataset_handles.append(handle)

    neuron_handles = [
        mlines.Line2D([], [], color=color_on, linewidth=2, label="on neurons"),
        mlines.Line2D([], [], color=color_off, linewidth=2, label="off neurons"),
    ]

    ax.legend(
        handles=dataset_handles + neuron_handles,
        frameon=False,
    )
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Mean neuron activity (Hz)")
    ax.set_title("Error population activity vs prediction error")
    sns.despine(ax=ax, top=True, right=True)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fname = "neural_activities"
    plt.savefig(FIGURES_DIR / f"{fname}.png", dpi=300)
    plt.savefig(FIGURES_DIR / f"{fname}.pdf")
    print(f"Saved figures/{fname}.{{png,pdf}}")


if __name__ == "__main__":
    main()
