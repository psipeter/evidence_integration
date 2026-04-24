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
import numpy as np
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
COUNTING_OBS_RANGE = {
    "carrabin": (2, 5),
    "jiang": (0, 30),
    "yoo": (2, 30),
}
PE_BIN_TYPE = "equally_spaced"  # "equally_spaced" or "quantile"
PE_BIN_N = 10
PE_BIN_RANGE = (-1.5, 1.5)
TIMECOURSE_OBS_MIN = 2
TIMECOURSE_OBS_MAX = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses_folder", type=str, default="joint_loss")
    parser.add_argument("--activity_folder", type=str, default="save_activities")
    args = parser.parse_args()

    activities_data: dict[str, pd.DataFrame | None] = {}
    encoders_data: dict[str, pd.DataFrame | None] = {}
    counting_activities_data: dict[str, pd.DataFrame | None] = {}
    counting_encoders_data: dict[str, pd.DataFrame | None] = {}
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
        counting_activities_path = (
            data_path("experiments")
            / args.activity_folder
            / f"activities_counting_{dataset}.pkl"
        )
        counting_encoders_path = (
            data_path("experiments")
            / args.activity_folder
            / f"encoders_counting_{dataset}.pkl"
        )
        responses_path = (
            RUNS_DIR / args.responses_folder / f"{MODEL_TYPE}_{dataset}_responses.pkl"
        )
        raw_path = data_path(f"{dataset}.pkl")

        if counting_activities_path.exists() and counting_encoders_path.exists():
            counting_activities_data[dataset] = pd.read_pickle(counting_activities_path)
            counting_encoders_data[dataset] = pd.read_pickle(counting_encoders_path)
        else:
            counting_activities_data[dataset] = None
            counting_encoders_data[dataset] = None

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

    windowed_path = (
        data_path("experiments")
        / args.activity_folder
        / "activities_windowed_error_carrabin.npz"
    )
    windowed_carrabin = np.load(windowed_path) if windowed_path.exists() else None
    if windowed_carrabin is None:
        print(f"Warning: missing windowed activities: {windowed_path}")

    yoo_params_path = (
        RUNS_DIR / args.responses_folder / "NEF_recurrent_yoo_params.pkl"
    )
    if yoo_params_path.exists():
        yoo_params = pd.read_pickle(yoo_params_path)[["pid", "lambda_"]].drop_duplicates()
    else:
        print(f"Warning: missing yoo params file: {yoo_params_path}")
        yoo_params = None

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
        jiang_raw = pd.read_pickle(data_path("jiang.pkl"))
        who_rd = jiang_raw[["pid", "trial", "stage", "who", "rd"]].copy()

        def add_row_idx(df):
            df = df.copy()
            df["row_idx"] = df.groupby(["pid", "trial", "stage"]).cumcount()
            return df

        activities_data["jiang"] = add_row_idx(activities_data["jiang"])
        who_rd = add_row_idx(who_rd)
        activities_data["jiang"] = (
            activities_data["jiang"]
            .merge(
                who_rd[["pid", "trial", "stage", "row_idx", "who", "rd"]].rename(
                    columns={"rd": "rd_true"}
                ),
                on=["pid", "trial", "stage", "row_idx"],
                how="left",
            )
            .drop(columns=["row_idx"])
        )

        stage2_rd = (
            activities_data["jiang"][activities_data["jiang"]["stage"] == 2][
                ["pid", "trial", "who", "rd_true"]
            ]
            .drop_duplicates()
            .rename(columns={"rd_true": "rd_stage2"})
        )
        activities_data["jiang"] = activities_data["jiang"].merge(
            stage2_rd, on=["pid", "trial", "who"], how="left"
        )
        activities_data["jiang"].loc[
            activities_data["jiang"]["stage"] == 1, "rd_true"
        ] = activities_data["jiang"].loc[
            activities_data["jiang"]["stage"] == 1, "rd_stage2"
        ]
        activities_data["jiang"] = activities_data["jiang"].drop(columns=["rd_stage2"])

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

            if dataset in ("yoo", "jiang"):
                weight_on_idx = pid_enc[
                    pid_enc["enc_dim_0"] > ENCODER_THRESHOLD
                ]["neuron_idx"].values
                weight_on_cols = [f"n{i}" for i in weight_on_idx if f"n{i}" in neuron_cols]
                activities_df.loc[mask, "mean_activity_weight_on"] = (
                    activities_df.loc[mask, weight_on_cols].mean(axis=1)
                )

        activities_data[dataset] = activities_df

    carrabin_on_idx_by_pid: dict[int, np.ndarray] = {}
    if "carrabin" in available_datasets and encoders_data["carrabin"] is not None:
        for pid, pid_enc in encoders_data["carrabin"].groupby("pid"):
            on_idx = pid_enc[
                pid_enc["enc_dim_1"] > ENCODER_THRESHOLD
            ]["neuron_idx"].values
            carrabin_on_idx_by_pid[int(pid)] = on_idx

    timecourse_mean = None
    timecourse_std = None
    t_axis: np.ndarray | None = None

    if windowed_carrabin is not None and carrabin_on_idx_by_pid:
        acts = windowed_carrabin["activities"]
        pid_ids = windowed_carrabin["pid_ids"]
        # dt_sample = float(windowed_carrabin["dt_sample"])
        dt_sample = 0.01  # matches --dt_sample used when saving
        obs_slice = slice(TIMECOURSE_OBS_MIN - 1, TIMECOURSE_OBS_MAX)

        pid_timecourses = []
        for i, pid in enumerate(pid_ids):
            on_idx = carrabin_on_idx_by_pid.get(int(pid))
            if on_idx is None or len(on_idx) == 0:
                continue
            pid_acts = acts[i]
            selected = pid_acts[:, obs_slice, :, :][:, :, :, on_idx]
            pid_mean = np.nanmean(selected, axis=(0, 1, 3))
            pid_timecourses.append(pid_mean)

        if pid_timecourses:
            pid_timecourses_arr = np.stack(pid_timecourses, axis=0)
            timecourse_mean = np.nanmean(pid_timecourses_arr, axis=0)
            timecourse_std = np.nanstd(pid_timecourses_arr, axis=0)
            t_axis = np.arange(len(timecourse_mean)) * dt_sample

    for dataset in available_datasets:
        activities_df = counting_activities_data.get(dataset)
        encoders_df = counting_encoders_data.get(dataset)
        if activities_df is None or encoders_df is None:
            continue

        neuron_cols = [c for c in activities_df.columns if c.startswith("n")]
        for pid, pid_enc in encoders_df.groupby("pid"):
            pos_idx = pid_enc[pid_enc["enc_dim_0"] > ENCODER_THRESHOLD]["neuron_idx"].values
            mask = activities_df["pid"] == pid
            pos_cols = [f"n{i}" for i in pos_idx if f"n{i}" in neuron_cols]
            activities_df.loc[mask, "mean_activity_pos"] = (
                activities_df.loc[mask, pos_cols].mean(axis=1)
            )

        counting_activities_data[dataset] = activities_df

    if "yoo" in available_datasets and yoo_params is not None:
        activities_data["yoo"] = activities_data["yoo"].merge(
            yoo_params, on="pid", how="left"
        )
        lambda_bin_edges = np.unique(
            np.quantile(activities_data["yoo"]["lambda_"].dropna(), [0, 1/3, 2/3, 1])
        )
        n_bins = len(lambda_bin_edges) - 1
        bin_labels = [
            f"λ = [{lambda_bin_edges[i]:.2f} - {lambda_bin_edges[i+1]:.2f}]"
            for i in range(n_bins)
        ]
        activities_data["yoo"]["lambda_bin"] = pd.cut(
            activities_data["yoo"]["lambda_"],
            bins=lambda_bin_edges,
            labels=bin_labels,
            include_lowest=True,
        )

    apply_style()
    fig, axes = plt.subplots(
        1, 5, figsize=FIGURE_SIZE, constrained_layout=True, sharey=False
    )
    ax_pe = axes[0]
    ax_count = axes[1]
    ax_weight = axes[2]
    ax_rd = axes[3]
    ax_time = axes[4]

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
        if dataset == "jiang":
            plot_df = (
                plot_df.sort_values(["pid", "trial", "stage"])
                .groupby(["pid", "trial", "stage"], sort=False)
                .first()
                .reset_index()
            )
        if plot_df.empty:
            continue

        style = dataset_styles[dataset]
        if PE_BIN_TYPE == "equally_spaced":
            pe_bin_edges = np.linspace(PE_BIN_RANGE[0], PE_BIN_RANGE[1], PE_BIN_N + 1)
        elif PE_BIN_TYPE == "quantile":
            pe_bin_edges = np.quantile(
                plot_df[pe_col].dropna(), np.linspace(0, 1, PE_BIN_N + 1)
            )
        else:
            raise ValueError(f"Unknown PE_BIN_TYPE: {PE_BIN_TYPE!r}")
        for y_col, color, label in [
            ("mean_activity_on", color_on, "on neurons"),
            ("mean_activity_off", color_off, "off neurons"),
        ]:
            binned = (
                plot_df.groupby(
                    pd.cut(plot_df[pe_col], bins=pe_bin_edges, include_lowest=True)
                )[y_col]
                .agg(["mean", "std"])
                .dropna()
            )
            bin_centers = [interval.mid for interval in binned.index]
            ax_pe.errorbar(
                bin_centers,
                binned["mean"],
                yerr=binned["std"],
                fmt=style["marker"],
                color=color,
                alpha=0.8,
                markersize=5,
                linewidth=1.2,
                zorder=5,
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
                ax=ax_pe,
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

    ax_pe.legend(
        handles=dataset_handles + neuron_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    ax_pe.set_xlabel("Prediction error")
    sns.despine(ax=ax_pe, top=True, right=True)

    for dataset in available_datasets:
        activities_df = counting_activities_data.get(dataset)
        if activities_df is None or "mean_activity_pos" not in activities_df.columns:
            continue
        count_x_col = "trial_obs_idx" if dataset == "jiang" else "observation"
        obs_min, obs_max = COUNTING_OBS_RANGE[dataset]
        plot_df = activities_df[
            (activities_df[count_x_col] >= obs_min)
            & (activities_df[count_x_col] <= obs_max)
        ].copy()

        style = dataset_styles[dataset]
        if plot_df.empty or "mean_activity_pos" not in plot_df.columns:
            continue

        n_unique = int(plot_df[count_x_col].nunique())
        sns.regplot(
            data=plot_df,
            x=count_x_col,
            y="mean_activity_pos",
            x_bins=n_unique,
            scatter=True,
            scatter_kws={
                "alpha": 0.0,
                "s": 0,
            },
            line_kws={
                "color": cb_palette[0],
                "linewidth": 2,
                "linestyle": style["linestyle"],
            },
            ax=ax_count,
        )
        binned_means = plot_df.groupby(count_x_col)["mean_activity_pos"].agg(
            ["mean", "std"]
        )
        ax_count.errorbar(
            binned_means.index,
            binned_means["mean"],
            yerr=binned_means["std"],
            fmt=style["marker"],
            color=cb_palette[0],
            alpha=0.8,
            markersize=5,
            linewidth=1.2,
            zorder=5,
        )

    ax_count.legend(handles=dataset_handles, frameon=False)
    ax_count.set_xlabel("Observation index (within trial)")
    sns.despine(ax=ax_count, top=True, right=True)
    ax_count.set_xticks(range(0, 31, 5))

    if "yoo" in available_datasets:
        plot_df_w = activities_data["yoo"]
        if (
            plot_df_w is not None
            and "mean_activity_weight_on" in plot_df_w.columns
            and "lambda_bin" in plot_df_w.columns
        ):
            obs_min, obs_max = OBS_RANGE["yoo"]
            plot_df_w = plot_df_w[
                (plot_df_w["observation"] >= obs_min)
                & (plot_df_w["observation"] <= obs_max)
            ].copy()
            lambda_palette = sns.color_palette("colorblind", n_colors=3)
            for i, (bin_label, bin_df) in enumerate(
                plot_df_w.groupby("lambda_bin", observed=True)
            ):
                if bin_df.empty:
                    continue
                binned = bin_df.groupby("observation")["mean_activity_weight_on"].agg(
                    ["mean", "std"]
                )
                ax_weight.errorbar(
                    binned.index,
                    binned["mean"],
                    yerr=binned["std"],
                    fmt=dataset_styles["yoo"]["marker"],
                    color=lambda_palette[i],
                    alpha=0.8,
                    markersize=5,
                    linewidth=1.2,
                    label=str(bin_label),
                )
                sns.regplot(
                    data=bin_df,
                    x="observation",
                    y="mean_activity_weight_on",
                    scatter=False,
                    line_kws={
                        "color": lambda_palette[i],
                        "linewidth": 2,
                        "linestyle": dataset_styles["yoo"]["linestyle"],
                    },
                    ax=ax_weight,
                )
            ax_weight.legend(frameon=False)
            ax_weight.set_xlabel("Observation number")
            sns.despine(ax=ax_weight, top=True, right=True)
        else:
            ax_weight.set_visible(False)
    else:
        ax_weight.set_visible(False)
    ax_weight.set_xticks(range(0, 31, 5))

    if "jiang" in available_datasets:
        plot_df_rd = activities_data["jiang"]
        if (
            plot_df_rd is not None
            and "mean_activity_weight_on" in plot_df_rd.columns
            and "rd_true" in plot_df_rd.columns
        ):
            plot_df_rd = plot_df_rd[plot_df_rd["stage"].isin([1, 2, 3])].copy()
            stage_palette = sns.color_palette("colorblind", n_colors=3)
            rd_bin_edges = np.quantile(
                plot_df_rd["rd_true"].dropna(), np.linspace(0, 1, 11)
            )
            for i, stage_val in enumerate(sorted(plot_df_rd["stage"].unique())):
                stage_df = plot_df_rd[plot_df_rd["stage"] == stage_val]
                binned = (
                    stage_df.groupby(
                        pd.cut(
                            stage_df["rd_true"],
                            bins=rd_bin_edges,
                            include_lowest=True,
                        )
                    )["mean_activity_weight_on"]
                    .agg(["mean", "std"])
                    .dropna()
                )
                bin_centers = [interval.mid for interval in binned.index]
                ax_rd.errorbar(
                    bin_centers,
                    binned["mean"],
                    yerr=binned["std"],
                    fmt=dataset_styles["jiang"]["marker"],
                    color=stage_palette[i],
                    alpha=0.8,
                    markersize=5,
                    linewidth=1.2,
                    label=f"stage {stage_val}",
                    zorder=5,
                )
                sns.regplot(
                    data=stage_df,
                    x="rd_true",
                    y="mean_activity_weight_on",
                    x_bins=rd_bin_edges,
                    scatter=False,
                    line_kws={
                        "color": stage_palette[i],
                        "linewidth": 2,
                        "linestyle": dataset_styles["jiang"]["linestyle"],
                    },
                    ax=ax_rd,
                )
            ax_rd.legend(frameon=False)
            ax_rd.set_xlabel("Neighbor degree (rd)")
            sns.despine(ax=ax_rd, top=True, right=True)
        else:
            ax_rd.set_visible(False)
    else:
        ax_rd.set_visible(False)
    ax_rd.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    if timecourse_mean is not None and t_axis is not None:
        ax_time.fill_between(
            t_axis,
            timecourse_mean - timecourse_std,
            timecourse_mean + timecourse_std,
            color=cb_palette[0],
            alpha=0.3,
        )
        ax_time.plot(
            t_axis,
            timecourse_mean,
            color=cb_palette[0],
            linewidth=2,
        )
        ax_time.set_xlabel("Time within observation (s)")
        ax_time.set_title("Error neuron timecourse")
        sns.despine(ax=ax_time, top=True, right=True)
    else:
        ax_time.set_visible(False)

    ax_pe.set_title("Error-sensitive neurons")
    ax_count.set_title("Observation count neurons")
    ax_weight.set_title("Dynamic learning rate neurons")
    ax_rd.set_title("Neighbor information neurons")
    ax_pe.set_ylabel("Mean neuron activity (Hz)")
    ax_count.set_ylabel("")
    ax_weight.set_ylabel("")
    ax_rd.set_ylabel("")
    ax_time.set_ylabel("")
    ax_pe.set_xticks([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fname = "neural_activities"
    plt.savefig(FIGURES_DIR / f"{fname}.png", dpi=300)
    plt.savefig(FIGURES_DIR / f"{fname}.pdf")
    print(f"Saved figures/{fname}.{{png,pdf}}")


if __name__ == "__main__":
    main()
