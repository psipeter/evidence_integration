#!/usr/bin/env python3
"""
Plot error ensemble activity against reconstructed prediction error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

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
    "carrabin": (1, 5),
    "jiang": (1, 30),
    "yoo": (1, 30),
}
PE_BIN_TYPE = "equally_spaced"  # "equally_spaced" or "quantile"
PE_BIN_N = 10
PE_BIN_RANGE = (-1.5, 1.5)
LAMBDA_N = 5  # number of lowest/highest lambda pids to use in panel 3
ALPHA_N = 3  # number of pids to show for top/bottom alpha_0
ERROR_STYLE = "ci"  # "ci" for seaborn default 95% CI, or "sd" for standard deviation
TIMECOURSE_OBS_MIN = 1
TIMECOURSE_OBS_MAX = 5


def make_binned_df(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    bin_edges: np.ndarray | int,
    hue_col: str | None = None,
    hue_val=None,
) -> pd.DataFrame:
    """Return long-form df with x replaced by bin centers, ready for sns.lineplot."""
    df = df.copy()
    if isinstance(bin_edges, int):
        bins = pd.cut(df[x_col], bins=bin_edges, include_lowest=True)
    else:
        bins = pd.cut(df[x_col], bins=bin_edges, include_lowest=True)
    df["bin_center"] = bins.apply(lambda b: b.mid if pd.notna(b) else np.nan)
    df = df.dropna(subset=["bin_center", y_col])
    if hue_col is not None and hue_val is not None:
        df[hue_col] = hue_val
    return df[["bin_center", y_col] + ([hue_col] if hue_col else [])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, default="NEF200")
    args = parser.parse_args()
    run_dir = RUNS_DIR / args.run_folder

    activities_data: dict[str, pd.DataFrame | None] = {}
    encoders_data: dict[str, pd.DataFrame | None] = {}
    counting_activities_data: dict[str, pd.DataFrame | None] = {}
    counting_encoders_data: dict[str, pd.DataFrame | None] = {}
    responses_data: dict[str, pd.DataFrame | None] = {}
    raw_data: dict[str, pd.DataFrame | None] = {}

    available_datasets: list[str] = []
    for dataset in DATASETS:
        activities_path = run_dir / f"activities_error_{dataset}.pkl"
        encoders_path = run_dir / f"encoders_error_{dataset}.pkl"
        counting_activities_path = run_dir / f"activities_counting_{dataset}.pkl"
        counting_encoders_path = run_dir / f"encoders_counting_{dataset}.pkl"
        responses_path = run_dir / f"{MODEL_TYPE}_{dataset}_responses.pkl"
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

    windowed_path = run_dir / "activities_windowed_error_carrabin.npz"
    windowed_carrabin = np.load(windowed_path) if windowed_path.exists() else None
    if windowed_carrabin is None:
        print(f"Warning: missing windowed activities: {windowed_path}")
    carrabin_encoders_path = run_dir / "encoders_error_carrabin.pkl"
    carrabin_params_path = run_dir / "NEF_recurrent_carrabin_params.pkl"
    sorted_alpha = None
    if carrabin_params_path.exists():
        carrabin_params = pd.read_pickle(carrabin_params_path)
        sorted_alpha = carrabin_params.sort_values("alpha_0")
        bottom_alpha_pids = sorted_alpha.head(ALPHA_N)["pid"].tolist()
        top_alpha_pids = sorted_alpha.tail(ALPHA_N)["pid"].tolist()
    else:
        bottom_alpha_pids, top_alpha_pids = [], []

    yoo_params_path = run_dir / f"{MODEL_TYPE}_yoo_params.pkl"
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
    carrabin_encoders_df = None
    if carrabin_encoders_path.exists():
        carrabin_encoders_df = pd.read_pickle(carrabin_encoders_path)
    elif "carrabin" in available_datasets and encoders_data["carrabin"] is not None:
        print(
            f"Warning: missing run-folder carrabin encoders, falling back to old path: "
            f"{carrabin_encoders_path}"
        )
        carrabin_encoders_df = encoders_data["carrabin"]

    if carrabin_encoders_df is not None:
        for pid, pid_enc in carrabin_encoders_df.groupby("pid"):
            on_idx = pid_enc[
                pid_enc["enc_dim_1"] > ENCODER_THRESHOLD
            ]["neuron_idx"].values
            carrabin_on_idx_by_pid[int(pid)] = on_idx

    timecourse_df = None

    if windowed_carrabin is not None and carrabin_on_idx_by_pid:
        acts = windowed_carrabin["activities"]
        pid_ids = windowed_carrabin["pid_ids"]
        # dt_sample = float(windowed_carrabin["dt_sample"])
        dt_sample = 0.01  # matches --dt_sample used when saving
        timecourse_rows = []
        for ex_pid in bottom_alpha_pids:
            if ex_pid not in pid_ids:
                continue
            i = list(pid_ids).index(ex_pid)
            on_idx = carrabin_on_idx_by_pid.get(ex_pid)
            if on_idx is None or len(on_idx) == 0:
                continue
            pid_acts = acts[i]  # (n_trials, n_obs, n_timesteps, n_neurons)
            obs_slice = slice(TIMECOURSE_OBS_MIN - 1, TIMECOURSE_OBS_MAX)
            selected = pid_acts[:, obs_slice, :, :]  # (n_trials, n_obs, n_timesteps, n_neurons)
            selected_on = selected[:, :, :, on_idx]  # (n_trials, n_obs, n_timesteps, n_on)
            mean_over_neurons = np.nanmean(selected_on, axis=3)  # (n_trials, n_obs, n_timesteps)
            n_trials, n_obs_sel, n_timesteps = mean_over_neurons.shape
            for trial_i in range(n_trials):
                for obs_i in range(n_obs_sel):
                    for t_idx in range(n_timesteps):
                        timecourse_rows.append(
                            {
                                "t": t_idx * dt_sample,
                                "activity": float(mean_over_neurons[trial_i, obs_i, t_idx]),
                                "alpha_group": "low",
                            }
                        )
        for ex_pid in top_alpha_pids:
            if ex_pid not in pid_ids:
                continue
            i = list(pid_ids).index(ex_pid)
            on_idx = carrabin_on_idx_by_pid.get(ex_pid)
            if on_idx is None or len(on_idx) == 0:
                continue
            pid_acts = acts[i]  # (n_trials, n_obs, n_timesteps, n_neurons)
            obs_slice = slice(TIMECOURSE_OBS_MIN - 1, TIMECOURSE_OBS_MAX)
            selected = pid_acts[:, obs_slice, :, :]  # (n_trials, n_obs, n_timesteps, n_neurons)
            selected_on = selected[:, :, :, on_idx]  # (n_trials, n_obs, n_timesteps, n_on)
            mean_over_neurons = np.nanmean(selected_on, axis=3)  # (n_trials, n_obs, n_timesteps)
            n_trials, n_obs_sel, n_timesteps = mean_over_neurons.shape
            for trial_i in range(n_trials):
                for obs_i in range(n_obs_sel):
                    for t_idx in range(n_timesteps):
                        timecourse_rows.append(
                            {
                                "t": t_idx * dt_sample,
                                "activity": float(mean_over_neurons[trial_i, obs_i, t_idx]),
                                "alpha_group": "high",
                            }
                        )
        timecourse_df = pd.DataFrame(timecourse_rows)

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
    if "yoo" in available_datasets and activities_data["yoo"] is not None:
        obs_min, obs_max = OBS_RANGE["yoo"]
        plot_df = activities_data["yoo"]
        plot_df = plot_df[
            (plot_df["observation"] >= obs_min) & (plot_df["observation"] <= obs_max)
        ].copy()
        if PE_BIN_TYPE == "equally_spaced":
            pe_bin_edges = np.linspace(PE_BIN_RANGE[0], PE_BIN_RANGE[1], PE_BIN_N + 1)
        else:
            pe_bin_edges = np.quantile(
                plot_df[pe_col].dropna(), np.linspace(0, 1, PE_BIN_N + 1)
            )

        on_df = make_binned_df(
            plot_df, pe_col, "mean_activity_on", pe_bin_edges, "neuron_type", "on"
        )
        off_df = make_binned_df(
            plot_df, pe_col, "mean_activity_off", pe_bin_edges, "neuron_type", "off"
        )
        on_df = on_df.rename(columns={"mean_activity_on": "activity"})
        off_df = off_df.rename(columns={"mean_activity_off": "activity"})
        pe_long = pd.concat([on_df, off_df], ignore_index=True)

        sns.lineplot(
            data=pe_long,
            x="bin_center",
            y="activity",
            hue="neuron_type",
            palette={"on": cb_palette[0], "off": cb_palette[1]},
            errorbar=ERROR_STYLE,
            ax=ax_pe,
        )
        ax_pe.legend(frameon=False)
        ax_pe.get_legend().set_title("Encoding")
        ax_pe.set_xlabel("Prediction error")
        sns.despine(ax=ax_pe, top=True, right=True)
    else:
        ax_pe.set_visible(False)

    count_dfs = []
    for dataset in reversed(list(available_datasets)):
        activities_df = counting_activities_data.get(dataset)
        if activities_df is None or "mean_activity_pos" not in activities_df.columns:
            continue
        count_x_col = "trial_obs_idx" if dataset == "jiang" else "observation"
        obs_min, obs_max = COUNTING_OBS_RANGE[dataset]
        plot_df = activities_df.copy()
        if dataset == "jiang":
            plot_df["obs_plot"] = plot_df[count_x_col] + 1
        else:
            plot_df["obs_plot"] = plot_df[count_x_col]
        plot_df = plot_df[
            (plot_df["obs_plot"] >= obs_min) & (plot_df["obs_plot"] <= obs_max)
        ]
        if plot_df.empty:
            continue
        tmp = plot_df[["obs_plot", "mean_activity_pos"]].copy()
        tmp["dataset"] = dataset
        count_dfs.append(tmp)

    if count_dfs:
        count_long = pd.concat(count_dfs, ignore_index=True)
        dataset_palette = {
            ds: cb_palette[i] for i, ds in enumerate(reversed(list(available_datasets)))
        }
        sns.lineplot(
            data=count_long,
            x="obs_plot",
            y="mean_activity_pos",
            hue="dataset",
            palette=dataset_palette,
            errorbar=ERROR_STYLE,
            ax=ax_count,
        )
        ax_count.legend(frameon=False)
        ax_count.get_legend().set_title("Task")
        ax_count.set_xlabel("Observation number")
        ax_count.set_xticks(range(0, 31, 5))
        sns.despine(ax=ax_count, top=True, right=True)
    else:
        ax_count.set_visible(False)

    if "yoo" in available_datasets:
        plot_df_w = activities_data["yoo"]
        if (
            plot_df_w is not None
            and "mean_activity_weight_on" in plot_df_w.columns
            and "lambda_" in plot_df_w.columns
        ):
            obs_min, obs_max = OBS_RANGE["yoo"]
            plot_df_w = plot_df_w[
                (plot_df_w["observation"] >= obs_min)
                & (plot_df_w["observation"] <= obs_max)
            ].copy()
            lambdas_sorted = plot_df_w.groupby("pid")["lambda_"].first().sort_values()
            low_pids = lambdas_sorted.index[:LAMBDA_N].tolist()
            high_pids = lambdas_sorted.index[-LAMBDA_N:].tolist()
            low_thresh_lambda = lambdas_sorted.iloc[LAMBDA_N - 1]
            high_thresh_lambda = lambdas_sorted.iloc[-LAMBDA_N]

            low_df = plot_df_w[plot_df_w["pid"].isin(low_pids)].copy()
            high_df = plot_df_w[plot_df_w["pid"].isin(high_pids)].copy()

            low_label = f"low (λ<{low_thresh_lambda:.2f}, n={LAMBDA_N})"
            high_label = f"high (λ>{high_thresh_lambda:.2f}, n={LAMBDA_N})"
            low_df["lambda_group"] = low_label
            high_df["lambda_group"] = high_label

            plot_df_w_filtered = pd.concat([low_df, high_df], ignore_index=True)
            lambda_palette = {low_label: cb_palette[0], high_label: cb_palette[1]}
            sns.lineplot(
                data=plot_df_w_filtered,
                x="observation",
                y="mean_activity_weight_on",
                hue="lambda_group",
                palette=lambda_palette,
                errorbar=ERROR_STYLE,
                ax=ax_weight,
            )
            ax_weight.legend(frameon=False)
            ax_weight.get_legend().set_title("Temporal discounting")
            ax_weight.set_xlabel("Observation number")
            ax_weight.set_xticks(range(0, 31, 5))
            sns.despine(ax=ax_weight, top=True, right=True)
        else:
            ax_weight.set_visible(False)
    else:
        ax_weight.set_visible(False)

    if "jiang" in available_datasets:
        plot_df_rd = activities_data["jiang"]
        if (
            plot_df_rd is not None
            and "mean_activity_weight_on" in plot_df_rd.columns
            and "rd_true" in plot_df_rd.columns
        ):
            plot_df_rd = plot_df_rd[plot_df_rd["stage"].isin([1, 2, 3])].copy()
            rd_bin_edges = np.quantile(
                plot_df_rd["rd_true"].dropna(), np.linspace(0, 1, 11)
            )
            plot_df_rd["rd_bin"] = pd.cut(
                plot_df_rd["rd_true"], bins=rd_bin_edges, include_lowest=True
            ).apply(lambda b: b.mid if pd.notna(b) else np.nan)
            plot_df_rd = plot_df_rd.dropna(subset=["rd_bin", "mean_activity_weight_on"])
            plot_df_rd["stage"] = plot_df_rd["stage"].astype(str)
            stage_palette = {str(s): cb_palette[i] for i, s in enumerate([1, 2, 3])}
            sns.lineplot(
                data=plot_df_rd,
                x="rd_bin",
                y="mean_activity_weight_on",
                hue="stage",
                palette=stage_palette,
                errorbar=ERROR_STYLE,
                ax=ax_rd,
            )
            ax_rd.legend(frameon=False)
            ax_rd.get_legend().set_title("Jiang task stage")
            ax_rd.set_xlabel("Neighbor network degree")
            ax_rd.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            sns.despine(ax=ax_rd, top=True, right=True)
        else:
            ax_rd.set_visible(False)
    else:
        ax_rd.set_visible(False)

    if timecourse_df is not None:
        alpha_palette = {"low": cb_palette[0], "high": cb_palette[1]}
        sns.lineplot(
            data=timecourse_df,
            x="t",
            y="activity",
            hue="alpha_group",
            palette=alpha_palette,
            errorbar=ERROR_STYLE,
            ax=ax_time,
        )
        if sorted_alpha is not None and len(sorted_alpha) >= ALPHA_N:
            low_thresh = sorted_alpha.iloc[ALPHA_N - 1]["alpha_0"]
            high_thresh = sorted_alpha.iloc[-ALPHA_N]["alpha_0"]
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    color=cb_palette[0],
                    linewidth=1.5,
                    label=f"low (α₀<{low_thresh:.2f}, n={ALPHA_N})",
                ),
                Line2D(
                    [0],
                    [0],
                    color=cb_palette[1],
                    linewidth=1.5,
                    label=f"high (α₀>{high_thresh:.2f}, n={ALPHA_N})",
                ),
            ]
            ax_time.legend(
                handles=legend_handles,
                title="Base Learning Rate",
                frameon=False,
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
    ax_pe.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fname = "neural_activities"
    plt.savefig(FIGURES_DIR / f"{fname}.png", dpi=300)
    plt.savefig(FIGURES_DIR / f"{fname}.pdf")
    print(f"Saved figures/{fname}.{{png,pdf}}")


if __name__ == "__main__":
    main()
