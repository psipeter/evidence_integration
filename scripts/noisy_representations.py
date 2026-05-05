#!/usr/bin/env python3
"""Run noisy representation experiments for NEF carrabin fits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, RUNS_DIR, data_path
from utils.plot_style import FIGURE_SIZE, apply_style

SAMPLE_PIDS = [6, 7]  # high/low alpha_0 example pids for panel 1
SAMPLE_QID = None  # auto-select qid with most repeats
MIN_REPEATS = 10  # minimum trial repeats per qid for analysis
READOUT_OFFSET = 0.5  # seconds into obs window for readout
N_NEURONS_LIST = [50, 75, 100, 150, 200, 300, 500]


def _load_probe_metrics(pids, out_dir, human, nef_resp, nef_params, qid_map):
    """
    Returns DataFrame with columns:
    pid, alpha_0, lambda_, mean_error1, mean_std1, mean_cv1, response_noise
    """
    from collections import defaultdict

    from fitting.losses import _mean_qid_std

    rows = []
    for pid in pids:
        probe_path = out_dir / f"probe_NEF_recurrent_carrabin_{pid}.pkl"
        if not probe_path.exists():
            continue
        probes_raw = pd.read_pickle(probe_path)
        probes = probes_raw if isinstance(probes_raw, list) else [probes_raw]
        params = probes[0]["params"]
        t_iti = float(params["t_iti"])
        t_step = float(params["t_obs"]) + t_iti

        human_pid = human[human["pid"] == pid]
        trial_qid = (
            human_pid.groupby(["trial", "observation"])["qid"].first().reset_index()
        )
        n_obs = int(human_pid["observation"].max())

        qid_obs_vals = defaultdict(list)
        qid_neuron_vals = defaultdict(list)
        for probe in probes:
            trial = int(probe["trial"])
            t = probe["t"]
            error1 = np.abs(probe["error"][:, 1])
            error_neurons = probe.get("error_neurons")
            trial_map = trial_qid[trial_qid["trial"] == trial].set_index(
                "observation"
            )["qid"]
            for n in range(1, n_obs + 1):
                if n not in trial_map.index:
                    continue
                qid = trial_map[n]
                t_readout = t_iti + (n - 1) * t_step + READOUT_OFFSET
                idx = int(np.argmin(np.abs(t - t_readout)))
                qid_obs_vals[(qid, n)].append(float(error1[idx]))
                if error_neurons is not None and (n - 1) < len(error_neurons):
                    qid_neuron_vals[(qid, n)].append(error_neurons[n - 1])

        cvs, means, stds = [], [], []
        for vals in qid_obs_vals.values():
            if len(vals) < MIN_REPEATS:
                continue
            m, s = np.mean(vals), np.std(vals)
            if m > 0:
                cvs.append(s / m)
                means.append(m)
                stds.append(s)

        if not means:
            continue

        qid_neural_stds = []
        for vals in qid_neuron_vals.values():
            if len(vals) < MIN_REPEATS:
                continue
            arr = np.stack(vals)
            qid_neural_stds.append(arr.std(axis=0).mean())
        mean_neural_std = (
            float(np.mean(qid_neural_stds)) if qid_neural_stds else float("nan")
        )

        grp = nef_resp[nef_resp["pid"] == pid]
        grp_qid = grp.merge(qid_map, on=["pid", "trial", "observation"], how="left")
        noise = _mean_qid_std(grp_qid)
        p = nef_params[nef_params["pid"] == pid].iloc[0]
        rows.append(
            {
                "pid": pid,
                "alpha_0": float(p["alpha_0"]),
                "lambda_": float(p["lambda_"]),
                "mean_error1": float(np.mean(means)),
                "mean_std1": float(np.mean(stds)),
                "mean_cv1": float(np.mean(cvs)),
                "response_noise": noise,
                "mean_neural_std": mean_neural_std,
                "_qid_obs_vals": qid_obs_vals,
                "_probes": probes,
                "_params": params,
            }
        )
    return pd.DataFrame(rows)


def _load_pred_error_std(
    out_dir: Path,
    human: pd.DataFrame,
    n_neurons_list: list[int],
    pids: list[int],
    min_repeats: int = 5,
    readout_offset: float = 0.5,
) -> pd.DataFrame:
    """
    For each (n_neurons, pid, qid), compute std of |error[1]| at readout
    timepoints across trials.
    """
    from collections import defaultdict

    rows = []
    for n_neurons in n_neurons_list:
        for pid in pids:
            probe_path = out_dir / f"probe_n{n_neurons}_carrabin_{pid}.pkl"
            if not probe_path.exists():
                continue
            probes_raw = pd.read_pickle(probe_path)
            probes = probes_raw if isinstance(probes_raw, list) else [probes_raw]
            params = probes[0]["params"]
            t_iti = float(params["t_iti"])
            t_step = float(params["t_obs"]) + t_iti

            human_pid = human[human["pid"] == pid]
            trial_qid = (
                human_pid.groupby(["trial", "observation"])["qid"].first().reset_index()
            )

            qid_vals = defaultdict(list)
            for probe in probes:
                trial = int(probe["trial"])
                t = probe["t"]
                error1 = np.abs(probe["error"][:, 1])
                trial_map = trial_qid[trial_qid["trial"] == trial].set_index(
                    "observation"
                )["qid"]
                for obs_n in range(1, 6):
                    if obs_n not in trial_map.index:
                        continue
                    qid = trial_map[obs_n]
                    t_readout = t_iti + (obs_n - 1) * t_step + readout_offset
                    idx = int(np.argmin(np.abs(t - t_readout)))
                    qid_vals[(qid, obs_n)].append(float(error1[idx]))

            for (qid, _), vals in qid_vals.items():
                if len(vals) < min_repeats:
                    continue
                rows.append(
                    {
                        "n_neurons": n_neurons,
                        "pid": pid,
                        "qid": qid,
                        "pred_error_std": float(np.std(vals)),
                    }
                )

    return pd.DataFrame(rows)


def _plot_panel1(ax, sample_pid_data, human, color_0, color_1):
    colors = [color_0, color_1]

    for pid_row, color in zip(sample_pid_data, colors):
        pid = int(pid_row["pid"])
        probes = pid_row["_probes"]
        params = pid_row["_params"]
        t_iti = float(params["t_iti"])
        t_obs = float(params["t_obs"])
        t_step = t_obs + t_iti

        # find length-1 qid with most repeats
        human_pid = human[human["pid"] == pid]
        qid_lengths = human_pid.groupby("qid")["observation"].count()
        length1_qids = set(qid_lengths[qid_lengths == 1].index)
        qid_obs_vals = pid_row["_qid_obs_vals"]

        candidates = [(k, v) for k, v in qid_obs_vals.items() if k[0] in length1_qids]
        if not candidates:
            candidates = list(qid_obs_vals.items())
        best_qid, best_obs = max(candidates, key=lambda x: len(x[1]))[0]

        matching_trials = set(
            human_pid[
                (human_pid["qid"] == best_qid) & (human_pid["observation"] == best_obs)
            ]["trial"].values
        )
        t_start = t_iti + (best_obs - 1) * t_step
        t_end = t_start + t_obs

        # plot all matching trial traces
        first = True
        for probe in probes:
            if int(probe["trial"]) not in matching_trials:
                continue
            t = probe["t"]
            error1 = np.abs(probe["error"][:, 1])
            mask = (t >= t_start) & (t <= t_end)
            label = f"pid={pid} (α₀={pid_row['alpha_0']:.2f})" if first else None
            ax.plot(
                t[mask] - t_start,
                error1[mask],
                color=color,
                linewidth=0.5,
                label=label,
            )
            first = False

    # readout line (no label — just visual reference)
    ax.axvline(READOUT_OFFSET, color='k', linewidth=1.0, linestyle="--")
    ax.set_xlabel("Time within observation (s)")
    ax.set_ylabel("|Prediction error|")
    ax.set_title("Prediction error timecourse")

    # legend: colored lines matching the traces
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            color=colors[i],
            linewidth=1.5,
            label=f"pid={int(sample_pid_data[i]['pid'])} (α₀={sample_pid_data[i]['alpha_0']:.2f})",
        )
        for i in range(len(sample_pid_data))
    ]
    ax.legend(handles=handles, frameon=False)
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel2(ax, metrics_df, color_0, color_1):
    std_max = float(np.nanmax(np.abs(metrics_df["mean_std1"])))
    noise_max = float(np.nanmax(np.abs(metrics_df["response_noise"])))
    sns.regplot(
        data=metrics_df,
        x="alpha_0",
        y="response_noise",
        scatter_kws={"color": color_0, "s": 30},
        line_kws={"color": color_0, "linewidth": 1.5},
        label="Mean response noise",
        ci=95,
        ax=ax,
    )
    sns.regplot(
        data=metrics_df,
        x="alpha_0",
        y="mean_std1",
        scatter_kws={"color": color_1, "s": 30},
        line_kws={"color": color_1, "linewidth": 1.5},
        label="Std prediction error",
        ci=95,
        ax=ax,
    )
    ax.legend(frameon=False)
    ax.set_xlabel("Fitted α₀")
    ax.set_ylabel("Value")
    ax.set_title("Fitted learning rate α₀ affects neural and response noise")
    sns.despine(ax=ax, top=True, right=True)

    from scipy.stats import pearsonr

    r_std, p_std = pearsonr(metrics_df["alpha_0"], metrics_df["mean_std1"])
    r_noise, p_noise = pearsonr(metrics_df["alpha_0"], metrics_df["response_noise"])
    print(f"Panel 2 — mean_std1 vs alpha_0: r={r_std:.3f}, p={p_std:.4f}")
    print(f"Panel 2 — response_noise vs alpha_0: r={r_noise:.3f}, p={p_noise:.4f}")


def _plot_panel3(ax, metrics_df, color_0):
    sns.regplot(
        data=metrics_df,
        x="mean_std1",
        y="response_noise",
        scatter_kws={"color": color_0, "s": 30},
        line_kws={"color": color_0, "linewidth": 1.5},
        ax=ax,
        ci=95,
    )
    ax.set_xlabel("Std prediction error")
    ax.set_ylabel("Mean response noise")
    ax.set_title("Higher prediction error noise → more response variability")
    sns.despine(ax=ax, top=True, right=True)

    from scipy.stats import pearsonr

    r, p = pearsonr(metrics_df["mean_std1"], metrics_df["response_noise"])
    print(f"Panel 3 — response_noise vs mean_std1: r={r:.3f}, p={p:.4f}")


def _plot_panel4(ax, scan_per_qid, pred_error_df, scan_pid, color_0, color_1):
    if scan_per_qid.empty:
        ax.text(
            0.5,
            0.5,
            "n_neurons scan\n(TBD)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color=color_0,
        )
        ax.set_title("Neural noise vs response noise")
        sns.despine(ax=ax, top=True, right=True)
        return

    sns.lineplot(
        data=scan_per_qid,
        x="n_neurons",
        y="response_noise",
        color=color_0,
        linewidth=2.0,
        marker="o",
        markersize=5,
        errorbar="ci",
        ax=ax,
        label="Mean response noise",
    )
    if not pred_error_df.empty:
        sns.lineplot(
            data=pred_error_df,
            x="n_neurons",
            y="pred_error_std",
            color=color_1,
            linewidth=2.0,
            marker="s",
            markersize=5,
            errorbar="ci",
            ax=ax,
            label="Std prediction error",
        )
    ax.set_xlabel("Number of neurons")
    ax.set_ylabel("Value")
    ax.set_title("More neurons → less neural and response noise")
    ax.legend(frameon=False)
    sns.despine(ax=ax, top=True, right=True)


def _sensitivity_analysis(run_folder: Path) -> None:
    from scipy.stats import pearsonr, spearmanr

    from fitting.losses import _mean_qid_std

    nef_params = pd.read_pickle(run_folder / "NEF_recurrent_carrabin_params.pkl")
    nef_resp = pd.read_pickle(run_folder / "NEF_recurrent_carrabin_responses.pkl")
    human = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map = human[["pid", "trial", "observation", "qid"]].drop_duplicates()

    rows = []
    for pid, grp in nef_resp.groupby("pid"):
        grp_qid = grp.merge(qid_map, on=["pid", "trial", "observation"], how="left")
        noise = _mean_qid_std(grp_qid)
        p = nef_params[nef_params["pid"] == pid].iloc[0]
        rows.append(
            {
                "pid": pid,
                "response_noise": noise,
                "alpha_0": float(p["alpha_0"]),
                "lambda_": float(p["lambda_"]),
            }
        )

    df = pd.DataFrame(rows).dropna()
    print(f"\nSensitivity analysis (NEF carrabin, n={len(df)} pids):")
    for col in ("alpha_0", "lambda_"):
        r, p_val = pearsonr(df[col], df["response_noise"])
        rs, ps = spearmanr(df[col], df["response_noise"])
        print(
            f"  response_noise vs {col}: "
            f"pearson r={r:.3f} (p={p_val:.4f}), "
            f"spearman r={rs:.3f} (p={ps:.4f})"
        )


def _run_probe_pids(
    pids: list[int],
    run_folder: Path,
    out_folder: str,
    run_simulation: bool,
) -> None:
    from fitting.model_params import MODEL_PARAMS
    from models.NEF import PARAM_DEFAULTS, run as nef_run

    out_dir = RUNS_DIR / out_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_pids = pids
    if run_simulation:
        for pid in sim_pids:
            params = pd.read_pickle(
                run_folder / f"NEF_recurrent_carrabin_{pid}_params.pkl"
            ).iloc[0].to_dict()
            fixed = MODEL_PARAMS["carrabin"]["NEF_recurrent"].get("fixed", {})
            params = {**PARAM_DEFAULTS, **fixed, **params}
            params["nef_type"] = "recurrent"
            params["dataset"] = "carrabin"
            params["model_type"] = "NEF_recurrent"
            print(
                f"Running pid={pid} (alpha_0={params['alpha_0']:.3f}, "
                f"lambda_={params['lambda_']:.3f})..."
            )
            nef_run(params, save_probes=True)
            src = data_path(f"probe_NEF_recurrent_carrabin_{pid}.pkl")
            dst = out_dir / f"probe_NEF_recurrent_carrabin_{pid}.pkl"
            if src.exists():
                Path(src).rename(dst)
                print(f"  Saved to {dst}")

    all_probe_files = sorted(out_dir.glob("probe_NEF_recurrent_carrabin_*.pkl"))
    analysis_pids = [int(f.stem.split("_")[-1]) for f in all_probe_files]
    print(f"\nFound probe data for {len(analysis_pids)} pids: {analysis_pids}")

    human = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map = human[["pid", "trial", "observation", "qid"]].drop_duplicates()
    nef_params = pd.read_pickle(run_folder / "NEF_recurrent_carrabin_params.pkl")
    nef_resp = pd.read_pickle(run_folder / "NEF_recurrent_carrabin_responses.pkl")
    metrics_df = _load_probe_metrics(
        analysis_pids, out_dir, human, nef_resp, nef_params, qid_map
    )
    print(f"Loaded metrics for {len(metrics_df)} pids")

    n_neurons_list = N_NEURONS_LIST
    scan_pid = 14
    scan_dfs = []
    response_files = sorted(out_dir.glob("responses_carrabin_*_n*.pkl"))
    all_scan_pids = sorted({int(f.stem.split("_")[2]) for f in response_files})
    for pid in all_scan_pids:
        for n_neurons in n_neurons_list:
            path = out_dir / f"responses_carrabin_{pid}_n{n_neurons}.pkl"
            if path.exists():
                scan_dfs.append(pd.read_pickle(path))

    if scan_dfs:
        scan_resp = pd.concat(scan_dfs, ignore_index=True)
        scan_resp = scan_resp.merge(
            qid_map,
            on=["pid", "trial", "observation"],
            how="left",
        )
        scan_per_qid = (
            scan_resp.groupby(["n_neurons", "pid", "qid"])["response"]
            .std()
            .reset_index()
        )
        scan_per_qid.columns = ["n_neurons", "pid", "qid", "response_noise"]
    else:
        scan_per_qid = pd.DataFrame()

    all_scan_pids = [
        int(f.stem.split("_")[2])
        for f in sorted(out_dir.glob("responses_carrabin_*_n200.pkl"))
    ]
    pred_error_df = _load_pred_error_std(
        out_dir, human, N_NEURONS_LIST, all_scan_pids
    )

    print("\nProbe data:")
    for pid in analysis_pids:
        probe_path = out_dir / f"probe_NEF_recurrent_carrabin_{pid}.pkl"
        if probe_path.exists():
            probes_raw = pd.read_pickle(probe_path)
            n_trials = (
                len(probes_raw) if isinstance(probes_raw, list) else 1
            )
            print(f"  pid={pid}: {n_trials} trials of probe data available")

    if metrics_df.empty:
        print(
            "Skipping probe figure — no metrics (missing probes or insufficient repeats)."
        )
        return

    apply_style()
    palette = sns.color_palette("colorblind")
    color_0 = palette[0]
    color_1 = palette[1]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(FIGURE_SIZE[0] * 2.5, FIGURE_SIZE[1]),
        constrained_layout=True,
    )

    sample_rows = [
        metrics_df[metrics_df["pid"] == pid].iloc[0].to_dict()
        for pid in SAMPLE_PIDS
        if pid in metrics_df["pid"].values
    ]

    _plot_panel1(axes[0], sample_rows, human, color_0, color_1)
    _plot_panel2(axes[1], metrics_df, color_0, color_1)
    _plot_panel3(axes[2], metrics_df, color_0)
    _plot_panel4(axes[3], scan_per_qid, pred_error_df, scan_pid, color_0, color_1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "noisy_representations.png", dpi=300)
    plt.savefig(FIGURES_DIR / "noisy_representations.pdf")
    print("Saved figures/noisy_representations.{png,pdf}")
    plt.close(fig)


def _run_n_neurons_scan(
    scan_pids: list[int],
    n_neurons_list: list[int],
    run_folder: Path,
    out_folder: str,
    run_simulation: bool,
) -> None:
    from fitting.losses import _mean_qid_std
    from fitting.model_params import MODEL_PARAMS
    from models.NEF import PARAM_DEFAULTS, run as nef_run

    out_dir = RUNS_DIR / out_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    if run_simulation:
        for pid in scan_pids:
            base_params = pd.read_pickle(
                run_folder / f"NEF_recurrent_carrabin_{pid}_params.pkl"
            ).iloc[0].to_dict()
            fixed = MODEL_PARAMS["carrabin"]["NEF_recurrent"].get("fixed", {})
            base_params = {**PARAM_DEFAULTS, **fixed, **base_params}
            base_params["nef_type"] = "recurrent"
            base_params["dataset"] = "carrabin"
            base_params["model_type"] = "NEF_recurrent"

            for n_neurons in n_neurons_list:
                print(f"Simulating pid={pid}, n_neurons={n_neurons}...")
                p = {**base_params, "n_neurons": n_neurons}
                responses = nef_run(p, save_probes=True)
                responses["n_neurons"] = n_neurons
                out_path = out_dir / f"responses_carrabin_{pid}_n{n_neurons}.pkl"
                responses.to_pickle(out_path)
                src = data_path(f"probe_NEF_recurrent_carrabin_{pid}.pkl")
                dst = out_dir / f"probe_n{n_neurons}_carrabin_{pid}.pkl"
                if src.exists():
                    src.rename(dst)
                print(f"  Saved responses and probes for pid={pid}, n={n_neurons}")

    qid_map = pd.read_pickle(data_path("carrabin.pkl"))[
        ["pid", "trial", "observation", "qid"]
    ].drop_duplicates()
    all_scan_dfs = []
    response_files = sorted(out_dir.glob("responses_carrabin_*_n*.pkl"))
    all_scan_pids = sorted({int(f.stem.split("_")[2]) for f in response_files})
    for pid in all_scan_pids:
        for n_neurons in n_neurons_list:
            path = out_dir / f"responses_carrabin_{pid}_n{n_neurons}.pkl"
            if path.exists():
                all_scan_dfs.append(pd.read_pickle(path))

    if all_scan_dfs:
        scan_resp = pd.concat(all_scan_dfs, ignore_index=True)
        scan_resp = scan_resp.merge(
            qid_map, on=["pid", "trial", "observation"], how="left"
        )
        scan_per_qid = (
            scan_resp.groupby(["n_neurons", "pid", "qid"])["response"]
            .std()
            .reset_index()
        )
        scan_per_qid.columns = ["n_neurons", "pid", "qid", "response_noise"]
        summary = (
            scan_per_qid.groupby("n_neurons")["response_noise"]
            .mean()
            .reset_index()
            .sort_values("n_neurons")
        )
        print("\nn_neurons scan mean response noise across pids/qids:")
        for _, row in summary.iterrows():
            print(f"  n_neurons={int(row['n_neurons'])}: {row['response_noise']:.4f}")
    else:
        print("\nn_neurons scan analysis TBD.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="probe_pids",
        choices=["probe_pids", "n_neurons_scan"],
        help="Which experiment to run",
    )
    parser.add_argument("--run_simulation", action="store_true", default=False)
    parser.add_argument(
        "--run_folder",
        type=str,
        default="response",
        help="Source folder for fitted NEF params",
    )
    parser.add_argument("--out_folder", type=str, default="noisy_representations")
    parser.add_argument(
        "--pids",
        type=int,
        nargs="+",
        default=[6, 7],
        help="PIDs to simulate for probe_pids experiment",
    )
    parser.add_argument(
        "--scan_pid",
        type=int,
        default=None,
        help="Single PID alias for --scan_pids",
    )
    parser.add_argument(
        "--scan_pids",
        type=int,
        nargs="+",
        default=[14],
        help="PIDs to use for n_neurons_scan (default: [14])",
    )
    parser.add_argument(
        "--n_neurons_list",
        type=int,
        nargs="+",
        default=[50, 75, 100, 150, 200, 300, 500],
    )
    args = parser.parse_args()

    run_folder = RUNS_DIR / args.run_folder
    out_folder = args.out_folder
    if args.scan_pid is not None:
        args.scan_pids = [args.scan_pid]

    _sensitivity_analysis(run_folder)

    if args.experiment == "probe_pids":
        _run_probe_pids(args.pids, run_folder, out_folder, args.run_simulation)
    elif args.experiment == "n_neurons_scan":
        _run_n_neurons_scan(
            args.scan_pids,
            args.n_neurons_list,
            run_folder,
            out_folder,
            args.run_simulation,
        )


if __name__ == "__main__":
    main()
