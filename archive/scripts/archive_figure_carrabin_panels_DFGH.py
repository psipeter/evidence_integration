"""
scripts/archive_figure_carrabin_panels_DFGH.py

Archived panels D–H from figure_carrabin.py (old response-noise metric).
These used qid-based response noise (mean_qid_std) and NEF probe data.
Replaced by RNN-based sigma noise metric in the new panel D.

Archived: see git history or this file for full panel implementations.
"""

# ── Original imports needed by these panels ───────────────────────────────────
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils.paths import data_path, RUNS_DIR
from utils.plot_style import mean_qid_std

MIN_REPEATS  = 5
READOUT_OFFSET = 0.5

def _plot_panel_d(ax, run_folder: str, palette: dict, model_order: list[str]) -> None:
    run_dir = data_path("runs") / run_folder
    loss_df = _load_loss_long(run_dir, model_order, "carrabin")
    if loss_df.empty:
        _placeholder(ax, "No model data")
        return

    loss_df["model_disp"] = loss_df["model_type"].apply(_display)
    models_with_loss = loss_df["model_disp"].unique().tolist()
    df = loss_df.groupby("pid").filter(
        lambda g: len(g) == len(models_with_loss)
    ).copy()
    if df.empty:
        _placeholder(ax, "No model data")
        return

    order = [_display(m) for m in model_order]
    available = [m for m in order if m in set(df["model_disp"])]
    pal = {m: palette.get(m, "0.5") for m in available}
    sns.boxplot(
        data=df,
        x="model_disp",
        y="loss",
        order=available,
        hue="model_disp",
        palette=pal,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Response noise difference (sequence-wise)")
    sns.despine(ax=ax, top=True, right=True)


def _load_probe_metrics(pids, probes_list, human, nef_resp, nef_params, qid_map):
    """
    Returns DataFrame with columns:
    pid, alpha_0, lambda_, mean_error1, mean_std1, mean_cv1, response_noise
    """
    from collections import defaultdict

    from utils.plot_style import mean_qid_std

    rows = []
    for pid in pids:
        probes = [p for p in probes_list if int(p["pid"]) == pid]
        if not probes:
            continue
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
        param_rows = nef_params[nef_params["pid"] == pid]
        if grp.empty or param_rows.empty:
            continue
        noise = mean_qid_std(grp_qid)
        p = param_rows.iloc[0]
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


def _load_noisy_representations_figure_data(
    run_folder: Path,
    out_folder: str,
    n_neurons_list: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load combined extras_carrabin outputs from data/runs/<out_folder>/."""
    out_dir = RUNS_DIR / out_folder
    human = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map = human[["pid", "trial", "observation", "qid"]].drop_duplicates()

    nef_params_path = run_folder / "NEF_carrabin_params.pkl"
    nef_resp_path = run_folder / "NEF_carrabin_responses.pkl"
    if not nef_params_path.exists() or not nef_resp_path.exists():
        return pd.DataFrame(), human, pd.DataFrame(), pd.DataFrame()

    nef_params = pd.read_pickle(nef_params_path)
    nef_resp = pd.read_pickle(nef_resp_path)

    probe_combined = out_dir / "probe_pids_carrabin.pkl"
    if probe_combined.exists():
        all_probes = pd.read_pickle(probe_combined)
        analysis_pids = sorted({int(p["pid"]) for p in all_probes})
    else:
        all_probes = []
        analysis_pids = []

    metrics_df = _load_probe_metrics(
        analysis_pids, all_probes, human, nef_resp, nef_params, qid_map
    )

    scan_resp_path = out_dir / "scan_responses_carrabin.pkl"
    if scan_resp_path.exists():
        scan_resp = pd.read_pickle(scan_resp_path)
        scan_resp = scan_resp.merge(
            qid_map, on=["pid", "trial", "observation"], how="left"
        )
        scan_per_qid = (
            scan_resp.groupby(["n_neurons", "pid", "qid"])["response"]
            .std()
            .reset_index()
        )
        scan_per_qid.columns = ["n_neurons", "pid", "qid", "response_noise"]
        pred_error_df = (
            scan_resp.groupby(["n_neurons", "pid", "qid"])["abs_pred_error"]
            .std()
            .reset_index()
        )
        pred_error_df.columns = ["n_neurons", "pid", "qid", "pred_error_std"]
    else:
        scan_per_qid = pd.DataFrame()
        pred_error_df = pd.DataFrame()

    return metrics_df, human, scan_per_qid, pred_error_df


def _plot_panel_e(ax, sample_pid_data, human, color_0, color_1):
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
    ax.axvline(READOUT_OFFSET, color="k", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Time within observation (s)")
    ax.set_ylabel("Decoded prediction error")
    # ax.set_title("Prediction error timecourse")

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


def _plot_panel_f(ax, metrics_df, color_0, color_1):
    std_max = float(np.nanmax(np.abs(metrics_df["mean_std1"])))
    noise_max = float(np.nanmax(np.abs(metrics_df["response_noise"])))
    sns.regplot(
        data=metrics_df,
        x="alpha_0",
        y="response_noise",
        scatter_kws={"color": color_0, "s": 30},
        line_kws={"color": color_0, "linewidth": 1.5},
        # label="Mean response noise",
        ci=95,
        ax=ax,
    )
    # sns.regplot(
    #     data=metrics_df,
    #     x="alpha_0",
    #     y="mean_std1",
    #     scatter_kws={"color": color_1, "s": 30},
    #     line_kws={"color": color_1, "linewidth": 1.5},
    #     label="Std prediction error",
    #     ci=95,
    #     ax=ax,
    # )
    ax.legend(frameon=False)
    ax.set_xlabel("Fitted α₀")
    # ax.set_ylabel("Value")
    ax.set_ylabel("Mean response noise")
    # ax.set_title("Fitted learning rate \n affects neural and response noise")
    sns.despine(ax=ax, top=True, right=True)

    from scipy.stats import pearsonr

    r_std, p_std = pearsonr(metrics_df["alpha_0"], metrics_df["mean_std1"])
    r_noise, p_noise = pearsonr(metrics_df["alpha_0"], metrics_df["response_noise"])
    print(f"Panel 2 — mean_std1 vs alpha_0: r={r_std:.3f}, p={p_std:.4f}")
    print(f"Panel 2 — response_noise vs alpha_0: r={r_noise:.3f}, p={p_noise:.4f}")


def _plot_panel_g(ax, metrics_df, color_0):
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
    # ax.set_title("Prediction error noise →\nresponse variability")
    sns.despine(ax=ax, top=True, right=True)

    from scipy.stats import pearsonr

    r, p = pearsonr(metrics_df["mean_std1"], metrics_df["response_noise"])
    print(f"Panel 3 — response_noise vs mean_std1: r={r:.3f}, p={p:.4f}")


def _plot_panel_h(ax, scan_per_qid, pred_error_df, scan_pid, color_0, color_1):
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
        # ax.set_title("Neural noise vs response noise")
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
    # ax.set_title("More neurons →\nless neural and response noise")
    ax.legend(frameon=False)
    sns.despine(ax=ax, top=True, right=True)


def _save_panel_c_kde(human: pd.DataFrame) -> None:
    """Bottom-left KDE panel from ``scripts/response_noise_carrabin.py`` (verbatim)."""
    from utils.plot_style import mean_qid_std

    apply_style()
    PALETTE = {"Human": HUMAN_NEUTRAL_COLOR}
    LINESTYLES = ["solid", "dashed", "dotted"]
    SAMPLE_PIDS = {"low": 14, "medium": 18, "high": 17}

    pid_stds: list[float] = []
    for _pid, grp in human.groupby("pid"):
        pid_stds.append(mean_qid_std(grp))

    pid_stds_vals = [s for s in pid_stds if np.isfinite(s)]
    fig, ax_kde = plt.subplots(figsize=(3, 3))
    if pid_stds_vals:
        sns.kdeplot(
            pid_stds_vals, ax=ax_kde, color=PALETTE["Human"], fill=True, alpha=0.3
        )
        kde_fn = gaussian_kde(pid_stds_vals)
        for i, (_, pid) in enumerate(SAMPLE_PIDS.items()):
            std_val = mean_qid_std(human[human["pid"] == pid])
            if not np.isfinite(std_val):
                continue
            kde_height = float(kde_fn(np.array([std_val]))[0])
            ax_kde.plot(
                [std_val, std_val],
                [0, kde_height],
                color=PALETTE["Human"],
                linestyle=LINESTYLES[i],
                linewidth=1.5,
                label=f"#{pid}",
            )
        ax_kde.legend(title="Participant", frameon=False)
    ax_kde.set_xlabel("Response noise")
    ax_kde.set_ylabel("Density")
    ax_kde.set_title("Population response noise distribution")
    sns.despine(ax=ax_kde, top=True, right=True)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIGURES_DIR / "carrabin_response_noise_kde.pdf"
    out_svg = FIGURES_DIR / "carrabin_response_noise_kde.svg"
    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"Saved {out_pdf.name} and {out_svg.name}")


def _save_qid_kde(human: pd.DataFrame) -> None:
    """Per-PID KDE of trial responses demeaned within each valid qid, then pooled."""
    from utils.plot_style import QID_MIN_TRIALS

    apply_style()
    PALETTE = {"Human": HUMAN_NEUTRAL_COLOR}
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for pid in (14, 18, 17):
        human_pid = human[human["pid"] == pid]
        qid_counts = human_pid.groupby("qid")["trial"].nunique()
        valid_qids = qid_counts[qid_counts >= QID_MIN_TRIALS].index

        demeaned_pieces: list[np.ndarray] = []
        for qid in valid_qids:
            responses = (
                human_pid[human_pid["qid"] == qid]["response"].values.astype(float)
            )
            if responses.size == 0:
                continue
            m = float(responses.mean())
            demeaned_pieces.append(responses - m)
        demeaned = (
            np.concatenate(demeaned_pieces)
            if demeaned_pieces
            else np.array([], dtype=float)
        )

        fig, ax = plt.subplots(figsize=(3, 3))
        if demeaned.size >= 2:
            sns.kdeplot(
                demeaned,
                ax=ax,
                fill=True,
                alpha=0.4,
                color=PALETTE["Human"],
            )
        ax.axvline(
            0.0,
            color=PALETTE["Human"],
            linestyle="--",
            linewidth=1.0,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.tight_layout()
        stem = FIGURES_DIR / f"carrabin_demeaned_kde_pid{pid}"
        fig.savefig(stem.with_suffix(".svg"))
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)

