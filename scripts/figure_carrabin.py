#!/usr/bin/env python3
"""Carrabin summary figure: row 1 (panels A–D), row 2 = noisy representations (E–H)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, RUNS_DIR, data_path
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette, label_panels


MODEL_ORDER_B = ["Bayes", "RL", "NoisyCounting", "NEF_recurrent"]
MODEL_ORDER_D = ["Human", "Bayes", "RL", "NoisyCounting", "NEF_recurrent"]

# --- noisy representations row (copied from scripts/noisy_representations.py) ---
SAMPLE_PIDS = [6, 7]  # high/low alpha_0 example pids for panel 1
MIN_REPEATS = 10  # minimum trial repeats per qid for analysis
READOUT_OFFSET = 0.5  # seconds into obs window for readout
N_NEURONS_LIST = [50, 75, 100, 150, 200, 300, 500]


def _display(model_type: str) -> str:
    return "NEF" if model_type.startswith("NEF") else model_type


def _placeholder(ax, text: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="0.5",
        style="italic",
    )


def _empty_pdf_panel(ax) -> None:
    """Panels A/C: bare axes matching embedded-PDF styling, no decorative text."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
    ax.set_anchor("C")


def _plot_panel_a(ax) -> None:
    """Render first page of figures/carrabin_task.pdf into panel A."""
    pdf_path = FIGURES_DIR / "carrabin_task.pdf"
    if not pdf_path.exists():
        _empty_pdf_panel(ax)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / "carrabin_task"
        cmd = [
            "pdftoppm",
            "-png",
            "-singlefile",
            str(pdf_path),
            str(out_prefix),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            _empty_pdf_panel(ax)
            return
        img_path = out_prefix.with_suffix(".png")
        if not img_path.exists():
            _empty_pdf_panel(ax)
            return
        img = mpimg.imread(img_path)

    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Preserve aspect ratio while maximizing panel coverage.
    ax.set_aspect("equal")
    ax.set_anchor("C")


def _plot_panel_c(ax) -> None:
    """Render first page of figures/response_noise_schematic.pdf into panel C."""
    pdf_path = FIGURES_DIR / "response_noise_schematic.pdf"
    if not pdf_path.exists():
        _empty_pdf_panel(ax)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / "response_noise_schematic"
        cmd = [
            "pdftoppm",
            "-png",
            "-singlefile",
            str(pdf_path),
            str(out_prefix),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            _empty_pdf_panel(ax)
            return
        img_path = out_prefix.with_suffix(".png")
        if not img_path.exists():
            _empty_pdf_panel(ax)
            return
        img = mpimg.imread(img_path)

    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
    ax.set_anchor("C")


def _get_loss(perf_df: pd.DataFrame) -> pd.Series:
    """Return response_component if available, else cv_loss_mean."""
    if "response_component" in perf_df.columns:
        rc = perf_df["response_component"]
        if rc.notna().all():
            return rc
    return perf_df["cv_loss_mean"]


def _plot_panel_b(ax, run_folder: str, palette: dict) -> None:
    run_dir = data_path("runs") / run_folder
    rows = []
    for mt in MODEL_ORDER_B:
        f = run_dir / f"{mt}_carrabin_performance.pkl"
        if not f.exists():
            continue
        perf = pd.read_pickle(f).copy()
        perf["plot_loss"] = _get_loss(perf)
        perf["model_disp"] = _display(mt)
        rows.append(perf[["pid", "model_disp", "plot_loss"]])

    if not rows:
        _placeholder(ax, "No performance data")
        return

    df = pd.concat(rows, ignore_index=True)
    order = [_display(m) for m in MODEL_ORDER_B]
    available = [m for m in order if m in set(df["model_disp"])]
    pal = {m: palette.get(m, "0.5") for m in available}
    sns.boxplot(
        data=df,
        x="model_disp",
        y="plot_loss",
        order=available,
        hue="model_disp",
        palette=pal,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Response error (trial-wise RMSE)")
    sns.despine(ax=ax, top=True, right=True)


def _mean_qid_std_per_pid(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, grp in df.groupby("pid"):
        qid_std = grp.groupby("qid")["response"].std()
        qid_std = qid_std.dropna()
        if len(qid_std) == 0:
            continue
        rows.append({"pid": int(pid), "response_noise": float(qid_std.mean())})
    return pd.DataFrame(rows)


def _load_loss_long(
    run_dir: Path,
    model_order: list[str],
    dataset: str,
) -> pd.DataFrame:
    """
    Load per-pid shape loss for each model.
    Prefers shape_component from performance files when available and
    non-NaN; falls back to recomputing via losses.shape_loss().
    Returns DataFrame with columns: pid, model_type, loss.
    """
    import fitting.losses as losses_mod

    rows = []
    human_full = pd.read_pickle(data_path(f"{dataset}.pkl"))

    for mt in model_order:
        perf_path = run_dir / f"{mt}_{dataset}_performance.pkl"
        resp_path = run_dir / f"{mt}_{dataset}_responses.pkl"
        if not perf_path.exists():
            continue
        perf = pd.read_pickle(perf_path)

        if "shape_component" in perf.columns and perf["shape_component"].notna().all():
            for _, row in perf.iterrows():
                rows.append(
                    {
                        "pid": int(row["pid"]),
                        "model_type": mt,
                        "loss": float(row["shape_component"]),
                    }
                )
            continue

        if not resp_path.exists():
            print(f"Warning: missing {resp_path.name}, cannot compute loss for {mt}")
            continue
        responses = pd.read_pickle(resp_path)
        for pid, model_pid in responses.groupby("pid"):
            human_pid = human_full[human_full["pid"] == pid]
            params = {"dataset": dataset, "pid": int(pid)}
            try:
                loss = losses_mod.shape_loss(params, model_pid, human_pid)
                rows.append({"pid": int(pid), "model_type": mt, "loss": loss})
            except Exception as e:
                print(f"Warning: shape_loss failed for {mt} pid={pid}: {e}")

    return pd.DataFrame(rows)


def _plot_panel_d(ax, run_folder: str, palette: dict) -> None:
    run_dir = data_path("runs") / run_folder
    loss_df = _load_loss_long(run_dir, MODEL_ORDER_B, "carrabin")
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

    order = [_display(m) for m in MODEL_ORDER_B]
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


def _save_panel_c_kde(human: pd.DataFrame) -> None:
    """Bottom-left KDE panel from ``scripts/response_noise_carrabin.py`` (verbatim)."""
    from fitting.losses import _mean_qid_std

    apply_style()
    PALETTE = get_palette()
    LINESTYLES = ["solid", "dashed", "dotted"]
    SAMPLE_PIDS = {"low": 14, "medium": 18, "high": 17}

    pid_stds: list[float] = []
    for _pid, grp in human.groupby("pid"):
        pid_stds.append(_mean_qid_std(grp))

    pid_stds_vals = [s for s in pid_stds if np.isfinite(s)]
    fig, ax_kde = plt.subplots(figsize=(3, 3))
    if pid_stds_vals:
        sns.kdeplot(
            pid_stds_vals, ax=ax_kde, color=PALETTE["Human"], fill=True, alpha=0.3
        )
        kde_fn = gaussian_kde(pid_stds_vals)
        for i, (_, pid) in enumerate(SAMPLE_PIDS.items()):
            std_val = _mean_qid_std(human[human["pid"] == pid])
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
    from fitting.losses import QID_MIN_TRIALS

    apply_style()
    PALETTE = get_palette()
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


def _plot_panel2(ax, metrics_df, color_0, color_1):
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
    # ax.set_title("Prediction error noise →\nresponse variability")
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


def _load_noisy_representations_figure_data(
    run_folder: Path,
    out_folder: str,
    n_neurons_list: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assemble probe / scan data as in noisy_representations._run_probe_pids."""
    out_dir = RUNS_DIR / out_folder
    all_probe_files = sorted(out_dir.glob("probe_NEF_recurrent_carrabin_*.pkl"))
    analysis_pids = [int(f.stem.split("_")[-1]) for f in all_probe_files]

    human = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map = human[["pid", "trial", "observation", "qid"]].drop_duplicates()
    nef_params_path = run_folder / "NEF_recurrent_carrabin_params.pkl"
    nef_resp_path = run_folder / "NEF_recurrent_carrabin_responses.pkl"
    if not nef_params_path.exists() or not nef_resp_path.exists():
        return pd.DataFrame(), human, pd.DataFrame(), pd.DataFrame()

    nef_params = pd.read_pickle(nef_params_path)
    nef_resp = pd.read_pickle(nef_resp_path)
    metrics_df = _load_probe_metrics(
        analysis_pids, out_dir, human, nef_resp, nef_params, qid_map
    )

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
        out_dir, human, n_neurons_list, all_scan_pids
    )

    return metrics_df, human, scan_per_qid, pred_error_df


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
        default=list(N_NEURONS_LIST),
    )
    args = parser.parse_args()

    if args.scan_pid is not None:
        args.scan_pids = [args.scan_pid]

    apply_style()
    palette = get_palette()
    if "Human" not in palette:
        palette["Human"] = "black"

    fig, axes = plt.subplots(2, 4, figsize=FIGURE_SIZE, constrained_layout=True)
    row0, row1 = axes[0], axes[1]

    _plot_panel_a(row0[0])
    _plot_panel_b(row0[1], args.run_folder, palette)
    _plot_panel_c(row0[2])
    _plot_panel_d(row0[3], args.run_folder, palette)

    palette_cb = sns.color_palette("colorblind")
    color_0, color_1 = palette_cb[0], palette_cb[1]
    metrics_df, human, scan_per_qid, pred_error_df = (
        _load_noisy_representations_figure_data(
            RUNS_DIR / args.run_folder,
            args.out_folder,
            list(args.n_neurons_list),
        )
    )

    if metrics_df.empty:
        for ax in row1:
            _placeholder(
                ax,
                "No probe data — generate with scripts/noisy_representations.py "
                "(matching --run_folder/--out_folder).",
            )
    else:
        sample_rows = [
            metrics_df[metrics_df["pid"] == pid].iloc[0].to_dict()
            for pid in SAMPLE_PIDS
            if pid in metrics_df["pid"].values
        ]
        scan_pid_plot = args.scan_pids[0] if args.scan_pids else 14
        _plot_panel1(row1[0], sample_rows, human, color_0, color_1)
        _plot_panel2(row1[1], metrics_df, color_0, color_1)
        _plot_panel3(row1[2], metrics_df, color_0)
        _plot_panel4(
            row1[3],
            scan_per_qid,
            pred_error_df,
            scan_pid_plot,
            color_0,
            color_1,
        )

    label_panels(axes)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "figure_carrabin.png", dpi=300)
    plt.savefig(FIGURES_DIR / "figure_carrabin.pdf")
    plt.savefig(FIGURES_DIR / "figure_carrabin.svg")
    print("Saved figures/figure_carrabin.{png,pdf,svg}")
    _save_panel_c_kde(human)
    _save_qid_kde(human)


if __name__ == "__main__":
    main()
