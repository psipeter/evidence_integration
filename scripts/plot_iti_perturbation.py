#!/usr/bin/env python3
"""Plot figures for ITI perturbation (probe timecourses + noise amplitude scan summary)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, RUNS_DIR, data_path
from utils.plot_style import FIGURE_SIZE, apply_style

from fitting.losses import QID_MIN_TRIALS

CONDITION_LABELS = {
    "no_noise": "No noise",
    "amp0p1": "Noise 0.1",
    "amp0p05": "Noise 0.05",
}


def _trial_is_two_plus_ones_after_obs2(
    trial: int, trial_qid: pd.DataFrame
) -> bool:
    """True if obs 1 and 2 exist and the qid after obs 2 starts with '11' (two +1s).

    Probes do not store qids; this must match ``carrabin.pkl`` semantics: the ``qid``
    string grows with cumulative evidence (+1 ↦ '1', etc.), so ``'11'`` appears at
    observation 2, not at observation 1 (where qid is ``'1'``).
    """
    tq = trial_qid[trial_qid["trial"] == trial].set_index("observation")["qid"]
    if 1 not in tq.index or 2 not in tq.index:
        return False
    return str(tq.loc[2]).startswith("11")


def _plot_panel1(ax, out_dir, pid, colors, human):
    human_pid = human[human["pid"] == pid]
    trial_qid = human_pid.groupby(["trial", "observation"])["qid"].first().reset_index()

    for i, (label, display) in enumerate(CONDITION_LABELS.items()):
        probe_path = out_dir / f"probe_panel1_{pid}_{label}.pkl"
        if not probe_path.exists():
            print(f"Warning: missing {probe_path.name}")
            continue
        probes = pd.read_pickle(probe_path)
        params = probes[0]["params"]
        t_obs = float(params["t_obs"])
        t_iti = float(params["t_iti"])

        rows = []
        for probe in probes:
            trial = int(probe["trial"])
            if not _trial_is_two_plus_ones_after_obs2(trial, trial_qid):
                continue
            t = probe["t"]
            value = np.abs(probe["value"])
            t_start = t_iti
            # Stop before ITI #2 (noise spikes again after obs 2); exclusive upper edge
            t_before_iti2 = t_start + t_obs + t_iti + t_obs
            mask = (t >= t_start) & (t < t_before_iti2)
            t_rel = t[mask] - t_start
            for t_val, v_val in zip(t_rel, value[mask]):
                rows.append({"t": float(t_val), "value": float(v_val)})

        if not rows:
            continue

        df = pd.DataFrame(rows)
        # downsample t to reduce plot density (every 10th timepoint)
        t_vals = sorted(df["t"].unique())
        t_keep = t_vals[::10]
        df = df[df["t"].isin(t_keep)]

        sns.lineplot(
            data=df,
            x="t",
            y="value",
            color=colors[i],
            linewidth=1.5,
            errorbar="ci",
            label=display,
            ax=ax,
        )

    no_noise_path = out_dir / f"probe_panel1_{pid}_no_noise.pkl"
    if no_noise_path.exists():
        ref_params = pd.read_pickle(no_noise_path)[0]["params"]
        t_obs_ref = float(ref_params["t_obs"])
        t_iti_ref = float(ref_params["t_iti"])
        ax.axvspan(
            t_obs_ref,
            t_obs_ref + t_iti_ref,
            alpha=0.12,
            color="gray",
            linewidth=0,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Decoded value estimate")
    handles, labels_ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels_, frameon=False)
    sns.despine(ax=ax, top=True, right=True)


def _plot_noise_envelope(ax, out_dir: Path, pid: int, *, title: str) -> None:
    t_obs_ref = None
    t_iti_ref = None
    y_chunks: list[np.ndarray] = []
    for i, (label, _) in enumerate(CONDITION_LABELS.items()):
        probe_path = out_dir / f"probe_panel1_{pid}_{label}.pkl"
        if not probe_path.exists():
            continue
        probes = pd.read_pickle(probe_path)
        probe = probes[0]
        if "iti_noise" not in probe:
            continue
        params = probe["params"]
        t_obs = float(params["t_obs"])
        t_iti = float(params["t_iti"])
        if t_obs_ref is None:
            t_obs_ref = t_obs
            t_iti_ref = t_iti
        t = probe["t"]
        noise = probe["iti_noise"]
        t_start = t_iti
        t_before_iti2 = t_start + t_obs + t_iti + t_obs
        mask = (t >= t_start) & (t < t_before_iti2)
        t_rel = t[mask] - t_start
        y = noise[mask]
        if y.size:
            y_chunks.append(np.asarray(y, dtype=float).ravel())
        ax.plot(
            t_rel,
            y,
            color="0.12",
            linewidth=0.95,
            alpha=0.55 + 0.15 * i,
            zorder=5 + i,
        )
    if t_obs_ref is not None:
        ax.axvspan(t_obs_ref, t_obs_ref + t_iti_ref,
                   alpha=0.12, color="gray", linewidth=0)

    if y_chunks:
        y_all = np.concatenate(y_chunks)
        y_lo = float(np.nanmin(y_all))
        y_hi = float(np.nanmax(y_all))
        span = y_hi - y_lo
        pad = 0.08 * span if span > 0 else max(0.02 * max(abs(y_lo), abs(y_hi)), 1e-6)
        ax.set_ylim(y_lo - pad, y_hi + pad)

    ax.set_title(title)
    ax.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_ylabel("Noise", fontsize=7, labelpad=2)
    sns.despine(ax=ax, top=True, right=True, bottom=True, left=False)


def _response_noise_rows_per_qid(
    resp: pd.DataFrame, qid_map: pd.DataFrame, amp: float, pid: int
) -> list[dict]:
    """One row per qid: std of model responses for that qid (same filter as fitting loss)."""
    grp = resp.merge(qid_map, on=["pid", "trial", "observation"], how="left")
    counts = grp.groupby("qid")["trial"].nunique()
    valid_qids = counts[counts >= QID_MIN_TRIALS].index
    if len(valid_qids) == 0:
        return []
    sub = grp[grp["qid"].isin(valid_qids)]
    stds = sub.groupby("qid")["response"].std()
    return [
        {
            "iti_noise_amplitude": amp,
            "pid": pid,
            "qid": qid,
            "response_noise_qid": float(s),
        }
        for qid, s in stds.items()
    ]


def _plot_panel2(ax, out_dir: Path, qid_map: pd.DataFrame) -> None:
    """Per-qid response std vs amplitude (mean ± default Seaborn CI across qids & pids)."""
    rows_qid: list[dict] = []
    for path in sorted(out_dir.glob("noise_scan_*.pkl")):
        resp = pd.read_pickle(path)
        if resp.empty:
            continue
        amp = float(resp["iti_noise_amplitude"].iloc[0])
        pid = int(resp["pid"].iloc[0])
        rows_qid.extend(_response_noise_rows_per_qid(resp, qid_map, amp, pid))

    if not rows_qid:
        ax.text(
            0.5,
            0.5,
            "No noise_scan data in out_folder.\nRun noise_scan for one or more pids.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="gray",
        )
        ax.set_title("Response noise vs noise amplitude")
        sns.despine(ax=ax, top=True, right=True)
        return

    palette = sns.color_palette("colorblind")
    df_q = pd.DataFrame(rows_qid)
    sns.lineplot(
        data=df_q,
        x="iti_noise_amplitude",
        y="response_noise_qid",
        marker="o",
        ax=ax,
        color=palette[0],
        # label="Per-qid response std (mean ± 95% CI)",
    )
    ax.set_xlabel("ITI noise amplitude")
    ax.set_ylabel("Response Noise")
    ax.set_title("ITI noise increases response noise")
    ax.legend(frameon=False, loc="best")
    sns.despine(ax=ax, top=True, right=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, default="response")
    parser.add_argument("--out_folder", type=str, default="iti_perturbation")
    parser.add_argument("--pid", type=int, default=14)
    args = parser.parse_args()

    out_dir = RUNS_DIR / args.out_folder
    human = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map = human[
        ["pid", "trial", "observation", "qid", "value"]
    ].drop_duplicates()

    apply_style()
    palette = sns.color_palette("colorblind")
    colors = [palette[0], palette[1], palette[2]]

    fig = plt.figure(
        figsize=(FIGURE_SIZE[0] * 1.7, FIGURE_SIZE[1]),
        constrained_layout=True,
    )
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1.0])
    inner_gs = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[0, 0], height_ratios=[1, 4], hspace=0.05
    )
    ax_noise = fig.add_subplot(inner_gs[0])
    ax_value = fig.add_subplot(inner_gs[1])
    ax2 = fig.add_subplot(gs[0, 1])

    _plot_noise_envelope(
        ax_noise,
        out_dir,
        args.pid,
        title="ITI noise disrupts value representation",
    )
    _plot_panel1(ax_value, out_dir, args.pid, colors, human)
    ax_noise.sharex(ax_value)

    no_noise_path = out_dir / f"probe_panel1_{args.pid}_no_noise.pkl"
    if no_noise_path.exists():
        ref = pd.read_pickle(no_noise_path)[0]["params"]
        t_obs_r = float(ref["t_obs"])
        t_iti_r = float(ref["t_iti"])
        ax_value.set_xlim(0.0, t_obs_r + t_iti_r + t_obs_r)

    _plot_panel2(ax2, out_dir, qid_map)

    fig.align_titles()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "iti_perturbation.png", dpi=300)
    plt.savefig(FIGURES_DIR / "iti_perturbation.pdf")
    print("Saved figures/iti_perturbation.{png,pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
