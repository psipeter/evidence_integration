#!/usr/bin/env python3
"""Plot figures for ITI perturbation experiments (probe timecourses + scan stubs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, RUNS_DIR, data_path
from utils.plot_style import FIGURE_SIZE, apply_style

CONDITION_LABELS = {
    "no_noise": "No noise",
    "med_noise": "Medium noise (ITI)",
    "long_iti": "Small noise, long ITI",
}


def _plot_panel1(ax, out_dir, pid, colors, human):
    human_pid = human[human["pid"] == pid]
    all_qids = human_pid["qid"].unique()
    target_qids = {q for q in all_qids if str(q).startswith("11")}
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
            tmap = trial_qid[trial_qid["trial"] == trial].set_index("observation")[
                "qid"
            ]
            idx_set = set(tmap.index)
            for obs_n, qid in tmap.items():
                if qid not in target_qids:
                    continue
                if obs_n != 1:
                    continue
                if 2 not in idx_set:
                    continue
                t = probe["t"]
                value = np.abs(probe["value"])
                t_start = t_iti
                t_end = t_start + t_obs + t_iti + t_obs
                mask = (t >= t_start) & (t <= t_end)
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

    ax.set_xlabel("Time from obs 1 onset (s)")
    ax.set_ylabel("|Value estimate|")
    ax.set_title("ITI noise disrupts value representation")
    ax.legend(frameon=False)
    sns.despine(ax=ax, top=True, right=True)


def _plot_noise_envelope(ax, out_dir, pid, colors):
    t_obs_ref = None
    t_iti_ref = None
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
        t_end = t_start + t_obs + t_iti + t_obs
        mask = (t >= t_start) & (t <= t_end)
        t_rel = t[mask] - t_start
        ax.plot(t_rel, noise[mask], color=colors[i], linewidth=0.8, alpha=0.9)
    if t_obs_ref is not None:
        ax.axvspan(t_obs_ref, t_obs_ref + t_iti_ref,
                   alpha=0.12, color="gray", linewidth=0)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("Noise\ninput", fontsize=7, labelpad=2)
    sns.despine(ax=ax, top=True, right=True, bottom=True, left=True)


def _plot_panel2(ax):
    ax.text(
        0.5,
        0.5,
        "Noise amplitude scan\n(TBD)",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        color="gray",
    )
    ax.set_title("Response noise vs noise amplitude")
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel3(ax):
    ax.text(
        0.5,
        0.5,
        "ITI duration scan\n(TBD)",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        color="gray",
    )
    ax.set_title("Response noise vs ITI duration")
    sns.despine(ax=ax, top=True, right=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, default="response")
    parser.add_argument("--out_folder", type=str, default="iti_perturbation")
    parser.add_argument("--pid", type=int, default=14)
    args = parser.parse_args()

    out_dir = RUNS_DIR / args.out_folder
    human = pd.read_pickle(data_path("carrabin.pkl"))

    apply_style()
    palette = sns.color_palette("colorblind")
    colors = [palette[0], palette[1], palette[2]]

    fig = plt.figure(
        figsize=(FIGURE_SIZE[0] * 2.0, FIGURE_SIZE[1]),
        constrained_layout=True,
    )
    gs = GridSpec(1, 3, figure=fig)
    inner_gs = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[0, 0], height_ratios=[1, 4], hspace=0.05
    )
    ax_noise = fig.add_subplot(inner_gs[0])
    ax_value = fig.add_subplot(inner_gs[1])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    _plot_noise_envelope(ax_noise, out_dir, args.pid, colors)
    _plot_panel1(ax_value, out_dir, args.pid, colors, human)
    ax_noise.sharex(ax_value)

    _plot_panel2(ax2)
    _plot_panel3(ax3)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "iti_perturbation.png", dpi=300)
    plt.savefig(FIGURES_DIR / "iti_perturbation.pdf")
    print("Saved figures/iti_perturbation.{png,pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
