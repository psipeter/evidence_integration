#!/usr/bin/env python3
"""
Plot population dynamics from a single NEF.py probe run.

Usage:
    python scripts/NEF_plots.py --dataset carrabin --pid 1 --model_type NEF_recurrent
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import DATA_DIR, FIGURES_DIR, data_path
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _ideal_alpha_steps(params: dict, n_obs: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_steps, alpha_steps) as a step function for ideal alpha(n)."""
    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti
    alpha_0 = float(params.get("alpha_0", 1.0))
    lambda_ = float(params.get("lambda_", 0.0))
    ts, alphas = [], []
    for i in range(n_obs):
        t_start = t_iti + i * t_step
        t_end = t_start + t_obs
        alpha = alpha_0 / ((i + 1) ** lambda_)
        ts.extend([t_start, t_end])
        alphas.extend([alpha, alpha])
    return np.array(ts), np.array(alphas)


def _ideal_count_steps(params, n_obs):
    t_obs = params["t_obs"]
    t_iti = params["t_iti"]
    t_step = t_obs + t_iti
    ts, counts = [], []
    for i in range(n_obs):
        t_start = t_iti + i * t_step
        t_end = t_start + t_obs
        ts.extend([t_start, t_end])
        counts.extend([i + 1, i + 1])
    return np.array(ts), np.array(counts)


def plot_dynamics(probe_data: dict) -> None:
    apply_style()
    palette = get_palette()
    nef_color = palette["NEF"]

    t = probe_data["t"]
    obs = probe_data["obs"]
    error = probe_data["error"]
    value = probe_data["value"]
    params = probe_data["params"]
    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti
    n_obs = int(round((t[-1] + float(params["dt"])) / t_step))

    fig, axes = plt.subplots(
        1,
        6,
        figsize=(FIGURE_SIZE[0] * 2.2, FIGURE_SIZE[1] * 0.6),
        constrained_layout=True,
    )

    axes[0].plot(t, obs, color="0.3", linewidth=0.8)
    axes[0].set_title("Observation")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("o(t)")

    axes[1].plot(t, error[:, 0] * error[:, 1], color=nef_color, linewidth=0.8)
    axes[1].axhline(0, color="0.7", linewidth=0.5, linestyle="--")
    axes[1].set_title("Error")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("alpha(n) * (o - v)")

    axes[2].plot(t, error[:, 0], color=palette["Bayes"], linewidth=0.8)
    t_steps, alpha_steps = _ideal_alpha_steps(params, n_obs)
    axes[2].plot(t_steps, alpha_steps, color="0.4", linewidth=1.0,
                 linestyle="--", label="ideal")
    axes[2].plot(t, error[:, 0], color=palette["Bayes"], linewidth=0.8,
                 label="NEF error[0]")
    axes[2].legend(frameon=False, fontsize=7)
    axes[2].set_title("Alpha(n)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("alpha(n)")

    axes[3].plot(t, value, color=nef_color, linewidth=0.8)
    axes[3].axhline(0, color="0.7", linewidth=0.5, linestyle="--")
    axes[3].set_title("Value")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("v(t)")

    if "counting_weight" in probe_data:
        ax = axes[4]
        ax.plot(
            t, probe_data["counting_weight"], color=nef_color, linewidth=0.8, label="decoded"
        )
        t_steps, alpha_steps = _ideal_alpha_steps(params, n_obs)
        ax.plot(
            t_steps,
            alpha_steps,
            color="0.4",
            linewidth=1.0,
            linestyle="--",
            label="ideal",
        )
        ax.set_title("Counting weight")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("alpha(n)")
        ax.legend(frameon=False, fontsize=7)
        _despine(ax)

    if "counting_count" in probe_data:
        ax = axes[5]
        ax.plot(
            t, probe_data["counting_count"], color=nef_color, linewidth=0.8, label="decoded"
        )
        t_steps_c, count_steps = _ideal_count_steps(params, n_obs)
        ax.plot(
            t_steps_c,
            count_steps,
            color="0.4",
            linewidth=1.0,
            linestyle="--",
            label="ideal",
        )
        ax.set_title("Counting count")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("n")
        ax.legend(frameon=False, fontsize=7)
        _despine(ax)

    for ax in axes:
        _despine(ax)

    fig.suptitle(
        f"{params['model_type']} | {params['dataset']} pid={params['pid']} "
        f"seed={params['seed']}",
        fontsize=9,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = (
        f"NEF_dynamics_{params['model_type']}_{params['dataset']}_{params['pid']}"
    )
    plt.savefig(FIGURES_DIR / f"{stem}.png", dpi=300)
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.{{png,pdf}}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot NEF probe dynamics")
    p.add_argument("--dataset", type=str, default="carrabin")
    p.add_argument("--pid", type=int, default=None)
    p.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Required, e.g. NEF_recurrent or NEF_synaptic",
    )
    args = p.parse_args()
    if not args.model_type:
        p.error("--model_type is required (e.g. NEF_recurrent, NEF_synaptic)")

    probe_path = None
    if args.pid is not None:
        fname = f"probe_{args.model_type}_{args.dataset}_{args.pid}.pkl"
        candidate = data_path(fname)
        if candidate.exists():
            probe_path = candidate
    if probe_path is None:
        pattern = f"probe_{args.model_type}_{args.dataset}_*.pkl"
        candidates = sorted(DATA_DIR.glob(pattern))
        if not candidates:
            raise FileNotFoundError(
                f"No probe file found matching {pattern} in {DATA_DIR}"
            )
        probe_path = candidates[0]
    probe_data = pd.read_pickle(probe_path)
    plot_dynamics(probe_data)


if __name__ == "__main__":
    main()
