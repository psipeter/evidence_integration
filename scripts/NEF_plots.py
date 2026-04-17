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

from utils.paths import FIGURES_DIR, data_path
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_dynamics(probe_data: dict) -> None:
    apply_style()
    palette = get_palette()
    nef_color = palette["NEF"]

    t = probe_data["t"]
    obs = probe_data["obs"]
    error = probe_data["error"]
    value = probe_data["value"]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(FIGURE_SIZE[0] * 1.5, FIGURE_SIZE[1] * 0.6),
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
    axes[2].set_title("Alpha(n)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("alpha(n)")

    axes[3].plot(t, value, color=nef_color, linewidth=0.8)
    axes[3].axhline(0, color="0.7", linewidth=0.5, linestyle="--")
    axes[3].set_title("Value")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("v(t)")

    for ax in axes:
        _despine(ax)

    params = probe_data["params"]
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
    p.add_argument("--pid", type=int, default=1)
    p.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Required, e.g. NEF_recurrent or NEF_synaptic",
    )
    args = p.parse_args()
    if not args.model_type:
        p.error("--model_type is required (e.g. NEF_recurrent, NEF_synaptic)")

    fname = f"probe_{args.model_type}_{args.dataset}_{args.pid}.pkl"
    probe_data = pd.read_pickle(data_path(fname))
    plot_dynamics(probe_data)


if __name__ == "__main__":
    main()
