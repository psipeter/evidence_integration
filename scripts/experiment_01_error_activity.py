#!/usr/bin/env python3
"""
Plot mean error-population activity vs prediction error for Experiment 01.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, data_path
from utils.plot_style import apply_style, get_palette

EXPERIMENT_NAME = "experiment_01_error_activity"
DATASET = "carrabin"
OBSERVATION = 3  # which observation to plot (can override via CLI)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DATASET)
    parser.add_argument("--observation", type=int, default=OBSERVATION)
    parser.add_argument("--run_folder", type=str, default="joint_loss")
    args = parser.parse_args()

    data_file = (
        data_path("experiments")
        / EXPERIMENT_NAME
        / args.dataset
        / f"{EXPERIMENT_NAME}_{args.dataset}.pkl"
    )
    if not data_file.exists():
        raise FileNotFoundError(
            f"Combined output not found: {data_file}\n"
            f"Run: python experiments/{EXPERIMENT_NAME}.py --collect --dataset {args.dataset}"
        )
    df = pd.read_pickle(data_file)

    obs_col = "stage" if args.dataset == "jiang" else "observation"
    plot_df = df[df[obs_col] == args.observation].copy()
    if plot_df.empty:
        raise ValueError(
            f"No rows found for dataset={args.dataset!r}, {obs_col}={args.observation} "
            f"in {data_file}"
        )

    apply_style()
    palette = get_palette()
    color = palette.get("NEF", palette.get("NEF_recurrent", "0.3"))

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    sns.regplot(
        data=plot_df,
        x="prediction_error",
        y="mean_activity",
        scatter=True,
        scatter_kws={"alpha": 0.3, "s": 10, "color": color},
        line_kws={"color": color, "linewidth": 2},
        ax=ax,
    )

    ax.set_xlabel("Prediction error (o − v)")
    ax.set_ylabel("Mean error population activity (Hz)")
    ax.set_title(f"{args.dataset} — observation {args.observation}")
    sns.despine(ax=ax, top=True, right=True)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"experiment_01_{args.dataset}_obs{args.observation}"
    plt.savefig(FIGURES_DIR / f"{fname}.png", dpi=300)
    plt.savefig(FIGURES_DIR / f"{fname}.pdf")
    print(f"Saved figures/{fname}.{{png,pdf}}")
    plt.close(fig)


if __name__ == "__main__":
    main()
