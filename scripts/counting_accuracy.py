#!/usr/bin/env python3
"""
Sweep n_neurons_counting for integrator and lmu, measuring count/weight
error and runtime. Saves a summary figure.

Usage:
    python scripts/counting_accuracy.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.counting_integrator import (
    _eval_idx as _eval_idx_int,
    build_network as build_integrator,
    decode_outputs as decode_integrator,
    simulate_network as simulate_integrator,
)
from models.counting_lmu import (
    _eval_idx,
    build_network as build_lmu,
    decode_outputs as decode_lmu,
    simulate_network as simulate_lmu,
)
from fitting.model_params import _NEF_FIXED
from utils.paths import FIGURES_DIR
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette

N_NEURONS_VALUES = [1000, 1500, 2000]
N_SEEDS = 5
SEED_START = 0

BASE_PARAMS = {
    **_NEF_FIXED,
    "n_obs": 30,
    "n_neurons": 300,  # script-specific override for sweep
    "seed": 0,
    "n_seeds": 1,
    "lambda_": 0.5,
    "alpha_0": 1.0,
}


def run_one(mechanism: str, n_neurons_counting: int, seed: int) -> dict:
    p = {**BASE_PARAMS, "seed": seed, "n_neurons_counting": n_neurons_counting}
    t0 = time.time()
    if mechanism == "lmu":
        net_train = build_lmu(p, train=True)
        raw = simulate_lmu(net_train, p, train=True)
        decoders = decode_lmu(raw, p)
        net_test = build_lmu(p, train=False, decoders=decoders)
        out = simulate_lmu(net_test, p, train=False)
        idx = _eval_idx(p, len(out["ideal_count"]))
        count_dec = out["lmu_neural_count"][idx]
        weight_dec = out["lmu_neural_weight"][idx]
    else:
        net_train = build_integrator(p, train=True)
        raw = simulate_integrator(net_train, p, train=True)
        decoders = decode_integrator(raw, p)
        net_test = build_integrator(p, train=False, decoders=decoders)
        out = simulate_integrator(net_test, p, train=False)
        idx = _eval_idx_int(p, len(out["ideal_count"]))
        count_dec = out["count"][idx]
        weight_dec = out["weight"][idx]
    elapsed = time.time() - t0

    n_true = out["ideal_count"][idx]
    w_true = out["ideal_weight"][idx]
    return {
        "mechanism": mechanism,
        "n_neurons_counting": n_neurons_counting,
        "seed": seed,
        "rmse_count": float(np.sqrt(np.mean((count_dec - n_true) ** 2))),
        "rmse_weight": float(np.sqrt(np.mean((weight_dec - w_true) ** 2))),
        "runtime": elapsed,
    }


def main() -> None:
    apply_style()
    palette = get_palette()

    rows = []
    for mechanism in ("integrator", "lmu"):
        for n in N_NEURONS_VALUES:
            for s in range(SEED_START, SEED_START + N_SEEDS):
                print(f"  {mechanism}, n_neurons_counting={n}, seed={s}")
                rows.append(run_one(mechanism, n, s))
    df = pd.DataFrame(rows)
    print(df)

    colors = {"integrator": palette["RL"], "lmu": palette["NEF"]}
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE, constrained_layout=True)

    titles = ["Count RMSE", "Weight RMSE", "Runtime (s)"]
    ylabels = ["RMSE", "RMSE", "seconds"]
    ys = ["rmse_count", "rmse_weight", "runtime"]
    for ax, title, ylabel, y in zip(axes, titles, ylabels, ys):
        sns.lineplot(
            data=df,
            x="n_neurons_counting",
            y=y,
            hue="mechanism",
            palette=colors,
            errorbar="sd",
            marker="o",
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("n_neurons_counting")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=7)
        sns.despine(ax=ax, top=True, right=True)

    fig.suptitle(
        "Counting circuit: error and runtime vs n_neurons_counting", fontsize=9
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "counting_accuracy.png", dpi=300)
    plt.savefig(FIGURES_DIR / "counting_accuracy.pdf")
    print("Saved figures/counting_accuracy.{png,pdf}")


if __name__ == "__main__":
    main()
