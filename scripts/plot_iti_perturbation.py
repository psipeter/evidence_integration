#!/usr/bin/env python3
"""Plot scripts/neural_experiments.py's `iti_perturbation*` output.

--mode summary (default): TWO panels vs observation, one line per
(model_type, strength) in each -- rmse (vs the sequence's own fixed
true_mean) on the left, sigma (response std across repeats) on the right.
Both panels are built from the SAME per-repeat raw responses via
sns.lineplot's estimator/errorbar machinery (a closure over true_mean for
rmse, estimator=np.std for sigma), so both get a seaborn-bootstrapped 95%
CI, not just a point estimate. Reported together deliberately -- sigma
alone can't distinguish "protected from noise" from "responses collapsed
toward a common wrong answer" (rmse would still get worse in the latter
case even as sigma shrinks). See run_iti_perturbation's own docstring for
the experiment design.

--mode dose_response: same rmse/sigma pair, but collapsed to ONLY the last
observation (i.e. after the full fixed sequence has been seen) and with
perturbation STRENGTH on the x-axis instead of observation -- "how does
strength affect accuracy/variance after viewing a fixed sequence" rather
than "how do these quantities grow across observations within a fixed
strength". Reuses the exact same raw dataframe as --mode summary, no new
simulation needed as long as the strengths you want were included in the
run_iti_perturbation call that produced it.

--mode dynamics: single-trial full decoded-`value` traces in ONE panel,
color distinguishing both NEF implementation and strength (2 implementations
x 2 strengths = 4 lines; same repeat-seed across all four for direct
comparability), from `iti_perturbation_dynamics` -- for manually inspecting
what the perturbation does to value's own trajectory, per
run_iti_perturbation_dynamics' docstring.

Usage:
    python scripts/plot_iti_perturbation.py --task soltani_numbers
    python scripts/plot_iti_perturbation.py --task soltani_numbers --mode dose_response
    python scripts/plot_iti_perturbation.py --task soltani_numbers --mode dynamics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fitting.model_params import MODEL_PARAMS
from scripts.neural_experiments import (
    ITI_PERTURBATION_FREQ_HZ,
    ITI_PERTURBATION_SEQUENCES,
    OUT_DIR,
)
from utils.paths import FIGURES_DIR
from utils.plot_style import FIGURE_SIZE, apply_style

# Shared across all three plot modes -- "NEF implementation" as the one
# legend/column concept spanning model_type's two values.
IMPL_COLUMN = "NEF implementation"
IMPL_LABELS = {"NEF": "working memory", "NEF_synaptic": "synaptic"}


def _with_impl_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(**{IMPL_COLUMN: df["model_type"].map(lambda m: IMPL_LABELS.get(m, m))})


def _rmse_estimator_for(true_mean: float):
    def _estimator(y: np.ndarray) -> float:
        y = np.asarray(y)
        return float(np.sqrt(np.mean((y - true_mean) ** 2)))
    return _estimator


def plot_summary(raw_df: pd.DataFrame, task: str) -> None:
    """RMSE (left) and sigma (right) vs observation, both bootstrapped by
    seaborn directly from the per-repeat raw responses -- see this module's
    own top-of-file note on why both are shown together."""
    apply_style()
    plot_df = _with_impl_column(raw_df)
    strengths = sorted(raw_df["strength"].unique())
    impl_order = [IMPL_LABELS.get(mt, mt) for mt in sorted(raw_df["model_type"].unique())]
    palette = dict(zip(strengths, sns.color_palette("viridis", n_colors=len(strengths))))
    dashes = {"working memory": (1, 0), "synaptic": (4, 1.5)}
    true_mean = float(raw_df["true_mean"].iloc[0])
    rmse_estimator = _rmse_estimator_for(true_mean)

    fig, (ax_rmse, ax_sigma) = plt.subplots(
        1, 2, figsize=(FIGURE_SIZE[0] * 1.8, FIGURE_SIZE[1]), constrained_layout=True,
    )
    panels = [
        (ax_rmse, rmse_estimator, "RMSE vs true_mean (95% CI)", False),
        (ax_sigma, np.std, "Sigma (response std across repeats, 95% CI)", True),
    ]
    for ax, estimator, ylabel, show_legend in panels:
        sns.lineplot(
            data=plot_df,
            x="observation", y="response",
            hue="strength", hue_order=strengths, palette=palette,
            style=IMPL_COLUMN, style_order=impl_order,
            dashes={mt: dashes.get(mt, (1, 0)) for mt in impl_order},
            estimator=estimator, errorbar=("ci", 95), seed=0,
            linewidth=1.6, ax=ax, legend=show_legend,
        )
        ax.set_xlabel("Observation")
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)
    ax_rmse.set_title("Task accuracy (vs ground truth)")
    ax_sigma.set_title("Response variability across repeats")
    ax_sigma.legend(frameon=False, fontsize=8)
    fig.suptitle(f"ITI-perturbation: {task}, {ITI_PERTURBATION_FREQ_HZ:g} Hz", fontsize=10)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"iti_perturbation_{task}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_dose_response(raw_df: pd.DataFrame, task: str) -> None:
    """RMSE (left) and sigma (right) vs perturbation STRENGTH, using only
    the last observation's response (i.e. after the full fixed sequence has
    been seen) -- "how does strength affect accuracy/variance after
    viewing a fixed sequence", as opposed to plot_summary's own "how do
    these grow across observations within a fixed strength"."""
    apply_style()
    last_obs = int(raw_df["observation"].max())
    final_df = _with_impl_column(raw_df[raw_df["observation"] == last_obs])
    impl_order = [IMPL_LABELS.get(mt, mt) for mt in sorted(raw_df["model_type"].unique())]
    true_mean = float(raw_df["true_mean"].iloc[0])
    rmse_estimator = _rmse_estimator_for(true_mean)

    fig, (ax_rmse, ax_sigma) = plt.subplots(
        1, 2, figsize=(FIGURE_SIZE[0] * 1.8, FIGURE_SIZE[1]), constrained_layout=True,
    )
    panels = [
        (ax_rmse, rmse_estimator, "RMSE vs true_mean (95% CI)", False),
        (ax_sigma, np.std, "Sigma (response std across repeats, 95% CI)", True),
    ]
    for ax, estimator, ylabel, show_legend in panels:
        sns.lineplot(
            data=final_df,
            x="strength", y="response",
            hue=IMPL_COLUMN, hue_order=impl_order,
            estimator=estimator, errorbar=("ci", 95), seed=0,
            marker="o", linewidth=1.6, ax=ax, legend=show_legend,
        )
        ax.set_xlabel("Perturbation strength")
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)
    ax_rmse.set_title("Task accuracy after full sequence")
    ax_sigma.set_title("Response variability after full sequence")
    ax_sigma.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"ITI-perturbation dose-response: {task}, obs={last_obs} (after full sequence)",
        fontsize=10,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"iti_perturbation_dose_response_{task}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def _iti_shading(ax, n_obs: int, t_iti: float, t_step: float, t_obs: float) -> None:
    """Shade every ITI window, including the pre-obs-1 one (not perturbed --
    see _iti_perturbation_gate -- but still architecturally an ITI)."""
    ax.axvspan(0, t_iti, alpha=0.08, color="gray", linewidth=0, zorder=0)
    for i in range(n_obs):
        start = t_iti + i * t_step + t_obs
        ax.axvspan(start, start + t_iti, alpha=0.08, color="gray", linewidth=0, zorder=0)


# 2 colors per implementation (strength[0] -> first, strength[1] -> second)
# -- ONLY valid for exactly 2 strengths, matching this plot's own
# 2-implementation x 2-strength = 4-line design.
_DYNAMICS_COLOR_PAIRS = {"NEF": ("tab:blue", "tab:green"), "NEF_synaptic": ("tab:orange", "gold")}


def plot_dynamics(df: pd.DataFrame, task: str) -> None:
    apply_style()
    model_types = sorted(df["model_type"].unique())
    strengths = sorted(df["strength"].unique())
    if len(strengths) != 2:
        raise ValueError(
            f"plot_dynamics' 4-line color scheme assumes exactly 2 strengths, got {strengths} "
            f"-- re-run iti_perturbation_dynamics with exactly 2 --strengths."
        )

    fixed = MODEL_PARAMS[task]["NEF"]["fixed"]
    t_obs, t_iti = float(fixed["t_obs"]), float(fixed["t_iti"])
    t_step = t_obs + t_iti
    n_obs = len(ITI_PERTURBATION_SEQUENCES[task])

    fig, ax = plt.subplots(figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 0.7), constrained_layout=True)
    _iti_shading(ax, n_obs, t_iti, t_step, t_obs)
    for model_type in model_types:
        colors = _DYNAMICS_COLOR_PAIRS.get(model_type, ("tab:gray", "tab:pink"))
        label_base = IMPL_LABELS.get(model_type, model_type)
        g_model = df[df["model_type"] == model_type]
        for strength, color in zip(strengths, colors):
            g = g_model[g_model["strength"] == strength].sort_values("t")
            ax.plot(g["t"], g["value"], color=color, linewidth=1.2,
                    label=f"{label_base}, strength={strength:g}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Decoded value")
    ax.margins(x=0)
    sns.despine(ax=ax)
    ax.legend(title=IMPL_COLUMN, frameon=False, fontsize=8)
    fig.suptitle(f"ITI-perturbation single-trial dynamics: {task}", fontsize=10)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"iti_perturbation_dynamics_{task}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--mode", choices=["summary", "dose_response", "dynamics"], default="summary")
    args = p.parse_args()

    if args.mode in ("summary", "dose_response"):
        raw_path = OUT_DIR / f"iti_perturbation_{args.task}_raw.pkl"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"No {raw_path} -- run scripts/neural_experiments.py iti_perturbation "
                f"--task {args.task} first."
            )
        raw_df = pd.read_pickle(raw_path)
        if args.mode == "summary":
            plot_summary(raw_df, args.task)
        else:
            plot_dose_response(raw_df, args.task)
    else:
        dyn_path = OUT_DIR / f"iti_perturbation_dynamics_{args.task}.pkl"
        if not dyn_path.exists():
            raise FileNotFoundError(
                f"No {dyn_path} -- run scripts/neural_experiments.py "
                f"iti_perturbation_dynamics --task {args.task} first."
            )
        plot_dynamics(pd.read_pickle(dyn_path), args.task)


if __name__ == "__main__":
    main()
