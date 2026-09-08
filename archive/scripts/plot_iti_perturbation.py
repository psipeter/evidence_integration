#!/usr/bin/env python3
"""Plot scripts/neural_experiments.py's `iti_perturbation*` output.

--mode dose_response (default): RMSE (left) and sigma (right) vs
perturbation strength, one line per NEF implementation -- "how does
strength affect accuracy/reliability across the synthetic pool" (200
sessions x 32 trials each). Built via a two-stage, qid-aware hierarchy
mirroring how human data is aggregated elsewhere in this project (see
_session_level_stats' own docstring): each SESSION reduces to one sigma
value and one rmse value, then sns.lineplot's default mean/errorbar='se'
shows mean +/- SEM across the 200 sessions -- NOT a bootstrap over raw
rows. Reported together deliberately -- sigma alone can't distinguish
"protected from noise" from "responses collapsed toward a common wrong
answer" (rmse would still get worse in the latter case even as sigma
shrinks). See run_iti_perturbation's own docstring for the experiment
design.

--mode dynamics: single-trial full decoded-`value` traces in ONE panel,
color distinguishing both NEF implementation and strength (2 implementations
x 2 strengths = 4 lines; same repeat-seed across all four for direct
comparability), from `iti_perturbation_dynamics` -- for manually inspecting
what the perturbation does to value's own trajectory, per
run_iti_perturbation_dynamics' docstring.

Usage:
    python scripts/plot_iti_perturbation.py --task soltani_numbers
    python scripts/plot_iti_perturbation.py --task soltani_numbers --mode dynamics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fitting.model_params import MODEL_PARAMS
from scripts.neural_experiments import (
    ITI_PERTURBATION_PREFIX_LENGTH,
    OUT_DIR,
    _session_level_stats,
)
from utils.paths import FIGURES_DIR
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette

# Shared across both plot modes -- "NEF implementation" as the one
# legend/column concept spanning model_type's two values.
IMPL_COLUMN = "NEF implementation"
IMPL_LABELS = {"NEF": "working memory", "NEF_synaptic": "synaptic"}

# Paper-wide standard width (make_paper_figures.py's own FIGURE_SIZE[0]) --
# this plot's --mode dynamics is sized to sit as one of two side-by-side
# columns within that width, not a standalone full-width figure.
PAPER_WIDTH = 10.6
HALF_COLUMN_WIDTH = PAPER_WIDTH / 2

# One color per model_type, shared by both --mode dose_response and
# --mode dynamics: NEF ("recurrent") takes the standard colorblind
# palette's pink (index 4 -- matching make_paper_figures.py's own
# MODEL_COLORS["NEF"]); NEF_synaptic takes the next color in that same
# palette (index 5, brown) -- not a hand-picked "different part of color
# space" pair, just "pink, then whichever color is next", per instruction.
# --mode dynamics' own strength dimension is now carried by LINESTYLE
# (sns.lineplot's `style`) instead of a second color per model_type.
_PALETTE = get_palette(6)
_MODEL_TYPE_COLORS = {"NEF": _PALETTE[4], "NEF_synaptic": _PALETTE[5]}


def _with_impl_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(**{IMPL_COLUMN: df["model_type"].map(lambda m: IMPL_LABELS.get(m, m))})


# Beyond this strength, RMSE/sigma turn nonmonotonic (an artifact of
# implausibly large injected noise, not a real dose-response regime) --
# see chat for the visual confirmation this was based on.
DOSE_RESPONSE_MAX_STRENGTH = 0.5


def plot_dose_response(raw_df: pd.DataFrame, task: str) -> None:
    """RMSE (solid) and sigma (dashed) vs perturbation strength, sharing
    one panel via a dual y-axis (RMSE left, sigma right) -- see
    _session_level_stats' own docstring for the aggregation hierarchy.
    x-axis restricted to [0, DOSE_RESPONSE_MAX_STRENGTH] (see that
    constant's own comment)."""
    apply_style()
    stats_df = _with_impl_column(_session_level_stats(raw_df))
    stats_df = stats_df[stats_df["strength"] <= DOSE_RESPONSE_MAX_STRENGTH]
    model_types = sorted(raw_df["model_type"].unique())

    # Sized as one of two side-by-side columns within the paper's standard
    # 10.6in width (see HALF_COLUMN_WIDTH) -- not a standalone full-width figure.
    fig, ax_rmse = plt.subplots(
        figsize=(HALF_COLUMN_WIDTH, HALF_COLUMN_WIDTH * FIGURE_SIZE[1] / FIGURE_SIZE[0]),
        constrained_layout=True,
    )
    ax_sigma = ax_rmse.twinx()

    handles = []
    for model_type in model_types:
        color = _MODEL_TYPE_COLORS.get(model_type, "gray")
        g = stats_df[stats_df["model_type"] == model_type]
        sns.lineplot(
            data=g, x="strength", y="rmse", color=color, linestyle="-",
            estimator="mean", errorbar="se",
            linewidth=1.6, ax=ax_rmse, legend=False,
        )
        sns.lineplot(
            data=g, x="strength", y="sigma", color=color, linestyle="--",
            estimator="mean", errorbar="se",
            linewidth=1.6, ax=ax_sigma, legend=False,
        )
        handles.append(Line2D([0], [0], color=color, lw=1.6, label=IMPL_LABELS.get(model_type, model_type)))
    handles.append(Line2D([0], [0], color="black", lw=1.6, linestyle="-", label="RMSE"))
    handles.append(Line2D([0], [0], color="black", lw=1.6, linestyle="--", label="Sigma"))

    ax_rmse.set_xlabel("Perturbation strength")
    ax_rmse.set_ylabel("Model RMSE")
    ax_sigma.set_ylabel(r"Model $\sigma_R$")
    ax_rmse.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    sns.despine(ax=ax_rmse, right=True)
    sns.despine(ax=ax_sigma, top=True, right=False, left=True, bottom=True)
    ax_sigma.tick_params(axis="y", which="both", right=True, left=False)
    ax_rmse.legend(handles=handles, frameon=True, framealpha=0.85, fontsize=7)
    fig.suptitle("ITI perturbation dose-response", fontsize=10)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"iti_perturbation_{task}.pdf"
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


def _alpha_for_strengths(strengths: list, lo: float = 0.35, hi: float = 1.0) -> dict:
    """Map each strength to an opacity level, highest strength -> most
    opaque (hi) since the perturbed condition is the main story here,
    lowest -> most faded (lo), linear in between. NOT linestyle: dashes
    visually alias into a solid line at this trace's own high-frequency
    oscillation (dash spacing << the value trace's own wiggle period), so
    alpha is the channel that actually reads here."""
    if len(strengths) == 1:
        return {strengths[0]: hi}
    return {s: lo + (hi - lo) * i / (len(strengths) - 1) for i, s in enumerate(strengths)}


def plot_dynamics(df: pd.DataFrame, task: str) -> None:
    """Single-trial full decoded-`value` traces: color = model_type (via
    _MODEL_TYPE_COLORS), alpha = strength (see _alpha_for_strengths)."""
    apply_style()
    plot_df = _with_impl_column(df)
    impl_order = [IMPL_LABELS.get(mt, mt) for mt in sorted(df["model_type"].unique())]
    strengths = sorted(df["strength"].unique())
    alpha_for_strength = _alpha_for_strengths(strengths)
    palette = {IMPL_LABELS.get(mt, mt): c for mt, c in _MODEL_TYPE_COLORS.items()}

    fixed = MODEL_PARAMS[task]["NEF"]["fixed"]
    t_obs, t_iti = float(fixed["t_obs"]), float(fixed["t_iti"])
    t_step = t_obs + t_iti
    n_obs = ITI_PERTURBATION_PREFIX_LENGTH[task]

    # Sized as one of two side-by-side columns within the paper's standard
    # 10.6in width (see HALF_COLUMN_WIDTH) -- not a standalone full-width figure.
    fig, ax = plt.subplots(
        figsize=(HALF_COLUMN_WIDTH, HALF_COLUMN_WIDTH * FIGURE_SIZE[1] / FIGURE_SIZE[0]),
        constrained_layout=True,
    )
    _iti_shading(ax, n_obs, t_iti, t_step, t_obs)
    for strength in strengths:
        g = plot_df[plot_df["strength"] == strength]
        sns.lineplot(
            data=g, x="t", y="value",
            hue=IMPL_COLUMN, hue_order=impl_order, palette=palette,
            estimator=None, linewidth=1.2, alpha=alpha_for_strength[strength],
            ax=ax, legend=False,
        )
    handles = [Line2D([0], [0], color=c, lw=1.6, label=IMPL_LABELS.get(mt, mt))
               for mt, c in _MODEL_TYPE_COLORS.items()]
    handles += [Line2D([0], [0], color="0.3", lw=1.6, alpha=a, label=f"strength={s:g}")
                for s, a in alpha_for_strength.items()]
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Decoded value")
    ax.margins(x=0)
    sns.despine(ax=ax)
    ax.legend(handles=handles, fontsize=7, frameon=True, framealpha=0.85)
    fig.suptitle("ITI Perturbation Dynamics", fontsize=10)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"iti_perturbation_dynamics_{task}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--mode", choices=["dose_response", "dynamics"], default="dose_response")
    args = p.parse_args()

    if args.mode == "dose_response":
        raw_path = OUT_DIR / f"iti_perturbation_{args.task}_raw.pkl"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"No {raw_path} -- run scripts/neural_experiments.py iti_perturbation "
                f"--task {args.task} --mode collect first."
            )
        plot_dose_response(pd.read_pickle(raw_path), args.task)
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
