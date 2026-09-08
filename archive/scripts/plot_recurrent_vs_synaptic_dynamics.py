#!/usr/bin/env python3
"""Plot scripts/neural_experiments.py's `recurrent_vs_synaptic_dynamics`
output: mean +/- 95% CI decoded `value` trace for NEF (recurrent) vs
NEF_synaptic on ONE fixed 4-observation pool sequence, NO perturbation
(strength=0.0 throughout) -- the properly-powered baseline-consistency
check, superseding this session's earlier single-trace-per-model eyeball
comparisons. One sns.lineplot call does the mean/CI aggregation directly
from the raw per-seed traces (estimator="mean", errorbar=("ci", 95)) --
no manual aggregation step, unlike --mode dose_response's own two-stage
qid-aware hierarchy (there's no qid/session structure here, just repeat
seeds of one fixed stimulus).

Usage:
    python scripts/plot_recurrent_vs_synaptic_dynamics.py --task soltani_numbers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fitting.model_params import MODEL_PARAMS
from scripts.neural_experiments import ITI_PERTURBATION_PREFIX_LENGTH, OUT_DIR
from scripts.plot_iti_perturbation import (
    HALF_COLUMN_WIDTH,
    IMPL_COLUMN,
    IMPL_LABELS,
    _MODEL_TYPE_COLORS,
    _iti_shading,
    _with_impl_column,
)
from utils.paths import FIGURES_DIR
from utils.plot_style import FIGURE_SIZE, apply_style

# Same per-model_type colors as plot_iti_perturbation.py's own dose_response/
# dynamics plots (NEF/"working memory" -> palette pink, NEF_synaptic/
# "synaptic" -> the next palette color, brown) -- imported directly rather
# than recomputed, so this figure is guaranteed to match rather than merely
# resemble them.
_PALETTE = {IMPL_LABELS[mt]: color for mt, color in _MODEL_TYPE_COLORS.items()}


def plot_recurrent_vs_synaptic_dynamics(df: pd.DataFrame, task: str) -> None:
    apply_style()
    plot_df = _with_impl_column(df)
    impl_order = sorted(plot_df[IMPL_COLUMN].unique())

    fixed = MODEL_PARAMS[task]["NEF"]["fixed"]
    t_obs, t_iti = float(fixed["t_obs"]), float(fixed["t_iti"])
    t_step = t_obs + t_iti
    n_obs = ITI_PERTURBATION_PREFIX_LENGTH[task]

    fig, ax = plt.subplots(
        figsize=(HALF_COLUMN_WIDTH, HALF_COLUMN_WIDTH * FIGURE_SIZE[1] / FIGURE_SIZE[0]),
        constrained_layout=True,
    )
    _iti_shading(ax, n_obs, t_iti, t_step, t_obs)
    sns.lineplot(
        data=plot_df, x="t", y="value",
        hue=IMPL_COLUMN, hue_order=impl_order, palette=_PALETTE,
        estimator="mean", errorbar=("ci", 95),
        linewidth=1.6, ax=ax,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Decoded value")
    ax.margins(x=0)
    sns.despine(ax=ax)
    ax.legend(title=IMPL_COLUMN, title_fontsize=7, fontsize=7, frameon=True, framealpha=0.85)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"recurrent_vs_synaptic_dynamics_{task}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    args = p.parse_args()

    path = OUT_DIR / f"recurrent_vs_synaptic_dynamics_{args.task}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No {path} -- run scripts/neural_experiments.py "
            f"recurrent_vs_synaptic_dynamics --task {args.task} first."
        )
    plot_recurrent_vs_synaptic_dynamics(pd.read_pickle(path), args.task)


if __name__ == "__main__":
    main()
