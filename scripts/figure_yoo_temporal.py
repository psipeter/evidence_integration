#!/usr/bin/env python3
"""figure_yoo_temporal.py — T group figure for yoo task.

Layout: 1×2
  Panel A (T1): Estimation error — RMSE to cumulative true mean vs observation
  Panel B (T2): Mean |Δresponse| vs observation (obs ≥ 2)

Run:
    python scripts/figure_yoo_temporal.py
    python scripts/figure_yoo_temporal.py --run_folder yoo --nef_folder refit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, RUNS_DIR, data_path
from utils.plot_style import (
    FIGURE_SIZE,
    apply_style,
    get_palette,
    label_panels,
)

MODEL_ORDER = ["Mean", "PrimacyRecency", "LeakyIntegrator", "NEF"]
HUMAN_COLOR = "0.3"
OBS_TICKS   = [5, 10, 15, 20, 25, 30]


def _display(model_type: str) -> str:
    return "NEF" if model_type.startswith("NEF") else model_type


def _resp_path(mt: str, run_dir: Path, nef_dir: Path) -> Path:
    d = nef_dir if mt == "NEF" else run_dir
    return d / f"{mt}_yoo_responses.pkl"


def _abs_delta_long(df: pd.DataFrame) -> pd.DataFrame:
    """Per-trial |Δresponse| at each obs ≥ 2 (diff from previous observation)."""
    pieces = []
    for (pid, trial), g in df.groupby(["pid", "trial"], sort=False):
        g = g.sort_values("observation").copy()
        g["delta"] = g["response"].diff().abs()
        pieces.append(g)
    if not pieces:
        return pd.DataFrame(columns=["pid", "trial", "observation", "delta"])
    out = pd.concat(pieces, ignore_index=True)
    return out[out["observation"] >= 2].dropna(subset=["delta"])


# ── Panel A (T1) — Estimation error vs observation ───────────────────────────

def _plot_panel_a(ax, run_folder: str, palette: dict,
                  model_order: list[str], nef_folder: str | None) -> None:
    """Panel A (T1): RMSE to cumulative true mean as a function of observation.

    true_mean at obs t = mean(value[1..t]) for that trial.
    Mean ± SEM across pids at each observation position.
    """
    run_dir = RUNS_DIR / run_folder
    nef_dir = RUNS_DIR / nef_folder if nef_folder else run_dir
    yoo     = pd.read_pickle(data_path("yoo.pkl"))

    yoo_s = yoo.sort_values(["pid", "trial", "observation"]).copy()
    yoo_s["true_mean"] = (yoo_s.groupby(["pid", "trial"])["value"]
                               .expanding().mean().values)
    true_map = yoo_s[["pid", "trial", "observation", "true_mean"]].drop_duplicates()

    handles, labels = [], []

    # Human
    h = yoo_s.assign(sq_err=(yoo_s["response"] - yoo_s["true_mean"]) ** 2)
    stats_h = (h.groupby(["pid", "observation"])["sq_err"].mean()
                .apply(np.sqrt).reset_index(name="rmse")
                .groupby("observation")["rmse"].agg(["mean", "sem"]).reset_index())
    ax.plot(stats_h["observation"], stats_h["mean"], "-", color=HUMAN_COLOR, lw=1.8)
    ax.fill_between(stats_h["observation"],
                    stats_h["mean"] - stats_h["sem"],
                    stats_h["mean"] + stats_h["sem"],
                    color=HUMAN_COLOR, alpha=0.2)
    handles.append(Line2D([0], [0], color=HUMAN_COLOR, lw=1.5))
    labels.append("Human")

    # Models
    for mt in model_order:
        rp = _resp_path(mt, run_dir, nef_dir)
        if not rp.exists():
            continue
        mdf = pd.read_pickle(rp).merge(
            true_map, on=["pid", "trial", "observation"], how="left")
        stats_m = (mdf.assign(sq_err=(mdf["response"] - mdf["true_mean"]) ** 2)
                   .groupby(["pid", "observation"])["sq_err"].mean()
                   .apply(np.sqrt).reset_index(name="rmse")
                   .groupby("observation")["rmse"].agg(["mean", "sem"]).reset_index())
        color = palette.get(_display(mt), "0.5")
        ax.plot(stats_m["observation"], stats_m["mean"], "-", color=color, lw=1.8)
        ax.fill_between(stats_m["observation"],
                        stats_m["mean"] - stats_m["sem"],
                        stats_m["mean"] + stats_m["sem"],
                        color=color, alpha=0.2)
        handles.append(Line2D([0], [0], color=color, lw=1.5))
        labels.append(_display(mt))

    ax.set_xlabel("Observation")
    ax.set_ylabel("Estimation error (RMSE to true mean)")
    ax.set_xticks(OBS_TICKS)
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Panel B (T2) — Mean |Δresponse| vs observation ───────────────────────────

def _plot_panel_b(ax, run_folder: str, palette: dict,
                  model_order: list[str], nef_folder: str | None) -> None:
    """Panel B (T2): Mean |Δresponse| vs observation (obs ≥ 2 only).

    delta = |response(t) - response(t-1)|, undefined at obs 1 (dropped).
    Uses sns.lineplot with CI across trials for smooth uncertainty bands,
    matching the approach in figure_yoo.py panel C.
    """
    run_dir = RUNS_DIR / run_folder
    nef_dir = RUNS_DIR / nef_folder if nef_folder else run_dir
    yoo     = pd.read_pickle(data_path("yoo.pkl"))

    handles, labels = [], []

    # Human — per-trial rows, CI across trials
    long_h = _abs_delta_long(yoo)
    sns.lineplot(data=long_h, x="observation", y="delta",
                 color=HUMAN_COLOR, lw=1.8, errorbar="ci", ax=ax,
                 label="_nolegend_")
    handles.append(Line2D([0], [0], color=HUMAN_COLOR, lw=1.5))
    labels.append("Human")

    # Models
    for mt in model_order:
        rp = _resp_path(mt, run_dir, nef_dir)
        if not rp.exists():
            continue
        long_m = _abs_delta_long(pd.read_pickle(rp))
        color  = palette.get(_display(mt), "0.5")
        sns.lineplot(data=long_m, x="observation", y="delta",
                     color=color, lw=1.8, errorbar="ci", ax=ax,
                     label="_nolegend_")
        handles.append(Line2D([0], [0], color=color, lw=1.5))
        labels.append(_display(mt))

    ax.set_xlabel("Observation")
    ax.set_ylabel("Mean |Δresponse|")
    ax.set_xticks(OBS_TICKS)
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, default="yoo")
    parser.add_argument("--nef_folder", type=str, default=None,
                        help="Load NEF data from this folder instead of --run_folder")
    parser.add_argument("--extra_models", nargs="*", default=[])
    args = parser.parse_args()

    model_order = MODEL_ORDER + [
        m for m in args.extra_models if m not in MODEL_ORDER]

    apply_style()
    pal     = get_palette(len(model_order) + 1)
    palette = {m: pal[i] for i, m in enumerate(model_order)}
    for mt in model_order:
        disp = _display(mt)
        if disp not in palette:
            palette[disp] = palette[mt]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(FIGURE_SIZE[0] * 0.55, FIGURE_SIZE[1] / 2),
        constrained_layout=True,
    )

    _plot_panel_a(axes[0], args.run_folder, palette, model_order, args.nef_folder)
    _plot_panel_b(axes[1], args.run_folder, palette, model_order, args.nef_folder)

    label_panels(axes.reshape(1, -1))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure_yoo_temporal"
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.pdf")


if __name__ == "__main__":
    main()
