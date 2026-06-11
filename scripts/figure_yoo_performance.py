#!/usr/bin/env python3
"""figure_yoo_performance.py — P group figure for yoo task.

Layout: 1×3
  Panel A: Task schematic (yoo_task.pdf)
  Panel B (P1): Estimation error — RMSE to cumulative true mean, per pid
  Panel C (P2): Model fit — RMSE to human responses, per pid

Run:
    python scripts/figure_yoo_performance.py
    python scripts/figure_yoo_performance.py --run_folder yoo --nef_folder refit
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, RUNS_DIR, data_path
from utils.plot_style import (
    FIGURE_SIZE,
    apply_style,
    get_palette,
    label_panels,
    annotate_nef_comparisons,
)

MODEL_ORDER = ["Mean", "PrimacyRecency", "LeakyIntegrator", "NEF"]
HUMAN_COLOR = "0.3"


def _display(model_type: str) -> str:
    if model_type.startswith("NEF"):
        return "NEF"
    if model_type == "RL_lambda":
        return "RL_lambda"
    return model_type


def _get_loss(perf_df: pd.DataFrame) -> pd.Series:
    if "loss" in perf_df.columns:
        return perf_df["loss"]
    return perf_df["cv_loss_mean"]


# ── Panel A — Task schematic ──────────────────────────────────────────────────

def _plot_panel_a(ax) -> None:
    """Panel A: render yoo_task.pdf into axis."""
    pdf_path = FIGURES_DIR / "yoo_task.pdf"
    if not pdf_path.exists():
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.5, 0.5, "yoo_task.pdf\nnot found",
                ha="center", va="center", transform=ax.transAxes,
                color="0.5", style="italic", fontsize=8)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / "yoo_task"
        cmd = ["pdftoppm", "-png", "-singlefile", str(pdf_path), str(out_prefix)]
        try:
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        img_path = out_prefix.with_suffix(".png")
        if not img_path.exists():
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            return
        img = mpimg.imread(img_path)

    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal"); ax.set_anchor("C")


# ── Panel B (P1) — Estimation error ──────────────────────────────────────────

def _plot_panel_b(ax, run_folder: str, palette: dict,
                  model_order: list[str],
                  nef_folder: str | None = None) -> None:
    """Panel B (P1): RMSE to cumulative true mean per pid.

    true_mean at obs t = mean(value[1..t]) for that trial.
    Both responses and true_mean are on the [-1, 1] scale.
    nef_folder: if set, NEF responses are loaded from this folder instead.
    """
    run_dir = RUNS_DIR / run_folder
    nef_dir = RUNS_DIR / nef_folder if nef_folder else run_dir
    yoo     = pd.read_pickle(data_path("yoo.pkl"))

    # Compute cumulative true mean per (pid, trial, obs)
    yoo_s = yoo.sort_values(["pid", "trial", "observation"]).copy()
    yoo_s["true_mean"] = (yoo_s.groupby(["pid", "trial"])["value"]
                               .expanding().mean().values)
    true_map = yoo_s[["pid", "trial", "observation", "true_mean"]].drop_duplicates()

    rows = []

    # Human
    h = yoo_s.copy()
    human_rmse = (
        h.assign(sq_err=(h["response"] - h["true_mean"]) ** 2)
        .groupby("pid")["sq_err"].mean()
        .apply(np.sqrt)
        .reset_index(name="rmse")
    )
    human_rmse["source"] = "Human"
    rows.append(human_rmse)

    # Models
    for mt in model_order:
        resp_dir  = nef_dir if (nef_folder and mt == "NEF") else run_dir
        resp_path = resp_dir / f"{mt}_yoo_responses.pkl"
        if not resp_path.exists():
            continue
        mdf = pd.read_pickle(resp_path).merge(
            true_map, on=["pid", "trial", "observation"], how="left")
        rmse = (
            mdf.assign(sq_err=(mdf["response"] - mdf["true_mean"]) ** 2)
            .groupby("pid")["sq_err"].mean()
            .apply(np.sqrt)
            .reset_index(name="rmse")
        )
        rmse["source"] = _display(mt)
        rows.append(rmse)

    plot_df = pd.concat(rows, ignore_index=True)
    order   = ["Human"] + [_display(m) for m in model_order
                            if _display(m) in plot_df["source"].unique()]
    pal     = {s: palette.get(s, "0.5") for s in order}
    pal["Human"] = HUMAN_COLOR

    sns.boxplot(data=plot_df, x="source", y="rmse", order=order,
                hue="source", palette=pal, legend=False, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Performance error vs ground truth (RMSE)")
    ax.tick_params(axis="x", rotation=45)
    sns.despine(ax=ax, top=True, right=True)


# ── Panel C (P2) — Model fit ──────────────────────────────────────────────────

def _plot_panel_c(ax, run_folder: str, palette: dict,
                  model_order: list[str],
                  nef_folder: str | None = None) -> None:
    """Panel C (P2): RMSE to human responses per pid.
    nef_folder: if set, NEF performance is loaded from this folder instead.
    """
    run_dir = RUNS_DIR / run_folder
    nef_dir = RUNS_DIR / nef_folder if nef_folder else run_dir
    rows    = []

    for mt in model_order:
        perf_dir  = nef_dir if (nef_folder and mt == "NEF") else run_dir
        perf_path = perf_dir / f"{mt}_yoo_performance.pkl"
        if not perf_path.exists():
            continue
        perf = pd.read_pickle(perf_path).copy()
        perf["plot_loss"] = _get_loss(perf)
        perf["source"]    = _display(mt)
        rows.append(perf[["pid", "source", "plot_loss"]])

    if not rows:
        ax.text(0.5, 0.5, "No performance data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    plot_df = pd.concat(rows, ignore_index=True)
    order   = [_display(m) for m in model_order
               if _display(m) in plot_df["source"].unique()]
    pal     = {_display(m): palette.get(_display(m), palette.get(m, "0.5"))
               for m in model_order}

    sns.boxplot(data=plot_df, x="source", y="plot_loss", order=order,
                hue="source", palette=pal, legend=False, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Model fit (RMSE to human responses)")
    ax.tick_params(axis="x", rotation=45)
    sns.despine(ax=ax, top=True, right=True)

    nef_disp = _display("NEF")
    if nef_disp in order:
        annotate_nef_comparisons(
            ax, plot_df, "source", "plot_loss", order,
            nef_label=nef_disp,
            compare_only=[_display(m) for m in model_order if m != "NEF"],
        )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder",   type=str, default="yoo")
    parser.add_argument("--nef_folder",   type=str, default=None,
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
    palette["Human"] = HUMAN_COLOR

    fig, axes = plt.subplots(
        1, 3,
        figsize=(FIGURE_SIZE[0] * 0.75, FIGURE_SIZE[1] / 2),
        constrained_layout=True,
    )

    _plot_panel_a(axes[0])
    _plot_panel_b(axes[1], args.run_folder, palette, model_order, nef_folder=args.nef_folder)
    _plot_panel_c(axes[2], args.run_folder, palette, model_order, nef_folder=args.nef_folder)

    label_panels(axes.reshape(1, -1))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure_yoo_performance"
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.pdf")


if __name__ == "__main__":
    main()
