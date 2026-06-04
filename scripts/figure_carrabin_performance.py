#!/usr/bin/env python3
"""figure_carrabin_performance.py — P group figure for carrabin task.

Layout: 1×2
  Panel A (P1): Task error — RMSE to true_p per pid, human and all models
  Panel B (P2): Model fit — RMSE to human responses per pid (existing metric)

Run:
    python scripts/figure_carrabin_performance.py --run_folder carrabin
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import subprocess
import tempfile
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, data_path, resolve_run_folder
from utils.plot_style import (
    FIGURE_SIZE,
    apply_style,
    get_palette,
    label_panels,
    annotate_nef_comparisons,
    pvalue_to_stars,
)

MODEL_ORDER   = ["Mean", "PrimacyRecency", "LeakyIntegrator", "NEF"]
HUMAN_COLOR   = "0.3"
HUMAN_NEUTRAL_COLOR = "0.3"


def _display(model_type: str) -> str:
    return "NEF" if model_type.startswith("NEF") else model_type


def _get_loss(perf_df: pd.DataFrame) -> pd.Series:
    if "loss" in perf_df.columns:
        return perf_df["loss"]
    return perf_df["cv_loss_mean"]



def _plot_schematic(ax) -> None:
    """Render carrabin_task.pdf into the panel."""
    pdf_path = FIGURES_DIR / "carrabin_task.pdf"
    if not pdf_path.exists():
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.text(0.5, 0.5, "carrabin_task.pdf\nnot found",
                ha="center", va="center", transform=ax.transAxes,
                color="0.5", style="italic")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / "carrabin_task"
        cmd = ["pdftoppm", "-png", "-singlefile", str(pdf_path), str(out_prefix)]
        try:
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            return
        img_path = out_prefix.with_suffix(".png")
        if not img_path.exists():
            return
        img = mpimg.imread(img_path)
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_aspect("equal"); ax.set_anchor("C")


# ── Panel B — Task error (RMSE to true_p) ────────────────────────────────────

def _plot_panel_c(ax, run_folder: str, palette: dict,
                  model_order: list[str]) -> None:
    """Panel A (P1): RMSE to true_p per pid for humans and all models.

    true_p is the actual generating probability for each sequence.
    Responses and true_p are compared on the same [-1,1] scale.
    One box per source, one data point per pid.
    """
    df = pd.read_pickle(data_path("carrabin.pkl"))
    run_dir = resolve_run_folder(run_folder)

    # true_p is in [0,1]; convert to [-1,1] response scale
    df["true_p_resp"] = df["true_p"] * 2 - 1

    rows = []

    # Human
    human_rmse = (
        df.assign(sq_err=(df["response"] - df["true_p_resp"]) ** 2)
        .groupby("pid")["sq_err"].mean()
        .apply(np.sqrt)
        .reset_index()
        .rename(columns={"sq_err": "rmse"})
    )
    human_rmse["source"] = "Human"
    rows.append(human_rmse)

    # Models
    true_p_map = df[["pid", "trial", "observation", "true_p_resp"]].drop_duplicates()
    for mt in model_order:
        resp_path = run_dir / f"{mt}_carrabin_responses.pkl"
        if not resp_path.exists():
            continue
        mdf = pd.read_pickle(resp_path).merge(
            true_p_map, on=["pid", "trial", "observation"], how="left"
        )
        rmse = (
            mdf.assign(sq_err=(mdf["response"] - mdf["true_p_resp"]) ** 2)
            .groupby("pid")["sq_err"].mean()
            .apply(np.sqrt)
            .reset_index()
            .rename(columns={"sq_err": "rmse"})
        )
        rmse["source"] = _display(mt)
        rows.append(rmse)

    plot_df = pd.concat(rows, ignore_index=True)
    order    = ["Human"] + [_display(m) for m in model_order
                            if _display(m) in plot_df["source"].unique()]
    pal      = {s: palette.get(s, "0.5") for s in order}
    pal["Human"] = HUMAN_COLOR

    sns.boxplot(
        data=plot_df, x="source", y="rmse", order=order,
        hue="source", palette=pal, legend=False, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Estimation error (RMSE to hidden probability)")
    ax.tick_params(axis="x", rotation=45)
    sns.despine(ax=ax, top=True, right=True)




# ── Panel B — Model fit (RMSE to human responses) ────────────────────────────

def _plot_panel_b(ax, run_folder: str, palette: dict,
                  model_order: list[str]) -> None:
    """Panel B (P2): RMSE to human responses per pid.

    Replicates figure_carrabin panel B for direct comparison with P1.
    """
    run_dir = resolve_run_folder(run_folder)
    rows = []
    for mt in model_order:
        f = run_dir / f"{mt}_carrabin_performance.pkl"
        if not f.exists():
            continue
        perf = pd.read_pickle(f).copy()
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

    sns.boxplot(
        data=plot_df, x="source", y="plot_loss", order=order,
        hue="source", palette=pal, legend=False, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Model fit (RMSE to human responses)")
    ax.tick_params(axis="x", rotation=45)
    sns.despine(ax=ax, top=True, right=True)

    nef_disp = _display("NEF")
    if nef_disp in order:
        annotate_nef_comparisons(
            ax, plot_df, "source", "plot_loss", order,
            nef_label=nef_disp,
            compare_only=["Mean", "LeakyIntegrator", "PrimacyRecency"],
        )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder",   type=str, default="carrabin")
    parser.add_argument("--out_folder",   type=str, default="carrabin")
    parser.add_argument(
        "--extra_models", nargs="*", default=[],
        help="Additional models beyond MODEL_ORDER",
    )
    args = parser.parse_args()

    model_order = MODEL_ORDER + [
        m for m in args.extra_models if m not in MODEL_ORDER
    ]

    apply_style()
    pal     = get_palette(len(model_order) + 1)
    palette = {m: pal[i] for i, m in enumerate(model_order)}
    for mt in model_order:
        disp = _display(mt)
        if disp not in palette:
            palette[disp] = palette[mt]
    palette["Human"] = HUMAN_COLOR

    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_SIZE[0] * 0.75, FIGURE_SIZE[1] / 2),
                             constrained_layout=True)

    _plot_schematic(axes[0])
    _plot_panel_c(axes[1], args.run_folder, palette, model_order)
    _plot_panel_b(axes[2], args.run_folder, palette, model_order)

    label_panels(axes.reshape(1, -1))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure_carrabin_performance"
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
