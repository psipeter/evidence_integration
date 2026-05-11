#!/usr/bin/env python3
"""Usher summary figure: row 1 (panels A–D), row 2 reserved (E–H)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, data_path
from utils.plot_style import (
    FIGURE_SIZE,
    annotate_violins,
    apply_style,
    get_palette,
    label_panels,
)

MODEL_ORDER = ["Mean", "RL", "PopulationCoding"]
DATASET = "usher"


def _display(model_type: str) -> str:
    if model_type.startswith("NEF"):
        return "NEF"
    if model_type == "PopulationCoding":
        return "PopCode"
    return model_type


def _placeholder(ax, text: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="0.5",
        style="italic",
    )


def _empty_pdf_panel(ax) -> None:
    """Bare axes when PDF is missing or conversion fails."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
    ax.set_anchor("C")


def _blank_panel(ax) -> None:
    """Reserved / placeholder: no ticks, spines, or text."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")


def _plot_panel_a(ax) -> None:
    """Render first page of figures/usher_task.pdf into panel A."""
    pdf_path = FIGURES_DIR / "usher_task.pdf"
    if not pdf_path.exists():
        _empty_pdf_panel(ax)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / "usher_task"
        cmd = [
            "pdftoppm",
            "-png",
            "-singlefile",
            str(pdf_path),
            str(out_prefix),
        ]
        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            _empty_pdf_panel(ax)
            return
        img_path = out_prefix.with_suffix(".png")
        if not img_path.exists():
            _empty_pdf_panel(ax)
            return
        img = mpimg.imread(img_path)

    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
    ax.set_anchor("C")


def _get_loss(perf_df: pd.DataFrame) -> pd.Series:
    """Return response_component if available, else cv_loss_mean (model_performance)."""
    if "response_component" in perf_df.columns:
        rc = perf_df["response_component"]
        if rc.notna().all():
            return rc
    return perf_df["cv_loss_mean"]


def _plot_panel_b(ax, run_folder: str, palette: dict) -> None:
    """Per-pid RMSE distribution per model (logic aligned with scripts/model_performance.py)."""
    run_dir = data_path("runs") / run_folder
    rows = []
    for mt in MODEL_ORDER:
        f = run_dir / f"{mt}_{DATASET}_performance.pkl"
        if not f.exists():
            continue
        perf = pd.read_pickle(f).copy()
        perf["plot_loss"] = _get_loss(perf)
        perf["model_disp"] = _display(mt)
        rows.append(perf[["pid", "model_disp", "plot_loss"]])

    if not rows:
        _placeholder(ax, "No performance data")
        return

    df = pd.concat(rows, ignore_index=True)
    order = [_display(m) for m in MODEL_ORDER]
    available = [m for m in order if m in set(df["model_disp"])]
    pal = {}
    for m in available:
        if m == "PopCode":
            # TODO: add PopulationCoding to get_palette() in plot_style.py; model_performance
            #  uses get_palette() keys only — no PopulationCoding entry yet
            pal[m] = palette.get(
                "PopulationCoding",
                sns.color_palette("colorblind")[4],
            )
        else:
            pal[m] = palette.get(m, "0.5")
    sns.boxplot(
        data=df,
        x="model_disp",
        y="plot_loss",
        order=available,
        hue="model_disp",
        palette=pal,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Response error (trial-wise RMSE)")
    sns.despine(ax=ax, top=True, right=True)

    if len(available) >= 2:
        annotate_violins(ax, df, "model_disp", "plot_loss", available)


def main() -> None:
    parser = argparse.ArgumentParser(description="Usher summary figure (panels A–H).")
    parser.add_argument(
        "--run_folder",
        type=str,
        default="response",
        help="Run folder under data/runs/ for performance pickles",
    )
    args = parser.parse_args()

    apply_style()
    palette = get_palette()

    fig, axes = plt.subplots(2, 4, figsize=FIGURE_SIZE, constrained_layout=True)
    row0, row1 = axes[0], axes[1]

    _plot_panel_a(row0[0])
    _plot_panel_b(row0[1], args.run_folder, palette)
    _blank_panel(row0[2])
    _blank_panel(row0[3])
    for ax in row1:
        _blank_panel(ax)

    label_panels(axes)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "figure_usher.png", dpi=300)
    plt.savefig(FIGURES_DIR / "figure_usher.pdf")
    plt.savefig(FIGURES_DIR / "figure_usher.svg")
    print("Saved figures/figure_usher.{png,pdf,svg}")


if __name__ == "__main__":
    main()
