#!/usr/bin/env python3
"""First row of carrabin summary figure (panels A-D)."""

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

from utils.paths import FIGURES_DIR, data_path
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette, label_panels


MODEL_ORDER_B = ["Bayes", "RL", "NoisyCounting", "NEF_recurrent"]
MODEL_ORDER_D = ["Human", "Bayes", "RL", "NoisyCounting", "NEF_recurrent"]


def _display(model_type: str) -> str:
    return "NEF" if model_type.startswith("NEF") else model_type


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


def _plot_panel_a(ax) -> None:
    """Render first page of figures/carrabin_task.pdf into panel A."""
    pdf_path = FIGURES_DIR / "carrabin_task.pdf"
    if not pdf_path.exists():
        _placeholder(ax, "Task structure (manual)")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / "carrabin_task"
        cmd = [
            "pdftoppm",
            "-png",
            "-singlefile",
            str(pdf_path),
            str(out_prefix),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            _placeholder(ax, "Task structure (manual)")
            return
        img_path = out_prefix.with_suffix(".png")
        if not img_path.exists():
            _placeholder(ax, "Task structure (manual)")
            return
        img = mpimg.imread(img_path)

    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Preserve aspect ratio while maximizing panel coverage.
    ax.set_aspect("equal")
    ax.set_anchor("C")


def _plot_panel_c(ax) -> None:
    """Render first page of figures/response_noise_schematic.pdf into panel C."""
    pdf_path = FIGURES_DIR / "response_noise_schematic.pdf"
    if not pdf_path.exists():
        _placeholder(ax, "Response noise explainer (manual)")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / "response_noise_schematic"
        cmd = [
            "pdftoppm",
            "-png",
            "-singlefile",
            str(pdf_path),
            str(out_prefix),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            _placeholder(ax, "Response noise explainer (manual)")
            return
        img_path = out_prefix.with_suffix(".png")
        if not img_path.exists():
            _placeholder(ax, "Response noise explainer (manual)")
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
    """Return response_component if available, else cv_loss_mean."""
    if "response_component" in perf_df.columns:
        rc = perf_df["response_component"]
        if rc.notna().all():
            return rc
    return perf_df["cv_loss_mean"]


def _plot_panel_b(ax, run_folder: str, palette: dict) -> None:
    run_dir = data_path("runs") / run_folder
    rows = []
    for mt in MODEL_ORDER_B:
        f = run_dir / f"{mt}_carrabin_performance.pkl"
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
    order = [_display(m) for m in MODEL_ORDER_B]
    available = [m for m in order if m in set(df["model_disp"])]
    pal = {m: palette.get(m, "0.5") for m in available}
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


def _mean_qid_std_per_pid(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, grp in df.groupby("pid"):
        qid_std = grp.groupby("qid")["response"].std()
        qid_std = qid_std.dropna()
        if len(qid_std) == 0:
            continue
        rows.append({"pid": int(pid), "response_noise": float(qid_std.mean())})
    return pd.DataFrame(rows)


def _load_loss_long(
    run_dir: Path,
    model_order: list[str],
    dataset: str,
) -> pd.DataFrame:
    """
    Load per-pid shape loss for each model.
    Prefers shape_component from performance files when available and
    non-NaN; falls back to recomputing via losses.shape_loss().
    Returns DataFrame with columns: pid, model_type, loss.
    """
    import fitting.losses as losses_mod

    rows = []
    human_full = pd.read_pickle(data_path(f"{dataset}.pkl"))

    for mt in model_order:
        perf_path = run_dir / f"{mt}_{dataset}_performance.pkl"
        resp_path = run_dir / f"{mt}_{dataset}_responses.pkl"
        if not perf_path.exists():
            continue
        perf = pd.read_pickle(perf_path)

        if "shape_component" in perf.columns and perf["shape_component"].notna().all():
            for _, row in perf.iterrows():
                rows.append(
                    {
                        "pid": int(row["pid"]),
                        "model_type": mt,
                        "loss": float(row["shape_component"]),
                    }
                )
            continue

        if not resp_path.exists():
            print(f"Warning: missing {resp_path.name}, cannot compute loss for {mt}")
            continue
        responses = pd.read_pickle(resp_path)
        for pid, model_pid in responses.groupby("pid"):
            human_pid = human_full[human_full["pid"] == pid]
            params = {"dataset": dataset, "pid": int(pid)}
            try:
                loss = losses_mod.shape_loss(params, model_pid, human_pid)
                rows.append({"pid": int(pid), "model_type": mt, "loss": loss})
            except Exception as e:
                print(f"Warning: shape_loss failed for {mt} pid={pid}: {e}")

    return pd.DataFrame(rows)


def _plot_panel_d(ax, run_folder: str, palette: dict) -> None:
    run_dir = data_path("runs") / run_folder
    loss_df = _load_loss_long(run_dir, MODEL_ORDER_B, "carrabin")
    if loss_df.empty:
        _placeholder(ax, "No model data")
        return

    loss_df["model_disp"] = loss_df["model_type"].apply(_display)
    models_with_loss = loss_df["model_disp"].unique().tolist()
    df = loss_df.groupby("pid").filter(
        lambda g: len(g) == len(models_with_loss)
    ).copy()
    if df.empty:
        _placeholder(ax, "No model data")
        return

    order = [_display(m) for m in MODEL_ORDER_B]
    available = [m for m in order if m in set(df["model_disp"])]
    pal = {m: palette.get(m, "0.5") for m in available}
    sns.boxplot(
        data=df,
        x="model_disp",
        y="loss",
        order=available,
        hue="model_disp",
        palette=pal,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Response noise error (sequence-wise difference)")
    sns.despine(ax=ax, top=True, right=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, default="response")
    args = parser.parse_args()

    apply_style()
    palette = get_palette()
    if "Human" not in palette:
        palette["Human"] = "black"

    fig, axes = plt.subplots(1, 4, figsize=FIGURE_SIZE, constrained_layout=True)

    _plot_panel_a(axes[0])
    _plot_panel_b(axes[1], args.run_folder, palette)
    _plot_panel_c(axes[2])
    _plot_panel_d(axes[3], args.run_folder, palette)

    label_panels(axes)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "figure_carrabin.png", dpi=300)
    plt.savefig(FIGURES_DIR / "figure_carrabin.pdf")
    print("Saved figures/figure_carrabin.{png,pdf}")


if __name__ == "__main__":
    main()
