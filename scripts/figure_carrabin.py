#!/usr/bin/env python3
"""Carrabin summary figure: 2×4 layout, panels A–H."""

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
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, RUNS_DIR, data_path, resolve_run_folder
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette, label_panels, annotate_nef_comparisons

MODEL_ORDER = ["Mean", "LeakyIntegrator", "PrimacyRecency", "NEF"]
MODEL_ORDER_B = MODEL_ORDER
MODEL_ORDER_D = ["Human", "Mean", "LeakyIntegrator", "PrimacyRecency", "NEF"]

HUMAN_NEUTRAL_COLOR = "0.3"

# --- bottom row (E–H): data from scripts/extras_carrabin.py ---
SAMPLE_PIDS = [6, 7]  # high/low alpha_0 example pids for panel 1
MIN_REPEATS = 10  # minimum trial repeats per qid for analysis
READOUT_OFFSET = 0.5  # seconds into obs window for readout
N_NEURONS_LIST = [50, 75, 100, 150, 200, 300, 500]


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


def _empty_pdf_panel(ax) -> None:
    """Panels A/C: bare axes matching embedded-PDF styling, no decorative text."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
    ax.set_anchor("C")


def _plot_panel_a(ax) -> None:
    """Render first page of figures/carrabin_task.pdf into panel A."""
    pdf_path = FIGURES_DIR / "carrabin_task.pdf"
    if not pdf_path.exists():
        _empty_pdf_panel(ax)
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
    # Preserve aspect ratio while maximizing panel coverage.
    ax.set_aspect("equal")
    ax.set_anchor("C")


def _get_loss(perf_df: pd.DataFrame) -> pd.Series:
    # "loss" is the current column name; fall back to "cv_loss_mean"
    # for performance pickles produced before the column rename.
    if "loss" in perf_df.columns:
        return perf_df["loss"]
    return perf_df["cv_loss_mean"]


def _plot_panel_b(ax, run_folder: str, palette: dict, model_order: list[str]) -> None:
    run_dir = data_path("runs") / run_folder
    rows = []
    for mt in model_order:
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
    order = [_display(m) for m in model_order]
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
    ax.tick_params(axis="x", rotation=45)
    sns.despine(ax=ax, top=True, right=True)
    nef_disp = _display("NEF")
    if nef_disp in available:
        annotate_nef_comparisons(ax, df, "model_disp", "plot_loss", available, nef_label=nef_disp)


def _plot_panel_c(ax) -> None:
    """Render first page of figures/response_noise_schematic.pdf into panel C."""
    pdf_path = FIGURES_DIR / "response_noise_schematic.pdf"
    if not pdf_path.exists():
        _empty_pdf_panel(ax)
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


def _load_loss_long(
    run_dir: Path,
    model_order: list[str],
    dataset: str,
) -> pd.DataFrame:
    """
    Load per-pid qid-std shape metric for each model.
    Returns DataFrame with columns: pid, model_type, loss.
    """
    from utils.plot_style import mean_qid_std

    rows = []
    human_full = pd.read_pickle(data_path(f"{dataset}.pkl"))

    for mt in model_order:
        resp_path = run_dir / f"{mt}_{dataset}_responses.pkl"
        if not resp_path.exists():
            print(f"Warning: missing {resp_path.name}, cannot compute loss for {mt}")
            continue
        responses = pd.read_pickle(resp_path)
        for pid, model_pid in responses.groupby("pid"):
            human_pid = human_full[human_full["pid"] == pid]
            qid_map = human_pid[["pid", "trial", "observation", "qid"]].drop_duplicates()
            model_with_qid = model_pid.merge(
                qid_map, on=["pid", "trial", "observation"], how="left"
            )
            loss = abs(mean_qid_std(human_pid) - mean_qid_std(model_with_qid))
            rows.append({"pid": int(pid), "model_type": mt, "loss": loss})

    return pd.DataFrame(rows)


def _plot_panel_d(ax, run_folder: str, palette: dict, model_order: list[str]) -> None:
    """Panel D: KDE of per-participant sigma (RNN residual noise) for each source.

    Human data shown as a filled KDE.
    Stochastic models (NoisyCounting, NEF) shown as KDEs.
    Deterministic models shown as vertical lines at their mean sigma.
    Sources not present in the noise file are silently skipped.
    """
    from scipy.stats import gaussian_kde

    run_dir  = data_path("runs") / run_folder
    noise_f  = run_dir / "RNN_sigma_carrabin_sigma.pkl"
    if not noise_f.exists():
        _placeholder(ax, "No RNN noise data (run models/RNN.py --all_sources)")
        return

    sigma_df = pd.read_pickle(noise_f)

    # Sources treated as distributions (KDE) vs point estimates (vertical line)
    STOCHASTIC = {"human", "NoisyCounting", "NEF"}

    # Display order: human first, then models in model_order
    sources_in_order = ["human"] + [
        m for m in model_order if m in sigma_df["source"].unique()
    ]

    x_max = sigma_df["sigma"].quantile(0.98) * 1.1
    x     = np.linspace(0, x_max, 300)

    for src in sources_in_order:
        sub = sigma_df[sigma_df["source"] == src]["sigma"].dropna()
        if len(sub) == 0:
            continue

        color = palette.get(src, palette.get(_display(src), "0.5"))
        label = "Human" if src == "human" else _display(src)

        if src in STOCHASTIC and len(sub) >= 4:
            # KDE
            kde = gaussian_kde(sub, bw_method="scott")
            density = kde(x)
            if src == "human":
                ax.fill_between(x, density, alpha=0.25, color=color)
                ax.plot(x, density, lw=2, color=color, label=label)
            else:
                ax.plot(x, density, lw=1.5, color=color,
                        linestyle="--", label=label)
        else:
            # Vertical line at mean sigma
            mean_sigma = float(sub.mean())
            ax.axvline(mean_sigma, color=color, lw=1.5,
                       linestyle=":", label=f"{label} (det.)")

    ax.set_xlabel("σ (response noise)")
    ax.set_ylabel("Density")
    ax.set_xlim(left=0)
    ax.legend(fontsize=7, framealpha=0.8)
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel_e(ax, *args, **kwargs) -> None:
    """Placeholder — archived, pending new noise analysis."""
    _placeholder(ax, "Panel E\n(archived)")


def _plot_panel_f(ax, *args, **kwargs) -> None:
    """Placeholder — archived, pending new noise analysis."""
    _placeholder(ax, "Panel F\n(archived)")


def _plot_panel_g(ax, *args, **kwargs) -> None:
    """Placeholder — archived, pending new noise analysis."""
    _placeholder(ax, "Panel G\n(archived)")


def _plot_panel_h(ax, *args, **kwargs) -> None:
    """Placeholder — archived, pending new noise analysis."""
    _placeholder(ax, "Panel H\n(archived)")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_folder",
        type=str,
        default="carrabin",
        help="Source folder for fitted NEF params",
    )
    parser.add_argument("--out_folder", type=str, default="carrabin")
    parser.add_argument(
        "--scan_pid",
        type=int,
        default=None,
        help="Single PID alias for --scan_pids",
    )
    parser.add_argument(
        "--scan_pids",
        type=int,
        nargs="+",
        default=[14],
        help="PIDs to use for n_neurons_scan (default: [14])",
    )
    parser.add_argument(
        "--n_neurons_list",
        type=int,
        nargs="+",
        default=list(N_NEURONS_LIST),
    )
    parser.add_argument(
        "--extra_models",
        nargs="*",
        default=["RNN"],
        help="Additional models to include in top-row panels (default: ['RNN'])",
    )
    args = parser.parse_args()

    if args.scan_pid is not None:
        args.scan_pids = [args.scan_pid]

    model_order = MODEL_ORDER + [m for m in args.extra_models if m not in MODEL_ORDER]

    apply_style()
    _pal = get_palette(len(model_order))
    palette = {m: _pal[i] for i, m in enumerate(model_order)}
    for mt in model_order:
        disp = _display(mt)
        if disp not in palette:
            palette[disp] = palette[mt]
    palette["Human"] = HUMAN_NEUTRAL_COLOR

    fig, axes = plt.subplots(2, 4, figsize=FIGURE_SIZE, constrained_layout=True)
    row0, row1 = axes[0], axes[1]

    _plot_panel_a(row0[0])
    _plot_panel_b(row0[1], args.run_folder, palette, model_order)
    _plot_panel_c(row0[2])
    _plot_panel_d(row0[3], args.run_folder, palette, model_order)

    # ── bottom row: E–H archived, pending new noise analysis ─────────────────
    _plot_panel_e(row1[0])
    _plot_panel_f(row1[1])
    _plot_panel_g(row1[2])
    _plot_panel_h(row1[3])

    label_panels(axes)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "figure_carrabin.png", dpi=300)
    plt.savefig(FIGURES_DIR / "figure_carrabin.pdf")
    plt.savefig(FIGURES_DIR / "figure_carrabin.svg")
    print("Saved figures/figure_carrabin.{png,pdf,svg}")


if __name__ == "__main__":
    main()
