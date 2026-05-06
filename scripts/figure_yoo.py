#!/usr/bin/env python3
"""Yoo summary figure: row 1 (panels A–D), row 2 reserved (E–H)."""

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

from fitting.losses import (
    POWER_LAW_SMOOTH_WINDOW,
    _fit_power_law_params,
    _smooth_curve,
)
from utils.paths import FIGURES_DIR, data_path
from utils.plot_style import (
    FIGURE_SIZE,
    apply_style,
    get_palette,
    label_panels,
)

# TODO: [decision needed] Row-2 empty panels use minimal decoration only; confirm
# if future panels need a shared aspect or different spine visibility.

MODEL_ORDER = ["Mean", "RL", "ADM", "NEF_recurrent"]

OBS_MAX = 30
# Panel C: line / fit styling
PANEL_C_MEAN_LINE_COLOR = "#1a1a2e"
PANEL_C_GROUP_FIT_COLOR = "#c0392b"
PANEL_C_PID_LINE_COLOR = "0.75"
# Shift whole slope triangle right (observation units) so it clears the curve
PANEL_C_TRIANGLE_SHIFT_N = 2.5


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


def _empty_row_panel(ax) -> None:
    """Reserved row: blank axes (no ticks, spines, or labels)."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")


def _plot_panel_a(ax) -> None:
    """Render first page of figures/yoo_task.pdf into panel A."""
    pdf_path = FIGURES_DIR / "yoo_task.pdf"
    if not pdf_path.exists():
        _empty_pdf_panel(ax)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / "yoo_task"
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


def _yoo_abs_delta_long(human: pd.DataFrame) -> pd.DataFrame:
    """Long-format per-trial |Δresponse| (same trial construction as fitting.losses)."""
    pieces = []
    for (pid, trial), tgrp in human.groupby(["pid", "trial"], sort=False):
        g = tgrp.sort_values("observation").copy()
        g["delta"] = g["response"].diff().abs()
        pieces.append(g)
    if not pieces:
        return pd.DataFrame(columns=["pid", "trial", "observation", "delta"])
    return pd.concat(pieces, ignore_index=True)


def _fit_power_law_to_mean_curve(curve: pd.Series) -> tuple[float, float] | None:
    """Fit A·n^(−λ) to a mean |Δ| curve (smoothing + log-log regression, as in losses.py)."""
    from scipy.stats import linregress

    curve = curve[curve.index >= 2]
    if len(curve) < 3:
        return None
    d = _smooth_curve(curve.values.astype(float), POWER_LAW_SMOOTH_WINDOW)
    n = curve.index.values.astype(float)
    if np.any(d <= 0) or not np.all(np.isfinite(d)):
        return None
    slope, intercept, _, _, _ = linregress(np.log(n), np.log(d))
    return float(np.exp(intercept)), float(-slope)


def _plot_group_slope_triangle(
    ax,
    amp_g: float,
    lam_g: float,
    color: str,
    *,
    n_lo: float = 2.0,
    n_hi: float = 8.0,
    lift_frac: float = 0.25,
    zorder: float = 5,
) -> None:
    """
    Right-angle slope triangle for y = amp_g * n^(-lam_g): horizontal span n_lo..n_hi,
    tangent slope at mid-observation; lifted above the curve by lift_frac * y-axis span.
    """
    if n_hi <= n_lo:
        return

    n_lo = n_lo + PANEL_C_TRIANGLE_SHIFT_N
    n_hi = min(n_hi + PANEL_C_TRIANGLE_SHIFT_N, float(OBS_MAX))

    def f(n):
        return float(amp_g * np.asarray(n, dtype=float) ** (-lam_g))

    def fp(n):
        return float(-lam_g * amp_g * np.asarray(n, dtype=float) ** (-lam_g - 1))

    h_run = n_hi - n_lo
    n_c = 0.5 * (n_lo + n_hi)
    m = fp(n_c)

    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    lift = lift_frac * y_span
    # On [n_lo, n_hi] the group curve is decreasing in n for λ>0; stay above that arc.
    y_max_arc = max(f(n_lo), f(n_hi))
    y_horiz = y_max_arc + lift
    y_tip = y_horiz + m * h_run

    ax.plot(
        [n_lo, n_hi],
        [y_horiz, y_horiz],
        color=color,
        linewidth=0.9,
        zorder=zorder,
        clip_on=False,
        solid_capstyle="round",
    )
    ax.plot(
        [n_hi, n_hi],
        [y_horiz, y_tip],
        color=color,
        linewidth=0.9,
        zorder=zorder,
        clip_on=False,
        solid_capstyle="round",
    )
    # Hypotenuse: top-left (n_lo, y_horiz) to bottom-right (n_hi, y_tip)
    ax.plot(
        [n_lo, n_hi],
        [y_horiz, y_tip],
        color=color,
        linewidth=0.9,
        linestyle=":",
        zorder=zorder,
        clip_on=False,
        solid_capstyle="round",
    )

    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax.text(
        (n_lo + n_hi) / 2,
        y_horiz + 0.018 * y_span,
        r"$n$",
        ha="center",
        va="bottom",
        fontsize=7,
        color=color,
        clip_on=False,
    )
    ax.text(
        n_hi + 0.015 * x_span,
        (y_horiz + y_tip) / 2,
        r"$\Delta r$",
        ha="left",
        va="center",
        fontsize=7,
        color=color,
        clip_on=False,
    )
    cx = (n_lo + n_hi + n_hi) / 3.0
    cy = (y_horiz + y_horiz + y_tip) / 3.0
    ax.text(
        cx,
        cy,
        r"$\lambda$",
        ha="center",
        va="center",
        fontsize=7,
        color=color,
        clip_on=False,
    )

    ymin_cur, ymax_cur = ax.get_ylim()
    pad = 0.02 * max(ymax_cur - ymin_cur, 1e-12)
    new_top = max(ymax_cur, y_horiz + pad)
    ax.set_ylim(0.0, new_top)


def _plot_panel_c(ax, _palette: dict) -> None:
    """
    Group mean |Δresponse| vs observation (seaborn lineplot + CI), group power-law
    overlay, and per-participant power-law curves (fitting.losses logic).
    """
    human_path = data_path("yoo.pkl")
    if not human_path.exists():
        _placeholder(ax, "No human data")
        return

    human = pd.read_pickle(human_path)
    long_df = _yoo_abs_delta_long(human)
    long_df = long_df[long_df["delta"].notna() & (long_df["observation"] >= 2)].copy()
    if long_df.empty:
        _placeholder(ax, "No human data")
        return

    curve_mean = long_df.groupby("observation")["delta"].mean()
    group_fit = _fit_power_law_to_mean_curve(curve_mean)

    try:
        per_pid = _fit_power_law_params(human)
    except Exception:
        per_pid = pd.DataFrame(columns=["pid", "A", "lambda_"])

    n_grid_pid = np.arange(2, OBS_MAX + 1, dtype=float)
    for _, row in per_pid.iterrows():
        lam = float(row["lambda_"])
        amp = float(row["A"])
        if not (np.isfinite(lam) and np.isfinite(amp) and amp > 0):
            continue
        y_pid = amp * (n_grid_pid**-lam)
        ax.plot(
            n_grid_pid,
            y_pid,
            color=PANEL_C_PID_LINE_COLOR,
            alpha=0.28,
            linewidth=0.9,
            zorder=1,
            clip_on=False,
        )

    sns.lineplot(
        data=long_df,
        x="observation",
        y="delta",
        color=PANEL_C_MEAN_LINE_COLOR,
        linewidth=2.0,
        errorbar="ci",
        ax=ax,
        zorder=2,
    )

    if group_fit is not None:
        amp_g, lam_g = group_fit
        n_fine = np.linspace(1.0, float(OBS_MAX), 200)
        y_fit = amp_g * (n_fine**-lam_g)
        ax.plot(
            n_fine,
            y_fit,
            color=PANEL_C_GROUP_FIT_COLOR,
            linewidth=2.2,
            linestyle="-",
            zorder=4,
        )

    ax.set_xlim(0.5, float(OBS_MAX))
    _ymin, ymax = ax.get_ylim()
    ax.set_ylim(0.0, max(ymax, 1e-9))

    ax.set_xlabel("Observation")
    ax.set_ylabel("Response change")
    ax.text(
        0.98,
        0.98,
        r"$\Delta r = \frac{\alpha_0}{n^{\lambda}}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=plt.rcParams.get("font.size", 9),
    )

    if group_fit is not None:
        amp_g, lam_g = group_fit
        n_alpha = 1.0
        y_alpha = amp_g * (n_alpha**-lam_g)
        ax.annotate(
            r"$\alpha_0$",
            xy=(n_alpha, y_alpha),
            xytext=(22, 28),
            textcoords="offset points",
            color=PANEL_C_GROUP_FIT_COLOR,
            fontsize=plt.rcParams.get("font.size", 9),
            fontweight="bold",
            arrowprops=dict(
                arrowstyle="-|>",
                color=PANEL_C_GROUP_FIT_COLOR,
                shrinkA=0,
                shrinkB=3,
                lw=0.8,
                mutation_scale=8,
            ),
            zorder=6,
        )
        _plot_group_slope_triangle(
            ax,
            amp_g,
            lam_g,
            PANEL_C_GROUP_FIT_COLOR,
            n_lo=2.0,
            n_hi=14.0,
            lift_frac=0.1,
            zorder=5,
        )

    sns.despine(ax=ax, top=True, right=True)


def _get_loss(perf_df: pd.DataFrame) -> pd.Series:
    """Return response_component if available, else cv_loss_mean."""
    if "response_component" in perf_df.columns:
        rc = perf_df["response_component"]
        if rc.notna().all():
            return rc
    return perf_df["cv_loss_mean"]


def _plot_panel_b(ax, run_folder: str, palette: dict) -> None:
    """Per-pid RMSE (response loss) distribution — logic from scripts/model_performance.py."""
    run_dir = data_path("runs") / run_folder
    dataset = "yoo"
    rows = []
    for mt in MODEL_ORDER:
        f = run_dir / f"{mt}_{dataset}_performance.pkl"
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

    Copied from scripts/response_change_yoo.py.
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
            if dataset == "jiang":
                params_path = run_dir / f"{mt}_{dataset}_params.pkl"
                if params_path.exists():
                    params_df = pd.read_pickle(params_path)
                    beta_row = params_df[params_df["pid"] == pid]
                    if not beta_row.empty and "beta" in beta_row.columns:
                        params["beta"] = float(beta_row["beta"].iloc[0])
            try:
                loss = losses_mod.shape_loss(params, model_pid, human_pid)
                rows.append({"pid": int(pid), "model_type": mt, "loss": loss})
            except Exception as e:
                print(f"Warning: shape_loss failed for {mt} pid={pid}: {e}")

    return pd.DataFrame(rows)


def _plot_panel_d(ax, run_folder: str, palette: dict) -> None:
    """Shape loss boxplot — colors match scripts/response_change_yoo.py plot_palette."""
    run_dir = data_path("runs") / run_folder
    loss_df = _load_loss_long(run_dir, MODEL_ORDER, "yoo")
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

    order = [_display(m) for m in MODEL_ORDER]
    available = [m for m in order if m in set(df["model_disp"])]
    plot_palette = {
        _display(mt): palette.get(mt, palette.get(_display(mt), "gray"))
        for mt in MODEL_ORDER
    }
    pal = {m: plot_palette[m] for m in available}
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
    ax.set_ylabel("Power law fit error (|ΔA| + |Δλ|)")
    sns.despine(ax=ax, top=True, right=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, default="response")
    args = parser.parse_args()

    apply_style()
    palette = get_palette()
    if "Human" not in palette:
        palette["Human"] = "black"

    fig, axes = plt.subplots(2, 4, figsize=FIGURE_SIZE, constrained_layout=True)
    row0, row1 = axes[0], axes[1]

    _plot_panel_a(row0[0])
    _plot_panel_b(row0[1], args.run_folder, palette)
    _plot_panel_c(row0[2], palette)
    _plot_panel_d(row0[3], args.run_folder, palette)

    for ax in row1:
        _empty_row_panel(ax)

    label_panels(axes)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "figure_yoo.png", dpi=300)
    plt.savefig(FIGURES_DIR / "figure_yoo.pdf")
    print("Saved figures/figure_yoo.{png,pdf}")


if __name__ == "__main__":
    main()
