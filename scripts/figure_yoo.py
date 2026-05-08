#!/usr/bin/env python3
"""Yoo summary figure: rows A–D, E–H, I–L."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from scipy.stats import linregress, pearsonr

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

# --- Panel F: λ vs early−late Δ (model behavior + error activity) ---
PANEL_F_MODEL_TYPE = "NEF_recurrent"
PANEL_F_DATASET = "yoo"
PANEL_F_EARLY_OBS = (1, 5)
PANEL_F_LATE_OBS = (26, 30)

# --- Panel G (copied from scripts/response_change_vs_weight_activity.py) ---
PANEL_G_ENCODER_THRESHOLD = 0.5  # minimum enc_dim_0 to be classified as on-weight neuron
PANEL_G_OBS_MIN = 2
PANEL_G_OBS_MAX = 30
PANEL_G_MODEL_TYPE = "NEF_recurrent"
PANEL_G_DATASET = "yoo"

# --- Panel E (copied from scripts/plot_activities.py panels 2 & 3, yoo / NEF_recurrent only)
PANEL_E_ENCODER_THRESHOLD = 0.5
PANEL_E_MODEL_TYPE = "NEF_recurrent"
PANEL_E_PE_COL = "prediction_error_raw"
PANEL_E_OBS_RANGE_YOO = (2, 30)
PANEL_E_COUNTING_OBS_RANGE_YOO = (1, 30)
PANEL_E_LAMBDA_N = 5
PANEL_E_ERROR_STYLE = "ci"


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
    pdf_path = FIGURES_DIR / "yoo_task_wide.pdf"
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
    ax.set_ylabel("Power law fit error (|Δα₀| + |Δλ|)")
    sns.despine(ax=ax, top=True, right=True)


def _panel_e_load_counting_yoo(run_dir: Path) -> tuple[Optional[pd.DataFrame], int]:
    """Mean positive counting-encoder activity vs observation (plot_activities panel 2)."""
    counting_activities_path = run_dir / "activities_counting_yoo.pkl"
    counting_encoders_path = run_dir / "encoders_counting_yoo.pkl"
    if not counting_activities_path.exists() or not counting_encoders_path.exists():
        return None, 0

    activities_df = pd.read_pickle(counting_activities_path)
    encoders_df = pd.read_pickle(counting_encoders_path)
    neuron_cols = [c for c in activities_df.columns if c.startswith("n")]
    for pid, pid_enc in encoders_df.groupby("pid"):
        pos_idx = pid_enc[pid_enc["enc_dim_0"] > PANEL_E_ENCODER_THRESHOLD][
            "neuron_idx"
        ].values
        mask = activities_df["pid"] == pid
        pos_cols = [f"n{i}" for i in pos_idx if f"n{i}" in neuron_cols]
        if pos_cols:
            activities_df.loc[mask, "mean_activity_pos"] = activities_df.loc[
                mask, pos_cols
            ].mean(axis=1)

    if "mean_activity_pos" not in activities_df.columns:
        return None, 0

    obs_min, obs_max = PANEL_E_COUNTING_OBS_RANGE_YOO
    plot_df = activities_df.copy()
    plot_df["obs_plot"] = plot_df["observation"]
    plot_df = plot_df[
        (plot_df["obs_plot"] >= obs_min) & (plot_df["obs_plot"] <= obs_max)
    ]
    if plot_df.empty:
        return None, 0
    n_pids = int(plot_df["pid"].nunique())
    return plot_df[["obs_plot", "mean_activity_pos"]].copy(), n_pids


def _panel_e_load_weight_yoo(
    run_dir: Path,
) -> tuple[Optional[pd.DataFrame], Optional[float], Optional[float]]:
    """
    Error-ensemble weight-on activity split by λ group (plot_activities panel 3).
    Returns (dataframe, low_thresh_lambda, high_thresh_lambda) where thresholds match
    ``lambdas_sorted.iloc[LAMBDA_N-1]`` and ``lambdas_sorted.iloc[-LAMBDA_N]``.
    """
    activities_path = run_dir / "activities_error_yoo.pkl"
    encoders_path = run_dir / "encoders_error_yoo.pkl"
    responses_path = run_dir / f"{PANEL_E_MODEL_TYPE}_yoo_responses.pkl"
    raw_path = data_path("yoo.pkl")
    params_path = run_dir / f"{PANEL_E_MODEL_TYPE}_yoo_params.pkl"
    required = [activities_path, encoders_path, responses_path, raw_path, params_path]
    if not all(p.exists() for p in required):
        return None, None, None

    activities_df = pd.read_pickle(activities_path)
    encoders_df = pd.read_pickle(encoders_path)
    responses_df = pd.read_pickle(responses_path)
    raw_df = pd.read_pickle(raw_path)
    yoo_params = pd.read_pickle(params_path)[["pid", "lambda_"]].drop_duplicates()

    yoo_merged = responses_df.merge(
        raw_df[["pid", "trial", "observation", "value"]],
        on=["pid", "trial", "observation"],
        how="left",
    )
    yoo_merged = yoo_merged.sort_values(["pid", "trial", "observation"])
    yoo_merged["prev_response"] = (
        yoo_merged.groupby(["pid", "trial"])["response"].shift(1).fillna(0.0)
    )
    yoo_merged[PANEL_E_PE_COL] = yoo_merged["value"] - yoo_merged["prev_response"]

    activities_df = activities_df.merge(
        yoo_merged[["pid", "trial", "observation", PANEL_E_PE_COL]],
        on=["pid", "trial", "observation"],
        how="left",
    )

    neuron_cols = [c for c in activities_df.columns if c.startswith("n")]
    for pid, pid_enc in encoders_df.groupby("pid"):
        on_idx = pid_enc[pid_enc["enc_dim_1"] > PANEL_E_ENCODER_THRESHOLD][
            "neuron_idx"
        ].values
        off_idx = pid_enc[pid_enc["enc_dim_1"] < -PANEL_E_ENCODER_THRESHOLD][
            "neuron_idx"
        ].values
        mask = activities_df["pid"] == pid

        on_cols = [f"n{i}" for i in on_idx if f"n{i}" in neuron_cols]
        off_cols = [f"n{i}" for i in off_idx if f"n{i}" in neuron_cols]

        if on_cols:
            activities_df.loc[mask, "mean_activity_on"] = activities_df.loc[
                mask, on_cols
            ].mean(axis=1)
        if off_cols:
            activities_df.loc[mask, "mean_activity_off"] = activities_df.loc[
                mask, off_cols
            ].mean(axis=1)

        weight_on_idx = pid_enc[pid_enc["enc_dim_0"] > PANEL_E_ENCODER_THRESHOLD][
            "neuron_idx"
        ].values
        weight_on_cols = [f"n{i}" for i in weight_on_idx if f"n{i}" in neuron_cols]
        if weight_on_cols:
            activities_df.loc[mask, "mean_activity_weight_on"] = activities_df.loc[
                mask, weight_on_cols
            ].mean(axis=1)

    activities_df = activities_df.merge(yoo_params, on="pid", how="left")

    if (
        "mean_activity_weight_on" not in activities_df.columns
        or "lambda_" not in activities_df.columns
    ):
        return None, None, None

    obs_min, obs_max = PANEL_E_OBS_RANGE_YOO
    plot_df = activities_df[
        (activities_df["observation"] >= obs_min)
        & (activities_df["observation"] <= obs_max)
    ].copy()
    if plot_df.empty:
        return None, None, None

    lambdas_sorted = plot_df.groupby("pid")["lambda_"].first().sort_values()
    if len(lambdas_sorted) < PANEL_E_LAMBDA_N:
        return None, None, None

    low_pids = lambdas_sorted.index[:PANEL_E_LAMBDA_N].tolist()
    high_pids = lambdas_sorted.index[-PANEL_E_LAMBDA_N:].tolist()
    low_thresh_lambda = float(lambdas_sorted.iloc[PANEL_E_LAMBDA_N - 1])
    high_thresh_lambda = float(lambdas_sorted.iloc[-PANEL_E_LAMBDA_N])

    low_df = plot_df[plot_df["pid"].isin(low_pids)].copy()
    high_df = plot_df[plot_df["pid"].isin(high_pids)].copy()

    low_label = f"low (λ<{low_thresh_lambda:.2f}, n={PANEL_E_LAMBDA_N})"
    high_label = f"high (λ>{high_thresh_lambda:.2f}, n={PANEL_E_LAMBDA_N})"
    low_df["lambda_group"] = low_label
    high_df["lambda_group"] = high_label

    out = pd.concat([low_df, high_df], ignore_index=True)
    return out, low_thresh_lambda, high_thresh_lambda


def _plot_panel_e(ax, run_folder: str) -> None:
    """
    Error/weight activity on a single axis (λ groups).
    """
    from matplotlib.lines import Line2D

    run_dir = data_path("runs") / run_folder
    cb_palette = sns.color_palette("colorblind")

    weight_df, low_thr_lambda, high_thr_lambda = _panel_e_load_weight_yoo(run_dir)

    if weight_df is None:
        _placeholder(ax, "No activity data")
        return

    ax_left = ax

    legend_handles: list[Line2D] = []

    if weight_df is not None:
        low_df = weight_df[weight_df["lambda_group"].str.startswith("low")].copy()
        high_df = weight_df[weight_df["lambda_group"].str.startswith("high")].copy()

        sns.lineplot(
            data=low_df,
            x="observation",
            y="mean_activity_weight_on",
            color=cb_palette[0],
            errorbar=PANEL_E_ERROR_STYLE,
            ax=ax_left,
            legend=False,
        )
        sns.lineplot(
            data=high_df,
            x="observation",
            y="mean_activity_weight_on",
            color=cb_palette[1],
            errorbar=PANEL_E_ERROR_STYLE,
            ax=ax_left,
            legend=False,
        )
        ax_left.set_ylabel("Error neuron activity (Hz)")
        legend_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=cb_palette[1],
                    linewidth=2.0,
                    label=(
                        f"High discounting (λ > {high_thr_lambda:.2f}, "
                        f"n={PANEL_E_LAMBDA_N})"
                    ),
                ),
                Line2D(
                    [0],
                    [0],
                    color=cb_palette[0],
                    linewidth=2.0,
                    label=(
                        f"Low discounting (λ < {low_thr_lambda:.2f}, "
                        f"n={PANEL_E_LAMBDA_N})"
                    ),
                ),
            ]
        )

    ax_left.set_xticks(range(0, 31, 5))
    ax_left.set_xlabel("Observation")
    sns.despine(ax=ax_left, top=True, right=True)

    if legend_handles:
        ax_left.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            frameon=False,
            fontsize=plt.rcParams.get("legend.fontsize", 8),
        )


def _plot_g_regplot_panel(
    ax,
    pid_results_subset: list[dict],
    mean_activity: np.ndarray,
    mean_delta: np.ndarray,
    color_sig: str,
    color_nonsig: str,
    title: str = "",
    ylabel: str | None = "Mean on-weight neuron activity (Hz)",
) -> None:
    """Panel 1 from response_change_vs_weight_activity.py: significance + pop mean."""
    n_sig = sum(1 for p in pid_results_subset if p["pval"] < 0.05)
    n_nonsig = len(pid_results_subset) - n_sig

    for pid_data in pid_results_subset:
        pid_df = pd.DataFrame(
            {"delta": pid_data["delta"], "activity": pid_data["activity"]}
        )
        c = color_sig if pid_data["pval"] < 0.05 else color_nonsig
        sns.regplot(
            data=pid_df,
            x="delta",
            y="activity",
            scatter=True,
            line_kws={"color": c, "linewidth": 0.8},
            ci=95,
            ax=ax,
        )

    mean_df = pd.DataFrame({"activity": mean_activity, "delta": mean_delta})
    fin = np.isfinite(mean_df["activity"].values) & np.isfinite(mean_df["delta"].values)
    if fin.sum() >= 2:
        sns.regplot(
            data=mean_df,
            x="delta",
            y="activity",
            scatter=False,
            line_kws={"color": "black", "linewidth": 2.5},
            ci=95,
            ax=ax,
        )

    ax.set_xlabel("Response change")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles = [
        Line2D(
            [0],
            [0],
            color=color_sig,
            linewidth=1.5,
            label=f"Significant (p<0.05, n={n_sig})",
        ),
        Line2D(
            [0],
            [0],
            color=color_nonsig,
            linewidth=1.5,
            label=f"Non-significant (n={n_nonsig})",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.5,
            label="Population mean",
        ),
    ]
    ax.legend(handles=handles, frameon=False)
    sns.despine(ax=ax, top=True, right=True)


def _panel_g_prepare_data(
    run_dir: Path,
) -> tuple[list[dict], np.ndarray, np.ndarray] | None:
    """
    Correlation prep from scripts/response_change_vs_weight_activity.py main().
    Returns (pid_results, mean_activity, mean_delta) or None if inputs missing.
    """
    acts_all_p = run_dir / f"activities_error_{PANEL_G_DATASET}.pkl"
    encs_all_p = run_dir / f"encoders_error_{PANEL_G_DATASET}.pkl"
    human_p = data_path(f"{PANEL_G_DATASET}.pkl")
    if not acts_all_p.exists() or not encs_all_p.exists() or not human_p.exists():
        return None

    acts_all = pd.read_pickle(acts_all_p)
    encs_all = pd.read_pickle(encs_all_p)
    human = pd.read_pickle(human_p)

    nef_params_path = run_dir / f"{PANEL_G_MODEL_TYPE}_{PANEL_G_DATASET}_params.pkl"
    if nef_params_path.exists():
        nef_params = pd.read_pickle(nef_params_path).set_index("pid")
    else:
        nef_params = None

    neuron_cols = [c for c in acts_all.columns if c.startswith("n")]
    obs_range = np.arange(PANEL_G_OBS_MIN, PANEL_G_OBS_MAX + 1, dtype=int)

    pid_results: list[dict] = []
    activity_rows: list[np.ndarray] = []
    delta_rows: list[np.ndarray] = []

    pids = sorted(human["pid"].unique())

    for pid in pids:
        enc_pid = encs_all[encs_all["pid"] == pid]
        on_idx = enc_pid[enc_pid["enc_dim_0"] > PANEL_G_ENCODER_THRESHOLD][
            "neuron_idx"
        ].values
        cols = [f"n{i}" for i in on_idx if f"n{i}" in neuron_cols]
        if not cols:
            continue

        acts_pid = acts_all[acts_all["pid"] == pid].copy()
        acts_pid["mean_weight_on"] = acts_pid[cols].mean(axis=1)

        hum_pid = human[human["pid"] == pid].sort_values(["trial", "observation"])
        hum_pid = hum_pid.copy()
        hum_pid["prev_response"] = hum_pid.groupby("trial")["response"].shift(1)
        hum_pid["delta_abs"] = (hum_pid["response"] - hum_pid["prev_response"]).abs()

        merged = acts_pid.merge(
            hum_pid[["trial", "observation", "delta_abs"]],
            on=["trial", "observation"],
            how="inner",
        )

        g_act = merged.groupby("observation")["mean_weight_on"].mean()
        g_del = merged.groupby("observation")["delta_abs"].mean()

        activity = np.array(
            [float(g_act[o]) if o in g_act.index else np.nan for o in obs_range]
        )
        delta = np.array(
            [float(g_del[o]) if o in g_del.index else np.nan for o in obs_range]
        )

        mask = np.isfinite(activity) & np.isfinite(delta)
        if int(mask.sum()) < 3:
            continue

        slope, _intercept, r_val, pval, _stderr = linregress(
            activity[mask], delta[mask]
        )
        slope = float(slope)
        r_val = float(r_val)
        pval = float(pval)

        if nef_params is not None and pid in nef_params.index:
            lambda_val = float(nef_params.loc[pid, "lambda_"])
        else:
            lambda_val = float("nan")

        pid_results.append(
            {
                "pid": int(pid),
                "delta": delta,
                "activity": activity,
                "slope": slope,
                "r": r_val,
                "pval": pval,
                "lambda_": lambda_val,
            }
        )

        activity_rows.append(activity)
        delta_rows.append(delta)

    if not pid_results:
        return None

    mean_activity = np.nanmean(activity_rows, axis=0)
    mean_delta = np.nanmean(delta_rows, axis=0)
    return pid_results, mean_activity, mean_delta


def _panel_f_prepare_early_late(run_dir: Path) -> Optional[pd.DataFrame]:
    """
    Per-pid fitted λ, Δ|Δresponse| (model, early−late obs), and Δ error activity.
    """
    responses_path = run_dir / f"{PANEL_F_MODEL_TYPE}_{PANEL_F_DATASET}_responses.pkl"
    activities_path = run_dir / f"activities_error_{PANEL_F_DATASET}.pkl"
    params_path = run_dir / f"{PANEL_F_MODEL_TYPE}_{PANEL_F_DATASET}_params.pkl"
    if not all(p.exists() for p in (responses_path, activities_path, params_path)):
        return None

    responses_all = pd.read_pickle(responses_path)
    acts_all = pd.read_pickle(activities_path)
    params = pd.read_pickle(params_path)
    if "pid" not in params.columns or "lambda_" not in params.columns:
        return None
    params = params[["pid", "lambda_"]].drop_duplicates(subset=["pid"], keep="first")

    neuron_cols = [c for c in acts_all.columns if c.startswith("n")]
    if not neuron_cols:
        return None

    lo_e, hi_e = PANEL_F_EARLY_OBS
    lo_l, hi_l = PANEL_F_LATE_OBS
    rows: list[dict] = []

    for pid in sorted(responses_all["pid"].unique()):
        pr = params[params["pid"] == pid]
        if pr.empty:
            continue
        lam = float(pr["lambda_"].iloc[0])

        rpid = responses_all[responses_all["pid"] == pid].sort_values(
            ["trial", "observation"]
        )
        rpid = rpid.copy()
        rpid["prev_response"] = rpid.groupby("trial")["response"].shift(1)
        rpid["delta_abs"] = (rpid["response"] - rpid["prev_response"]).abs()
        early_beh = rpid[rpid["observation"].between(lo_e, hi_e)]["delta_abs"].mean()
        late_beh = rpid[rpid["observation"].between(lo_l, hi_l)]["delta_abs"].mean()
        if not (np.isfinite(early_beh) and np.isfinite(late_beh)):
            continue
        beh_delta = float(early_beh - late_beh)

        ap = acts_all[acts_all["pid"] == pid]
        if ap.empty:
            continue
        ap = ap.copy()
        ap["mean_error"] = ap[neuron_cols].mean(axis=1)
        early_neu = ap[ap["observation"].between(lo_e, hi_e)]["mean_error"].mean()
        late_neu = ap[ap["observation"].between(lo_l, hi_l)]["mean_error"].mean()
        if not (np.isfinite(early_neu) and np.isfinite(late_neu)):
            continue
        neural_delta = float(early_neu - late_neu)

        rows.append(
            {
                "pid": int(pid),
                "lambda_": lam,
                "beh_delta": beh_delta,
                "neural_delta": neural_delta,
            }
        )

    if not rows:
        return None
    return pd.DataFrame(rows)


def _plot_panel_f(ax, run_folder: str) -> None:
    """Fitted λ vs early−late Δ (model |Δresponse| left, error activity right)."""
    run_dir = data_path("runs") / run_folder
    df = _panel_f_prepare_early_late(run_dir)
    if df is None or df.empty:
        _placeholder(ax, "No data")
        return

    cb = sns.color_palette("colorblind")
    c_beh = cb[0]
    lw = plt.rcParams.get("lines.linewidth", 1.5)

    plot_df = df[["lambda_", "beh_delta"]].copy()
    sns.regplot(
        data=plot_df,
        x="lambda_",
        y="beh_delta",
        ax=ax,
        scatter=True,
        line_kws={
            "color": c_beh,
            "linewidth": lw,
        },
        scatter_kws={
            "color": c_beh,
            "s": 28,
            "edgecolors": c_beh,
            "linewidths": 0.5,
            "zorder": 4,
        },
    )
    ax.set_xlabel("Fitted λ")
    ax.set_ylabel("Δ response change (early − late)")
    ax.tick_params(axis="y")
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel_g(
    ax, run_folder: str, panel_g_show_significance: bool = False
) -> None:
    """Single plot from scripts/response_change_vs_weight_activity.py."""
    run_dir = data_path("runs") / run_folder
    prep = _panel_g_prepare_data(run_dir)
    if prep is None:
        _placeholder(ax, "No activity data")
        return
    pid_results, mean_activity, mean_delta = prep
    if not pid_results:
        _placeholder(ax, "No activity data")
        return

    pal = get_palette()
    if panel_g_show_significance:
        _plot_g_regplot_panel(
            ax,
            pid_results,
            mean_activity,
            mean_delta,
            pal["Bayes"],
            pal["RL"],
            title="",
            ylabel="Error neuron activity (Hz)",
        )
        return

    mean_df = pd.DataFrame({"activity": mean_activity, "delta": mean_delta})
    fin = np.isfinite(mean_df["activity"].values) & np.isfinite(mean_df["delta"].values)
    if fin.sum() >= 2:
        sns.regplot(
            data=mean_df,
            x="activity",
            y="delta",
            scatter=True,
            line_kws={"color": pal["Bayes"], "linewidth": 2.5},
            scatter_kws={"alpha": 0.6, "s": 20, "color": pal["Bayes"]},
            ci=95,
            truncate=True,
            ax=ax,
        )
    ax.set_xlabel("Mean error neuron activity (Hz)")
    ax.set_ylabel("Mean response change")
    ax.set_title("")
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel_h(ax, run_folder: str) -> None:
    """Panel H: 2×2 human/NEF heatmaps of task error and response change by pid."""
    human_path = data_path("yoo.pkl")
    if not human_path.exists():
        _empty_row_panel(ax)
        return

    human = pd.read_pickle(human_path)
    if human.empty:
        _empty_row_panel(ax)
        return

    # Reuse existing human per-pid power-law fitting logic used by panel C.
    try:
        fit_df = _fit_power_law_params(human)
    except Exception:
        fit_df = pd.DataFrame(columns=["pid", "lambda_"])
    if fit_df.empty or not {"pid", "lambda_"}.issubset(fit_df.columns):
        _empty_row_panel(ax)
        return

    # Human: per-trial running-mean task error and response change, then mean across trials.
    rows_task: list[dict] = []
    rows_delta: list[dict] = []
    for (pid, trial), tdf in human.groupby(["pid", "trial"]):
        tdf = tdf.sort_values("observation").copy()
        running_mean = tdf["value"].expanding().mean()
        task_error = (tdf["response"] - running_mean).abs()
        delta = tdf["response"].diff().abs()
        for obs, te in zip(tdf["observation"], task_error):
            if np.isfinite(te):
                rows_task.append(
                    {"pid": int(pid), "observation": int(obs), "task_error": float(te)}
                )
        for obs, d in zip(tdf["observation"], delta):
            if np.isfinite(d):
                rows_delta.append(
                    {"pid": int(pid), "observation": int(obs), "response_change": float(d)}
                )

    if not rows_task or not rows_delta:
        _empty_row_panel(ax)
        return

    human_task_df = pd.DataFrame(rows_task).groupby(
        ["pid", "observation"], as_index=False
    )["task_error"].mean()
    human_delta_df = pd.DataFrame(rows_delta).groupby(
        ["pid", "observation"], as_index=False
    )["response_change"].mean()

    lam_by_pid = (
        fit_df[["pid", "lambda_"]]
        .dropna()
        .drop_duplicates(subset=["pid"], keep="first")
        .set_index("pid")["lambda_"]
    )
    pids_sorted = [int(p) for p in lam_by_pid.sort_values().index.tolist()]
    if not pids_sorted:
        _empty_row_panel(ax)
        return

    # Model: load per-pid response files, compute same metrics, then mean across trials.
    run_dir = data_path("runs") / run_folder
    value_map = human[["pid", "trial", "observation", "value"]].drop_duplicates()
    model_rows_task: list[dict] = []
    model_rows_delta: list[dict] = []
    model_files = sorted(run_dir.glob("NEF_recurrent_yoo_*_responses.pkl"))
    for f in model_files:
        stem = f.stem
        parts = stem.split("_")
        if len(parts) < 5:
            continue
        pid_token = parts[-2]
        if not pid_token.isdigit():
            continue
        pid = int(pid_token)
        mdf = pd.read_pickle(f)
        if mdf.empty:
            continue
        if not {"pid", "trial", "observation", "response"}.issubset(mdf.columns):
            continue
        # Model response pickles do not contain stimulus values; merge from human yoo data.
        mdf = mdf.merge(
            value_map[value_map["pid"] == pid],
            on=["pid", "trial", "observation"],
            how="left",
        )
        if "value" not in mdf.columns:
            continue
        for (_pid, trial), tdf in mdf.groupby(["pid", "trial"]):
            tdf = tdf.sort_values("observation").copy()
            tdf = tdf[tdf["value"].notna()].copy()
            if tdf.empty:
                continue
            running_mean = tdf["value"].expanding().mean()
            task_error = (tdf["response"] - running_mean).abs()
            delta = tdf["response"].diff().abs()
            for obs, te in zip(tdf["observation"], task_error):
                if np.isfinite(te):
                    model_rows_task.append(
                        {"pid": int(pid), "observation": int(obs), "task_error": float(te)}
                    )
            for obs, d in zip(tdf["observation"], delta):
                if np.isfinite(d):
                    model_rows_delta.append(
                        {
                            "pid": int(pid),
                            "observation": int(obs),
                            "response_change": float(d),
                        }
                    )

    if not model_rows_task or not model_rows_delta:
        _empty_row_panel(ax)
        return

    model_task_df = pd.DataFrame(model_rows_task).groupby(
        ["pid", "observation"], as_index=False
    )["task_error"].mean()
    model_delta_df = pd.DataFrame(model_rows_delta).groupby(
        ["pid", "observation"], as_index=False
    )["response_change"].mean()

    nef_params_path = run_dir / "NEF_recurrent_yoo_params.pkl"
    if not nef_params_path.exists():
        _empty_row_panel(ax)
        return
    nef_params = pd.read_pickle(nef_params_path)
    if not {"pid", "lambda_"}.issubset(nef_params.columns):
        _empty_row_panel(ax)
        return
    nef_lambda_by_pid = (
        nef_params[["pid", "lambda_"]]
        .dropna()
        .drop_duplicates(subset=["pid"], keep="first")
        .set_index("pid")["lambda_"]
    )
    pids_sorted_nef = [int(p) for p in nef_lambda_by_pid.sort_values().index.tolist()]
    if not pids_sorted_nef:
        _empty_row_panel(ax)
        return

    obs_cols = list(range(1, 31))
    human_task_mat = (
        human_task_df.pivot(index="pid", columns="observation", values="task_error")
        .reindex(index=pids_sorted, columns=obs_cols)
    )
    human_delta_mat = (
        human_delta_df.pivot(index="pid", columns="observation", values="response_change")
        .reindex(index=pids_sorted, columns=obs_cols)
    )
    model_task_mat = (
        model_task_df.pivot(index="pid", columns="observation", values="task_error")
        .reindex(index=pids_sorted_nef, columns=obs_cols)
    )
    model_delta_mat = (
        model_delta_df.pivot(index="pid", columns="observation", values="response_change")
        .reindex(index=pids_sorted_nef, columns=obs_cols)
    )

    human_task_vals = human_task_mat.to_numpy().ravel()
    human_task_vals = human_task_vals[np.isfinite(human_task_vals)]
    human_delta_vals = human_delta_mat.to_numpy().ravel()
    human_delta_vals = human_delta_vals[np.isfinite(human_delta_vals)]
    model_task_vals = model_task_mat.to_numpy().ravel()
    model_task_vals = model_task_vals[np.isfinite(model_task_vals)]
    model_delta_vals = model_delta_mat.to_numpy().ravel()
    model_delta_vals = model_delta_vals[np.isfinite(model_delta_vals)]
    if (
        human_task_vals.size == 0
        or human_delta_vals.size == 0
        or model_task_vals.size == 0
        or model_delta_vals.size == 0
    ):
        _empty_row_panel(ax)
        return
    vmin_human_task, vmax_human_task = float(human_task_vals.min()), float(
        human_task_vals.max()
    )
    vmin_human_delta, vmax_human_delta = float(human_delta_vals.min()), float(
        human_delta_vals.max()
    )
    vmin_model_task, vmax_model_task = float(model_task_vals.min()), float(
        model_task_vals.max()
    )
    vmin_model_delta, vmax_model_delta = float(model_delta_vals.min()), float(
        model_delta_vals.max()
    )

    fig = ax.figure
    parent_spec = ax.get_subplotspec()
    ax.remove()
    inner = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=parent_spec, wspace=0.02, hspace=0.01
    )
    ax_left_human = fig.add_subplot(inner[0, 0])
    ax_right_human = fig.add_subplot(inner[0, 1], sharey=ax_left_human)
    ax_left_model = fig.add_subplot(inner[1, 0], sharex=ax_left_human)
    ax_right_model = fig.add_subplot(inner[1, 1], sharex=ax_right_human, sharey=ax_left_model)

    hm_left_human = sns.heatmap(
        human_task_mat,
        cmap="viridis",
        cbar=False,
        vmin=vmin_human_task,
        vmax=vmax_human_task,
        ax=ax_left_human,
    )
    sns.heatmap(
        human_delta_mat,
        cmap="viridis",
        cbar=False,
        vmin=vmin_human_delta,
        vmax=vmax_human_delta,
        ax=ax_right_human,
    )
    hm_left_model = sns.heatmap(
        model_task_mat,
        cmap="viridis",
        cbar=False,
        vmin=vmin_model_task,
        vmax=vmax_model_task,
        ax=ax_left_model,
    )
    hm_right_model = sns.heatmap(
        model_delta_mat,
        cmap="viridis",
        cbar=False,
        vmin=vmin_model_delta,
        vmax=vmax_model_delta,
        ax=ax_right_model,
    )

    fig.colorbar(
        hm_left_model.collections[0],
        ax=[ax_left_human, ax_left_model],
        orientation="horizontal",
        location="bottom",
        pad=0.10,
        shrink=0.8,
    )
    fig.colorbar(
        hm_right_model.collections[0],
        ax=[ax_right_human, ax_right_model],
        orientation="horizontal",
        location="bottom",
        pad=0.10,
        shrink=0.8,
    )

    rep_idx = np.linspace(0, len(pids_sorted) - 1, 5, dtype=int)
    rep_idx_set = set(int(i) for i in rep_idx.tolist())
    labels = [
        (f"λ={lam_by_pid.loc[pid]:.2f}" if i in rep_idx_set else "")
        for i, pid in enumerate(pids_sorted)
    ]
    y_pos_human = np.arange(len(pids_sorted)) + 0.5
    ax_left_human.set_yticks(y_pos_human)
    ax_left_human.set_yticklabels(labels, fontsize=6, rotation=0)

    rep_idx_nef = np.linspace(0, len(pids_sorted_nef) - 1, 5, dtype=int)
    rep_idx_nef_set = set(int(i) for i in rep_idx_nef.tolist())
    labels_nef = [
        (f"λ={nef_lambda_by_pid.loc[pid]:.1f}" if i in rep_idx_nef_set else "")
        for i, pid in enumerate(pids_sorted_nef)
    ]
    y_pos_nef = np.arange(len(pids_sorted_nef)) + 0.5
    ax_left_model.set_yticks(y_pos_nef)
    ax_left_model.set_yticklabels(labels_nef, fontsize=6, rotation=0)
    ax_right_human.tick_params(axis="y", left=True, labelleft=False)
    ax_right_model.tick_params(axis="y", left=True, labelleft=False)

    x_pos = [0.5, 9.5, 19.5, 29.5]
    x_lab = ["1", "10", "20", "30"]
    ax_left_human.set_xticks(x_pos)
    ax_left_human.set_xticklabels([])
    ax_right_human.set_xticks(x_pos)
    ax_right_human.set_xticklabels([])
    ax_left_human.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_right_human.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_left_human.set_xlabel("")
    ax_right_human.set_xlabel("")

    ax_left_model.set_xticks(x_pos)
    ax_left_model.set_xticklabels(x_lab, rotation=0)
    ax_right_model.set_xticks(x_pos)
    ax_right_model.set_xticklabels(x_lab, rotation=0)
    ax_left_model.set_xlabel("Observation")
    ax_right_model.set_xlabel("Observation")

    ax_left_human.set_title("Task error")
    ax_right_human.set_title("Response change")
    ax_left_model.set_title("")
    ax_right_model.set_title("")

    for heat_ax in (ax_left_human, ax_right_human, ax_left_model, ax_right_model):
        heat_ax.tick_params(axis="y", length=0)
        heat_ax.tick_params(axis="x", length=0)
        heat_ax.set_ylabel("")

    ax_left_human.annotate(
        "Human",
        xy=(-0.35, 0.5),
        xycoords="axes fraction",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
    )
    ax_left_model.annotate(
        "NEF",
        xy=(-0.35, 0.5),
        xycoords="axes fraction",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
    )


def _plot_panel_k(ax, noise_folder: str, run_folder: str, palette: dict) -> None:
    """Response noise trajectory (drawn in panel K); multi-seed NEF responses."""
    noise_run_dir = data_path("runs") / noise_folder
    noise_path = noise_run_dir / "NEF_recurrent_yoo_all_responses.pkl"
    if not noise_path.exists():
        _empty_row_panel(ax)
        return

    noise_df = pd.read_pickle(noise_path)
    required = {"pid", "trial", "observation", "response"}
    if not required.issubset(noise_df.columns):
        _empty_row_panel(ax)
        return

    per_qid_noise = (
        noise_df.groupby(["pid", "trial", "observation"], as_index=False)["response"]
        .agg(lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0)))
        .rename(columns={"response": "response_noise"})
    )
    if per_qid_noise.empty:
        _empty_row_panel(ax)
        return

    split_run_dir = data_path("runs") / run_folder
    weight_df, low_thr_lambda, high_thr_lambda = _panel_e_load_weight_yoo(split_run_dir)
    if weight_df is None or high_thr_lambda is None:
        _empty_row_panel(ax)
        return

    low_pids = set(
        weight_df[weight_df["lambda_group"].str.startswith("low")]["pid"]
        .dropna()
        .astype(int)
        .tolist()
    )
    high_pids = set(
        weight_df[weight_df["lambda_group"].str.startswith("high")]["pid"]
        .dropna()
        .astype(int)
        .tolist()
    )
    if not low_pids or not high_pids:
        _empty_row_panel(ax)
        return

    low_noise = per_qid_noise[per_qid_noise["pid"].isin(low_pids)].copy()
    high_noise = per_qid_noise[per_qid_noise["pid"].isin(high_pids)].copy()
    if low_noise.empty or high_noise.empty:
        _empty_row_panel(ax)
        return

    cb_palette = sns.color_palette("colorblind")
    # Same convention as panel C: seaborn computes mean line + default CI region.
    sns.lineplot(
        data=high_noise,
        x="observation",
        y="response_noise",
        color=cb_palette[1],
        label=f"High discounting (λ > {high_thr_lambda:.2f})",
        ax=ax,
    )
    sns.lineplot(
        data=low_noise,
        x="observation",
        y="response_noise",
        color=cb_palette[0],
        label=f"Low discounting (λ < {low_thr_lambda:.2f})",
        ax=ax,
    )
    ax.set_xlabel("Observation")
    ax.set_ylabel("Predicted response noise")
    ax.legend(frameon=False)
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel_i(ax, run_folder: str) -> None:
    """α₀ vs λ correlation from fitted NEF_recurrent yoo params."""
    run_dir = data_path("runs") / run_folder
    params_path = run_dir / "NEF_recurrent_yoo_params.pkl"
    if not params_path.exists():
        _placeholder(ax, "No params data")
        return
    params = pd.read_pickle(params_path)
    if not {"alpha_0", "lambda_"}.issubset(params.columns):
        _placeholder(ax, "No params data")
        return
    plot_df = (
        params.groupby("pid")[["alpha_0", "lambda_"]]
        .first()
        .dropna()
    )
    if len(plot_df) < 2:
        _placeholder(ax, "No params data")
        return

    cb0 = sns.color_palette("colorblind")[0]
    sns.regplot(
        data=plot_df,
        x="alpha_0",
        y="lambda_",
        scatter=True,
        truncate=True,
        scatter_kws={"alpha": 0.6, "s": 20, "color": cb0},
        line_kws={"color": cb0, "linewidth": 2.0},
        ci=95,
        ax=ax,
    )
    r_val, p_val = pearsonr(
        plot_df["alpha_0"].to_numpy(dtype=float),
        plot_df["lambda_"].to_numpy(dtype=float),
    )
    ax.text(
        0.02,
        0.98,
        f"r = {float(r_val):.2f}, p = {float(p_val):.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    ax.set_xlabel("Fitted α₀")
    ax.set_ylabel("Fitted λ")
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    sns.despine(ax=ax, top=True, right=True)


def _panel_j_split_by_alpha0(
    run_dir: Path,
) -> tuple[set[int], set[int], float, float] | None:
    """Top/bottom PANEL_E_LAMBDA_N pids by α₀; thresholds match panel E λ-split style."""
    params_path = run_dir / "NEF_recurrent_yoo_params.pkl"
    if not params_path.exists():
        return None
    params = pd.read_pickle(params_path)
    if "alpha_0" not in params.columns:
        return None
    alphas_sorted = params.groupby("pid")["alpha_0"].first().sort_values()
    if len(alphas_sorted) < PANEL_E_LAMBDA_N:
        return None
    low_pids = set(int(p) for p in alphas_sorted.index[:PANEL_E_LAMBDA_N].tolist())
    high_pids = set(int(p) for p in alphas_sorted.index[-PANEL_E_LAMBDA_N:].tolist())
    low_thr = float(alphas_sorted.iloc[PANEL_E_LAMBDA_N - 1])
    high_thr = float(alphas_sorted.iloc[-PANEL_E_LAMBDA_N])
    return low_pids, high_pids, low_thr, high_thr


def _plot_panel_j(ax, noise_folder: str, run_folder: str, _palette: dict) -> None:
    """Same as panel H but split by high/low α₀ instead of λ."""
    noise_run_dir = data_path("runs") / noise_folder
    noise_path = noise_run_dir / "NEF_recurrent_yoo_all_responses.pkl"
    if not noise_path.exists():
        _empty_row_panel(ax)
        return

    noise_df = pd.read_pickle(noise_path)
    required = {"pid", "trial", "observation", "response"}
    if not required.issubset(noise_df.columns):
        _empty_row_panel(ax)
        return

    per_qid_noise = (
        noise_df.groupby(["pid", "trial", "observation"], as_index=False)["response"]
        .agg(lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0)))
        .rename(columns={"response": "response_noise"})
    )
    if per_qid_noise.empty:
        _empty_row_panel(ax)
        return

    split_run_dir = data_path("runs") / run_folder
    split = _panel_j_split_by_alpha0(split_run_dir)
    if split is None:
        _empty_row_panel(ax)
        return
    low_pids, high_pids, low_thr, high_thr = split
    if not low_pids or not high_pids:
        _empty_row_panel(ax)
        return

    low_noise = per_qid_noise[per_qid_noise["pid"].isin(low_pids)].copy()
    high_noise = per_qid_noise[per_qid_noise["pid"].isin(high_pids)].copy()
    if low_noise.empty or high_noise.empty:
        _empty_row_panel(ax)
        return

    cb_palette = sns.color_palette("colorblind")
    sns.lineplot(
        data=high_noise,
        x="observation",
        y="response_noise",
        color=cb_palette[1],
        label=f"High initial learning (α₀ > {high_thr:.2f})",
        ax=ax,
    )
    sns.lineplot(
        data=low_noise,
        x="observation",
        y="response_noise",
        color=cb_palette[0],
        label=f"Low initial learning (α₀ < {low_thr:.2f})",
        ax=ax,
    )
    ax.set_xlabel("Observation")
    ax.set_ylabel("Predicted response noise")
    ax.legend(frameon=False)
    sns.despine(ax=ax, top=True, right=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, default="response")
    parser.add_argument("--noise_folder", type=str, default="yoo_response_noise")
    parser.add_argument("--panel_g_show_significance", action="store_true", default=False)
    args = parser.parse_args()

    apply_style()
    palette = get_palette()
    if "Human" not in palette:
        palette["Human"] = "black"

    fig, axes = plt.subplots(3, 4, figsize=FIGURE_SIZE, constrained_layout=True)
    row0, row1, row2 = axes[0], axes[1], axes[2]

    _plot_panel_a(row0[0])
    _plot_panel_b(row0[1], args.run_folder, palette)
    _plot_panel_c(row0[2], palette)
    _plot_panel_d(row0[3], args.run_folder, palette)

    _plot_panel_e(row1[0], args.run_folder)
    _plot_panel_f(row1[1], args.run_folder)
    _plot_panel_g(row1[2], args.run_folder, args.panel_g_show_significance)
    _plot_panel_h(row1[3], args.run_folder)

    _plot_panel_i(row2[0], args.run_folder)
    _plot_panel_j(row2[1], args.noise_folder, args.run_folder, palette)
    _plot_panel_k(row2[2], args.noise_folder, args.run_folder, palette)
    _empty_row_panel(row2[3])

    label_panels(axes)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "figure_yoo.png", dpi=300)
    plt.savefig(FIGURES_DIR / "figure_yoo.pdf")
    print("Saved figures/figure_yoo.{png,pdf}")


if __name__ == "__main__":
    main()
