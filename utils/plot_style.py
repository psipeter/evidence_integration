"""
Shared plotting style for all notebooks and figures.

Import and call apply_style() at the top of every notebook to ensure
consistent aesthetics across figures. Optimized for readability in
JupyterLab and crispness when loaded into LaTeX/Overleaf as PDF.
"""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D  # noqa: F401 (re-exported for convenience)
from scipy.stats import wilcoxon


def apply_style() -> None:
    """Apply project-wide matplotlib and seaborn style defaults."""
    sns.set_theme(style="ticks")
    plt.rcParams.update(
        {
            # figure
            "figure.dpi": 150,
            "savefig.dpi": 300,
            # fonts
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            # lines and markers
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 2.5,
            # saving
            "savefig.bbox": "tight",
            "savefig.transparent": False,
        }
    )


# -- shared figure constants ---------------------------------------------------

FIGURE_SIZE = (14, 7)
POWER_LAW_SMOOTH_WINDOW = 5  # smoothing window for power-law fits in yoo figures
QID_MIN_TRIALS = 10  # minimum trials per qid in carrabin qid-std diagnostic


def mean_qid_std(df: pd.DataFrame, qid_min_trials: int = QID_MIN_TRIALS) -> float:
    """
    Mean per-qid response std for carrabin diagnostics, using only qids with
    at least qid_min_trials trials. Returns nan if no valid qids.
    """
    counts = df.groupby("qid")["trial"].nunique()
    valid_qids = counts[counts >= qid_min_trials].index
    if len(valid_qids) == 0:
        return float("nan")
    stds = df[df["qid"].isin(valid_qids)].groupby("qid")["response"].std()
    return float(stds.mean())


def smooth_curve(arr: np.ndarray, window: int) -> np.ndarray:
    """Apply centered rolling average of given window size to 1D array."""
    if window <= 1:
        return arr
    result = arr.astype(float).copy()
    half = window // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        result[i] = float(arr[lo:hi].mean())
    return result


def fit_power_law_params(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a power law A * n^(-lambda) to each pid's smoothed mean |delta response|
    curve. Returns DataFrame with columns: pid, A, lambda_.
    """
    from scipy.stats import linregress

    rows = []
    for pid, grp in df.groupby("pid"):
        pieces = []
        for _, tgrp in grp.groupby("trial"):
            g = tgrp.sort_values("observation").copy()
            g["delta"] = g["response"].diff().abs()
            pieces.append(g)
        delta = pd.concat(pieces, ignore_index=True)
        curve = delta.groupby("observation")["delta"].mean().dropna()
        curve = curve[curve.index >= 2]
        if len(curve) < 3:
            continue
        d = smooth_curve(curve.values, POWER_LAW_SMOOTH_WINDOW)
        if np.any(d <= 0):
            continue
        n = curve.index.values.astype(float)
        slope, intercept, _, _, _ = linregress(np.log(n), np.log(d))
        rows.append({"pid": pid, "A": float(np.exp(intercept)), "lambda_": float(-slope)})
    return pd.DataFrame(rows)


def label_panels(axes, labels=None, **kwargs) -> None:
    """Add bold panel labels in upper-left corner of each axes."""
    axs = np.ravel(axes).tolist()
    if labels is None:
        labels = [chr(ord("A") + i) for i in range(len(axs))]
    defaults = {
        "x": -0.1,
        "y": 1.1,
        "ha": "left",
        "va": "top",
        "fontweight": "bold",
        "fontsize": plt.rcParams.get("axes.titlesize", 16),
    }
    defaults.update(kwargs)
    for ax, lab in zip(axs, labels):
        ax.text(s=lab, transform=ax.transAxes, **defaults)


def get_palette(n: int = 10) -> list:
    """Return the seaborn colorblind palette as a list of colors.

    Colors are assigned by index — callers should plot items in a consistent
    order so that the same item always gets the same color.
    """
    return sns.color_palette("colorblind", n)


def pvalue_to_stars(p: float) -> str:
    if p <= 1e-4:
        return "****"
    elif p <= 1e-3:
        return "***"
    elif p <= 1e-2:
        return "**"
    elif p <= 0.05:
        return "*"
    return "ns"


def draw_sig_line(ax, x1, x2, y, stars, linewidth=0.9, fontsize=7):
    """Draw a flat horizontal significance line (no end ticks) with stars above."""
    ax.plot([x1, x2], [y, y], color="black", linewidth=linewidth, clip_on=False)
    ax.text(
        (x1 + x2) / 2,
        y,
        stars,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="black",
    )


def annotate_nef_comparisons(
    ax,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    order: list,
    nef_label: str = "NEF",
    dy_fraction: float = 0.04,
    compare_only: list | None = None,
) -> None:
    """
    Run paired Wilcoxon tests between NEF and other models, and draw
    horizontal significance lines (no end ticks) above the boxplot.

    Only comparisons involving `nef_label` are shown.
    Non-significant pairs (p > 0.05) are omitted.

    Parameters
    ----------
    ax           : matplotlib Axes
    data         : DataFrame with columns [pid, x_col, y_col]
    x_col        : column with model/group labels (x-axis categories)
    y_col        : column with the metric values
    order        : list of category names in x-axis order (sets x positions)
    nef_label    : display name of the NEF model in x_col
    dy_fraction  : line-spacing as fraction of current y-axis range
    compare_only : if given, only compare NEF against these models
    """
    x_positions = {model: i for i, model in enumerate(order)}
    if nef_label not in x_positions:
        return

    y_lo, y_hi = ax.get_ylim()
    dy_step = (y_hi - y_lo) * dy_fraction

    # collect NEF vs other pairs, restricted to compare_only if given
    candidates = [m for m in order if m != nef_label]
    if compare_only is not None:
        candidates = [m for m in candidates if m in compare_only]
    pairs = sorted(
        [(m, nef_label) for m in candidates],
        key=lambda p: abs(x_positions[p[0]] - x_positions[p[1]]),
    )

    sig_lines = []
    for m_other, m_nef in pairs:
        p1 = data.loc[data[x_col] == m_other, ["pid", y_col]]
        p2 = data.loc[data[x_col] == m_nef,   ["pid", y_col]]
        merged = p1.merge(p2, on="pid", suffixes=("_1", "_2"))
        if len(merged) < 4:
            continue
        d1 = merged[f"{y_col}_1"].to_numpy(dtype=float)
        d2 = merged[f"{y_col}_2"].to_numpy(dtype=float)
        diff = d1 - d2
        if np.all(diff == 0) or np.nanstd(diff) == 0:
            continue
        try:
            res = wilcoxon(d1, d2)
        except ValueError:
            continue
        p = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])
        stars = pvalue_to_stars(p)
        if stars == "ns":
            continue  # omit non-significant
        sig_lines.append((x_positions[m_other], x_positions[m_nef], stars))

    # stack lines from lowest span to highest
    sig_lines.sort(key=lambda t: abs(t[1] - t[0]))
    y_current = y_hi + dy_step * 0.5
    for x1, x2, stars in sig_lines:
        draw_sig_line(ax, x1, x2, y_current, stars)
        y_current += dy_step * 2.0

    if sig_lines:
        ax.set_ylim(top=y_current + dy_step)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, prune="upper"))
