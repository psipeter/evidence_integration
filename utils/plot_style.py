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


def draw_bracket(ax, x1, x2, y_bot, dy_step, text, linewidth=0.8, fontsize=7):
    """Draw a significance bracket between x1 and x2 at height y_bot."""
    y = y_bot + dy_step
    ax.plot(
        [x1, x1, x2, x2],
        [y_bot, y, y, y_bot],
        color="black",
        linewidth=linewidth,
        clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="black",
    )


def annotate_violins(
    ax,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    order: list,
    dy_fraction: float = 0.03,
) -> None:
    """
    Run paired Wilcoxon tests for all combinations of `order` and draw
    significance brackets above the distribution plot on `ax`.

    Parameters
    ----------
    ax         : matplotlib Axes
    data       : DataFrame with columns [pid, x_col, y_col]
    x_col      : column name for model/group (x-axis categories)
    y_col      : column name for the metric (y-axis values)
    order      : list of category names in x-axis order
    dy_fraction: bracket spacing as fraction of current y-axis range
    """
    pairs = list(combinations(order, 2))
    x_positions = {model: i for i, model in enumerate(order)}
    y_top = ax.get_ylim()[1]
    dy_step = (ax.get_ylim()[1] - ax.get_ylim()[0]) * dy_fraction

    bracket_stars = []
    for m1, m2 in pairs:
        p1 = data.loc[data[x_col] == m1, ["pid", y_col]]
        p2 = data.loc[data[x_col] == m2, ["pid", y_col]]
        merged = p1.merge(p2, on="pid", suffixes=("_1", "_2"))
        if len(merged) < 2:
            continue
        d1 = merged[f"{y_col}_1"].to_numpy(dtype=float)
        d2 = merged[f"{y_col}_2"].to_numpy(dtype=float)
        diff = np.array(d1) - np.array(d2)
        if len(diff) == 0 or np.all(diff == 0) or np.nanstd(diff) == 0:
            continue
        try:
            res = wilcoxon(d1, d2)
        except ValueError:
            continue
        p = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])
        bracket_stars.append((m1, m2, pvalue_to_stars(p)))

    y_current = y_top
    for m1, m2, stars in bracket_stars:
        draw_bracket(
            ax, x_positions[m1], x_positions[m2], y_current, dy_step, stars
        )
        star_y = y_current + dy_step
        mid_x = (x_positions[m1] + x_positions[m2]) / 2
        y_current += dy_step * 2

    ax.set_ylim(top=y_current + dy_step)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, prune="upper"))
