"""
Shared plotting style for all notebooks and figures.

Import and call apply_style() at the top of every notebook to ensure
consistent aesthetics across figures. Optimized for readability in
JupyterLab and crispness when loaded into LaTeX/Overleaf as PDF.
"""

from itertools import combinations

import matplotlib.pyplot as plt
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
            "lines.linewidth": 1.5,
            # saving
            "savefig.bbox": "tight",
            "savefig.transparent": False,
        }
    )


# -- shared figure constants ---------------------------------------------------

FIGURE_SIZE = (14, 7)


def get_palette() -> dict:
    """
    Return the project-wide model color palette.
    Call after apply_style() to ensure seaborn palette is initialized.
    Colors assigned by model role:
        optimal  (Bayes, Mean):                       palette[0]
        naive    (RL):                                palette[1]
        human-matching (NoisyCounting, DeGroot, ADM): palette[2]
        Human data:                                   "0.3" (neutral grey)
    """
    palette = sns.color_palette("colorblind")
    return {
        "Human":         "0.3",
        "Bayes":         palette[0],
        "Mean":          palette[0],
        "RL":            palette[1],
        "NoisyCounting": palette[2],
        "DeGroot":       palette[2],
        "ADM":           palette[2],
        "NEF":           palette[3],
        "NEF_recurrent": palette[3],
        "NEF_synaptic":  palette[3],
    }


# Marker shapes for sample participants (narrow/low, medium, broad/high)
SAMPLE_MARKERS = ["o", "s", "^"]


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
    significance brackets above the violin plot on `ax`.

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
        y_current += dy_step * 2

    ax.set_ylim(top=y_current + dy_step)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, prune="upper"))
