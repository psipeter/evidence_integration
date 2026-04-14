"""
Shared plotting style for all notebooks and figures.

Import and call apply_style() at the top of every notebook to ensure
consistent aesthetics across figures. Optimized for readability in
JupyterLab and crispness when loaded into LaTeX/Overleaf as PDF.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D  # noqa: F401 (re-exported for convenience)


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
    }


# Marker shapes for sample participants (narrow/low, medium, broad/high)
SAMPLE_MARKERS = ["o", "s", "^"]
