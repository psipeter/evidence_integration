"""
Shared plotting style for all notebooks and figures.

Import and call apply_style() at the top of every notebook to ensure
consistent aesthetics across figures. Optimized for readability in
JupyterLab and crispness when loaded into LaTeX/Overleaf as PDF.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def apply_style() -> None:
    """Apply project-wide matplotlib and seaborn style defaults."""
    sns.set_theme(style="ticks")
    plt.rcParams.update(
        {
            # figure
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "figure.figsize": (7, 4.5),
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
