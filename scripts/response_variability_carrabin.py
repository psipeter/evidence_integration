#!/usr/bin/env python3
"""
Intertrial variability figure -- carrabin task.

On first run (SAMPLE_PIDS = None), prints a pid/std table so you can choose
representative participants, then exits. Set SAMPLE_PIDS at the top of this
file and rerun to generate the figure.

Usage:
    python scripts/response_variability_carrabin.py [run_folder]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

# -- path setup ----------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import data_path, FIGURES_DIR
from utils.plot_style import annotate_violins, apply_style, FIGURE_SIZE, get_palette

# -- CLI ----------------------------------------------------------------------
_parser = argparse.ArgumentParser()
_parser.add_argument("--run_folder", type=str, default="MSE")
_args, _ = _parser.parse_known_args()
RUN_FOLDER = _args.run_folder

# -- configuration (edit here) -------------------------------------------------
# Set to None on first run to print pid/std table, then fill in and rerun.
SAMPLE_PIDS = {"narrow": 20, "medium": 18, "broad": 4}

LINESTYLES = ["solid", "dashed", "dotted"]  # narrow / medium / broad

# -- CLI ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Carrabin response variability figure")
parser.add_argument(
    "--run_folder",
    type=str,
    default=RUN_FOLDER,
    help="Run folder under data/runs/ for math model pickles",
)
args = parser.parse_args()

RUN_FOLDER = args.run_folder

MODEL_ORDER = ["Bayes", "RL", "NoisyCounting", "NEF_recurrent"]

# -- style ---------------------------------------------------------------------
apply_style()
PALETTE = get_palette()


def _display(mt: str) -> str:
    if mt.startswith("NEF"):
        return "NEF"
    return mt


def _kde_panel_title(label: str) -> str:
    if label == "Human":
        return "Human"
    return _display(label)


def _kde_color(label: str) -> str:
    if label == "Human":
        return PALETTE["Human"]
    return PALETTE[_display(label)]

# -- load data -----------------------------------------------------------------
run_dir = data_path("runs") / RUN_FOLDER

human = pd.read_pickle(data_path("carrabin.pkl"))

models: dict[str, pd.DataFrame] = {}
perf_dfs = []
loaded_models: list[str] = []
for mt in MODEL_ORDER:
    resp_f = run_dir / f"{mt}_carrabin_responses.pkl"
    perf_f = run_dir / f"{mt}_carrabin_performance.pkl"
    if not resp_f.exists():
        print(f"Warning: missing {resp_f.name}, skipping {mt}")
        continue
    if not perf_f.exists():
        print(f"Warning: missing performance file for {mt}, skipping")
        continue
    models[mt] = pd.read_pickle(resp_f)
    perf_dfs.append(pd.read_pickle(perf_f))
    loaded_models.append(mt)

MODEL_ORDER = loaded_models
DISPLAY_ORDER = [_display(mt) for mt in MODEL_ORDER]

if perf_dfs:
    perf = pd.concat(perf_dfs, ignore_index=True)
else:
    perf = pd.DataFrame(columns=["model_type", "cv_loss_mean"])
perf_display = perf.copy()
perf_display["model_type"] = perf_display["model_type"].apply(_display)

# -- per-participant std -------------------------------------------------------
pid_std = (
    human.groupby("pid")["response"]
    .std()
    .rename("response_std")
    .reset_index()
    .sort_values("response_std")
    .reset_index(drop=True)
)

if SAMPLE_PIDS is None:
    print("pid / response_std table (sorted narrow -> broad):")
    print(pid_std.to_string(index=False))
    print("\nSet SAMPLE_PIDS at the top of this script and rerun.")
    sys.exit(0)

sample_labels = ["narrow", "medium", "broad"]
sample_pids = [SAMPLE_PIDS[l] for l in sample_labels]

# -- figure layout -------------------------------------------------------------
fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=True)
gs = gridspec.GridSpec(
    2,
    5,
    figure=fig,
    height_ratios=[1, 1.2],
)

# Row 1: KDE panels (Human + available models)
ax_kde = []
for i in range(5):
    sharey = ax_kde[0] if i > 0 else None
    ax_kde.append(fig.add_subplot(gs[0, i], sharey=sharey))

# Row 2: std (2 cols) + violin (3 cols)
ax_std = fig.add_subplot(gs[1, :2])
ax_viol = fig.add_subplot(gs[1, 2:])

# -- row 1: KDE panels ---------------------------------------------------------
sources = [("Human", human)] + [(mt, models[mt]) for mt in MODEL_ORDER]

# Compute shared x/y limits across all panels and all pids
all_responses = pd.concat(
    [src[["response"]] for _, src in sources], ignore_index=True
)
x_min, x_max = all_responses["response"].min(), all_responses["response"].max()
x_pad = (x_max - x_min) * 0.05
x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 400)

max_density = 0.0
kde_cache: dict[tuple[str, int], np.ndarray] = {}

for label, src in sources:
    for pid in sample_pids:
        vals = src[src["pid"] == pid]["response"].values
        kde = gaussian_kde(vals)
        density = kde(x_grid)
        kde_cache[(label, pid)] = density
        max_density = max(max_density, density.max())

y_max = max_density * 1.1

for ax, (label, _) in zip(ax_kde, sources):
    color = _kde_color(label)
    for pid, ls in zip(sample_pids, LINESTYLES):
        density = kde_cache[(label, pid)]
        ax.fill_between(x_grid, density, alpha=0.3, color=color)
        ax.plot(x_grid, density, color=color, linestyle=ls, linewidth=1.5)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(0, y_max)
    ax.set_title(_kde_panel_title(label))
    ax.set_xlabel("Response")
    ax.set_ylabel("Density" if label == "Human" else "")
    if label != "Human":
        plt.setp(ax.get_yticklabels(), visible=False)
    sns.despine(ax=ax, top=True, right=True)
for ax in ax_kde[len(sources):]:
    ax.axis("off")

# -- row 2a: population std distribution ---------------------------------------
std_vals = pid_std["response_std"].values
kde_pop = gaussian_kde(std_vals)
x_std = np.linspace(std_vals.min() * 0.9, std_vals.max() * 1.1, 400)
ax_std.fill_between(x_std, kde_pop(x_std), alpha=0.3, color="0.5")
ax_std.plot(x_std, kde_pop(x_std), color="0.3", linewidth=1.5)

# Mark sample participants
for pid, ls in zip(sample_pids, LINESTYLES):
    std_val = pid_std.loc[pid_std["pid"] == pid, "response_std"].values[0]
    kde_height = float(kde_pop(np.array([std_val]))[0])
    ax_std.plot(
        [std_val, std_val],
        [0, kde_height],
        color="0.3",
        linestyle=ls,
        linewidth=1.5,
    )

ax_std.set_xlabel("Response std")
ax_std.set_ylabel("Density")
ax_std.set_title("Population variability")
sns.despine(ax=ax_std, top=True, right=True)

# -- row 2b: Wasserstein violin plots ------------------------------------------
if DISPLAY_ORDER and not perf_display.empty:
    sns.violinplot(
        data=perf_display,
        x="model_type",
        y="cv_loss_mean",
        order=DISPLAY_ORDER,
        hue="model_type",
        palette=PALETTE,
        inner=None,
        legend=False,
        cut=0,
        ax=ax_viol,
    )
else:
    ax_viol.text(0.5, 0.5, "no model data", ha="center", va="center", transform=ax_viol.transAxes)
ax_viol.set_title("Distance to human response distribution")
ax_viol.set_ylabel("Wasserstein distance")
ax_viol.set_xlabel("")
sns.despine(ax=ax_viol, top=True, right=True)

if len(DISPLAY_ORDER) >= 2 and not perf_display.empty:
    annotate_violins(
        ax_viol, perf_display, "model_type", "cv_loss_mean", DISPLAY_ORDER
    )

# -- save ----------------------------------------------------------------------
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "response_variability_carrabin.png", dpi=300)
plt.savefig(FIGURES_DIR / "response_variability_carrabin.pdf")
print("Saved figures/response_variability_carrabin.{png,pdf}")
