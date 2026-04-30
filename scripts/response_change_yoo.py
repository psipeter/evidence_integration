#!/usr/bin/env python3
"""
Yoo task: power-law decay of response change (same fitting as ``fitting.losses``).

Row 1: Human and models — per-participant power-law curves (same fit as
``fitting.losses`` for yoo). Row 2: human decay vs amplitude KDE with sample
markers; violin plot of shape loss vs human.

On first run (SAMPLE_PIDS = None), prints pid / lambda_ / A for humans (sorted
by lambda_ descending), then exits. Set SAMPLE_PIDS and rerun.

Usage:
    python scripts/response_change_yoo.py [run_folder]

Data: data/yoo.pkl and data/runs/{run_folder}/{Mean,RL,ADM}_yoo_responses.pkl;
NEF: data/runs/{run_folder}/{NEF_*}_yoo_responses.pkl
No CSV/pickle outputs (figures only).
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

# -- path setup ----------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fitting.losses import POWER_LAW_SMOOTH_WINDOW  # noqa: F401
from fitting.losses import _fit_power_law_params
from utils.paths import data_path, FIGURES_DIR
from utils.plot_style import (
    annotate_violins,
    apply_style,
    FIGURE_SIZE,
    get_palette,
    SAMPLE_MARKERS,
)

# -- CLI ----------------------------------------------------------------------
_parser = argparse.ArgumentParser()
_parser.add_argument("--run_folder", type=str, default="joint")
_parser.add_argument("--include_rl_lambda", action="store_true", default=False)
_args, _ = _parser.parse_known_args()
RUN_FOLDER = _args.run_folder

# -- configuration (edit here) -------------------------------------------------
# None: print human pid / tau / y_int table (sorted by tau) and exit.
# Else: e.g. {"fast": 1, "medium": 2, "slow": 3}
SAMPLE_PIDS: dict[str, int] | None = {"fast": 1, "medium": 25, "slow": 17}

LINESTYLES = ["solid", "dashed", "dotted"]  # fast / medium / slow
SAMPLE_LABELS = ["fast", "medium", "slow"]

OBS_MIN = 2
OBS_MAX = 30

# -- CLI ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Yoo response change / decay figure")
parser.add_argument(
    "--run_folder",
    type=str,
    default=RUN_FOLDER,
    help="Run folder under data/runs/ for math model pickles",
)
parser.add_argument("--include_rl_lambda", action="store_true", default=False)
args = parser.parse_args()

RUN_FOLDER = args.run_folder

MODEL_ORDER = ["Mean", "RL", "ADM", "NEF_recurrent"]
if args.include_rl_lambda:
    MODEL_ORDER.append("RL_lambda")

# -- style ---------------------------------------------------------------------
apply_style()
PALETTE = get_palette()


def _display(mt: str) -> str:
    if mt.startswith("NEF"):
        return "NEF"
    return mt


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


def _row1_title(label: str) -> str:
    if label == "Human":
        return "Human"
    return _display(label)


def _row1_color(label: str) -> str:
    if label == "Human":
        return PALETTE["Human"]
    return PALETTE.get(label, PALETTE.get(_display(label), "gray"))


# -- load human ----------------------------------------------------------------
human = pd.read_pickle(data_path("yoo.pkl"))

if SAMPLE_PIDS is None:
    tbl = _fit_power_law_params(human).sort_values("lambda_", ascending=False)
    print("pid / lambda_ / A (human, sorted by lambda_):")
    print(tbl.to_string(index=False))
    print("\nSet SAMPLE_PIDS at the top of this script and rerun.")
    sys.exit(0)

# -- load models ---------------------------------------------------------------
run_dir = data_path("runs") / RUN_FOLDER
models: dict[str, pd.DataFrame] = {}
loaded_models: list[str] = []
for mt in MODEL_ORDER:
    f = run_dir / f"{mt}_yoo_responses.pkl"
    if not f.exists():
        print(f"Warning: missing {f.name}, skipping {mt}")
        continue
    models[mt] = pd.read_pickle(f)
    loaded_models.append(mt)

MODEL_ORDER = loaded_models
DISPLAY_ORDER = [_display(mt) for mt in MODEL_ORDER]

# human power law params (using same function as losses.py)
human_pl = _fit_power_law_params(human).set_index("pid")

# model power law params
model_pl: dict[str, pd.DataFrame] = {}
for mt in MODEL_ORDER:
    model_pl[mt] = _fit_power_law_params(models[mt]).set_index("pid")

sample_pids = [int(SAMPLE_PIDS[k]) for k in SAMPLE_LABELS]
for lab, pid in zip(SAMPLE_LABELS, sample_pids):
    if pid not in human_pl.index:
        raise ValueError(f"SAMPLE_PIDS[{lab!r}]={pid} not in human power-law fits")
    lam = float(human_pl.loc[pid, "lambda_"])
    a0 = float(human_pl.loc[pid, "A"])
    if not (np.isfinite(lam) and np.isfinite(a0)):
        raise ValueError(f"SAMPLE_PIDS[{lab!r}]={pid} missing or invalid power-law fit")

loss_df = _load_loss_long(run_dir, MODEL_ORDER, "yoo")
loss_df["model_type"] = loss_df["model_type"].apply(_display)
if MODEL_ORDER and not loss_df.empty:
    _models_with_loss = loss_df["model_type"].unique().tolist()
    loss_plot = loss_df.groupby("pid").filter(
        lambda g: len(g) == len(_models_with_loss)
    ).copy()
else:
    loss_plot = pd.DataFrame(columns=["pid", "model_type", "loss"])

sources: list[tuple[str, pd.DataFrame]] = [("Human", human)] + [
    (mt, models[mt]) for mt in MODEL_ORDER
]
n_model_cols = len(sources)

# -- figure --------------------------------------------------------------------
fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=True)
gs = gridspec.GridSpec(2, n_model_cols, figure=fig, height_ratios=[1.0, 1.2])

ax_row1: list = []
for i in range(n_model_cols):
    sharey = ax_row1[0] if i > 0 else None
    ax_row1.append(fig.add_subplot(gs[0, i], sharey=sharey))

ax_param = fig.add_subplot(gs[1, :2])
ax_viol = fig.add_subplot(gs[1, 2:])

_markers = SAMPLE_MARKERS
n_grid = np.arange(OBS_MIN, OBS_MAX + 1, dtype=float)

# Row 1: power-law fit curves only (same fit as losses.shape_loss for yoo)
for ax, (label, _) in zip(ax_row1, sources):
    color = _row1_color(label)
    pl_df = human_pl if label == "Human" else model_pl[label]
    for pid, ls in zip(sample_pids, LINESTYLES):
        if pid not in pl_df.index:
            continue
        tau = float(pl_df.loc[pid, "lambda_"])
        y_int = float(pl_df.loc[pid, "A"])
        if np.isfinite(tau) and np.isfinite(y_int) and y_int > 0:
            y_fit = y_int * n_grid ** (-tau)
            ax.plot(n_grid, y_fit, color=color, linestyle=ls, linewidth=1.5, zorder=4)
    if label == "Human":
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [0],
                [0],
                color=PALETTE["Human"],
                linestyle=ls,
                linewidth=1.5,
                label=f"#{pid}",
            )
            for pid, ls in zip(sample_pids, LINESTYLES)
        ]
        ax.legend(handles=handles, title="Participant", frameon=False, loc="upper right")

    ax.set_xlim(0, float(OBS_MAX))
    ax.set_title(_row1_title(label))
    ax.set_xlabel("Observation")
    ax.set_ylabel("Power-law fit (|Δ|)" if label == "Human" else "")
    if label != "Human":
        plt.setp(ax.get_yticklabels(), visible=False)
    sns.despine(ax=ax, top=True, right=True)
for ax in ax_row1[len(sources):]:
    ax.axis("off")

ymax = max(ax.get_ylim()[1] for ax in ax_row1)
for ax in ax_row1:
    ax.set_ylim(0.0, ymax)

# Row 2 left: lambda_ vs A (human)
h_tbl = human_pl.reset_index()[["pid", "lambda_", "A"]]
h_tbl = h_tbl[np.isfinite(h_tbl["lambda_"]) & np.isfinite(h_tbl["A"])]
sns.kdeplot(
    data=h_tbl,
    x="lambda_",
    y="A",
    fill=True,
    alpha=0.6,
    color=PALETTE["Human"],
    ax=ax_param,
)
for pid, ls, mkr in zip(sample_pids, LINESTYLES, _markers):
    row = h_tbl[h_tbl["pid"] == pid]
    if row.empty:
        continue
    x = float(row["lambda_"].iloc[0])
    y = float(row["A"].iloc[0])
    # draw marker
    ax_param.plot(
        x,
        y,
        marker=mkr,
        color="none",
        markeredgecolor=PALETTE["Human"],
        markeredgewidth=1.5,
        markersize=10,
        zorder=5,
    )

from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker=mkr, color="none",
           markeredgecolor=PALETTE["Human"], markeredgewidth=1.5,
           markersize=8, label=f"#{pid}")
    for pid, mkr in zip(sample_pids, _markers)
]
ax_param.legend(handles=handles, title="Participant", frameon=False)
ax_param.set_xlabel("λ (decay exponent)")
ax_param.set_ylabel("A (amplitude)")
ax_param.set_title("Human power-law parameters")
sns.despine(ax=ax_param, top=True, right=True)

# Row 2 right: shape loss violins
plot_palette = {
    _display(mt): PALETTE.get(mt, PALETTE.get(_display(mt), "gray"))
    for mt in MODEL_ORDER
}
if DISPLAY_ORDER and not loss_plot.empty:
    sns.violinplot(
        data=loss_plot,
        x="model_type",
        y="loss",
        order=DISPLAY_ORDER,
        hue="model_type",
        palette=plot_palette,
        inner=None,
        legend=False,
        cut=0,
        ax=ax_viol,
    )
else:
    ax_viol.text(0.5, 0.5, "no model data", ha="center", va="center", transform=ax_viol.transAxes)
ax_viol.set_title("Power-law parameter distance")
ax_viol.set_ylabel("Shape loss (|ΔA| + |Δλ|)")
ax_viol.set_xlabel("")
sns.despine(ax=ax_viol, top=True, right=True)

if len(DISPLAY_ORDER) >= 2 and not loss_plot.empty:
    annotate_violins(ax_viol, loss_plot, "model_type", "loss", DISPLAY_ORDER)

# -- save ----------------------------------------------------------------------
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "response_change_yoo.png", dpi=300)
plt.savefig(FIGURES_DIR / "response_change_yoo.pdf")
print("Saved figures/response_change_yoo.{png,pdf}")
