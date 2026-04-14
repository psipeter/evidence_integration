#!/usr/bin/env python3
"""
Yoo task: mean absolute response change across observations and power-law decay.

Row 1: Human and three models — mean |Δresponse| ± SE vs observation with
per-participant power-law fits for three sample participants (fast / medium /
slow decay). Row 2: human (tau, y_int) KDE with sample markers; violin plot of
Wasserstein distance between human and model mean curves per participant.

On first run (SAMPLE_PIDS = None), prints pid / tau / y_int for humans (sorted
by tau), then exits. Set SAMPLE_PIDS and rerun.

Usage:
    python scripts/response_change_yoo.py

Data: data/yoo.pkl and data/runs/MSE/{Mean,RL,ADM}_yoo_responses.pkl
No CSV/pickle outputs (figures only).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance
# -- path setup ----------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import data_path, FIGURES_DIR
from utils.plot_style import (
    annotate_violins,
    apply_style,
    FIGURE_SIZE,
    get_palette,
    SAMPLE_MARKERS,
)

# -- configuration (edit here) -------------------------------------------------
RUN_FOLDER = "MSE"

# None: print human pid / tau / y_int table (sorted by tau) and exit.
# Else: e.g. {"fast": 1, "medium": 2, "slow": 3}
# SAMPLE_PIDS: dict[str, int] | None = None
SAMPLE_PIDS: dict[str, int] | None = {"fast": 1, "medium": 25, "slow": 17}

MODEL_ORDER = ["Mean", "RL", "ADM"]
LINESTYLES = ["solid", "dashed", "dotted"]  # fast / medium / slow
SAMPLE_LABELS = ["fast", "medium", "slow"]

OBS_MIN = 2
OBS_MAX = 30

# -- style ---------------------------------------------------------------------
apply_style()
PALETTE = get_palette()


def abs_delta_long(df: pd.DataFrame) -> pd.DataFrame:
    """|response[n] - response[n-1]| per (pid, trial, observation); obs 2–30."""
    pieces: list[pd.DataFrame] = []
    for (pid, trial), grp in df.groupby(["pid", "trial"], sort=False):
        g = grp.sort_values("observation").copy()
        g["delta"] = g["response"].astype(float).diff().abs()
        g = g[(g["observation"] >= OBS_MIN) & (g["observation"] <= OBS_MAX)]
        g = g[["pid", "trial", "observation", "delta"]].dropna(subset=["delta"])
        pieces.append(g)
    if not pieces:
        return pd.DataFrame(columns=["pid", "trial", "observation", "delta"])
    return pd.concat(pieces, ignore_index=True)


def mean_curve_by_pid(delta_df: pd.DataFrame) -> dict[int, pd.Series]:
    """Mean |Δresponse| at each observation, averaged across trials; index = obs."""
    g = delta_df.groupby(["pid", "observation"])["delta"].mean()
    out: dict[int, pd.Series] = {}
    for pid in g.index.get_level_values(0).unique():
        s = g.loc[int(pid)].sort_index()
        out[int(pid)] = s
    return out


def fit_power_law(mean_ser: pd.Series) -> tuple[float, float]:
    """
    Fit mean_delta(n) = y_int * n^(-tau) via log-log linear regression.
    Returns (tau, y_int); (nan, nan) if incomplete or any mean_delta <= 0.
    """
    mean_ser = mean_ser.sort_index()
    idx = mean_ser.index.astype(float).to_numpy()
    vals = mean_ser.to_numpy(dtype=float)
    mask = (idx >= OBS_MIN) & (idx <= OBS_MAX)
    idx = idx[mask]
    vals = vals[mask]
    if len(idx) != (OBS_MAX - OBS_MIN + 1) or np.any(~np.isfinite(vals)) or np.any(vals <= 0):
        return float("nan"), float("nan")
    log_n = np.log(idx)
    log_y = np.log(vals)
    slope, intercept = np.polyfit(log_n, log_y, 1)
    tau = -float(slope)
    y_int = float(np.exp(intercept))
    return tau, y_int


def curve_vector(mean_ser: pd.Series) -> np.ndarray | None:
    """29-point array for observations OBS_MIN..OBS_MAX; None if incomplete."""
    obs_range = list(range(OBS_MIN, OBS_MAX + 1))
    reindexed = mean_ser.reindex(obs_range)
    if reindexed.isna().any() or not np.all(np.isfinite(reindexed.to_numpy(dtype=float))):
        return None
    return reindexed.to_numpy(dtype=float)


def build_wasserstein_loss_long(
    human_means: dict[int, pd.Series],
    model_means: dict[str, dict[int, pd.Series]],
) -> pd.DataFrame:
    rows: list[dict] = []
    pids = sorted(human_means.keys())
    for pid in pids:
        h_vec = curve_vector(human_means[pid])
        if h_vec is None:
            continue
        for mt in MODEL_ORDER:
            mser = model_means[mt].get(pid)
            if mser is None:
                continue
            m_vec = curve_vector(mser)
            if m_vec is None:
                continue
            loss = float(wasserstein_distance(h_vec, m_vec))
            rows.append({"pid": pid, "model_type": mt, "loss": loss})
    return pd.DataFrame(rows)


# -- load human ----------------------------------------------------------------
human = pd.read_pickle(data_path("yoo.pkl"))
delta_human = abs_delta_long(human)
human_means = mean_curve_by_pid(delta_human)

params_human: dict[int, tuple[float, float]] = {}
for pid, ser in human_means.items():
    params_human[pid] = fit_power_law(ser)

if SAMPLE_PIDS is None:
    rows = [
        {"pid": pid, "tau": params_human[pid][0], "y_int": params_human[pid][1]}
        for pid in sorted(params_human.keys())
    ]
    tbl = pd.DataFrame(rows).sort_values("tau", na_position="last")
    print("pid / tau / y_int (human, sorted by tau):")
    print(tbl.to_string(index=False))
    print("\nSet SAMPLE_PIDS at the top of this script and rerun.")
    sys.exit(0)

# -- load models ---------------------------------------------------------------
run_dir = data_path("runs") / RUN_FOLDER
models: dict[str, pd.DataFrame] = {}
for mt in MODEL_ORDER:
    f = run_dir / f"{mt}_yoo_responses.pkl"
    if not f.exists():
        raise FileNotFoundError(f"Missing model responses: {f}")
    models[mt] = pd.read_pickle(f)

delta_by_source: dict[str, pd.DataFrame] = {"Human": delta_human}
model_means: dict[str, dict[int, pd.Series]] = {}
for mt in MODEL_ORDER:
    dmt = abs_delta_long(models[mt])
    delta_by_source[mt] = dmt
    model_means[mt] = mean_curve_by_pid(dmt)

params_by_source: dict[str, dict[int, tuple[float, float]]] = {"Human": params_human}
for mt in MODEL_ORDER:
    params_by_source[mt] = {
        pid: fit_power_law(ser) for pid, ser in model_means[mt].items()
    }

sample_pids = [int(SAMPLE_PIDS[k]) for k in SAMPLE_LABELS]
for lab, pid in zip(SAMPLE_LABELS, sample_pids):
    tau, y0 = params_human.get(pid, (float("nan"), float("nan")))
    if pid not in params_human or not (np.isfinite(tau) and np.isfinite(y0)):
        raise ValueError(f"SAMPLE_PIDS[{lab!r}]={pid} missing or invalid power-law fit")

loss_df = build_wasserstein_loss_long(human_means, model_means)
_complete = loss_df.groupby("pid").filter(lambda g: len(g) == len(MODEL_ORDER))
if _complete.empty:
    raise RuntimeError("No participants with valid Wasserstein loss for all models.")
loss_plot = _complete.copy()

sources: list[tuple[str, pd.DataFrame]] = [("Human", human)] + [
    (mt, models[mt]) for mt in MODEL_ORDER
]

# -- figure --------------------------------------------------------------------
fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=True)
gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.2])

ax_row1: list = []
for i in range(4):
    sharey = ax_row1[0] if i > 0 else None
    ax_row1.append(fig.add_subplot(gs[0, i], sharey=sharey))

ax_param = fig.add_subplot(gs[1, :2])
ax_viol = fig.add_subplot(gs[1, 2:])

_markers = SAMPLE_MARKERS
n_grid = np.arange(OBS_MIN, OBS_MAX + 1, dtype=float)

# Row 1: mean |Δresponse| ± SE + power-law overlay
for ax, (label, _) in zip(ax_row1, sources):
    color = PALETTE[label]
    delta_src = delta_by_source[label]
    param_src = params_by_source[label]
    for pid, ls, mkr in zip(sample_pids, LINESTYLES, _markers):
        obs_pid = delta_src[delta_src["pid"] == pid]
        if obs_pid.empty:
            continue
        sns.lineplot(
            data=obs_pid,
            x="observation",
            y="delta",
            errorbar="se",
            err_style="band",
            estimator="mean",
            color=color,
            linestyle="none",
            linewidth=0,
            ax=ax,
            legend=False,
        )
        tau, y_int = param_src.get(pid, (float("nan"), float("nan")))
        if np.isfinite(tau) and np.isfinite(y_int) and y_int > 0:
            y_fit = y_int * n_grid ** (-tau)
            ax.plot(n_grid, y_fit, color=color, linestyle=ls, linewidth=1.5, zorder=4)
            ax.scatter(2.0, y_int * 2.0 ** (-tau), color=color, marker=mkr,
                       s=60, zorder=5)

    ax.set_xlim(0, float(OBS_MAX))
    ax.set_title(label)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Mean |Δresponse|" if label == "Human" else "")
    if label != "Human":
        plt.setp(ax.get_yticklabels(), visible=False)
    sns.despine(ax=ax, top=True, right=True)

ymax = max(ax.get_ylim()[1] for ax in ax_row1)
for ax in ax_row1:
    ax.set_ylim(0.0, ymax)

# Row 2 left: tau vs y_int (human)
h_tbl = pd.DataFrame(
    [{"pid": p, "tau": params_human[p][0], "y_int": params_human[p][1]} for p in params_human]
)
h_tbl = h_tbl[np.isfinite(h_tbl["tau"]) & np.isfinite(h_tbl["y_int"])]
sns.kdeplot(
    data=h_tbl,
    x="tau",
    y="y_int",
    fill=True,
    alpha=0.6,
    color=PALETTE["Human"],
    ax=ax_param,
)
y_text_pad = 0.02 * float(h_tbl["y_int"].max()) if len(h_tbl) else 0.0
for pid, mkr, lbl in zip(sample_pids, _markers, SAMPLE_LABELS):
    row = h_tbl[h_tbl["pid"] == pid]
    if row.empty:
        continue
    ax_param.scatter(
        float(row["tau"].iloc[0]),
        float(row["y_int"].iloc[0]),
        s=80,
        facecolors="none",
        edgecolors=PALETTE["Human"],
        linewidths=1.5,
        marker=mkr,
        zorder=5,
    )
    ax_param.text(
        float(row["tau"].iloc[0]),
        float(row["y_int"].iloc[0]) + y_text_pad,
        lbl,
        ha="center",
        va="bottom",
        fontsize=7,
        color=PALETTE["Human"],
    )

ax_param.set_xlabel("τ (decay exponent)")
ax_param.set_ylabel("y_int (scale)")
ax_param.set_title("Human power-law parameters")
sns.despine(ax=ax_param, top=True, right=True)

# Row 2 right: Wasserstein violins
plot_palette = {k: PALETTE[k] for k in MODEL_ORDER}
sns.violinplot(
    data=loss_plot,
    x="model_type",
    y="loss",
    order=MODEL_ORDER,
    hue="model_type",
    palette=plot_palette,
    inner=None,
    legend=False,
    cut=0,
    ax=ax_viol,
)
# np.random.seed(42)
# sns.stripplot(
#     data=loss_plot,
#     x="model_type",
#     y="loss",
#     order=MODEL_ORDER,
#     color="0.2",
#     alpha=0.5,
#     jitter=0.2,
#     size=4,
#     ax=ax_viol,
# )
ax_viol.set_title("Distance to human response change curve")
ax_viol.set_ylabel("Wasserstein distance")
ax_viol.set_xlabel("")
sns.despine(ax=ax_viol, top=True, right=True)

annotate_violins(ax_viol, loss_plot, "model_type", "loss", MODEL_ORDER)

# -- save ----------------------------------------------------------------------
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "response_change_yoo.png", dpi=300)
plt.savefig(FIGURES_DIR / "response_change_yoo.pdf")
print("Saved figures/response_change_yoo.{png,pdf}")
