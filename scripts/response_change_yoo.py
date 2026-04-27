#!/usr/bin/env python3
"""
Yoo task: mean absolute response change across observations and power-law decay.

Row 1: Human and four models — mean |Δresponse| ± SE vs observation with
per-participant power-law fits for three sample participants (fast / medium /
slow decay). Row 2: human (tau, y_int) KDE with sample markers; violin plot of
Wasserstein distance between human and model mean curves per participant.

On first run (SAMPLE_PIDS = None), prints pid / tau / y_int for humans (sorted
by tau), then exits. Set SAMPLE_PIDS and rerun.

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
_parser.add_argument("--run_folder", type=str, default="joint_loss")
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

ROLLING_WINDOW = 3

# -- CLI ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Yoo response change / decay figure")
parser.add_argument(
    "--run_folder",
    type=str,
    default=RUN_FOLDER,
    help="Run folder under data/runs/ for math model pickles",
)
args = parser.parse_args()

RUN_FOLDER = args.run_folder

MODEL_ORDER = ["Mean", "RL", "ADM", "NEF_recurrent"]

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
    return PALETTE[_display(label)]


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


def smooth_mean_curve(
    means: dict[int, pd.Series], window: int
) -> dict[int, pd.Series]:
    """Apply centered rolling average of given window size to each pid's curve."""
    return {
        pid: ser.rolling(window, center=True, min_periods=1).mean()
        for pid, ser in means.items()
    }


def smooth_delta_df(delta_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Apply rolling average per (pid, trial) to smooth delta values."""
    if delta_df.empty:
        return delta_df
    pieces = []
    for (pid, trial), grp in delta_df.groupby(["pid", "trial"], sort=False):
        g = grp.sort_values("observation").copy()
        g["delta"] = g["delta"].rolling(window, center=True, min_periods=1).mean()
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


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


# -- load human ----------------------------------------------------------------
human = pd.read_pickle(data_path("yoo.pkl"))
delta_human = abs_delta_long(human)

if SAMPLE_PIDS is None:
    human_means_tbl = mean_curve_by_pid(delta_human)
    if ROLLING_WINDOW > 1:
        human_means_tbl = smooth_mean_curve(human_means_tbl, ROLLING_WINDOW)
    params_human_tbl = {
        pid: fit_power_law(ser) for pid, ser in human_means_tbl.items()
    }
    rows = [
        {"pid": pid, "tau": params_human_tbl[pid][0], "y_int": params_human_tbl[pid][1]}
        for pid in sorted(params_human_tbl.keys())
    ]
    tbl = pd.DataFrame(rows).sort_values("tau", na_position="last")
    print("pid / tau / y_int (human, sorted by tau):")
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

delta_by_source: dict[str, pd.DataFrame] = {"Human": delta_human}
for mt in MODEL_ORDER:
    delta_by_source[mt] = abs_delta_long(models[mt])

if ROLLING_WINDOW > 1:
    for key in delta_by_source:
        delta_by_source[key] = smooth_delta_df(delta_by_source[key], ROLLING_WINDOW)

human_means = mean_curve_by_pid(delta_by_source["Human"])
model_means: dict[str, dict[int, pd.Series]] = {}
for mt in MODEL_ORDER:
    model_means[mt] = mean_curve_by_pid(delta_by_source[mt])

if ROLLING_WINDOW > 1:
    human_means = smooth_mean_curve(human_means, ROLLING_WINDOW)
    for mt in MODEL_ORDER:
        model_means[mt] = smooth_mean_curve(model_means[mt], ROLLING_WINDOW)

params_human: dict[int, tuple[float, float]] = {
    pid: fit_power_law(ser) for pid, ser in human_means.items()
}
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

loss_df = _load_loss_long(run_dir, MODEL_ORDER, "yoo")
loss_df["model_type"] = loss_df["model_type"].apply(_display)
if MODEL_ORDER and not loss_df.empty:
    loss_plot = loss_df.groupby("pid").filter(
        lambda g: len(g) == len(MODEL_ORDER)
    ).copy()
else:
    loss_plot = pd.DataFrame(columns=["pid", "model_type", "loss"])

sources: list[tuple[str, pd.DataFrame]] = [("Human", human)] + [
    (mt, models[mt]) for mt in MODEL_ORDER
]

# -- figure --------------------------------------------------------------------
fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=True)
gs = gridspec.GridSpec(2, 5, figure=fig, height_ratios=[1.0, 1.2])

ax_row1: list = []
for i in range(5):
    sharey = ax_row1[0] if i > 0 else None
    ax_row1.append(fig.add_subplot(gs[0, i], sharey=sharey))

ax_param = fig.add_subplot(gs[1, :2])
ax_viol = fig.add_subplot(gs[1, 2:])

_markers = SAMPLE_MARKERS
n_grid = np.arange(OBS_MIN, OBS_MAX + 1, dtype=float)

# Row 1: mean |Δresponse| ± SE + power-law overlay
for ax, (label, _) in zip(ax_row1, sources):
    color = _row1_color(label)
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
            ax.scatter(2.0, y_int * 2.0 ** (-tau), color=color, marker=mkr, s=60, zorder=5)

    ax.set_xlim(0, float(OBS_MAX))
    ax.set_title(_row1_title(label))
    ax.set_xlabel("Observation")
    ax.set_ylabel("Mean |Δresponse|" if label == "Human" else "")
    if label != "Human":
        plt.setp(ax.get_yticklabels(), visible=False)
    sns.despine(ax=ax, top=True, right=True)
for ax in ax_row1[len(sources):]:
    ax.axis("off")

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
plot_palette = {disp: PALETTE[disp] for disp in DISPLAY_ORDER}
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
ax_viol.set_title("Distance to human response change curve")
ax_viol.set_ylabel("Wasserstein distance")
ax_viol.set_xlabel("")
sns.despine(ax=ax_viol, top=True, right=True)

if len(DISPLAY_ORDER) >= 2 and not loss_plot.empty:
    annotate_violins(ax_viol, loss_plot, "model_type", "loss", DISPLAY_ORDER)

# -- save ----------------------------------------------------------------------
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "response_change_yoo.png", dpi=300)
plt.savefig(FIGURES_DIR / "response_change_yoo.pdf")
print("Saved figures/response_change_yoo.{png,pdf}")
