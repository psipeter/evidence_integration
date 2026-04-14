#!/usr/bin/env python3
"""
Jiang task: switch probability vs social conflict (post-hoc logistic fit).

Row 1: Human and three models — mean switch vs conflict with per-participant
logistic fits for three sample participants (low / medium / high conflict
sensitivity). Row 2: human (threshold, steepness) scatter with sample
markers; violin plot of Euclidean parameter distance vs human per model.

On first run (SAMPLE_PIDS = None), prints pid / threshold / steepness for
humans (sorted by threshold), then exits. Set SAMPLE_PIDS and rerun.

Usage:
    python scripts/switch_probability_jiang.py

Data: data/jiang.pkl and data/runs/switch_probability/*_jiang_responses.pkl
No other run folders. Does not write pickle/CSV outputs (figures only).
"""

from __future__ import annotations

import os
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid function
from statannotations.Annotator import Annotator

# -- path setup ----------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import data_path, FIGURES_DIR
from utils.plot_style import apply_style, FIGURE_SIZE, get_palette, SAMPLE_MARKERS

# -- configuration (edit here) -------------------------------------------------
RUN_FOLDER = "switch_probability"

# None: print human pid / threshold / steepness table and exit.
# Else: e.g. {"low": 12, "medium": 88, "high": 156}
# SAMPLE_PIDS: dict[str, int] | None = None
SAMPLE_PIDS: dict[str, int] | None = {"low": 154, "medium": 88, "high": 173}

MODEL_ORDER = ["Bayes", "RL", "DeGroot"]
LINESTYLES = ["solid", "dashed", "dotted"]  # low / medium / high sensitivity
SAMPLE_LABELS = ["low", "medium", "high"]
LINE_ARC = 0.2

# Exclude participants whose max loss across models exceeds this (violin panel).
LOSS_CUTOFF = 50.0
BETA_SAMPLE_SEED = 42

# -- style ---------------------------------------------------------------------
apply_style()
PALETTE = get_palette()


def _model_lookup_series(mdf: pd.DataFrame) -> pd.Series:
    """One model response per (pid, trial, stage)."""
    s = mdf.set_index(["pid", "trial", "stage"])["response"]
    if s.index.duplicated().any():
        s = s[~s.index.duplicated(keep="first")]
    return s


def apply_beta_sampling(
    responses_df: pd.DataFrame,
    params_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Replace continuous model responses with stochastic binary choices.
    For each (pid, trial, stage):
        P(choose +1) = sigmoid(beta * response)
        choice ~ Bernoulli(P), mapped to {-1, +1}
    Uses per-participant beta from params_df.
    """
    rng = np.random.RandomState(seed)
    df = responses_df.copy()
    beta_map = dict(zip(params_df["pid"], params_df["beta"]))
    p_pos = expit(
        df["response"].to_numpy(dtype=float)
        * df["pid"].map(beta_map).to_numpy(dtype=float)
    )
    samples = rng.binomial(1, p_pos)
    df["response"] = np.where(samples == 1, 1, -1).astype(float)
    return df


def observations_switch_conflict(
    human_df: pd.DataFrame,
    model_lookup: pd.Series | None,
) -> pd.DataFrame:
    """
    One row per (pid, trial, stage) with stage > 0.

    Uses human_df row structure (neighbor ``value`` counts). If ``model_lookup``
    is None, responses are human; else prior/current responses come from the
    model and conflict compares neighbor values to the model's sign at s-1.
    """
    rows: list[dict] = []
    for (pid, trial), grp in human_df.groupby(["pid", "trial"], sort=False):
        for st in sorted(grp["stage"].unique()):
            if st <= 0:
                continue
            prev = grp[grp["stage"] == st - 1]
            curr = grp[grp["stage"] == st]
            if len(prev) == 0 or len(curr) == 0:
                continue
            if model_lookup is None:
                prev_resp = float(prev["response"].iloc[0])
                curr_resp = float(curr["response"].iloc[0])
                switch = 1 if prev_resp != curr_resp else 0
                disagree = (curr["value"].astype(float) != prev_resp).sum()
            else:
                key_p = (int(pid), int(trial), int(st - 1))
                key_c = (int(pid), int(trial), int(st))
                try:
                    prev_resp = float(model_lookup.loc[key_p])
                    curr_resp = float(model_lookup.loc[key_c])
                except KeyError:
                    continue
                switch = int(
                    not np.isclose(prev_resp, curr_resp, rtol=0.0, atol=1e-5)
                )
                prev_dir = float(np.sign(prev_resp))
                if prev_dir == 0.0:
                    prev_dir = 1.0
                disagree = (curr["value"].astype(float) != prev_dir).sum()
            n = len(curr)
            conflict = float(disagree / n)
            rows.append(
                {
                    "pid": int(pid),
                    "trial": int(trial),
                    "stage": int(st),
                    "switch": switch,
                    "conflict": conflict,
                }
            )
    return pd.DataFrame(rows)


def agg_conflict_switch(obs: pd.DataFrame) -> pd.DataFrame:
    """Mean switch and count at each natural conflict level (display only)."""
    gb = obs.groupby("conflict", as_index=False).agg(
        n=("switch", "size"),
        mean_switch=("switch", "mean"),
    )
    return gb.sort_values("conflict")


def logistic(x: np.ndarray, threshold: float, steepness: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-steepness * (x - threshold)))


def fit_logistic_pid(obs: pd.DataFrame) -> tuple[float, float]:
    """
    Fit logistic regression to raw binary (switch ~ conflict) data via MLE.
    Reparameterized so threshold = sigmoid(t_raw) in (0,1)
    and steepness = exp(s_raw) > 0.
    Returns (threshold, steepness) in original parameterization.
    """
    if obs["conflict"].nunique() < 2:
        return float("nan"), float("nan")

    x = obs["conflict"].to_numpy(dtype=float)
    y = obs["switch"].to_numpy(dtype=float)

    def neg_log_likelihood(params):
        t_raw, s_raw = params
        threshold = expit(t_raw)
        steepness = np.exp(s_raw)
        p = logistic(x, threshold, steepness)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    try:
        result = minimize(
            neg_log_likelihood,
            x0=[0.0, 1.0],  # threshold=0.5, steepness=e≈2.7
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6},
        )
        if not result.success:
            return float("nan"), float("nan")
        t_raw, s_raw = result.x
        threshold = float(expit(t_raw))
        steepness = float(np.exp(s_raw))
        return threshold, steepness
    except Exception:
        return float("nan"), float("nan")


def fit_all_pids(obs: pd.DataFrame) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for pid, sub in obs.groupby("pid"):
        out[int(pid)] = fit_logistic_pid(sub)
    return out


def params_to_tidy(params: dict[int, tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for pid, (threshold, steepness) in params.items():
        rows.append(
            {
                "pid": int(pid),
                "midpoint": float(threshold),
                "steepness": float(steepness),
                "tangent": float(steepness) / 4.0,
            }
        )
    return pd.DataFrame(rows)


def build_loss_long(
    human_params: dict[int, tuple[float, float]],
    model_params: dict[str, dict[int, tuple[float, float]]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for pid, (th_h, st_h) in human_params.items():
        if not (np.isfinite(th_h) and np.isfinite(st_h)):
            continue
        for mt in MODEL_ORDER:
            if pid not in model_params[mt]:
                continue
            th_m, st_m = model_params[mt][pid]
            if not (np.isfinite(th_m) and np.isfinite(st_m)):
                continue
            loss = float(np.hypot(th_h - th_m, st_h - st_m))
            rows.append({"pid": pid, "model_type": mt, "loss": loss})
    return pd.DataFrame(rows)


# -- load data & human-only analysis ------------------------------------------
human = pd.read_pickle(data_path("jiang.pkl"))

obs_human = observations_switch_conflict(human, None)
params_human = fit_all_pids(obs_human)

# First run: human summary only (no model pickles required)
if SAMPLE_PIDS is None:
    tbl = pd.DataFrame(
        [
            {
                "pid": pid,
                "threshold": params_human[pid][0],
                "steepness": params_human[pid][1],
            }
            for pid in sorted(params_human.keys())
        ]
    )
    tbl = tbl[np.isfinite(tbl["threshold"]) & np.isfinite(tbl["steepness"])]
    tbl = tbl.sort_values("threshold").reset_index(drop=True)
    print("pid / threshold / steepness (human, sorted by threshold):")
    print(tbl.to_string(index=False))
    print("\nSet SAMPLE_PIDS = {'low': ..., 'medium': ..., 'high': ...} and rerun.")
    sys.exit(0)

run_dir = data_path("runs") / RUN_FOLDER
models: dict[str, pd.DataFrame] = {}
for mt in MODEL_ORDER:
    path = run_dir / f"{mt}_jiang_responses.pkl"
    params_path = run_dir / f"{mt}_jiang_params.pkl"
    assert path.exists(), f"Missing model responses: {path}"
    assert params_path.exists(), f"Missing model params: {params_path}"
    responses = pd.read_pickle(path)
    params_mt = pd.read_pickle(params_path)
    models[mt] = apply_beta_sampling(responses, params_mt, seed=BETA_SAMPLE_SEED)

params_models: dict[str, dict[int, tuple[float, float]]] = {}
obs_models: dict[str, pd.DataFrame] = {}

for mt, mdf in models.items():
    lookup = _model_lookup_series(mdf)
    obs_m = observations_switch_conflict(human, lookup)
    obs_models[mt] = obs_m
    params_models[mt] = fit_all_pids(obs_m)

params_by_source: dict[str, pd.DataFrame] = {"Human": params_to_tidy(params_human)}
for mt in MODEL_ORDER:
    params_by_source[mt] = params_to_tidy(params_models[mt])

sample_pids = [int(SAMPLE_PIDS[k]) for k in SAMPLE_LABELS]
for lab, pid in zip(SAMPLE_LABELS, sample_pids):
    if pid not in params_human or not np.isfinite(params_human[pid][0]):
        raise ValueError(f"SAMPLE_PIDS[{lab!r}]={pid} missing or invalid fit")

loss_df = build_loss_long(params_human, params_models)
# Paired tests: keep pids with all three models
_complete = loss_df.groupby("pid").filter(lambda g: len(g) == 3)
if _complete.empty:
    raise RuntimeError("No participants with valid loss for all three models.")
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

# Row 1: logistic regplot + midpoint/tangent overlays
obs_by_source = {"Human": obs_human}
obs_by_source.update(obs_models)
conflict_bins = np.linspace(0, 1.0, 5)

for ax, (label, _) in zip(ax_row1, sources):
    color = PALETTE[label]
    obs_src = obs_by_source[label]
    param_src = params_by_source[label]
    for pid, ls, mkr in zip(sample_pids, LINESTYLES, _markers):
        obs_pid = obs_src[obs_src["pid"] == pid]
        if obs_pid.empty:
            continue
        # Logistic curve with CI band
        sns.regplot(
            data=obs_pid,
            x="conflict",
            y="switch",
            x_bins=conflict_bins,
            logistic=True,
            scatter=False,
            color=color,
            line_kws={"linestyle": ls, "linewidth": 1.5},
            ax=ax,
        )
        prow = param_src[param_src["pid"] == pid]
        if prow.empty:
            continue
        midpoint = float(prow["midpoint"].iloc[0])
        tangent = float(prow["tangent"].iloc[0])
        # Midpoint marker
        ax.scatter(midpoint, 0.5, color=color, marker=mkr, s=60, zorder=5)
        # Tangent at inflection
        line_span = LINE_ARC / np.sqrt(1.0 + tangent ** 2)
        x_tan = np.linspace(midpoint - line_span, midpoint + line_span, 100)
        y_tan = tangent * (x_tan - midpoint) + 0.5
        ax.plot(
            x_tan,
            y_tan,
            color="black",
            linestyle="-",
            linewidth=2.0,
            alpha=0.9,
            zorder=6,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(label)
    ax.set_xlabel("Conflict")
    ax.set_ylabel("P(switch)" if label == "Human" else "")
    if label != "Human":
        plt.setp(ax.get_yticklabels(), visible=False)
    sns.despine(ax=ax, top=True, right=True)

# Row 2 left: threshold vs steepness
h_tbl = pd.DataFrame(
    [{"pid": p, "threshold": params_human[p][0], "steepness": params_human[p][1]} for p in params_human]
)
h_tbl = h_tbl[np.isfinite(h_tbl["threshold"]) & np.isfinite(h_tbl["steepness"])]
x_clip = float(np.percentile(h_tbl["threshold"], 99))
n_outside = int((h_tbl["threshold"] > x_clip).sum())
if n_outside > 0:
    print(
        f"Threshold axis clipped at {x_clip:.3f} "
        f"({n_outside} participant(s) excluded from scatter and violin plots)"
    )
valid_pids = h_tbl[h_tbl["threshold"] <= x_clip]["pid"].values
h_tbl = h_tbl[h_tbl["threshold"] <= x_clip].copy()
loss_plot = loss_plot[loss_plot["pid"].isin(valid_pids)].copy()
sns.kdeplot(
    data=h_tbl,
    x="threshold",
    y="steepness",
    fill=True,
    alpha=0.6,
    color=PALETTE["Human"],
    ax=ax_param,
)
for pid, mkr, lbl in zip(sample_pids, _markers, SAMPLE_LABELS):
    row = h_tbl[h_tbl["pid"] == pid]
    if row.empty:
        continue
    ax_param.scatter(
        row["threshold"],
        row["steepness"],
        s=80,
        facecolors="none",
        edgecolors=PALETTE["Human"],
        linewidths=1.5,
        marker=mkr,
        zorder=5,
    )
    ax_param.text(
        float(row["threshold"].iloc[0]),
        float(row["steepness"].iloc[0]) + 0.02 * h_tbl["steepness"].max(),
        lbl,
        ha="center",
        va="bottom",
        fontsize=7,
        color=PALETTE["Human"],
    )

ax_param.set_xlabel("Threshold (conflict)")
ax_param.set_ylabel("Steepness")
ax_param.set_title("Participant logistic parameters")
ax_param.set_xlim(0.4, 1.0)
sns.despine(ax=ax_param, top=True, right=True)

# Row 2 right: loss violins
# Temporary: exclude participants with extreme model losses pending
# improved model fitting. Remove when model fits are fixed.
bad_pids = loss_plot.groupby("pid")["loss"].max()
extreme_pids = bad_pids[bad_pids > LOSS_CUTOFF].index
if len(extreme_pids) > 0:
    print(
        f"Loss cutoff {LOSS_CUTOFF}: {len(extreme_pids)} participant(s) "
        f"excluded from violin plot (pending model fit improvements)"
    )
loss_plot = loss_plot[~loss_plot["pid"].isin(extreme_pids)].copy()

plot_palette = {k: PALETTE[k] for k in MODEL_ORDER}
sns.violinplot(
    data=loss_plot,
    x="model_type",
    y="loss",
    order=MODEL_ORDER,
    hue="model_type",
    palette=plot_palette,
    inner="point",
    legend=False,
    cut=0,
    ax=ax_viol,
)
np.random.seed(42)
sns.stripplot(
    data=loss_plot,
    x="model_type",
    y="loss",
    order=MODEL_ORDER,
    color="0.2",
    alpha=0.5,
    jitter=0.2,
    size=4,
    ax=ax_viol,
)
ax_viol.set_title("Distance to human parameters")
ax_viol.set_ylabel("Euclidean loss")
ax_viol.set_xlabel("")
sns.despine(ax=ax_viol, top=True, right=True)

pairs = list(combinations(MODEL_ORDER, 2))
annotator = Annotator(
    ax_viol,
    pairs,
    data=loss_plot,
    x="model_type",
    y="loss",
    order=MODEL_ORDER,
)
annotator.configure(test="Wilcoxon", text_format="star", loc="inside")
with open(os.devnull, "w") as devnull:
    old_stdout = sys.stdout
    sys.stdout = devnull
    annotator.apply_and_annotate()
    sys.stdout = old_stdout

# -- save ----------------------------------------------------------------------
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "switch_probability_jiang.png", dpi=300)
plt.savefig(FIGURES_DIR / "switch_probability_jiang.pdf")
print("Saved figures/switch_probability_jiang.{png,pdf}")
# bad_pids = loss_plot.groupby("pid")["loss"].max()
# print(f"Pids excluded at cutoff 50: {(bad_pids > 50).sum()} / {len(bad_pids)}")