#!/usr/bin/env python3
"""
Jiang task: P(switch | neighbor disagrees) vs neighbor rd.

Top row: Human and models — logistic curves for sample participants (linestyle
per pid). Bottom left: KDE of per-participant logistic slopes (rd → switch).
Bottom right: shape-loss violins (performance shape_component or fallback).
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
from utils.plot_style import annotate_violins, apply_style, FIGURE_SIZE, get_palette

SAMPLE_PIDS = {"low": 179, "medium": 81, "high": 202}
LINESTYLES = ["solid", "dashed", "dotted"]  # high / medium / low
MODEL_ORDER = ["Bayes", "RL", "DeGroot", "NEF_recurrent"]


def _display(mt: str) -> str:
    if mt.startswith("NEF"):
        return "NEF"
    if mt == "RL_lambda_rd":
        return "RL_λ_rd"
    return mt


def _apply_beta(
    resp_df: pd.DataFrame, params_df: pd.DataFrame, seed: int = 42
) -> pd.DataFrame:
    """Apply beta sampling to convert continuous responses to binary ±1."""
    from scipy.special import expit

    rng = np.random.RandomState(seed)
    beta_map = dict(zip(params_df["pid"], params_df["beta"]))
    df = resp_df.copy()
    beta_vals = df["pid"].map(beta_map).fillna(1.0).values
    p_pos = expit(df["response"].values * beta_vals)
    df["response"] = np.where(rng.binomial(1, p_pos) == 1, 1.0, -1.0)
    return df


def _build_switch_rd_df(
    resp_df: pd.DataFrame,
    human_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each (pid, trial, stage 2+3), find disagreeing neighbors and record
    (pid, true_rd, switch). One row per disagreeing neighbor observation.
    resp_df must have binary ±1 responses (after beta sampling).
    """
    model_resp = (
        resp_df[resp_df["stage"].isin([1, 2, 3])][["pid", "trial", "stage", "response"]]
        .drop_duplicates(["pid", "trial", "stage"])
        .rename(columns={"response": "model_response"})
    )
    rows = []
    for (pid, trial), grp in human_df.groupby(["pid", "trial"]):
        for stage in [2, 3]:
            curr = grp[grp["stage"] == stage]
            prev_stage = stage - 1
            prev_resp = model_resp.query(
                "pid == @pid & trial == @trial & stage == @prev_stage"
            )
            curr_resp = model_resp.query(
                "pid == @pid & trial == @trial & stage == @stage"
            )
            if prev_resp.empty or curr.empty or curr_resp.empty:
                continue
            prev_sign = float(prev_resp["model_response"].iloc[0])
            curr_sign = float(curr_resp["model_response"].iloc[0])
            switch = int(prev_sign != curr_sign)
            for _, neighbor in curr.iterrows():
                if float(neighbor["value"]) != prev_sign:
                    rows.append(
                        {
                            "pid": int(pid),
                            "true_rd": float(neighbor["true_rd"]),
                            "switch": switch,
                        }
                    )
    return pd.DataFrame(rows)


def _compute_logistic_slope(switch_rd_df: pd.DataFrame) -> pd.Series:
    """Fit logistic regression slope of switch vs true_rd per pid."""
    from scipy.optimize import minimize
    from scipy.special import expit

    slopes = {}
    for pid, grp in switch_rd_df.groupby("pid"):
        if len(grp) < 10:
            continue
        x = grp["true_rd"].values
        y = grp["switch"].values.astype(float)

        def neg_log_lik(params):
            a, b = params
            p = np.clip(expit(a * x + b), 1e-7, 1 - 1e-7)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        res = minimize(neg_log_lik, [1.0, 0.0], method="Nelder-Mead")
        slopes[pid] = float(res.x[0])
    return pd.Series(slopes)


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


parser = argparse.ArgumentParser(description="Jiang social influence figure")
parser.add_argument("--run_folder", type=str, default="joint_loss")
parser.add_argument("--include_rl_lambda", action="store_true", default=False)
args = parser.parse_args()

run_folder = args.run_folder
model_order = MODEL_ORDER.copy()
if args.include_rl_lambda:
    model_order.append("RL_lambda_rd")

apply_style()
PALETTE = get_palette()

run_dir = data_path("runs") / run_folder
human = pd.read_pickle(data_path("jiang.pkl"))

models: dict[str, pd.DataFrame] = {}
loaded_models: list[str] = []
for mt in model_order:
    resp_path = run_dir / f"{mt}_jiang_responses.pkl"
    params_path = run_dir / f"{mt}_jiang_params.pkl"
    if not resp_path.exists():
        print(f"Warning: missing {resp_path.name}, skipping {mt}")
        continue
    resp = pd.read_pickle(resp_path)
    if params_path.exists():
        params_df = pd.read_pickle(params_path)
        if "beta" in params_df.columns:
            resp = _apply_beta(resp, params_df)
    models[mt] = resp
    loaded_models.append(mt)

MODEL_ORDER = loaded_models
DISPLAY_ORDER = [_display(mt) for mt in MODEL_ORDER]

human_resp_dedup = (
    human.groupby(["pid", "trial", "stage"])["response"]
    .first()
    .reset_index()
)
sources: list[tuple[str, pd.DataFrame]] = [("Human", human_resp_dedup)] + [
    (mt, models[mt]) for mt in MODEL_ORDER
]
switch_rd_dfs: dict[str, pd.DataFrame] = {}
for label, resp_df in sources:
    switch_rd_dfs[label] = _build_switch_rd_df(resp_df, human)

slope_series: dict[str, pd.Series] = {}
for label in switch_rd_dfs:
    slope_series[label] = _compute_logistic_slope(switch_rd_dfs[label])

n_top = len(loaded_models) + 1

fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=True)
n_gs_cols = max(n_top, 4)
gs = gridspec.GridSpec(2, n_gs_cols, figure=fig, height_ratios=[1, 1.2])
ax_top: list = []
for i in range(n_top):
    sharey = ax_top[0] if i > 0 else None
    ax_top.append(fig.add_subplot(gs[0, i], sharey=sharey))
ax_kde = fig.add_subplot(gs[1, : n_gs_cols // 2])
ax_viol = fig.add_subplot(gs[1, n_gs_cols // 2 :])

# Top row: P(switch | disagree) vs neighbor rd
for ax, (label, resp_df) in zip(ax_top, sources):
    color = (
        PALETTE["Human"]
        if label == "Human"
        else PALETTE.get(label, PALETTE.get(_display(label), "gray"))
    )
    for i, (pid_label, pid) in enumerate(SAMPLE_PIDS.items()):
        pid_df = switch_rd_dfs[label][switch_rd_dfs[label]["pid"] == pid]
        if pid_df.empty:
            continue
        sns.regplot(
            data=pid_df,
            x="true_rd",
            y="switch",
            logistic=True,
            scatter=False,
            line_kws={"color": color, "linewidth": 2, "linestyle": LINESTYLES[i]},
            ci=95,
            ax=ax,
        )

    if label == "Human":
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [0],
                [0],
                color=PALETTE["Human"],
                linestyle=LINESTYLES[i],
                linewidth=2,
                label=f"#{pid}",
            )
            for i, (_, pid) in enumerate(SAMPLE_PIDS.items())
        ]
        ax.legend(handles=handles, title="Participant", frameon=False, loc="upper left")

    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Neighbor rd")
    ax.set_ylabel("P(switch | neighbor disagrees)" if label == "Human" else "")
    ax.set_title(_display(label))
    if label != "Human":
        plt.setp(ax.get_yticklabels(), visible=False)
    sns.despine(ax=ax, top=True, right=True)

# Bottom left: KDE of logistic slopes
human_slopes = slope_series["Human"]
if len(human_slopes.dropna()) >= 2:
    sns.kdeplot(
        human_slopes.values,
        ax=ax_kde,
        color=PALETTE["Human"],
        fill=True,
        alpha=0.3,
    )
    from scipy.stats import gaussian_kde

    kde_fn = gaussian_kde(human_slopes.dropna().values)
    for i, (_, pid) in enumerate(SAMPLE_PIDS.items()):
        slope_val = human_slopes.get(pid, np.nan)
        if not np.isfinite(slope_val):
            continue
        kde_h = float(kde_fn(np.array([slope_val]))[0])
        ax_kde.plot(
            [slope_val, slope_val],
            [0, kde_h],
            color=PALETTE["Human"],
            linestyle=LINESTYLES[i],
            linewidth=1.5,
        )
else:
    ax_kde.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax_kde.transAxes)
ax_kde.set_xlabel("Logistic slope (rd → switch)")
ax_kde.set_ylabel("Density")
ax_kde.set_title("Population rd sensitivity")
sns.despine(ax=ax_kde, top=True, right=True)

# Bottom right: shape loss violins (stub)
loss_df = _load_loss_long(run_dir, MODEL_ORDER, "jiang")
if not loss_df.empty:
    loss_df["model_type"] = loss_df["model_type"].apply(_display)
if MODEL_ORDER and not loss_df.empty:
    _models_with_loss = loss_df["model_type"].unique().tolist()
    loss_plot = loss_df.groupby("pid").filter(
        lambda g: len(g) == len(_models_with_loss)
    ).copy()
else:
    loss_plot = pd.DataFrame(columns=["pid", "model_type", "loss"])

plot_palette = {
    _display(mt): PALETTE.get(mt, PALETTE.get(_display(mt), "gray")) for mt in MODEL_ORDER
}
if DISPLAY_ORDER and not loss_plot.empty:
    sns.boxplot(
        data=loss_plot,
        x="model_type",
        y="loss",
        order=DISPLAY_ORDER,
        hue="model_type",
        palette=plot_palette,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 5,
        },
        legend=False,
        ax=ax_viol,
    )
else:
    ax_viol.text(0.5, 0.5, "no model data", ha="center", va="center", transform=ax_viol.transAxes)
ax_viol.set_title("rd-consistency slope distance")
ax_viol.set_ylabel("Shape loss (|Δslope|)")
ax_viol.set_xlabel("")
sns.despine(ax=ax_viol, top=True, right=True)

if len(DISPLAY_ORDER) >= 2 and not loss_plot.empty:
    annotate_violins(ax_viol, loss_plot, "model_type", "loss", DISPLAY_ORDER)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "social_influence_jiang.png", dpi=300)
plt.savefig(FIGURES_DIR / "social_influence_jiang.pdf")
print("Saved figures/social_influence_jiang.{png,pdf}")
