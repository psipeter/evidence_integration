#!/usr/bin/env python3
"""
Per-qid response variability (noise) for carrabin: human vs fitted models.

Top row: for each source, mean response ± mean per-qid std vs sequence length
(1–5), aggregating qids with sufficient trials. Bottom: population KDE of
mean per-qid std (human) and shape-loss violins from run performance files.

Usage:
    python scripts/response_noise_carrabin.py [--run_folder RUN] [--include_rl_lambda]
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

from fitting.losses import _mean_qid_std
from utils.paths import data_path, FIGURES_DIR
from utils.plot_style import annotate_violins, apply_style, FIGURE_SIZE, get_palette

# -- CLI ----------------------------------------------------------------------
_parser = argparse.ArgumentParser()
_parser.add_argument("--run_folder", type=str, default="joint")
_parser.add_argument("--include_rl_lambda", action="store_true", default=False)
_args, _ = _parser.parse_known_args()
RUN_FOLDER = _args.run_folder

# -- configuration -------------------------------------------------------------
SAMPLE_PIDS = {"low": 14, "medium": 18, "high": 17}
QID_MIN_TRIALS = 10
MODEL_ORDER = ["Bayes", "RL", "NoisyCounting", "NEF_recurrent"]
LINESTYLES = ["solid", "dashed", "dotted"]
HATCHES = ["", "///", "xxx"]  # narrow / medium / broad

# -- CLI ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Carrabin per-qid response noise figure")
parser.add_argument(
    "--run_folder",
    type=str,
    default=RUN_FOLDER,
    help="Run folder under data/runs/ for model pickles",
)
parser.add_argument("--include_rl_lambda", action="store_true", default=False)
args = parser.parse_args()

RUN_FOLDER = args.run_folder

if args.include_rl_lambda:
    MODEL_ORDER = list(MODEL_ORDER) + ["RL_lambda"]

# -- style ---------------------------------------------------------------------
apply_style()
PALETTE = get_palette()


def _display(mt: str) -> str:
    if mt.startswith("NEF"):
        return "NEF"
    return mt


def _qid_length(qid) -> int:
    return len(str(qid).strip())


def _agg_by_length(
    df: pd.DataFrame, pid: int, qid_counts: pd.Series
) -> pd.DataFrame:
    """
    For a given pid, compute mean-of-means and mean-of-stds per sequence length.
    qid_counts: Series of n_trials per qid for this pid (used for filtering).
    Returns DataFrame with columns: length, mean_response, mean_std.
    """
    valid_qids = qid_counts[qid_counts >= QID_MIN_TRIALS].index
    pid_df = df[(df["pid"] == pid) & (df["qid"].isin(valid_qids))]
    pid_df = pid_df.copy()
    pid_df["length"] = pid_df["qid"].apply(_qid_length)
    stats = (
        pid_df.groupby(["length", "qid"])["response"].agg(["mean", "std"]).reset_index()
    )
    agg = stats.groupby("length").agg(
        mean_response=("mean", "mean"),
        mean_std=("std", "mean"),
        std_std=("std", "std"),
    ).reset_index()
    return agg.sort_values("length")


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


# -- load data -----------------------------------------------------------------
run_dir = data_path("runs") / RUN_FOLDER

human = pd.read_pickle(data_path("carrabin.pkl"))

models: dict[str, pd.DataFrame] = {}
loaded_models: list[str] = []
for mt in MODEL_ORDER:
    resp_f = run_dir / f"{mt}_carrabin_responses.pkl"
    if not resp_f.exists():
        print(f"Warning: missing {resp_f.name}, skipping {mt}")
        continue
    models[mt] = pd.read_pickle(resp_f)
    loaded_models.append(mt)

MODEL_ORDER = loaded_models
DISPLAY_ORDER = [_display(mt) for mt in MODEL_ORDER]

qid_map = human[["pid", "trial", "observation", "qid"]].drop_duplicates()


def _with_qid(df: pd.DataFrame) -> pd.DataFrame:
    if "qid" in df.columns:
        return df
    return df.merge(qid_map, on=["pid", "trial", "observation"], how="left")


loss_df = _load_loss_long(run_dir, MODEL_ORDER, "carrabin")
loss_df["model_type"] = loss_df["model_type"].apply(_display)
if MODEL_ORDER and not loss_df.empty:
    loss_plot = loss_df.groupby("pid").filter(
        lambda g: len(g) == len(MODEL_ORDER)
    ).copy()
else:
    loss_plot = pd.DataFrame(columns=["pid", "model_type", "loss"])

# -- ensure sample pids yield length-aggregated data ----------------------------
_has_agg = False
for _pid in SAMPLE_PIDS.values():
    qid_counts = human[human["pid"] == _pid].groupby("qid")["trial"].nunique()
    if not _agg_by_length(human, _pid, qid_counts).empty:
        _has_agg = True
        break
if not _has_agg:
    print("No length-aggregated data for sample pids (check QID_MIN_TRIALS); exiting.")
    sys.exit(1)

# -- figure layout -------------------------------------------------------------
n_top = len(loaded_models) + 1
n_gs = max(n_top, 2)
fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=True)
gs = gridspec.GridSpec(2, n_gs, figure=fig, height_ratios=[1, 1.2])

axes_top: list = []
for i in range(n_top):
    sharey = axes_top[0] if i > 0 else None
    axes_top.append(fig.add_subplot(gs[0, i], sharey=sharey))

left_cols = max(1, n_top // 2)
ax_kde = fig.add_subplot(gs[1, :left_cols])
ax_viol = fig.add_subplot(gs[1, left_cols:])

sources = [("Human", human)] + [
    (mt, _with_qid(models[mt])) for mt in MODEL_ORDER
]

# -- top row -------------------------------------------------------------------
for ax, (label, df) in zip(axes_top, sources):
    if label == "Human":
        color = PALETTE["Human"]
    else:
        color = PALETTE.get(label, PALETTE.get(_display(label), "gray"))

    n_pids = len(SAMPLE_PIDS)
    lengths = [1, 2, 3, 4, 5]
    bar_width = 0.8 / n_pids
    offsets = (
        np.linspace(-(n_pids - 1) / 2, (n_pids - 1) / 2, n_pids) * bar_width
    )

    for i, (_pid_label, pid) in enumerate(SAMPLE_PIDS.items()):
        qid_counts = human[human["pid"] == pid].groupby("qid")["trial"].nunique()
        agg = _agg_by_length(df, pid, qid_counts)
        if agg.empty:
            continue
        agg = agg.set_index("length").reindex(lengths)
        x = np.array(lengths) + offsets[i]
        ax.bar(
            x,
            agg["mean_std"].fillna(0),
            width=bar_width,
            color=color,
            hatch=HATCHES[i],
            edgecolor="white",
            yerr=agg["std_std"].fillna(0),
            capsize=3,
            error_kw={"elinewidth": 1.0, "ecolor": "gray"},
        )
    if label == "Human":
        from matplotlib.patches import Patch

        handles = [
            Patch(
                facecolor=PALETTE["Human"],
                hatch=HATCHES[i],
                edgecolor="white",
                label=f"#{pid}",
            )
            for i, (_, pid) in enumerate(SAMPLE_PIDS.items())
        ]
        ax.legend(handles=handles, title="Participant", frameon=False, loc="upper left")

    ax.set_xticks(lengths)
    ax.set_xticklabels([str(l) for l in lengths])
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Response noise" if label == "Human" else "")
    ax.set_title(_display(label))
    if label != "Human":
        plt.setp(ax.get_yticklabels(), visible=False)
    sns.despine(ax=ax, top=True, right=True)

for ax in axes_top[len(sources) :]:
    ax.axis("off")

# -- bottom left: population KDE -----------------------------------------------
pid_stds: list[float] = []
for _pid, grp in human.groupby("pid"):
    pid_stds.append(_mean_qid_std(grp))

pid_stds_vals = [s for s in pid_stds if np.isfinite(s)]
if pid_stds_vals:
    sns.kdeplot(pid_stds_vals, ax=ax_kde, color=PALETTE["Human"], fill=True, alpha=0.3)
    kde_fn = gaussian_kde(pid_stds_vals)
    for i, (pid_label, pid) in enumerate(SAMPLE_PIDS.items()):
        std_val = _mean_qid_std(human[human["pid"] == pid])
        if not np.isfinite(std_val):
            continue
        kde_height = float(kde_fn(np.array([std_val]))[0])
        ax_kde.plot(
            [std_val, std_val],
            [0, kde_height],
            color=PALETTE["Human"],
            linestyle=LINESTYLES[i],
            linewidth=1.5,
            label=f"#{pid}",
        )
    ax_kde.legend(title="Participant", frameon=False)
ax_kde.set_xlabel("Response noise")
ax_kde.set_ylabel("Density")
ax_kde.set_title("Population response noise distribution")
sns.despine(ax=ax_kde, top=True, right=True)

# -- bottom right: shape loss violins -----------------------------------------
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
    ax_viol.text(
        0.5,
        0.5,
        "no model data",
        ha="center",
        va="center",
        transform=ax_viol.transAxes,
    )
ax_viol.set_title("Shape loss by model")
ax_viol.set_ylabel("Shape loss")
ax_viol.set_xlabel("Model")
sns.despine(ax=ax_viol, top=True, right=True)

if len(DISPLAY_ORDER) >= 2 and not loss_plot.empty:
    annotate_violins(ax_viol, loss_plot, "model_type", "loss", DISPLAY_ORDER)

# -- save ----------------------------------------------------------------------
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "response_noise_carrabin.png", dpi=300)
plt.savefig(FIGURES_DIR / "response_noise_carrabin.pdf")
print("Saved figures/response_noise_carrabin.{png,pdf}")
