#!/usr/bin/env python3
"""
Jiang task: ND-weighted regression metric (OLS of current sign on unweighted /
orthogonalized ND-weighted neighbor sums) per Jiang et al.

Top row: ND coefficient bars with stages on the x-axis, color by Human/model,
hatch pattern by sample participant (low/medium/high).
Bottom left: KDE of human stage-2 ND coefficients across pids with sample lines.
Bottom right: boxplots of shape loss per model (ND coefficient distance).
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import data_path, FIGURES_DIR
from utils.plot_style import annotate_violins, apply_style, FIGURE_SIZE, get_palette

SAMPLE_PIDS = {"low": 109, "medium": 132, "high": 94}
HATCHES = ["", "///", "xxx"]  # low / medium / high pid
LINESTYLES = ["solid", "dashed", "dotted"]  # aligned with SAMPLE_PIDS order
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


def _compute_nd_coefs(
    resp_df: pd.DataFrame,
    human_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each pid and stage, compute OLS coefficient for ND-weighted sum
    (orthogonalized against unweighted sum) predicting current response sign.

    Returns DataFrame with columns: pid, stage, coef_nd, coef_uw, se_nd, pval_nd.
    ``resp_df`` must have binary ±1 responses (after beta sampling if needed).
    """
    from numpy.linalg import lstsq
    from scipy.stats import t as t_dist

    rows: list[dict] = []
    for (pid, trial), grp in human_df.groupby(["pid", "trial"], sort=False):
        for stage in [1, 2]:
            curr = grp[grp["stage"] == stage]
            curr_resp = resp_df[
                (resp_df["pid"] == pid)
                & (resp_df["trial"] == trial)
                & (resp_df["stage"] == stage)
            ]
            if curr.empty or curr_resp.empty:
                continue
            curr_sign = float(curr_resp["response"].iloc[0])
            obs = curr["value"].astype(float).values
            rd = curr["true_rd"].astype(float).values
            unweighted = float(np.sum(obs))
            nd_weighted = float(np.sum(rd * obs))
            rows.append(
                {
                    "pid": int(pid),
                    "trial": int(trial),
                    "stage": int(stage),
                    "curr_sign": curr_sign,
                    "unweighted": unweighted,
                    "nd_weighted": nd_weighted,
                }
            )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    coefs: list[dict] = []
    for stage in [1, 2]:
        s_df = df[df["stage"] == stage].copy()
        if len(s_df) < 10:
            continue
        b = lstsq(
            s_df["unweighted"].values.reshape(-1, 1),
            s_df["nd_weighted"].values,
            rcond=None,
        )[0][0]
        s_df["nd_orth"] = s_df["nd_weighted"] - b * s_df["unweighted"]

        for pid, grp in s_df.groupby("pid"):
            if len(grp) < 5:
                continue
            x_uw = grp["unweighted"].values
            x_nd = grp["nd_orth"].values
            y = grp["curr_sign"].values.astype(float)
            X = np.column_stack([np.ones(len(y)), x_uw, x_nd])
            coeffs, _, _, _ = lstsq(X, y, rcond=None)
            n, p = len(y), X.shape[1]
            y_hat = X @ coeffs
            sigma2 = np.sum((y - y_hat) ** 2) / max(n - p, 1)
            cov = sigma2 * np.linalg.pinv(X.T @ X)
            se_nd = float(np.sqrt(max(cov[2, 2], 0)))
            t_stat = coeffs[2] / se_nd if se_nd > 0 else 0.0
            pval = float(2 * t_dist.sf(abs(t_stat), df=max(n - p, 1)))
            coefs.append(
                {
                    "pid": int(pid),
                    "stage": int(stage),
                    "coef_nd": float(coeffs[2]),
                    "coef_uw": float(coeffs[1]),
                    "se_nd": se_nd,
                    "pval_nd": pval,
                }
            )
    return pd.DataFrame(coefs)


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
    human.groupby(["pid", "trial", "stage"])["response"].first().reset_index()
)
human_coefs = _compute_nd_coefs(human_resp_dedup, human)

model_coefs: dict[str, pd.DataFrame] = {}
for mt in MODEL_ORDER:
    if mt not in models:
        continue
    model_coefs[mt] = _compute_nd_coefs(models[mt], human)

sources: list[tuple[str, pd.DataFrame]] = [("Human", human_resp_dedup)] + [
    (mt, models[mt]) for mt in MODEL_ORDER
]

coefs_by_label: dict[str, pd.DataFrame] = {"Human": human_coefs}
for mt in MODEL_ORDER:
    if mt in model_coefs:
        coefs_by_label[mt] = model_coefs[mt]

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

# Top row: stages on x; bar color Human/model palette; hatch = participant
x = np.arange(2)  # stages 1, 2
bar_width = 0.8 / len(SAMPLE_PIDS)
offsets = (
    np.linspace(-(len(SAMPLE_PIDS) - 1) / 2, (len(SAMPLE_PIDS) - 1) / 2, len(SAMPLE_PIDS))
    * bar_width
)

for ax, (label, _) in zip(ax_top, sources):
    coefs_df = coefs_by_label.get(label, pd.DataFrame())
    if coefs_df.empty:
        ax.text(
            0.5,
            0.5,
            "insufficient data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(_display(label))
        sns.despine(ax=ax, top=True, right=True)
        continue

    color = (
        PALETTE["Human"]
        if label == "Human"
        else PALETTE.get(label, PALETTE.get(_display(label), "gray"))
    )
    for i, (pid_label, pid) in enumerate(SAMPLE_PIDS.items()):
        coef_vals = []
        se_vals = []
        for stage in [1, 2]:
            row = coefs_df[(coefs_df["pid"] == pid) & (coefs_df["stage"] == stage)]
            coef_vals.append(
                float(row["coef_nd"].iloc[0]) if not row.empty else 0.0
            )
            se_vals.append(float(row["se_nd"].iloc[0]) if not row.empty else 0.0)
        ax.bar(
            x + offsets[i],
            coef_vals,
            bar_width,
            color=color,
            hatch=HATCHES[i],
            edgecolor="white",
            yerr=se_vals,
            capsize=3,
            error_kw={"elinewidth": 1.0, "ecolor": "gray"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Stage 1", "Stage 2"])
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(None)
    ax.set_ylabel("ND coefficient" if label == "Human" else "")
    ax.set_title(_display(label))
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
        ax.legend(handles=handles, title="Participant", frameon=False)
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
    sns.despine(ax=ax, top=True, right=True)

# Bottom left: KDE of stage-2 human ND coef
stage2_coefs = human_coefs[human_coefs["stage"] == 2] if not human_coefs.empty else pd.DataFrame()
vals_s2 = stage2_coefs["coef_nd"].dropna().values if not stage2_coefs.empty else np.array([])
if len(vals_s2) >= 2:
    sns.kdeplot(vals_s2, ax=ax_kde, color=PALETTE["Human"], fill=True, alpha=0.3)
    from scipy.stats import gaussian_kde

    kde_fn = gaussian_kde(vals_s2)
    for i, (_, pid) in enumerate(SAMPLE_PIDS.items()):
        row = stage2_coefs[stage2_coefs["pid"] == pid]
        if row.empty:
            continue
        val = float(row["coef_nd"].iloc[0])
        kde_h = float(kde_fn(np.array([val]))[0])
        ax_kde.plot(
            [val, val],
            [0, kde_h],
            color=PALETTE["Human"],
            linestyle=LINESTYLES[i],
            linewidth=1.5,
        )
elif len(vals_s2) == 1:
    ax_kde.axvline(float(vals_s2[0]), color=PALETTE["Human"], linewidth=1.5)
else:
    ax_kde.text(
        0.5,
        0.5,
        "insufficient data",
        ha="center",
        va="center",
        transform=ax_kde.transAxes,
    )
ax_kde.set_xlabel("ND coefficient (stage 2)")
ax_kde.set_ylabel("Density")
ax_kde.set_title("Population rd sensitivity")
sns.despine(ax=ax_kde, top=True, right=True)

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
    _display(mt): PALETTE.get(mt, PALETTE.get(_display(mt), "gray"))
    for mt in MODEL_ORDER
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

ax_viol.set_title("ND coefficient distance (mean |Δcoef|)")
ax_viol.set_ylabel("Shape loss (stage 2)")
ax_viol.set_xlabel("")
sns.despine(ax=ax_viol, top=True, right=True)

if len(DISPLAY_ORDER) >= 2 and not loss_plot.empty:
    annotate_violins(ax_viol, loss_plot, "model_type", "loss", DISPLAY_ORDER)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "social_influence_jiang.png", dpi=300)
plt.savefig(FIGURES_DIR / "social_influence_jiang.pdf")
print("Saved figures/social_influence_jiang.{png,pdf}")
