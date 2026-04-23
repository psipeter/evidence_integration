#!/usr/bin/env python3
"""
Plot results from experiment_01_error_activity.

Regplot of mean error population activity vs prediction error
across all pids, trials, and observations.

Usage:
    python scripts/plot_experiment_01.py
    python scripts/plot_experiment_01.py --dataset yoo --pe_type decoded
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, data_path
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette

DATASET = "carrabin"
OBS_MIN = 3
OBS_MAX = 5

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=DATASET)
parser.add_argument(
    "--pe_type",
    type=str,
    default="raw",
    choices=("raw", "decoded"),
    help="Which prediction error to plot",
)
args = parser.parse_args()

data_file = (
    data_path("experiments") / "experiment_01" / f"experiment_01_{args.dataset}.pkl"
)
if not data_file.exists():
    raise FileNotFoundError(
        f"Combined output not found: {data_file}\n"
        f"Run: python experiments/experiment_01_error_activity.py "
        f"--collect --dataset {args.dataset}"
    )

df = pd.read_pickle(data_file)
obs_col = "stage" if args.dataset == "jiang" else "observation"
plot_df = df[(df[obs_col] >= OBS_MIN) & (df[obs_col] <= OBS_MAX)].copy()

if plot_df.empty:
    raise ValueError(f"No data for {obs_col} {OBS_MIN}-{OBS_MAX} in {args.dataset}")

# load human responses for trial-level response variability
human = pd.read_pickle(data_path(f"{args.dataset}.pkl"))
if args.dataset in ("carrabin", "yoo"):
    obs_max_col = human.groupby(["pid", "trial"])["observation"].max().reset_index()
    obs_max_col.columns = ["pid", "trial", "max_obs"]
    human_final = human.merge(obs_max_col, on=["pid", "trial"])
    human_final = human_final[human_final["observation"] == human_final["max_obs"]]
    trial_response = human_final[["pid", "trial", "response"]].rename(
        columns={"response": "human_response"}
    )
elif args.dataset == "jiang":
    # for jiang use last stage response
    stage_max = human.groupby(["pid", "trial"])["stage"].max().reset_index()
    stage_max.columns = ["pid", "trial", "max_stage"]
    human_final = human.merge(stage_max, on=["pid", "trial"])
    human_final = human_final[human_final["stage"] == human_final["max_stage"]]
    trial_response = human_final[["pid", "trial", "response"]].rename(
        columns={"response": "human_response"}
    )
else:
    raise ValueError(f"Unsupported dataset: {args.dataset}")

pid_response_std = (
    human.groupby("pid")["response"].std().rename("response_std").reset_index()
)

df_merged = df.merge(trial_response, on=["pid", "trial"], how="left")
df_merged["pe_sign"] = (df_merged["prediction_error_raw"] >= 0).astype(int)

variability_rows = []
for (pid, obs, pe_sign), group in df_merged.groupby(["pid", obs_col, "pe_sign"]):
    if obs < OBS_MIN or obs > OBS_MAX:
        continue
    if len(group) < 2:
        continue
    resp_std = float(group["human_response"].std())
    fr_std_on = float(group["mean_activity_on"].std())
    fr_std_off = float(group["mean_activity_off"].std())
    fr_std_mean = float(np.mean([fr_std_on, fr_std_off]))
    variability_rows.append(
        {
            "pid": pid,
            obs_col: obs,
            "pe_sign": pe_sign,
            "response_std": resp_std,
            "firing_rate_std": fr_std_mean,
        }
    )
var_df = pd.DataFrame(variability_rows)

resid_rows = []
for pid, pid_df in df_merged.groupby("pid"):
    obs_filtered = pid_df[
        (pid_df[obs_col] >= OBS_MIN) & (pid_df[obs_col] <= OBS_MAX)
    ].copy()
    if len(obs_filtered) < 5:
        continue

    X = obs_filtered[[obs_col, "prediction_error_raw"]].to_numpy(dtype=float)
    X_aug = np.column_stack([np.ones(len(X)), X])

    resid_stds = []
    for neuron_col in ["mean_activity_on", "mean_activity_off"]:
        y = obs_filtered[neuron_col].to_numpy(dtype=float)
        coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        y_pred = X_aug @ coeffs
        residuals = y - y_pred
        resid_stds.append(float(residuals.std()))

    resid_rows.append(
        {
            "pid": pid,
            "resid_firing_rate_std": float(np.mean(resid_stds)),
        }
    )

resid_df = pd.DataFrame(resid_rows)
resid_df = resid_df.merge(pid_response_std, on="pid")

pe_col = f"prediction_error_{args.pe_type}"

apply_style()
PALETTE = get_palette()
color_on = PALETTE.get("NEF_recurrent", PALETTE.get("NEF", "C0"))
color_off = PALETTE.get("RL", "C1")

fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE, constrained_layout=True)
ax_regplot = axes[0]
ax_scatter = axes[1]
ax_resid = axes[2]
sns.regplot(
    data=plot_df,
    x=pe_col,
    y="mean_activity_on",
    scatter=True,
    scatter_kws={"alpha": 0.15, "s": 6, "color": color_on},
    line_kws={"color": color_on, "linewidth": 2, "label": "on neurons"},
    ax=ax_regplot,
)
sns.regplot(
    data=plot_df,
    x=pe_col,
    y="mean_activity_off",
    scatter=True,
    scatter_kws={"alpha": 0.15, "s": 6, "color": color_off},
    line_kws={"color": color_off, "linewidth": 2, "label": "off neurons"},
    ax=ax_regplot,
)
ax_regplot.legend()
ax_regplot.set_xlabel(f"Prediction error ({args.pe_type})")
ax_regplot.set_ylabel("Mean neuron activity (Hz)")
ax_regplot.set_title(f"{args.dataset} — {obs_col}s {OBS_MIN}–{OBS_MAX}")
sns.despine(ax=ax_regplot, top=True, right=True)

color = PALETTE.get("NEF_recurrent", PALETTE.get("NEF", "C0"))
sns.regplot(
    data=var_df,
    x="response_std",
    y="firing_rate_std",
    scatter=True,
    scatter_kws={"alpha": 0.4, "s": 20, "color": color},
    line_kws={"color": color, "linewidth": 2},
    ax=ax_scatter,
)
ax_scatter.set_xlabel("Response std (within group)")
ax_scatter.set_ylabel("Mean firing rate std (Hz)")
ax_scatter.set_title("Response vs neural variability")
sns.despine(ax=ax_scatter, top=True, right=True)

sns.regplot(
    data=resid_df,
    x="response_std",
    y="resid_firing_rate_std",
    scatter=True,
    scatter_kws={"alpha": 0.7, "s": 40, "color": color},
    line_kws={"color": color, "linewidth": 2},
    ax=ax_resid,
)
ax_resid.set_xlabel("Response std (per pid)")
ax_resid.set_ylabel("Residual firing rate std (Hz)")
ax_resid.set_title("Response vs residual neural variability")
sns.despine(ax=ax_resid, top=True, right=True)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fname = f"experiment_01_{args.dataset}"
plt.savefig(FIGURES_DIR / f"{fname}.png", dpi=300)
plt.savefig(FIGURES_DIR / f"{fname}.pdf")
print(f"Saved figures/{fname}.{{png,pdf}}")
