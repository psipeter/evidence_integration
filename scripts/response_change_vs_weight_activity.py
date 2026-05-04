#!/usr/bin/env python3
"""
Yoo task: correlation between on-weight neuron activity and |Δresponse| across
observations. Per-participant faded traces plus population mean with regression.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import linregress

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, RUNS_DIR, data_path
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette

ENCODER_THRESHOLD = 0.5  # minimum enc_dim_0 to be classified as on-weight neuron
OBS_MIN = 2  # first observation with defined delta response
OBS_MAX = 30
ALPHA_PID = 0.15  # transparency for per-pid lines
MODEL_TYPE = "NEF_recurrent"
DATASET = "yoo"


def _plot_panel(
    ax,
    pid_results_subset: list[dict],
    mean_activity: np.ndarray,
    mean_delta: np.ndarray,
    color_sig: str,
    color_nonsig: str,
    title: str = "",
    ylabel: str | None = "Mean on-weight neuron activity (Hz)",
) -> None:
    """Panel 1: significance coloring + black population mean."""
    n_sig = sum(1 for p in pid_results_subset if p["pval"] < 0.05)
    n_nonsig = len(pid_results_subset) - n_sig

    for pid_data in pid_results_subset:
        pid_df = pd.DataFrame(
            {"delta": pid_data["delta"], "activity": pid_data["activity"]}
        )
        c = color_sig if pid_data["pval"] < 0.05 else color_nonsig
        sns.regplot(
            data=pid_df,
            x="delta",
            y="activity",
            scatter=False,
            line_kws={"color": c, "linewidth": 0.8},
            ci=95,
            ax=ax,
        )

    mean_df = pd.DataFrame({"activity": mean_activity, "delta": mean_delta})
    fin = np.isfinite(mean_df["activity"].values) & np.isfinite(mean_df["delta"].values)
    if fin.sum() >= 2:
        sns.regplot(
            data=mean_df,
            x="delta",
            y="activity",
            scatter=False,
            line_kws={"color": "black", "linewidth": 2.5},
            ci=95,
            ax=ax,
        )

    ax.set_xlabel(
        "Mean absolute response change per observation (|Δresponse|)"
    )
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles = [
        Line2D(
            [0],
            [0],
            color=color_sig,
            linewidth=1.5,
            label=f"Significant (p<0.05, n={n_sig})",
        ),
        Line2D(
            [0],
            [0],
            color=color_nonsig,
            linewidth=1.5,
            label=f"Non-significant (n={n_nonsig})",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.5,
            label="Population mean",
        ),
    ]
    ax.legend(handles=handles, frameon=False)
    sns.despine(ax=ax, top=True, right=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, default="response")
    parser.add_argument(
        "--split_lambda",
        action="store_true",
        default=False,
        help="Add second panel coloring pids by lambda_ threshold",
    )
    parser.add_argument(
        "--lambda_threshold",
        type=float,
        default=0.4,
        help="Lambda threshold for coloring (default: 0.4)",
    )
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_folder

    acts_all = pd.read_pickle(run_dir / f"activities_error_{DATASET}.pkl")
    encs_all = pd.read_pickle(run_dir / f"encoders_error_{DATASET}.pkl")
    human = pd.read_pickle(data_path(f"{DATASET}.pkl"))

    # always load NEF params for contingency analysis
    nef_params_path = run_dir / f"NEF_recurrent_{DATASET}_params.pkl"
    if nef_params_path.exists():
        nef_params = pd.read_pickle(nef_params_path).set_index("pid")
    else:
        nef_params = None

    neuron_cols = [c for c in acts_all.columns if c.startswith("n")]
    obs_range = np.arange(OBS_MIN, OBS_MAX + 1, dtype=int)

    pid_results: list[dict] = []
    slopes_sig: list[float] = []
    slopes_nonsig: list[float] = []
    activity_rows: list[np.ndarray] = []
    delta_rows: list[np.ndarray] = []

    pids = sorted(human["pid"].unique())

    for pid in pids:
        enc_pid = encs_all[encs_all["pid"] == pid]
        on_idx = enc_pid[enc_pid["enc_dim_0"] > ENCODER_THRESHOLD]["neuron_idx"].values
        cols = [f"n{i}" for i in on_idx if f"n{i}" in neuron_cols]
        if not cols:
            continue

        acts_pid = acts_all[acts_all["pid"] == pid].copy()
        acts_pid["mean_weight_on"] = acts_pid[cols].mean(axis=1)

        hum_pid = human[human["pid"] == pid].sort_values(["trial", "observation"])
        hum_pid = hum_pid.copy()
        hum_pid["prev_response"] = hum_pid.groupby("trial")["response"].shift(1)
        hum_pid["delta_abs"] = (hum_pid["response"] - hum_pid["prev_response"]).abs()

        merged = acts_pid.merge(
            hum_pid[["trial", "observation", "delta_abs"]],
            on=["trial", "observation"],
            how="inner",
        )

        g_act = merged.groupby("observation")["mean_weight_on"].mean()
        g_del = merged.groupby("observation")["delta_abs"].mean()

        activity = np.array([float(g_act[o]) if o in g_act.index else np.nan for o in obs_range])
        delta = np.array([float(g_del[o]) if o in g_del.index else np.nan for o in obs_range])

        mask = np.isfinite(activity) & np.isfinite(delta)
        if int(mask.sum()) < 3:
            continue

        slope, _intercept, r_val, pval, _stderr = linregress(
            activity[mask], delta[mask]
        )
        slope = float(slope)
        r_val = float(r_val)
        pval = float(pval)

        if nef_params is not None and pid in nef_params.index:
            lambda_val = float(nef_params.loc[pid, "lambda_"])
        else:
            lambda_val = float("nan")

        pid_results.append(
            {
                "pid": int(pid),
                "delta": delta,
                "activity": activity,
                "slope": slope,
                "r": r_val,
                "pval": pval,
                "lambda_": lambda_val,
            }
        )

        activity_rows.append(activity)
        delta_rows.append(delta)

        if pval < 0.05:
            slopes_sig.append(slope)
        else:
            slopes_nonsig.append(slope)

    n = len(pid_results)
    n_sig = len(slopes_sig)
    n_nonsig = len(slopes_nonsig)

    mean_activity = np.nanmean(activity_rows, axis=0)
    mean_delta = np.nanmean(delta_rows, axis=0)
    pop_mask = np.isfinite(mean_activity) & np.isfinite(mean_delta)
    if int(pop_mask.sum()) >= 2:
        _slo, _int, pop_r, pop_p, _se = linregress(
            mean_activity[pop_mask], mean_delta[pop_mask]
        )
        pop_r = float(pop_r)
        pop_p = float(pop_p)
    else:
        pop_r = float("nan")
        pop_p = float("nan")

    mean_slope_sig = float(np.mean(slopes_sig)) if slopes_sig else float("nan")
    mean_slope_nonsig = float(np.mean(slopes_nonsig)) if slopes_nonsig else float("nan")

    print(f"Significant pids (p<0.05): {n_sig}/{n}")
    print(f"Non-significant pids: {n_nonsig}/{n}")
    print(f"Mean slope (sig): {mean_slope_sig}")
    print(f"Mean slope (non-sig): {mean_slope_nonsig}")
    print(f"Population mean r={pop_r}, p={pop_p}")

    if nef_params is not None:
        from scipy.stats import fisher_exact

        thresh = args.lambda_threshold
        rows = []
        for p in pid_results:
            pid = p["pid"]
            if pid not in nef_params.index:
                continue
            rows.append(
                {
                    "pid": pid,
                    "sig": p["pval"] < 0.05,
                    "high_lambda": float(nef_params.loc[pid, "lambda_"]) >= thresh,
                    "lambda_": float(nef_params.loc[pid, "lambda_"]),
                }
            )
        ct_df = pd.DataFrame(rows)
        if len(ct_df) > 0:
            table = pd.crosstab(ct_df["high_lambda"], ct_df["sig"])
            n_high = int(ct_df["high_lambda"].sum())
            n_low = int((~ct_df["high_lambda"]).sum())
            sig_high = int(ct_df[ct_df["high_lambda"]]["sig"].sum())
            sig_low = int(ct_df[~ct_df["high_lambda"]]["sig"].sum())
            if table.shape == (2, 2):
                _, fisher_p = fisher_exact(table.values)
            else:
                fisher_p = float("nan")
            print(f"\nContingency table (lambda threshold = {thresh}):")
            print(
                table.rename(
                    index={True: f"λ≥{thresh}", False: f"λ<{thresh}"},
                    columns={True: "sig (p<0.05)", False: "not sig"},
                )
            )
            print(f"Fisher's exact p = {fisher_p:.4f}")
            print("\nSummary:")
            print(
                f"Among participants with λ≥{thresh} (n={n_high}), {sig_high}/{n_high} showed "
                f"a significant positive relationship between weight neuron activity and response "
                f"change (Fisher's exact p={fisher_p:.3f}). In contrast, among participants with "
                f"λ<{thresh} (n={n_low}), only {sig_low}/{n_low} showed significant relationships."
            )

    if args.split_lambda:
        high = [p for p in pid_results if p["lambda_"] >= args.lambda_threshold]
        low = [p for p in pid_results if p["lambda_"] < args.lambda_threshold]
        print(
            f"\nλ >= {args.lambda_threshold} (n={len(high)}): "
            f"sig={sum(p['pval'] < 0.05 for p in high)}/{len(high)}"
        )
        print(
            f"λ <  {args.lambda_threshold} (n={len(low)}):  "
            f"sig={sum(p['pval'] < 0.05 for p in low)}/{len(low)}"
        )

    apply_style()
    PALETTE = get_palette()
    color_sig = PALETTE["Bayes"]
    color_nonsig = PALETTE["RL"]

    n_panels = 2 if args.split_lambda else 1
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(FIGURE_SIZE[0] * n_panels, FIGURE_SIZE[1]),
        constrained_layout=True,
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]
    ax = axes[0]

    _plot_panel(
        ax,
        pid_results,
        mean_activity,
        mean_delta,
        color_sig,
        color_nonsig,
        # title="Colored by p-value",
        # ylabel="Mean on-weight neuron activity (Hz)",
        ylabel="Mean activity of error neurons (Hz)",
    )

    if args.split_lambda:
        ax2 = axes[1]
        thresh = args.lambda_threshold
        for pid_data in pid_results:
            if np.isnan(pid_data["lambda_"]):
                continue
            c = color_sig if pid_data["lambda_"] >= thresh else color_nonsig
            pid_df = pd.DataFrame(
                {"delta": pid_data["delta"], "activity": pid_data["activity"]}
            )
            sns.regplot(
                data=pid_df,
                x="delta",
                y="activity",
                scatter=False,
                line_kws={"color": c, "linewidth": 0.8},
                ci=95,
                ax=ax2,
            )
        mean_df = pd.DataFrame({"activity": mean_activity, "delta": mean_delta})
        fin = np.isfinite(mean_df["activity"].values) & np.isfinite(
            mean_df["delta"].values
        )
        if fin.sum() >= 2:
            sns.regplot(
                data=mean_df,
                x="delta",
                y="activity",
                scatter=False,
                line_kws={"color": "black", "linewidth": 2.5},
                ci=95,
                ax=ax2,
            )
        ax2.set_xlabel(
            "Mean absolute response change per observation (|Δresponse|)"
        )
        ax2.set_ylabel("")
        ax2.set_title(f"Colored by λ threshold")
        n_high = sum(1 for p in pid_results if p["lambda_"] >= thresh)
        n_low = sum(1 for p in pid_results if p["lambda_"] < thresh)
        handles2 = [
            Line2D(
                [0],
                [0],
                color=color_sig,
                linewidth=1.5,
                label=f"λ ≥ {thresh} (n={n_high})",
            ),
            Line2D(
                [0],
                [0],
                color=color_nonsig,
                linewidth=1.5,
                label=f"λ < {thresh} (n={n_low})",
            ),
        ]
        ax2.legend(handles=handles2, frameon=False)
        sns.despine(ax=ax2, top=True, right=True)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "response_change_vs_weight_activity.png", dpi=300)
    plt.savefig(FIGURES_DIR / "response_change_vs_weight_activity.pdf")
    print("Saved figures/response_change_vs_weight_activity.{png,pdf}")


if __name__ == "__main__":
    main()
