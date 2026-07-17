#!/usr/bin/env python3
"""figure_soltani_temporal.py — T group figure for the soltani task/ pilot
(task-continuous + task-binary, Prolific pilot 1).

Layout: 2x5
  Row 1 = task-binary, Row 2 = task-continuous (standing row-order
  convention for soltani figures — see figure_soltani_performance.py)
  Col 1 (~carrabin temporal panel A / T1): Performance error (RMSE to
    ground truth) vs observation. Human: mean +/- SEM across pids, with a
    flag to overlay each pid as a thin grey line (no CI/band on those).
    Mean model: bootstrapped mean +/- CI pooled directly across every
    sequence a participant actually saw (see _plot_panel_performance's
    inline comment for why this differs from the Human aggregation).
  Col 2 (~yoo temporal panel B / T2): Mean |Delta response| vs observation
    (obs >= 2, matching yoo's own threshold), via sns.lineplot(errorbar=
    "ci"). Same individual-pid overlay flag as col 1, and the same Mean
    model line (computed on real stimuli, pooled across all sequences).
  Col 3 (~carrabin temporal's RENDERED panel C / T3): Residual variance
    growth — std(resid | obs, qid) vs observation.
  Col 4 (~carrabin temporal's RENDERED panel D / T4): Within-trial lag-k
    residual autocorrelation (lag 1-3).
  Col 5 (~yoo temporal panel C / T3): Split-half reliability of the
    decay-rate lambda fitted to |Delta response| vs observation, but with
    scatter=True (raw per-pid points shown, not just the fit line).

NOTE ON CARRABIN'S "PANEL C"/"PANEL D" LABELS
------------------------------------------------
figure_carrabin_temporal.py's own docstring calls its autocorrelation panel
"C (T4)" and its variance-growth panel "D (T3)", but its main() actually
plots them in the order [A, B, D(variance growth), C(autocorrelation)], so
the RENDERED, lettered panel C is variance growth and rendered panel D is
autocorrelation. Columns 3/4 here follow the rendered lettering (i.e. what
you'd see if you opened figure_carrabin_temporal.pdf), not the internal
function names.

WHY COLS 3/4 ARE RESTRICTED TO THE PREFIX REGION (observation < 4)
------------------------------------------------------------------------
Same reason as figure_soltani_variability.py: this task's qid repeats are
only identical over the first `prefix_length` (=4) observations; the
suffix differs by design on every repeat (steered toward different
targets). carrabin's residual-vs-qid-mean approach assumes the whole
trial is identical across a qid's repeats, which only holds here within
the prefix. Applying it to the full trial would fold real signal
differences (not noise) into "residuals" for observation >= 4, so cols 3
and 4 are computed only over observation < prefix_length. Cols 1, 2, and 5
don't rely on qid-repeated identical inputs (they use ground truth,
trial-to-trial response change, and whole-trial decay dynamics
respectively) and so are NOT prefix-restricted.

PLACEHOLDER NOTE
----------------
Human only — no models have been fit to this pilot yet. Once fits exist,
add model overlays to every panel, mirroring figure_carrabin_temporal.py /
figure_yoo_temporal.py.

Run:
    python scripts/figure_soltani_temporal.py
    python scripts/figure_soltani_temporal.py --results_file task_results_pilot1.pkl
    python scripts/figure_soltani_temporal.py --hide_individual
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
from scipy.optimize import curve_fit as scipy_curve_fit
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, data_path
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette, label_panels
from utils.binary_transform import apply_binary_transform
from utils.participant_filters import filter_participants

TASK_ROWS     = ["binary", "continuous"]  # standing row-order convention
PREFIX_LENGTH = 4
HUMAN_COLOR   = "0.3"
INDIV_COLOR   = "0.7"
MIN_CORR_N    = 3  # matches the threshold used in figure_soltani_variability.py


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Successful attempts only, one row per (pid, trial, observation)."""
    return (df[df["timed_out"] == False]
            .drop_duplicates(subset=["prolific_pid", "trial", "observation"]))


def _ground_truth(sub: pd.DataFrame, task: str) -> pd.Series:
    return sub["true_p"] * 100 if task == "binary" else sub["true_mean"]


def _mean_model_response(df_task: pd.DataFrame, task: str) -> pd.DataFrame:
    """Deterministic Mean-model response (expanding mean of the observed
    `value` stream, with the same Laplace-smoothing binary_transform used
    elsewhere for task_binary) computed on every (prolific_pid, trial,
    observation) sequence actually shown in this pilot — i.e. real stimuli,
    not resimulated. One row per sequence step; columns
    [prolific_pid, trial, observation, model_response]."""
    sub = (df_task[["prolific_pid", "trial", "observation", "value"]]
           .drop_duplicates(subset=["prolific_pid", "trial", "observation"])
           .sort_values(["prolific_pid", "trial", "observation"]))
    sub["model_mean"] = (sub.groupby(["prolific_pid", "trial"])["value"]
                          .expanding().mean().values)
    if task == "binary":
        smoothed = apply_binary_transform(
            sub[["observation"]].assign(response=sub["model_mean"]), "task_binary")
        sub["model_response"] = (smoothed["response"].to_numpy() + 1) / 2 * 100
    else:
        sub["model_response"] = sub["model_mean"]
    return sub[["prolific_pid", "trial", "observation", "model_response"]]


# ── Col 1 — Performance error vs observation ────────────────────────────────

def _rmse_per_pid_obs(df_task: pd.DataFrame, task: str) -> pd.DataFrame:
    sub = _dedup(df_task).assign(ground_truth=lambda d: _ground_truth(d, task))
    return (sub.assign(sq_err=(sub["response"] - sub["ground_truth"]) ** 2)
            .groupby(["prolific_pid", "observation"])["sq_err"].mean()
            .apply(np.sqrt).reset_index(name="rmse"))


def _plot_panel_performance(ax, df_task: pd.DataFrame, task: str,
                            show_individual: bool, mean_color: str) -> None:
    rmse_df = _rmse_per_pid_obs(df_task, task)
    stats = rmse_df.groupby("observation")["rmse"].agg(["mean", "sem"]).reset_index()

    handles = [Line2D([0], [0], color=HUMAN_COLOR, lw=1.8)]
    labels = ["Human"]

    if show_individual:
        for pid, g in rmse_df.groupby("prolific_pid"):
            g = g.sort_values("observation")
            ax.plot(g["observation"], g["rmse"], color=INDIV_COLOR,
                    lw=0.6, alpha=0.5, zorder=1)
        handles.append(Line2D([0], [0], color=INDIV_COLOR, lw=0.8))
        labels.append("Individual pids")

    ax.plot(stats["observation"], stats["mean"], "o-", color=HUMAN_COLOR,
            lw=1.8, ms=5, zorder=3)
    ax.fill_between(stats["observation"], stats["mean"] - stats["sem"],
                    stats["mean"] + stats["sem"], color=HUMAN_COLOR, alpha=0.2, zorder=2)

    # Mean model: computed on the same real stimulus sequences participants
    # actually saw (not resimulated). Pooled directly across every
    # (prolific_pid, trial) sequence — NOT pre-averaged per pid first, unlike
    # the Human line above — via a bootstrapped RMSE estimator, since the
    # model has no meaningful "individual differences" of its own; its only
    # source of across-sequence spread is which real stimuli each sequence
    # happened to contain.
    mean_df = _mean_model_response(df_task, task)
    gt_df = (_dedup(df_task)
             .assign(ground_truth=lambda d: _ground_truth(d, task))
             [["prolific_pid", "trial", "observation", "ground_truth"]])
    mean_merged = mean_df.merge(gt_df, on=["prolific_pid", "trial", "observation"])
    mean_merged["sq_err"] = (mean_merged["model_response"] - mean_merged["ground_truth"]) ** 2

    sns.lineplot(data=mean_merged, x="observation", y="sq_err",
                estimator=lambda a: float(np.sqrt(np.mean(a))),
                errorbar="ci", color=mean_color, lw=1.8, ax=ax,
                label="_nolegend_", zorder=4)
    handles.append(Line2D([0], [0], color=mean_color, lw=1.8))
    labels.append("Mean")

    ax.set_xlabel("Observation")
    ax.set_ylabel("Performance error vs ground truth (RMSE)")
    ax.set_xticks(sorted(df_task["observation"].unique()))
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Col 2 — Mean |Delta response| vs observation ────────────────────────────

def _abs_delta_long(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for (_, _), g in df.groupby(["prolific_pid", "trial"], sort=False):
        g = g.sort_values("observation").copy()
        g["delta"] = g["response"].diff().abs()
        pieces.append(g)
    if not pieces:
        return pd.DataFrame(columns=["prolific_pid", "trial", "observation", "delta"])
    out = pd.concat(pieces, ignore_index=True)
    return out[out["observation"] >= 2].dropna(subset=["delta"])  # matches yoo's own threshold


def _plot_panel_delta(ax, df_task: pd.DataFrame, task: str,
                      show_individual: bool, mean_color: str) -> None:
    delta_df = _abs_delta_long(_dedup(df_task))

    sns.lineplot(data=delta_df, x="observation", y="delta", color=HUMAN_COLOR,
                lw=1.8, errorbar="ci", ax=ax, label="_nolegend_", zorder=3)

    handles = [Line2D([0], [0], color=HUMAN_COLOR, lw=1.8, alpha=0.65)]
    labels = ["Human"]

    if show_individual:
        indiv = (delta_df.groupby(["prolific_pid", "observation"])["delta"]
                 .mean().reset_index())
        for pid, g in indiv.groupby("prolific_pid"):
            g = g.sort_values("observation")
            ax.plot(g["observation"], g["delta"], color=INDIV_COLOR,
                    lw=0.6, alpha=0.5, zorder=1)
        handles.append(Line2D([0], [0], color=INDIV_COLOR, lw=0.8))
        labels.append("Individual pids")

    # Mean model: same real stimulus sequences, same |delta| definition,
    # pooled directly across all sequences (see _plot_panel_performance's
    # comment for why this isn't pre-averaged per pid).
    mean_resp = _mean_model_response(df_task, task).rename(
        columns={"model_response": "response"})
    mean_delta_df = _abs_delta_long(mean_resp)
    sns.lineplot(data=mean_delta_df, x="observation", y="delta", color=mean_color,
                lw=1.8, errorbar="ci", ax=ax, label="_nolegend_", zorder=4)
    handles.append(Line2D([0], [0], color=mean_color, lw=1.8))
    labels.append("Mean")

    ax.set_xlabel("Observation")
    ax.set_ylabel("Mean |\u0394response|")
    ax.set_xticks(sorted(df_task["observation"].unique()))
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Cols 3/4 shared helper — residuals within the prefix region only ───────

def _add_resid_prefix(df_task: pd.DataFrame) -> pd.DataFrame:
    sub = _dedup(df_task)
    sub = sub[sub["observation"] < PREFIX_LENGTH]
    means = (sub.groupby(["prolific_pid", "observation", "qid"])["response"]
             .mean().reset_index().rename(columns={"response": "qid_mean"}))
    df2 = sub.merge(means, on=["prolific_pid", "observation", "qid"])
    df2["resid"] = df2["response"] - df2["qid_mean"]
    return df2


# ── Col 3 — Residual variance growth (prefix only) ──────────────────────────

def _plot_panel_variance_growth(ax, df_task: pd.DataFrame) -> None:
    df2 = _add_resid_prefix(df_task)
    MIN = 2
    grp = (df2.groupby(["prolific_pid", "observation", "qid"])["resid"]
           .apply(lambda x: x.std() if len(x) >= MIN else np.nan)
           .dropna().reset_index(name="std"))
    if grp.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    by_pid_obs = grp.groupby(["prolific_pid", "observation"])["std"].mean().reset_index()
    stats = by_pid_obs.groupby("observation")["std"].agg(["mean", "std"]).reset_index()
    n_pid = by_pid_obs["prolific_pid"].nunique()
    stats["se"] = stats["std"] / np.sqrt(n_pid)

    ax.plot(stats["observation"], stats["mean"], "o-", color=HUMAN_COLOR, lw=1.8, ms=5)
    ax.fill_between(stats["observation"], stats["mean"] - stats["se"],
                    stats["mean"] + stats["se"], color=HUMAN_COLOR, alpha=0.25)

    ax.set_xlabel("Observation (prefix only)")
    ax.set_ylabel("Response variability")
    ax.set_xticks(range(PREFIX_LENGTH))
    ax.set_ylim(bottom=0)
    ax.legend([Line2D([0], [0], color=HUMAN_COLOR, lw=1.5)], ["Human"],
              fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Col 4 — Within-trial residual autocorrelation (prefix only) ────────────

def _plot_panel_autocorr(ax, df_task: pd.DataFrame) -> None:
    df2 = _add_resid_prefix(df_task)
    lags = [1, 2, 3]
    pid_rs: dict[int, list[float]] = {lag: [] for lag in lags}

    for _, pid_df in df2.groupby("prolific_pid"):
        for lag in lags:
            pairs = []
            for (_, _), g in pid_df.groupby(["prolific_pid", "trial"]):
                r = g.sort_values("observation")["resid"].values
                if len(r) > lag:
                    pairs.extend(zip(r[:-lag], r[lag:]))
            if len(pairs) < 3:
                continue
            arr = np.array(pairs)
            rv, _ = pearsonr(arr[:, 0], arr[:, 1])
            pid_rs[lag].append(rv)

    if all(len(v) == 0 for v in pid_rs.values()):
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    means = np.array([np.mean(pid_rs[lag]) if pid_rs[lag] else np.nan for lag in lags])
    sems = np.array([np.std(pid_rs[lag]) / np.sqrt(len(pid_rs[lag]))
                    if len(pid_rs[lag]) > 1 else np.nan for lag in lags])

    ax.plot(lags, means, "o-", color=HUMAN_COLOR, lw=1.8, ms=5)
    ax.fill_between(lags, means - sems, means + sems, color=HUMAN_COLOR, alpha=0.2)
    ax.axhline(0, color="0.7", lw=0.8, ls="--")

    ax.set_xlabel("Lag (observations, within prefix)")
    ax.set_ylabel("Autocorrelation of trial-to-trial deviations")
    ax.set_xticks(lags)
    ax.legend([Line2D([0], [0], color=HUMAN_COLOR, lw=1.5)], ["Human"],
              fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Col 5 — Split-half reliability of lambda (scatter=True) ────────────────

def _fit_lambda_curve_fit(df: pd.DataFrame) -> pd.Series:
    def power_law(n, A, lam):
        return A * np.power(np.asarray(n, dtype=float), -lam)
    out: dict = {}
    for pid, grp in df.groupby("prolific_pid"):
        pieces = []
        for _, tg in grp.groupby("trial"):
            g = tg.sort_values("observation").copy()
            g["delta"] = g["response"].diff().abs()
            pieces.append(g)
        delta = pd.concat(pieces, ignore_index=True)
        curve = delta.groupby("observation")["delta"].mean().dropna()
        curve = curve[curve.index >= 2]
        if len(curve) < 3:
            continue
        n = curve.index.values.astype(float)
        y = curve.values.astype(float)
        if not (np.all(np.isfinite(n)) and np.all(np.isfinite(y))):
            continue
        try:
            popt, _ = scipy_curve_fit(power_law, n, y, p0=[0.1, 0.5],
                                      bounds=([0.0, 0.0], [2.0, 2.0]), maxfev=2000)
            out[pid] = float(popt[1])
        except Exception:
            pass
    return pd.Series(out, name="lambda_")


def _fit_lambda_split_half(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, grp in df.groupby("prolific_pid"):
        trials = sorted(grp["trial"].unique())
        mid = len(trials) // 2
        if mid < 3:
            continue
        for half_label, trial_set in [("first", trials[:mid]), ("second", trials[mid:])]:
            sub = grp[grp["trial"].isin(trial_set)].copy()
            lam = _fit_lambda_curve_fit(sub.assign(prolific_pid=pid))
            if pid in lam.index:
                rows.append({"prolific_pid": pid, "half": half_label,
                             "lambda_": float(lam[pid])})
    if not rows:
        return pd.DataFrame(columns=["prolific_pid", "first", "second"])
    wide = (pd.DataFrame(rows)
            .pivot(index="prolific_pid", columns="half", values="lambda_")
            .dropna())
    wide.columns.name = None
    return wide.reset_index()


def _plot_panel_splithalf_lambda(ax, df_task: pd.DataFrame) -> None:
    wide = _fit_lambda_split_half(_dedup(df_task))

    if len(wide) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    sns.regplot(data=wide, x="first", y="second", ax=ax, color=HUMAN_COLOR,
               ci=95 if len(wide) >= MIN_CORR_N else None,
               scatter=True, line_kws={"lw": 1.5},
               scatter_kws={"s": 20, "alpha": 0.7})

    if len(wide) >= MIN_CORR_N:
        r, p = pearsonr(wide["first"], wide["second"])
        from utils.plot_style import pvalue_to_stars
        ax.legend(handles=[Line2D([0], [0], color=HUMAN_COLOR, lw=1.5)],
                  labels=[f"Human r={r:.2f}{pvalue_to_stars(p)}"],
                  fontsize=8, frameon=True, framealpha=0.9)
    else:
        ax.text(0.02, 0.98, f"n={len(wide)} (too few for r)",
                ha="left", va="top", transform=ax.transAxes,
                fontsize=7, style="italic", color="0.5")

    ax.set_xlabel("\u03bb (first half of trials)")
    ax.set_ylabel("\u03bb (second half of trials)")
    sns.despine(ax=ax, top=True, right=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="task_results_pilot1.pkl",
                        help="Filename under data/ produced by task/parse_results.py")
    parser.add_argument("--show_individual", dest="show_individual",
                        action="store_true", default=True,
                        help="Overlay each pid as a thin grey line in cols 1-2 (default on)")
    parser.add_argument("--hide_individual", dest="show_individual",
                        action="store_false")
    parser.add_argument("--skip_filters", action="store_true",
                        help="Skip utils/participant_filters exclusion (default: applied)")
    args = parser.parse_args()

    df = pd.read_pickle(data_path(args.results_file))
    if not args.skip_filters:
        df = filter_participants(df, verbose=True)
    apply_style()
    pal = get_palette(2)
    mean_color = pal[0]

    fig, axes = plt.subplots(
        2, 5,
        figsize=(FIGURE_SIZE[0] * 1.25, FIGURE_SIZE[1]),
        constrained_layout=True,
    )

    for row, task in enumerate(TASK_ROWS):
        sub = df[df["task"] == task]
        _plot_panel_performance(axes[row, 0], sub, task, args.show_individual, mean_color)
        _plot_panel_delta(axes[row, 1], sub, task, args.show_individual, mean_color)
        _plot_panel_variance_growth(axes[row, 2], sub)
        _plot_panel_autocorr(axes[row, 3], sub)
        _plot_panel_splithalf_lambda(axes[row, 4], sub)
        axes[row, 0].set_title(f"task-{task}", loc="left", fontsize=9, style="italic")

    label_panels(axes)

    fig.text(0.5, -0.02,
              "PLACEHOLDER: human only (no models fit yet). Cols 3-4 restricted to "
              "observation < prefix_length=4, the only region guaranteed identical "
              "across a qid's repeats in this task's design (unlike carrabin).",
              ha="center", va="top", fontsize=7, style="italic", color="0.4")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure_soltani_temporal"
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.pdf")


if __name__ == "__main__":
    main()
