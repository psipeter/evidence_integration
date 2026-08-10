#!/usr/bin/env python3
"""scripts/inspect_participant_temporal.py
============================================
figure_soltani_temporal.py's own 5-panel layout (performance error,
|delta response|, residual variance growth, residual autocorrelation,
split-half lambda reliability), scoped down to ONE real participant
pulled directly from Supabase -- not the population pkl pipeline
(build_model_inputs.py/figure_soltani_temporal.py itself still needs
that, no adapter exists yet, see CLAUDE.md/TODO.md).

This is NOT a mechanical re-run of figure_soltani_temporal.py against a
1-row slice of data -- three of its five panels are defined ACROSS PIDS
(cols 1/2 aggregate mean+/-SEM across pids; col 5 scatters one pid's
lambda against another's), which is undefined or degenerate at N=1.
Adaptations made here, each deliberate:

  Col 1 (RMSE) / Col 2 (|delta response|): the original collapses each
    pid down to ONE number per observation (mean over that pid's own
    trials) BEFORE aggregating mean+/-SEM across pids -- at N=1 that
    throws away all the real variability this participant's own 32
    trials contain. Adapted to aggregate mean+/-SEM ACROSS THIS
    PARTICIPANT'S TRIALS directly instead, which is still a real,
    informative quantity (how consistent is their error at this
    observation number, across their different trials) -- just a
    different axis of aggregation than the original panel.
  Col 3 (variance growth) / Col 4 (autocorrelation): NOT adapted --
    these were already defined within ONE participant's own qid
    repeats (residual = response - mean(response | this pid, this
    observation, this qid)), so they work as-is at N=1. Whether they
    show real data depends entirely on whether THIS TASK repeats qids
    at all for a single participant -- confirmed empirically before
    writing this script (numbers: yes, 8 qids x 4 repeats; colors: no,
    every qid appears exactly once, by design) -- so a colors
    participant will honestly show "insufficient data" here, which is
    a fact about colors' generation method, not a bug in this script.
  Col 5 (split-half lambda reliability): the original is fundamentally
    a cross-pid correlation (does pid A's first-half lambda predict
    their own second-half lambda, compared across MANY pids) --
    meaningless as a correlation at N=1. Adapted to show a single
    point (this participant's own first-half lambda vs. second-half
    lambda) with an explicit "n=1, no r defined" note, rather than
    hiding the panel or fabricating a correlation from one point.

Usage:
    python scripts/inspect_participant_temporal.py --prolific_pid f0079fb --task colors
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.paths import FIGURES_DIR
from utils.plot_style import apply_style, label_panels

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_participant import fetch_participant_events, rows_to_df, TASK_INTERNAL

HUMAN_COLOR = "0.25"
PREFIX_LENGTH = 4  # numbers only -- see module docstring on cols 3/4


def _to_pct(x, task_internal):
    return (x + 1.0) / 2.0 * 100.0 if task_internal == "binary" else (x + 1.0) * 50.0


def load_one_participant(prolific_pid: str, task: str) -> pd.DataFrame:
    """Same columns figure_soltani_temporal.py's _load_human returns:
    [trial, observation, qid, response, ground_truth], response/
    ground_truth on the 0-100 display scale -- but response/true_mean/
    true_p come straight off the DB's own 0-100/0-1 native scale (no
    -1..1 round-trip needed, unlike the pkl pipeline's carrabin/yoo-
    inherited [-1,1] convention)."""
    rows = fetch_participant_events(prolific_pid, task)
    df = rows_to_df(rows)
    if df.empty:
        return df
    out = df[["trial", "observation", "qid", "response"]].copy()
    if task == "colors":
        out["ground_truth"] = df["true_p"] * 100.0
    else:
        out["ground_truth"] = df["true_mean"]
    return out


# ── Cols 1/2 — mean +/- SEM ACROSS THIS PARTICIPANT'S TRIALS ───────────────

def _plot_across_trials(ax, df: pd.DataFrame, value_col: str, ylabel: str):
    stats = df.groupby("observation")[value_col].agg(["mean", "sem", "count"]).reset_index()
    ax.plot(stats["observation"], stats["mean"], "o-", color=HUMAN_COLOR, lw=1.8, ms=5)
    ax.fill_between(stats["observation"], stats["mean"] - stats["sem"].fillna(0),
                    stats["mean"] + stats["sem"].fillna(0), color=HUMAN_COLOR, alpha=0.2)
    ax.set_xlabel("Observation")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(stats["observation"]))
    ax.set_ylim(bottom=0)
    n_trials = int(stats["count"].max())
    ax.legend([Line2D([0], [0], color=HUMAN_COLOR, lw=1.5)],
              [f"This participant (n={n_trials} trials)"], fontsize=7, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def _panel_error(ax, df: pd.DataFrame):
    d = df.copy()
    d["abs_err"] = (d["response"] - d["ground_truth"]).abs()
    _plot_across_trials(ax, d, "abs_err", "|error| vs ground truth")


def _panel_delta(ax, df: pd.DataFrame):
    pieces = []
    for _, g in df.groupby("trial"):
        g = g.sort_values("observation").copy()
        g["delta"] = g["response"].diff().abs()
        pieces.append(g)
    d = pd.concat(pieces, ignore_index=True)
    d = d[d["observation"] >= 1].dropna(subset=["delta"])
    _plot_across_trials(ax, d, "delta", "Mean |\u0394response|")


# ── Cols 3/4 — residuals within this pid's OWN qid repeats ─────────────────
# Unmodified from figure_soltani_temporal.py's own logic -- already
# defined within one participant, see module docstring.

def _add_resid_prefix(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["observation"] < PREFIX_LENGTH]
    means = (sub.groupby(["observation", "qid"])["response"]
             .mean().reset_index().rename(columns={"response": "qid_mean"}))
    df2 = sub.merge(means, on=["observation", "qid"])
    df2["resid"] = df2["response"] - df2["qid_mean"]
    return df2


def _panel_variance_growth(ax, df: pd.DataFrame):
    df2 = _add_resid_prefix(df)
    grp = (df2.groupby(["observation", "qid"])["resid"]
           .apply(lambda x: x.std() if len(x) >= 2 else np.nan)
           .dropna().reset_index(name="std"))
    if grp.empty:
        ax.text(0.5, 0.5, "Insufficient data\n(no qid repeats for this task)",
                ha="center", va="center", transform=ax.transAxes, color="0.5", style="italic")
        return
    stats = grp.groupby("observation")["std"].mean().reset_index()
    ax.plot(stats["observation"], stats["std"], "o-", color=HUMAN_COLOR, lw=1.8, ms=5)
    ax.set_xlabel("Observation (prefix only)")
    ax.set_ylabel("Response variability across qid repeats")
    ax.set_xticks(range(PREFIX_LENGTH))
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax, top=True, right=True)


def _panel_autocorr(ax, df: pd.DataFrame):
    df2 = _add_resid_prefix(df)
    # A qid with only 1 repeat produces a trivially-zero residual (its
    # "mean" is just itself), not a genuine absence of data -- the same
    # degenerate case _panel_variance_growth already guards against via
    # its own `len(x) >= 2` check. Apply the identical guard here before
    # computing anything, rather than relying on df2 being empty (it
    # never is in this case -- it's full of meaningless zeros, which is
    # what produced scipy's "constant input" warning before this fix).
    repeat_counts = df2.groupby(["observation", "qid"]).size()
    if not (repeat_counts >= 2).any():
        ax.text(0.5, 0.5, "Insufficient data\n(no qid repeats for this task)",
                ha="center", va="center", transform=ax.transAxes, color="0.5", style="italic")
        return
    from scipy.stats import pearsonr
    lags = [1, 2, 3]
    rs = []
    for lag in lags:
        pairs = []
        for _, g in df2.groupby("trial"):
            r = g.sort_values("observation")["resid"].values
            if len(r) > lag:
                pairs.extend(zip(r[:-lag], r[lag:]))
        if len(pairs) < 3:
            rs.append(np.nan)
            continue
        arr = np.array(pairs)
        rv, _ = pearsonr(arr[:, 0], arr[:, 1])
        rs.append(rv)
    if all(np.isnan(rs)):
        ax.text(0.5, 0.5, "Insufficient data\n(no qid repeats for this task)",
                ha="center", va="center", transform=ax.transAxes, color="0.5", style="italic")
        return
    ax.plot(lags, rs, "o-", color=HUMAN_COLOR, lw=1.8, ms=5)
    ax.axhline(0, color="0.7", lw=0.8, ls="--")
    ax.set_xlabel("Lag (observations, within prefix)")
    ax.set_ylabel("Autocorrelation of deviations")
    ax.set_xticks(lags)
    sns.despine(ax=ax, top=True, right=True)


# ── Col 5 — this pid's own first-half vs. second-half lambda (single point)

def _fit_lambda(df: pd.DataFrame) -> float | None:
    pieces = []
    for _, g in df.groupby("trial"):
        g = g.sort_values("observation").copy()
        g["delta"] = g["response"].diff().abs()
        pieces.append(g)
    d = pd.concat(pieces, ignore_index=True)
    curve = d.groupby("observation")["delta"].mean().dropna()
    curve = curve[curve.index >= 1]
    if len(curve) < 3:
        return None
    n, y = curve.index.values.astype(float), curve.values.astype(float)
    try:
        popt, _ = scipy_curve_fit(lambda n, A, lam: A * n ** (-lam), n, y,
                                  p0=[0.1, 0.5], bounds=([0, 0], [2, 2]), maxfev=2000)
        return float(popt[1])
    except Exception:
        return None


def _panel_splithalf_point(ax, df: pd.DataFrame):
    trials = sorted(df["trial"].unique())
    mid = len(trials) // 2
    if mid < 3:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    lam_first = _fit_lambda(df[df["trial"].isin(trials[:mid])])
    lam_second = _fit_lambda(df[df["trial"].isin(trials[mid:])])
    if lam_first is None or lam_second is None:
        ax.text(0.5, 0.5, "Fit failed", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    lims = (0, max(1.0, lam_first, lam_second) * 1.15)
    ax.plot(lims, lims, color="0.7", lw=0.8, ls="--")
    ax.scatter([lam_first], [lam_second], s=60, color=HUMAN_COLOR, zorder=3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("\u03bb (first half of trials)")
    ax.set_ylabel("\u03bb (second half of trials)")
    ax.text(0.02, 0.98, "n=1 participant\nno r defined for a single point",
            ha="left", va="top", transform=ax.transAxes, fontsize=7, style="italic", color="0.5")
    sns.despine(ax=ax, top=True, right=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prolific_pid", required=True)
    p.add_argument("--task", required=True, choices=["numbers", "colors"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    print(f"Fetching {args.prolific_pid} / {args.task} from Supabase...")
    df = load_one_participant(args.prolific_pid, args.task)
    if df.empty:
        print(f"No trial-phase rows found for ({args.prolific_pid}, {args.task}).")
        return
    print(f"{len(df)} distinct (trial, observation) rows across {df['trial'].nunique()} trials")

    apply_style()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)

    _panel_error(axes[0], df)
    _panel_delta(axes[1], df)
    _panel_variance_growth(axes[2], df)
    _panel_autocorr(axes[3], df)
    _panel_splithalf_point(axes[4], df)
    axes[0].set_title(f"{args.prolific_pid} -- {args.task}", loc="left", fontsize=9, style="italic")
    label_panels(axes.reshape(1, -1))

    fig.text(0.5, -0.05,
             "Adapted from figure_soltani_temporal.py's own 5-panel layout for ONE real "
             "participant -- cols 1-2 aggregate across this participant's own trials "
             "(not across pids, undefined at N=1); col 5 shows a single first-half-vs-"
             "second-half point (no correlation defined at N=1); cols 3-4 unmodified "
             "(already within-participant) -- see this script's own module docstring.",
             ha="center", va="top", fontsize=7, style="italic", color="0.4")

    out = Path(args.out) if args.out else FIGURES_DIR / f"inspect_participant_temporal_{args.prolific_pid}_{args.task}.pdf"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    print("JOB_COMPLETE")


if __name__ == "__main__":
    main()
