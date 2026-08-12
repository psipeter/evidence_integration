#!/usr/bin/env python3
"""figure_soltani_temporal.py — T group figure for the soltani task/ pilot
(task-numbers + task-colors).

Layout: 2x6
  Row 1 = task-colors, Row 2 = task-numbers (standing row-order
  convention for soltani figures — see figure_soltani_performance.py)
  Col 1 (~carrabin temporal panel A / T1): Performance error (RMSE to
    the RUNNING MEAN of the observed stimulus stream, NOT the fixed
    generative true_mean/true_p -- see _add_running_mean_ground_truth's
    own docstring for why) vs observation. Human always shown; pass
    --plot_models to add all 4 fitted models (Mean, LeakyIntegrator, PrimacyRecency,
    RL_lambda): mean +/- SEM across pids (per-pid RMSE computed first,
    collapsing over that pid's own trials, then mean/SEM across pids) --
    SAME hierarchy for every line, so they're directly comparable. Only
    Human gets the individual-pid thin-line overlay
    (--show_individual/--hide_individual); models show mean/CI only.
  Col 2 (~yoo temporal panel B / T2): Mean |Delta response| vs observation
    (obs >= 1 -- this task's `observation` is 0-indexed, unlike yoo's own
    1-indexed column; see _abs_delta_long's inline comment). Same
    hierarchy and same Human-only individual-pid overlay as col 1; same
    --plot_models gate for the 4 fitted models.
  Col 3 (~carrabin temporal's RENDERED panel C / T3): Residual variance
    growth -- std(resid | obs, qid) vs observation. Human only, permanently
    (see below).
  Col 4 (~carrabin temporal's RENDERED panel D / T4): Within-trial lag-k
    residual autocorrelation (lag 1-3). Human only, permanently (see below).
  Col 5 (~yoo temporal panel C / T3): Split-half reliability (ODD/EVEN
    trials) of the decay-rate lambda fitted to |Delta response| vs
    observation, with scatter=True. Human AND each fitted model, one
    regplot per source -- matching figure_yoo_temporal.py's own T3.
  Col 6 (~yoo temporal panel D / T4): lambda_model vs lambda_human per pid,
    one regplot per model plus an identity line -- does a model reproduce
    each INDIVIDUAL's discounting rate, not just the group average? Both
    axes use the same model-free estimator. Mean is retained as a null
    control (no free parameters, so it cannot track individual differences);
    see _plot_panel_lambda_model_vs_human's own docstring, which also notes
    where this deliberately diverges from yoo's T4.

COLS 3-4 ARE PERMANENTLY HUMAN-ONLY (until a STOCHASTIC model exists)
-----------------------------------------------------------------------
This is a settled decision backed by a direct check, not an unfinished
follow-up. Both panels are built on residuals against a qid-conditional mean:
resid = response - mean(response | pid, observation, qid). Every model
currently fit here (Mean, LeakyIntegrator, PrimacyRecency, RL_lambda) is
DETERMINISTIC, and a qid's repeats share an identical prefix by design, so a
deterministic model returns the identical response on every repeat and its
residual is EXACTLY ZERO. Verified against real pilot 5 data: max|resid| was
0.000e+00 for all four models over 1152 qualifying prefix rows, versus 0.68 for
Human. Adding them would draw four flat lines at zero.

These two metrics (carrabin's T5/T6) exist precisely to measure
state-persistent response variability, which is a property only a noisy
generative process has. They become meaningful for NEF (spiking noise) or
NoisyCounting, and should be extended when NEF is fit for these datasets --
not before. Col 5 does NOT have this problem and DOES include models; see
_plot_panel_splithalf_lambda's own docstring for why.

Cols 3-4 DO now use quasi-qids for colors (task-colors)'s human data --
colors' own literal `qid` column never repeats, so a DIFFERENT repeat
structure is empirically derived instead: see utils/colors_quasi_qids.py's
own module docstring for the full definition and the empirical sweep
that settled its PREFIX_LENGTH=4/MIN_REPEATS=3 defaults. Numbers
(task-numbers) uses its real, designed qid repeats unchanged. Col 5
doesn't use qid at all (it's a lambda power-law fit on the |delta
response| curve alone), so it's unaffected either way.

DATA SOURCE
-----------
Both human and model data come from data/soltani_numbers.pkl / data/
soltani_colors.pkl and data/runs/{run_folder}/{model_type}_{dataset}_
responses.pkl -- NOT from a raw task_results_pilot*.pkl. Participant
filtering and the prolific_pid -> int pid mapping already happened when
those files were built (scripts/build_model_inputs.py), and model
responses were fit directly against them, so this script does no
filtering itself and merges everything on integer `pid`. Both are stored
on the canonical [-1,1] scale carrabin/yoo use; converted back to [0,100]
here purely for readability (see _to_pct).

NOTE ON CARRABIN'S "PANEL C"/"PANEL D" LABELS
------------------------------------------------
figure_carrabin_temporal.py's own docstring calls its autocorrelation panel
"C (T4)" and its variance-growth panel "D (T3)", but its main() actually
plots them in the order [A, B, D(variance growth), C(autocorrelation)], so
the RENDERED, lettered panel C is variance growth and rendered panel D is
autocorrelation. Columns 3/4 here follow the rendered lettering.

WHY COLS 3/4 ARE RESTRICTED TO THE PREFIX REGION (observation < 4)
------------------------------------------------------------------------
This task's qid repeats are only identical over the first `prefix_length`
(=4) observations; the suffix differs by design on every repeat (steered
toward different targets). carrabin's residual-vs-qid-mean approach
assumes the whole trial is identical across a qid's repeats, which only
holds here within the prefix.

Run:
    python scripts/figure_soltani_temporal.py
    python scripts/figure_soltani_temporal.py --plot_models
    python scripts/figure_soltani_temporal.py --plot_models --datafile pilot5
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
from scipy.stats import linregress, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, data_path, dataset_stem, resolve_run_folder
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette, label_panels, pvalue_to_stars
from utils.colors_quasi_qids import add_quasi_qids

TASK_ROWS        = ["colors", "numbers"]  # standing row-order convention
DATASET_FOR_TASK = {"colors": "soltani_colors", "numbers": "soltani_numbers"}
MODEL_ORDER       = ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda"]
PREFIX_LENGTH     = 4
HUMAN_COLOR       = "0.3"
INDIV_COLOR       = "0.7"
MIN_CORR_N        = 3  # matches the threshold used in figure_soltani_variability.py


def _to_pct(x: pd.Series, task: str) -> pd.Series:
    if task == "colors":
        return (x + 1.0) / 2.0 * 100.0
    return (x + 1.0) * 50.0


def _load_human(task: str, datafile: str | None = None) -> pd.DataFrame | None:
    """Human data for one task, on the [0,100] percent scale. Columns:
    [pid, trial, observation, qid, value, response, ground_truth].
    ground_truth is the RUNNING mean of `value` (see
    _add_running_mean_ground_truth below), NOT the fixed true_mean/
    true_p -- see that function's own docstring for why.

    datafile: optional suffix (e.g. 'pilot4', 'pilot5') appended to the
    dataset stem -- see figure_soltani_performance.py's own _load_human
    docstring for the full rationale (this file's own convention, kept
    consistent rather than reinvented). Returns None if that task has no
    file at all for this datafile (e.g. a numbers-only pilot has no
    colors file yet) -- caller's job to handle gracefully, not this
    function's."""
    dataset = DATASET_FOR_TASK[task]
    if datafile:
        dataset = f"{dataset}_{datafile}"
    path = data_path(f"{dataset}.pkl")
    if not path.exists():
        return None
    df = pd.read_pickle(path)
    out = df[["pid", "trial", "observation", "qid", "value"]].copy()
    out["response"] = _to_pct(df["response"], task)
    out = _add_running_mean_ground_truth(out, task)
    return out


def _add_running_mean_ground_truth(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Ground truth = the RUNNING mean/ratio of the observed stimulus
    stream itself, per (pid, trial) -- i.e. what a perfect 'just average
    what you've seen so far' agent would report at each observation --
    NOT the fixed generative true_mean/true_p. Matches the same
    running_mean/running_p convention already established elsewhere in
    this project (scripts/plot_sequences.py's own gt_mode='running_mean';
    task_backend's own live 'correct answer' panel shows real
    participants exactly this quantity during the actual task, never the
    fixed target). Requires `value` (raw stimulus, native pkl scale --
    NOT yet through _to_pct) already present in df."""
    df = df.sort_values(["pid", "trial", "observation"]).copy()
    running = df.groupby(["pid", "trial"])["value"].transform(lambda s: s.expanding().mean())
    df["ground_truth"] = _to_pct(running, task)
    return df


def _load_model(task: str, model_type: str, run_dir: Path,
                datafile: str | None = None) -> pd.DataFrame | None:
    """Fitted model responses for one (task, model_type), on the [0,100]
    percent scale. Returns None if not yet fit/collected. Columns:
    [pid, trial, observation, response].

    `datafile` MUST be the same suffix _load_human was given: fits are named
    after the dataset STEM (family + data-version suffix, see
    utils.paths.dataset_stem), so passing it is what guarantees the model
    responses were actually fit against the human data plotted beside them.
    Omitting it here was a real defect: human data came from
    data/{dataset}_{datafile}.pkl while models came from the unsuffixed
    {model}_{dataset}_responses.pkl, and the two were then merged on `pid`
    even when they described different participants entirely."""
    dataset = dataset_stem(DATASET_FOR_TASK[task], datafile)
    resp_path = run_dir / f"{model_type}_{dataset}_responses.pkl"
    if not resp_path.exists():
        print(f"  (missing {resp_path.name} -- skipping {model_type} for {task})")
        return None
    df = pd.read_pickle(resp_path)
    out = df[["pid", "trial", "observation"]].copy()
    out["response"] = _to_pct(df["response"], task)
    return out


# ── Col 1 — Performance error vs observation ────────────────────────────────

def _rmse_per_pid_obs(df: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """df: [pid,trial,observation,response]; ground_truth: adds
    [pid,trial,observation,ground_truth]. Returns per-(pid,observation) RMSE."""
    merged = df.merge(ground_truth[["pid", "trial", "observation", "ground_truth"]],
                      on=["pid", "trial", "observation"])
    return (merged.assign(sq_err=(merged["response"] - merged["ground_truth"]) ** 2)
            .groupby(["pid", "observation"])["sq_err"].mean()
            .apply(np.sqrt).reset_index(name="rmse"))


def _plot_hierarchical_line(ax, per_pid_df: pd.DataFrame, value_col: str,
                            color: str, zorder_line: float, zorder_fill: float) -> None:
    """Shared plotting for the mean+/-SEM-across-pids line: per_pid_df must
    already be one row per (pid, observation) -- i.e. already collapsed
    over that pid's own trials -- for both Human and every model, so all
    lines in a panel use the identical hierarchy."""
    stats = per_pid_df.groupby("observation")[value_col].agg(["mean", "sem"]).reset_index()
    ax.plot(stats["observation"], stats["mean"], "o-", color=color,
            lw=1.8, ms=5, zorder=zorder_line)
    ax.fill_between(stats["observation"], stats["mean"] - stats["sem"],
                    stats["mean"] + stats["sem"], color=color, alpha=0.2, zorder=zorder_fill)


def _plot_panel_performance(ax, human: pd.DataFrame, models: dict[str, pd.DataFrame],
                            show_individual: bool, palette: dict) -> None:
    rmse_df = _rmse_per_pid_obs(human[["pid", "trial", "observation", "response"]], human)

    handles = [Line2D([0], [0], color=HUMAN_COLOR, lw=1.8)]
    labels = ["Human"]

    if show_individual:
        for pid, g in rmse_df.groupby("pid"):
            g = g.sort_values("observation")
            ax.plot(g["observation"], g["rmse"], color=INDIV_COLOR,
                    lw=0.6, alpha=0.5, zorder=2)
        handles.append(Line2D([0], [0], color=INDIV_COLOR, lw=0.8))
        labels.append("Individual pids")

    _plot_hierarchical_line(ax, rmse_df, "rmse", HUMAN_COLOR, zorder_line=3, zorder_fill=1)

    for i, (model_type, mdf) in enumerate(models.items()):
        model_rmse_df = _rmse_per_pid_obs(mdf, human)
        color = palette[model_type]
        _plot_hierarchical_line(ax, model_rmse_df, "rmse", color,
                                zorder_line=4 + i, zorder_fill=1)
        handles.append(Line2D([0], [0], color=color, lw=1.8))
        labels.append(model_type)

    obs_ticks = sorted(set(human["observation"]) | {o for m in models.values()
                                                    for o in m["observation"]})
    ax.set_xlabel("Observation")
    ax.set_ylabel("Performance error vs running mean (RMSE)")
    ax.set_xticks(obs_ticks)
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=7, frameon=True, framealpha=0.9, ncol=1)
    sns.despine(ax=ax, top=True, right=True)


# ── Col 2 — Mean |Delta response| vs observation ────────────────────────────

def _abs_delta_long(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for (_, _), g in df.groupby(["pid", "trial"], sort=False):
        g = g.sort_values("observation").copy()
        g["delta"] = g["response"].diff().abs()
        pieces.append(g)
    if not pieces:
        return pd.DataFrame(columns=["pid", "trial", "observation", "delta"])
    out = pd.concat(pieces, ignore_index=True)
    # First defined delta is at observation=1 (response[1]-response[0]),
    # since this task's `observation` is 0-indexed. NOT >=2 -- that's only
    # correct for yoo's own 1-indexed `observation` column, where the first
    # defined delta lands at observation=2.
    return out[out["observation"] >= 1].dropna(subset=["delta"])


def _plot_panel_delta(ax, human: pd.DataFrame, models: dict[str, pd.DataFrame],
                      show_individual: bool, palette: dict) -> None:
    # Per-pid mean |delta| first (pooling over that pid's own trials) --
    # this is both what the thin individual-pid lines plot directly AND
    # what every bold line's mean/SEM is computed from, so thin lines and
    # bold lines are guaranteed consistent, and every model uses the exact
    # same hierarchy as Human.
    delta_df = _abs_delta_long(human[["pid", "trial", "observation", "response"]])
    per_pid = (delta_df.groupby(["pid", "observation"])["delta"]
              .mean().reset_index())

    handles = [Line2D([0], [0], color=HUMAN_COLOR, lw=1.8)]
    labels = ["Human"]

    if show_individual:
        for pid, g in per_pid.groupby("pid"):
            g = g.sort_values("observation")
            ax.plot(g["observation"], g["delta"], color=INDIV_COLOR,
                    lw=0.6, alpha=0.5, zorder=2)
        handles.append(Line2D([0], [0], color=INDIV_COLOR, lw=0.8))
        labels.append("Individual pids")

    _plot_hierarchical_line(ax, per_pid, "delta", HUMAN_COLOR, zorder_line=3, zorder_fill=1)

    obs_ticks = set(human["observation"])
    for i, (model_type, mdf) in enumerate(models.items()):
        model_delta_df = _abs_delta_long(mdf)
        model_per_pid = (model_delta_df.groupby(["pid", "observation"])["delta"]
                        .mean().reset_index())
        color = palette[model_type]
        _plot_hierarchical_line(ax, model_per_pid, "delta", color,
                                zorder_line=4 + i, zorder_fill=1)
        handles.append(Line2D([0], [0], color=color, lw=1.8))
        labels.append(model_type)
        obs_ticks |= set(mdf["observation"])

    ax.set_xlabel("Observation")
    ax.set_ylabel("Mean |\u0394response|")
    ax.set_xticks(sorted(obs_ticks))
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=7, frameon=True, framealpha=0.9, ncol=1)
    sns.despine(ax=ax, top=True, right=True)


# ── Cols 3/4 shared helper — residuals within the prefix region only ───────
# Human only in this pass -- see module docstring.

def _add_resid_prefix(human: pd.DataFrame) -> pd.DataFrame:
    sub = human[human["observation"] < PREFIX_LENGTH]
    means = (sub.groupby(["pid", "observation", "qid"])["response"]
             .mean().reset_index().rename(columns={"response": "qid_mean"}))
    df2 = sub.merge(means, on=["pid", "observation", "qid"])
    df2["resid"] = df2["response"] - df2["qid_mean"]
    return df2


# ── Col 3 — Residual variance growth (prefix only) ──────────────────────────

def _plot_panel_variance_growth(ax, human: pd.DataFrame) -> None:
    df2 = _add_resid_prefix(human)
    MIN = 2
    grp = (df2.groupby(["pid", "observation", "qid"])["resid"]
           .apply(lambda x: x.std() if len(x) >= MIN else np.nan)
           .dropna().reset_index(name="std"))
    if grp.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    by_pid_obs = grp.groupby(["pid", "observation"])["std"].mean().reset_index()
    stats = by_pid_obs.groupby("observation")["std"].agg(["mean", "std"]).reset_index()
    n_pid = by_pid_obs["pid"].nunique()
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

def _plot_panel_autocorr(ax, human: pd.DataFrame) -> None:
    df2 = _add_resid_prefix(human)
    # A qid with only 1 repeat produces a trivially-zero residual (its
    # "mean" is just itself), not a genuine signal to autocorrelate --
    # the same degenerate case _plot_panel_variance_growth already guards
    # against via its own `len(x) >= MIN` check. Apply the identical guard
    # here before computing anything, rather than relying on df2 being
    # empty (it never is in this case -- it's full of meaningless zeros,
    # which is what produced scipy's "constant input" warning here for
    # task-colors before this fix: colors' current design gives every
    # qid exactly one repeat per participant, confirmed directly this
    # session -- see chat history). This is a correctness/honesty fix
    # only, NOT the qid-repeat redefinition itself (deliberately deferred
    # -- see module docstring).
    repeat_counts = df2.groupby(["pid", "observation", "qid"]).size()
    if not (repeat_counts >= 2).any():
        ax.text(0.5, 0.5, "Insufficient data\n(no qid repeats for this task)",
                ha="center", va="center", transform=ax.transAxes, color="0.5", style="italic")
        return
    lags = [1, 2, 3]
    pid_rs: dict[int, list[float]] = {lag: [] for lag in lags}

    for _, pid_df in df2.groupby("pid"):
        for lag in lags:
            pairs = []
            for (_, _), g in pid_df.groupby(["pid", "trial"]):
                # Pair by ACTUAL observation index, not array position --
                # a missing checkpoint can leave gaps (e.g. observations
                # [1,3] logged, 0 and 2 missing), and pairing by position
                # (old: r[:-lag], r[lag:]) would wrongly treat obs=1 and
                # obs=3 as a "lag=1" pair when they're actually 2 apart.
                # Confirmed as a real, if rare, issue against pilot 4's
                # data directly (2/160 prefix-trials affected) before
                # fixing this -- see chat history.
                obs_to_resid = dict(zip(g["observation"], g["resid"]))
                for o, resid_o in obs_to_resid.items():
                    if (o + lag) in obs_to_resid:
                        pairs.append((resid_o, obs_to_resid[o + lag]))
            if len(pairs) < 3:
                continue
            arr = np.array(pairs)
            # Guard against a single pid/lag combination happening to have
            # zero variance on one side (e.g. lag=3 with PREFIX_LENGTH=4
            # pairs exactly ONE point per trial -- observation 0 vs
            # observation 3 -- and a real participant who never moves the
            # slider away from its fixed per-trial starting position on
            # their very first observation would have an exactly-zero
            # residual there for EVERY trial. This is a genuine, real
            # behavioral pattern, not a bug -- but pearsonr silently
            # returns NaN for it rather than raising, which would
            # otherwise poison this pid's contribution to the whole
            # lag's cross-pid mean below via plain np.mean). Skip this
            # one (pid, lag) point rather than let one participant's edge
            # case NaN out an entire lag's aggregate.
            if arr[:, 0].std() <= 1e-9 or arr[:, 1].std() <= 1e-9:
                continue
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
# Human only in this pass -- see module docstring.

def _fit_lambda_curve_fit(df: pd.DataFrame) -> pd.Series:
    """Fits the power-law decay A*n^(-lambda) to each pid's own mean
    |delta response| vs observation curve -- in LOG-LOG SPACE (a plain
    linear regression of log(delta) on log(observation), lambda = -slope)
    rather than scipy.optimize.curve_fit's bounded nonlinear least squares
    directly on the raw curve. This is a real fix, not a style choice:
    with only ~32 trials per pid to average over 14 observation steps,
    these curves are noisy and often close to flat (confirmed directly
    against real task_backend data -- see chat history), and the bounded
    nonlinear optimizer was reliably getting stuck exactly at its own
    lam=0 lower bound for data like this (returning lambda~1e-11 to
    1e-21 -- a degenerate floor artifact, not a genuine near-zero
    estimate), rather than genuinely fitting anything. Log-log linear
    regression has no bounds to stick to and handles a flat or even
    slightly INCREASING curve (a real, honest possibility for noisy human
    data, unlike the old [0,2]-bounded fit, which couldn't express
    'no decay' as anything other than exactly 0) by simply returning a
    small or negative lambda instead of degenerating."""
    out: dict = {}
    for pid, grp in df.groupby("pid"):
        pieces = []
        for _, tg in grp.groupby("trial"):
            g = tg.sort_values("observation").copy()
            g["delta"] = g["response"].diff().abs()
            pieces.append(g)
        delta = pd.concat(pieces, ignore_index=True)
        curve = delta.groupby("observation")["delta"].mean().dropna()
        curve = curve[curve.index >= 1]
        if len(curve) < 3:
            continue
        n = curve.index.values.astype(float)
        y = curve.values.astype(float)
        if not (np.all(np.isfinite(n)) and np.all(np.isfinite(y))):
            continue
        # A delta of exactly 0 is possible (identical consecutive
        # responses) but undefined in log-space -- clip to a small floor
        # rather than drop the observation entirely, since dropping would
        # bias the fit toward whichever observations happened to have
        # nonzero movement.
        y = np.clip(y, 1e-6, None)
        try:
            slope, intercept, _, _, _ = linregress(np.log(n), np.log(y))
            out[pid] = -float(slope)
        except Exception:
            pass
    return pd.Series(out, name="lambda_")


def _fit_lambda_split_half(df: pd.DataFrame) -> pd.DataFrame:
    """Split-half by ODD/EVEN trial index, not first-half/second-half --
    a strict chronological split confounds genuine estimation noise (what
    split-half reliability is meant to measure) with any systematic
    drift in behavior over the session (learning, fatigue, boredom): a
    real drift would show up as LOWER reliability even if the
    moment-to-moment estimate itself is perfectly stable. Interleaving
    odd/even trials samples both halves from the same span of session-
    time, isolating noise from drift -- the standard recommendation in
    psychometrics over a strict first/second split (see chat history)."""
    rows = []
    for pid, grp in df.groupby("pid"):
        trials = sorted(grp["trial"].unique())
        halves = {"first": trials[0::2], "second": trials[1::2]}
        if min(len(halves["first"]), len(halves["second"])) < 3:
            continue
        for half_label, trial_set in halves.items():
            sub = grp[grp["trial"].isin(trial_set)].copy()
            lam = _fit_lambda_curve_fit(sub.assign(pid=pid))
            if pid in lam.index:
                rows.append({"pid": pid, "half": half_label,
                             "lambda_": float(lam[pid])})
    if not rows:
        return pd.DataFrame(columns=["pid", "first", "second"])
    wide = (pd.DataFrame(rows)
            .pivot(index="pid", columns="half", values="lambda_")
            .dropna())
    wide.columns.name = None
    return wide.reset_index()


def _plot_panel_splithalf_lambda(ax, human: pd.DataFrame,
                                 models: dict[str, pd.DataFrame],
                                 palette: dict) -> None:
    """Split-half (odd/even trial) reliability of the fitted decay rate lambda,
    one regplot per source. Mirrors figure_yoo_temporal.py's own T3 panel,
    which likewise shows Human plus every model rather than Human alone.

    Unlike cols 3-4 (see their own note in the module docstring), this panel IS
    meaningful for the deterministic math models: lambda is fitted to each
    source's own mean |delta response| curve, and that curve differs between
    odd and even trials because the underlying STIMULUS sequences differ -- no
    response noise is needed for the split to be informative. Models are
    therefore expected to be MORE reliable than Human here (having no response
    noise, only sequence sampling separates their halves); that gap is a
    result, not an artefact.
    """
    sources: list[tuple[str, pd.DataFrame, str]] = [("Human", human, HUMAN_COLOR)]
    for model_type in MODEL_ORDER:
        mdf = models.get(model_type)
        if mdf is not None:
            sources.append((model_type, mdf, palette.get(model_type, "0.5")))

    handles, labels = [], []
    for label, df, color in sources:
        wide = _fit_lambda_split_half(df)
        if len(wide) < 2:
            continue
        sns.regplot(data=wide, x="first", y="second", ax=ax, color=color,
                   ci=95 if len(wide) >= MIN_CORR_N else None,
                   scatter=True, line_kws={"lw": 1.5},
                   scatter_kws={"s": 20, "alpha": 0.7})
        handles.append(Line2D([0], [0], color=color, lw=1.5))
        if len(wide) >= MIN_CORR_N:
            r, p = pearsonr(wide["first"], wide["second"])
            labels.append(f"{label} r={r:.2f}{pvalue_to_stars(p)}")
        else:
            labels.append(f"{label} n={len(wide)}")

    if not handles:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    ax.legend(handles=handles, labels=labels, fontsize=7,
              frameon=True, framealpha=0.9)

    ax.set_xlabel("\u03bb (odd-indexed trials)")
    ax.set_ylabel("\u03bb (even-indexed trials)")
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel_lambda_model_vs_human(ax, human: pd.DataFrame,
                                      models: dict[str, pd.DataFrame],
                                      palette: dict) -> None:
    """Col 6 (~yoo temporal panel D / T4): lambda_model vs lambda_human per pid,
    one regplot per model, with an identity line.

    Both axes use the SAME model-free estimator (_fit_lambda_curve_fit, log-log
    regression on that source's own mean |delta response| curve) -- so this asks
    whether a model reproduces each individual's discounting RATE, not just the
    group average. It is the individual-differences counterpart to col 5's
    reliability check: col 5 establishes that lambda is measurable at all
    (Human r=0.96 on pilot 5), and only then is a cross-source correlation
    interpretable rather than noise-on-noise.

    DIFFERS FROM figure_yoo_temporal.py's own T4, which EXCLUDES Mean and
    LeakyIntegrator. All four are kept here, and Mean specifically earns its
    place as a NULL CONTROL: Mean has no free parameters, so its per-pid lambda
    varies only because the stimulus sequences differ, and it therefore CANNOT
    track genuine individual differences. A near-zero Mean correlation
    alongside a strong one for the parameterised models is direct evidence that
    those correlations reflect fitted individual variation rather than an
    artefact of the lambda-fitting procedure itself. If Mean ever correlates as
    strongly as the fitted models, something is wrong with the estimator.
    """
    lam_h = _fit_lambda_curve_fit(human)
    if lam_h.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    handles, labels = [], []
    for model_type in MODEL_ORDER:
        mdf = models.get(model_type)
        if mdf is None:
            continue
        lam_m = _fit_lambda_curve_fit(mdf)
        merged = pd.DataFrame({"human": lam_h, "model": lam_m}).dropna()
        if len(merged) < MIN_CORR_N:
            continue
        color = palette.get(model_type, "0.5")
        sns.regplot(data=merged, x="human", y="model", ax=ax, color=color,
                   ci=95, scatter=True, line_kws={"lw": 1.5},
                   scatter_kws={"s": 18, "alpha": 0.7})
        r, p = pearsonr(merged["human"], merged["model"])
        handles.append(Line2D([0], [0], color=color, lw=1.5))
        labels.append(f"{model_type} r={r:.2f}{pvalue_to_stars(p)}")

    if not handles:
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, "No fitted models\n(pass --plot_models)",
                ha="center", va="center", transform=ax.transAxes,
                color="0.5", style="italic")
        sns.despine(ax=ax, top=True, right=True)
        return

    # Identity line: a model on this line reproduces each individual's decay
    # rate exactly. Drawn from the union of both axis ranges so it always spans
    # the visible area, and computed AFTER the regplots so those set the limits.
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color="0.7", lw=0.8, ls="--", zorder=0)

    ax.set_xlabel("\u03bb (human)")
    ax.set_ylabel("\u03bb (model)")
    ax.legend(handles, labels, fontsize=7, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, default="soltani",
                        help="Folder under data/runs/ with fitting.submit + "
                             "fitting.collect output. Holds BOTH tasks: each "
                             "filename carries its own dataset stem, so "
                             "soltani_numbers and soltani_colors coexist.")
    parser.add_argument("--show_individual", dest="show_individual",
                        action="store_true", default=True,
                        help="Overlay each pid as a thin grey line in cols 1-2 "
                             "(Human only; default on)")
    parser.add_argument("--hide_individual", dest="show_individual",
                        action="store_false")
    parser.add_argument("--plot_models", dest="plot_models",
                        action="store_true", default=False,
                        help="Overlay fitted models in cols 1-2 (mean/CI lines) "
                             "and cols 5-6 (one regplot per source/model). Default off, to "
                             "keep pilot-stage human data most visible; pass this "
                             "flag to add Mean/LeakyIntegrator/PrimacyRecency/"
                             "RL_lambda. Cols 3-4 are human-only by necessity -- "
                             "see the module docstring.")
    parser.add_argument("--datafile", default=None,
                       help="Suffix identifying which dataset to load, e.g. 'pilot4' -> "
                            "data/soltani_numbers_pilot4.pkl / soltani_colors_pilot4.pkl. "
                            "Omit to use the canonical data/soltani_numbers.pkl / "
                            "soltani_colors.pkl.")
    args = parser.parse_args()

    run_dir = resolve_run_folder(args.run_folder)
    apply_style()
    pal = get_palette(len(MODEL_ORDER))
    palette = {m: pal[i] for i, m in enumerate(MODEL_ORDER)}

    fig, axes = plt.subplots(
        2, 6,
        figsize=(FIGURE_SIZE[0] * 1.5, FIGURE_SIZE[1]),
        constrained_layout=True,
    )

    for row, task in enumerate(TASK_ROWS):
        print(f"task-{task}:")
        human = _load_human(task, args.datafile)
        if human is None:
            print(f"  no data file found for this task/datafile combination -- skipping row")
            for col in range(6):
                axes[row, col].axis("off")
            axes[row, 0].text(0.5, 0.5, f"No {task} data\nfor this dataset",
                             ha="center", va="center", transform=axes[row, 0].transAxes,
                             color="0.5", style="italic")
            axes[row, 0].set_title(f"task-{task}", loc="left", fontsize=9, style="italic")
            continue
        models = {}
        if args.plot_models:
            for model_type in MODEL_ORDER:
                mdf = _load_model(task, model_type, run_dir, args.datafile)
                if mdf is not None:
                    models[model_type] = mdf

        _plot_panel_performance(axes[row, 0], human, models, args.show_individual, palette)
        _plot_panel_delta(axes[row, 1], human, models, args.show_individual, palette)
        human_for_repeats = add_quasi_qids(human) if task == "colors" else human
        _plot_panel_variance_growth(axes[row, 2], human_for_repeats)
        _plot_panel_autocorr(axes[row, 3], human_for_repeats)
        _plot_panel_splithalf_lambda(axes[row, 4], human, models, palette)
        _plot_panel_lambda_model_vs_human(axes[row, 5], human, models, palette)
        axes[row, 0].set_title(f"task-{task}", loc="left", fontsize=9, style="italic")

    label_panels(axes)

    if args.plot_models:
        footer = (f"Cols 1-2, 5-6 model fits: {', '.join(MODEL_ORDER)} from run "
                 f"'{args.run_folder}'. Cols 3-4 are human-only by necessity: "
                 "these models are deterministic, so their residual against a "
                 "qid-conditional mean is exactly zero. Cols 3-4 restricted to "
                 "observation < prefix_length=4, the only region guaranteed "
                 "identical across a qid's repeats in this task's design.")
    else:
        footer = ("Human data only (--plot_models off by default, to keep "
                 "pilot-stage human data most visible; pass --plot_models to "
                 "add fitted Mean/LeakyIntegrator/PrimacyRecency/RL_lambda "
                 "to cols 1-2 and 5-6). Cols 3-4 restricted to observation < "
                 "prefix_length=4, the only region guaranteed identical across "
                 "a qid's repeats in this task's design.")
    fig.text(0.5, -0.02, footer,
              ha="center", va="top", fontsize=7, style="italic", color="0.4")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure_soltani_temporal"
    if args.datafile:
        stem = f"{stem}_{args.datafile}"
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.pdf")


if __name__ == "__main__":
    main()
