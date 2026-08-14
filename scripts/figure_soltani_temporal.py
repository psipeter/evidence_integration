#!/usr/bin/env python3
"""figure_soltani_temporal.py — T group figure for the soltani task/ pilot
(task-numbers + task-colors).

Layout: 2x6
  Row 1 = task-colors, Row 2 = task-numbers (standing row-order
  convention for soltani figures — see figure_soltani_performance.py)
  Col 1 (~carrabin temporal panel A / T1): Performance error (RMSE to
    the RUNNING MEAN of the observed stimulus stream, NOT the fixed
    generative true_mean/true_p BY DEFAULT -- pass --gt_mode true to switch;
    see _add_ground_truth's own docstring for how the two differ and for the
    true_mean/true_p scale mismatch it handles) vs observation. Human always shown; pass
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
    growth -- std(resid | obs, qid) vs observation. Human + stochastic
    models only (see below).
  Col 4 (~carrabin temporal's RENDERED panel D / T4): Within-trial lag-k
    residual autocorrelation (lag 1-3). Human + stochastic models only
    (see below).
  Col 5 (~yoo temporal panel C / T3): Split-half reliability (ODD/EVEN
    trials) of the decay-rate lambda fitted to |Delta response| vs
    observation, with scatter=True. Human AND each fitted model, one
    regplot per source -- matching figure_yoo_temporal.py's own T3.
  Col 6: cross-task comparison of the fitted decay rate lambda -- lambda
    (colors) on x, lambda (numbers) on y, one point per pid who did BOTH
    tasks. Tests whether an individual's response-change decay is a
    STABLE trait that transfers across tasks, or something specific to
    one task's own stimulus structure. Human only, matching
    figure_soltani_variability.py's own cross-task panel ("panel F" --
    row 2/numbers, col 3 there) exactly in both purpose and
    implementation (_plot_panel_crosstask there, _plot_panel_lambda_crosstask
    here) -- computed once across both tasks' human data, not per-row like
    every other column, and rendered only in the numbers row (row 1's own
    slot is turned off), same convention as that file uses for its own
    cross-task panel.

COLS 3-4 SHOW HUMAN + STOCHASTIC MODELS ONLY (never deterministic ones)
------------------------------------------------------------------------
Both panels are built on residuals against a qid-conditional mean:
resid = response - mean(response | pid, observation, qid). A qid's repeats share
an identical prefix by design, so a DETERMINISTIC model returns the identical
response on every repeat and its residual is EXACTLY ZERO -- verified against
real pilot 5 data: max|resid| was 0.000e+00 for Mean, LeakyIntegrator,
PrimacyRecency and RL_lambda over 1152 qualifying prefix rows, versus 0.68 for
Human. Including them would draw four flat lines at zero.

These two metrics (carrabin's T5/T6) exist precisely to measure
state-persistent response variability, which only a noisy generative process
has -- so eligibility is decided by _STOCHASTIC_MODELS (NEF, NoisyCounting),
not by MODEL_ORDER. Do not "fix" the missing deterministic curves by widening
that set.

Model responses carry no `qid` column, so _attach_qid merges it across from the
human frame -- specifically from human_for_repeats, i.e. AFTER add_quasi_qids
for colors, so a model is grouped by the same repeat structure Human is.

Col 5 does NOT have this problem and includes ALL models, deterministic ones
included; see _plot_panel_splithalf_lambda's own docstring for why.

Cols 3-4 DO now use quasi-qids for colors (task-colors)'s human data --
colors' own literal `qid` column never repeats, so a DIFFERENT repeat
structure is empirically derived instead: see utils/colors_quasi_qids.py's
own module docstring for the full definition and the empirical sweep
that settled its PREFIX_LENGTH/MIN_REPEATS defaults (now 5/3; numbers
is fixed at its designed 4 -- see NUMBERS_PREFIX_LENGTH). Numbers
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
filtering itself and merges everything on integer `pid`.

RESPONSE SCALE: everything here is on the canonical [-1,1] scale that
carrabin/yoo use, and is LEFT there -- no percent conversion anywhere. That is
deliberate: it means RMSE, mean |delta response|, and response variability are
numerically comparable with the yoo and carrabin figures, and it means any
estimator borrowed from those figures (notably the bounded power-law fit in
_fit_lambda_curve_fit, whose A bound assumes this scale) drops in unmodified.
An earlier version converted to [0,100] percent for readability, which required
round-trip compensation factors and produced exactly one such bug -- see
_fit_lambda_curve_fit's own docstring. The one remaining conversion is colors'
true_p, which build_from_df leaves on [0,1]; see _add_ground_truth.

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
from scipy.optimize import curve_fit as scipy_curve_fit
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, data_path, dataset_stem, resolve_run_folder
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette, label_panels, pvalue_to_stars
from utils.colors_quasi_qids import (
    MIN_REPEATS as QQ_MIN_REPEATS,
    PREFIX_LENGTH as QQ_PREFIX_LENGTH,
    add_quasi_qids,
)

TASK_ROWS        = ["colors", "numbers"]  # standing row-order convention
DATASET_FOR_TASK = {"colors": "soltani_colors", "numbers": "soltani_numbers"}
MODEL_ORDER       = ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda", "NEF"]

# Models with a NOISY generative process, so their response to a repeated
# stimulus prefix varies across repeats. Cols 3-4 measure exactly that
# variability, so only these are eligible there: a deterministic model returns
# the identical response on every repeat of a qid, making its residual against
# the qid-conditional mean identically zero (verified: max|resid| = 0.000e+00
# for all four math models). See the module docstring.
_STOCHASTIC_MODELS = frozenset({"NEF", "NoisyCounting"})
# Observations over which a qid's repeats share an identical stimulus prefix --
# PER TASK, because the two tasks get their repeat structure completely
# differently. numbers has a DESIGNED prefix of exactly 4 (verified: within
# (pid, qid), `value` is identical across trials for observations 0-3 in 216/216
# groups, and identical in 0/216 at observation 4) so raising it would silently
# admit non-shared stimuli and turn genuine stimulus differences into "response
# variability". colors has no designed prefix; its groups are constructed by
# utils.colors_quasi_qids, so its length is that module's tunable parameter.
NUMBERS_PREFIX_LENGTH = 4


def _prefix_length(task: str, colors_prefix: int = QQ_PREFIX_LENGTH) -> int:
    return colors_prefix if task == "colors" else NUMBERS_PREFIX_LENGTH
HUMAN_COLOR       = "0.3"
INDIV_COLOR       = "0.7"
MIN_CORR_N        = 3  # matches the threshold used in figure_soltani_variability.py


def _load_human(task: str, datafile: str | None = None,
                gt_mode: str = "running_mean") -> pd.DataFrame | None:
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
    out["response"] = df["response"]
    # Carry through whichever generative-truth column this task has, so
    # gt_mode='true' can use it. numbers -> true_mean, colors -> true_p; they
    # are on DIFFERENT scales, which _add_ground_truth handles.
    for col in ("true_mean", "true_p"):
        if col in df.columns:
            out[col] = df[col]
    out = _add_ground_truth(out, task, gt_mode=gt_mode)
    return out


# Ground-truth conventions for col 1, mirroring scripts/plot_sequences.py's own
# GT_MODES / --gt_mode toggle rather than inventing a second vocabulary.
GT_MODES = ("running_mean", "true")

# Axis-label wording per mode. "true mean/ratio" rather than naming a column,
# since the underlying column differs by task (true_mean vs true_p).
GT_LABEL = {"running_mean": "running mean", "true": "true mean/ratio"}

def _power_law(n, A, lam):
    """A * n**(-lam). Same form figure_yoo_temporal.py fits."""
    return A * np.power(np.asarray(n, dtype=float), -lam)


def _add_ground_truth(df: pd.DataFrame, task: str,
                      gt_mode: str = "running_mean") -> pd.DataFrame:
    """Add a `ground_truth` column on the [0,100] percent scale.

    gt_mode='running_mean' (default): the RUNNING mean/ratio of the observed
      stimulus stream itself, per (pid, trial) -- what a perfect "average what
      you've seen so far" agent would report at each observation. This is the
      quantity task_backend's own live "correct answer" panel shows real
      participants during the actual task; they are never shown the fixed
      generative target. Error DECAYS toward 0 as the estimate converges.

    gt_mode='true': the FIXED generative true_mean (numbers) / true_p (colors),
      constant within a trial (verified: nunique==1 per (pid,trial) for both
      tasks). Error PLATEAUS rather than decaying, at the sampling error between
      that trial's actual drawn values and the mean they were drawn from -- so
      the two modes answer different questions and are not rescalings of each
      other. Matches plot_sequences.py's own historical default.

    SCALE WARNING for gt_mode='true': build_from_df does NOT put these two
    columns on the same scale, and this is the ONE place in the soltani figures
    where a conversion is still required. true_mean is already rescaled to
    [-1,1] like value/response (verified [-0.700, 0.700]) and is used as-is.
    true_p is deliberately LEFT on its native [0,1] probability scale (verified
    [0.133, 0.867]) even though colors' `response` is on [-1,1] -- so it must be
    mapped with 2p-1. Using true_p directly would silently compress colors'
    ground truth into [0,1], i.e. half the response range and never negative.
    See utils/binary_transform.py's own module docstring, which flags the same
    mismatch.

    Requires `value` (raw stimulus, [-1,1]) already present in df for the
    running_mean mode.
    """
    if gt_mode not in GT_MODES:
        raise ValueError(f"gt_mode must be one of {GT_MODES}, got {gt_mode!r}")

    df = df.sort_values(["pid", "trial", "observation"]).copy()

    if gt_mode == "running_mean":
        running = df.groupby(["pid", "trial"])["value"].transform(
            lambda s: s.expanding().mean())
        df["ground_truth"] = running
        return df

    if task == "colors":
        if "true_p" not in df.columns:
            raise KeyError("gt_mode='true' needs a true_p column for colors")
        # true_p is P(blue) on [0,1]; map to the response scale as 2p-1.
        df["ground_truth"] = df["true_p"] * 2.0 - 1.0
    else:
        if "true_mean" not in df.columns:
            raise KeyError("gt_mode='true' needs a true_mean column for numbers")
        df["ground_truth"] = df["true_mean"]
    return df


# Back-compat alias: this was the only ground-truth function before the toggle.
def _add_running_mean_ground_truth(df: pd.DataFrame, task: str) -> pd.DataFrame:
    return _add_ground_truth(df, task, gt_mode="running_mean")


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
    out["response"] = df["response"]
    return out


# ── Col 1 — Performance error vs observation ────────────────────────────────

def _sq_err_long(df: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Per-(pid, trial, observation) SQUARED error against `ground_truth`.

    Kept un-aggregated on purpose. Col 1's error metric involves an averaging
    step by construction -- RMSE is sqrt(mean(sq_err)) -- so a "flat median of
    RMSEs" is not a well-defined thing to ask for: the mean over trials has
    already happened before you get an RMSE to take a median of. Returning the
    raw squared errors lets the aggregation choice be applied ONCE, afterwards,
    by composing it with the sqrt (see _plot_error_aggregate).
    """
    merged = df.merge(ground_truth[["pid", "trial", "observation", "ground_truth"]],
                      on=["pid", "trial", "observation"])
    return merged.assign(sq_err=(merged["response"] - merged["ground_truth"]) ** 2)


def _rmse_per_pid_obs(df: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Per-(pid, observation) RMSE. Used for the thin individual-pid overlay,
    which is per-pid by definition regardless of the across-pid aggregation."""
    long = _sq_err_long(df, ground_truth)
    return (long.groupby(["pid", "observation"])["sq_err"].mean()
            .apply(np.sqrt).reset_index(name="rmse"))


# Aggregation schemes for col 2's |delta response| curve. See
# _plot_delta_aggregate for what each does and why the default is what it is.
AGGREGATE_MODES = ("flat_median", "flat_mean", "hier_mean_sem", "hier_mean_median")

# y-axis wording per mode, so the label never claims an estimator that wasn't used.
# Col 1's metric depends on the aggregation, not just its estimator: a median of
# squared errors under a sqrt IS the median absolute error, so the label must not
# keep saying "RMSE". See _plot_error_aggregate.
ERROR_METRIC_LABEL = {
    "flat_median": "Median |error|",
    "flat_mean": "Pooled RMSE",
    "hier_mean_sem": "Performance error (RMSE)",
    "hier_mean_median": "Median of per-pid RMSE",
}

# Band options for cols 1-2. Two DIFFERENT kinds of thing, deliberately kept
# distinguishable because they answer different questions and a reader will
# assume whichever one the caption implies:
#   'ci'  -- percentile bootstrap CI of the ESTIMATOR (inferential: how well is
#            the central tendency pinned down?). seaborn ('ci', 95).
#   'se'  -- standard error of the MEAN across pids. Only meaningful with a mean
#            estimator; pairing it with a median would be simply wrong.
#   'iqr' -- 25th-75th percentile of the underlying values (DESCRIPTIVE: how
#            spread out are participants?). seaborn ('pi', 50). NOT a confidence
#            interval -- do not describe it as one.
#   'pi80'-- as 'iqr' but the 10th-90th percentile.
ERRORBAR_SPEC = {
    "ci": ("ci", 95),
    "se": "se",
    "iqr": ("pi", 50),
    "pi80": ("pi", 80),
}

# Default band per aggregation mode.
#
# hier_mean_median uses 'ci' -- a percentile bootstrap CI of the median over
# pids, the correct inferential interval for this estimator. Its edge looks
# step-like, and that is INHERENT rather than a resampling artefact: with ~27
# participants the bootstrap median can only land on a small set of order
# statistics (measured: 19 distinct values across 20000 resamples), so raising
# n_boot does nothing -- upper-edge mean |step| is 0.0188 / 0.0183 / 0.0183 at
# n_boot 1e3 / 1e4 / 1e5. Do not try to smooth it by raising n_boot.
#
# 'iqr' was considered and rejected as the default: smoother (0.0125) but WIDER
# (mean width 0.094 vs 0.069) and, more importantly, it answers a different
# question -- between-participant spread rather than how precisely the median is
# pinned down. Use --errorbar iqr (or --show_individual) when individual
# variability is the point. Two caveats on 'ci': at this n a percentile bootstrap
# CI of a median can be mildly anticonservative and asymmetric, so do not
# over-read a marginal non-overlap between two curves.
ERRORBAR_DEFAULT = {
    "flat_median": "ci",
    "flat_mean": "ci",
    "hier_mean_sem": "se",
    "hier_mean_median": "ci",
}

AGGREGATE_LABEL = {
    "flat_median": "Median",
    "flat_mean": "Mean",
    "hier_mean_sem": "Mean",
    "hier_mean_median": "Median of per-pid mean",
}


def _plot_delta_aggregate(ax, delta_df: pd.DataFrame, color: str, mode: str,
                          zorder_line: float, zorder_fill: float,
                          errorbar_kind: str | None = None) -> None:
    """Draw one source's |delta response| curve for col 2 under `mode`.

    `delta_df` is LONG per-trial data -- one row per (pid, trial, observation)
    with a `delta` column -- so every mode starts from the same input and only
    the aggregation differs. Applied identically to Human and to every model.

    WHY THE DEFAULT IS A POOLED (FLAT) MEDIAN. Per-pid |delta| LEVEL varies 3-4x
    across participants (yoo 0.026-0.238, soltani numbers 0.025-0.251), and the
    high-amplitude participants tend to be FLAT -- their movement is dominated by
    response noise that does not decay. Under a mean they contribute in
    proportion to their amplitude, so they dominate the late observations where
    everyone else has already converged, roughly HALVING the visible decay: the
    curve's implied power-law lambda is 0.243 under a mean vs 0.475 under a
    median for soltani numbers (yoo: 0.132 vs 0.226). A median counts each
    observation once regardless of amplitude.

    'flat' vs 'hier' matters less than it looks for the MEAN: trial counts are
    perfectly balanced (32/32 soltani, 30/30 yoo), so the pooled mean EQUALS the
    mean of per-pid means to floating point (verified, max|diff| 1.4e-17). The
    hierarchy was never doing anything for a mean. It matters a lot for a median,
    and the pooled version is also visibly smoother -- 0/13 upward steps for
    soltani vs 4/13 for the median of per-pid means -- because it takes a median
    over ~860 values per observation instead of 27.

    CI CAVEAT, read before quoting an interval. Seaborn bootstraps ROWS. For the
    'hier_*' modes the rows are already one-per-(pid, observation), so its
    bootstrap is correctly a bootstrap over PARTICIPANTS. For the 'flat_*' modes
    the rows are individual trials, which are NOT independent within a
    participant, so the interval is pseudo-replicated and TOO NARROW. It is
    still useful as a visual indication of central-tendency stability, but do not
    report it as a confidence interval over participants. A proper cluster
    bootstrap would resample pids and recompute the pooled statistic; that is
    deliberately not hand-rolled here.

    WHY 'hier_mean_median' AVOIDS THE FLAT MEDIAN'S STAIRCASE. Responses come
    from a 101-value slider, so a median taken over raw per-trial deltas lands
    exactly on a grid value and STAYS there across observations -- numbers'
    flat_median curve reads 0.06, 0.06, 0.06, 0.06, 0.06, 0.04, 0.04, visibly
    quantised. Averaging within a participant first produces a continuous
    per-pid value, so the median across pids is no longer pinned to the grid.
    It also gives a CI over PARTICIPANTS rather than a row bootstrap, so the band
    means what a reader will assume it means.

    This panel is descriptive. The per-participant characterisation of decay
    lives in cols 5-6, which fit lambda per pid and do inference across pids --
    so col 2 does not need to carry that burden.
    """
    if mode not in AGGREGATE_MODES:
        raise ValueError(f"aggregate must be one of {AGGREGATE_MODES}, got {mode!r}")

    if mode.startswith("hier"):
        # One row per (pid, observation) -> seaborn's row bootstrap IS a
        # participant bootstrap, which is what makes hier_mean_median's CI
        # correct without hand-rolling anything: errorbar=('ci',95) percentile-
        # bootstraps the MEDIAN over pids. (An earlier version of this function
        # left that mode as a stub claiming it needed an explicit bootstrap --
        # that was wrong; only pairing a median with errorbar='se' would have
        # been.) errorbar='se' with the mean reproduces the original
        # mean+/-SEM-across-pids behaviour exactly.
        data = (delta_df.groupby(["pid", "observation"])["delta"]
                .mean().reset_index())
        estimator = "median" if mode == "hier_mean_median" else "mean"
    else:
        data = delta_df
        estimator = "median" if mode == "flat_median" else "mean"
    errorbar = ERRORBAR_SPEC[errorbar_kind or ERRORBAR_DEFAULT[mode]]

    sns.lineplot(data=data, x="observation", y="delta", ax=ax,
                 estimator=estimator, errorbar=errorbar,
                 color=color, marker="o", markersize=5, linewidth=1.8,
                 err_kws={"alpha": 0.2, "zorder": zorder_fill},
                 zorder=zorder_line, legend=False)


def _plot_error_aggregate(ax, sq_err_df: pd.DataFrame, color: str, mode: str,
                          zorder_line: float, zorder_fill: float,
                          errorbar_kind: str | None = None) -> None:
    """Draw one source's col-1 error curve under `mode`, from raw squared errors.

    The aggregation is composed with the sqrt so it is applied exactly once:
      flat_median   -> sqrt(median(sq_err)) == MEDIAN ABSOLUTE ERROR
      flat_mean     -> sqrt(mean(sq_err))   == pooled RMSE
      hier_mean_sem -> per-(pid, observation) RMSE, then mean +/- SEM across pids
                       (the previous behaviour)

    NOTE flat_median changes the METRIC, not just the estimator: it is a median
    absolute error, not an RMSE, and the y-axis says so. That is the honest
    consequence of asking for a median here -- a median of squared errors under a
    sqrt IS the median absolute error, since sqrt is monotone. It is also far less
    sensitive to the handful of participants whose responses barely track the
    target at all.

    NOTE ALSO flat_mean != hier_mean_sem's point estimate here, unlike in col 2.
    Col 2's two versions coincide because trial counts are balanced and the
    operation is linear; here mean-of-sqrt differs from sqrt-of-mean by Jensen's
    inequality, so pooled RMSE sits at or above the mean of per-pid RMSEs.

    Same row-bootstrap CI caveat as col 2 -- see _plot_delta_aggregate.
    """
    if mode not in AGGREGATE_MODES:
        raise ValueError(f"aggregate must be one of {AGGREGATE_MODES}, got {mode!r}")

    if mode.startswith("hier"):
        # Per-pid RMSE first (sqrt of that pid's mean squared error), so the
        # across-pid step operates on one value per participant.
        data = (sq_err_df.groupby(["pid", "observation"])["sq_err"].mean()
                .apply(np.sqrt).reset_index(name="v"))
        ycol = "v"
        estimator = "median" if mode == "hier_mean_median" else "mean"
    else:
        data = sq_err_df
        ycol = "sq_err"
        estimator = ((lambda x: float(np.sqrt(np.median(x))))
                     if mode == "flat_median"
                     else (lambda x: float(np.sqrt(np.mean(x)))))
    errorbar = ERRORBAR_SPEC[errorbar_kind or ERRORBAR_DEFAULT[mode]]

    sns.lineplot(data=data, x="observation", y=ycol, ax=ax,
                 estimator=estimator, errorbar=errorbar,
                 color=color, marker="o", markersize=5, linewidth=1.8,
                 err_kws={"alpha": 0.2, "zorder": zorder_fill},
                 zorder=zorder_line, legend=False)


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
                            show_individual: bool, palette: dict,
                            gt_mode: str = "true",
                            aggregate: str = "hier_mean_median",
                            errorbar_kind: str | None = None) -> None:
    # Order matters: `ground_truth` was already resolved by _add_ground_truth per
    # gt_mode when the data was loaded, so the aggregation below is applied to
    # errors against whichever target that chose.
    hcols = human[["pid", "trial", "observation", "response"]]
    sq_err_df = _sq_err_long(hcols, human)
    rmse_df = _rmse_per_pid_obs(hcols, human)

    handles = [Line2D([0], [0], color=HUMAN_COLOR, lw=1.8)]
    labels = ["Human"]

    if show_individual:
        for pid, g in rmse_df.groupby("pid"):
            g = g.sort_values("observation")
            ax.plot(g["observation"], g["rmse"], color=INDIV_COLOR,
                    lw=0.6, alpha=0.5, zorder=2)
        handles.append(Line2D([0], [0], color=INDIV_COLOR, lw=0.8))
        labels.append("Individual pids")

    _plot_error_aggregate(ax, sq_err_df, HUMAN_COLOR, aggregate,
                          zorder_line=3, zorder_fill=1,
                          errorbar_kind=errorbar_kind)

    for i, (model_type, mdf) in enumerate(models.items()):
        color = palette[model_type]
        _plot_error_aggregate(ax, _sq_err_long(mdf, human), color, aggregate,
                              zorder_line=4 + i, zorder_fill=1,
                              errorbar_kind=errorbar_kind)
        handles.append(Line2D([0], [0], color=color, lw=1.8))
        labels.append(model_type)

    obs_ticks = sorted(set(human["observation"]) | {o for m in models.values()
                                                    for o in m["observation"]})
    ax.set_xlabel("Observation")
    ax.set_ylabel(f"{ERROR_METRIC_LABEL[aggregate]}\nvs {GT_LABEL[gt_mode]}")
    ax.set_xticks(obs_ticks)
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=7, frameon=True, framealpha=0.9, ncol=1)
    sns.despine(ax=ax, top=True, right=True)


# ── Col 2 — Mean |Delta response| vs observation ────────────────────────────

# First observation whose |delta response| is worth plotting in col 2, PER TASK.
# numbers: 1, the first defined delta (response[1]-response[0]).
# colors: 2, deliberately dropping the first delta. With binary evidence the
# response change at observation 1 is near-degenerate -- the running mean either
# does not move (second draw matches the first) or jumps the whole way -- so the
# per-trial distribution is BIMODAL on essentially two values. Measured on
# complete_pairs: the colors Mean model has just 2 distinct delta values there
# with 58% exactly 0, giving median 0.000 against mean 0.424; colors humans have
# 46% exact zeros, median 0.060 against mean 0.394 (a 6.5x discrepancy). A median
# lands in the zero mode and produces a spurious dip-then-spike. numbers has no
# such problem (7-9% zeros, ~87 distinct values, median 0.100 vs mean 0.145).
DELTA_MIN_OBS = {"colors": 2, "numbers": 1}


def _abs_delta_long(df: pd.DataFrame, min_observation: int = 1) -> pd.DataFrame:
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
    return out[out["observation"] >= min_observation].dropna(subset=["delta"])


def _plot_panel_delta(ax, human: pd.DataFrame, models: dict[str, pd.DataFrame],
                      show_individual: bool, palette: dict,
                      aggregate: str = "hier_mean_median",
                      min_observation: int = 1,
                      errorbar_kind: str | None = None) -> None:
    # Per-pid mean |delta| first (pooling over that pid's own trials) --
    # this is both what the thin individual-pid lines plot directly AND
    # what every bold line's mean/SEM is computed from, so thin lines and
    # bold lines are guaranteed consistent, and every model uses the exact
    # same hierarchy as Human.
    delta_df = _abs_delta_long(human[["pid", "trial", "observation", "response"]],
                               min_observation)
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

    _plot_delta_aggregate(ax, delta_df, HUMAN_COLOR, aggregate,
                          zorder_line=3, zorder_fill=1,
                          errorbar_kind=errorbar_kind)

    obs_ticks = set(human["observation"])
    for i, (model_type, mdf) in enumerate(models.items()):
        model_delta_df = _abs_delta_long(mdf, min_observation)
        color = palette[model_type]
        _plot_delta_aggregate(ax, model_delta_df, color, aggregate,
                              zorder_line=4 + i, zorder_fill=1,
                              errorbar_kind=errorbar_kind)
        handles.append(Line2D([0], [0], color=color, lw=1.8))
        labels.append(model_type)
        obs_ticks |= set(mdf["observation"])

    ax.set_xlabel("Observation")
    ax.set_ylabel(f"{AGGREGATE_LABEL[aggregate]} |\u0394response|")
    ax.set_xticks(sorted(obs_ticks))
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=7, frameon=True, framealpha=0.9, ncol=1)
    sns.despine(ax=ax, top=True, right=True)


# ── Cols 3/4 shared helpers — residuals within the prefix region only ──────
# Human + STOCHASTIC models only (_STOCHASTIC_MODELS) -- see module docstring.

def _attach_qid(mdf: pd.DataFrame, human_for_repeats: pd.DataFrame) -> pd.DataFrame:
    """Give a model's responses the same `qid` labelling as the human data.

    Model response files carry only [pid, trial, observation, response] -- qid
    lives in the human data. Merging it across is what lets a model's residuals
    be computed against the SAME repeat groups as Human, which is the only way
    the two are comparable in cols 3-4.

    Takes human_for_repeats (post-add_quasi_qids for colors) rather than the raw
    human frame, so colors' empirically-derived quasi-qid grouping is applied to
    models too instead of its real, non-repeating qid.
    """
    return mdf.merge(
        human_for_repeats[["pid", "trial", "observation", "qid"]],
        on=["pid", "trial", "observation"],
        how="inner",
    )


def _stochastic_models(models: dict[str, pd.DataFrame],
                       human_for_repeats: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """(model_type, df-with-qid) for each fitted STOCHASTIC model, in
    MODEL_ORDER. Deterministic models are excluded by construction."""
    out = []
    for model_type in MODEL_ORDER:
        if model_type not in _STOCHASTIC_MODELS:
            continue
        mdf = models.get(model_type)
        if mdf is None:
            continue
        out.append((model_type, _attach_qid(mdf, human_for_repeats)))
    return out


def _add_resid_prefix(human: pd.DataFrame, prefix_length: int) -> pd.DataFrame:
    sub = human[human["observation"] < prefix_length]
    means = (sub.groupby(["pid", "observation", "qid"])["response"]
             .mean().reset_index().rename(columns={"response": "qid_mean"}))
    df2 = sub.merge(means, on=["pid", "observation", "qid"])
    df2["resid"] = df2["response"] - df2["qid_mean"]
    return df2


# ── Col 3 — Residual variance growth (prefix only) ──────────────────────────

def _variance_growth_stats(df: pd.DataFrame, prefix_length: int,
                           return_per_pid: bool = False):
    """Per-observation mean/SE of within-qid residual SD. Same hierarchy for
    Human and every model, so their curves are directly comparable.

    return_per_pid=True also returns the intermediate per-(pid, observation)
    frame, for the thin individual-pid overlay. That overlay matters more here
    than it looks: the cross-pid SEM band is a BETWEEN-subject interval, while
    the trend it sits on is a WITHIN-subject one, and absolute variability level
    differs enormously across participants (numbers 0.5-10.0, colors 0.03-24.05
    on complete_pairs). So a visually ambiguous band can coexist with a strong
    per-pid trend -- for numbers, 23/27 pids have a positive slope, Wilcoxon
    p=0.0004, against a band that looks marginal. Showing the individual lines
    is what makes that heterogeneity legible rather than hiding it inside the
    band. NOTE the aggregation itself is deliberately identical to
    figure_carrabin_temporal.py's own panel D (std within (pid, obs, qid) ->
    mean over qid within (pid, obs) -> mean/SD across pids, SE = SD/sqrt(n_pid))
    and to cols 1-2 of this figure; do not switch these two columns to a
    different error convention in isolation.
    """
    df2 = _add_resid_prefix(df, prefix_length)
    MIN = 2
    grp = (df2.groupby(["pid", "observation", "qid"])["resid"]
           .apply(lambda x: x.std() if len(x) >= MIN else np.nan)
           .dropna().reset_index(name="std"))
    if grp.empty:
        return (None, None) if return_per_pid else None
    by_pid_obs = grp.groupby(["pid", "observation"])["std"].mean().reset_index()
    stats = by_pid_obs.groupby("observation")["std"].agg(["mean", "std"]).reset_index()
    n_pid = by_pid_obs["pid"].nunique()
    stats["se"] = stats["std"] / np.sqrt(n_pid)
    if return_per_pid:
        return stats, by_pid_obs
    return stats


def _plot_panel_variance_growth(ax, human: pd.DataFrame,
                                models: dict[str, pd.DataFrame] | None = None,
                                palette: dict | None = None,
                                show_individual: bool = True,
                                prefix_length: int = NUMBERS_PREFIX_LENGTH) -> None:
    palette = palette or {}
    stats, per_pid = _variance_growth_stats(human, prefix_length, return_per_pid=True)
    if stats is None:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    handles, labels = [], []

    # Human individual pids, same convention as cols 1-2 (Human only; models
    # show mean/CI only) and gated by the same --show_individual/
    # --hide_individual flag.
    if show_individual:
        for _, g in per_pid.groupby("pid"):
            g = g.sort_values("observation")
            ax.plot(g["observation"], g["std"], color=INDIV_COLOR,
                    lw=0.6, alpha=0.5, zorder=2)

    ax.plot(stats["observation"], stats["mean"], "o-", color=HUMAN_COLOR,
            lw=1.8, ms=5, zorder=5)
    ax.fill_between(stats["observation"], stats["mean"] - stats["se"],
                    stats["mean"] + stats["se"], color=HUMAN_COLOR, alpha=0.25,
                    zorder=3)
    handles.append(Line2D([0], [0], color=HUMAN_COLOR, lw=1.5))
    labels.append("Human")
    if show_individual:
        handles.append(Line2D([0], [0], color=INDIV_COLOR, lw=0.8))
        labels.append("Individual pids")

    for i, (model_type, mdf) in enumerate(_stochastic_models(models or {}, human)):
        mstats = _variance_growth_stats(mdf, prefix_length)
        if mstats is None:
            continue
        color = palette.get(model_type, "0.5")
        ax.plot(mstats["observation"], mstats["mean"], "o-", color=color,
                lw=1.8, ms=5, zorder=4 + i)
        ax.fill_between(mstats["observation"], mstats["mean"] - mstats["se"],
                        mstats["mean"] + mstats["se"], color=color, alpha=0.25,
                        zorder=1)
        handles.append(Line2D([0], [0], color=color, lw=1.5))
        labels.append(model_type)

    ax.set_xlabel("Observation (prefix only)")
    ax.set_ylabel("Response variability")
    ax.set_xticks(range(prefix_length))
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Col 4 — Within-trial residual autocorrelation (prefix only) ────────────

def _autocorr_stats(df: pd.DataFrame, prefix_length: int,
                    return_per_pid: bool = False):
    """Cross-pid mean/SEM of within-trial residual autocorrelation at lags 1-3.

    Returns (lags, means, sems), or the string "no_repeats" / "insufficient"
    when it cannot be computed -- the caller renders the message, so this stays
    reusable for Human and for each stochastic model. With return_per_pid=True,
    appends the per-pid {lag: r} mapping for the thin individual-pid overlay.

    As in col 3, the SEM band here is BETWEEN-subject while the lag-decay claim
    is WITHIN-subject: for numbers, 13/14 pids have r(lag1) > r(lag3), Wilcoxon
    p=0.0002, which a between-pid band understates. Individual lines make that
    visible without changing the error convention cols 1-2 use.

    Note lag 3 is much weaker here than in carrabin's own panel: the prefix
    restriction leaves progressively fewer residual pairs per
    trial at lags 1/2/3, versus carrabin's 4/3/2 over its full 5-observation
    sequence. Lag 3 is therefore a single pair per trial and should not be
    over-read; it is also why the zero-variance guard below is needed at all.
    """
    df2 = _add_resid_prefix(df, prefix_length)
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
        return "no_repeats"
    lags = list(range(1, prefix_length))
    pid_rs: dict[int, list[float]] = {lag: [] for lag in lags}
    # pid -> {lag: r}. pid_rs above deliberately discards which pid each value
    # came from (it only needs the cross-pid aggregate), so track it separately
    # rather than changing that structure.
    per_pid_rs: dict[object, dict[int, float]] = {}

    for pid_key, pid_df in df2.groupby("pid"):
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
            # zero variance on one side (e.g. the longest lag in a short prefix
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
            per_pid_rs.setdefault(pid_key, {})[lag] = rv

    if all(len(v) == 0 for v in pid_rs.values()):
        return "insufficient"

    means = np.array([np.mean(pid_rs[lag]) if pid_rs[lag] else np.nan for lag in lags])
    sems = np.array([np.std(pid_rs[lag]) / np.sqrt(len(pid_rs[lag]))
                    if len(pid_rs[lag]) > 1 else np.nan for lag in lags])
    if return_per_pid:
        return lags, means, sems, per_pid_rs
    return lags, means, sems


def _plot_panel_autocorr(ax, human: pd.DataFrame,
                         models: dict[str, pd.DataFrame] | None = None,
                         palette: dict | None = None,
                         show_individual: bool = True,
                         prefix_length: int = NUMBERS_PREFIX_LENGTH) -> None:
    palette = palette or {}
    res = _autocorr_stats(human, prefix_length, return_per_pid=True)
    if isinstance(res, str):
        msg = ("Insufficient data\n(no qid repeats for this task)"
               if res == "no_repeats" else "Insufficient data")
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return
    lags, means, sems, per_pid_rs = res

    handles, labels = [], []

    # Human individual pids, same convention as cols 1-3. A pid is drawn only
    # across the lags it actually has: the zero-variance and <3-pairs guards
    # above can drop individual (pid, lag) points, so joining across a gap
    # would imply a value that was never computed.
    if show_individual:
        for _, lag_map in per_pid_rs.items():
            xs = [lag for lag in lags if lag in lag_map]
            if len(xs) < 2:
                continue
            ax.plot(xs, [lag_map[lag] for lag in xs], color=INDIV_COLOR,
                    lw=0.6, alpha=0.5, zorder=2)

    ax.plot(lags, means, "o-", color=HUMAN_COLOR, lw=1.8, ms=5, zorder=5)
    ax.fill_between(lags, means - sems, means + sems, color=HUMAN_COLOR,
                    alpha=0.2, zorder=3)
    handles.append(Line2D([0], [0], color=HUMAN_COLOR, lw=1.5))
    labels.append("Human")
    if show_individual:
        handles.append(Line2D([0], [0], color=INDIV_COLOR, lw=0.8))
        labels.append("Individual pids")

    for i, (model_type, mdf) in enumerate(_stochastic_models(models or {}, human)):
        mres = _autocorr_stats(mdf, prefix_length)
        if isinstance(mres, str):
            continue
        mlags, mmeans, msems = mres
        color = palette.get(model_type, "0.5")
        ax.plot(mlags, mmeans, "o-", color=color, lw=1.8, ms=5, zorder=4 + i)
        ax.fill_between(mlags, mmeans - msems, mmeans + msems, color=color,
                        alpha=0.2, zorder=1)
        handles.append(Line2D([0], [0], color=color, lw=1.5))
        labels.append(model_type)

    ax.set_xlabel("Lag (observations, within prefix)")
    ax.set_ylabel("Autocorrelation of trial-to-trial deviations")
    ax.set_xticks(lags)
    ax.legend(handles, labels, fontsize=8, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Col 5 — Split-half reliability of lambda (scatter=True) ────────────────
# Human AND all models (deterministic included) -- see the panel's docstring.

def _fit_lambda_curve_fit(df: pd.DataFrame) -> pd.Series:
    """Per-pid decay exponent lambda, fitting A*n^(-lambda) to that pid's own
    mean |delta response| vs n curve by BOUNDED NONLINEAR LEAST SQUARES
    (scipy.optimize.curve_fit), identical to figure_yoo_temporal.py's own
    estimator -- same functional form, same p0=[0.1, 0.5], same
    bounds=([0,0],[2,2]) -- so soltani and yoo lambdas are directly comparable.

    TWO THINGS HAVE TO BE RIGHT FOR THAT COMPARABILITY, and both were wrong here
    before; see the notes below. Do not "simplify" either one away.

    (1) n MEANS "NUMBER OF OBSERVATIONS SEEN", NOT THE RAW observation VALUE.
    yoo is 1-indexed, so its first valid delta sits at observation 2 with 2
    observations seen, and n == observation. The soltani datasets are 0-indexed
    (observation 0-14), so the first valid delta sits at observation 1 with 2
    observations seen -- n must therefore be observation + 1. Feeding the raw
    0-indexed value understated n by one, which has enormous leverage at the low
    end of a power law (log(1) = 0 vs log(2) = 0.693) and made soltani lambdas
    silently non-comparable to yoo's: human mean 0.279 raw vs 0.346 corrected.

    (2) THE CURVE MUST BE ON THE [-1,1] SCALE BEFORE FITTING, which it now is
    throughout this figure. yoo fits responses on the canonical [-1,1] scale,
    where the |delta| curve starts around 0.11 and A in [0,2] is a comfortable
    bound. An earlier version of this figure converted responses to [0,100]
    percent for readability, putting the same curve near 6.0 -- so A saturated at
    its upper bound of 2, the fit could not reach the curve, and lambda collapsed
    to 0 for most pids. That BOUNDS artefact was mistaken for the nonlinear
    optimizer being unusable on short, noisy soltani curves, and was why it got
    (wrongly) replaced by a log-log regression. Do not reintroduce a percent
    conversion upstream of this fit.

    DO NOT USE A LOG-LOG LINEAR REGRESSION FOR LAMBDA -- here or anywhere else in
    this project. It looks equivalent (it fits the same power law) but it is not
    the same estimator: it minimises squared error in LOG space, which weights
    the small late-observation deltas far more heavily than the large early ones
    and is acutely sensitive to the arbitrary floor you must impose on
    delta == 0. It also silently diverges from yoo/carrabin, which is exactly how
    the two bugs above went unnoticed. On real complete_pairs data the two
    methods correlate r=0.91 (human) / r=0.95 (RL_lambda) but differ in level --
    human mean 0.433 nonlinear vs 0.346 log-log -- so switching between them
    shifts every lambda in the figure.

    Pids whose fit does not converge are omitted from the returned Series (2/27
    for human numbers on complete_pairs), matching yoo's behaviour rather than
    substituting a degenerate value.
    """
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
        # (1) observations seen, not the 0-indexed observation value.
        n = curve.index.values.astype(float) + 1.0
        # (2) back to the [-1,1] scale yoo fits on, so A in [0,2] is meaningful.
        y = curve.values.astype(float)
        if not (np.all(np.isfinite(n)) and np.all(np.isfinite(y))):
            continue
        try:
            popt, _ = scipy_curve_fit(_power_law, n, y, p0=[0.1, 0.5],
                                      bounds=([0.0, 0.0], [2.0, 2.0]),
                                      maxfev=2000)
            out[int(pid)] = float(popt[1])
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


def _plot_panel_lambda_crosstask(ax, lambda_colors: pd.Series, lambda_numbers: pd.Series) -> None:
    """Col 6: cross-task comparison of the fitted decay rate lambda, one
    point per pid who did BOTH tasks. Mirrors
    figure_soltani_variability.py's own _plot_panel_crosstask exactly --
    same merge-on-integer-pid logic, same MIN_CORR_N gate, same graceful
    "too few"/"none did both" messages -- just for lambda instead of
    prefix response variability. lambda_colors/lambda_numbers are each a
    pd.Series indexed by pid (the return type of _fit_lambda_curve_fit),
    not a DataFrame -- merge happens directly on that shared index, which
    is valid regardless of colors' quasi-qids: quasi-qid relabeling only
    ever affects cols 3-4's OWN repeat-structure computation, never
    touches lambda fitting (col 5/6), which works on |delta response|
    alone and has no qid dependency at all."""
    merged = pd.DataFrame({"colors": lambda_colors, "numbers": lambda_numbers}).dropna()

    if len(merged) < 2:
        msg = ("No pids completed both tasks" if len(merged) == 0
              else f"Only {len(merged)} pid completed both tasks (need >=2 to plot)")
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
                color="0.5", style="italic")
        return

    ax.scatter(merged["colors"], merged["numbers"], color=HUMAN_COLOR, s=30, alpha=0.8, zorder=3)

    if len(merged) >= MIN_CORR_N:
        sns.regplot(data=merged, x="colors", y="numbers", ax=ax, color=HUMAN_COLOR,
                   ci=95, scatter=False, line_kws={"lw": 1.5})
        r, p = pearsonr(merged["colors"], merged["numbers"])
        ax.legend(handles=[Line2D([0], [0], color=HUMAN_COLOR, lw=1.5)],
                  labels=[f"Human r={r:.2f}{pvalue_to_stars(p)}"],
                  fontsize=8, frameon=True, framealpha=0.9)
    else:
        ax.text(0.02, 0.98, f"n={len(merged)} (too few for r)",
                ha="left", va="top", transform=ax.transAxes,
                fontsize=7, style="italic", color="0.5")

    ax.set_xlabel("\u03bb (colors)")
    ax.set_ylabel("\u03bb (numbers)")
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
                        help="Overlay each pid as a thin grey line in cols 1-4 "
                             "(Human only; models show mean/CI regardless). "
                             "Default on. Worth turning off for publication: "
                             "individual lines stretch the y-axis considerably, "
                             "since absolute variability differs by up to ~770x "
                             "across participants in cols 3-4.")
    parser.add_argument("--hide_individual", dest="show_individual",
                        action="store_false")
    parser.add_argument("--plot_models", dest="plot_models",
                        action="store_true", default=False,
                        help="Overlay fitted models in cols 1-2 (mean/CI lines) "
                             "and col 5 (one regplot per source/model). Default off, to "
                             "keep pilot-stage human data most visible; pass this "
                             "flag to add Mean/LeakyIntegrator/PrimacyRecency/"
                             "RL_lambda. Cols 3-4 are human-only by necessity, and "
                             "col 6 (cross-task lambda) is human-only by design -- "
                             "see the module docstring for both.")
    parser.add_argument("--aggregate", choices=AGGREGATE_MODES, default="hier_mean_median",
                        help="COLS 1-2: how the per-trial error (col 1) and "
                             "|delta response| (col 2) are aggregated. "
                             "'hier_mean_median' (default) takes the mean over each pid's "
                             "trials then the MEDIAN across pids -- mean where trials "
                             "are exchangeable replicates, median where participants "
                             "differ in kind. Avoids both a mean's outlier sensitivity "
                             "and the response-grid quantisation a pooled median "
                             "suffers. 'flat_median' pools all trials and pids and "
                             "takes the median -- robust to the 3-4x spread in per-pid "
                             "response-change AMPLITUDE, which under a mean roughly "
                             "halves the visible decay. 'flat_mean' pools and means "
                             "(identical to hier_mean_sem's point estimate, since trial "
                             "counts are balanced). 'hier_mean_sem' is the previous "
                             "behaviour: mean over each pid's trials, then mean+/-SEM "
                             "across pids. 'hier_mean_median' is a STUB. Note the "
                             "flat_* CI is a row bootstrap and so too narrow -- see "
                             "_plot_delta_aggregate's docstring.")
    parser.add_argument("--errorbar", choices=sorted(ERRORBAR_SPEC), default=None,
                        help="COLS 1-2 band. Default depends on --aggregate "
                             "(se for hier_mean_sem, ci otherwise). 'ci'/'se' are INFERENTIAL (how "
                             "precisely is the central tendency pinned down); "
                             "'iqr'/'pi80' are DESCRIPTIVE percentile spreads of the "
                             "underlying values and must NOT be called confidence "
                             "intervals.")
    parser.add_argument("--colors_prefix_length", type=int, default=QQ_PREFIX_LENGTH,
                        help="COLORS ONLY: how many leading observations a "
                             "quasi-qid group must share (cols 3-4). Colors has no "
                             "designed prefix, so this is free; numbers is fixed at "
                             "its designed 4 and is NOT affected by this flag. "
                             f"Default {QQ_PREFIX_LENGTH}.")
    parser.add_argument("--colors_min_repeats", type=int, default=QQ_MIN_REPEATS,
                        help="COLORS ONLY: minimum trials sharing a prefix for a "
                             "quasi-qid group to qualify. Note cols 3 and 4 want "
                             "different values -- col 3 needs >=3 for a usable "
                             "within-group SD, col 4 only needs residual PAIRS and "
                             f"does better at 2. Default {QQ_MIN_REPEATS}.")
    parser.add_argument("--gt_mode", choices=GT_MODES, default="true",
                        help="Col 1 ground truth. 'true' (default): the FIXED "
                             "generative true_mean/true_p, constant within a trial, "
                             "so error starts high and DECAYS as evidence "
                             "accumulates. 'running_mean': the running mean/ratio of "
                             "the observations so far -- what the task's own live "
                             "feedback shows participants -- against which error is "
                             "flat-to-rising, because the target itself moves. The two "
                             "converge late in a trial as the running mean approaches "
                             "the true mean. Mirrors plot_sequences.py's own "
                             "--gt_mode.")
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

    lambda_by_task: dict[str, pd.Series] = {}

    for row, task in enumerate(TASK_ROWS):
        print(f"task-{task}:")
        human = _load_human(task, args.datafile, args.gt_mode)
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

        _plot_panel_performance(axes[row, 0], human, models, args.show_individual,
                                palette, args.gt_mode, args.aggregate,
                                args.errorbar)
        _plot_panel_delta(axes[row, 1], human, models, args.show_individual,
                          palette, args.aggregate,
                          DELTA_MIN_OBS.get(task, 1), args.errorbar)
        colors_prefix = args.colors_prefix_length
        human_for_repeats = (add_quasi_qids(human, prefix_length=colors_prefix,
                                           min_repeats=args.colors_min_repeats)
                             if task == "colors" else human)
        prefix_length = _prefix_length(task, colors_prefix)
        _plot_panel_variance_growth(axes[row, 2], human_for_repeats, models,
                                    palette, args.show_individual, prefix_length)
        _plot_panel_autocorr(axes[row, 3], human_for_repeats, models,
                             palette, args.show_individual, prefix_length)
        _plot_panel_splithalf_lambda(axes[row, 4], human, models, palette)
        lambda_by_task[task] = _fit_lambda_curve_fit(human)
        axes[row, 0].set_title(f"task-{task}", loc="left", fontsize=9, style="italic")

    # Col 6: cross-task lambda comparison -- computed once across both
    # tasks' human data (not per-row like every other column), rendered
    # only in the numbers row -- same convention
    # figure_soltani_variability.py's own cross-task panel uses for
    # itself ("panel F", row 2/numbers, col 3 there).
    numbers_row = TASK_ROWS.index("numbers")
    colors_row = TASK_ROWS.index("colors")
    axes[colors_row, 5].axis("off")
    if "colors" in lambda_by_task and "numbers" in lambda_by_task:
        _plot_panel_lambda_crosstask(axes[numbers_row, 5], lambda_by_task["colors"], lambda_by_task["numbers"])
    else:
        axes[numbers_row, 5].axis("off")
        axes[numbers_row, 5].text(0.5, 0.5, "Cross-task comparison needs\nboth tasks' data",
                                 ha="center", va="center", transform=axes[numbers_row, 5].transAxes,
                                 color="0.5", style="italic")

    label_panels(axes)

    if args.plot_models:
        footer = (f"Cols 1-2, 5 model fits: {', '.join(MODEL_ORDER)} from run "
                 f"'{args.run_folder}'. Cols 3-4 are human-only by necessity: "
                 "these models are deterministic, so their residual against a "
                 "qid-conditional mean is exactly zero. Col 6 (cross-task lambda) "
                 "is human-only by design -- an individual-differences check, not "
                 "a model-fit panel. " "Cols 3-4 restricted to the shared-prefix window "
                 f"(numbers {NUMBERS_PREFIX_LENGTH} obs, by design; colors "
                 f"{args.colors_prefix_length} obs, min_repeats="
                 f"{args.colors_min_repeats}, constructed).")
    else:
        footer = ("Human data only (--plot_models off by default, to keep "
                 "pilot-stage human data most visible; pass --plot_models to "
                 "add fitted Mean/LeakyIntegrator/PrimacyRecency/RL_lambda "
                 "to cols 1-2 and 5-6). " "Cols 3-4 restricted to the shared-prefix window "
                 f"(numbers {NUMBERS_PREFIX_LENGTH} obs, by design; colors "
                 f"{args.colors_prefix_length} obs, min_repeats="
                 f"{args.colors_min_repeats}, constructed)." " Col 6 "
                 "(cross-task lambda) is human-only by design, unaffected by "
                 "--plot_models.")
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
