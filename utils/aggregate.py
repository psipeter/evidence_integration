"""utils/aggregate.py

Shared aggregation for the temporal figures' error and |delta response| curves
(soltani cols 1-2, yoo panels A-B, carrabin panels A-B).

Extracted so the three figures cannot drift apart again. They previously used
three DIFFERENT schemes without that being visible anywhere: carrabin took the
mean over each pid's trials then mean +/- SEM across pids ('hier_mean_sem'); yoo
called sns.lineplot straight on long per-trial data with errorbar='ci', i.e. a
pooled mean with a row bootstrap ('flat_mean'); soltani had its own copy. Any
estimator borrowed between them silently inherited a different aggregation.

Everything here is estimator/plumbing only -- no dataset-specific constants, no
knowledge of which column is which panel. Callers pass long per-trial frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns


# Aggregation schemes for col 2's |delta response| curve. See
# plot_delta_aggregate for what each does and why the default is what it is.
AGGREGATE_MODES = ("flat_median", "flat_mean", "hier_mean_sem", "hier_mean_median")

# y-axis wording per mode, so the label never claims an estimator that wasn't used.
# Col 1's metric depends on the aggregation, not just its estimator: a median of
# squared errors under a sqrt IS the median absolute error, so the label must not
# keep saying "RMSE". See plot_error_aggregate.
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


def plot_delta_aggregate(ax, delta_df: pd.DataFrame, color: str, mode: str,
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
                 color=color, linewidth=1.8,
                 err_kws={"alpha": 0.2, "zorder": zorder_fill},
                 zorder=zorder_line, legend=False)


def plot_error_aggregate(ax, sq_err_df: pd.DataFrame, color: str, mode: str,
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

    Same row-bootstrap CI caveat as col 2 -- see plot_delta_aggregate.
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
                 color=color, linewidth=1.8,
                 err_kws={"alpha": 0.2, "zorder": zorder_fill},
                 zorder=zorder_line, legend=False)


def add_aggregate_args(parser, default_mode: str = "hier_mean_median") -> None:
    """Add --aggregate/--errorbar to a figure's parser, worded identically
    everywhere so the three temporal figures document the same choice the same
    way."""
    parser.add_argument(
        "--aggregate", choices=AGGREGATE_MODES, default=default_mode,
        help="How the per-trial error and |delta response| curves are "
             "aggregated. 'hier_mean_median' takes the mean over each pid's "
             "trials then the MEDIAN across pids -- mean where trials are "
             "exchangeable replicates, median where participants differ in "
             "kind. Avoids both a mean's outlier sensitivity and the "
             "response-grid quantisation a pooled median suffers. "
             "'hier_mean_sem' is mean +/- SEM across pids; 'flat_*' pool all "
             "trials and pids. See utils/aggregate.py for the measured "
             f"differences. Default {default_mode}.")
    parser.add_argument(
        "--errorbar", choices=sorted(ERRORBAR_SPEC), default=None,
        help="Band around those curves. Default depends on --aggregate (se for "
             "hier_mean_sem, ci otherwise). 'ci'/'se' are INFERENTIAL (how "
             "precisely is the central tendency pinned down); 'iqr'/'pi80' are "
             "DESCRIPTIVE percentile spreads of the underlying values and must "
             "NOT be called confidence intervals.")
