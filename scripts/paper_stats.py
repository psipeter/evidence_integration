#!/usr/bin/env python3
"""scripts/paper_stats.py

Named paper_stats.py, NOT statistics.py -- the obvious name shadows the
Python standard library's own `statistics` module (imported by
seaborn._statistics, among others), which silently broke scripts/
make_figures.py's own import chain the first time this was tried (any
script placed in scripts/ implicitly puts that directory on sys.path,
so a same-named local file wins over the stdlib one). Do not rename this
back to statistics.py.

Central home for statistical analyses whose numeric results (r values, p
values, significance stars, fractions of participants meeting some
criterion, etc.) get reported directly in paper/main.tex. Figure scripts
(scripts/make_figures.py, scripts/neural_experiments.py) still compute
plenty of their own inline statistics for figure annotations (e.g. paired
Wilcoxon stars on a boxplot) -- those stay where they are for now. This
script is for analyses whose PRIMARY purpose is a number reported in the
text, not a plotted annotation, starting with new analyses going forward.

TODO (not yet done): port the existing ad hoc statistical computations
already scattered through make_figures.py and neural_experiments.py (e.g.
the split-half reliability numbers, lambda/sigma cross-task correlations,
neural_main's covariance r-values) into this script, so every number that
ends up in the paper has a single, findable source. Not started -- see
docs/SCIENCE.md's "Future extensions" for the tracked entry. Each still-
unported group of numbers has its own report_*() stub below
(report_lambda_reliability, report_sigma_reliability,
report_sigma_dynamics, report_neural_covariance) -- each stub's docstring
names the exact make_figures.py/neural_experiments.py function(s) that
currently compute it and the paper/main.tex location the hand-copied
numbers live in, so start there instead of grepping cold.
report_lambda_distribution is the one exception -- NOT a stub, since its
numbers (median fitted lambda_, fraction with lambda_>=1) had no source
anywhere in the codebase at all (unlike every other number this file's
audit checked) and were cheap to reproduce directly from
test_lambda_fit_quality's already-computed per-participant fits.

Run:
    venv/bin/python scripts/paper_stats.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.make_figures import (
    LAMBDA_TASK_PANELS,
    LAMBDA_N_OFFSET,
    _fit_lambda_series,
    _load_lambda_delta,
    _human_data_path,
    _power_law,
)


def test_response_change_decline(task_key: str, alpha: float = 0.05) -> pd.DataFrame:
    """Per-participant test for a reliable MONOTONIC decline in |delta
    response| across observations within a trial -- the "fraction of
    participants that show significant decline in delta R" number
    reported in paper/main.tex's "Decay in magnitude of response change"
    subsection.

    Uses Spearman rank correlation (not a linear-regression slope test)
    between observation index and each participant's own mean |delta
    response| (already averaged across trials at each observation,
    matching make_figures._fit_lambda_series' own aggregation) -- a
    monotonic-trend test is the right match here since the assumed
    underlying decay is a power law (steep early, flattening out), not a
    straight line; an OLS slope test on the untransformed observation
    index is under-powered against that shape and undercounts real
    declines, confirmed directly against a similar test in an earlier
    version of this paper (see chat). Whether a model fits the decay
    better as a power law than a straight line is a separate question,
    reported alongside fitted lambda, not here.

    A decline is flagged as rho<0 with p<alpha. Participants with fewer
    than 3 observations (after this task's own LAMBDA_MIN_OBS filtering)
    are omitted, matching _fit_lambda_series' own convention, rather than
    substituted with a degenerate value.

    Returns a DataFrame indexed by pid with columns: rho, pvalue, declined.
    """
    human_delta = _load_lambda_delta(task_key, _human_data_path(task_key))
    n_offset = LAMBDA_N_OFFSET[task_key]
    rows = []
    for pid, g in human_delta.groupby("pid"):
        curve = g.groupby("observation")["delta"].mean().dropna().sort_index()
        if len(curve) < 3:
            continue
        n = curve.index.values.astype(float) + n_offset
        y = curve.values.astype(float)
        result = spearmanr(n, y)
        rows.append({
            "pid": int(pid),
            "rho": result.statistic,
            "pvalue": result.pvalue,
            "declined": bool(result.statistic < 0 and result.pvalue < alpha),
        })
    return pd.DataFrame(rows).set_index("pid")


def test_lambda_fit_quality(task_key: str, r2_threshold: float = 0.5) -> pd.DataFrame:
    """Per-participant goodness-of-fit for the power-law curve
    (A * n^-lambda) fit to |delta response| vs observation -- the SAME
    fit make_figures._fit_lambda_series performs to estimate lambda_,
    just also scoring how well it actually fits each participant's own
    curve. Flags a "good" fit as R^2 >= r2_threshold.

    r2_threshold=0.5 is a PLACEHOLDER default, not a settled choice --
    confirm the actual threshold to report in the paper before treating
    this function's output as final (could also report the R^2
    distribution directly instead of a hard threshold/fraction).

    Returns a DataFrame indexed by pid with columns: lambda_, r2, good_fit.
    """
    human_delta = _load_lambda_delta(task_key, _human_data_path(task_key))
    n_offset = LAMBDA_N_OFFSET[task_key]
    rows = []
    for pid, g in human_delta.groupby("pid"):
        curve = g.groupby("observation")["delta"].mean().dropna().sort_index()
        if len(curve) < 3:
            continue
        n = curve.index.values.astype(float) + n_offset
        y = curve.values.astype(float)
        try:
            popt, _ = curve_fit(_power_law, n, y, p0=[0.1, 0.5],
                                bounds=([0.0, 0.0], [2.0, 2.0]), maxfev=2000)
        except Exception:
            continue
        y_pred = _power_law(n, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot <= 0:
            continue
        r2 = 1.0 - ss_res / ss_tot
        rows.append({
            "pid": int(pid),
            "lambda_": popt[1],
            "r2": r2,
            "good_fit": bool(r2 >= r2_threshold),
        })
    return pd.DataFrame(rows).set_index("pid")


def report_lambda_distribution(task_key: str) -> dict:
    """Per-task median fitted discounting rate lambda_ and the fraction
    of participants with lambda_>=1 (near-optimal integration or a
    primacy bias, vs. the more common lambda_<1 recency bias) -- the
    quantitative version of the qualitative claim already in Fig.~
    lambda_main's own caption ("most individuals show...a recency
    bias...but a substantial minority show near-optimal discounting or
    even a primacy bias"), with the actual medians/percentages
    hand-copied into paper/main.tex's SI "Individual differences in
    temporal discounting" STUB instead (not the caption itself, which
    stays qualitative): medians: Binary 0.89, Continuous 0.33, Value
    0.16; fraction lambda_>=1: 39%, 7%, 0%.

    Unlike this file's four report_*() stubs above, this one is NOT a
    stub -- these numbers had no source anywhere in the codebase at all
    (not even an inline print in make_figures.py, unlike every other
    stat this session's audit checked), so there was nothing to port
    from; they were reproduced here directly from
    test_lambda_fit_quality's own already-fitted lambda_ column (SAME
    population: curve_fit succeeded, >=3 observations, no R^2/
    significance filtering beyond that), and verified this session to
    match paper/main.tex's existing numbers exactly.

    Returns a dict: n, median_lambda, frac_lambda_ge1.
    """
    fit_quality = test_lambda_fit_quality(task_key)
    return {
        "n": len(fit_quality),
        "median_lambda": float(fit_quality["lambda_"].median()),
        "frac_lambda_ge1": float((fit_quality["lambda_"] >= 1).mean()),
    }


def test_lambda_significance(task_key: str, alpha: float = 0.05) -> pd.DataFrame:
    """Per-participant test for whether the fitted power-law discounting
    rate lambda_ is significantly different from 0 (i.e. whether ANY
    reliable decay was detected), rather than scoring how much variance
    the power-law curve explains (see test_lambda_fit_quality's own R^2
    threshold, which this is an alternative to -- no arbitrary threshold
    needed here beyond the usual alpha).

    Reuses the SAME fit as test_lambda_fit_quality/_fit_lambda_series
    (A * n^-lambda, bounds=([0,0],[2,2])), but this time keeps curve_fit's
    covariance matrix (pcov) instead of discarding it, to get lambda_'s
    own standard error. Tests lambda_ != 0 via a two-sided t-test
    (df = n_observations - 2 free parameters).

    Participants whose fit lands exactly on a bound (lambda_ ~= 0 or
    ~= 2, or A ~= 0) are excluded -- curve_fit's covariance estimate is
    unreliable at a constrained optimum (the standard error can come out
    artificially small OR the covariance matrix can fail to invert
    cleanly), so neither the estimate nor its p-value can be trusted
    there. Checked directly: this excludes a small minority of
    participants per task, not a systematic chunk.

    Returns a DataFrame indexed by pid with columns: lambda_, se_lambda,
    pvalue, significant.
    """
    from scipy.stats import t as t_dist

    human_delta = _load_lambda_delta(task_key, _human_data_path(task_key))
    n_offset = LAMBDA_N_OFFSET[task_key]
    bound_eps = 1e-3
    rows = []
    for pid, g in human_delta.groupby("pid"):
        curve = g.groupby("observation")["delta"].mean().dropna().sort_index()
        if len(curve) < 3:
            continue
        n = curve.index.values.astype(float) + n_offset
        y = curve.values.astype(float)
        try:
            popt, pcov = curve_fit(_power_law, n, y, p0=[0.1, 0.5],
                                   bounds=([0.0, 0.0], [2.0, 2.0]), maxfev=2000)
        except Exception:
            continue
        A_fit, lam_fit = popt
        if (A_fit < bound_eps or lam_fit < bound_eps
                or lam_fit > 2.0 - bound_eps):
            continue
        se_lambda = np.sqrt(pcov[1, 1])
        if not np.isfinite(se_lambda) or se_lambda <= 0:
            continue
        df = len(n) - 2
        if df < 1:
            continue
        t_stat = lam_fit / se_lambda
        pvalue = 2.0 * t_dist.sf(abs(t_stat), df)
        rows.append({
            "pid": int(pid),
            "lambda_": lam_fit,
            "se_lambda": se_lambda,
            "pvalue": pvalue,
            "significant": bool(pvalue < alpha),
        })
    return pd.DataFrame(rows).set_index("pid")


def report_lambda_reliability() -> dict:
    """Within-task split-half (odd/even trial) and across-task (colors vs.
    numbers) reliability of fitted lambda_, across LAMBDA_TASK_PANELS's own
    three tasks (Binary/Continuous/Value integration -- Proportion
    Inference has no lambda fit at all, see report_lambda_distribution).
    Ported from make_figures.py's make_lambda_reliability/
    make_lambda_sigma_crosstask (_fit_lambda_split_half/
    _plot_lambda_crosstask_panel), which compute the SAME numbers inline
    for Fig.~lambda_sigma_reliability's own panels A-C; this function is
    now the single source those SI numbers should match.

    Returns {task_key: {"n", "r", "p"}} for the three within-task
    split-half correlations, plus "crosstask" (colors vs. numbers, human
    only, one point per pid who completed both).
    """
    from scipy.stats import pearsonr

    from scripts.make_figures import _fit_lambda_split_half

    out = {}
    for task_key, _ in LAMBDA_TASK_PANELS:
        wide = _fit_lambda_split_half(task_key, _human_data_path(task_key))
        if len(wide) < 3:
            out[task_key] = {"n": len(wide), "r": float("nan"), "p": float("nan")}
            continue
        r, p = pearsonr(wide["odd"], wide["even"])
        out[task_key] = {"n": len(wide), "r": float(r), "p": float(p)}

    lam_colors = _fit_lambda_series(
        _load_lambda_delta("colors", _human_data_path("colors")), LAMBDA_N_OFFSET["colors"])
    lam_numbers = _fit_lambda_series(
        _load_lambda_delta("numbers", _human_data_path("numbers")), LAMBDA_N_OFFSET["numbers"])
    merged = pd.DataFrame({"colors": lam_colors, "numbers": lam_numbers}).dropna()
    r, p = pearsonr(merged["colors"], merged["numbers"])
    out["crosstask"] = {"n": len(merged), "r": float(r), "p": float(p)}
    return out


def report_sigma_reliability() -> dict:
    """Within-task split-half (odd/even trial) and across-task (colors vs.
    numbers) reliability of response noise (sigma), across
    VARIABILITY_TASK_PANELS's own three tasks (Proportion Inference/
    Binary/Continuous integration -- Value Integration has no repeated
    sequences at all, see make_sigma_main's own docstring). Ported from
    make_figures.py's make_sigma_reliability/make_lambda_sigma_crosstask's
    sigma panel (_fit_sigma_split_half/_plot_sigma_crosstask_panel).

    Returns a dict shaped like report_lambda_reliability's own: {task_key:
    {"n", "r", "p"}} for the three within-task correlations, plus
    "crosstask" (colors vs. numbers, human only).
    """
    from scipy.stats import pearsonr

    from scripts.make_figures import (
        _fit_sigma_split_half,
        _variability_qid_map,
        _variability_series,
    )

    out = {}
    for task_key in ["balls", "colors", "numbers"]:
        qid_map, prefix = _variability_qid_map(task_key)
        wide = _fit_sigma_split_half(task_key, _human_data_path(task_key), qid_map, prefix)
        if len(wide) < 3:
            out[task_key] = {"n": len(wide), "r": float("nan"), "p": float("nan")}
            continue
        r, p = pearsonr(wide["odd"], wide["even"])
        out[task_key] = {"n": len(wide), "r": float(r), "p": float(p)}

    qid_map_c, prefix_c = _variability_qid_map("colors")
    qid_map_n, prefix_n = _variability_qid_map("numbers")
    sigma_colors = _variability_series("colors", _human_data_path("colors"), qid_map_c, prefix_c)
    sigma_numbers = _variability_series("numbers", _human_data_path("numbers"), qid_map_n, prefix_n)
    merged = pd.DataFrame({"colors": sigma_colors, "numbers": sigma_numbers}).dropna()
    r, p = pearsonr(merged["colors"], merged["numbers"])
    out["crosstask"] = {"n": len(merged), "r": float(r), "p": float(p)}
    return out


def report_sigma_dynamics() -> dict:
    """Growth of sigma across observations within a repeated sequence
    (Fig.~sigma_main D--F) and the lag-1 autocorrelation of each
    participant's residual from their own typical response (Fig.~
    sigma_main G--I), for Human, each of the four noise-augmented math
    models, and SNN, across all three RESID_TASK_PANELS tasks (Proportion
    Inference/Binary/Continuous Integration). Ported from make_figures.py's
    make_sigma_main row 2/3 loaders (_load_variance_growth_data/
    _load_variance_autocorr_data), called with the EXACT SAME arguments
    that figure itself uses, so this is guaranteed to match what's
    plotted.

    Growth is reported as the last-observation/first-observation ratio of
    each source's own (unnormalized) per-observation mean residual SD --
    equivalent to the normalized-curve's own endpoint, since normalizing
    by the first observation's value and then reading the last point is
    the same ratio. Autocorrelation is reported at lag 1 (the shortest lag
    common to all three tasks' own RESID_LAGS).

    Returns {task_key: {"growth": {source: ratio}, "autocorr_lag1":
    {source: r}}}, source keys being "Human", each math model's own name,
    and "SNN".
    """
    from scripts.make_figures import (
        NLL_RESP_NOISE_MODELS,
        RESID_TASK_PANELS,
        _load_variance_autocorr_data,
        _load_variance_growth_data,
        _nll_resp_noise_responses_path,
    )

    growth_data = _load_variance_growth_data(
        models=NLL_RESP_NOISE_MODELS, responses_path_fn=_nll_resp_noise_responses_path)
    autocorr_data = _load_variance_autocorr_data(
        models=NLL_RESP_NOISE_MODELS, responses_path_fn=_nll_resp_noise_responses_path,
        include_nef=True)

    out = {}
    for task_key, _ in RESID_TASK_PANELS:
        human_stats, model_stats, nef_stats = growth_data[task_key]
        growth = {}
        h = human_stats.sort_values("observation")
        growth["Human"] = float(h["mean"].iloc[-1] / h["mean"].iloc[0])
        for m, s in model_stats.items():
            s = s.sort_values("observation")
            growth[m] = float(s["mean"].iloc[-1] / s["mean"].iloc[0])
        if nef_stats is not None:
            n = nef_stats.sort_values("observation")
            growth["SNN"] = float(n["mean"].iloc[-1] / n["mean"].iloc[0])

        human_res, model_results, lags = autocorr_data[task_key]
        autocorr = {"Human": float(human_res[1][0])}
        for m, res in model_results.items():
            source = "SNN" if m == "NEF" else m
            autocorr[source] = float(res[1][0])

        out[task_key] = {"growth": growth, "autocorr_lag1": autocorr}
    return out


def report_neural_covariance() -> None:
    """STUB -- not yet implemented here. The three neural_main experiments'
    parameter-vs-outcome r-values are currently computed inline via
    scipy.stats.pearsonr calls scattered through make_figures.py's panel
    helpers, reading pre-generated grids from neural_experiments.py's
    `outlier`/`param_scan`/`n_neurons_snr` experiments (see
    scripts/neural_experiments.py and .claude/skills/
    neural-simulation-pipeline/SKILL.md for how those grids are produced
    and generated/rechecked -- these grids ARE cluster-run/expensive to
    regenerate, unlike the other three report_* stubs' plain human/model
    response data, but once the .pkl grids already exist under
    data/runs/neural_experiments/, recomputing these pearsonr values from
    them locally is cheap -- confirmed directly by rerunning
    `venv/bin/python scripts/make_figures.py neural_main --mode paper`,
    which reads the cached grids and reprints every r/p value below
    without resimulating anything):
      Row 1 (alpha_0, outlier experiment) -- _plot_outlier_param_effect
        (peak |PE| vs. alpha_0; PE decay rate vs. alpha_0) and
        _plot_outlier_dv_scatter (peak |PE| vs. decay rate, direct).
        Values already reported in paper/main.tex's Results (Section
        2.6, first paragraph): peak r=0.91, decay rate r=0.95, direct
        covariance r=0.93, all p<0.0001.
      Row 2 (lambda_, param_scan experiment) -- _plot_neural_main_decay_vs_param
        (activity decay % vs. lambda_; deltaR decay % vs. lambda_) and
        _plot_param_scan_dv_scatter (activity decay % vs. deltaR decay
        %, direct). Values reported in paper/main.tex's Results (Section
        2.6, second paragraph): activity decay r=0.89, response-change
        decay r=0.92, direct covariance r=0.86, all p<0.0001.
      Row 3 (n_neurons, n_neurons_snr experiment) -- _plot_n_neurons_snr_pair
        (PE noise CV% vs. n_neurons; sigma vs. n_neurons) and
        _plot_n_neurons_snr_dv_scatter (PE noise CV% vs. sigma, direct).
        Values reported in paper/main.tex's Results (Section 2.6, third
        paragraph): PE noise r=-0.82, sigma r=-0.86, direct covariance
        r=0.82, all p<0.0001 (rerun with cluster_spread=1.0, matching the
        outlier experiment's own trial design -- an earlier version of
        this grid used cluster_spread=15.0, wider than the outlier
        deviation itself, which the M&M text now avoids describing as an
        "outlier" trial; see chat).
    Porting this here (even just row 1, which IS already in the text) is
    tracked in docs/SCIENCE.md's "Future extensions"; not started.
    """
    raise NotImplementedError(
        "port make_neural_main's per-row pearsonr computations here -- "
        "see this function's docstring")


def report_decoder_free_reliability(task: str = "soltani_numbers") -> dict:
    """Decoder-free corroboration of neural_main's own row-3 (n_neurons)
    finding (Supplementary Text's "A decoder-free corroboration of the
    neurons-vs-SNR result"): split-half spike-count reliability
    (split_half_r_mean -- no decoded PE/value signal involved, unlike the
    main text's own PE-noise CV%) vs. n_neurons, sigma vs. n_neurons (same
    grid, restated here for convenience), and the direct covariance between
    split-half reliability and sigma. All three read the SAME
    n_neurons_snr grid make_figures.py's own
    _plot_n_neurons_splithalf_pair/_plot_n_neurons_splithalf_dv_scatter
    plot (one point per (cluster_center, outlier_deviation, n_neurons)
    cell, 50 cells total for the production grid).

    Returns {"n_cells": int, "by_n_neurons": {n_neurons: mean split-half r},
    "split_half_vs_n_neurons": {"r", "p"}, "sigma_vs_n_neurons": {"r", "p"},
    "split_half_vs_sigma": {"r", "p"}}.
    """
    from scripts.make_figures import NEURAL_EXP_DIR

    path = NEURAL_EXP_DIR / f"n_neurons_snr_{task}.pkl"
    grid = pd.read_pickle(path)["grid"].copy()
    grid["response_std"] = np.sqrt(grid["response_variance"])

    by_n = grid.groupby("n_neurons")["split_half_r_mean"].mean().to_dict()

    r1, p1 = pearsonr(grid["n_neurons"], grid["split_half_r_mean"])
    r2, p2 = pearsonr(grid["n_neurons"], grid["response_std"])
    r3, p3 = pearsonr(grid["split_half_r_mean"], grid["response_std"])
    return {
        "n_cells": len(grid),
        "by_n_neurons": {int(k): float(v) for k, v in by_n.items()},
        "split_half_vs_n_neurons": {"r": float(r1), "p": float(p1)},
        "sigma_vs_n_neurons": {"r": float(r2), "p": float(p2)},
        "split_half_vs_sigma": {"r": float(r3), "p": float(p3)},
    }


def main() -> None:
    for task_key, title in LAMBDA_TASK_PANELS:
        decline = test_response_change_decline(task_key)
        n_declined = int(decline["declined"].sum())
        n_total = len(decline)
        print(f"{title:24s} {n_declined:3d}/{n_total:3d} participants "
              f"({100 * n_declined / n_total:.0f}%) show a significant "
              f"monotonic decline in |delta response|")

        fit_quality = test_lambda_fit_quality(task_key)
        n_good = int(fit_quality["good_fit"].sum())
        n_fit = len(fit_quality)
        print(f"{'':24s} {n_good:3d}/{n_fit:3d} participants "
              f"({100 * n_good / n_fit:.0f}%) have a power-law fit with "
              f"R^2 >= 0.5")

        significance = test_lambda_significance(task_key)
        n_sig = int(significance["significant"].sum())
        n_lam = len(significance)
        print(f"{'':24s} {n_sig:3d}/{n_lam:3d} participants "
              f"({100 * n_sig / n_lam:.0f}%) have lambda significantly "
              f"different from 0 (p<0.05)")

        dist = report_lambda_distribution(task_key)
        print(f"{'':24s} median lambda={dist['median_lambda']:.2f}, "
              f"{100 * dist['frac_lambda_ge1']:.0f}% have lambda>=1 "
              f"(n={dist['n']})")

    print()
    lam_rel = report_lambda_reliability()
    for task_key, _ in LAMBDA_TASK_PANELS:
        r = lam_rel[task_key]
        print(f"lambda split-half   {task_key:8s} r={r['r']:.2f} p={r['p']:.2g} (n={r['n']})")
    ct = lam_rel["crosstask"]
    print(f"lambda cross-task    colors-numbers r={ct['r']:.2f} p={ct['p']:.2g} (n={ct['n']})")

    print()
    sig_rel = report_sigma_reliability()
    for task_key in ["balls", "colors", "numbers"]:
        r = sig_rel[task_key]
        print(f"sigma split-half     {task_key:8s} r={r['r']:.2f} p={r['p']:.2g} (n={r['n']})")
    ct = sig_rel["crosstask"]
    print(f"sigma cross-task     colors-numbers r={ct['r']:.2f} p={ct['p']:.2g} (n={ct['n']})")

    print()
    dynamics = report_sigma_dynamics()
    for task_key, d in dynamics.items():
        g = d["growth"]
        print(f"sigma growth (last/first)  {task_key:8s} " +
              " ".join(f"{k}={v:.2f}" for k, v in g.items()))
    for task_key, d in dynamics.items():
        a = d["autocorr_lag1"]
        print(f"sigma autocorr lag-1       {task_key:8s} " +
              " ".join(f"{k}={v:.3f}" for k, v in a.items()))

    print()
    df = report_decoder_free_reliability()
    print(f"decoder-free split-half r by n_neurons (n={df['n_cells']} cells): " +
          " ".join(f"{k}={v:.2f}" for k, v in sorted(df["by_n_neurons"].items())))
    for label, key in [("split-half r vs n_neurons", "split_half_vs_n_neurons"),
                       ("sigma vs n_neurons", "sigma_vs_n_neurons"),
                       ("split-half r vs sigma", "split_half_vs_sigma")]:
        r = df[key]
        print(f"  {label:26s} r={r['r']:.3f} p={r['p']:.2g}")


if __name__ == "__main__":
    main()
