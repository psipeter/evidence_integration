"""
archive/utils/archive_participant_filters_legacy.py

The original (pre-Cohen's-f2-reframe) exclusion criteria from
utils/participant_filters.py, archived once the f2-based approach became
the default there. Nothing here still runs as part of the active
pipeline -- see utils/participant_filters.py's own module docstring for
what's actually live now, and CLAUDE.md's "Participant exclusion
criteria" section for the current, settled account.

WHY THIS WAS REPLACED, NOT JUST TUNED AGAIN
----------------------------------------------
The active criteria (below, in their original form here) were three
different statistical objects, each with its own bespoke threshold-
picking process:
  - flag_no_integration: a raw tolerance-based fraction (frac_copy >= 0.80)
  - flag_noncontingent_sign: a binomial test's significance (p > 0.05)
  - flag_noncontingent_magnitude: a raw Pearson correlation's significance
    (p > 0.05)
  - flag_recency_only: a partial Pearson correlation against a hand-picked
    effect-size cutoff (r < 0.10, motivated by a real gap found in two
    batches of data, not a citable convention on its own)

Defensible individually, but the COLLECTION reads as an ad hoc patchwork
rather than one principled measurement -- hard to answer "why these
specific tests, with these specific cutoffs, and how do you know you've
covered every way someone could be inattentive?" The PI raised exactly
this concern directly, prompting the reframe.

flag_recency_only ALSO turned out to have a real, if narrow, blind spot:
it was originally built and validated (as flag_no_integration's
replacement) using a significance test (p > alpha), then switched to the
r=0.10 effect-size cutoff seen below once the significance version proved
uselessly lenient at this task's sample size (~448 updates/pid -- nearly
any nonzero correlation is "significant" there). But r=0.10 alone missed
2 real participants that the ORIGINAL flag_no_integration (frac_copy>=0.8)
had caught (partial_r=0.12, 0.13 for two moderate-but-real copiers) --
confirmed directly against real data, not assumed. The current
f2-based flag_recency_only (utils/participant_filters.py) closes this gap
on its own, as a side effect of using Cohen's regression-specific f2=0.02
convention (~r=0.14 equivalent) rather than his bivariate-correlation
r=0.10 convention -- not because it was re-tuned to.

THE REPLACEMENT: one consistent statistical framework (nested regression
+ Cohen's f2 effect size), applied to the same three underlying questions
(recency, sign-contingency, magnitude-contingency) -- see
utils/participant_filters.py's own module docstring for the full account
and the empirical comparison that motivated switching over.

CRITERION 1 -- NO_INTEGRATION (superseded by the f2-based flag_recency_only)
------------------------------------------------------------------------------
Response is (nearly) identical to the just-observed stimulus on almost
every trial, i.e. they are reporting the raw stimulus value rather than
any running estimate. Found by direct inspection in an early pilot: one
numbers participant matched the displayed value on 98.75% of rows
(e.g. value=63 -> response=62, value=73 -> response=72).

CRITERIA 2 & 3 -- NON-CONTINGENT UPDATING (sign / magnitude), original
significance-based versions
-------------------------------------------------------------------------
  - flag_noncontingent_sign: does the response move toward the new
    observation relative to the participant's own previous response, more
    often than chance (binomial test)? Flagged if p_direction > 0.05.
  - flag_noncontingent_magnitude: does |update| correlate with
    |discrepancy|? Flagged if that correlation's p_magnitude > 0.05, or
    undefined.

CRITERION 4 -- RECENCY-ONLY UPDATING, original r=0.10 partial-correlation
version
----------------------------------------------------------------------------
Partial correlation between response and the mean of strictly-prior
values, controlling for the current value. Flagged if that partial
correlation was below 0.10 (Cohen's conventional "small effect" boundary
for a bare bivariate correlation) or undefined.

HOW TO RESTORE
----------------
Copy flag_no_integration, _direction_and_magnitude_stats,
flag_noncontingent_sign, flag_noncontingent_magnitude, flag_recency_only,
and compute_exclusion_report below back into utils/participant_filters.py
(renaming the current f2-based compute_exclusion_report to something
else first, e.g. compute_exclusion_report_f2, to avoid a name collision --
the reverse of the rename that created this archive). DEFAULT_COPY_TOLERANCE/
DEFAULT_COPY_FRAC_THRESHOLD/DEFAULT_ALPHA/DEFAULT_MIN_EFFECT_R would need
restoring too. The shared helpers these depend on
(_assert_single_task/_dedup/_value_on_response_scale/_compute_updates/
_compute_recency_features) were NOT archived -- they're still live in
utils/participant_filters.py, used by the current f2-based criteria too,
so this file imports them from there rather than duplicating them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr

from utils.participant_filters import (
    _compute_updates, _compute_recency_features,
)

DEFAULT_COPY_TOLERANCE      = 2.0   # |response - value_scale| considered "identical"
DEFAULT_COPY_FRAC_THRESHOLD = 0.80  # fraction of observations that must be "identical"
DEFAULT_ALPHA               = 0.05  # significance threshold for noncontingent-sign/magnitude
DEFAULT_MIN_EFFECT_R        = 0.10  # recency_only's original threshold on partial_r_recency
DEFAULT_MIN_UPDATES         = 10    # below this, we don't have enough data to judge


# ── Criterion 1: NO_INTEGRATION (copying the stimulus) ──────────────────────

def flag_no_integration(df_task: pd.DataFrame, task: str,
                        tolerance: float = DEFAULT_COPY_TOLERANCE,
                        frac_threshold: float = DEFAULT_COPY_FRAC_THRESHOLD
                        ) -> pd.DataFrame:
    """Per prolific_pid: fraction of observations where response is within
    `tolerance` of the just-observed stimulus (on the [0,100] scale).
    Flagged if that fraction >= frac_threshold."""
    from utils.participant_filters import _dedup, _value_on_response_scale
    sub = _dedup(df_task).copy()
    sub["value_scale"] = _value_on_response_scale(sub, task)
    sub["is_copy"] = (sub["response"] - sub["value_scale"]).abs() <= tolerance

    report = (sub.groupby("prolific_pid")["is_copy"].mean()
              .reset_index(name="frac_copy"))
    report["task"] = task
    report["flagged_no_integration"] = report["frac_copy"] >= frac_threshold
    return report[["prolific_pid", "task", "frac_copy", "flagged_no_integration"]]


# ── Criteria 2 & 3: NON-CONTINGENT UPDATING (sign / magnitude, kept separate) ─

def _direction_and_magnitude_stats(df_task: pd.DataFrame, task: str,
                                   min_updates: int = DEFAULT_MIN_UPDATES) -> pd.DataFrame:
    """Shared per-pid statistics underlying both non-contingent-updating
    filters: binomial test on update direction vs chance, and Pearson
    correlation of |update| vs |discrepancy|."""
    updates = _compute_updates(df_task, task)
    rows = []
    for pid, g in updates.groupby("prolific_pid"):
        actual_dir = np.sign(g["update"].to_numpy())
        expected_dir = np.sign(g["discrepancy"].to_numpy())
        nonzero = actual_dir != 0
        n_match = int((actual_dir[nonzero] == expected_dir[nonzero]).sum())
        n_total = int(nonzero.sum())

        if n_total < min_updates:
            rows.append({"prolific_pid": pid, "task": task, "n_updates": n_total,
                         "frac_correct_direction": np.nan, "p_direction": np.nan,
                         "r_magnitude": np.nan, "p_magnitude": np.nan,
                         "note": f"n_updates={n_total} < min_updates={min_updates}"})
            continue

        frac_correct = n_match / n_total
        p_direction = binomtest(n_match, n_total, 0.5).pvalue

        if g["discrepancy"].abs().std() > 1e-9 and g["update"].abs().std() > 1e-9:
            r_mag, p_mag = pearsonr(g["discrepancy"].abs(), g["update"].abs())
        else:
            r_mag, p_mag = np.nan, np.nan

        rows.append({"prolific_pid": pid, "task": task, "n_updates": n_total,
                     "frac_correct_direction": frac_correct, "p_direction": p_direction,
                     "r_magnitude": r_mag, "p_magnitude": p_mag, "note": ""})
    return pd.DataFrame(rows)


def flag_noncontingent_sign(df_task: pd.DataFrame, task: str,
                            alpha: float = DEFAULT_ALPHA,
                            min_updates: int = DEFAULT_MIN_UPDATES) -> pd.DataFrame:
    """Per prolific_pid: binomial test on whether update DIRECTION matches
    the evidence more often than chance. Flagged if that test fails to
    reach significance (p_direction > alpha)."""
    stats = _direction_and_magnitude_stats(df_task, task, min_updates)
    stats["flagged_noncontingent_sign"] = (
        stats["p_direction"] > alpha) & stats["p_direction"].notna()
    return stats[["prolific_pid", "task", "n_updates", "frac_correct_direction",
                 "p_direction", "flagged_noncontingent_sign"]]


def flag_noncontingent_magnitude(df_task: pd.DataFrame, task: str,
                                 alpha: float = DEFAULT_ALPHA,
                                 min_updates: int = DEFAULT_MIN_UPDATES) -> pd.DataFrame:
    """Per prolific_pid: Pearson correlation of |update| vs |discrepancy|.
    Flagged if that correlation fails to reach significance (p_magnitude >
    alpha, or undefined)."""
    stats = _direction_and_magnitude_stats(df_task, task, min_updates)
    stats["flagged_noncontingent_magnitude"] = (
        stats["p_magnitude"].isna() | (stats["p_magnitude"] > alpha)
    ) & stats["n_updates"].ge(min_updates)
    return stats[["prolific_pid", "task", "n_updates", "r_magnitude",
                 "p_magnitude", "flagged_noncontingent_magnitude"]]


# ── Criterion 4: RECENCY-ONLY UPDATING (original r=0.10 version) ────────────

def flag_recency_only(df_task: pd.DataFrame, task: str,
                      min_effect_r: float = DEFAULT_MIN_EFFECT_R,
                      min_updates: int = DEFAULT_MIN_UPDATES) -> pd.DataFrame:
    """Per prolific_pid: partial correlation of response & prior_mean,
    controlling for current_value. Flagged if that partial correlation is
    below `min_effect_r` (or undefined)."""
    feats = _compute_recency_features(df_task, task)
    rows = []
    for pid, g in feats.groupby("prolific_pid"):
        n = len(g)
        if n < min_updates:
            rows.append({"prolific_pid": pid, "task": task, "n_updates": n,
                        "partial_r_recency": np.nan, "p_recency": np.nan,
                        "flagged_recency_only": False,
                        "note": f"n_updates={n} < min_updates={min_updates}"})
            continue

        cur = g["current_value"].to_numpy()
        prior = g["prior_mean"].to_numpy()
        resp = g["response"].to_numpy()

        if np.std(cur) < 1e-9:
            rows.append({"prolific_pid": pid, "task": task, "n_updates": n,
                        "partial_r_recency": np.nan, "p_recency": np.nan,
                        "flagged_recency_only": False,
                        "note": "current_value has no variance -- can't compute a partial correlation"})
            continue

        resp_resid  = resp  - np.polyval(np.polyfit(cur, resp, 1), cur)
        prior_resid = prior - np.polyval(np.polyfit(cur, prior, 1), cur)

        if np.std(resp_resid) > 1e-9 and np.std(prior_resid) > 1e-9:
            partial_r, p_recency = pearsonr(prior_resid, resp_resid)
        else:
            partial_r, p_recency = np.nan, np.nan

        flagged = bool(np.isnan(partial_r) or partial_r < min_effect_r)
        rows.append({"prolific_pid": pid, "task": task, "n_updates": n,
                    "partial_r_recency": partial_r, "p_recency": p_recency,
                    "flagged_recency_only": flagged, "note": ""})
    return pd.DataFrame(rows)


# ── Combined report (legacy) ────────────────────────────────────────────────

def compute_exclusion_report(df: pd.DataFrame, tasks: tuple[str, ...] = ("colors", "numbers"),
                             alpha: float = DEFAULT_ALPHA,
                             min_updates: int = DEFAULT_MIN_UPDATES,
                             min_effect_r: float = DEFAULT_MIN_EFFECT_R) -> pd.DataFrame:
    """Full diagnostic report, one row per (prolific_pid, task) present in
    `df`, using the ORIGINAL (pre-f2) criteria."""
    reports = []
    for task in tasks:
        df_task = df[df["task"] == task]
        if df_task.empty:
            continue
        r1 = flag_recency_only(df_task, task, min_effect_r, min_updates)
        r2 = flag_noncontingent_sign(df_task, task, alpha, min_updates).drop(columns=["n_updates"])
        r3 = flag_noncontingent_magnitude(df_task, task, alpha, min_updates).drop(columns=["n_updates"])
        merged = r1.merge(r2, on=["prolific_pid", "task"], how="outer")
        merged = merged.merge(r3, on=["prolific_pid", "task"], how="outer")
        reports.append(merged)

    if not reports:
        return pd.DataFrame(columns=["prolific_pid", "task", "n_updates",
                                     "partial_r_recency", "p_recency",
                                     "flagged_recency_only",
                                     "frac_correct_direction", "p_direction",
                                     "flagged_noncontingent_sign", "r_magnitude",
                                     "p_magnitude", "flagged_noncontingent_magnitude",
                                     "excluded"])

    report = pd.concat(reports, ignore_index=True)
    report["excluded"] = (report["flagged_recency_only"].fillna(False) |
                          report["flagged_noncontingent_sign"].fillna(False) |
                          report["flagged_noncontingent_magnitude"].fillna(False))
    return report
