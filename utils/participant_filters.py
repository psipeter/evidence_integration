"""
utils/participant_filters.py

Quantifiable, adjustable exclusion criteria for identifying participants who
show no evidence of genuinely attempting the soltani numbers or colors
estimation task, as opposed to participants who attempt the task but update
sub-optimally (whom we explicitly do NOT want to exclude).

All criteria are computed per (prolific_pid, task) — a participant who did
both tasks can be excluded from one without being excluded from the other.

CRITERION 1 — NO_INTEGRATION
-----------------------------
Response is (nearly) identical to the just-observed stimulus on almost
every trial, i.e. they are reporting the raw stimulus value rather than
any running estimate. Found by direct inspection in this pilot: one
numbers participant matched the displayed value on 98.75% of rows
(e.g. value=63 -> response=62, value=73 -> response=72). Colors values are
mapped onto the response's [0,100] percent scale (100 for blue/+1, 0 for
red/-1) so the same tolerance-based check applies to both tasks.

CRITERIA 2 & 3 — NON-CONTINGENT UPDATING (sign / magnitude)
-------------------------------------------------------------
Participants ARE updating their response every trial — the failure mode
here isn't a lack of updating, it's updating in a way that shows no
reliable relationship to (isn't "contingent on") the evidence just shown.
Two independent, separately-flaggable sub-patterns:

  - flag_noncontingent_sign: does the response move toward the new
    observation relative to the participant's own previous response, more
    often than chance (binomial test)? If not, the DIRECTION of their
    updates carries no information about the stimulus.
  - flag_noncontingent_magnitude: does |update| correlate with
    |discrepancy| (bigger surprises -> bigger movements)? If not, the SIZE
    of their updates carries no information about the stimulus either,
    even if direction sometimes happens to line up.

These are deliberately kept as two SEPARATE filters (each can exclude on
its own) rather than bundled into one "both must fail" criterion, so each
failure mode is independently visible and independently adjustable. In
this pilot the one clear case (one colors-task participant) happened to
fail both simultaneously (p_direction=0.96, p_magnitude=0.56), so the
distinction doesn't change who gets excluded yet — but it will matter as
more pilot data comes in and these can diverge.

Neither non-contingent-updating filter flags participants who update in a
directionally/magnitude-sensitive but non-Bayesian way (e.g. a constant
learning rate instead of properly shrinking it over the trial) — that's
the pattern most participants in this pilot actually show, and it's
exactly the kind of "imperfect but genuine effort" behavior we do NOT want
to exclude.

All three criteria catch different failure modes and are combined with OR
at the top level: the copier from criterion 1 actually PASSES both
non-contingent-updating checks (their updates track the stimulus
perfectly, since response==value exactly), so criterion 1 is the only
thing that catches them.

All criteria are intentionally conservative and meant to catch "no
apparent effort" participants, not "imperfect" ones. All thresholds are
exposed as parameters and expected to be revisited as more pilot data
comes in and more failure patterns are discovered — see
figure_soltani_temporal.py and the investigation that motivated this
module for the diagnostics these thresholds were calibrated against.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_COPY_TOLERANCE      = 2.0   # |response - value_scale| considered "identical"
DEFAULT_COPY_FRAC_THRESHOLD = 0.80  # fraction of observations that must be "identical"
DEFAULT_ALPHA               = 0.05  # significance threshold for each noncontingent-* test
DEFAULT_MIN_UPDATES         = 10    # below this, we don't have enough data to judge


def _assert_single_task(df: pd.DataFrame) -> None:
    tasks = df["task"].unique()
    if len(tasks) != 1:
        raise ValueError(
            f"Expected a single-task dataframe, got tasks={list(tasks)}. "
            "Filter to one task BEFORE calling into participant_filters — "
            "task and (trial, observation) numbering overlap across tasks, "
            "so mixing tasks before any dedup/groupby silently corrupts "
            "results for participants who did both (this bit us once "
            "already; see the conversation that produced this module)."
        )


def _dedup(df_task: pd.DataFrame) -> pd.DataFrame:
    _assert_single_task(df_task)
    return (df_task[df_task["timed_out"] == False]
            .drop_duplicates(subset=["prolific_pid", "trial", "observation"]))


def _value_on_response_scale(df_task: pd.DataFrame, task: str) -> pd.Series:
    """Map the raw stimulus `value` onto the same [0,100] scale as `response`.
    numbers: value is already on that scale. colors: value in {-1,+1}
    (red/blue) -> {0, 100} (this treats a single binary draw like an
    "extreme" observation on the percent-blue scale, consistent with how
    figure_soltani_temporal.py's direction/magnitude diagnostics defined it)."""
    if task == "colors":
        return np.where(df_task["value"] == 1, 100.0, 0.0)
    return df_task["value"].astype(float)


def _compute_updates(df_task: pd.DataFrame, task: str) -> pd.DataFrame:
    """One row per (prolific_pid, trial, observation>=1): the response
    update and the discrepancy it should track (new evidence relative to
    the participant's OWN previous response), in long form across pids."""
    sub = _dedup(df_task).copy()
    sub["value_scale"] = _value_on_response_scale(sub, task)
    rows = []
    for pid, pg in sub.groupby("prolific_pid"):
        for trial, g in pg.groupby("trial"):
            g = g.sort_values("observation")
            resp = g["response"].to_numpy()
            val_scale = g["value_scale"].to_numpy()
            obs = g["observation"].to_numpy()
            for i in range(1, len(resp)):
                rows.append({
                    "prolific_pid": pid, "trial": trial, "observation": obs[i],
                    "update": resp[i] - resp[i - 1],
                    "discrepancy": val_scale[i] - resp[i - 1],
                })
    return pd.DataFrame(rows, columns=["prolific_pid", "trial", "observation",
                                       "update", "discrepancy"])


# ── Criterion 1: NO_INTEGRATION (copying the stimulus) ──────────────────────

def flag_no_integration(df_task: pd.DataFrame, task: str,
                        tolerance: float = DEFAULT_COPY_TOLERANCE,
                        frac_threshold: float = DEFAULT_COPY_FRAC_THRESHOLD
                        ) -> pd.DataFrame:
    """Per prolific_pid: fraction of observations where response is within
    `tolerance` of the just-observed stimulus (on the [0,100] scale).
    Flagged if that fraction >= frac_threshold."""
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
    correlation of |update| vs |discrepancy|. Kept as one helper so the two
    filters below don't duplicate the underlying computation, even though
    they flag independently on their own single test."""
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
    reach significance (p_direction > alpha) — direction carries no
    information about the stimulus."""
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
    alpha, or undefined) — update size carries no information about how
    surprising the evidence was."""
    stats = _direction_and_magnitude_stats(df_task, task, min_updates)
    stats["flagged_noncontingent_magnitude"] = (
        stats["p_magnitude"].isna() | (stats["p_magnitude"] > alpha)
    ) & stats["n_updates"].ge(min_updates)
    return stats[["prolific_pid", "task", "n_updates", "r_magnitude",
                 "p_magnitude", "flagged_noncontingent_magnitude"]]


# ── Combined report + filtering ─────────────────────────────────────────────

def compute_exclusion_report(df: pd.DataFrame, tasks: tuple[str, ...] = ("colors", "numbers"),
                             copy_tolerance: float = DEFAULT_COPY_TOLERANCE,
                             copy_frac_threshold: float = DEFAULT_COPY_FRAC_THRESHOLD,
                             alpha: float = DEFAULT_ALPHA,
                             min_updates: int = DEFAULT_MIN_UPDATES) -> pd.DataFrame:
    """Full diagnostic report, one row per (prolific_pid, task) present in
    `df`. `df` may span multiple tasks — this function does the required
    per-task slicing before calling into any criterion itself."""
    reports = []
    for task in tasks:
        df_task = df[df["task"] == task]
        if df_task.empty:
            continue
        r1 = flag_no_integration(df_task, task, copy_tolerance, copy_frac_threshold)
        r2 = flag_noncontingent_sign(df_task, task, alpha, min_updates)
        r3 = flag_noncontingent_magnitude(df_task, task, alpha, min_updates).drop(columns=["n_updates"])
        merged = r1.merge(r2, on=["prolific_pid", "task"], how="outer")
        merged = merged.merge(r3, on=["prolific_pid", "task"], how="outer")
        reports.append(merged)

    if not reports:
        return pd.DataFrame(columns=["prolific_pid", "task", "frac_copy",
                                     "flagged_no_integration", "n_updates",
                                     "frac_correct_direction", "p_direction",
                                     "flagged_noncontingent_sign", "r_magnitude",
                                     "p_magnitude", "flagged_noncontingent_magnitude",
                                     "excluded"])

    report = pd.concat(reports, ignore_index=True)
    report["excluded"] = (report["flagged_no_integration"].fillna(False) |
                          report["flagged_noncontingent_sign"].fillna(False) |
                          report["flagged_noncontingent_magnitude"].fillna(False))
    return report


def excluded_pairs(df: pd.DataFrame, **kwargs) -> set[tuple[str, str]]:
    """Set of (prolific_pid, task) pairs that fail any exclusion criterion."""
    report = compute_exclusion_report(df, **kwargs)
    excl = report[report["excluded"]]
    return set(zip(excl["prolific_pid"], excl["task"]))


def filter_participants(df: pd.DataFrame, report: pd.DataFrame | None = None,
                        verbose: bool = True, **kwargs) -> pd.DataFrame:
    """Return `df` with rows belonging to any excluded (prolific_pid, task)
    pair removed. A participant excluded from one task keeps their data
    from the other task, if they did both and only failed one.

    Pass a pre-computed `report` (from compute_exclusion_report) to reuse
    it and avoid recomputing; otherwise one is computed from `df` using
    **kwargs (forwarded to compute_exclusion_report)."""
    if report is None:
        report = compute_exclusion_report(df, **kwargs)

    excl = report[report["excluded"]]
    excl_pairs = set(zip(excl["prolific_pid"], excl["task"]))

    if verbose:
        if excl_pairs:
            print(f"Excluding {len(excl_pairs)} (prolific_pid, task) pair(s):")
            for _, row in excl.iterrows():
                reasons = []
                if row.get("flagged_no_integration"):
                    reasons.append(f"no_integration (frac_copy={row['frac_copy']:.2f})")
                if row.get("flagged_noncontingent_sign"):
                    reasons.append(f"noncontingent_sign (p_direction={row['p_direction']:.2f})")
                if row.get("flagged_noncontingent_magnitude"):
                    reasons.append(f"noncontingent_magnitude (p_magnitude={row['p_magnitude']:.2f})")
                print(f"  {row['prolific_pid']} / {row['task']}: {', '.join(reasons)}")
        else:
            print("No participants excluded.")

    keep_mask = ~df.apply(lambda r: (r["prolific_pid"], r["task"]) in excl_pairs, axis=1)
    return df[keep_mask].copy()


if __name__ == "__main__":
    import argparse
    from utils.paths import data_path

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="task_results_pilot1.pkl")
    args = parser.parse_args()

    df = pd.read_pickle(data_path(args.results_file))
    report = compute_exclusion_report(df)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 14)
    print(report.round(3).to_string(index=False))
    print()
    filter_participants(df, report=report)
