"""archive/utils/archive_exclusion_criteria.py

SUPERSEDED participant-exclusion criteria, kept for provenance only. Do NOT
import from live code -- utils/participant_filters.py holds the criteria in use.

Both were built and tested during the exclusion-criteria investigation, and both
were superseded by `non_integrator`. Retained because the REASONS they lose are
substantive and easy to re-derive by accident:

  flag_performance_outlier ('performance')
    carrabin's own rule: mean absolute error against the true generative
    parameter, more than N SDs above the retained group's mean. Model-free and
    matches published practice (carrabin excluded 4/25 = 16% this way), but it is
    ACCURACY-based, so it cannot distinguish an inaccurate integrator from a
    non-integrator -- the exact confusion the final criterion had to avoid.
    Also: carrabin's literal >6 SD threshold excludes ZERO participants here,
    because our error distribution is continuous where theirs had a 6-SD gap.

  flag_no_integration ('integration')
    a skill score, 1 - err_participant/err_copy_latest, targeting the running
    mean. Its threshold was defensible (a 0.29-wide empirical void in the score
    distribution), but the METRIC is not monotone in integration depth. Measured
    on synthetic leaky integrators, which genuinely integrate all history:
        alpha 0.10 (near-optimal)  skill +0.603
        alpha 0.20 (mild recency)  skill +0.745   <- PEAK, above near-optimal
        alpha 0.35                 skill +0.660
        alpha 0.70 + noise         skill +0.115   <- a hair above its own 0.10
                                                     threshold
    With 15 observations, mild recency overweighting tracks the running mean
    BETTER than a sluggish filter does, so the score conflates integration depth
    with proximity to the target. It discards inaccurate integrators.

See docs/HISTORY.md ("Participant exclusion criteria: four candidates") for the
full progression, including the candidates that were measured and rejected before
reaching an operationalisation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# These helpers live in utils/participant_filters.py; imported here so the
# archived criteria remain runnable for provenance checks.
from utils.participant_filters import (  # noqa: F401
    DEFAULT_MIN_UPDATES,
    _assert_single_task,
    _dedup,
    _value_on_response_scale,
)


# ── Criterion set 2: GROSS PERFORMANCE OUTLIER (model-free) ─────────────────
#
# The f2 criteria above test whether updating is CONTINGENT on prediction error,
# which is mechanistically upstream of what the temporal figures measure (error
# decay, |delta response| decay, residual variance growth). That makes them open
# to the charge of excluding participants who "didn't fit the pattern we wanted"
# rather than ones who didn't follow the instructions -- and on complete_pairs
# they exclude 25/60 (42%) for numbers and 19/60 (32%) for colors, 33/60 (55%) as
# a union.
#
# For comparison, the two published datasets this project fits:
#   carrabin (Prat-Carrabin 2024): excluded 4/25 (16%) on ONE model-free quantity
#     -- mean absolute error against the true generative parameter. The excluded
#     group averaged .263 (SD .0298) vs .176 (SD .0132) for the retained 21, a
#     separation of more than 6 SDs.
#   yoo: excluded 8/46 (17%), of which SEVEN were fMRI-technical (1 structural
#     abnormality, 6 head motion >3mm) and exactly ONE was behavioural -- a
#     post-experiment questionnaire in which the subject said they tracked
#     pairwise differences rather than the average. Behavioural rate ~2%.
#
# Neither used a model-based contingency test, and both are 2-4x more
# conservative than ours. This criterion set implements carrabin's rule.
#
# NOTE our error distribution is CONTINUOUS where carrabin's had a 6-SD gap, so
# carrabin's literal >6 SD threshold excludes ZERO participants here. Measured on
# complete_pairs: >3 SD excludes 6 (10%) numbers / 1 (2%) colors; >2 SD excludes
# 9 (15%) each; the single largest gap is one participant in each task. Default
# is 2 SD. The two criterion sets largely agree on WHO is worst -- 19 of the 25
# f2-excluded numbers pids are among the 25 worst by absolute error (16 of 19 for
# colors) -- so this draws the same line much further up the same distribution.

DEFAULT_MAX_ERROR_SD = 2.0   # SDs above the retained group's mean


def flag_performance_outlier(df_task: pd.DataFrame, task: str,
                             max_error_sd: float = DEFAULT_MAX_ERROR_SD) -> pd.DataFrame:
    """Flag participants whose mean absolute error against the TRUE generative
    parameter sits more than `max_error_sd` SDs above the mean of the retained
    group. Model-free: references no decay/integration model, only the task's own
    stated objective.

    Applied iteratively from the worst participant down, recomputing the retained
    group's mean and SD each time, so a single extreme participant cannot inflate
    the SD and thereby shield the next one. Stops at the first participant who
    falls inside the threshold.

    `df_task` must be single-task and carry the truth column for that task
    (true_mean for numbers, true_p for colors); `response` and the truth are
    compared on the NATIVE response scale, as carrabin's rule does.
    """
    _assert_single_task(df_task)
    d = _dedup(df_task).copy()

    if task == "colors":
        if "true_p" not in d.columns:
            raise KeyError("flag_performance_outlier needs true_p for colors")
        truth = pd.to_numeric(d["true_p"], errors="coerce") * 100.0
    else:
        if "true_mean" not in d.columns:
            raise KeyError("flag_performance_outlier needs true_mean for numbers")
        truth = pd.to_numeric(d["true_mean"], errors="coerce")

    d["abs_err"] = (pd.to_numeric(d["response"], errors="coerce") - truth).abs()
    per = (d.dropna(subset=["abs_err"])
           .groupby("prolific_pid")["abs_err"].mean().sort_values())

    flagged: set = set()
    vals = per.values.copy()
    names = list(per.index)
    i = len(vals) - 1
    while i > 1:
        rest = vals[:i]
        sd = rest.std(ddof=1)
        if sd > 0 and vals[i] > rest.mean() + max_error_sd * sd:
            flagged.add(names[i])
            i -= 1
        else:
            break

    return pd.DataFrame({
        "prolific_pid": names,
        "task": task,
        "mean_abs_error": per.values,
        "flagged_performance_outlier": [n in flagged for n in names],
    })


# ── Combined report + filtering ─────────────────────────────────────────────

# ── Criterion set 3: NO EVIDENCE OF INTEGRATION (model-free) ────────────────
#
# Named for what the participant fails to DO -- integrate evidence beyond the most
# recent observation -- rather than for the mechanism used to detect it. An earlier
# name, 'baseline', described only the mechanism (compare against a baseline
# strategy) and was uninformative: `performance` also compares against a baseline,
# just a group-level one.
#
# IMPORTANT RELATIONSHIP TO `recency_only`: this criterion and the f2-based
# recency_only criterion measure THE SAME CONSTRUCT ("uses only the most recent
# observation") by two different methods -- incremental regression variance from
# adding prior_mean over current_value, versus error against a copy baseline. That
# is almost certainly WHY they agree on 23 of 25 excluded numbers participants and
# 18 of 19 colors. Present that agreement as two operationalisations of one
# construct converging, NOT as two independent lines of evidence.
#
# The task instruction is "report the mean of ALL observations so far". Two
# reference agents can be evaluated on each participant's OWN sequences:
#
#   optimal   report the running mean            -> lower bound on achievable error
#   last      report only the latest observation -> what "not averaging" achieves
#
# skill = (err_last - err_pid) / (err_last - err_optimal)
#
#   skill >= 1  at the optimal running-mean level
#   skill ~  0  no better than reporting only the most recent observation
#   skill <  0  WORSE than that -- cannot be following the instruction at all
#
# NOTE skill normalises by (err_last - err_optimal) computed on the participant's
# OWN sequences, which is what makes it robust to the objection that copying is
# nearly correct at low true_std: if copying is nearly correct then err_last
# shrinks and the yardstick shrinks with it. Measured headroom on complete_pairs:
# numbers err_last 8.35 vs err_optimal 3.28 (gap 5.07, and at least 3.99 for
# EVERY participant); colors 39.35 vs 11.75 (gap 27.60). So there is real room to
# discriminate even at std=10. Prefer this over a raw "fraction of responses
# matching the displayed value", which IS confounded by true_std.
#
# ONE baseline, deliberately: "report the most recent observation" is the single
# strategy the instruction most directly rules out ("the mean of ALL observations
# so far"). An earlier version also tried a constant-midpoint baseline and used
# whichever bound tighter per task, which made the criterion's MEANING
# task-dependent -- for colors the midpoint bound and for numbers the last value.
# Removed for interpretability: one criterion, one sentence, same meaning in both
# tasks.
#
# CONSEQUENCE, worth knowing: for colors this baseline is near-VACUOUS on its own.
# Reporting the last binary draw means slamming to 0% or 100% every trial (error
# 39.35 vs optimal 11.75), so almost any behaviour beats it and skill < 0 excludes
# essentially nobody. Colors exclusions therefore come almost entirely from the
# both-tasks requirement propagating numbers' exclusions -- see
# `require_both_tasks` in filter_participants.
#
# WHY THIS RATHER THAN "no improvement with more observations". That would be the
# natural way to phrase an instruction-compliance test, but error-decay-with-
# observation is exactly what temporal col 1 plots, and |delta response| decay is
# col 2 -- excluding on the shape and then reporting the shape makes those panels
# partly definitional. This is a LEVEL comparison against a NAMED alternative
# strategy, so it shares no quantity with those panels.
#
# THE THRESHOLD IS READ OFF AN EMPIRICAL DISCONTINUITY, not tuned. Two candidate
# anchors, and the second is stronger:
#
#   skill < 0     "did not beat a strategy the instructions rule out". Principled,
#                 but LENIENT: a pure copier scores exactly 0 and is RETAINED, so
#                 it excludes only participants worse than copying. Excludes 31/60
#                 numbers, 0/60 colors.
#   skill < 0.10  "moved at least slightly toward the true mean, relative to
#                 copying" -- i.e. some evidence of integrating beyond the most
#                 recent observation. DEFAULT.
#
# 0.10 is not a tuned value: the numbers skill distribution has a 0.29-wide VOID
# immediately above the copying cluster. Sorted values run
#   ... -0.0031, -0.0004, 0.0102, 0.0354, 0.0387, 0.0388, 0.0406, | 0.3339, 0.3829, ...
# so ANY threshold in (0.041, 0.334) produces the IDENTICAL partition: 36/60
# numbers excluded, 24 retained after require_both_tasks. Verified across
# 0.05/0.10/0.15/0.20/0.25/0.30/0.33. The partition only starts moving at 0.35
# (37) and 0.40 (39). Colors is barely affected either way (1-2 exclusions),
# because its last-value baseline is near-vacuous -- see above.
#
# That void is itself the interesting finding: participants either barely beat
# copying (<=0.041, n=36) or clearly integrate (>=0.334, n=24), with nobody in
# between. It is the closest thing in this data to the 6-SD separation carrabin
# reported, and it is what makes a threshold here defensible rather than chosen.
#
# WHAT IT FOUND, and why it settles the "are we excluding too many?" question:
# skill < 0 excludes 31/60 (52%) for numbers and 24/60 (40%) for colors -- and it
# picks out 23 of the 25 and 18 of the 19 participants the CONTINGENCY criteria
# exclude. So an independent, model-free, non-circular criterion agrees on
# essentially the same people. The high exclusion rate is not an artefact of an
# aggressive model-based filter: roughly half of participants really do perform
# worse than ignoring all but one observation, or worse than answering "50%"
# every time. Note this is well above carrabin's 16% and yoo's 17%, but both were
# supervised lab studies (yoo paying a $35 base); an unsupervised 32-trial x
# 15-observation Prolific session is a different population.

DEFAULT_MIN_SKILL = 0.10


def flag_no_integration(df_task: pd.DataFrame, task: str,
                        min_skill: float = DEFAULT_MIN_SKILL) -> pd.DataFrame:
    """Flag participants who show no evidence of integrating beyond the most
    recent observation: skill < min_skill, where skill=0 is "no better than
    copying the latest value" and skill=1 is optimal. Model-free, and shares no
    quantity with temporal cols 1-2 -- see the block comment above for why the
    default threshold sits in a 0.29-wide empirical void rather than being tuned.

    `df_task` must be single-task and carry that task's truth column.
    """
    _assert_single_task(df_task)
    d = _dedup(df_task).copy()

    d["value_num"] = pd.to_numeric(d["value"], errors="coerce")
    d["resp_num"] = pd.to_numeric(d["response"], errors="coerce")
    if task == "colors":
        if "true_p" not in d.columns:
            raise KeyError("flag_no_integration needs true_p for colors")
        d["truth"] = pd.to_numeric(d["true_p"], errors="coerce") * 100.0
        # value is +-1 (blue/red) -> the response scale, as _value_on_response_scale
        d["val_scale"] = np.where(d["value_num"] == 1, 100.0, 0.0)
    else:
        if "true_mean" not in d.columns:
            raise KeyError("flag_no_integration needs true_mean for numbers")
        d["truth"] = pd.to_numeric(d["true_mean"], errors="coerce")
        d["val_scale"] = d["value_num"]

    d = d.sort_values(["prolific_pid", "trial", "observation"])
    d["running"] = (d.groupby(["prolific_pid", "trial"])["val_scale"]
                    .transform(lambda s: s.expanding().mean()))

    rows = []
    for pid, g in d.groupby("prolific_pid"):
        g = g.dropna(subset=["resp_num", "truth", "val_scale"])
        if g.empty:
            rows.append(dict(prolific_pid=pid, task=task, skill=np.nan,
                             err_pid=np.nan, err_optimal=np.nan,
                             err_baseline=np.nan, baseline_used="none"))
            continue
        err_pid = (g["resp_num"] - g["truth"]).abs().mean()
        err_opt = (g["running"] - g["truth"]).abs().mean()
        err_base = (g["val_scale"] - g["truth"]).abs().mean()   # last observation
        skill = ((err_base - err_pid) / (err_base - err_opt)
                 if err_base > err_opt else np.nan)
        rows.append(dict(prolific_pid=pid, task=task, skill=skill,
                         err_pid=err_pid, err_optimal=err_opt,
                         err_baseline=err_base, baseline_used="last_value"))

    out = pd.DataFrame(rows)
    # NaN skill means the baseline was already at/below optimal for this pid --
    # the comparison is undefined, so do not flag on it.
    out["flagged_no_integration"] = out["skill"].lt(min_skill).fillna(False)
    return out
