"""
utils/participant_filters.py

Quantifiable, adjustable exclusion criteria for identifying participants who
show no evidence of genuinely attempting the soltani numbers or colors
estimation task, as opposed to participants who attempt the task but update
sub-optimally (whom we explicitly do NOT want to exclude).

All criteria are computed per (prolific_pid, task) — a participant who did
both tasks can be excluded from one without being excluded from the other.

THE CORE METHODOLOGY: NESTED REGRESSION + COHEN'S f²
--------------------------------------------------------
An earlier version of this module used three different statistical
objects (a tolerance-based fraction, a binomial test, a raw Pearson
correlation, a partial Pearson correlation with a hand-picked cutoff) --
defensible individually, but the COLLECTION read as an ad hoc patchwork
rather than one principled measurement of "genuinely attempting the
task." The PI raised exactly this concern directly (see the conversation
that produced this rewrite): why these specific tests, with these
specific cutoffs, and how do we know we've covered every way someone
could be inattentive?

The replacement, used by all three criteria below: does adding ONE
specific piece of task-relevant information to a regression explain a
NON-TRIVIAL amount of additional variance, measured by Cohen's f² (the
standard effect size for an added predictor's incremental contribution to
a regression: f² = (R²_full - R²_reduced) / (1 - R²_full)) against Cohen's
own conventional f²=0.02 "small effect" boundary. One established,
citable convention, applied the same way three times, instead of three
different ad hoc constructions each needing its own separate
justification. See `_r2_ols`/`_cohens_f2` for the shared implementation.

Deliberately an EFFECT-SIZE threshold, not a significance test (p > alpha)
-- checked directly against two real batches (numbers and colors "part A",
~448 updates/pid each) before settling on this: at that sample size, a
plain significance test declares virtually ANY nonzero effect
significant, making it uselessly lenient as an exclusion criterion, AND
makes the flagging decision depend on how much data a participant
happens to have (more timeouts -> less power -> easier to "pass" with the
same underlying behavior) rather than on how large the effect actually
is, which is what we care about. A fixed effect-size cutoff has neither
problem.

CRITERION 1 — RECENCY-ONLY UPDATING
--------------------------------------
Does the participant's response carry any information about the trial's
PRIOR history (everything before the current observation) BEYOND what the
single current observation already explains? f² of adding `prior_mean`
(the mean of all strictly-prior values in the trial) to a regression that
already has `current_value`, predicting `response`. A participant who's
genuinely trying to integrate but doing so poorly or noisily (a real,
if weak, contribution from prior history) is NOT flagged -- only someone
whose response is explained by the single most recent observation, with
prior history adding nothing beyond that.

This SUBSUMES the old literal-copy check ("no_integration") that used to
be its own separate criterion, not just in theory: every participant that
old check flagged (response matches the current stimulus almost exactly)
shows an even MORE extreme version of this exact signature -- confirmed
directly against real data. Some literal copiers produce a 0/0-degenerate
f² (both the reduced and full model already fit at ~R²=1, so neither the
numerator nor denominator of Cohen's formula is informative on its own) --
`_cohens_f2` resolves this correctly as f²≈0 (flagged), not the
`+inf` a naive implementation would give from dividing by a near-zero
denominator alone, checked directly against a real case, not assumed.

CRITERIA 2 & 3 — NON-CONTINGENT UPDATING (sign / magnitude)
-----------------------------------------------------------------
Participants ARE updating their response every trial — the failure mode
here isn't a lack of updating, it's updating in a way that shows no
reliable relationship to (isn't "contingent on") the evidence just shown.
Two independent, separately-flaggable sub-patterns, both now the same
f²-of-an-added-predictor question as criterion 1 above:

  - flag_noncontingent_sign: f² of adding sign(discrepancy) to an
    intercept-only model, predicting `update` (the participant's own
    response change). If near-zero, the DIRECTION of their updates
    carries no information about the stimulus.
  - flag_noncontingent_magnitude: f² of adding |discrepancy| to an
    intercept-only model, predicting |update|. If near-zero, the SIZE of
    their updates carries no information about how surprising the
    evidence was, even if direction sometimes happens to line up.

These are deliberately kept as two SEPARATE filters (each can exclude on
its own) rather than bundled into one "both must fail" criterion, so each
failure mode is independently visible and independently adjustable.

Neither non-contingent-updating filter flags participants who update in a
directionally/magnitude-sensitive but non-Bayesian way (e.g. a constant
learning rate instead of properly shrinking it over the trial) — that's
the pattern most genuinely-trying participants actually show, and it's
exactly the kind of "imperfect but genuine effort" behavior we do NOT want
to exclude.

All three criteria catch different failure modes and are combined with OR
at the top level. All are intentionally conservative and meant to catch
"no apparent effort" participants, not "imperfect" ones. All thresholds
are exposed as parameters and expected to be revisited as more data comes
in — see figure_soltani_temporal.py and the conversation that produced
this module for the diagnostics these were calibrated against.

SUPERSEDED VERSIONS
-----------------------
The original three-different-statistical-tools version of this module
(tolerance-based no_integration, significance-based sign/magnitude,
r=0.10-effect-size-based recency_only) is archived at
`archive/utils/archive_participant_filters_legacy.py`, including the full
account of exactly what it caught differently from the current version
and why the switch happened -- not deleted, since it's genuine prior
methodology worth being able to point to, just no longer live.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_MIN_UPDATES = 10    # below this, we don't have enough data to judge
DEFAULT_MIN_F2      = 0.02  # Cohen's conventional "small effect" f^2 boundary -- see module docstring


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
    the participant's OWN previous response), in long form across pids.
    Shared by both non-contingent-updating criteria below."""
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


def _compute_recency_features(df_task: pd.DataFrame, task: str) -> pd.DataFrame:
    """One row per (prolific_pid, trial, observation>=1): the participant's
    own response, that observation's current stimulus value (on the
    response scale), and the mean of all STRICTLY PRIOR values within the
    same trial (excluding the current observation) -- also on the response
    scale. observation=0 has no prior history at all and is excluded,
    matching _compute_updates' own convention above. Used by
    flag_recency_only below."""
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
                    "response": resp[i], "current_value": val_scale[i],
                    "prior_mean": float(np.mean(val_scale[:i])),
                })
    return pd.DataFrame(rows)


# ── Shared statistical core: nested OLS + Cohen's f² ────────────────────────

def _r2_ols(y: np.ndarray, X: np.ndarray) -> float:
    """R^2 of a plain OLS fit of y on X (intercept added automatically).
    X can be 1D (one predictor) or 2D (n_obs x n_predictors, already laid
    out that way -- callers build it as np.column_stack([...]))."""
    X = np.atleast_2d(X)
    if X.shape[0] != len(y):
        X = X.T
    X_design = np.column_stack([np.ones(len(y)), X])
    coefs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    y_pred = X_design @ coefs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


def _cohens_f2(y: np.ndarray, X_reduced: np.ndarray | None, X_full: np.ndarray) -> float:
    """Cohen's f^2 for the incremental contribution of whatever's in
    X_full but not X_reduced: f2 = (R2_full - R2_reduced) / (1 - R2_full).
    X_reduced=None means "intercept-only" reduced model, whose R^2 is 0 by
    construction (predicting the mean explains none of its own variance),
    so this simplifies to f2 = R2_full / (1 - R2_full) -- used directly by
    the sign/magnitude criteria below, which have no other predictor to
    control for.

    Guards a real 0/0 case, not just a divide-by-zero: if y is a
    near-perfect deterministic function of X_reduced ALONE (e.g. a
    literal copier, response ≈ current_value exactly), adding X_full's
    extra predictor(s) can't improve an already-near-perfect fit, so BOTH
    R2_reduced and R2_full approach 1 together -- the numerator
    (R2_full-R2_reduced) approaches 0 at the same rate the denominator
    (1-R2_full) does. Checking only the denominator and returning +inf
    whenever it's tiny (an earlier version of this function did exactly
    that) is wrong here: it reads a genuinely NULL incremental effect as
    an enormous one, the opposite of correct -- confirmed by tracing a
    real literal-copier case directly, not assumed. The fix: when the
    denominator is ~0, look at the numerator too -- ~0 numerator means ~0
    incremental effect (f2=0, correctly flaggable), not infinity; only a
    genuinely large numerator alongside a ~0 denominator (a near-perfect
    fit that ONLY the full model achieves) is a real, extreme
    large-effect case.
    """
    r2_full = _r2_ols(y, X_full)
    r2_reduced = 0.0 if X_reduced is None else _r2_ols(y, X_reduced)
    numerator = r2_full - r2_reduced
    denom = 1 - r2_full
    if denom > 1e-9:
        return float(numerator / denom)
    return 0.0 if numerator < 1e-9 else float("inf")


# ── Criterion 1: RECENCY-ONLY UPDATING ──────────────────────────────────────

def flag_recency_only(df_task: pd.DataFrame, task: str,
                      min_f2: float = DEFAULT_MIN_F2,
                      min_updates: int = DEFAULT_MIN_UPDATES) -> pd.DataFrame:
    """Per prolific_pid: f2 of adding prior_mean to a model that already
    has current_value (predicting response). Flagged if f2 < min_f2 --
    see module docstring's CRITERION 1 for the full rationale."""
    feats = _compute_recency_features(df_task, task)
    rows = []
    for pid, g in feats.groupby("prolific_pid"):
        n = len(g)
        if n < min_updates:
            rows.append({"prolific_pid": pid, "task": task, "n_updates": n,
                        "f2_recency": np.nan, "flagged_recency_only": False,
                        "note": f"n_updates={n} < min_updates={min_updates}"})
            continue

        cur = g["current_value"].to_numpy()
        prior = g["prior_mean"].to_numpy()
        resp = g["response"].to_numpy()

        if np.std(cur) < 1e-9:
            # A degenerate STIMULUS sequence (no variance in current_value
            # at all across this pid's updates), not evidence about their
            # BEHAVIOR -- we simply can't judge, so don't flag without
            # evidence either way. Essentially never triggers in practice
            # given how much real stimuli vary; kept as a defensive guard.
            rows.append({"prolific_pid": pid, "task": task, "n_updates": n,
                        "f2_recency": np.nan, "flagged_recency_only": False,
                        "note": "current_value has no variance"})
            continue

        f2 = _cohens_f2(resp, X_reduced=cur, X_full=np.column_stack([cur, prior]))
        rows.append({"prolific_pid": pid, "task": task, "n_updates": n,
                    "f2_recency": f2, "flagged_recency_only": bool(f2 < min_f2),
                    "note": ""})
    return pd.DataFrame(rows)


# ── Criteria 2 & 3: NON-CONTINGENT UPDATING (sign / magnitude, kept separate) ─

def flag_noncontingent_sign(df_task: pd.DataFrame, task: str,
                            min_f2: float = DEFAULT_MIN_F2,
                            min_updates: int = DEFAULT_MIN_UPDATES) -> pd.DataFrame:
    """Per prolific_pid: f2 of adding sign(discrepancy) to an intercept-
    only model (predicting update). Flagged if f2 < min_f2 -- see module
    docstring's CRITERIA 2 & 3 for the full rationale."""
    updates = _compute_updates(df_task, task)
    rows = []
    for pid, g in updates.groupby("prolific_pid"):
        n = len(g)
        if n < min_updates:
            rows.append({"prolific_pid": pid, "task": task, "n_updates": n,
                        "f2_sign": np.nan, "flagged_noncontingent_sign": False,
                        "note": f"n_updates={n} < min_updates={min_updates}"})
            continue

        update = g["update"].to_numpy()
        sign_discrepancy = np.sign(g["discrepancy"].to_numpy())

        if np.std(sign_discrepancy) < 1e-9 or np.std(update) < 1e-9:
            rows.append({"prolific_pid": pid, "task": task, "n_updates": n,
                        "f2_sign": np.nan, "flagged_noncontingent_sign": True,
                        "note": "no variance in sign(discrepancy) or update"})
            continue

        f2 = _cohens_f2(update, X_reduced=None, X_full=sign_discrepancy)
        rows.append({"prolific_pid": pid, "task": task, "n_updates": n,
                    "f2_sign": f2, "flagged_noncontingent_sign": bool(f2 < min_f2),
                    "note": ""})
    return pd.DataFrame(rows)


def flag_noncontingent_magnitude(df_task: pd.DataFrame, task: str,
                                 min_f2: float = DEFAULT_MIN_F2,
                                 min_updates: int = DEFAULT_MIN_UPDATES) -> pd.DataFrame:
    """Per prolific_pid: f2 of adding |discrepancy| to an intercept-only
    model (predicting |update|). Flagged if f2 < min_f2 -- see module
    docstring's CRITERIA 2 & 3 for the full rationale."""
    updates = _compute_updates(df_task, task)
    rows = []
    for pid, g in updates.groupby("prolific_pid"):
        n = len(g)
        if n < min_updates:
            rows.append({"prolific_pid": pid, "task": task, "n_updates": n,
                        "f2_magnitude": np.nan, "flagged_noncontingent_magnitude": False,
                        "note": f"n_updates={n} < min_updates={min_updates}"})
            continue

        abs_update = g["update"].abs().to_numpy()
        abs_discrepancy = g["discrepancy"].abs().to_numpy()

        if np.std(abs_discrepancy) < 1e-9 or np.std(abs_update) < 1e-9:
            rows.append({"prolific_pid": pid, "task": task, "n_updates": n,
                        "f2_magnitude": np.nan, "flagged_noncontingent_magnitude": True,
                        "note": "no variance in |discrepancy| or |update|"})
            continue

        f2 = _cohens_f2(abs_update, X_reduced=None, X_full=abs_discrepancy)
        rows.append({"prolific_pid": pid, "task": task, "n_updates": n,
                    "f2_magnitude": f2, "flagged_noncontingent_magnitude": bool(f2 < min_f2),
                    "note": ""})
    return pd.DataFrame(rows)


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

# ── Criterion set 3: WORSE THAN A TRIVIAL BASELINE (model-free) ─────────────
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
# THE THRESHOLD NEEDS NO TUNING: skill < 0 means "did not beat a strategy the
# instructions rule out", which is principled in a way that a chosen SD multiple
# is not.
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

DEFAULT_MIN_SKILL = 0.0


def flag_below_baseline(df_task: pd.DataFrame, task: str,
                        min_skill: float = DEFAULT_MIN_SKILL) -> pd.DataFrame:
    """Flag participants who do not beat the best trivial strategy on their own
    sequences. Model-free, and shares no quantity with temporal cols 1-2 -- see
    the block comment above.

    `df_task` must be single-task and carry that task's truth column.
    """
    _assert_single_task(df_task)
    d = _dedup(df_task).copy()

    d["value_num"] = pd.to_numeric(d["value"], errors="coerce")
    d["resp_num"] = pd.to_numeric(d["response"], errors="coerce")
    if task == "colors":
        if "true_p" not in d.columns:
            raise KeyError("flag_below_baseline needs true_p for colors")
        d["truth"] = pd.to_numeric(d["true_p"], errors="coerce") * 100.0
        # value is +-1 (blue/red) -> the response scale, as _value_on_response_scale
        d["val_scale"] = np.where(d["value_num"] == 1, 100.0, 0.0)
    else:
        if "true_mean" not in d.columns:
            raise KeyError("flag_below_baseline needs true_mean for numbers")
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
    out["flagged_below_baseline"] = out["skill"].lt(min_skill).fillna(False)
    return out


# Which criterion set decides `excluded`. Both are always COMPUTED and reported;
# only the `excluded` column differs, so a report always carries the diagnostics
# for the method you did not choose.
EXCLUSION_METHODS = ("contingency", "performance", "baseline")
DEFAULT_EXCLUSION_METHOD = "contingency"


def compute_exclusion_report(df: pd.DataFrame, tasks: tuple[str, ...] = ("colors", "numbers"),
                             min_f2: float = DEFAULT_MIN_F2,
                             min_updates: int = DEFAULT_MIN_UPDATES,
                             method: str = DEFAULT_EXCLUSION_METHOD,
                             max_error_sd: float = DEFAULT_MAX_ERROR_SD,
                             min_skill: float = DEFAULT_MIN_SKILL) -> pd.DataFrame:
    """Full diagnostic report, one row per (prolific_pid, task) present in
    `df`. `df` may span multiple tasks — this function does the required
    per-task slicing before calling into any criterion itself.

    `method` selects which criterion set drives the `excluded` column:
      'contingency'  the three Cohen's f2 tests (recency-only, non-contingent
                     sign, non-contingent magnitude). Model-BASED: it asks
                     whether updating is contingent on prediction error, which
                     is upstream of what the temporal figures measure.
      'performance'  gross performance outlier on mean absolute error against the
                     true generative parameter, carrabin's own rule. Model-FREE.
      'baseline'     did not beat the best TRIVIAL strategy (latest observation,
                     or the scale midpoint) on their own sequences. Model-FREE,
                     needs no threshold tuning, and shares no quantity with
                     temporal cols 1-2. See flag_below_baseline.
    Both criterion sets are computed either way, so the report always carries the
    diagnostics for the method you did not pick -- which is the point: the
    contingency measures remain reportable as a description of heterogeneity
    (e.g. "n of N retained participants showed recency-only updating") even when
    they are not driving exclusion.
    """
    if method not in EXCLUSION_METHODS:
        raise ValueError(f"method must be one of {EXCLUSION_METHODS}, got {method!r}")

    reports = []
    for task in tasks:
        df_task = df[df["task"] == task]
        if df_task.empty:
            continue
        r1 = flag_recency_only(df_task, task, min_f2, min_updates)
        r2 = flag_noncontingent_sign(df_task, task, min_f2, min_updates).drop(columns=["n_updates", "note"])
        r3 = flag_noncontingent_magnitude(df_task, task, min_f2, min_updates).drop(columns=["n_updates", "note"])
        r4 = flag_performance_outlier(df_task, task, max_error_sd)
        r5 = flag_below_baseline(df_task, task, min_skill)
        merged = r1.merge(r2, on=["prolific_pid", "task"], how="outer")
        merged = merged.merge(r3, on=["prolific_pid", "task"], how="outer")
        merged = merged.merge(r4, on=["prolific_pid", "task"], how="outer")
        merged = merged.merge(r5, on=["prolific_pid", "task"], how="outer")
        reports.append(merged)

    if not reports:
        return pd.DataFrame(columns=["prolific_pid", "task", "n_updates",
                                     "f2_recency", "flagged_recency_only",
                                     "f2_sign", "flagged_noncontingent_sign",
                                     "f2_magnitude", "flagged_noncontingent_magnitude",
                                     "mean_abs_error", "flagged_performance_outlier",
                                     "skill", "err_pid", "err_optimal",
                                     "err_baseline", "baseline_used",
                                     "flagged_below_baseline", "excluded"])

    report = pd.concat(reports, ignore_index=True)
    report["flagged_any_contingency"] = (
        report["flagged_recency_only"].fillna(False) |
        report["flagged_noncontingent_sign"].fillna(False) |
        report["flagged_noncontingent_magnitude"].fillna(False))
    if method == "contingency":
        report["excluded"] = report["flagged_any_contingency"]
    elif method == "performance":
        report["excluded"] = report["flagged_performance_outlier"].fillna(False)
    else:
        report["excluded"] = report["flagged_below_baseline"].fillna(False)
    report.attrs["exclusion_method"] = method
    return report


def excluded_pairs(df: pd.DataFrame, **kwargs) -> set[tuple[str, str]]:
    """Set of (prolific_pid, task) pairs that fail any exclusion criterion."""
    report = compute_exclusion_report(df, **kwargs)
    excl = report[report["excluded"]]
    return set(zip(excl["prolific_pid"], excl["task"]))


def filter_participants(df: pd.DataFrame, report: pd.DataFrame | None = None,
                        verbose: bool = True, require_both_tasks: bool = True,
                        **kwargs) -> pd.DataFrame:
    """Return `df` with rows belonging to any excluded (prolific_pid, task)
    pair removed.

    `require_both_tasks=True` (default) makes exclusion SUBJECT-level: a
    participant who fails in either task is dropped from BOTH, so every task ends
    up with the same participants. Set False for the older per-(pid, task)
    behaviour, where a participant excluded from one task kept their data from
    the other.

    Subject-level is the default because per-task exclusion silently degrades
    every WITHIN-SUBJECT cross-task analysis whenever the criterion is not
    equally strict in both tasks. Measured directly: under the 'baseline' method
    with per-task exclusion, numbers retained 29 and colors 36, and the 26-pid
    intersection was a differently-selected group from either sample -- the
    cross-task lambda correlation fell to r=0.331 (p=0.099) from r=0.587
    (p=0.0013) under a symmetric filter. That was NOT a power loss or a
    reliability loss: lambda's split-half reliability was if anything higher
    (colors 0.836 vs 0.796), the attenuation ceilings were indistinguishable
    (0.791 vs 0.780), and lambda's SD and range were unchanged -- so the drop was
    purely the composition of the intersection. Requiring both tasks makes the
    intersection the sample by construction.

    Only applies when `df` actually spans more than one task; single-task frames
    are unaffected.

    Pass a pre-computed `report` (from compute_exclusion_report) to reuse
    it and avoid recomputing; otherwise one is computed from `df` using
    **kwargs (forwarded to compute_exclusion_report)."""
    if report is None:
        report = compute_exclusion_report(df, **kwargs)

    excl = report[report["excluded"]]
    excl_pairs = set(zip(excl["prolific_pid"], excl["task"]))

    tasks_present = sorted(df["task"].unique())
    n_propagated = 0
    if require_both_tasks and len(tasks_present) > 1:
        failed_any = {pid for pid, _ in excl_pairs}
        expanded = {(pid, t) for pid in failed_any for t in tasks_present}
        n_propagated = len(expanded) - len(excl_pairs)
        excl_pairs = expanded

    if verbose:
        method = report.attrs.get("exclusion_method", DEFAULT_EXCLUSION_METHOD)
        if n_propagated:
            print(f"require_both_tasks: propagated {n_propagated} additional "
                  f"(pid, task) exclusion(s) so every task keeps the same "
                  f"{len(set(df['prolific_pid'])) - len({p for p, _ in excl_pairs})} "
                  f"participants")
        if excl_pairs:
            print(f"Excluding {len(excl_pairs)} (prolific_pid, task) pair(s) "
                  f"[method={method}]:")
            for _, row in excl.iterrows():
                reasons = []
                if row.get("flagged_below_baseline"):
                    sk = row.get("skill")
                    sk_str = f"{sk:.3f}" if pd.notna(sk) else "undefined"
                    reasons.append(f"below_baseline (skill={sk_str} vs "
                                   f"{row.get('baseline_used')})")
                if row.get("flagged_performance_outlier"):
                    err = row.get("mean_abs_error")
                    err_str = f"{err:.2f}" if pd.notna(err) else "undefined"
                    reasons.append(f"performance_outlier (mean |error|={err_str})")
                if row.get("flagged_recency_only"):
                    f2_str = f"{row['f2_recency']:.4f}" if pd.notna(row['f2_recency']) else "undefined"
                    reasons.append(f"recency_only (f2={f2_str})")
                if row.get("flagged_noncontingent_sign"):
                    f2_str = f"{row['f2_sign']:.4f}" if pd.notna(row['f2_sign']) else "undefined"
                    reasons.append(f"noncontingent_sign (f2={f2_str})")
                if row.get("flagged_noncontingent_magnitude"):
                    f2_str = f"{row['f2_magnitude']:.4f}" if pd.notna(row['f2_magnitude']) else "undefined"
                    reasons.append(f"noncontingent_magnitude (f2={f2_str})")
                print(f"  {row['prolific_pid']} / {row['task']}: {', '.join(reasons)}")
        else:
            print(f"No participants excluded [method={method}].")

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
