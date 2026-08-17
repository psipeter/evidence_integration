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

DEFAULT_MIN_UPDATES = 10
DEFAULT_MIN_TRIALS  = 8     # below this a trial-level bootstrap is meaningless    # below this, we don't have enough data to judge
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


# ── Criterion set 4: NON-INTEGRATOR (definition-first, model-free) ──────────
#
# THE DEFINITION, which the other three criterion sets only approximate:
#
#   A NON-INTEGRATOR is a participant for whom observations BEFORE the most
#   recent one make no reliable contribution to predicting their responses.
#
# Stated as a property of information rather than of accuracy or of weighting,
# which is what makes it the right definition for this project:
#
#   - It does NOT require accuracy. A participant who integrates all history but
#     inaccurately -- with strong recency bias, or with large response noise --
#     still has a reliable contribution from prior observations, and is RETAINED.
#     That is the case the accuracy-based criteria get wrong (see below).
#   - It does NOT require a particular WEIGHT on history, only that the weight be
#     distinguishable from zero. So there is no threshold to choose: the cut is a
#     significance level, not a magnitude.
#   - It catches BOTH observed failure modes with one test. A copier gets no
#     predictive benefit from history; a random/drifting responder gets no benefit
#     from anything. Both fail.
#
# OPERATIONALISATION. Per participant, regress the response on the most recent
# observation AND the mean of all strictly-prior observations in that trial:
#
#     response_t ~ 1 + value_t + mean(value_0 .. value_{t-1})
#
# and ask whether the coefficient on the prior mean is reliably nonzero. Inference
# is a TRIAL-LEVEL CLUSTER BOOTSTRAP (resample whole trials with replacement),
# because responses within a trial are strongly dependent -- the slider persists
# and the running mean changes slowly -- so ordinary OLS standard errors are far
# too small here. Retained if the bootstrap CI excludes zero.
#
# WHY NOT the alternatives that were built and tested first:
#
#   'contingency'  the closest of the three, and closer to right than the two
#                  below: recency_only tests the same construct. It differs in
#                  using an f^2 EFFECT-SIZE threshold (0.02) rather than
#                  reliability, and in using in-sample variance without
#                  clustering. This criterion is that idea done properly.
#   'performance'  a gross-outlier rule on mean absolute error. Accuracy-based, so
#                  it cannot distinguish an inaccurate integrator from a
#                  non-integrator -- exactly the confusion to avoid.
#   'integration'  the skill score. Measured directly on synthetic leaky
#                  integrators, it is NOT MONOTONE in integration depth: it peaks
#                  at alpha=0.20 (+0.745) and is LOWER for a near-optimal
#                  alpha=0.10 agent (+0.603), because with 15 observations a mild
#                  overweighting of recent evidence tracks the running mean better
#                  than a sluggish filter does. Worse, a genuine integrator with
#                  alpha=0.70 and realistic response noise scores +0.115 -- a hair
#                  above the 0.10 threshold. It discards inaccurate integrators.
#
# ALSO REJECTED, after testing on this data:
#   Thresholding the serial-position weight on the latest observation (g_lag0).
#   g_lag0 recovers alpha almost exactly (0.100/0.200/0.350/0.494/0.687/0.959 for
#   true alpha 0.10-1.00) and is nearly immune to response noise, so it is the
#   right MEASURE of integration depth -- but it is continuous with no natural
#   cutoff (largest gap 0.076 in a 0.03-1.00 range), and it CANNOT catch random
#   responders: unstructured responses give diffuse weights scoring ~0.12, which
#   is indistinguishable from optimal. Any weight-based test is blind to the
#   "nothing predicts them" failure mode. Report g_lag0 as a descriptive measure;
#   do not filter on it.
#   A one-sided version of this test (requiring a POSITIVE contribution, to catch
#   scale inversion such as reporting % red instead of % blue). Tested: 1 of 61
#   numbers participants and 0 of 61 colors have a reliably negative coefficient,
#   and the one case is marginal (b=-0.074, CI [-0.171,-0.013]). Not worth the
#   added directional assumption.
#   Requiring stability across session HALVES. Tested: 26% of retained numbers
#   participants pass pooled but not both halves -- and the asymmetry runs the
#   WRONG way for a fatigue story. 12 participants integrate only in the second
#   half against 4 only in the first, i.e. the instability is mostly LATE
#   LEARNING, consistent with error falling 19% from the first 8 to the last 8
#   trials. Requiring both halves would penalise a slow start, which is a
#   consequence of the tutorial having no comprehension gate rather than a
#   participant defect.
#   Restricting evaluation to trials 8-31 (a burn-in). Tested: moves retention by
#   exactly ONE participant per task (numbers 42->41, colors 43->44), and the two
#   retained sets are indistinguishable in accuracy on the same late trials
#   (median |error vs running| 4.90 vs 4.79 for numbers, 6.94 vs 7.03 for colors).
#   No gain, so use all 32 trials.
#
# WHAT IT DOES ON complete_pairs: excludes 19/61 numbers and 18/61 colors (~30%),
# against carrabin's 16% and yoo's 17%. The excluded group is far worse on an
# accuracy measure the criterion never sees -- median |error vs running mean| 8.33
# vs 4.90 (numbers) and 27.37 vs 6.94 (colors) -- which is the validation that it
# is not carving the distribution arbitrarily.
#
# KNOWN GAPS, deliberately not engineered around. The definition retains anyone
# whose responses reliably use history, so it does NOT catch: integrating the
# WRONG STATISTIC (running sum, max, a hand-picked subset); SCALE COMPRESSION
# (correct direction, only using 40-60 of the slider -- fine for temporal panels,
# bad for accuracy panels); or ANCHORED-WITH-A-NUDGE (parked near 50, shifting
# slightly with the evidence). The first two are arguably correct to retain; the
# third is a real miss. Accuracy-sensitive analyses may still want the
# 'performance' criterion in addition -- one filter serving both jobs was probably
# never the right goal.
#
# AND ONE STRUCTURAL LIMITATION: this is a SIGNIFICANCE test, so it is
# power-dependent. The same behaviour passes with 32 trials and fails with 16. The
# ~30% rate is tied to this design's trial count and does not transfer.

# n_boot is high because MONTE CARLO NOISE ALONE MOVES MEMBERSHIP. Measured on
# complete_pairs numbers: at n_boot=2000, seeds 0/1/2 give 17/18/19 flagged, and
# n_boot=500 gives 19 -- so two to three participants' fate was decided by the
# random seed. (An earlier comment in this file claimed raising n_boot to 2000
# "fixed" that; it reduced it, it did not eliminate it.) The Gram-matrix bootstrap
# below makes 20000 resamples cheap, so there is no reason to economise. `seed` is
# returned in the report so any exclusion set is reproducible.
DEFAULT_N_BOOT = 20000
DEFAULT_CI = 95.0

# SENSITIVITY OF THIS CRITERION, measured -- report the range, not just the point.
# The criterion removes the ARBITRARY MAGNITUDE threshold that the other three
# have (f^2=0.02, 2 SD, skill<0.10), which was the main objection to them. It does
# NOT make the criterion threshold-free. On complete_pairs:
#
#   ci level     numbers 16 / 17 / 24 flagged at ci = 90 / 95 / 99
#                colors  17 / 17 / 20
#                90<->95 is stable; 99 adds 7 for numbers (+41%). Churn is
#                one-directional, as it must be (a wider CI is more likely to
#                include zero), but the significance level is consequential.
#   seed         +/- 2 participants at n_boot=2000; hence the default above.
#   predictors   the LARGEST source of variation. Using the last 3 lags
#                individually plus an older-mean term instead of a single
#                prior_mean gives numbers 23 (churn +10/-4) and colors 15
#                (+3/-5) -- 14 participants change status in numbers.
#
# WHY prior_mean IS NEVERTHELESS THE RIGHT FORM, and not merely convenient: it
# asks the definitional question ("does history beyond the latest observation
# contribute?") as ONE test. The full-lag version splits that signal across four
# correlated predictors, so every CI widens (power loss -> MORE flagged, as in
# numbers) while simultaneously giving four uncorrected chances at significance
# (multiplicity -> FEWER flagged, as in colors). Those two errors moving in
# opposite directions across tasks is the signature of an ill-posed test. Do not
# "improve" this by adding lag predictors without correcting for multiplicity.


def flag_non_integrator(df_task: pd.DataFrame, task: str,
                        n_boot: int = DEFAULT_N_BOOT,
                        ci: float = DEFAULT_CI,
                        seed: int = 0) -> pd.DataFrame:
    """Flag participants for whom observations before the most recent make no
    reliable contribution to predicting their responses. See the block comment
    above for the definition, the operationalisation, and what it does and does
    not catch.

    Returns one row per participant with the prior-mean coefficient, its
    bootstrap CI, the coefficient on the latest observation for reference, and
    `flagged_non_integrator`.
    """
    _assert_single_task(df_task)
    d = _dedup(df_task).copy()

    d["resp_num"] = pd.to_numeric(d["response"], errors="coerce")
    d["val_scale"] = _value_on_response_scale(d, task)
    d = d.sort_values(["prolific_pid", "trial", "observation"])
    # Mean of STRICTLY PRIOR observations within the trial. shift() first so the
    # current observation is excluded -- otherwise the two predictors are
    # collinear by construction and the test is meaningless.
    d["prior_mean"] = (d.groupby(["prolific_pid", "trial"])["val_scale"]
                       .transform(lambda s: s.shift().expanding().mean()))
    d = d.dropna(subset=["resp_num", "val_scale", "prior_mean"])

    rng = np.random.default_rng(seed)
    lo_q, hi_q = (100.0 - ci) / 2.0, 100.0 - (100.0 - ci) / 2.0

    def _solve(AtA, Aty):
        """OLS coefficients from normal equations. Returns (b_latest, b_prior)."""
        try:
            b = np.linalg.solve(AtA, Aty)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        return float(b[1]), float(b[2])

    rows = []
    for pid, sub in d.groupby("prolific_pid"):
        trials = sub["trial"].unique()
        if len(trials) < DEFAULT_MIN_TRIALS or len(sub) < DEFAULT_MIN_UPDATES:
            rows.append(dict(prolific_pid=pid, task=task, b_latest=np.nan,
                             b_prior=np.nan, b_prior_lo=np.nan, b_prior_hi=np.nan,
                             n_trials=len(trials), n_boot=n_boot, ci=ci, seed=seed,
                             note="too few trials/updates"))
            continue

        # Per-trial X'X and X'y. A trial-level bootstrap resample is then just a
        # SUM of these, so each resample costs one 3x3 solve instead of rebuilding
        # a DataFrame -- which is what makes n_boot=20000 affordable (the previous
        # pd.concat version took minutes per task at n_boot=2000).
        AtA = np.empty((len(trials), 3, 3))
        Aty = np.empty((len(trials), 3))
        for j, t in enumerate(trials):
            g = sub[sub["trial"] == t]
            X = np.column_stack([np.ones(len(g)),
                                 g["val_scale"].to_numpy(float),
                                 g["prior_mean"].to_numpy(float)])
            y = g["resp_num"].to_numpy(float)
            AtA[j] = X.T @ X
            Aty[j] = X.T @ y

        b_latest, b_prior = _solve(AtA.sum(0), Aty.sum(0))

        idx = rng.integers(0, len(trials), size=(n_boot, len(trials)))
        boots = np.empty(n_boot)
        for i in range(n_boot):
            sel = idx[i]
            boots[i] = _solve(AtA[sel].sum(0), Aty[sel].sum(0))[1]
        boots = boots[np.isfinite(boots)]
        if len(boots) < n_boot // 2:
            lo = hi = np.nan
        else:
            lo, hi = (float(x) for x in np.percentile(boots, [lo_q, hi_q]))
        rows.append(dict(prolific_pid=pid, task=task, b_latest=b_latest,
                         b_prior=b_prior, b_prior_lo=lo, b_prior_hi=hi,
                         n_trials=len(trials), n_boot=n_boot, ci=ci, seed=seed,
                         note=""))

    out = pd.DataFrame(rows)
    # Reliable contribution = CI excludes zero. A NaN CI (too little data, or a
    # degenerate design matrix) is NOT evidence of integration, so it flags.
    reliable = (out["b_prior_lo"] > 0) | (out["b_prior_hi"] < 0)
    out["flagged_non_integrator"] = ~reliable.fillna(False)
    return out


# Which criterion set decides `excluded`. Both are always COMPUTED and reported;
# only the `excluded` column differs, so a report always carries the diagnostics
# for the method you did not choose.
# 'performance' and 'integration' were moved to
# archive/utils/archive_exclusion_criteria.py -- see that module for why each
# loses. 'contingency' is RETAINED as a computed diagnostic even though it no
# longer decides anything: recency_only tests the same construct as
# non_integrator by a different method (in-sample f^2 vs trial-clustered
# bootstrap reliability), and their agreement -- 23 of 25 and 18 of 19 excluded
# participants -- is the strongest available evidence that the exclusions are not
# an artefact of one statistic. Every exclusion in the verbose log therefore
# carries both criteria's numbers.
EXCLUSION_METHODS = ("contingency", "non_integrator")
DEFAULT_EXCLUSION_METHOD = "non_integrator"


def compute_exclusion_report(df: pd.DataFrame, tasks: tuple[str, ...] = ("colors", "numbers"),
                             min_f2: float = DEFAULT_MIN_F2,
                             min_updates: int = DEFAULT_MIN_UPDATES,
                             method: str = DEFAULT_EXCLUSION_METHOD,
                             n_boot: int = DEFAULT_N_BOOT) -> pd.DataFrame:
    """Full diagnostic report, one row per (prolific_pid, task) present in
    `df`. `df` may span multiple tasks — this function does the required
    per-task slicing before calling into any criterion itself.

    `method` selects which criterion set drives the `excluded` column:
      'non_integrator' (DEFAULT)  prior observations make no reliable
                     contribution to predicting the response, by trial-clustered
                     bootstrap. Definition-first, model-free, no magnitude
                     threshold. See flag_non_integrator.
      'contingency'  the three Cohen's f2 tests (recency-only, non-contingent
                     sign, non-contingent magnitude). Model-BASED: it asks
                     whether updating is contingent on prediction error, which
                     is upstream of what the temporal figures measure. Retained
                     mainly as a DIAGNOSTIC -- recency_only tests the same
                     construct as non_integrator by a different method, and their
                     agreement is the evidence that the exclusions are not an
                     artefact of one statistic.

    Two further criteria ('performance', 'integration') were tested and moved to
    archive/utils/archive_exclusion_criteria.py; see that module for why each
    loses.

    BOTH live criterion sets are computed either way, so the report always carries
    the diagnostics for the method you did not pick. That is the point: the
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
        r6 = flag_non_integrator(df_task, task, n_boot=n_boot).drop(columns=["note"])
        merged = r1.merge(r2, on=["prolific_pid", "task"], how="outer")
        merged = merged.merge(r3, on=["prolific_pid", "task"], how="outer")
        merged = merged.merge(r6, on=["prolific_pid", "task"], how="outer")
        reports.append(merged)

    if not reports:
        return pd.DataFrame(columns=["prolific_pid", "task", "n_updates",
                                     "f2_recency", "flagged_recency_only",
                                     "f2_sign", "flagged_noncontingent_sign",
                                     "f2_magnitude", "flagged_noncontingent_magnitude",
                                     "b_latest",
                                     "b_prior", "b_prior_lo", "b_prior_hi",
                                     "n_trials", "n_boot", "ci", "seed",
                                     "flagged_non_integrator", "excluded"])

    report = pd.concat(reports, ignore_index=True)
    report["flagged_any_contingency"] = (
        report["flagged_recency_only"].fillna(False) |
        report["flagged_noncontingent_sign"].fillna(False) |
        report["flagged_noncontingent_magnitude"].fillna(False))
    if method == "contingency":
        report["excluded"] = report["flagged_any_contingency"]
    else:
        report["excluded"] = report["flagged_non_integrator"].fillna(False)
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
                if row.get("flagged_non_integrator"):
                    b = row.get("b_prior")
                    lo, hi = row.get("b_prior_lo"), row.get("b_prior_hi")
                    if pd.notna(b) and pd.notna(lo):
                        reasons.append(f"non_integrator (b_prior={b:+.3f}, "
                                       f"CI [{lo:+.3f},{hi:+.3f}] includes 0)")
                    else:
                        reasons.append("non_integrator (CI undefined)")
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
