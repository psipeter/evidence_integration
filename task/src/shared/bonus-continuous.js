/**
 * bonus-continuous.js
 * Per-OBSERVATION error + per-OBSERVATION bonus-payment formula for the
 * continuous task (chat history).
 *
 * Error is computed ONCE per observation, at the moment that response is
 * given, and stored directly on that observation's own JATOS row (see
 * build-trial-timeline.js / build-tutorial-timeline.js's on_finish
 * callbacks) -- flows into JATOS via the SAME per-trial on_trial_finish
 * append every other trial already uses, no separate append call needed.
 * The trial-summary screen's `total_error` is then just the SUM of those
 * already-stored per-observation errors (see plugin-trial-summary-
 * continuous.js's caller) -- NOT independently recomputed from responses
 * + a reference at summary time. One source of truth: whatever's stored
 * per-observation is exactly what gets summed, so the two can never drift
 * apart from each other (e.g. a future formula tweak applied in one place
 * but not the other).
 *
 * REWARD FORMULA (recalibrated this session, chat history -- REPLACES an
 * earlier per-TRIAL formula, see below for why):
 *
 *     normError = rawError / MAX_POSSIBLE_ERROR        (scaled to 0-1)
 *     reward    = max(0, MAX_REWARD * (1 - BONUS_DECAY * normError))
 *
 * computed PER OBSERVATION now, not once per trial from a SUMMED error --
 * "each response can give reward bonus" (chat history) -- and the
 * trial's own total reward is just the SUM of these per-observation
 * rewards (computeTrialReward below), the same "store once per
 * observation, sum at summary time" pattern this file already used for
 * error itself. MAX_POSSIBLE_ERROR=100 because both tasks' responses/refs
 * live on the same bounded 0-100 scale (continuous's slider, binary's
 * percentage) -- the worst possible single response (0 vs a reference of
 * 100, or vice versa) is exactly 100 error, so dividing by 100 always
 * lands normError in [0,1] regardless of task. Parameters (chat history,
 * explicitly meant to be iterated on -- tuned twice already this session,
 * from an initial MAX_REWARD=4/BONUS_DECAY=5, then MAX_REWARD=3/
 * BONUS_DECAY=10): MAX_REWARD=3 (cents), BONUS_DECAY=15 -- reward hits
 * exactly 0 once normError >= 1/BONUS_DECAY ≈ 0.067, i.e. once a single
 * response is off by ~6.7+ (on the 0-100 scale) -- tighter still than the
 * previous 10.
 *
 * REPLACED (chat history, this session) a per-TRIAL formula that instead
 * summed every observation's raw error FIRST, then applied one decay/
 * floor to that sum:
 *
 *     reward = max(0, 100 - BONUS_DECAY_OLD * totalError)
 *
 * BONUS_DECAY_OLD was itself a fix for an EARLIER bug (kept here for
 * history): originally a plain 1, which implicitly treated totalError as
 * if it lived on the SAME 0-100 scale as a single observation's error.
 * It didn't -- totalError was a SUM across every observation in the
 * trial (N_OBS_TO_RUN of them, ~15), so it saturated the max(0, ...)
 * floor almost immediately. Scaling BONUS_DECAY_OLD by N_OBS_TO_RUN
 * patched that specific bug, but the whole SUM-then-decay shape still
 * meant a single very-wrong response and many small errors could produce
 * the same totalError -- indistinguishable to the formula. The new
 * per-observation formula above sidesteps that entirely by design, not
 * just by re-tuning the same shape again.
 *
 * ERROR_MODE (chat history, config-base.js's DEFAULTS.ERROR_MODE) -- what
 * a response's error is measured AGAINST:
 *   'true_mean'    -- a single fixed value, the trial's generative mean.
 *   'running_mean' -- a per-observation value, that observation's own
 *                     cumulative mean of the RAW OBSERVED VALUES (not
 *                     responses) up to and including it -- see
 *                     computeRunningMeans below. Also changes what the
 *                     summary chart's per-row blue tick shows (draw-
 *                     performance-continuous.js) and the "true mean"/
 *                     "running mean" wording in its legend and the
 *                     tutorial-summary blue banner -- see those files' own
 *                     docstrings.
 * Binary uses the SAME functions here (computeResponseError/
 * computeResponseReward/computeTrialReward/refForObservation are already
 * fully task-agnostic -- they just compare a 0-100 response to a 0-100
 * reference, regardless of what that reference represents), plus its own
 * computeRunningRatios below (binary's analog of computeRunningMeans) and
 * its own errorMode values 'true_p'/'running_p' (chat history -- kept as
 * distinct strings from continuous's 'true_mean'/'running_mean' rather
 * than reusing those, since each task's config.js can set its OWN value
 * via config-base.js's overrides mechanism independently of the other
 * task's). This file's own name is now a little dated ("-continuous" in a
 * file used by both tasks) -- not renamed, to avoid a wide-reaching
 * import-path change across every file that already imports from here,
 * for what would otherwise be a pure naming/hygiene concern.
 */

// Reward parameters (chat history, this session -- explicitly meant to be
// iterated on, same as the old BONUS_DECAY's own "decide iteratively"
// framing) -- plain exported constants, not config-base.js's DEFAULTS,
// since they're specific to this one formula, not a general task-timing/
// UI parameter.
export const MAX_REWARD  = 3;   // cents, per observation, at normError=0
export const BONUS_DECAY = 15;

// Both tasks' responses/refs live on a 0-100 scale (continuous's slider,
// binary's percentage) -- see module docstring's "REWARD FORMULA" note.
export const MAX_POSSIBLE_ERROR = 100;

/**
 * @param {number[]} values  one trial's raw observed values, in order
 * @returns {number[]} cumulative mean of values[0..i] for each i
 */
export function computeRunningMeans(values) {
  const out = [];
  let cum = 0;
  for (let i = 0; i < (values || []).length; i++) {
    cum += values[i];
    out.push(cum / (i + 1));
  }
  return out;
}

/**
 * Binary's analog of computeRunningMeans above -- running PERCENTAGE
 * (0-100 scale, matching the response slider's own scale) of +1 ("blue")
 * draws among values[0..i], for each i. +1/-1 is urn-binary.js's own
 * value encoding (1 = blue, -1 = red -- see binary-draw-animation.js's
 * own docstring for the same convention).
 * @param {number[]} values  one trial's raw observed values (+1/-1), in order
 * @returns {number[]} running percentage of +1 values, 0-100 scale
 */
export function computeRunningRatios(values) {
  const out = [];
  let countBlue = 0;
  for (let i = 0; i < (values || []).length; i++) {
    if (values[i] === 1) countBlue++;
    out.push((countBlue / (i + 1)) * 100);
  }
  return out;
}

/**
 * Resolves what a specific observation's reference value is, given either
 * a single scalar (true_mean mode -- same value for every observation) or
 * a per-observation array (running_mean mode, from computeRunningMeans
 * above). One place for this resolution, used identically by both
 * build-trial-timeline.js and build-tutorial-timeline.js, so the two never
 * implement the branch differently.
 */
export function refForObservation(refs, i) {
  return Array.isArray(refs) ? refs[i] : refs;
}

/**
 * @param {number|null} response  null (e.g. an unresolved timed-out
 *   attempt) always yields null error, not zero -- a genuinely different
 *   thing from "responded exactly at the reference".
 * @param {number} ref  from refForObservation() above
 * @returns {number|null}  raw (NOT normalized) absolute error, 0-100 scale
 */
export function computeResponseError(response, ref) {
  return response == null ? null : Math.abs(response - ref);
}

/**
 * Per-observation reward -- see module docstring's "REWARD FORMULA".
 * @param {number|null} error  from computeResponseError() above. null
 *   (timed-out/unresolved) yields 0 reward, not a formula result -- there
 *   was no response to reward.
 * @returns {number} reward for this ONE observation, in cents, never negative
 */
export function computeResponseReward(error) {
  if (error == null) return 0;
  const normError = error / MAX_POSSIBLE_ERROR;
  return Math.max(0, MAX_REWARD * (1 - BONUS_DECAY * normError));
}

/**
 * Trial reward = SUM of each observation's own computeResponseReward()
 * (module docstring's "REWARD FORMULA") -- NOT one formula applied to a
 * pre-summed total error (that was the REPLACED per-trial shape, see
 * module docstring).
 * @param {(number|null)[]} errors  one trial's stored per-observation
 *   errors, in order (each from computeResponseError() above -- may
 *   include nulls for timed-out observations, handled by
 *   computeResponseReward()).
 * @returns {number} total reward for the trial, in cents, never negative
 */
export function computeTrialReward(errors) {
  return (errors || []).reduce((sum, error) => sum + computeResponseReward(error), 0);
}
