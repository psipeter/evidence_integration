/**
 * scoring.js
 * Per-OBSERVATION error + per-OBSERVATION bonus-payment formula for the
 * numbers task (chat history).
 *
 * Error is computed ONCE per observation, at the moment that response is
 * given, and stored directly on that observation's own row (see
 * build-trial-timeline.js / build-tutorial-timeline.js's on_finish
 * callbacks). The trial-summary screen's `total_error` is then just the
 * SUM of those already-stored per-observation errors (see
 * plugin-trial-summary-numbers.js's caller) -- NOT independently
 * recomputed from responses + a reference at summary time. One source of
 * truth: whatever's stored per-observation is exactly what gets summed,
 * so the two can never drift apart from each other (e.g. a future formula
 * tweak applied in one place but not the other).
 *
 * REWARD FORMULA:
 *
 *     normError = rawError / MAX_POSSIBLE_ERROR        (scaled to 0-1)
 *     reward    = max(0, MAX_REWARD * (1 - bonusDecay * normError))
 *
 * computed PER OBSERVATION, not once per trial from a SUMMED error -- the
 * trial's own total reward is just the SUM of these per-observation
 * rewards (computeTrialReward below), the same "store once per
 * observation, sum at summary time" pattern this file already uses for
 * error itself. MAX_POSSIBLE_ERROR=100 because both tasks' responses/refs
 * live on the same bounded 0-100 scale (numbers's slider, colors's
 * percentage) -- the worst possible single response (0 vs a reference of
 * 100, or vice versa) is exactly 100 error, so dividing by 100 always
 * lands normError in [0,1] regardless of task. MAX_REWARD=2 (cents).
 *
 * bonusDecay is a REQUIRED parameter to computeResponseReward/
 * computeTrialReward below (no default) -- NUMBERS_BONUS_DECAY and
 * COLORS_BONUS_DECAY are deliberately SEPARATE constants, not one shared
 * value, and every call site must say explicitly which one it means
 * rather than silently falling back to a default that might be wrong for
 * that task. This split exists because a single shared decay value
 * doesn't stay comparable once the two tasks' underlying noise levels
 * differ: numbers' own std_fixed changed (15 -> 10, see
 * generate_sequences.py's own comment on that constant) while colors'
 * did not, and a fixed absolute error tolerance (originally ~6.7 points
 * on the 0-100 scale, from decay=15) becomes easier to stay inside on a
 * LOWER-noise numbers task even with no real change in skill -- confirmed
 * directly against real pilot data before this split existed (pilot 5,
 * std=10, showed RMSE/std_fixed ≈ 0.60 vs pilot 4's ≈ 0.68 at std=15 --
 * genuinely comparable RELATIVE precision, even though pilot 5's absolute
 * RMSE was much lower -- see chat history for the full analysis, including
 * a decay sweep (15/22/30/40/50/60) against real pilot 5 responses that
 * settled on 25 as a reasonable middle ground: fewer people hit the $5
 * manual-payment cap than at decay=15, without crushing lower performers'
 * pay as hard as decay=50 (needed to eliminate cap-hitting entirely)
 * would have). Colors' own decay is UNCHANGED at 15 -- nothing about
 * colors' task design has changed, so there was no reason to touch it.
 *
 * ACTUAL PAYMENT -- real bonus payments are given MANUALLY (outside this
 * codebase entirely) and clipped to a $5 ceiling, regardless of what this
 * formula's raw sum comes to across a full session.
 *
 * ERROR_MODE (config-base.js's DEFAULTS.ERROR_MODE) -- what a response's
 * error is measured AGAINST:
 *   'true_mean'    -- a single fixed value, the trial's generative mean.
 *   'running_mean' -- a per-observation value, that observation's own
 *                     cumulative mean of the RAW OBSERVED VALUES (not
 *                     responses) up to and including it -- see
 *                     computeRunningMeans below.
 * Colors uses the SAME functions here (computeResponseError/
 * computeResponseReward/computeTrialReward/refForObservation are already
 * fully task-agnostic -- they just compare a 0-100 response to a 0-100
 * reference, regardless of what that reference represents), plus its own
 * computeRunningRatios below (colors's analog of computeRunningMeans) and
 * its own errorMode values 'true_p'/'running_p'.
 */

// Reward parameters -- plain exported constants, not config-base.js's
// DEFAULTS, since they're specific to this one formula, not a general
// task-timing/UI parameter.
export const MAX_REWARD  = 2;   // cents, per observation, at normError=0
export const NUMBERS_BONUS_DECAY = 25;  // changed from 15 -- see module docstring's "REWARD FORMULA" note
export const COLORS_BONUS_DECAY  = 15;  // unchanged -- colors' task design hasn't changed

// Both tasks' responses/refs live on a 0-100 scale (numbers's slider,
// colors's percentage) -- see module docstring's "REWARD FORMULA" note.
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
 * Colors's analog of computeRunningMeans above -- running PERCENTAGE
 * (0-100 scale, matching the response slider's own scale) of +1 ("blue")
 * draws among values[0..i], for each i. +1/-1 is this task's own value
 * encoding (1 = blue, -1 = red).
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
 * @param {number} bonusDecay  NUMBERS_BONUS_DECAY or COLORS_BONUS_DECAY --
 *   REQUIRED, no default (see module docstring for why).
 * @returns {number} reward for this ONE observation, in cents, never negative
 */
export function computeResponseReward(error, bonusDecay) {
  if (error == null) return 0;
  const normError = error / MAX_POSSIBLE_ERROR;
  return Math.max(0, MAX_REWARD * (1 - bonusDecay * normError));
}

/**
 * Trial reward = SUM of each observation's own computeResponseReward()
 * (module docstring's "REWARD FORMULA") -- NOT one formula applied to a
 * pre-summed total error.
 * @param {(number|null)[]} errors  one trial's stored per-observation
 *   errors, in order (each from computeResponseError() above -- may
 *   include nulls for timed-out observations, handled by
 *   computeResponseReward()).
 * @param {number} bonusDecay  NUMBERS_BONUS_DECAY or COLORS_BONUS_DECAY --
 *   REQUIRED, no default (see module docstring for why).
 * @returns {number} total reward for the trial, in cents, never negative
 */
export function computeTrialReward(errors, bonusDecay) {
  return (errors || []).reduce((sum, error) => sum + computeResponseReward(error, bonusDecay), 0);
}

/**
 * computeErrorRefs -- resolves the full reference array/scalar a trial's
 * responses get compared against, given its task type and errorMode.
 * Pulled out of build-trial-timeline.js and build-tutorial-timeline.js,
 * which each had this exact branch duplicated verbatim (found during the
 * task_backend port review pass) -- one shared home for it now, so a
 * future errorMode change can't be applied to one loop and forgotten in
 * the other.
 *
 * @param {number[]} values   the trial's raw observed values, in order
 * @param {number}   trueRef  seq.true_mean (numbers) or seq.true_p (colors)
 * @param {object}   opts
 * @param {boolean}  opts.isColors
 * @param {string}   opts.errorMode  'true_mean'/'running_mean' (numbers) or 'true_p'/'running_p' (colors)
 * @returns {number|number[]}  a single scalar (fixed reference) or a
 *   per-observation array (running reference) -- resolve per-observation
 *   values via refForObservation() above either way.
 */
export function computeErrorRefs(values, trueRef, { isColors, errorMode }) {
  if (isColors) {
    return errorMode === 'running_p' ? computeRunningRatios(values) : trueRef * 100;
  }
  return errorMode === 'running_mean' ? computeRunningMeans(values) : trueRef;
}
