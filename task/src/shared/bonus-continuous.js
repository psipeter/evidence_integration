/**
 * bonus-continuous.js
 * Per-OBSERVATION error + per-TRIAL bonus-payment formula for the
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
 *     reward = max(0, 100 - BONUS_DECAY * totalError)
 *
 * BONUS_DECAY is the explicit "decide iteratively" knob here (chat
 * history: "we can play with different bonus_decay values when testing")
 * -- kept as a plain exported constant in this file, not config-base.js's
 * DEFAULTS, since it's specific to this one formula, not a general
 * task-timing/UI parameter.
 *
 * BUG FOUND AND FIXED (chat history): BONUS_DECAY was originally a plain
 * 1, which implicitly treated totalError as if it lived on the SAME 0-100
 * scale as a single observation's error. It doesn't -- totalError is a
 * SUM across every observation in the trial (N_OBS_TO_RUN of them, ~15),
 * so it saturates the max(0, ...) floor almost immediately: at
 * BONUS_DECAY=1, reward hits exactly 0 once the AVERAGE per-observation
 * error exceeds just 100/15 ≈ 6.67 -- a genuinely GOOD result on a 0-100
 * scale, not a bad one. This is why bonus was reported as always showing
 * 0c in the summary screen; it wasn't a display bug, the underlying
 * reward really was 0 for nearly every real response. Scaling BONUS_DECAY
 * by N_OBS_TO_RUN makes the formula behave as reward ≈ 100 -
 * average_error_per_observation instead -- 0 reward now requires an
 * AVERAGE error of 100 (i.e. consistently as wrong as possible), not an
 * average of ~7.
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
 * computeTrialReward/refForObservation are already fully task-agnostic --
 * they just compare a 0-100 response to a 0-100 reference, regardless of
 * what that reference represents), plus its own computeRunningRatios
 * below (binary's analog of computeRunningMeans) and its own errorMode
 * values 'true_p'/'running_p' (chat history -- kept as distinct strings
 * from continuous's 'true_mean'/'running_mean' rather than reusing those,
 * since each task's config.js can set its OWN value via config-base.js's
 * overrides mechanism independently of the other task's). This file's own
 * name is now a little dated ("-continuous" in a file used by both tasks)
 * -- not renamed, to avoid a wide-reaching import-path change across every
 * file that already imports from here, for what would otherwise be a pure
 * naming/hygiene concern.
 */
import { DEFAULTS } from './config-base.js';

export const BONUS_DECAY = 1 / DEFAULTS.N_OBS_TO_RUN;

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
 * @returns {number|null}
 */
export function computeResponseError(response, ref) {
  return response == null ? null : Math.abs(response - ref);
}

/**
 * @param {number} totalError  sum of stored per-observation errors for one trial
 * @returns {number} reward for the trial, in cents, never negative
 */
export function computeTrialReward(totalError) {
  return Math.max(0, 100 - BONUS_DECAY * totalError);
}
