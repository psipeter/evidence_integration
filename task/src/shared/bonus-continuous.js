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
 */

export const BONUS_DECAY = 1;

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
