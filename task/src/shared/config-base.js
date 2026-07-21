/**
 * config-base.js
 * Shared config scaffolding for the continuous and binary task configs.
 * Each per-task config.js supplies taskType, sequencesData, tutorial values,
 * and any per-task overrides; this factory fills in the common structure
 * and default parameter values so the two configs can't drift out of sync.
 */

// ── Shared defaults (identical across tasks unless overridden) ─────────────
export const DEFAULTS = {
  N_OBS_TO_RUN:           15,
  SHOW_SLIDER_VALUE:      true,    // float label above thumb
  SLIDER_DEFAULT:         'last',  // 'none' | 'last' -- 'last' shows the
  // thumb pre-positioned at the participant's previous response the
  // instant a new observation loads, rather than starting invisible until
  // they first interact (init_pos already always uses lastResponse either
  // way -- this only controls whether that position is VISIBLE up front).
  // Changed from 'none' (chat history): helps participants remember where
  // they left their running estimate before the new number arrived.
  DEFAULT_VALUE:          50,      // midpoint of [0,100] slider
  BTI_MS:                 3000,
  ITI_SHORT_MS:           1000,
  T_OBS_MS:               7000,
  SHOW_TRIAL_PERFORMANCE: true,
  DISTRACTOR_TYPE:        'none',  // 'none' | 'iti_length' | 'popup'
  // Bonus-payment parameters (chat history) -- deliberately kept here
  // alongside the other governing constants, not hardcoded in
  // bonus-continuous.js, since these are exactly the "decide iteratively"
  // knobs this object already exists for. BASE_PAYMENT_DOLLARS is fixed,
  // paid-regardless-of-performance. The per-response BONUS scheme itself
  // (fixed cent bins on raw error, no session-wide total cap -- see
  // bonus-continuous.js's own docstring for why the original
  // total-capped/split-evenly design was replaced) has its own tunable
  // table (BONUS_BINS_CENTS) living in bonus-continuous.js instead of here,
  // since it's a bonus-specific data structure, not a single scalar.
  BASE_PAYMENT_DOLLARS:   10,
  // ERROR_MODE (chat history) -- 'true_mean': error is |response -
  // trueMean| for every response. 'running_mean' (set for testing, chat
  // history): error is |response - runningMean| where runningMean is the
  // cumulative mean of raw observed values up to that point -- also
  // changes the summary chart's per-row blue tick and the "true mean"/
  // "running mean" wording in its legend and the tutorial-summary blue
  // banner. See bonus-continuous.js's own docstring for the full math.
  ERROR_MODE:             'running_mean',
};

/**
 * pickTutorialExample — derives the tutorial's illustrative sequence from a
 * REAL trial in the main-task sequences, rather than a hand-picked literal
 * array. Hand-picked examples silently drift out of sync whenever
 * sequences.json is regenerated with different mean_range/p_range/std_fixed
 * (this happened once already — see CLAUDE.md's sequence-generation section).
 * Deriving from real data means the tutorial can never mismatch the actual
 * task parameters again.
 *
 * Selection: among all trials, find the one whose true_mean (continuous) or
 * true_p (binary) is closest to the midpoint (50 / 0.5) — i.e. a
 * pedagogically "typical", non-extreme example — subject to two checks on
 * the first `n` values, falling through to the next-closest candidate if
 * the top match fails either:
 *   1. Spread: >=2 above and >=2 below the mean for continuous; >=2 of each
 *      color for binary. Exists because moment-matched generation only
 *      guarantees the FULL ~15-observation sequence hits the target
 *      statistics, not any small window within it — a first-5 slice can
 *      occasionally look one-sided by chance even in a well-behaved full
 *      sequence.
 *   2. Directional consistency: the shown slice's OWN apparent direction
 *      (which color is more frequent, for binary; which side of the
 *      midpoint the slice's sample mean falls on, for continuous) must
 *      match the true parameter's actual direction. Spread alone doesn't
 *      guarantee this — e.g. 3 blue/2 red passes the spread check just
 *      fine even when true_p favors red, which visually teaches the exact
 *      opposite of what the tutorial is trying to demonstrate (confirmed
 *      as a real occurrence, not just a theoretical risk: qid 2 in the
 *      6-level binary design, true_p=0.4, drew a first-5 slice of
 *      3 blue/2 red). Skipped when the true value sits exactly at the
 *      midpoint, since there's no direction to be inconsistent with.
 *
 * For continuous, tutorialStd is the sample std of the FULL chosen trial
 * (not just the shown 5-value slice) — this tracks the true generative
 * std_fixed closely under moment-matched generation (verified: within
 * ~0.5 of nominal even at range edges), avoiding a second hardcoded
 * constant that would need to be kept in sync separately from std_fixed.
 */
export function pickTutorialExample(sequencesData, { isBinary, n = 5 } = {}) {
  const field  = isBinary ? 'true_p' : 'true_mean';
  const target = isBinary ? 0.5 : 50;

  const candidates = [...sequencesData].sort(
    (a, b) => Math.abs(a[field] - target) - Math.abs(b[field] - target)
  );

  const passesSpread = (trial) => {
    const vals     = trial.values.slice(0, n);
    const trueSide = Math.sign(trial[field] - target);

    if (isBinary) {
      const nBlue = vals.filter(v => v === 1).length;
      const nRed  = vals.filter(v => v === -1).length;
      if (!(nBlue >= 2 && nRed >= 2)) return false;
      if (trueSide === 0) return true;
      return Math.sign(nBlue - nRed) === trueSide;
    }

    const m       = trial[field];
    const nAbove  = vals.filter(v => v > m).length;
    const nBelow  = vals.filter(v => v < m).length;
    if (!(nAbove >= 2 && nBelow >= 2)) return false;
    if (trueSide === 0) return true;
    const sliceMean = vals.reduce((a, b) => a + b, 0) / vals.length;
    return Math.sign(sliceMean - target) === trueSide;
  };

  const chosen = candidates.find(passesSpread) ?? candidates[0];

  const values = chosen.values.slice(0, n);
  const mean   = chosen[field];
  let std = 0;
  if (!isBinary) {
    const full = chosen.values;
    const m    = full.reduce((a, b) => a + b, 0) / full.length;
    std        = Math.sqrt(full.reduce((a, b) => a + (b - m) ** 2, 0) / full.length);
  }
  return { values, mean, std };
}

/**
 * buildConfig — assembles a task config object from per-task inputs.
 *
 * @param {object} opts
 * @param {'continuous'|'binary'} opts.taskType
 * @param {Array}  opts.sequencesPool  array of independent pool members
 *                                     (each itself an array of trial
 *                                     objects) -- one is selected per
 *                                     participant by timeline-builder.js's
 *                                     poolIndexForParticipant, not here.
 * @param {Array}  opts.tutorialValues fixed tutorial sequence
 * @param {number} opts.tutorialMean   true_mean (continuous) or true_p (binary)
 * @param {number} [opts.tutorialStd]  continuous only; defaults to 0
 * @param {object} [opts.overrides]    any DEFAULTS keys this task wants to override
 */
export function buildConfig({
  taskType,
  sequencesPool,
  tutorialValues,
  tutorialMean,
  tutorialStd = 0,
  overrides = {},
}) {
  const P = { ...DEFAULTS, ...overrides };

  return {
    taskType,
    // Every pool member, trimmed to N_OBS_TO_RUN -- no slicing of WHICH
    // trials, only how many observations within each. Trial count per
    // member is fully implicit from however many trials that pool member's
    // JSON actually contains (matches the single-sequence design's own
    // "never silently drift out of sync" property, just applied per member).
    // timeline-builder.js selects cfg.sequences = cfg.sequencesPool[i] once
    // it knows which participant this is -- config-base.js/config.js don't
    // know about participant identity at all, deliberately (keeps sequence
    // *data* and participant *assignment* fully decoupled).
    sequencesPool: sequencesPool.map(
      pool => pool.map(s => ({ ...s, values: s.values.slice(0, P.N_OBS_TO_RUN) }))
    ),
    tutorialValues,
    tutorialMean,
    tutorialStd,
    showSliderValue:      P.SHOW_SLIDER_VALUE,
    sliderDefault:        P.SLIDER_DEFAULT,
    defaultValue:         P.DEFAULT_VALUE,
    btiMs:                P.BTI_MS,
    itiShortMs:           P.ITI_SHORT_MS,
    tObsMs:               P.T_OBS_MS,
    showTrialPerformance: P.SHOW_TRIAL_PERFORMANCE,
    distractorType:       P.DISTRACTOR_TYPE,
    basePaymentDollars:   P.BASE_PAYMENT_DOLLARS,
    errorMode:            P.ERROR_MODE,
  };
}
