/**
 * config-base.js
 * Shared config scaffolding for the numbers and colors task configs.
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
  // Bonus-payment parameters (chat history) -- deliberately kept here
  // alongside the other governing constants, not hardcoded in
  // scoring.js, since these are exactly the "decide iteratively"
  // knobs this object already exists for. BASE_PAYMENT_DOLLARS is fixed,
  // paid-regardless-of-performance. The per-response BONUS scheme itself
  // is computed PER OBSERVATION from a simple two-scalar formula
  // (MAX_REWARD, NUMBERS_BONUS_DECAY/COLORS_BONUS_DECAY -- see scoring.js's
  // own "REWARD FORMULA" docstring for the exact math and current
  // values, and for why the decay is split per-task now), not a
  // lookup table -- kept in scoring.js itself rather than here,
  // since it's specific to that one formula, not a general task-timing/
  // UI parameter this object otherwise holds.
  BASE_PAYMENT_DOLLARS:   10,
  // ERROR_MODE -- CONFIRMED PRODUCTION DEFAULT (chat history): 'running_mean'.
  // 'true_mean': error is |response - trueMean| for every response --
  // fixed target, never changes within a trial. 'running_mean' (the
  // actual production setting): error is |response - runningMean| where
  // runningMean is the cumulative mean of raw observed values up to that
  // point -- also changes the summary chart's per-row blue tick and the
  // "true mean"/"correct answer" wording in its legend and the
  // tutorial-summary blue banner. This was originally introduced for
  // testing before task_backend had real participants -- since
  // confirmed as the intended production choice, not a leftover test
  // setting; kept here rather than reverted. See scoring.js's own
  // docstring for the full math.
  ERROR_MODE:             'running_mean',
};

/**
 * Tutorial example (tutorialValues/tutorialMean/tutorialStd, passed into
 * buildConfig below) is no longer derived here at load time -- it comes
 * from a plain, pre-generated JSON snapshot instead
 * (tutorial_sequence_{numbers,colors}.json at the repo root), imported
 * directly by each task's own config.js. See
 * task_backend/generate_sequences.py's choose_tutorial_sequences for how
 * that fixed trial is chosen (a real trial from the production pool with
 * a genuinely large early swing in the running mean, plus continued
 * movement afterward -- not a hand-picked literal array, so it still
 * can't silently drift out of sync the way a truly hand-picked example
 * would, and not derived dynamically per-load either, so every
 * participant now sees the exact same tutorial example regardless of
 * their own pool assignment). Superseded an earlier dynamic
 * pickTutorialExample() that picked a "typical, near-midpoint" trial from
 * pool member 0 at load time -- removed entirely once both config.js
 * files stopped calling it (see git history to restore if ever needed).
 */

/**
 * buildConfig — assembles a task config object from per-task inputs.
 *
 * @param {object} opts
 * @param {'numbers'|'colors'} opts.taskType
 * @param {Array}  opts.sequencesPool  array of independent pool members
 *                                     (each itself an array of trial
 *                                     objects) -- one is selected per
 *                                     participant by timeline-builder.js's
 *                                     poolIndexForParticipant, not here.
 * @param {Array}  opts.tutorialValues fixed tutorial sequence
 * @param {number} opts.tutorialMean   true_mean (numbers) or true_p (colors)
 * @param {number} [opts.tutorialStd]  numbers only; defaults to 0
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
    basePaymentDollars:   P.BASE_PAYMENT_DOLLARS,
    errorMode:            P.ERROR_MODE,
  };
}
