/**
 * config-base.js
 * Shared config scaffolding for the continuous and binary task configs.
 * Each per-task config.js supplies taskType, sequencesData, tutorial values,
 * and any per-task overrides; this factory fills in the common structure
 * and default parameter values so the two configs can't drift out of sync.
 */

// ── Shared defaults (identical across tasks unless overridden) ─────────────
export const DEFAULTS = {
  TEST_MODE:              false,
  N_TRIALS_TO_RUN:        24,      // 6 seqs × 4 reps
  N_OBS_TO_RUN:           15,
  SHOW_SLIDER_VALUE:      true,    // float label above thumb
  SLIDER_DEFAULT:         'none',  // 'none' | 'last'
  DEFAULT_VALUE:          50,      // midpoint of [0,100] slider
  DEBUG_FAST:             false,   // set true for fast automated testing only
  BTI_MS_TEST:            500,
  BTI_MS_PROD:            3000,
  ITI_SHORT_MS:           1000,
  T_OBS_MS:               7000,
  SHOW_TRIAL_PERFORMANCE: true,
  DISTRACTOR_TYPE:        'none',  // 'none' | 'iti_length' | 'popup'
};

/**
 * buildConfig — assembles a task config object from per-task inputs.
 *
 * @param {object} opts
 * @param {'continuous'|'binary'} opts.taskType
 * @param {Array}  opts.sequencesData   raw sequences JSON for this task
 * @param {Array}  opts.tutorialValues  fixed tutorial sequence
 * @param {number} opts.tutorialMean    true_mean (continuous) or true_p (binary)
 * @param {number} [opts.tutorialStd]   continuous only; defaults to 0
 * @param {object} [opts.overrides]     any DEFAULTS keys this task wants to override
 */
export function buildConfig({
  taskType,
  sequencesData,
  tutorialValues,
  tutorialMean,
  tutorialStd = 0,
  overrides = {},
}) {
  const P = { ...DEFAULTS, ...overrides };

  const nTrialsDefault = P.TEST_MODE ? Math.min(20, P.N_TRIALS_TO_RUN) : P.N_TRIALS_TO_RUN;
  const btiMs           = P.TEST_MODE ? P.BTI_MS_TEST : P.BTI_MS_PROD;

  return {
    taskType,
    sequences: sequencesData.map(
      s => ({ ...s, values: s.values.slice(0, P.N_OBS_TO_RUN) })
    ),
    nTrialsDefault,
    tutorialValues,
    tutorialMean,
    tutorialStd,
    showSliderValue:      P.SHOW_SLIDER_VALUE,
    sliderDefault:        P.SLIDER_DEFAULT,
    defaultValue:         P.DEFAULT_VALUE,
    btiMs,
    itiShortMs:           P.ITI_SHORT_MS,
    tObsMs:               P.T_OBS_MS,
    showTrialPerformance: P.SHOW_TRIAL_PERFORMANCE,
    distractorType:       P.DISTRACTOR_TYPE,
    testMode:             P.TEST_MODE,
  };
}
