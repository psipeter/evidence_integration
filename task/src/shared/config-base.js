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
 * pedagogically "typical", non-extreme example — subject to a spread check
 * on the first `n` values (>=2 above and >=2 below the mean for continuous;
 * >=2 of each color for binary), falling through to the next-closest
 * candidate if the top match fails it. The spread check exists because
 * moment-matched generation only guarantees the FULL ~15-observation
 * sequence hits the target statistics, not any small window within it —
 * a first-5 slice can occasionally look one-sided by chance even in a
 * well-behaved full sequence.
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
    const vals = trial.values.slice(0, n);
    if (isBinary) {
      const nBlue = vals.filter(v => v === 1).length;
      const nRed  = vals.filter(v => v === -1).length;
      return nBlue >= 2 && nRed >= 2;
    }
    const m = trial[field];
    const nAbove = vals.filter(v => v > m).length;
    const nBelow = vals.filter(v => v < m).length;
    return nAbove >= 2 && nBelow >= 2;
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
