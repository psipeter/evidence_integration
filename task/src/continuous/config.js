/**
 * config.js — continuous task
 * Sequences loaded from task/sequences/continuous_sequences.json
 * (single master copy — never edit task/src/continuous/sequences.json)
 * (mean_range=-80..80, std_range=10..30, values in [-100,100])
 */
import sequencesData from '../../sequences/continuous_sequences.json';

// ── Parameters ────────────────────────────────────────────────────────────
const N_TRIALS_TO_RUN        = 2;       // ← set to n_total for full experiment
const SHOW_SLIDER_VALUE      = true;
const SLIDER_DEFAULT         = 'none';  // 'none' | 'last' | 'value'
const DEFAULT_VALUE          = 0;       // midpoint of [-100, 100]
const ITI_MS                 = 1000;
const T_OBS_MS               = 5000;
const SHOW_TRIAL_PERFORMANCE = true;

// ── Practice (tutorial) ───────────────────────────────────────────────────
// Fixed sequence: 5 values drawn from Normal(20, 25), all within 1σ of mean,
// at least 2 above and 2 below true_mean for pedagogical clarity.
// Generated with seed=5 of the constrained sampler.
const PRACTICE_MEAN = 20;
const PRACTICE_STD  = 25;
const practiceValues = [16, 27, 10, 33, -2];  // mean≈16.8, all in [-5,45]

export const config = {
  taskType:             'continuous',
  sequences:            sequencesData.slice(0, N_TRIALS_TO_RUN),
  practiceValues,
  practiceMean:         PRACTICE_MEAN,
  practiceStd:          PRACTICE_STD,
  showSliderValue:      SHOW_SLIDER_VALUE,
  sliderDefault:        SLIDER_DEFAULT,
  defaultValue:         DEFAULT_VALUE,
  itiMs:                ITI_MS,
  tObsMs:               T_OBS_MS,
  showTrialPerformance: SHOW_TRIAL_PERFORMANCE,
};
