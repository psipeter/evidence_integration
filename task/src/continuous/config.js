/**
 * config.js — continuous (Gaussian) task
 * Import this from experiment-continuous.js.
 */
import sequencesData from './sequences.json';

// ── Parameters ────────────────────────────────────────────────────────────
const N_TRIALS_TO_RUN        = 2;       // ← set to 100 for full experiment
const SHOW_SLIDER_VALUE      = true;
const SLIDER_DEFAULT         = 'none';  // 'none' | 'last' | 'value'
const DEFAULT_VALUE          = 54;
const ITI_MS                 = 1000;
const T_OBS_MS               = 5000;
const SHOW_TRIAL_PERFORMANCE = true;

const PRACTICE_N_OBS  = 5;
const PRACTICE_MEAN   = 55;
const PRACTICE_STD    = 10;
const PRACTICE_SEED   = 42;

// ── Practice sequence (seeded LCG — reproducible without a library) ────────
const practiceValues = (() => {
  let seed = PRACTICE_SEED;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  };
  const vals = [];
  while (vals.length < PRACTICE_N_OBS) {
    const v = Math.round(PRACTICE_MEAN + (rand() * 2 - 1) * PRACTICE_STD * 2.5);
    if (v >= 10 && v <= 99) vals.push(v);
  }
  return vals;
})();

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
