/**
 * config.js — binary (Bernoulli) task
 */

const N_TRIALS_TO_RUN        = 2;
const SHOW_SLIDER_VALUE      = true;
const SLIDER_DEFAULT         = 'none';
const DEFAULT_VALUE          = 50;   // midpoint of 0-100 probability scale
const ITI_MS                 = 1000;
const T_OBS_MS               = 5000;
const SHOW_TRIAL_PERFORMANCE = true;

const PRACTICE_N_OBS = 5;
const PRACTICE_P     = 0.65;  // true probability of blue for tutorial
const PRACTICE_SEED  = 99;

// Fixed tutorial sequence: B B R B R
const practiceValues = [1, 1, 0, 1, 0];

// Placeholder trial sequences — replace with generate_sequences_binary.py output
const PLACEHOLDER_SEQUENCES = Array.from({ length: N_TRIALS_TO_RUN }, (_, t) => {
  let seed = 1000 + t;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  };
  const p = 0.4 + rand() * 0.4;  // true_p between 0.4 and 0.8
  return {
    trial:     t,
    true_mean: p,   // true_mean = true_p for binary task
    true_std:  0,
    values:    Array.from({ length: 15 }, () => rand() < p ? 1 : 0),
  };
});

export const config = {
  taskType:             'binary',
  sequences:            PLACEHOLDER_SEQUENCES,
  practiceValues,
  practiceMean:         PRACTICE_P,
  practiceStd:          0,
  showSliderValue:      SHOW_SLIDER_VALUE,
  sliderDefault:        SLIDER_DEFAULT,
  defaultValue:         DEFAULT_VALUE,
  itiMs:                ITI_MS,
  tObsMs:               T_OBS_MS,
  showTrialPerformance: SHOW_TRIAL_PERFORMANCE,
};
