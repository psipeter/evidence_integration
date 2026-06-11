import { initJsPsych } from 'jspsych';
import jsPsychHtmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';
import jsPsychHtmlButtonResponse from '@jspsych/plugin-html-button-response';
import ItiClockPlugin from './plugin-iti-clock.js';
import ObservationPlugin from './plugin-observation.js';
import PracticeObservationPlugin from './plugin-practice-observation.js';
import PracticeSummaryPlugin from './plugin-practice-summary.js';
import TrialSummaryPlugin from './plugin-trial-summary.js';
import 'jspsych/css/jspsych.css';
import sequencesData from './sequences.json';

// ---------------------------------------------------------------------------
// JATOS shim
// ---------------------------------------------------------------------------
if (typeof jatos === 'undefined') {
  console.warn('[dev mode] jatos object not found — using no-op shim');
  window.jatos = {
    studySessionData: {},
    submitResultData: (data, onSuccess) => {
      console.log('[dev mode] submitResultData:', data);
      if (onSuccess) onSuccess();
    },
    endStudy: () => console.log('[dev mode] endStudy'),
  };
}

// ---------------------------------------------------------------------------
// Prolific PID
// ---------------------------------------------------------------------------
const urlParams   = new URLSearchParams(window.location.search);
const prolificPID = urlParams.get('PROLIFIC_PID') || 'dev_pid';

// ---------------------------------------------------------------------------
// Task parameters
// ---------------------------------------------------------------------------
const N_TRIALS_TO_RUN   = 3;       // ← set to 100 for full experiment
const SHOW_SLIDER_VALUE = true;    // ← false hides the numeric readout above thumb
const SLIDER_DEFAULT    = 'none';  // ← 'none' | 'last' | 'value'
const DEFAULT_VALUE     = 54;      // ← used by 'value' mode and first obs of 'last'
const ITI_MS            = 1000;    // ← inter-observation interval in ms
const SHOW_TRIAL_PERFORMANCE = true;  // ← show performance summary after each trial
const T_OBS_MS          = 5000;    // ← response deadline per observation in ms
const PRACTICE_T_OBS_MS = 10000;   // ← response deadline for practice observations
const PRACTICE_N_OBS    = 5;       // ← number of practice observations
const PRACTICE_MEAN     = 55;      // ← true mean for practice trial
const PRACTICE_STD      = 10;      // ← true std for practice trial
const PRACTICE_SEED     = 42;      // ← rng seed for reproducible practice sequence

// ---------------------------------------------------------------------------
const SEQUENCES = sequencesData.slice(0, N_TRIALS_TO_RUN);

// Practice sequence: seeded simple LCG so it's reproducible without a library
const practiceValues = (() => {
  let seed = PRACTICE_SEED;
  const rand = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
  const vals = [];
  while (vals.length < PRACTICE_N_OBS) {
    const v = Math.round(PRACTICE_MEAN + (rand() * 2 - 1) * PRACTICE_STD * 2.5);
    if (v >= 10 && v <= 99) vals.push(v);
  }
  return vals;
})();

// ---------------------------------------------------------------------------
// jsPsych init
// ---------------------------------------------------------------------------
const jsPsych = initJsPsych({
  on_finish: () => {
    const allData = jsPsych.data.get().json();
    jatos.submitResultData(allData, () => {
      window.location.href = 'https://app.prolific.com/submissions/complete?cc=PLACEHOLDER';
    });
  },
});

jsPsych.data.addProperties({ prolific_pid: prolificPID });

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const makeButton = (label, extraStyle = '') =>
  `<button class="jspsych-btn" style="font-size:1.1rem;padding:0.6rem 2.5rem;${extraStyle}">${label}</button>`;

const trialHeader = (t) =>
  `<div class="trial-counter">Trial ${t + 1} / ${SEQUENCES.length}</div>`;

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------
const timeline = [];

// --- Welcome ---
timeline.push({
  type: jsPsychHtmlButtonResponse,
  stimulus: `
    <div class="screen-wrap">
      <h2>Welcome</h2>
      <p>Thank you for taking part in this study.</p><br>
      <p>This study is conducted by <strong>Peter Duggins and Alireza Soltani</strong>
      at <strong>Dartmouth College</strong>, Department of
      <strong>Psychological and Brain Sciences</strong>.</p><br>
      <p>The study takes approximately <strong>30 minutes</strong> to complete.
      You will be compensated at the rate advertised on Prolific.</p><br>
      <p>Please click <em>Next</em> to read the consent form before beginning.</p>
    </div>`,
  choices: ['Next'],
  button_html: (c) => makeButton(c),
  data: { screen: 'welcome' },
});

// --- Consent ---
timeline.push({
  type: jsPsychHtmlButtonResponse,
  stimulus: `
    <div class="consent-outer">
      <h2 style="text-align:center;margin-bottom:1.5rem;">Informed Consent</h2>
      <div class="consent-scroll">
        <p><strong>Study title:</strong> [Placeholder Study Title]</p>
        <p><strong>Principal Investigators:</strong> Peter Duggins and Alireza Soltani,
        Dartmouth College, Department of Psychological and Brain Sciences</p>
        <p><strong>IRB Protocol:</strong> [Protocol Number]</p><br>
        <p><strong>Purpose:</strong> [Placeholder: research study about how people
        make judgments based on sequences of information.]</p><br>
        <p><strong>What you will do:</strong> [Placeholder: view a series of numbers
        and use a slider to indicate your best estimate after each. ~30 minutes.]</p><br>
        <p><strong>Risks and benefits:</strong> [Placeholder: no known risks beyond
        everyday computer use. Compensation as advertised on Prolific.]</p><br>
        <p><strong>Confidentiality:</strong> [Placeholder: responses stored by
        Prolific ID only. No personally identifying information collected.]</p><br>
        <p><strong>Voluntary participation:</strong> [Placeholder: you may withdraw
        at any time by closing the browser. This will not affect your Prolific account.]</p><br>
        <p><strong>Contact:</strong> [researcher@dartmouth.edu] |
        Dartmouth IRB: [irb@dartmouth.edu]</p>
      </div>
      <div class="consent-footer">
        <hr style="margin-bottom:1rem;">
        <label style="display:flex;align-items:flex-start;gap:0.75rem;cursor:pointer;">
          <input type="checkbox" id="consent-checkbox"
            style="margin-top:3px;width:18px;height:18px;flex-shrink:0;"
            onchange="document.getElementById('consent-btn').disabled=!this.checked;">
          <span>I have read and understood the information above. I am at least
          18 years old and I agree to participate in this study.</span>
        </label>
      </div>
    </div>`,
  choices: ['I agree and wish to continue'],
  button_html: (c) =>
    `<button id="consent-btn" class="jspsych-btn" disabled
      style="font-size:1.1rem;padding:0.6rem 2rem;margin-top:1.5rem;">${c}</button>`,
  data: { screen: 'consent' },
  on_finish: (data) => { data.consent_given = true; },
});

// --- Instructions ---
timeline.push({
  type: jsPsychHtmlButtonResponse,
  stimulus: `
    <div class="screen-wrap">
      <h2>Instructions</h2>
      <p>On each trial, you will see a sequence of <strong>15 numbers</strong>,
      one at a time.</p><br>
      <p>Each number is drawn from the <strong>same hidden distribution</strong>
      for that trial. Your job is to estimate the <strong>average</strong> of
      that distribution after seeing each number.</p><br>
      <p>After each number, use the <strong>slider</strong> to indicate your
      current best estimate of the average, then click <strong>Submit</strong>
      or press <strong>Space</strong> to confirm.</p><br>
      <p>The slider ranges from <strong>10</strong> to <strong>99</strong>.</p><br>
      <p>There are <strong>${N_TRIALS_TO_RUN} trials</strong> in total.</p>
    </div>`,
  choices: ['Start practice trial'],
  button_html: (c) => makeButton(c),
  data: { screen: 'instructions' },
});

// ---------------------------------------------------------------------------
// Practice block
// ---------------------------------------------------------------------------
let practiceLastResponse  = DEFAULT_VALUE;
let practiceResponses     = [];  // {value, response} per completed obs

for (let o = 0; o < PRACTICE_N_OBS; o++) {
  const _o     = o;
  const _value = practiceValues[o];

  // Each observation is wrapped in a loop node so a timeout repeats it
  const obsNode = {
    timeline: [
      // ITI between observations (skip before first)
      {
        timeline: [{
          type: ItiClockPlugin,
          duration_ms: 250,
          data: { screen: 'practice_iti', observation: _o },
        }],
        conditional_function: () => _o > 0,
      },
      // Observation
      {
        type: PracticeObservationPlugin,
        value:          _value,
        obs_num:        _o + 1,
        n_obs:          PRACTICE_N_OBS,
        true_mean:      PRACTICE_MEAN,
        true_std:       PRACTICE_STD,
        slider_default: SLIDER_DEFAULT,
        init_pos:       () => practiceLastResponse,
        show_value:     SHOW_SLIDER_VALUE,
        data: { screen: 'practice_observation', observation: _o, value: _value },
        on_finish: (data) => {
          if (data.response !== null) {
            practiceLastResponse = data.response;
            practiceResponses.push({ value: _value, response: data.response });
          }
        },
      },
    ],

  };

  timeline.push(obsNode);
}

// Practice summary
timeline.push({
  type: PracticeSummaryPlugin,
  true_mean:  PRACTICE_MEAN,
  true_std:   PRACTICE_STD,
  values:     () => practiceResponses.map(r => r.value),
  responses:  () => practiceResponses.map(r => r.response),
  data: { screen: 'practice_summary' },
});

// ---------------------------------------------------------------------------
// Trial loop
// ---------------------------------------------------------------------------
let lastResponse = DEFAULT_VALUE;
  let lastTimedOut = false;

for (let t = 0; t < SEQUENCES.length; t++) {
  const seq = SEQUENCES[t];
  lastResponse = DEFAULT_VALUE;
  const trialResponses = [];  // {observation, value, response} for each obs
    lastTimedOut = false;

  for (let o = 0; o < seq.values.length; o++) {
    const value = seq.values[o];

    // ITI before every observation except the first of each trial
    if (o > 0) {
      timeline.push({
        type: ItiClockPlugin,
        duration_ms: ITI_MS,
        timed_out: () => lastTimedOut,
        data: { screen: 'iti', trial: t, observation: o },
      });
    }

    // Determine initial slider position
    const initPos = (SLIDER_DEFAULT === 'value') ? DEFAULT_VALUE
                  : (SLIDER_DEFAULT === 'last')  ? lastResponse
                  : DEFAULT_VALUE;

    timeline.push({
      type: ObservationPlugin,
      value:          value,
      trial_num:      t + 1,
      n_trials:       SEQUENCES.length,
      t_obs_ms:       T_OBS_MS,
      slider_default: SLIDER_DEFAULT,
      init_pos:       initPos,
      show_value:     SHOW_SLIDER_VALUE,
      data: {
        screen:      'observation',
        trial:       t,
        observation: o,
        value:       value,
        true_mean:   seq.true_mean,
        true_std:    seq.true_std,
      },
      on_finish: (data) => {
        lastTimedOut = data.timed_out;
        if (data.response !== null) lastResponse = data.response;
        trialResponses.push({
          observation: o,
          value: seq.values[o],
          response: data.response,
        });
      },
    });
  }

  // Inter-trial summary screen
  if (t < SEQUENCES.length - 1) {
    // Snapshot values into const so the closure captures correctly
    const _t         = t;
    const _trueMean  = seq.true_mean;
    const _trueStd   = seq.true_std;
    const _values    = [...seq.values];
    const _responses = trialResponses;   // array built up by on_finish above

    timeline.push({
      type: TrialSummaryPlugin,
      trial_num:        _t + 1,
      true_mean:        _trueMean,
      true_std:         _trueStd,
      values:           _values,
      responses:        () => _responses.map(r => r.response),
      show_performance: SHOW_TRIAL_PERFORMANCE,
      data: { screen: 'inter_trial', trial: _t },
    });
  }
}

// --- Final trial summary (if flag set) ---
{
  const _t         = SEQUENCES.length - 1;
  const _seq       = SEQUENCES[_t];
  // trialResponses is still in scope from the last iteration of the loop
  const _responses = trialResponses;

  if (SHOW_TRIAL_PERFORMANCE) {
    timeline.push({
      type: TrialSummaryPlugin,
      trial_num:        _t + 1,
      true_mean:        _seq.true_mean,
      true_std:         _seq.true_std,
      values:           [..._seq.values],
      responses:        () => _responses.map(r => r.response),
      show_performance: true,
      data: { screen: 'inter_trial', trial: _t },
    });
  }
}

// --- End ---
timeline.push({
  type: jsPsychHtmlKeyboardResponse,
  stimulus: `
    <div class="screen-wrap" style="text-align:center;">
      <h2>Thank you!</h2>
      <p style="margin-top:1rem;">Your responses have been saved.</p>
      <p style="margin-top:1rem;color:#666;">
        Press any key to return to Prolific and complete your submission.
      </p>
    </div>`,
  data: { screen: 'end' },
});

jsPsych.run(timeline);
