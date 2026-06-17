/**
 * timeline-builder.js
 * Builds and runs a jsPsych timeline from a task config object.
 *
 * Config shape:
 * {
 *   // Core
 *   sequences:            array of { trial, true_mean, true_std, values[] }
 *   practiceValues:       number[]
 *   practiceMean:         number
 *   practiceStd:          number
 *
 *   // Display flags
 *   showSliderValue:      boolean
 *   sliderDefault:        'none' | 'last' | 'value'
 *   defaultValue:         number
 *   itiMs:                number
 *   tObsMs:               number
 *   showTrialPerformance: boolean
 * }
 */

import { initJsPsych } from 'jspsych';
import jsPsychHtmlButtonResponse   from '@jspsych/plugin-html-button-response';
import ItiClockPlugin              from './plugin-iti-clock.js';
import ObservationPlugin           from './plugin-observation.js';
import PracticeObservationPlugin   from './plugin-practice-observation.js';
import PracticeSummaryPlugin       from './plugin-practice-summary.js';
import TrialSummaryPlugin          from './plugin-trial-summary.js';
import 'jspsych/css/jspsych.css';
import PracticeObservationBinaryPlugin from './plugin-practice-observation-binary.js';
import TutorialIntroContinuousPlugin    from './plugin-tutorial-intro-continuous.js';
import TutorialIntroBinaryPlugin        from './plugin-tutorial-intro-binary.js';
import PracticeSummaryBinaryPlugin  from './plugin-practice-summary-binary.js';
import ObservationBinaryPlugin      from './plugin-observation-binary.js';
import TrialSummaryBinaryPlugin     from './plugin-trial-summary-binary.js';
import './style.css';

export function buildAndRun(cfg) {
  const {
    taskType = 'continuous',
    sequences,
    practiceValues,
    practiceMean,
    practiceStd,
    showSliderValue,
    sliderDefault,
    defaultValue,
    itiMs,
    tObsMs,
    showTrialPerformance,
  } = cfg;

  const isBinary = taskType === 'binary';
  const TutorialObsPlugin     = isBinary ? PracticeObservationBinaryPlugin : PracticeObservationPlugin;
  const TutorialSummaryPlugin = isBinary ? PracticeSummaryBinaryPlugin     : PracticeSummaryPlugin;
  const TrialObsPlugin        = isBinary ? ObservationBinaryPlugin          : ObservationPlugin;
  const TrialSummaryPlugin2   = isBinary ? TrialSummaryBinaryPlugin         : TrialSummaryPlugin;

  // Prolific PID
  const urlParams   = new URLSearchParams(window.location.search);
  const prolificPID = urlParams.get('PROLIFIC_PID') || 'dev_pid';

  // jsPsych
  const jsPsych = initJsPsych({
    on_finish: () => {
      const allData = jsPsych.data.get().json();
      jatos.submitResultData(allData, () => {
        window.location.href =
          'https://app.prolific.com/submissions/complete?cc=PLACEHOLDER';
      });
    },
  });

  jsPsych.data.addProperties({ prolific_pid: prolificPID, task: taskType });

  // Helpers
  const makeButton = (label, extraStyle = '') =>
    `<button class="jspsych-btn"
      style="font-size:1.1rem;padding:0.6rem 2.5rem;${extraStyle}">${label}</button>`;

  const timeline = [];

  // ── Welcome ──────────────────────────────────────────────────────────────
  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div class="screen-wrap">
        <h2>Welcome</h2>
        <p>Thank you for taking part in this study.</p><br>
        <p>This study is conducted by
        <strong>Peter Duggins and Alireza Soltani</strong>
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

  // ── Consent ───────────────────────────────────────────────────────────────
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
          at any time by closing the browser. This will not affect your Prolific
          account.]</p><br>
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

  // ── Tutorial intro (continuous only) ────────────────────────────────────────
  // The intro shows obs 0 with the interactive reveal, then finishes with a
  // response — so the practice loop starts from obs 1 (skipping obs 0).
  const practiceStartObs = 1;  // intro handles obs 0 for both tasks
  if (!isBinary) {
    timeline.push({
      type: TutorialIntroContinuousPlugin,
      example_value: practiceValues[0],
      true_mean:     practiceMean,
      true_std:      practiceStd,
      data: { screen: 'tutorial_intro', observation: 0, value: practiceValues[0] },
      on_finish: (data) => {
        if (data.response !== null && data.response !== undefined) {
          practiceLastResponse = data.response;
          practiceResponses.push({ value: practiceValues[0], response: data.response });
        }
      },
    });
  } else {
    timeline.push({
      type: TutorialIntroBinaryPlugin,
      example_value: practiceValues[0],
      true_p:        practiceMean,
      n_obs:         practiceValues.length,
      data: { screen: 'tutorial_intro', observation: 0, value: practiceValues[0] },
      on_finish: (data) => {
        if (data.response !== null && data.response !== undefined) {
          practiceLastResponse = data.response;
          practiceResponses.push({ value: practiceValues[0], response: data.response });
        }
      },
    });
  }

  // ── Practice block ────────────────────────────────────────────────────────
  let practiceLastResponse = defaultValue;
  let practiceResponses    = [];

  for (let o = practiceStartObs; o < practiceValues.length; o++) {
    const _o     = o;
    const _value = practiceValues[o];

    const obsNode = {
      timeline: [
        {
          timeline: [{
            type: ItiClockPlugin,
            duration_ms: 250,
            data: { screen: 'practice_iti', observation: _o },
          }],
          conditional_function: () => true,
        },
        {
          type: TutorialObsPlugin,
          value:          _value,
          obs_num:        _o + 1,
          n_obs:          practiceValues.length,
          true_mean:      practiceMean,
          true_std:       practiceStd,
          slider_default: sliderDefault,
          init_pos:       () => practiceLastResponse,
          show_value:     showSliderValue,
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

  timeline.push({
    type: TutorialSummaryPlugin,
    // continuous params
    true_mean:  practiceMean,
    true_std:   practiceStd,
    // binary params
    true_p:     practiceMean,
    values:     () => practiceResponses.map(r => r.value),
    responses:  () => practiceResponses.map(r => r.response),
    data: { screen: 'practice_summary' },
  });

  // ── Trial loop ────────────────────────────────────────────────────────────
  let lastResponse      = defaultValue;
  let lastTimedOut      = false;
  let lastTrialResponses = [];

  for (let t = 0; t < sequences.length; t++) {
    const seq = sequences[t];
    lastResponse = defaultValue;
    lastTimedOut  = false;
    const trialResponses = [];
    lastTrialResponses   = trialResponses;

    for (let o = 0; o < seq.values.length; o++) {
      if (o > 0) {
        timeline.push({
          type: ItiClockPlugin,
          duration_ms: itiMs,
          timed_out:   () => lastTimedOut,
          data: { screen: 'iti', trial: t, observation: o },
        });
      }

      const initPos = sliderDefault === 'value' ? defaultValue
                    : sliderDefault === 'last'  ? lastResponse
                    : defaultValue;

      timeline.push({
        type: TrialObsPlugin,
        value:          seq.values[o],
        trial_num:      t + 1,
        n_trials:       sequences.length,
        t_obs_ms:       tObsMs,
        slider_default: sliderDefault,
        init_pos:       initPos,
        show_value:     showSliderValue,
        data: {
          screen:        'observation',
          trial:         t,
          observation:   o,
          value:         seq.values[o],
          true_mean:     seq.true_mean,
          true_std:      seq.true_std,
          true_p:        seq.true_p   ?? null,
          qid:           seq.qid      ?? null,
          trial_type:    seq.trial_type ?? null,
          prefix_length: seq.prefix_length ?? null,
          std_condition: seq.std_condition ?? null,
        },
        on_finish: (data) => {
          lastTimedOut = data.timed_out;
          if (data.response !== null) lastResponse = data.response;
          trialResponses.push({ observation: o, value: seq.values[o], response: data.response });
        },
      });
    }

    // Inter-trial summary
    const _t         = t;
    const _trueMean  = seq.true_mean;
    const _trueStd   = seq.true_std;
    const _values    = [...seq.values];
    const _responses = trialResponses;
    const _isLast    = t === sequences.length - 1;

    if (!_isLast) {
      timeline.push({
        type: TrialSummaryPlugin2,
        trial_num:        _t + 1,
        true_mean:        _trueMean,
        true_std:         _trueStd,
        true_p:           seq.true_p ?? null,
        values:           _values,
        responses:        () => _responses.map(r => r.response),
        show_performance: showTrialPerformance,
        is_last:          false,
        data: { screen: 'inter_trial', trial: _t },
      });
    }
  }

  // Final trial summary
  {
    const _t        = sequences.length - 1;
    const _seq      = sequences[_t];
    const _responses = lastTrialResponses;
    if (showTrialPerformance) {
      timeline.push({
        type: TrialSummaryPlugin2,
        trial_num:        _t + 1,
        true_mean:        _seq.true_mean,
        true_std:         _seq.true_std,
        values:           [..._seq.values],
        responses:        () => _responses.map(r => r.response),
        show_performance: true,
        is_last:          true,
        data: { screen: 'inter_trial', trial: _t },
      });
    }
  }

  // ── End ───────────────────────────────────────────────────────────────────
  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div class="screen-wrap" style="text-align:center;">
        <h2>Thank you!</h2>
        <p style="margin-top:1rem;">Your responses have been saved.</p>
      </div>`,
    choices: ['Return to Prolific to complete your submission'],
    button_html: (c) => makeButton(c),
    data: { screen: 'end' },
  });

  jsPsych.run(timeline);
}
