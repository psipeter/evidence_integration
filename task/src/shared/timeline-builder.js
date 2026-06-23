/**
 * timeline-builder.js
 * Builds and runs a jsPsych timeline from a task config object.
 *
 * Config shape:
 * {
 *   taskType:             'continuous' | 'binary'
 *   sequences:            array of { trial, qid, true_mean, true_std, true_p,
 *                                    values[], prefix_length, iti_ms }
 *   practiceValues:       number[]
 *   practiceMean:         number  (true_mean for continuous, true_p for binary)
 *   practiceStd:          number  (continuous only)
 *   showSliderValue:      boolean
 *   sliderDefault:        'none' | 'last'
 *   defaultValue:         number  (slider initial position, 0-100)
 *   btiMs:                number  (between-trial interval, ms)
 *   itiShortMs:           number  (between-observation ITI in tutorial, ms)
 *   tObsMs:               number  (observation display timeout, ms)
 *   showTrialPerformance: boolean
 * }
 */

import { initJsPsych } from 'jspsych';
import jsPsychHtmlButtonResponse   from '@jspsych/plugin-html-button-response';
import ItiClockPlugin              from './plugin-iti-clock.js';
import InterTrialPlugin            from './plugin-inter-trial.js';
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
    btiMs,
    itiShortMs = 1000,
    tObsMs,
    showTrialPerformance,
  } = cfg;

  const isBinary = taskType === 'binary';
  const TutorialObsPlugin     = isBinary ? PracticeObservationBinaryPlugin : PracticeObservationPlugin;
  const TutorialSummaryPlugin = isBinary ? PracticeSummaryBinaryPlugin     : PracticeSummaryPlugin;
  const TrialObsPlugin        = isBinary ? ObservationBinaryPlugin          : ObservationPlugin;
  const TrialSummaryPlugin2   = isBinary ? TrialSummaryBinaryPlugin         : TrialSummaryPlugin;

  // Prolific PID — absent for pilot participants
  const urlParams   = new URLSearchParams(window.location.search);
  const prolificPID = urlParams.get('PROLIFIC_PID') || null;
  const isProlific  = prolificPID !== null;

  // jsPsych
  // ── beforeunload guard — warns participant before navigating away ──────────
  const beforeUnloadHandler = (e) => {
    e.preventDefault();
    e.returnValue = '';
  };
  window.addEventListener('beforeunload', beforeUnloadHandler);

  const jsPsych = initJsPsych({
    on_finish: () => {
      window.removeEventListener('beforeunload', beforeUnloadHandler);
      if (isProlific) {
        jatos.endStudyAndRedirect(
          'https://app.prolific.com/submissions/complete?cc=C3W3TF1O',
          jsPsych.data.get().json()
        );
      } else {
        jatos.endStudy(jsPsych.data.get().json());
      }
    },
  });

  // Use Prolific PID if present; fall back to JATOS worker ID for pilots
  const participantId = prolificPID ?? `pilot_${jatos.workerId}`;
  jsPsych.data.addProperties({ prolific_pid: participantId, task: taskType });

  // Helpers
  const makeButton = (label, extraStyle = '') =>
    `<button class="jspsych-btn"
      style="font-size:1.6rem;padding:1rem 3.5rem;${extraStyle}">${label}</button>`;

  const timeline = [];

  // ── Consent ───────────────────────────────────────────────────────────────
  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div class="consent-outer">
        <h2 style="text-align:center;margin-bottom:1.5rem;font-size:2.2rem;">Informed Consent</h2>
        <div class="consent-scroll" style="background:#fafafa;border:1px solid #d1d5db;border-radius:6px;padding:1rem;">
          <p><strong>Study title:</strong> [Placeholder Study Title]</p>
          <p><strong>Principal Investigators:</strong> Peter Duggins and Alireza Soltani,
          Dartmouth College, Department of Psychological and Brain Sciences</p>
          <p><strong>IRB Protocol:</strong> [Protocol Number]</p><br>
          <p><strong>Purpose:</strong> [Placeholder: research study about how people
          make judgments based on sequences of information.]</p><br>
          <p><strong>What you will do:</strong> [Placeholder: view a series of numbers
          and use a slider to indicate your best estimate after each. ~45 minutes.]</p><br>
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
        <div class="consent-info-boxes">
          <div class="consent-info-box">
            <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24"
              fill="none" stroke="#555" stroke-width="2" stroke-linecap="round"
              stroke-linejoin="round" style="flex-shrink:0;margin-top:1px;">
              <circle cx="12" cy="12" r="10"/>
              <polyline points="12 6 12 12 16 14"/>
            </svg>
            <span style="font-size:1.4rem;">The study takes approximately <strong>45 minutes</strong> to complete.
            You will be compensated at the rate advertised on Prolific.</span>
          </div>
          <div class="consent-info-box consent-info-box--warning">
            <span style="font-size:1.6rem;flex-shrink:0;margin-top:1px;">&#9888;</span>
            <span style="font-size:1.4rem;">Do not close, refresh, or navigate away during the task — your data
            will be lost and you will not be paid. If this happens accidentally,
            please request a return on Prolific.</span>
          </div>
        </div>
        <div class="consent-footer">
          <hr style="margin-bottom:1rem;">
          <label style="display:flex;align-items:center;justify-content:center;gap:0.75rem;cursor:pointer;">
            <input type="checkbox" id="consent-checkbox"
              style="margin-top:3px;width:18px;height:18px;flex-shrink:0;"
              onchange="document.getElementById('consent-btn').disabled=!this.checked;">
            <span style="font-size:1.4rem;">I have read and understood the information above. I am at least
            18 years old and I agree to participate in this study.</span>
          </label>
        </div>
      </div>`,
    choices: ['I agree and wish to continue'],
    button_html: (c) =>
      `<button id="consent-btn" class="jspsych-btn" disabled
        style="font-size:1.6rem;padding:1rem 3.5rem;margin-top:1.5rem;">${c}</button>`,
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
          type: ItiClockPlugin,
          duration_ms: itiShortMs,
          data: { screen: 'practice_iti', observation: _o },
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

  // Inter-trial reset before trial 1
  timeline.push({
    type:        InterTrialPlugin,
    trial_num:   1,
    n_trials:    sequences.length,
    duration_ms: btiMs,
    is_binary:   isBinary,
    data: { screen: 'inter_trial_reset', trial: -1 },
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
          duration_ms: seq.iti_ms ?? 1000,
          timed_out:   () => lastTimedOut,
          data: { screen: 'iti', trial: t, observation: o, iti_ms: seq.iti_ms ?? 1000 },
        });
      }

      timeline.push({
        type: TrialObsPlugin,
        value:          seq.values[o],
        trial_num:      t + 1,
        n_trials:       sequences.length,
        t_obs_ms:       tObsMs,
        // obs 0 of each trial: always start fresh (no carry-over)
        slider_default: o === 0 ? 'none' : sliderDefault,
        init_pos:       () => sliderDefault === 'last' && o > 0 ? lastResponse : defaultValue,
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

    const _t         = t;
    const _trueMean  = seq.true_mean;
    const _trueStd   = seq.true_std;
    const _values    = [...seq.values];
    const _responses = trialResponses;
    const _isLast    = t === sequences.length - 1;

    if (!_isLast) {
      timeline.push({
        type: TrialSummaryPlugin2,
        true_mean:        _trueMean,
        true_std:         _trueStd,
        true_p:           seq.true_p ?? null,
        values:           _values,
        responses:        () => _responses.map(r => r.response),
        show_performance: showTrialPerformance,
        is_last:          false,
        data: { screen: 'inter_trial', trial: _t },
      });
      // BTI (between-trial interval) reset screen — always 5s
      timeline.push({
        type:        InterTrialPlugin,
        trial_num:   _t + 2,
        n_trials:    sequences.length,
        duration_ms: btiMs,
        is_binary:   isBinary,
        data: { screen: 'inter_trial_reset', trial: _t },
      });
    }
  }

  // Final trial summary
  {
    const _t        = sequences.length - 1;
    const _seq      = sequences[_t];
    const _responses = lastTrialResponses;
    timeline.push({
      type: TrialSummaryPlugin2,
      true_mean:        _seq.true_mean,
      true_std:         _seq.true_std,
      values:           [..._seq.values],
      responses:        () => _responses.map(r => r.response),
      show_performance: showTrialPerformance,
      is_last:          true,
      data: { screen: 'inter_trial', trial: _t },
    });
  }

  // ── End ───────────────────────────────────────────────────────────────────
  const endStimulus = `
      <div class="screen-wrap" style="text-align:center;">
        <h2>Thank you!</h2>
        <p style="margin-top:1rem;">Your responses have been saved.</p>
        <p style="margin-top:0.75rem;background:#f0fdf4;border:1px solid #86efac;border-radius:6px;padding:0.6rem 0.75rem;">
          This is <strong>one half of a two-part study</strong>. The other part will
          appear on your Prolific dashboard shortly &mdash; please complete it today
          if possible.
        </p>
      </div>`;

  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: endStimulus,
    choices: ['Return to Prolific to complete your submission'],
    button_html: (c) => makeButton(c),
    data: { screen: 'end' },
  });

  jsPsych.run(timeline);
}
