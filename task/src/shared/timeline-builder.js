/**
 * timeline-builder.js
 * Builds and runs a jsPsych timeline from a task config object.
 *
 * Config shape:
 * {
 *   taskType:             'continuous' | 'binary'
 *   sequences:            array of { trial, qid, true_mean, true_p, values[], iti_ms }
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
import TimeoutDemoPlugin            from './plugin-timeout-demo.js';
import './style.css';
import { buildTrialTimeline } from './build-trial-timeline.js';

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
    distractorType = 'iti_length',
    testMode       = false,
  } = cfg;

  const isBinary = taskType === 'binary';
  const MAX_TIMEOUTS_PER_TRIAL = 3;
  const EARLY_EXIT_CODE        = 'EARLYEXIT'; // TODO: replace before publishing
  const TutorialObsPlugin     = isBinary ? PracticeObservationBinaryPlugin : PracticeObservationPlugin;
  const TutorialSummaryPlugin = isBinary ? PracticeSummaryBinaryPlugin     : PracticeSummaryPlugin;
  const TrialObsPlugin        = isBinary ? ObservationBinaryPlugin         : ObservationPlugin;

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

  const earlyExit = () => {
    window.removeEventListener('beforeunload', beforeUnloadHandler);
    const el = document.querySelector('#jspsych-content');
    if (!el) return;

    const FADE = 800;
    const showTerminated = () => {
      el.innerHTML = `
        <div class='screen-wrap' style='text-align:center;'>
          <h2>Session terminated</h2>
          <p style='margin-top:1rem;font-size:1.4rem;color:#555;'>
            You reached the maximum number of timed-out responses in one trial.
          </p>
          <p style='margin-top:0.75rem;font-size:1.4rem;color:#555;'>
            Your data has been saved and you will receive partial compensation.
          </p>
          <button id='early-exit-btn' class='jspsych-btn'
            style='font-size:1.6rem;padding:1rem 3.5rem;margin-top:2rem;'>
            Return to Prolific
          </button>
        </div>`;
      const btn = document.getElementById('early-exit-btn');
      if (btn) btn.addEventListener('pointerdown', () => {
        if (isProlific) {
          jatos.endStudyAndRedirect(
            `https://app.prolific.com/submissions/complete?cc=${EARLY_EXIT_CODE}`,
            jsPsych.data.get().json());
        } else {
          jatos.endStudy(jsPsych.data.get().json());
        }
      });
    };

    // Show "Too slow — all timeouts used" with fade-in/out/in, then terminated screen
    el.innerHTML = `
      <div class="iti-wrap" style="flex-direction:column;gap:1.2rem;">
        <span style="font-size:3rem;font-weight:bold;color:#ef4444;">Too slow</span>
        <span id="exit-pulse" style="
          font-size:2rem;font-style:italic;color:#555;
          opacity:0;transition:opacity ${FADE}ms ease;">
          0 timeouts remaining
        </span>
      </div>`;
    const pulse = el.querySelector('#exit-pulse');
    setTimeout(() => { if (pulse) pulse.style.opacity = '1'; }, 100);
    setTimeout(() => { if (pulse) pulse.style.opacity = '0'; }, 100 + FADE + 200);
    setTimeout(() => { if (pulse) pulse.style.opacity = '1'; }, 100 + FADE * 2 + 400);
    setTimeout(showTerminated, 100 + FADE * 3 + 600);
  };

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
          <div class="consent-info-box" style="border-color:#ef4444;background:#fef2f2;">
            <span style="font-size:1.6rem;flex-shrink:0;color:#ef4444;">&#9888;</span>
            <span style="font-size:1.4rem;color:#b91c1c;">
              You must respond within the ${tObsMs/1000}-second time limit.
              If you time out <strong>${MAX_TIMEOUTS_PER_TRIAL} times in one trial</strong>,
              the experiment will terminate and you will receive partial compensation.
            </span>
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

  // ── Tutorial ─────────────────────────────────────────────────────────────────
  // In test mode, show a 'Skip tutorial' button before the tutorial.
  let skipTutorial = false;
  if (testMode) {
    timeline.push({
      type: jsPsychHtmlButtonResponse,
      stimulus: `<div class='screen-wrap' style='text-align:center;'>
        <h2>Developer test mode</h2>
        <p style='margin-top:1rem;font-size:1.4rem;color:#555;'>
          20 trials · short BTI · no timeout
        </p></div>`,
      choices: ['Take tutorial', 'Skip tutorial'],
      button_html: (c) => makeButton(c),
      data: { screen: 'test_mode_choice' },
      on_finish: (data) => { skipTutorial = data.response === 1; },
    });
  }

  // ── Tutorial (conditionally skipped in test mode) ──────────────────────────
  {
    const tutorialTimeline = [];
  // ── Tutorial intro ────────────────────────────────────────────────────────────
  const practiceStartObs = 1;  // intro handles obs 0 for both tasks
  if (!isBinary) {
      tutorialTimeline.push({
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
      tutorialTimeline.push({
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

      tutorialTimeline.push(obsNode);
  }

    tutorialTimeline.push({
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

  // Timeout demo — three screens explaining the timeout mechanism
    tutorialTimeline.push({
    type:         TimeoutDemoPlugin,
    is_binary:    isBinary,
    t_obs_ms:     tObsMs,
    max_timeouts: MAX_TIMEOUTS_PER_TRIAL,
    data: { screen: 'timeout_demo' },
  });

    timeline.push({
      timeline:             tutorialTimeline,
      conditional_function: () => !skipTutorial,
    });
  }

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
  const { timeline: trialTimelineNodes, isExited } = buildTrialTimeline(
    {
      sequences, sliderDefault, defaultValue, btiMs, tObsMs,
      showSliderValue, showTrialPerformance,
      MAX_TIMEOUTS_PER_TRIAL, distractorType,
    },
    {
      ItiClockPlugin, TrialObsPlugin,
      TrialSummaryPlugin: isBinary ? TrialSummaryBinaryPlugin : TrialSummaryPlugin,
      InterTrialPlugin,
      isBinary,
    },
    jsPsych,
    earlyExit,
  );
  for (const node of trialTimelineNodes) timeline.push(node);

  // ── End ─────────────────────────────────────────────────────────────────────
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
    timeline: [{
      type: jsPsychHtmlButtonResponse,
      stimulus: endStimulus,
      choices: ['Return to Prolific to complete your submission'],
      button_html: (c) => makeButton(c),
      data: { screen: 'end' },
    }],
    conditional_function: () => !isExited(),
  });

  jsPsych.run(timeline);
}
