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
    showTutorial   = true,
    trialItiMs     = null,  // null = use seq.iti_ms; number = override all trial ITIs
    nTrialsDefault = null,  // set in config; overridden by dev page
  } = cfg;

  // Slice sequences to the correct number of trials
  const activeSequences = nTrialsDefault != null
    ? sequences.slice(0, nTrialsDefault)
    : sequences;

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
        <div class="consent-scroll" style="background:#fafafa;border:1.5px solid #d1d5db;border-radius:6px;padding:1rem;margin-bottom:0.75rem;">
          <p style="text-align:center;"><strong>CONSENT TO TAKE PART IN RESEARCH</strong><br>Dartmouth College</p>
          <p style="text-align:center;"><strong>Study title:</strong> Reward learning, choice, and attention<br>
          <strong>Principal Investigator:</strong> Alireza Soltani</p>
          <p><strong>You are being asked to take part in a research study.  Taking part in research is voluntary.</strong></p>
          <div style="text-align:left;">
          <p style="margin-top:0.75rem;"><strong>Study Summary:</strong><br>
          The purpose of this study is to better understand how we learn from reward feedback, make decisions based on that learning, and how these processes are influenced by attention. There are no physical or psychological risks associated with this study beyond the potential tiredness from looking at a computer screen and pressing keys on a keyboard. Monitoring with an infrared camera has no known health risks.</p>
          <p>Your decision whether or not to take part will have no effect on the quality of your academic standing or job status. Please ask questions if there is anything about this study that you do not understand.</p>
          <p style="margin-top:0.75rem;"><strong>What is the purpose of this study?</strong><br>
          The purpose of the study is to explore how people learn from rewards, use what they have learned to make decisions, and how their attention is controlled and affects these processes.</p>
          <p style="margin-top:0.75rem;"><strong>Will you benefit from taking part in this study?</strong><br>
          You will not personally benefit from being in this research study.  We hope to gather information that may help people in the future.</p>
          <p style="margin-top:0.75rem;"><strong>What does this study involve?</strong><br>
          Your participation in this study may last up to 6 hours, which is divided across multiple sessions (over two or more days).</p>
          <p>During the study, you will see one or more visual cues (such as images or shapes) or no cues at all, along with additional information displayed on a computer screen. You will then select between two or more choice options based on this information. On some trials, you will receive feedback, which will tell you whether your choice was correct or incorrect, or whether it was rewarded or not. At certain points, you may also be asked to report what you have learned from the cues and how confident you are about your learning or choices. You will use a keyboard, buttons, or your eye movements to make and report your decisions, etc. The visual cues you see may include photographs, drawings, simple geometric shapes, or patterns such as sinewave gratings.</p>
          <p>We may monitor your eye movements and pupil dilation using an infrared camera placed on the table in front of you. You will be asked to rest your chin on a chinrest to minimize head movement during this process. The infrared camera works like a regular camera but is sensitive to infrared light. Additionally, you may be asked to complete a questionnaire about your learning or performance on the task.</p>
          <p style="margin-top:0.75rem;"><strong>What are the options if you do not want to take part in this study?</strong><br>
          The alternative is not to take part in this study.</p>
          <p style="margin-top:0.75rem;"><strong>If you take part in this study, what activities will be done only for research purposes?</strong><br>
          If you take part in this study, the following activities will be done only for research purposes: monitoring your eye movements, recording your choices and responses on the keyboard.</p>
          <p style="margin-top:0.75rem;"><strong>What are the risks involved with being enrolled in this study?</strong><br>
          This study does not involve any physical or psychological side effects or risks, other than potential tiredness from looking at a computer screen and pressing keys on a keyboard. You are free to stop participating at any time.</p>
          <p>There are no known health risks associated with monitoring eye movements using an infrared camera.</p>
          <p>All data collected during this study will remain confidential, and your name will not appear in any reports or publications resulting from this research.</p>
          <p style="margin-top:0.75rem;"><strong>Will my data be deidentified and used in the future for other purposes?</strong><br>
          Your data (choice and responses on the keyboard and eye movements) might be stripped of identifiers and used for future research.
          Any future research that uses your data will be reviewed by the Committee for the Protection of Human Subjects at Dartmouth College, who will determine if the research requires your permission or may be properly done without further permission from you.</p>
          <p style="margin-top:0.75rem;"><strong>Other important items you should know:</strong></p>
          <ul style="margin:0.25rem 0 0.5rem 1.5rem;padding:0;">
            <li><strong>Leaving the study:</strong>  You may choose to stop taking part in this study at any time. If you decide to stop taking part, it will have no effect on your academic standing, or job status.</li>
            <li><strong>Number of people in this study:</strong>  We expect (500) people to enroll in this study here, and (0) at other study sites.</li>
            <li><strong>Funding:</strong>  National Science Foundation provides funding to Dartmouth College for this research.</li>
          </ul>
          <p style="margin-top:0.75rem;"><strong>How will your privacy be protected?</strong></p>
          <p style="margin-top:0;"><strong>The information collected as data for this study includes:</strong><br>
          The data collected in this study include behavioral measures, such as button-press data, eye movements, and pupil dilation. Your name and age will be collected along with your answers to the questionnaire.  We will also record your gender, handedness, age, and other basic information that will not and cannot be used to identify anything private about you. We may also ask for your average household income to better understand how the incentive provided by compensation from the experiment could affect your performance.</p>
          <p>Data collected for this study will be maintained indefinitely.</p>
          <p>We are careful to protect the identities of the people in this study.  We also keep the information collected for this study secure and confidential.  Behavioral data and responses from computer-based questionnaires will be stored on password-protected computers in the lab and identified only by randomized IDs.</p>
          <p>If research data is shared electronically or physically outside the institution, all identifying information will be removed to ensure anonymity.</p>
          <p style="margin-top:0.75rem;"><strong>Will you be paid to take part in this study?</strong><br>
          Yes or no.  You will be paid for your participation based on the number of hours you participate in the experiment, your performance, and what options you choose during the experiment (overall $10-20/hour). You may also choose to instead get one T-point for every hour of experimental data collection. Because of the nature of online experiments, the overall payment for online version of the experiment will be $5-10/hour.</p>
          <p style="margin-top:0.75rem;"><strong>Whom should you call with questions about this study?</strong><br>
          If you have questions about this study or concerns about a research related problem or injury, you can call the research director for this study: Dr. Alireza Soltani (603) 646-2998 during normal business hours.</p>
          <p>If you have questions, concerns, complaints, or suggestions about human research at Dartmouth, you may call the Office of the Committee for the Protection of Human Subjects at Dartmouth College (603) 646-6482 during normal business hours.</p>
          </div>
        </div>
        <div class="consent-info-boxes">
          <div id="reveal-box-0" class="consent-info-box" style="cursor:pointer;position:relative;">
            <span id="reveal-box-0-ph" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:1.3rem;font-weight:bold;color:#1d4ed8;white-space:nowrap;">Click to reveal</span>
            <span id="reveal-box-0-real" style="visibility:hidden;font-size:1.4rem;">The study takes approximately 20 minutes to complete.
            You will be compensated at the rate advertised on Prolific.</span>
          </div>
          <div id="reveal-box-1" class="consent-info-box consent-info-box--warning" style="cursor:pointer;position:relative;">
            <span id="reveal-box-1-ph" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:1.3rem;font-weight:bold;color:#92400e;white-space:nowrap;">Click to reveal</span>
            <span id="reveal-box-1-real" style="visibility:hidden;font-size:1.4rem;">Do not close, refresh, or navigate away during the task — your data
            will be lost and you will not be paid. If this happens accidentally,
            please request a return on Prolific.</span>
          </div>
          <div id="reveal-box-2" class="consent-info-box" style="cursor:pointer;position:relative;border:1.5px solid #ef4444;background:#fef2f2;">
            <span id="reveal-box-2-ph" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:1.3rem;font-weight:bold;color:#b91c1c;white-space:nowrap;">Click to reveal</span>
            <span id="reveal-box-2-real" style="visibility:hidden;font-size:1.4rem;color:#b91c1c;">
              You must respond within the ${tObsMs/1000}-second time limit.
              If you time out ${MAX_TIMEOUTS_PER_TRIAL} times in one trial,
              the experiment will terminate and you will receive partial compensation.
            </span>
          </div>
        </div>
        <div class="consent-footer" style="margin-top:0.75rem;">
          <!-- PILOT ONLY: name field for within-participant ID (remove for Prolific production) -->
          <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.75rem;justify-content:center;">
            <label for="pilot-name" style="font-size:1.4rem;flex-shrink:0;">Name:</label>
            <input type="text" id="pilot-name" placeholder="Enter your name"
              style="font-size:1.4rem;padding:0.4rem 0.75rem;border:1.5px solid #d1d5db;
                     border-radius:6px;width:220px;"
              oninput="_updateConsentBtn()">
          </div>
          <label style="display:flex;align-items:center;justify-content:center;gap:0.75rem;cursor:pointer;">
            <input type="checkbox" id="consent-checkbox"
              style="margin-top:3px;width:18px;height:18px;flex-shrink:0;"
              onchange="_updateConsentBtn()">
            <span style="font-size:1.4rem;">I have read the above information and I agree to take part in this study.</span>
          </label>
        </div>
      </div>`,
    choices: ['Begin experiment'],
    button_html: (c) =>
      `<button id="consent-btn" class="jspsych-btn" disabled
        style="font-size:1.6rem;padding:1rem 3.5rem;margin-top:1.5rem;">${c}</button>`,
    data: { screen: 'consent' },
    on_load: () => {
      const revealed = new Set();
      let _pilotName = '';

      const revealBox = (i) => {
        document.getElementById('reveal-box-' + i + '-ph').style.display      = 'none';
        document.getElementById('reveal-box-' + i + '-real').style.visibility = 'visible';
        document.getElementById('reveal-box-' + i).style.cursor = 'default';
        revealed.add(i);
        window._updateConsentBtn();
      };

      window._updateConsentBtn = () => {
        const checked = document.getElementById('consent-checkbox')?.checked;
        const name    = document.getElementById('pilot-name')?.value.trim() || '';
        _pilotName    = name;
        window._pilotNameCapture = name;  // accessible to on_finish after DOM cleared
        const btn     = document.getElementById('consent-btn');
        if (btn) btn.disabled = !(checked && name && revealed.size === 3);
      };

      [0, 1, 2].forEach(i => {
        const box = document.getElementById('reveal-box-' + i);
        if (box) box.addEventListener('click', () => revealBox(i), { once: true });
      });
    },
    on_finish: (data) => {
      data.consent_given = true;
      // PILOT ONLY: save name as prolific_pid substitute (remove for Prolific production)
      // Name captured in on_load closure (_pilotName) since DOM is cleared before on_finish
      if (window._pilotNameCapture) {
        data.pilot_name = window._pilotNameCapture;
        window._pilotNameCapture = null;
      }
      window._updateConsentBtn = null;
    },
  });

  // Tutorial skip — controlled by showTutorial from config (set by dev setup page)
  const skipTutorial = !showTutorial;

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
          duration_ms: 1000,  // tutorial ITI always 1s
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
    n_trials:    activeSequences.length,
    duration_ms: btiMs,
    is_binary:   isBinary,
    data: { screen: 'inter_trial_reset', trial: -1 },
  });

  // ── Trial loop ────────────────────────────────────────────────────────────
  const { timeline: trialTimelineNodes, isExited } = buildTrialTimeline(
    {
      sequences: activeSequences, sliderDefault, defaultValue,
      btiMs, trialItiMs, tObsMs,
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
