/**
 * timeline-builder.js
 * Builds and runs a jsPsych timeline from a task config object.
 *
 * This file is the orchestrator only — consent, tutorial, early-exit, and
 * end-screen logic each live in their own module (see shared/build-*.js and
 * shared/create-early-exit.js). Trial-loop logic lives in
 * shared/build-trial-timeline.js. Splitting these out (instead of one large
 * buildAndRun) makes each piece independently readable and testable.
 *
 * Config shape:
 * {
 *   taskType:             'continuous' | 'binary'
 *   sequences:            array of { trial, qid, true_mean, true_p, values[], iti_ms }
 *   tutorialValues:       number[]
 *   tutorialMean:         number  (true_mean for continuous, true_p for binary)
 *   tutorialStd:          number  (continuous only)
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
import ObservationContinuousPlugin           from './plugin-observation-continuous.js';
import TutorialObservationContinuousPlugin   from './plugin-tutorial-observation-continuous.js';
import TutorialSummaryContinuousPlugin       from './plugin-tutorial-summary-continuous.js';
import TrialSummaryContinuousPlugin          from './plugin-trial-summary-continuous.js';
import 'jspsych/css/jspsych.css';
import TutorialObservationBinaryPlugin from './plugin-tutorial-observation-binary.js';
import TutorialIntroContinuousPlugin    from './plugin-tutorial-intro-continuous.js';
import TutorialIntroBinaryPlugin        from './plugin-tutorial-intro-binary.js';
import TutorialSummaryBinaryPlugin  from './plugin-tutorial-summary-binary.js';
import ObservationBinaryPlugin      from './plugin-observation-binary.js';
import TrialSummaryBinaryPlugin     from './plugin-trial-summary-binary.js';
import './style.css';
import { buildTrialTimeline }      from './build-trial-timeline.js';
import { buildConsentScreen }      from './build-consent-screen.js';
import { buildTutorialTimeline }   from './build-tutorial-timeline.js';
import { createEarlyExit }         from './create-early-exit.js';
import { buildEndScreen }          from './build-end-screen.js';

const MAX_TIMEOUTS_PER_TRIAL = 3;
const EARLY_EXIT_CODE        = 'EARLYEXIT'; // TODO: replace before publishing

export function buildAndRun(cfg) {
  const {
    taskType = 'continuous',
    sequences,
    tutorialValues,
    tutorialMean,
    tutorialStd,
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
  // "Resolved" prefix distinguishes these task-specific selections from the
  // plain imported class names above (e.g. TutorialSummaryContinuousPlugin
  // vs. TutorialSummaryBinaryPlugin) so nothing shadows an import.
  const TutorialIntroPlugin           = isBinary ? TutorialIntroBinaryPlugin       : TutorialIntroContinuousPlugin;
  const ResolvedTutorialObsPlugin     = isBinary ? TutorialObservationBinaryPlugin : TutorialObservationContinuousPlugin;
  const ResolvedTutorialSummaryPlugin = isBinary ? TutorialSummaryBinaryPlugin     : TutorialSummaryContinuousPlugin;
  const TrialObsPlugin                = isBinary ? ObservationBinaryPlugin         : ObservationContinuousPlugin;
  const ResolvedTrialSummaryPlugin    = isBinary ? TrialSummaryBinaryPlugin        : TrialSummaryContinuousPlugin;

  // Prolific PID — absent for pilot participants
  const urlParams   = new URLSearchParams(window.location.search);
  const prolificPID = urlParams.get('PROLIFIC_PID') || null;
  const isProlific  = prolificPID !== null;

  // ── beforeunload guard — warns participant before navigating away ──────────
  const beforeUnloadHandler = (e) => {
    e.preventDefault();
    e.returnValue = '';
  };
  window.addEventListener('beforeunload', beforeUnloadHandler);

  const jsPsych = initJsPsych({
    // Exposes the current screen on <body data-screen="..."> so automated
    // tests can deterministically wait for a specific screen transition
    // (e.g. `body[data-screen="observation"]`) instead of guessing timing
    // with fixed sleeps. Harmless in production — just a DOM attribute.
    on_trial_start: (trial) => {
      document.body.dataset.screen = trial?.data?.screen ?? '';
      if (trial?.data?.trial != null)       document.body.dataset.trial       = String(trial.data.trial);
      if (trial?.data?.observation != null) document.body.dataset.observation = String(trial.data.observation);
    },
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

  const earlyExit = createEarlyExit({
    beforeUnloadHandler,
    isProlific,
    jsPsych,
    earlyExitCode: EARLY_EXIT_CODE,
  });

  const timeline = [];

  // ── Consent ───────────────────────────────────────────────────────────────
  timeline.push(buildConsentScreen(tObsMs, MAX_TIMEOUTS_PER_TRIAL));

  // ── Tutorial (conditionally skipped — controlled by showTutorial, set by
  //    config or the dev setup page) ──────────────────────────────────────────
  const skipTutorial = !showTutorial;
  const tutorialTimeline = buildTutorialTimeline(
    {
      isBinary, tutorialValues, tutorialMean, tutorialStd,
      sliderDefault, defaultValue, showSliderValue,
      tObsMs, maxTimeoutsPerTrial: MAX_TIMEOUTS_PER_TRIAL,
    },
    {
      TutorialIntroPlugin,
      TutorialObsPlugin:     ResolvedTutorialObsPlugin,
      TutorialSummaryPlugin: ResolvedTutorialSummaryPlugin,
    },
  );
  timeline.push({
    timeline:             tutorialTimeline,
    conditional_function: () => !skipTutorial,
  });

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
      TrialSummaryPlugin: ResolvedTrialSummaryPlugin,
      InterTrialPlugin,
      isBinary,
    },
    jsPsych,
    earlyExit,
  );
  for (const node of trialTimelineNodes) timeline.push(node);

  // ── End ─────────────────────────────────────────────────────────────────────
  timeline.push(buildEndScreen(isExited, jsPsychHtmlButtonResponse));

  jsPsych.run(timeline);
}
