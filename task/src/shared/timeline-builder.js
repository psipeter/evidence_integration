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
 *   sequencesPool:        array of pool members, each an array of
 *                         { trial, qid, true_mean, true_p, values[], iti_ms }
 *                         -- one member is selected per participant via
 *                         poolIndexForParticipant (deterministic hash of
 *                         their ID), not here
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
 *
 * Always runs the FULL trial set of whichever pool member gets assigned
 * (however many trials it contains), always shows the tutorial, and always
 * uses each trial's own iti_ms — there is no dev-page/override mechanism
 * anymore (removed along with index-dev.html; see CLAUDE.md).
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
import { buildWelcomeScreen }      from './build-welcome-screen.js';
import { buildConsentScreen }      from './build-consent-screen.js';
import { buildTutorialTimeline }   from './build-tutorial-timeline.js';
import { createEarlyExit }         from './create-early-exit.js';
import { buildEndScreen }          from './build-end-screen.js';
import { finishSession }          from './finish-session.js';

const MAX_TIMEOUTS_PER_TRIAL = 3;
// Real codes, filled in from the "Human Mixed Task" Prolific workspace
// project (see CLAUDE.md's "PROLIFIC_CODES" note for the full history,
// including an earlier now-superseded set tied to abandoned drafts).
// Grouped by task first (each task is a separate Prolific study with its
// own pair of codes) rather than as two same-shaped completion/early-exit
// objects side by side.
const PROLIFIC_CODES = {
  continuous: { completion: 'C1CNSEMJ', earlyExit: 'C1ARJ6LO' },
  binary:     { completion: 'C12FEFJU', earlyExit: 'C1L1GGHT' },
};

/**
 * poolIndexForParticipant — deterministic, stateless hash of a participant
 * ID into [0, poolSize). No server-side counter/database needed to track
 * "who got which pool member" -- the same ID always maps to the same
 * index, on any machine, any time, just from the string itself. Using the
 * SAME formula (not seeded by task) for both continuous and binary is what
 * gives a participant the same pool index for both tasks (decided
 * explicitly -- see chat history), given both pools are the same size
 * (200); no cross-task coordination code is needed for that beyond this
 * one shared function.
 *
 * Simple DJB2-style polynomial string hash -- not cryptographic, doesn't
 * need to be; just needs to spread participant IDs roughly uniformly
 * across the pool. `>>> 0` keeps it an unsigned 32-bit int so `% poolSize`
 * never sees a negative operand.
 */
export function poolIndexForParticipant(participantId, poolSize) {
  let hash = 5381;
  for (let i = 0; i < participantId.length; i++) {
    hash = ((hash * 33) ^ participantId.charCodeAt(i)) >>> 0;
  }
  return hash % poolSize;
}

export function buildAndRun(cfg) {
  const {
    taskType = 'continuous',
    sequencesPool,
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
  } = cfg;

  // Everything below reads jatos.* variables (workerId, urlQueryParameters),
  // which jatos.js's own reference docs say are only reliably populated
  // AFTER jatos.onLoad()'s callback fires -- reading them synchronously,
  // with no onLoad wrapper (as this function did before), is very likely
  // why several early real pilots recorded prolific_pid as literally
  // "pilot_undefined" (jatos.workerId read before it was ready) -- see
  // CLAUDE.md's "Pilot data files" note. jatos-shim.js's local dev/test
  // onLoad calls back immediately (nothing async to wait for locally), so
  // this wrapper is a no-op timing-wise outside real JATOS.
  jatos.onLoad(() => {

  const isBinary = taskType === 'binary';
  // "Resolved" prefix distinguishes these task-specific selections from the
  // plain imported class names above (e.g. TutorialSummaryContinuousPlugin
  // vs. TutorialSummaryBinaryPlugin) so nothing shadows an import.
  const TutorialIntroPlugin           = isBinary ? TutorialIntroBinaryPlugin       : TutorialIntroContinuousPlugin;
  const ResolvedTutorialObsPlugin     = isBinary ? TutorialObservationBinaryPlugin : TutorialObservationContinuousPlugin;
  const ResolvedTutorialSummaryPlugin = isBinary ? TutorialSummaryBinaryPlugin     : TutorialSummaryContinuousPlugin;
  const TrialObsPlugin                = isBinary ? ObservationBinaryPlugin         : ObservationContinuousPlugin;
  const ResolvedTrialSummaryPlugin    = isBinary ? TrialSummaryBinaryPlugin        : TrialSummaryContinuousPlugin;

  // Prolific PID -- absent for pilot participants. Read from
  // jatos.urlQueryParameters, NOT window.location.search: JATOS's own
  // internal redirect (/publix/{studyCode} -> /publix/{workerId}/
  // {studyUuid}/start, confirmed directly against this project's real
  // MindProbe deployment) strips the ORIGINAL query string from the
  // visible URL before this script ever runs, but jatos.js itself
  // captures those original parameters server-side and exposes them via
  // jatos.urlQueryParameters -- confirmed against JATOS's own "Use
  // Prolific" documentation and jatos.js reference, not assumed. Falling
  // back to window.location.search as a second check costs nothing and
  // covers any case where urlQueryParameters is empty for a reason not
  // yet seen (e.g. a future direct-link workflow that bypasses the
  // publix redirect entirely).
  const prolificPID = jatos.urlQueryParameters?.PROLIFIC_PID
    || new URLSearchParams(window.location.search).get('PROLIFIC_PID')
    || null;
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
      // Guard against the timeline's own "natural end" firing this before the
      // participant ever sees/clicks a button: once the timeout budget is
      // exhausted, all remaining trial-loop nodes become conditionally
      // skipped, so jsPsych reaches the end of its OWN timeline array and
      // fires this callback immediately — well before the early-exit
      // "Too slow" animation even starts, let alone the button click.
      // create-early-exit.js's earlyExit() is the ONLY correct place to call
      // jatos.endStudy/endStudyAndRedirect in that case; without this guard,
      // endStudy fires twice — once automatically here, once from the actual
      // button click — and on real JATOS a second call after the session is
      // already closed can silently fail to redirect (a real participant
      // reaching this path might never get sent back to Prolific to submit
      // their completion code, even though their data was saved by the
      // first call). Confirmed via mocked-jatos E2E test: without this
      // guard, endStudy is called once immediately when the timeout budget
      // is exhausted, then again on the visible button's click.
      if (isExited()) return;
      window.removeEventListener('beforeunload', beforeUnloadHandler);
      // finishSession (shared with create-early-exit.js's button handler) is
      // the single place that knows how a session actually ends -- see its
      // own docstring for why non-Prolific participants get a DOM update +
      // no redirect, not a redirect to a same-origin confirmation page
      // (confirmed broken on real JATOS: an access-rights error trying to
      // serve that file after the session had already ended).
      finishSession({
        isProlific,
        prolificCode: PROLIFIC_CODES[taskType].completion,
        jsPsych,
        contentEl: document.querySelector('#jspsych-content'),
      });
    },
  });

  // Use Prolific PID if present; fall back to JATOS worker ID for pilots.
  // Pilots draw from the pool too (decided explicitly -- see chat history),
  // rather than special-casing them onto some single fixed file, so local/
  // pilot testing exercises the exact same assignment path real participants
  // do.
  const participantId = prolificPID ?? `pilot_${jatos.workerId}`;
  const poolIndex = poolIndexForParticipant(participantId, sequencesPool.length);
  const sequences  = sequencesPool[poolIndex];
  jsPsych.data.addProperties({ prolific_pid: participantId, task: taskType, pool_index: poolIndex });

  const earlyExit = createEarlyExit({
    beforeUnloadHandler,
    isProlific,
    jsPsych,
    earlyExitCode: PROLIFIC_CODES[taskType].earlyExit,
  });

  const timeline = [];

  // ── Welcome / title screen ────────────────────────────────────────────────
  // continuous <-> 'Numbers', binary <-> 'Colors' -- decided HERE (nowhere
  // else), matching the study names given on Prolific (previously the
  // internal-only labels 'Part A'/'Part B'). Easy to find/change later if
  // the mapping should be the reverse or vary by participant instead of
  // being fixed per task build.
  timeline.push(buildWelcomeScreen(isBinary, isBinary ? 'Colors' : 'Numbers'));

  // ── Consent ───────────────────────────────────────────────────────────────
  timeline.push(buildConsentScreen(tObsMs, MAX_TIMEOUTS_PER_TRIAL));

  // ── Tutorial (always shown -- no skip mechanism anymore) ────────────────────
  const tutorialTimeline = buildTutorialTimeline(
    {
      isBinary, tutorialValues, tutorialMean, tutorialStd,
      sliderDefault, defaultValue, showSliderValue,
      tObsMs, maxTimeoutsPerTrial: MAX_TIMEOUTS_PER_TRIAL, itiShortMs,
    },
    {
      TutorialIntroPlugin,
      TutorialObsPlugin:     ResolvedTutorialObsPlugin,
      TutorialSummaryPlugin: ResolvedTutorialSummaryPlugin,
    },
  );
  for (const node of tutorialTimeline) timeline.push(node);

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
      sequences, sliderDefault, defaultValue,
      btiMs, tObsMs,
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
  timeline.push(buildEndScreen(isExited, jsPsychHtmlButtonResponse, isProlific));

  jsPsych.run(timeline);

  });
}
