/**
 * timeline-builder.js
 * Builds and runs a jsPsych timeline from a task config object, resuming
 * from wherever the participant's Supabase progress record says they left
 * off (see supabase/functions/progress-check and task_backend/TODO.md's
 * four-way resume branch).
 *
 * This file is the orchestrator only — consent, tutorial, terminate, and
 * end-screen logic each live in their own module. Trial-loop logic lives
 * in build-trial-timeline.js.
 *
 * BIGGEST CHANGE FROM THE OLD JATOS PIPELINE: there is no blanket
 * on_trial_finish hook that appends every single trial's data (the old
 * pipeline needed that because JATOS had no other way to know which
 * screens mattered). Under the new schema only 3-4 checkpoint types are
 * ever logged (welcome/consent/tutorial/trial) -- those checkpoint calls
 * now live inline, right where each response is already being processed
 * (build-trial-timeline.js / build-tutorial-timeline.js's own on_finish
 * handlers, and this file's welcome/consent wrapping below). See the
 * port review-pass inventory for the full old-screen -> new-phase
 * mapping and why this dropped ~13 of the old pipeline's 17 screen tags
 * from ever needing a network call at all.
 *
 * Config shape: unchanged from the JATOS-era version -- see config-base.js.
 */

import { initJsPsych } from 'jspsych';
import jsPsychHtmlButtonResponse   from '@jspsych/plugin-html-button-response';
import ItiClockPlugin              from './plugin-iti-clock.js';
import InterTrialPlugin            from './plugin-inter-trial.js';
import ObservationNumbersPlugin           from './plugin-observation-numbers.js';
import TutorialObservationNumbersPlugin   from './plugin-tutorial-observation-numbers.js';
import TutorialSummaryNumbersPlugin       from './plugin-tutorial-summary-numbers.js';
import TrialSummaryNumbersPlugin          from './plugin-trial-summary-numbers.js';
import 'jspsych/css/jspsych.css';
import TutorialObservationColorsPlugin from './plugin-tutorial-observation-colors.js';
import TutorialIntroNumbersPlugin    from './plugin-tutorial-intro-numbers.js';
import TutorialIntroColorsPlugin        from './plugin-tutorial-intro-colors.js';
import TutorialSummaryColorsPlugin  from './plugin-tutorial-summary-colors.js';
import ObservationColorsPlugin      from './plugin-observation-colors.js';
import TrialSummaryColorsPlugin     from './plugin-trial-summary-colors.js';
import './style.css';
import { buildTrialTimeline }        from './build-trial-timeline.js';
import { buildWelcomeScreen }        from './build-welcome-screen.js';
import { buildConsentScreen }        from './build-consent-screen.js';
import { buildTutorialTimeline }     from './build-tutorial-timeline.js';
import { createTerminateSession }    from './create-terminate-session.js';
import { buildEndScreen }            from './build-end-screen.js';
import { finishSession, renderCompletionScreen } from './finish-session.js';
import { checkProgress, createCheckpointSender } from './backend-client.js';
import { PHASES } from './phases.js';

const MAX_TIMEOUTS_PER_TRIAL = 3;

/**
 * poolIndexForParticipant — deterministic, stateless hash of a participant
 * ID into [0, poolSize). No server-side counter/database needed to track
 * "who got which pool member" -- the same ID always maps to the same
 * index, on any machine, any time, just from the string itself. Using the
 * SAME formula (not seeded by task) for both numbers and colors is what
 * gives a participant the same pool index for both tasks, given both pools
 * are the same size (200).
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

/** Minimal, non-blocking banner for the consecutive-checkpoint-failure
 * warning (backend-client.js's createCheckpointSender) -- the direct fix
 * for the "silent save failure" incident this backend exists to address.
 * Does not block the task; just makes a persistent failure VISIBLE
 * instead of invisible. */
function showSaveWarning() {
  if (document.getElementById('save-warning-banner')) return;
  const banner = document.createElement('div');
  banner.id = 'save-warning-banner';
  banner.style = `
    position:fixed;top:0;left:0;right:0;z-index:9999;
    background:#fef3c7;border-bottom:2px solid #f59e0b;color:#92400e;
    text-align:center;padding:0.5rem;font-size:1.1rem;`;
  banner.textContent = "We're having trouble saving your responses. Please check your internet connection -- your progress may not be saved.";
  document.body.appendChild(banner);
}
function hideSaveWarning() {
  document.getElementById('save-warning-banner')?.remove();
}

/** Returning participant whose session already ended (finished or
 * terminated) -- shown instead of running any jsPsych timeline at all,
 * so a completed/terminated participant re-visiting the link can never
 * re-run the study or get double-paid. Reuses finish-session.js's
 * renderCompletionScreen -- same code-visible-as-text + redirect/close
 * treatment as a normal session ending, not a bespoke duplicate. */
function showReturningParticipantScreen(status, prolificCode, isProlific) {
  const isFinished = status === PHASES.FINISHED;
  renderCompletionScreen(document.getElementById('jspsych-content') ?? document.body, {
    title: isFinished ? 'You already completed this study' : 'This study session already ended',
    message: isFinished
      ? 'Your data was already saved.'
      : 'Your partial data was already saved.',
    prolificCode,
    isProlific,
  });
}

export async function buildAndRun(cfg) {
  const {
    taskType = 'numbers',
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
    errorMode = 'true_mean',
  } = cfg;

  const isColors = taskType === 'colors';

  // Prolific PID -- absent for pilot/dev participants. Read directly from
  // window.location.search: unlike JATOS's publix redirect (which used to
  // strip the original query string before this script ever ran, forcing
  // a read from jatos.urlQueryParameters instead), there's no such
  // redirect in this pipeline -- the URL the participant actually opened
  // is the one this script sees.
  const prolificPID = new URLSearchParams(window.location.search).get('PROLIFIC_PID') || null;
  const isProlific  = prolificPID !== null;
  // Dev/local fallback -- replaces the old `pilot_${jatos.workerId}`
  // convention (there's no JATOS worker ID here at all). Timestamped so
  // repeated local runs without ?PROLIFIC_PID= don't collide with each
  // other's progress rows.
  const participantId = prolificPID ?? `dev_${Date.now()}`;
  const poolIndex = poolIndexForParticipant(participantId, sequencesPool.length);

  let progress;
  try {
    progress = await checkProgress({ prolificPid: participantId, task: taskType });
  } catch (err) {
    console.error('buildAndRun: progress-check failed, starting fresh as a fallback:', err);
    progress = { status: 'new', phase: null, resumeTrialIndex: null, prolificCode: null };
  }

  if (progress.status === PHASES.FINISHED || progress.status === PHASES.TERMINATED) {
    showReturningParticipantScreen(progress.status, progress.prolificCode, isProlific);
    return;
  }

  const sequences = sequencesPool[poolIndex];
  const resumeAtTutorial = progress.status === 'in_progress' && progress.phase === PHASES.TUTORIAL;
  const resumeAtTrial    = progress.status === 'in_progress' && progress.phase === PHASES.TRIAL;
  const startTrialIndex  = resumeAtTrial ? Math.min(progress.resumeTrialIndex, sequences.length) : 0;
  const skipToTrials      = resumeAtTrial;
  const skipToTutorial    = resumeAtTutorial;

  const isColorsPlugin = isColors;
  const TutorialIntroPlugin           = isColorsPlugin ? TutorialIntroColorsPlugin       : TutorialIntroNumbersPlugin;
  const ResolvedTutorialObsPlugin     = isColorsPlugin ? TutorialObservationColorsPlugin : TutorialObservationNumbersPlugin;
  const ResolvedTutorialSummaryPlugin = isColorsPlugin ? TutorialSummaryColorsPlugin     : TutorialSummaryNumbersPlugin;
  const TrialObsPlugin                = isColorsPlugin ? ObservationColorsPlugin         : ObservationNumbersPlugin;
  const ResolvedTrialSummaryPlugin    = isColorsPlugin ? TrialSummaryColorsPlugin        : TrialSummaryNumbersPlugin;

  // ── beforeunload guard — warns participant before navigating away ──────────
  const beforeUnloadHandler = (e) => {
    e.preventDefault();
    e.returnValue = '';
  };
  window.addEventListener('beforeunload', beforeUnloadHandler);

  const checkpointer = createCheckpointSender({ onWarning: showSaveWarning, onRecovered: hideSaveWarning });
  // checkpointLedger: every TRIAL-phase checkpoint payload ever attempted
  // this session, keyed by "trialIndex-observationIndex" -- stored
  // regardless of whether the send below actually succeeded. Exists so
  // endSession (finish-session.js) can resend the EXACT original payload
  // for whatever progress-finish reports as still missing at session end,
  // without needing to re-derive it from the sequence data (which would
  // duplicate build-trial-timeline.js's own error/reward computation).
  // Only trial-phase entries are kept -- welcome/consent/tutorial
  // checkpoints aren't part of progress-finish's completeness check.
  // ~480 small objects for a full session; not cleared until the tab
  // closes, which is fine at this size.
  const checkpointLedger = new Map();
  const sendCheckpoint = (partial) => {
    if (partial.phase === PHASES.TRIAL) {
      checkpointLedger.set(`${partial.trialIndex}-${partial.observationIndex}`, partial);
    }
    return checkpointer({
      prolificPid: participantId, task: taskType, poolIndex, ...partial,
    }).catch(() => {}); // fire-and-forget from the timeline's perspective; failure handling/warning already lives in the sender itself
  };

  let isExited = () => false;

  const jsPsych = initJsPsych({
    // Exposes the current screen on <body data-screen="..."> so automated
    // tests can deterministically wait for a specific screen transition.
    on_trial_start: (trial) => {
      document.body.dataset.screen = trial?.data?.screen ?? '';
      if (trial?.data?.trial != null)       document.body.dataset.trial       = String(trial.data.trial);
      if (trial?.data?.observation != null) document.body.dataset.observation = String(trial.data.observation);
    },
    on_finish: () => {
      // Guard against the timeline's own "natural end" firing this before
      // the participant ever sees/clicks a button: once the timeout
      // budget is exhausted, all remaining trial-loop nodes become
      // conditionally skipped, so jsPsych reaches the end of its OWN
      // timeline array and fires this callback immediately -- well before
      // terminateSession's "Too slow" animation even starts. Without this
      // guard, the session-end call fires twice.
      if (isExited()) return;
      window.removeEventListener('beforeunload', beforeUnloadHandler);
      finishSession({
        isProlific, prolificPid: participantId, task: taskType, poolIndex,
        expectedTrialCount: sequences.length,
        contentEl: document.querySelector('#jspsych-content'),
        checkpointLedger, sendCheckpoint, clearSaveWarning: hideSaveWarning,
      });
    },
  });

  jsPsych.data.addProperties({ prolific_pid: participantId, task: taskType, pool_index: poolIndex });

  const terminateSession = createTerminateSession({
    beforeUnloadHandler, isProlific, prolificPid: participantId, task: taskType, poolIndex,
  });

  const timeline = [];

  if (!skipToTutorial && !skipToTrials) {
    // ── Welcome / title screen ────────────────────────────────────────────
    const welcomeNode = buildWelcomeScreen(isColors, isColors ? 'Colors' : 'Numbers');
    welcomeNode.on_finish = () => sendCheckpoint({ phase: PHASES.WELCOME, trialIndex: -1, observationIndex: -1, attempt: 0 });
    timeline.push(welcomeNode);

    // ── Consent ───────────────────────────────────────────────────────────
    const consentNode = buildConsentScreen(tObsMs, MAX_TIMEOUTS_PER_TRIAL);
    const originalConsentOnFinish = consentNode.on_finish;
    consentNode.on_finish = (data) => {
      originalConsentOnFinish?.(data);
      sendCheckpoint({ phase: PHASES.CONSENT, trialIndex: -1, observationIndex: -1, attempt: 0 });
    };
    timeline.push(consentNode);
  }

  if (!skipToTrials) {
    // ── Tutorial (always restarts from the top on resume -- see TODO.md
    //    decision #3: trial-boundary resume, tutorial is cheap to redo) ──
    const tutorialTimeline = buildTutorialTimeline(
      {
        isColors, tutorialValues, tutorialMean, tutorialStd,
        sliderDefault, defaultValue, showSliderValue,
        tObsMs, itiShortMs, errorMode,
      },
      {
        TutorialIntroPlugin,
        TutorialObsPlugin:     ResolvedTutorialObsPlugin,
        TutorialSummaryPlugin: ResolvedTutorialSummaryPlugin,
      },
      sendCheckpoint,
    );
    for (const node of tutorialTimeline) timeline.push(node);
  }

  // ── Trial loop ────────────────────────────────────────────────────────────
  // startTrialIndex === sequences.length means every trial's last
  // observation was already checkpointed but the session never reached
  // progress-finish (e.g. a crash between the last trial and the end
  // screen) -- skip straight to the end screen rather than building an
  // empty trial loop (which would otherwise show a bogus final-trial
  // summary for a trial the participant never actually ran this session).
  if (startTrialIndex < sequences.length) {
    // Transition screen -- ALWAYS shown right before the trial loop
    // begins, whether this is a fresh start (trial 1) or a resume
    // (restarting the first incomplete trial from scratch, per the
    // trial-boundary-resume design -- see TODO.md decision #3). A
    // resuming participant lands back at observation 0 of that trial,
    // same as a normal between-trial transition -- this screen makes
    // that explicit ("Trial X/32, generating new sequence…") rather than
    // silently dropping them into a slider with no framing, which could
    // otherwise look like their prior progress on that trial just
    // vanished.
    timeline.push({
      type:        InterTrialPlugin,
      trial_num:   startTrialIndex + 1,
      n_trials:    sequences.length,
      duration_ms: btiMs,
      is_colors:   isColors,
      data: { screen: 'inter_trial_reset', trial: -1 },
    });

    const { timeline: trialTimelineNodes, isExited: isExitedFromTrials } = buildTrialTimeline(
      {
        sequences, sliderDefault, defaultValue,
        btiMs, tObsMs, startTrialIndex,
        showSliderValue, showTrialPerformance,
        MAX_TIMEOUTS_PER_TRIAL, distractorType, errorMode,
      },
      {
        ItiClockPlugin, TrialObsPlugin,
        TrialSummaryPlugin: ResolvedTrialSummaryPlugin,
        InterTrialPlugin,
        isColors,
      },
      jsPsych,
      terminateSession,
      sendCheckpoint,
    );
    for (const node of trialTimelineNodes) timeline.push(node);
    isExited = isExitedFromTrials;
  }

  // ── End ─────────────────────────────────────────────────────────────────────
  timeline.push(buildEndScreen(isExited, jsPsychHtmlButtonResponse, isProlific));

  jsPsych.run(timeline);
}
