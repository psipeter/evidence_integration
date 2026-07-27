/**
 * finish-session.js
 * How a session actually ends: call progress-finish (writes the terminal
 * marker row and returns the participant's Prolific code in ONE round
 * trip -- see supabase/functions/progress-finish), then show the code +
 * redirect (real Prolific participants) or show the code + a plain
 * confirmation (everyone else).
 *
 * SAVE-THEN-CONFIRM, EXPLICITLY GATED: only show a completion screen AFTER
 * progress-finish resolves successfully. On failure, the participant is
 * deliberately NOT redirected and sees an error screen instead -- the
 * whole point of this backend is to never tell a participant "you're
 * done" without confirming their data actually landed first (see
 * task_backend/TODO.md's incident history).
 *
 * THE CODE IS ALWAYS SHOWN AS VISIBLE TEXT, never just embedded silently
 * in a redirect URL. A previous version redirected immediately on success
 * with no on-screen fallback -- if that redirect ever failed (network
 * hiccup, popup blocker, Prolific-side issue), the participant would have
 * had no way to know what their own code was. Now: the code is rendered
 * first, a "Continue to Prolific" button and a short auto-redirect both
 * exist for convenience, but the code stays on-screen throughout --
 * closing the tab or a failed redirect never loses it.
 *
 * endSession is the shared chain; finishSession (below) and
 * terminateSession (terminate-session.js) are the two named entry points
 * -- kept as separate, clearly-named functions rather than one generic
 * function taking a phase flag, so call sites read as "this session
 * finished" or "this session was terminated" rather than a string
 * parameter buried in an options object.
 */
import { finishProgress } from './backend-client.js';
import { PHASES } from './phases.js';

const PROLIFIC_REDIRECT_BASE = 'https://app.prolific.com/submissions/complete?cc=';
const AUTO_REDIRECT_MS = 4000;

/**
 * Shared "here's your code" rendering -- used by endSession below AND by
 * timeline-builder.js's returning-participant screen (a completed/
 * terminated participant re-visiting the link needs the exact same
 * code-plus-redirect treatment, not a bespoke duplicate).
 *
 * @param {Element|null} contentEl  falls back to #jspsych-content, then document.body
 * @param {object} opts
 * @param {string} opts.title
 * @param {string} opts.message
 * @param {string} opts.prolificCode
 * @param {boolean} opts.isProlific  non-Prolific participants see the code
 *   too (useful for local/dev testing and manual pilot record-keeping) but
 *   get no redirect button/timer, since there's nowhere to redirect them.
 * @param {number} [opts.autoRedirectMs]  set null to disable auto-redirect
 */
export function renderCompletionScreen(contentEl, { title, message, prolificCode, isProlific, autoRedirectMs = AUTO_REDIRECT_MS }) {
  const root = contentEl ?? document.getElementById('jspsych-content') ?? document.body;
  const redirectUrl = `${PROLIFIC_REDIRECT_BASE}${prolificCode}`;

  root.innerHTML = `
    <div class="screen-wrap" style="text-align:center;">
      <h2>${title}</h2>
      <p style="margin-top:1rem;font-size:1.4rem;color:#555;">${message}</p>
      <p style="margin-top:1.5rem;font-size:1.2rem;color:#555;">Your completion code:</p>
      <div style="margin-top:0.5rem;font-size:2rem;font-weight:bold;letter-spacing:0.1em;
        font-family:monospace;user-select:all;padding:0.75rem 1.5rem;
        background:#f3f4f6;border-radius:8px;display:inline-block;">
        ${prolificCode}
      </div>
      ${isProlific ? `
        <p style="margin-top:1rem;font-size:1.1rem;color:#888;">
          If you are not redirected automatically, copy this code and enter it on Prolific.
        </p>
        <button id="continue-prolific-btn" class="jspsych-btn"
          style="font-size:1.6rem;padding:1rem 3.5rem;margin-top:1.5rem;">
          Continue to Prolific
        </button>` : `
        <p style="margin-top:1rem;font-size:1.1rem;color:#888;">You may now close this window.</p>`}
    </div>`;

  if (isProlific) {
    const go = () => { window.location.href = redirectUrl; };
    document.getElementById('continue-prolific-btn')?.addEventListener('click', go, { once: true });
    if (autoRedirectMs != null) setTimeout(go, autoRedirectMs);
  }
}

/**
 * @param {object} opts
 * @param {'finished'|'terminated'} opts.phase
 * @param {boolean} opts.isProlific
 * @param {string}  opts.prolificPid
 * @param {string}  opts.task
 * @param {number}  opts.poolIndex
 * @param {number}  [opts.expectedTrialCount]  required when phase is 'finished'
 * @param {Element} [opts.contentEl]  element to update with the completion
 *                                      or error screen
 */
export async function endSession({ phase, isProlific, prolificPid, task, poolIndex, expectedTrialCount, contentEl }) {
  try {
    const { prolificCode, dataComplete } = await finishProgress({
      prolificPid, task, poolIndex, phase, expectedTrialCount,
    });

    if (dataComplete === false) {
      // Logged, not blocking -- see progress-finish's own docstring and
      // TODO.md's "progress-finish mismatch handling" decision: don't
      // hold up a real participant's payment for our own bug, but make
      // sure it's visible for manual reconciliation.
      console.warn(`finish-session: dataComplete=false for prolific_pid=${prolificPid}`);
    }

    renderCompletionScreen(contentEl, {
      title: phase === PHASES.FINISHED ? 'Session complete' : 'Session terminated',
      message: phase === PHASES.FINISHED
        ? 'Thank you for participating. Your data has been saved.'
        : 'You reached the maximum number of timed-out responses. Your data has been saved.',
      prolificCode,
      isProlific,
    });
  } catch (err) {
    // Deliberately NOT redirecting and NOT retrying here. The participant
    // sees this on-screen; it's also logged to the console, which shows
    // up in the browser's own error reporting / any error-tracking
    // integration, rather than only being visible to a participant who
    // may just close the tab.
    console.error(`finish-session: progress-finish FAILED for prolific_pid=${prolificPid}: ${err}`);
    const root = contentEl ?? document.getElementById('jspsych-content') ?? document.body;
    root.innerHTML = `
      <div class="screen-wrap" style="text-align:center;">
        <h2>Something went wrong saving your data</h2>
        <p style="margin-top:1rem;font-size:1.4rem;color:#555;">
          Please do not close this window. If this message persists,
          contact the researcher.
        </p>
      </div>`;
  }
}

/** Normal completion -- the "Thank you" end-screen's button handler. */
export function finishSession(opts) {
  return endSession({ ...opts, phase: PHASES.FINISHED });
}
