/**
 * create-early-exit.js
 * Builds the earlyExit() callback used when a participant exhausts their
 * timeout budget mid-trial: shows a "too slow" pulse, then the session-
 * terminated screen with a save-and-exit button (text/behavior depend on
 * isProlific — pilots get generic wording, real Prolific participants get
 * "Return to Prolific" plus the partial-compensation note). Extracted from
 * timeline-builder.js — pure extraction, no behavior change.
 */

import { finishSession } from './finish-session.js';

/**
 * @param {object} opts
 * @param {Function} opts.beforeUnloadHandler  handler to remove once exited
 * @param {boolean}  opts.isProlific
 * @param {*}        opts.jsPsych
 * @param {string}   opts.earlyExitCode         Prolific partial-payment completion code
 * @returns {Function} earlyExit — call with no args to trigger the exit flow
 */
export function createEarlyExit({ beforeUnloadHandler, isProlific, jsPsych, earlyExitCode }) {
  return function earlyExit() {
    window.removeEventListener('beforeunload', beforeUnloadHandler);
    const el = document.querySelector('#jspsych-content');
    if (!el) return;

    const FADE = 800;
    const showTerminated = () => {
      // Mirrors on_trial_start's `document.body.dataset.screen = ...` in
      // timeline-builder.js -- this flow is a manual DOM injection outside
      // jsPsych's trial system (see module docstring), so it never fires
      // that hook on its own; setting the attribute here keeps automated
      // tests able to wait for `body[data-screen="terminated"]` the same
      // way they wait for every other screen, instead of falling back to
      // matching this screen's own copy (which is intentionally variable --
      // see isProlific below).
      document.body.dataset.screen = 'terminated';
      el.innerHTML = `
        <div class='screen-wrap' style='text-align:center;'>
          <h2>Session terminated</h2>
          <p style='margin-top:1rem;font-size:1.4rem;color:#555;'>
            You reached the maximum number of timed-out responses in one trial.
          </p>
          <p style='margin-top:0.75rem;font-size:1.4rem;color:#555;'>
            Click the button below to save your data and exit the study${isProlific ? '. You will receive partial compensation.' : '.'}
          </p>
          <button id='early-exit-btn' class='jspsych-btn'
            style='font-size:1.6rem;padding:1rem 3.5rem;margin-top:2rem;'>
            ${isProlific ? 'Return to Prolific' : 'Finish and exit'}
          </button>
        </div>`;
      const btn = document.getElementById('early-exit-btn');
      if (btn) btn.addEventListener('pointerdown', () => {
        // One-shot, matching how jsPsych's own button-response plugin
        // behaves everywhere else in this app (it disables its buttons on
        // click, so a second click can't re-fire the handler). This screen
        // is hand-rolled DOM rather than a real jsPsych trial -- jsPsych's
        // own timeline has already reached its natural end by the time a
        // participant sees it (see module docstring) -- so it doesn't get
        // that protection for free; { once: true } below plus disabling the
        // button here closes that gap explicitly. This matters more than a
        // generic double-click guard would elsewhere: calling
        // finishSession() (and therefore jatos.endStudy/endStudyAndRedirect)
        // twice is the exact bug class already documented in CLAUDE.md's
        // "Exit/redirect and data-saving architecture" -- a second call can
        // hit a DIFFERENT failure on real JATOS (session already closed),
        // not a harmless no-op.
        btn.disabled = true;
        finishSession({ isProlific, prolificCode: earlyExitCode, jsPsych, contentEl: el, progress: 'terminated' });
      }, { once: true });
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
}
