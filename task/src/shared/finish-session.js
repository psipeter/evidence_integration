/**
 * finish-session.js
 * Single shared implementation of "how does a session actually end" for
 * both exit paths (normal completion in timeline-builder.js's on_finish,
 * and early-exit in create-early-exit.js). Replaces the earlier
 * resolve-exit-url.js, which just picked a URL for jatos.endStudyAndRedirect
 * to send BOTH Prolific and non-Prolific participants to -- that turned out
 * to be broken for the non-Prolific branch (see below).
 *
 * Real Prolific participants: jatos.endStudyAndRedirect to the Prolific
 * completion URL. This is an EXTERNAL domain (app.prolific.com), entirely
 * outside anything JATOS itself controls access to.
 *
 * Everyone else (local dev/test AND non-Prolific JATOS/MindProbe pilots):
 * NO redirect at all -- confirmed via a REAL pilot run that redirecting to
 * a same-origin confirmation page (the old public/exit-complete.html) after
 * ending the study fails on real JATOS: "You tried to access the file
 * .../exit-complete.html but it seems you have no access rights." Best
 * explanation: jatos.endStudyAndRedirect ends/closes the session as part of
 * its own execution, and by the time the browser's follow-up navigation to
 * that file actually fires, JATOS's access-control layer sees the session
 * is no longer active and rejects the request for that study asset --
 * unlike the Prolific case, which was never subject to that check at all
 * since it's not a JATOS-served file in the first place. Confirmation is
 * shown via a DOM update in the CURRENTLY loaded page instead -- nothing
 * new is ever fetched from JATOS past this point, so there's no
 * session-closed access boundary to cross. jatos.endStudy(data) (single
 * call, data passed directly as the argument, no redirect) is also exactly
 * the mechanism empirically proven by task/dev-results/pilot7cont.txt.
 */

/**
 * @param {object} opts
 * @param {boolean} opts.isProlific
 * @param {string}  opts.prolificCode  Prolific completion code (unused when !isProlific)
 * @param {*}       opts.jsPsych
 * @param {Element} [opts.contentEl]   element to update with a confirmation
 *                                      message for non-Prolific participants
 *                                      (omit only if the caller is about to
 *                                      replace it itself right after)
 */
export function finishSession({ isProlific, prolificCode, jsPsych, contentEl }) {
  const data = jsPsych.data.get().json();

  if (isProlific) {
    const url = `https://app.prolific.com/submissions/complete?cc=${prolificCode}`;
    jatos.endStudyAndRedirect(url, data);
    return;
  }

  if (contentEl) {
    contentEl.innerHTML = `
      <div class="screen-wrap" style="text-align:center;">
        <h2>Session complete</h2>
        <p style="margin-top:1rem;font-size:1.4rem;color:#555;">
          Your data has been saved. You may now close this window.
        </p>
      </div>`;
  }
  jatos.endStudy(data);
}
