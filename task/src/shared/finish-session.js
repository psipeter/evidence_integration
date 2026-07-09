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
 *
 * CORRECTNESS NOTE (found during a generate_jzip.py review, not by any
 * local/E2E test -- see below for why): per JATOS's own jatos.js reference,
 * jatos.endStudy's showEndPage parameter DEFAULTS TO TRUE -- "redirects to
 * an end page (either the JATOS default one or the one configured in the
 * study properties) after the study is finished." Calling jatos.endStudy(data)
 * with nothing else, as this file did previously, leaves that default in
 * place, meaning JATOS itself -- not this app's JS -- may redirect
 * non-Prolific participants away from the "Session complete" screen below,
 * to either JATOS's generic end page or whatever's in the study's End
 * Redirect URL property (see generate_jzip.py). This is a SEPARATE,
 * platform-level completion mechanism that the local dev shim
 * (jatos-shim.js) has no representation of at all, since it only mocks
 * jatos.js's client-side API, not server-side study-property enforcement --
 * so no amount of local/E2E testing against the shim could ever have caught
 * this; it can only be confirmed (or ruled out) by an actual MindProbe run.
 * `pilot7cont.txt` only proves data saved correctly, not that no redirect
 * happened afterward -- those are independent facts that were previously
 * conflated. Fixed below by explicitly passing showEndPage=false, so the
 * JS remains the single source of truth for redirect behavior in both
 * local dev and real JATOS, matching the "thin wrapper, identical
 * behavior" principle generate_jzip.py should also follow (see its own
 * endRedirectUrl comment).
 *
 * SHOW_END_PAGE is exported (not just a literal `false` inline) so that
 * generate_jzip.py can import this exact module at build time and read the
 * real value, rather than either duplicating the assumption as a separate
 * constant (which could silently drift) or regex-matching this file's
 * source text (fragile to reformatting). If this ever needs to become
 * `true` for some future reason, that is a deliberate design change that
 * also requires deciding what generate_jzip.py's endRedirectUrl SHOULD be
 * at that point -- there's no automatically-correct value to derive, so
 * changing this constant alone is not sufficient; see CLAUDE.md's
 * "Two independent completion mechanisms" note for the full picture.
 */
export const SHOW_END_PAGE = false;

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
  // (data, successful=true, message=undefined, showEndPage=SHOW_END_PAGE) --
  // SHOW_END_PAGE=false is the actual fix; successful/message are passed
  // through as their own documented defaults rather than left to chance,
  // since jatos.js checks each positional argument independently and this
  // is the only way to reach the 4th one. See the module docstring's
  // CORRECTNESS NOTE for why this matters, and why SHOW_END_PAGE is a named
  // export rather than an inline literal here.
  jatos.endStudy(data, true, undefined, SHOW_END_PAGE);
}
