/**
 * finish-session.js
 * Single shared implementation of "how does a session actually end" for
 * both exit paths (normal completion in timeline-builder.js's on_finish,
 * and early-exit in create-early-exit.js).
 *
 * SAVE-THEN-END-THEN-REDIRECT, EXPLICITLY GATED (see chat history's
 * "leaked through" investigation -- Prolific showing completions with no
 * matching JATOS data): the previous version called
 * jatos.endStudyAndRedirect(url, data) / jatos.endStudy(data, ...) directly.
 * Tracing those through jatos.js's own source (not just its docs -- this
 * project has been burned by jatos.js docs being wrong twice before, see
 * CLAUDE.md) showed that the actual result-data upload and the small
 * "/end" ajax call that jatos.endStudy* waits on are two INDEPENDENT
 * requests. Nothing in the old call site confirmed the data upload had
 * actually succeeded before treating the session as finished and
 * redirecting a real Prolific participant to their completion code.
 *
 * Rewritten below into an explicit chain:
 *   jatos.appendResultData(data)          -- the actual data upload
 *     .then(() => jatos.endStudyWithoutRedirect(true))  -- mark finished
 *     .then(() => redirect (Prolific) / show confirmation (everyone else))
 *     .catch(() => log + show an error screen, do NOT redirect or end)
 *
 * On failure at ANY step, the participant is deliberately NOT redirected
 * to Prolific and the study is NOT marked finished -- the JATOS session is
 * left open/incomplete rather than risking a Prolific completion with no
 * matching real data. jatos.log records the failure so it's visible in
 * JATOS's own server log for manual follow-up, instead of vanishing
 * silently (which is what happened before -- neither call site attached
 * onError/onSuccess to anything).
 *
 * jatos.appendResultData (not submitResultData) is used deliberately: it
 * never overwrites, so this composes safely with the per-trial incremental
 * appends added elsewhere (timeline-builder.js).
 *
 * WHAT GETS SENT HERE: only a small completion marker (prolific_pid,
 * progress, task, pool_index), NOT jsPsych.data.get().json(). Every trial's
 * actual data -- including the final "end"/"terminated" screen's own trial,
 * which finishes before this function ever runs -- has already gone out
 * individually via timeline-builder.js's on_trial_finish hook. Sending the
 * full cumulative dataset again here would be a pure duplicate of
 * everything already appended (append never overwrites, so it wouldn't
 * corrupt anything, but it would double the row count and work against the
 * whole point of appending incrementally -- a lean, quickly-inspectable
 * JATOS results view). Append is now the ONLY way this app ever sends
 * trial data; this function's job is solely to mark completion.
 *
 * Everyone non-Prolific (local dev/test AND non-Prolific JATOS/MindProbe
 * pilots): NO redirect at all -- confirmed via a REAL pilot run that
 * redirecting to a same-origin confirmation page (the old
 * public/exit-complete.html) after ending the study fails on real JATOS:
 * "You tried to access the file .../exit-complete.html but it seems you
 * have no access rights." Confirmation is shown via a DOM update in the
 * CURRENTLY loaded page instead -- nothing new is ever fetched from JATOS
 * past this point, so there's no session-closed access boundary to cross.
 */

// Vestigial: no longer passed anywhere below (endStudyWithoutRedirect never
// shows a JATOS end-page by definition, so there's nothing left to
// disable). Still exported as `false` purely so generate_jzip.py's
// assert_show_end_page_disabled() build-time check keeps passing unchanged.
// Clean this up together when generate_jzip.py's own batch-config edit
// happens (see CLAUDE.md / chat history's consolidated change list) --
// removing it here alone would break that script's import.
export const SHOW_END_PAGE = false;

/**
 * @param {object} opts
 * @param {boolean} opts.isProlific
 * @param {string}  opts.prolificCode  Prolific completion code (unused when !isProlific)
 * @param {*}       opts.jsPsych
 * @param {Element} [opts.contentEl]   element to update with a confirmation
 *                                      or error message (omit only if the
 *                                      caller is about to replace it itself
 *                                      right after)
 * @param {string}  [opts.progress]    completion-marker label -- 'finished'
 *                                      (normal end screen) or 'terminated'
 *                                      (timeout-budget early exit). Callers
 *                                      pass this explicitly rather than it
 *                                      being inferred here, since this
 *                                      function has no other way to tell
 *                                      the two exit paths apart.
 */
export function finishSession({ isProlific, prolificCode, jsPsych, contentEl, progress = 'finished' }) {
  // First trial's data has prolific_pid/task/pool_index on it too (added via
  // jsPsych.data.addProperties before jsPsych.run() -- see timeline-builder.js),
  // so this doesn't depend on the participant having reached any particular
  // point; it's just the smallest available source for these three values.
  const first = jsPsych.data.get().values()[0] ?? {};
  const marker = {
    prolific_pid: first.prolific_pid ?? 'unknown',
    progress,
    task: first.task,
    pool_index: first.pool_index,
    is_prolific: isProlific,
  };

  jatos.appendResultData(marker)
    .then(() => jatos.endStudyWithoutRedirect(true))
    .then(() => {
      if (isProlific) {
        window.location.href = `https://app.prolific.com/submissions/complete?cc=${prolificCode}`;
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
    })
    .catch((err) => {
      // Deliberately NOT redirecting and NOT retrying here -- see module
      // docstring. The participant sees this on-screen; the failure is
      // also logged server-side (jatos.log) so it's visible in JATOS's own
      // results/log for manual follow-up, rather than only visible to a
      // participant who may just close the tab.
      jatos.log(`finishSession: save/end FAILED for prolific_pid=${marker.prolific_pid}: ${err}`);
      if (contentEl) {
        contentEl.innerHTML = `
          <div class="screen-wrap" style="text-align:center;">
            <h2>Something went wrong saving your data</h2>
            <p style="margin-top:1rem;font-size:1.4rem;color:#555;">
              Please do not close this window. If this message persists,
              contact the researcher.
            </p>
          </div>`;
      }
    });
}
