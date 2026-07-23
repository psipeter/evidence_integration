/**
 * build-end-screen.js
 * Builds the final "Thank you" jsPsych timeline node, shown only if the
 * participant was not early-exited.
 *
 * REMINDER (chat history, not yet implemented) -- actual bonus payments
 * are given manually, clipped to a $5 ceiling regardless of what the raw
 * per-observation reward formula sums to across the whole session (see
 * bonus-continuous.js's own "REWARD FORMULA" docstring for that formula,
 * and its own note on the theoretical ceiling exceeding $5 at the current
 * parameters). Since the clipping happens manually, outside this
 * codebase, a participant currently has no way to see what they actually
 * earned anywhere -- each per-trial summary screen shows that ONE
 * trial's reward, but nothing sums across the whole session, and this
 * end screen shows no reward info at all. Worth considering: a note HERE
 * showing their total earned bonus (pre-clip, with a caveat that the
 * actual payment may be capped) would close that gap. Every trial's data
 * already carries a `reward` field (see build-trial-timeline.js's
 * on_finish handlers), so the total is a simple jsPsych.data.get() sum
 * away -- this file would just need jsPsych passed in (it currently
 * isn't) to query it. Not implemented -- flagging for a future pass, not
 * blocking anything currently.
 */

/**
 * @param {Function} isExited              returns true if the participant already hit the
 *                                          early-exit (session-terminated) path
 * @param {*} jsPsychHtmlButtonResponse     the jsPsych html-button-response plugin
 * @param {boolean} isProlific              real Prolific participant vs pilot — the
 *                                          two-part-study/dashboard messaging and
 *                                          "Return to Prolific" button text only make
 *                                          sense for real Prolific participants; pilots
 *                                          get generic wording instead
 * @returns {object} jsPsych timeline node (wrapped with conditional_function)
 */
export function buildEndScreen(isExited, jsPsychHtmlButtonResponse, isProlific) {
  const endStimulus = `
      <div class="screen-wrap" style="text-align:center;">
        <h2>Thank you!</h2>
        <p style="margin-top:1rem;">
          Click the button below to save your responses and finish the study.
        </p>
        ${isProlific ? `
        <p style="margin-top:0.75rem;background:#f0fdf4;border:1px solid #86efac;border-radius:6px;padding:0.6rem 0.75rem;">
          This is <strong>one half of a two-part study</strong>. The other part will
          appear on your Prolific dashboard shortly &mdash; please complete it today
          if possible.
        </p>` : ''}
      </div>`;

  return {
    timeline: [{
      type: jsPsychHtmlButtonResponse,
      stimulus: endStimulus,
      choices: [isProlific ? 'Return to Prolific to complete your submission' : 'Finish and submit'],
      button_html: (c) =>
        `<button class="jspsych-btn" style="font-size:1.6rem;padding:1rem 3.5rem;">${c}</button>`,
      data: { screen: 'end' },
    }],
    conditional_function: () => !isExited(),
  };
}
