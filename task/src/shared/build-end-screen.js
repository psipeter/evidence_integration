/**
 * build-end-screen.js
 * Builds the final "Thank you" jsPsych timeline node, shown only if the
 * participant was not early-exited.
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
