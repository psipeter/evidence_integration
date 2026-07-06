/**
 * build-end-screen.js
 * Builds the final "Thank you" jsPsych timeline node, shown only if the
 * participant was not early-exited. Extracted from timeline-builder.js —
 * pure extraction, no behavior change.
 */

/**
 * @param {Function} isExited              returns true if the participant already hit the
 *                                          early-exit (session-terminated) path
 * @param {*} jsPsychHtmlButtonResponse     the jsPsych html-button-response plugin
 * @returns {object} jsPsych timeline node (wrapped with conditional_function)
 */
export function buildEndScreen(isExited, jsPsychHtmlButtonResponse) {
  const endStimulus = `
      <div class="screen-wrap" style="text-align:center;">
        <h2>Thank you!</h2>
        <p style="margin-top:1rem;">Your responses have been saved.</p>
        <p style="margin-top:0.75rem;background:#f0fdf4;border:1px solid #86efac;border-radius:6px;padding:0.6rem 0.75rem;">
          This is <strong>one half of a two-part study</strong>. The other part will
          appear on your Prolific dashboard shortly &mdash; please complete it today
          if possible.
        </p>
      </div>`;

  return {
    timeline: [{
      type: jsPsychHtmlButtonResponse,
      stimulus: endStimulus,
      choices: ['Return to Prolific to complete your submission'],
      button_html: (c) =>
        `<button class="jspsych-btn" style="font-size:1.6rem;padding:1rem 3.5rem;">${c}</button>`,
      data: { screen: 'end' },
    }],
    conditional_function: () => !isExited(),
  };
}
