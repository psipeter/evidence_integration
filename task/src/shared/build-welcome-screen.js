/**
 * build-welcome-screen.js
 * Builds the very first screen participants see \u2014 a title/branding page
 * before consent. Purely a "begin" gate; no interactivity beyond the button.
 */
import jsPsychHtmlButtonResponse from '@jspsych/plugin-html-button-response';

/**
 * @param {boolean} isBinary       unused for now (kept for signature parity
 *                                  with other build-*-screen.js functions
 *                                  that branch on task type; harmless if
 *                                  this screen ever needs to again)
 * @param {string}  partLabel      e.g. 'Part A' / 'Part B' \u2014 decided by the
 *                                  caller (timeline-builder.js), not
 *                                  hardcoded here, so the continuous/binary
 *                                  <-> A/B mapping lives in one obvious place.
 * @returns {object} jsPsych timeline node
 */
export function buildWelcomeScreen(isBinary, partLabel) {
  return {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div class="welcome-wrap">
        <div id="welcome-box" class="welcome-box">
          <h1 class="welcome-title">Evidence Integration</h1>
          <div class="welcome-subtitle">${partLabel}</div>
        </div>
      </div>`,
    choices: ['Proceed to tutorial'],
    button_html: (c) =>
      `<button id="welcome-begin-btn" class="jspsych-btn welcome-begin-btn">${c}</button>`,
    data: { screen: 'welcome' },
  };
}
