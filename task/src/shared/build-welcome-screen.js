/**
 * build-welcome-screen.js
 * Builds the very first screen participants see -- a title/branding page
 * before consent. Purely a "begin" gate; no interactivity beyond the button.
 */
import jsPsychHtmlButtonResponse from '@jspsych/plugin-html-button-response';

/**
 * @param {boolean} isBinary       determines the subtitle's color -- Part A
 *                                  (continuous) is blue, Part B (binary) is red.
 * @param {string}  partLabel      e.g. 'Part A' / 'Part B' -- decided by the
 *                                  caller (timeline-builder.js), not
 *                                  hardcoded here, so the continuous/binary
 *                                  <-> A/B mapping lives in one obvious place.
 * @returns {object} jsPsych timeline node
 */
export function buildWelcomeScreen(isBinary, partLabel) {
  const subtitleClass = isBinary ? 'welcome-subtitle-b' : 'welcome-subtitle-a';
  return {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div class="welcome-wrap">
        <div id="welcome-box" class="welcome-box">
          <h1 class="welcome-title">Evidence Integration</h1>
          <div class="welcome-subtitle ${subtitleClass}">${partLabel}</div>
        </div>
      </div>`,
    choices: ['Proceed to tutorial'],
    button_html: (c) =>
      `<button id="welcome-begin-btn" class="jspsych-btn welcome-begin-btn">${c}</button>`,
    data: { screen: 'welcome' },
  };
}
