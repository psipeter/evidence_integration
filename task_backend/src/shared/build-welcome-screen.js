/**
 * build-welcome-screen.js
 * Builds the very first screen participants see -- a title/branding page
 * before consent. Purely a "begin" gate; no interactivity beyond the button.
 */
import jsPsychHtmlButtonResponse from '@jspsych/plugin-html-button-response';

/**
 * @param {boolean} isColors       determines the subtitle's color -- 'Numbers'
 *                                  (numbers) is blue, 'Colors' (colors) is
 *                                  red.
 * @param {string}  partLabel      e.g. 'Numbers' / 'Colors' -- decided by the
 *                                  caller (timeline-builder.js), not
 *                                  hardcoded here, so the numbers/colors
 *                                  <-> label mapping lives in one obvious
 *                                  place. Matches the study names given on
 *                                  Prolific (not the earlier internal
 *                                  'Part A'/'Part B' labels).
 * @returns {object} jsPsych timeline node
 */
export function buildWelcomeScreen(isColors, partLabel) {
  const subtitleClass = isColors ? 'welcome-subtitle-b' : 'welcome-subtitle-a';
  return {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div class="welcome-wrap">
        <div id="welcome-box" class="welcome-box">
          <h1 class="welcome-title">Evidence Integration</h1>
          <div class="welcome-subtitle ${subtitleClass}">${partLabel}</div>
        </div>
      </div>`,
    choices: ['Begin'],
    button_html: (c) =>
      `<button id="welcome-begin-btn" class="jspsych-btn welcome-begin-btn">${c}</button>`,
    data: { screen: 'welcome' },
  };
}
