/**
 * plugin-tutorial-summary-continuous.js
 * End-of-tutorial summary (continuous task) using the same performance
 * canvas as the real trial summary. The total-error/reward box (chat
 * history) sits BELOW the blue banner and ABOVE the image, matching
 * plugin-trial-summary-continuous.js's positioning exactly -- the banner
 * is the one thing that screen doesn't have.
 */

import { buildPerformanceSVG, COIN_FILL, ERROR_GREEN, MEAN_BLUE } from './draw-performance-continuous.js';

const info = {
  name: 'tutorial-summary-continuous',
  parameters: {
    true_mean:    { type: 'FLOAT' },
    true_std:     { type: 'FLOAT' },
    values:       { type: 'OBJECT', default: [] },
    responses:    { type: 'OBJECT', default: [] },
    error_mode:   { type: 'STRING', default: 'true_mean' },
    total_error:  { type: 'FLOAT',  default: 0  },
    reward:       { type: 'FLOAT',  default: 0  },
  },
};

class TutorialSummaryContinuousPlugin {
  constructor(jsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';

    const { true_mean, true_std, values, responses, error_mode, total_error, reward } = trial;
    const meanWord = error_mode === 'running_mean' ? 'running mean' : 'true mean';
    display_el.innerHTML = `
      <div class="screen-wrap" style="text-align:center;width:70vw;">
        <div class="info-banner-blue" style="max-width:900px;font-size:1.15rem;white-space:nowrap;">
          You will receive <span style="color:${COIN_FILL};font-weight:bold;">bonus payments</span> for responses that are close to the <span style="color:${MEAN_BLUE};font-weight:bold;">${meanWord}</span>.
        </div>
        <div class="tutorial-info-block" style="margin-bottom:0.5rem;">
          <span>
            <span style="color:${ERROR_GREEN};font-weight:bold;">Total error: ${total_error.toFixed(1)}</span>
            &nbsp;&bull;&nbsp;
            <span style="color:${COIN_FILL};font-weight:bold;">Bonus: ${Math.round(reward)}¢</span>
          </span>
        </div>
        <div id="summary-svg"
          style="display:block;margin:0 auto 0.75rem;
                 width:100%;border:1px solid #e5e7eb;
                 border-radius:6px;background:#fff;padding:4px;">
        </div>
        <button id="proceed-btn" class="jspsych-btn"
          style="font-size:1.6rem;padding:1rem 3.5rem;margin-top:1rem;">
          Next
        </button>
      </div>`;

    display_el.querySelector('#summary-svg').innerHTML = buildPerformanceSVG(true_mean, true_std, values, responses, error_mode);

    display_el.querySelector('#proceed-btn').addEventListener('pointerdown', (e) => {
      e.preventDefault();
      this.jsPsych.finishTrial({ screen: 'tutorial_summary', total_error, reward });
    });
  }
}

TutorialSummaryContinuousPlugin.info = info;
export default TutorialSummaryContinuousPlugin;
