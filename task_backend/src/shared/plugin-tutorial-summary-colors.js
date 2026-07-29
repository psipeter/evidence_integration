/**
 * plugin-tutorial-summary-colors.js
 * End-of-tutorial summary for the colors task — bar chart + info boxes.
 * Mirrors plugin-tutorial-summary-numbers.js's identical pattern for
 * the blue banner + Total error/Bonus box -- see that file and bonus-
 * numbers.js's own docstrings for the full formula/rationale.
 */

import { buildSummaryBarSVG } from './draw-performance-colors.js';
import { COIN_FILL, ERROR_GREEN } from './draw-performance-numbers.js';
import { DIST_COLOR } from './tutorial-text-colors.js';

const info = {
  name: 'tutorial-summary-colors',
  parameters: {
    true_p:       { type: 'FLOAT' },
    values:       { type: 'OBJECT', default: []  },
    responses:    { type: 'OBJECT', default: []  },
    error_mode:   { type: 'STRING', default: 'true_p' },
    total_error:  { type: 'FLOAT',  default: 0  },
    reward:       { type: 'FLOAT',  default: 0  },
  },
};

class TutorialSummaryColorsPlugin {
  constructor(jsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    const { true_p, values, responses, error_mode, total_error, reward } = trial;
    // 'running_p' mode says "correct answer" here (chat history) --
    // mirrors numbers's identical meanWord swap, unifying with the
    // tutorial's own live correct-answer panel terminology rather than
    // "running percentage".
    const refWord = error_mode === 'running_p' ? 'correct answer' : 'true probability';

    display_el.innerHTML = `
      <div class="screen-wrap" style="text-align:center;width:70vw;">
        <div class="info-banner-blue" style="max-width:900px;font-size:1.15rem;white-space:nowrap;">
          You will receive <span style="color:${COIN_FILL};font-weight:bold;">bonus payments</span> for responses that are close to the <span style="color:${DIST_COLOR};font-weight:bold;">${refWord}</span>.
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

    display_el.querySelector('#summary-svg').innerHTML =
      buildSummaryBarSVG(true_p, values, responses, error_mode);

    display_el.querySelector('#proceed-btn').addEventListener('pointerdown', (e) => {
      e.preventDefault();
      this.jsPsych.finishTrial({ screen: 'tutorial_summary', total_error, reward });
    });
  }
}

TutorialSummaryColorsPlugin.info = info;
export default TutorialSummaryColorsPlugin;
