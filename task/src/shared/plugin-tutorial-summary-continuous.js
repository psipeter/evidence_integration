/**
 * plugin-tutorial-summary-continuous.js
 * End-of-tutorial summary (continuous task) using the same performance
 * canvas as the real trial summary.
 */

import { buildPerformanceSVG } from './draw-performance-continuous.js';

const info = {
  name: 'tutorial-summary-continuous',
  parameters: {
    true_mean:  { type: 'FLOAT',  default: 54 },
    true_std:   { type: 'FLOAT',  default: 10 },
    values:     { type: 'OBJECT', default: [] },
    responses:  { type: 'OBJECT', default: [] },
  },
};

class TutorialSummaryContinuousPlugin {
  constructor(jsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';

    const { true_mean, true_std, values, responses } = trial;
    display_el.innerHTML = `
      <div class="screen-wrap" style="text-align:center;width:70vw;">
        <div id="summary-svg"
          style="display:block;margin:0 auto 0.75rem;
                 width:100%;border:1px solid #e5e7eb;
                 border-radius:6px;background:#fff;padding:4px;">
        </div>
        <div style="margin-bottom:0.75rem;">
        </div>
        <button id="proceed-btn" class="jspsych-btn"
          style="font-size:1.6rem;padding:1rem 3.5rem;margin-top:1rem;">
          Next
        </button>
      </div>`;

    display_el.querySelector('#summary-svg').innerHTML = buildPerformanceSVG(true_mean, true_std, values, responses);

    display_el.querySelector('#proceed-btn').addEventListener('pointerdown', (e) => {
      e.preventDefault();
      this.jsPsych.finishTrial({ screen: 'tutorial_summary' });
    });
  }
}

TutorialSummaryContinuousPlugin.info = info;
export default TutorialSummaryContinuousPlugin;
