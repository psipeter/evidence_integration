/**
 * plugin-practice-summary.js
 * End-of-practice summary using the same performance canvas as trial summary.
 */

import { buildPerformanceSVG } from './draw-performance.js';

const info = {
  name: 'practice-summary',
  parameters: {
    true_mean:  { type: 'FLOAT',  default: 54 },
    true_std:   { type: 'FLOAT',  default: 10 },
    values:     { type: 'OBJECT', default: [] },
    responses:  { type: 'OBJECT', default: [] },
  },
};

class PracticeSummaryPlugin {
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
          <p class="practice-info-block" style="text-align:center;margin:0 auto;">In the experiment, you will not see the&nbsp;<span style="color:#2563eb;font-weight:bold;">true mean</span>&nbsp;or how many&nbsp;<span style="color:#ef4444;font-weight:bold;">observations</span>&nbsp;remain.</p>
        </div>
        <button id="proceed-btn" class="jspsych-btn"
          style="font-size:1.6rem;padding:1rem 3.5rem;margin-top:1rem;">
          Next
        </button>
      </div>`;

    display_el.querySelector('#summary-svg').innerHTML = buildPerformanceSVG(true_mean, true_std, values, responses);

    display_el.querySelector('#proceed-btn').addEventListener('pointerdown', (e) => {
      e.preventDefault();
      this.jsPsych.finishTrial({ screen: 'practice_summary' });
    });
  }
}

PracticeSummaryPlugin.info = info;
export default PracticeSummaryPlugin;
