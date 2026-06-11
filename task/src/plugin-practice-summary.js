/**
 * plugin-practice-summary.js
 * End-of-practice summary using the same performance canvas as trial summary.
 */

import { drawPerformance } from './draw-performance.js';

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
    const finalResp = [...responses].reverse().find(r => r !== null) ?? null;

    display_el.innerHTML = `
      <div class="screen-wrap" style="text-align:center;">
        <h2 style="margin-bottom:0.75rem;">Practice complete</h2>
        <canvas id="summary-canvas" width="520" height="150"
          style="display:block;margin:0 auto 0.75rem;border:1px solid #e5e7eb;
                 border-radius:6px;background:#fff;padding:4px;">
        </canvas>
        <div class="summary-info-row">
          <p class="practice-info-block">
            In the experiment, the
            <span style="color:#2563eb;font-weight:bold;">true mean</span>
            will not be shown.
          </p>
          <p class="practice-info-block">
            In the experiment, the number of
            <span style="color:#ef4444;font-weight:bold;">observations</span>
            per trial is unknown.
          </p>
        </div>
        <button id="proceed-btn" class="jspsych-btn"
          style="font-size:1.1rem;padding:0.6rem 2.5rem;margin-top:1rem;">
          Proceed to experiment
        </button>
      </div>`;

    drawPerformance(
      display_el.querySelector('#summary-canvas'),
      true_mean, true_std, values, responses
    );

    display_el.querySelector('#proceed-btn').addEventListener('pointerdown', (e) => {
      e.preventDefault();
      this.jsPsych.finishTrial({ screen: 'practice_summary' });
    });
  }
}

PracticeSummaryPlugin.info = info;
export default PracticeSummaryPlugin;
