/**
 * plugin-trial-summary-binary.js
 * Inter-trial summary for the binary task.
 * Same bar chart as practice summary, no info boxes.
 */

import { buildSummaryBarSVG } from './bar-chart.js';

const info = {
  name: 'trial-summary-binary',
  parameters: {
    trial_num:        { type: 'INT',     default: 1     },
    true_p:           { type: 'FLOAT',   default: 0.5   },
    values:           { type: 'OBJECT',  default: []    },
    responses:        { type: 'OBJECT',  default: []    },
    show_performance: { type: 'BOOLEAN', default: false },
    is_last:          { type: 'BOOLEAN', default: false },
  },
};

class TrialSummaryBinaryPlugin {
  constructor(jsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';

    const { trial_num, true_p, values, responses,
            show_performance, is_last } = trial;

    const perfHTML = show_performance ? `
      <div id="summary-svg"
        style="display:block;margin:0 auto 1rem;
               border:1px solid #e5e7eb;border-radius:6px;
               background:#fff;padding:4px;width:fit-content;">
      </div>` : '';

    display_el.innerHTML = `
      <div class="screen-wrap" style="text-align:center;">
        <p style="font-size:1.2rem;margin-bottom:0.75rem;">
          Trial ${trial_num} complete.
        </p>
        ${perfHTML}
        <button id="next-btn" class="jspsych-btn"
          style="font-size:1.1rem;padding:0.6rem 2.5rem;">
          ${is_last ? 'Finish' : 'Next trial'}
        </button>
      </div>`;

    if (show_performance) {
      display_el.querySelector('#summary-svg').innerHTML =
        buildSummaryBarSVG(true_p, values, responses);
    }

    display_el.querySelector('#next-btn').addEventListener('pointerdown', (e) => {
      e.preventDefault();
      this.jsPsych.finishTrial({ screen: 'inter_trial', trial_num });
    });
  }
}

TrialSummaryBinaryPlugin.info = info;
export default TrialSummaryBinaryPlugin;
