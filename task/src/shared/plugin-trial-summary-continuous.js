/**
 * plugin-trial-summary-continuous.js
 * Inter-trial summary — continuous task.
 * Identical layout to plugin-tutorial-summary-continuous.js; no title, no text box.
 */
import { buildPerformanceSVG } from './draw-performance-continuous.js';

const info = {
  name: 'trial-summary-continuous',
  parameters: {
    true_mean:        { type: 'FLOAT'   },
    true_std:         { type: 'FLOAT'   },
    values:           { type: 'OBJECT',  default: []    },
    responses:        { type: 'OBJECT',  default: []    },
    show_performance: { type: 'BOOLEAN', default: false },
    is_last:          { type: 'BOOLEAN', default: false },
  },
};

class TrialSummaryContinuousPlugin {
  constructor(jsPsych) { this.jsPsych = jsPsych; }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    const { true_mean, true_std, values, responses,
            show_performance, is_last, trial_num } = trial;

    display_el.innerHTML = `
      <div style="width:70vw;margin:0 auto;text-align:center;">
        ${show_performance ? `
        <div id="summary-svg"
          style="display:block;margin:0 auto 0.75rem;
                 width:100%;border:1px solid #e5e7eb;
                 border-radius:6px;background:#fff;padding:4px;">
        </div>` : ''}
        <button id="next-btn" class="jspsych-btn"
          style="font-size:1.6rem;padding:1rem 3.5rem;margin-top:1rem;">
          ${is_last ? 'Finish' : 'Next trial'}
        </button>
      </div>`;

    if (show_performance) {
      display_el.querySelector('#summary-svg').innerHTML =
        buildPerformanceSVG(true_mean, true_std, values, responses);
    }

    display_el.querySelector('#next-btn').addEventListener('pointerdown', (e) => {
      e.preventDefault();
      this.jsPsych.finishTrial({ screen: 'inter_trial' });
    });
  }
}

TrialSummaryContinuousPlugin.info = info;
export default TrialSummaryContinuousPlugin;
