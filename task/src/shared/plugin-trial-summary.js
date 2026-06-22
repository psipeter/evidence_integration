/**
 * plugin-trial-summary.js
 * Inter-trial summary — continuous task.
 * Identical layout to plugin-practice-summary.js; no title, no text box.
 */
import { buildPerformanceSVG } from './draw-performance.js';

const info = {
  name: 'trial-summary',
  parameters: {
    true_mean:        { type: 'FLOAT',   default: 54    },
    true_std:         { type: 'FLOAT',   default: 10    },
    values:           { type: 'OBJECT',  default: []    },
    responses:        { type: 'OBJECT',  default: []    },
    show_performance: { type: 'BOOLEAN', default: false },
    is_last:          { type: 'BOOLEAN', default: false },
  },
};

class TrialSummaryPlugin {
  constructor(jsPsych) { this.jsPsych = jsPsych; }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    const { true_mean, true_std, values, responses,
            show_performance, is_last, trial_num } = trial;

    display_el.innerHTML = `
      <div class="screen-wrap" style="text-align:center;width:70vw;">
        ${show_performance ? `
        <div id="summary-svg"
          style="display:block;margin:0 auto 0.75rem;
                 width:100%;border:1px solid #e5e7eb;
                 border-radius:6px;background:#fff;padding:4px;">
        </div>` : ''}
        <button id="next-btn" class="jspsych-btn"
          style="font-size:1.1rem;padding:0.6rem 2.5rem;margin-top:1rem;">
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

TrialSummaryPlugin.info = info;
export default TrialSummaryPlugin;
