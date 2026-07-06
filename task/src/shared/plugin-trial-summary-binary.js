/**
 * plugin-trial-summary-binary.js
 * Inter-trial summary — binary task.
 * Identical layout to plugin-tutorial-summary-binary.js; no title, no text box.
 */
import { buildSummaryBarSVG } from './draw-performance-binary.js';

const info = {
  name: 'trial-summary-binary',
  parameters: {
    true_p:           { type: 'FLOAT',   default: 0.5   },
    values:           { type: 'OBJECT',  default: []    },
    responses:        { type: 'OBJECT',  default: []    },
    show_performance: { type: 'BOOLEAN', default: false },
    is_last:          { type: 'BOOLEAN', default: false },
  },
};

class TrialSummaryBinaryPlugin {
  constructor(jsPsych) { this.jsPsych = jsPsych; }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    const { true_p, values, responses,
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
          style="font-size:1.6rem;padding:1rem 3.5rem;margin-top:1rem;">
          ${is_last ? 'Finish' : 'Next trial'}
        </button>
      </div>`;

    if (show_performance) {
      display_el.querySelector('#summary-svg').innerHTML =
        buildSummaryBarSVG(true_p, values, responses);
    }

    display_el.querySelector('#next-btn').addEventListener('pointerdown', (e) => {
      e.preventDefault();
      this.jsPsych.finishTrial({ screen: 'inter_trial' });
    });
  }
}

TrialSummaryBinaryPlugin.info = info;
export default TrialSummaryBinaryPlugin;
