import { buildPerformanceSVG } from './draw-performance.js';

const info = {
  name: 'trial-summary',
  parameters: {
    trial_num:        { type: 'INT',     default: 1     },
    true_mean:        { type: 'FLOAT',   default: 54    },
    true_std:         { type: 'FLOAT',   default: 10    },
    values:           { type: 'OBJECT',  default: []    },
    responses:        { type: 'OBJECT',  default: []    },
    show_performance: { type: 'BOOLEAN', default: false },
    is_last:          { type: 'BOOLEAN', default: false },
  },
};

class TrialSummaryPlugin {
  constructor(jsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';

    const { trial_num, true_mean, true_std,
            values, responses, show_performance, is_last } = trial;

    const perfHTML = show_performance ? `
      <div id="perf-svg"
        style="display:block;margin:0 auto 0.75rem;width:fit-content;
               border:1px solid #e5e7eb;border-radius:6px;
               background:#fff;padding:4px;">
      </div>
` : '';

    display_el.innerHTML = `
      <div class="screen-wrap" style="text-align:center;">
        <p style="font-size:1.2rem;margin-bottom:0.75rem;">Trial ${trial_num} complete.</p>
        ${perfHTML}
        <button id="next-btn" class="jspsych-btn"
          style="font-size:1.1rem;padding:0.6rem 2.5rem;">
          ${is_last ? "Finish" : "Next trial"}
        </button>
      </div>`;

    if (show_performance) {
      display_el.querySelector('#perf-svg').innerHTML =
        buildPerformanceSVG(true_mean, true_std, values, responses);
    }

    display_el.querySelector('#next-btn').addEventListener('pointerdown', (e) => {
      e.preventDefault();
      this.jsPsych.finishTrial({ screen: 'inter_trial', trial_num });
    });
  }
}

TrialSummaryPlugin.info = info;
export default TrialSummaryPlugin;
