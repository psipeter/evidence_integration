import { drawPerformance } from './draw-performance.js';

const info = {
  name: 'trial-summary',
  parameters: {
    trial_num:        { type: 'INT',     default: 1     },
    true_mean:        { type: 'FLOAT',   default: 54    },
    true_std:         { type: 'FLOAT',   default: 10    },
    values:           { type: 'OBJECT',  default: []    },
    responses:        { type: 'OBJECT',  default: []    },
    show_performance: { type: 'BOOLEAN', default: false },
  },
};

class TrialSummaryPlugin {
  constructor(jsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';

    const { trial_num, true_mean, true_std,
            values, responses, show_performance } = trial;

    const perfHTML = show_performance ? `
      <canvas id="perf-canvas" width="520" height="150"
        style="display:block;margin:0 auto 0.75rem;border:1px solid #e5e7eb;
               border-radius:6px;background:#fff;padding:4px;">
      </canvas>
` : '';

    display_el.innerHTML = `
      <div class="screen-wrap" style="text-align:center;">
        <p style="font-size:1.2rem;margin-bottom:0.75rem;">Trial ${trial_num} complete.</p>
        ${perfHTML}
        <p style="color:#555;margin:0.75rem 0 1rem;">
          Click <em>Next trial</em> when you are ready to continue.
        </p>
        <button id="next-btn" class="jspsych-btn"
          style="font-size:1.1rem;padding:0.6rem 2.5rem;">
          Next trial
        </button>
      </div>`;

    if (show_performance) {
      const canvas = display_el.querySelector('#perf-canvas');
      drawPerformance(canvas, true_mean, true_std, values, responses);
    }

    display_el.querySelector('#next-btn').addEventListener('pointerdown', (e) => {
      e.preventDefault();
      this.jsPsych.finishTrial({ screen: 'inter_trial', trial_num });
    });
  }
}

TrialSummaryPlugin.info = info;
export default TrialSummaryPlugin;
