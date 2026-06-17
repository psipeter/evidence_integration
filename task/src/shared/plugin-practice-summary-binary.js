/**
 * plugin-practice-summary-binary.js
 * Tutorial summary for the binary task — bar chart + info boxes.
 */

import { buildSummaryBarSVG } from './bar-chart.js';

const info = {
  name: 'practice-summary-binary',
  parameters: {
    true_p:    { type: 'FLOAT',  default: 0.6 },
    values:    { type: 'OBJECT', default: []  },
    responses: { type: 'OBJECT', default: []  },
  },
};

class PracticeSummaryBinaryPlugin {
  constructor(jsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    const { true_p, values, responses } = trial;

    display_el.innerHTML = `
      <div class="screen-wrap" style="text-align:center;">
        <h2 style="margin-bottom:0.75rem;">Tutorial complete</h2>
        <div id="summary-svg"
          style="display:inline-block;margin-bottom:0.75rem;
                 border:1px solid #e5e7eb;border-radius:6px;
                 background:#fff;padding:4px;">
        </div>
        <div style="margin-bottom:0.75rem;">
          <p class="practice-info-block" style="text-align:center;max-width:clamp(280px,50vw,600px);margin:0 auto;">
            In the experiment, you will not see the
            <span style="color:#16a34a;font-weight:bold;">distribution</span>,
            the <span style="color:#2563eb;font-weight:bold;">probability</span>,
            or how many
            <span style="color:#888;font-weight:bold;">observations</span>
            remain.
          </p>
        </div>
        <button id="proceed-btn" class="jspsych-btn"
          style="font-size:1.1rem;padding:0.6rem 2.5rem;margin-top:1rem;">
          Proceed to experiment
        </button>
      </div>`;

    display_el.querySelector('#summary-svg').innerHTML =
      buildSummaryBarSVG(true_p, values, responses);

    display_el.querySelector('#proceed-btn').addEventListener('pointerdown', (e) => {
      e.preventDefault();
      this.jsPsych.finishTrial({ screen: 'practice_summary' });
    });
  }
}

PracticeSummaryBinaryPlugin.info = info;
export default PracticeSummaryBinaryPlugin;
