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
      <div class="screen-wrap" style="text-align:center;width:70vw;">
        <div id="summary-svg"
          style="display:block;margin:0 auto 0.75rem;
                 width:100%;border:1px solid #e5e7eb;
                 border-radius:6px;background:#fff;padding:4px;">
        </div>
        <div style="margin-bottom:0.75rem;">
          <p class="practice-info-block" style="text-align:center;margin:0 auto;">
            In the experiment, you will not see the&nbsp;<span style="color:#16a34a;font-weight:bold;">true probability</span>&nbsp;or how many&nbsp;<span style="color:#888;font-weight:bold;">observations</span>&nbsp;remain.
          </p>
        </div>
        <button id="proceed-btn" class="jspsych-btn"
          style="font-size:1.6rem;padding:1rem 3.5rem;margin-top:1rem;">
          Next
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
