/**
 * plugin-practice-summary-binary.js
 * Tutorial summary for the binary task — bar chart + info boxes.
 */

import { buildSummaryBarSVG } from './bar-chart.js';

const DIST_COLOR = '#16a34a';

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
        <div class="summary-info-row">
          <p class="practice-info-block">
            In the experiment, the
            <span style="color:${DIST_COLOR};font-weight:bold;">true probability</span>
            will not be shown.
          </p>
          <p class="practice-info-block">
            In the experiment, the number of observations per trial is unknown.
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
