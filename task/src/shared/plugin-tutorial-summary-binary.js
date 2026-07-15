/**
 * plugin-tutorial-summary-binary.js
 * End-of-tutorial summary for the binary task — bar chart + info boxes.
 */

import { buildSummaryBarSVG } from './draw-performance-binary.js';

const info = {
  name: 'tutorial-summary-binary',
  parameters: {
    true_p:    { type: 'FLOAT' },
    values:    { type: 'OBJECT', default: []  },
    responses: { type: 'OBJECT', default: []  },
  },
};

class TutorialSummaryBinaryPlugin {
  constructor(jsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    const { true_p, values, responses } = trial;

    display_el.innerHTML = `
      <div class="screen-wrap" style="text-align:center;width:70vw;">
        <div class="info-banner-blue">
          You will be paid based on your performance.
        </div>
        <div id="summary-svg"
          style="display:block;margin:0 auto 0.75rem;
                 width:100%;border:1px solid #e5e7eb;
                 border-radius:6px;background:#fff;padding:4px;">
        </div>
        <div style="margin-bottom:0.75rem;">
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
      this.jsPsych.finishTrial({ screen: 'tutorial_summary' });
    });
  }
}

TutorialSummaryBinaryPlugin.info = info;
export default TutorialSummaryBinaryPlugin;
