/**
 * plugin-tutorial-observation-binary.js
 * Obs 2-5 of the binary tutorial — same layout as obs 1 but fully revealed.
 *
 * No timeout clock, deliberately — participants need unhurried time to read
 * and think during the tutorial. Wires the slider synchronously right after
 * setting innerHTML (Pattern A: no on_load, no async, no rAF/setTimeout
 * deferral).
 */

import { buildBinarySliderHTML, initBinarySlider } from './slider-binary.js';
import { buildUrnSVG } from './urn-binary.js';

const SAMPLE_BLUE = '#2563eb';
const SAMPLE_RED  = '#ef4444';
const DIST_COLOR  = '#16a34a';

const info = {
  name: 'tutorial-observation-binary',
  parameters: {
    value:          { type: 'INT',     default: 1      },
    obs_num:        { type: 'INT',     default: 1      },
    n_obs:          { type: 'INT',     default: 5      },
    true_p:         { type: 'FLOAT',   default: 0.7    },
    slider_default: { type: 'STRING',  default: 'none' },
    init_pos:       { type: 'INT',     default: 50     },
    show_value:     { type: 'BOOLEAN', default: true   },
  },
};

class TutorialObservationBinaryPlugin {
  constructor(jsPsych) { this.jsPsych = jsPsych; }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    if (document.activeElement && document.activeElement !== document.body)
      document.activeElement.blur();

    const { value, obs_num, true_p,
            slider_default, init_pos, show_value } = trial;
    const resolvedInitPos = typeof init_pos === 'function' ? init_pos() : (init_pos ?? 50);
    const unset    = slider_default === 'none';
    const ballCol  = value === 1 ? SAMPLE_BLUE : SAMPLE_RED;

    const BOX0 = `<span style="color:${SAMPLE_BLUE};font-weight:bold;">Blue</span> or
      <span style="color:${SAMPLE_RED};font-weight:bold;">red</span> balls
      are drawn one-at-a-time based on a
      <span style="color:${DIST_COLOR};font-weight:bold;">hidden probability</span>.`;
    const BOX1 = `Goal: estimate the <strong>proportion</strong> of
      <span style="color:${SAMPLE_BLUE};font-weight:bold;">blue</span> and
      <span style="color:${SAMPLE_RED};font-weight:bold;">red</span> balls.`;
    const BOX2 = `After each <span style="color:#888;font-weight:bold;">observation</span>,
      move the slider to update your estimate.`;

    display_el.innerHTML = `
      <div class="tutorial-title">Tutorial</div>
      <div class="tutorial-wrap">
        <div class="tutorial-top-row">
          <div class="tutorial-panel">
            <p class="tutorial-info-block"><span>${BOX0}</span></p>
            <p class="tutorial-info-block"><span>${BOX1}</span></p>
            <p class="tutorial-info-block"><span>${BOX2}</span></p>
          </div>
          <div class="tutorial-panel tutorial-panel-centre">
            <div class="binary-circle" style="background:${ballCol};"></div>
          </div>
          <div class="tutorial-panel tutorial-panel-right">
            <div id="urn-svg" class="dist-canvas" style="line-height:0;"></div>
          </div>
        </div>
        ${buildBinarySliderHTML({ unset, initPos: resolvedInitPos, showValue: show_value })}
        <div style="text-align:center;margin-top:0.5rem;">
          <button id="submit-btn" class="jspsych-btn"
            ${unset ? 'disabled' : ''}
            style="font-size:1.6rem;padding:1rem 3.5rem;min-width:200px;">
            Submit
          </button>
        </div>
      </div>`;

    // revealed=true: all groups visible immediately
    display_el.querySelector('#urn-svg').innerHTML =
      buildUrnSVG(true_p, value, obs_num, true);

    initBinarySlider(display_el, {
      unset, showValue: show_value,
      onFinish: () => {
        const response = parseInt(display_el.querySelector('#response-slider').value);
        this.jsPsych.finishTrial({ response, timed_out: false });
      },
    });
  }
}

TutorialObservationBinaryPlugin.info = info;
export default TutorialObservationBinaryPlugin;
