/**
 * plugin-tutorial-observation-binary.js
 * Obs 2-5 of the binary tutorial — same layout as obs 1 but fully revealed.
 *
 * No timeout clock, deliberately — participants need unhurried time to read
 * and think during the tutorial. Draw animation runs on load; submit stays
 * disabled until the animation completes.
 *
 * Box text is imported from tutorial-text-binary.js, shared with
 * plugin-tutorial-intro-binary.js -- never hardcode it here again (see that
 * module's own docstring for why).
 */

import { buildBinarySliderHTMLv2 as buildBinarySliderHTML, initBinarySliderV2 as initBinarySlider } from './slider-binary.js';
import { buildUrnSVG } from './urn-binary.js';
import { startBinaryDrawAnimation } from './binary-draw-animation.js';
import { BOX0, BOX1, BOX2, URN_CAPTION } from './tutorial-text-binary.js';

const info = {
  name: 'tutorial-observation-binary',
  parameters: {
    value:          { type: 'INT',     default: 1      },
    obs_num:        { type: 'INT',     default: 1      },
    n_obs:          { type: 'INT',     default: 5      },
    true_p:         { type: 'FLOAT' },
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
    const unset = slider_default === 'none';

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
            <div id="tut-ball" class="binary-circle" style="opacity:0;"></div>
          </div>
          <div class="tutorial-panel tutorial-panel-right">
            <div id="urn-svg" class="dist-canvas" style="line-height:0;"></div>
            <p id="urn-caption" class="tutorial-info-block"
               style="margin-top:0.5rem;background:#fffbeb;border:1px solid #fbbf24;">
              <span>${URN_CAPTION}</span>
            </p>
          </div>
        </div>
        ${buildBinarySliderHTML({ unset, initPos: resolvedInitPos, showValue: show_value })}
        <div style="text-align:center;margin-top:0.5rem;">
          <button id="submit-btn" class="jspsych-btn" disabled
            style="font-size:1.6rem;padding:1rem 3.5rem;min-width:200px;">
            Submit
          </button>
        </div>
      </div>`;

    display_el.querySelector('#urn-svg').innerHTML = buildUrnSVG(true_p, true);

    const submitBtn = display_el.querySelector('#submit-btn');
    const svgRoot   = display_el.querySelector('#urn-svg svg');
    const centerEl  = display_el.querySelector('#tut-ball');

    startBinaryDrawAnimation({
      svgRoot,
      centerEl,
      true_p,
      currentValue: value,
      obsNum:       obs_num,
      onComplete: () => {
        if (!unset) submitBtn.disabled = false;
      },
    });

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
