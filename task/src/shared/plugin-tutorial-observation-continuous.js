/**
 * plugin-tutorial-observation-continuous.js
 * Obs 2-5 of the continuous tutorial. Three-column layout:
 *   Left  — goal/observation text
 *   Centre — identical to main task (number + slider + submit)
 *   Right  — distribution plot
 *
 * No timeout clock, deliberately — participants need unhurried time to read
 * and think during the tutorial. Wires the slider synchronously right after
 * setting innerHTML (Pattern A: no on_load, no async, no rAF/setTimeout
 * deferral) — same shape as plugin-tutorial-observation-binary.js.
 *
 * The distribution SVG is built by the shared distribution-continuous.js
 * (revealed=true here — axis/dist/mean fully shown; the #tut-svg-obs marker
 * always starts hidden regardless, and the falling-bubble draw animation
 * (continuous-draw-animation.js) reveals it — same auto-running pattern as
 * plugin-tutorial-observation-binary.js's startBinaryDrawAnimation; submit
 * stays disabled until it completes.
 */

import { buildDistributionSVG } from './distribution-continuous.js';
import { buildSliderHTML, initSlider } from './slider-continuous.js';
import { startContinuousDrawAnimation } from './continuous-draw-animation.js';

const GOAL_COLOR   = '#2563eb';   // blue  — true mean / ???
const SAMPLE_COLOR = '#ef4444';   // red   — current observation
const DIST_COLOR   = '#16a34a';   // green — distribution curve

const info = {
  name: 'tutorial-observation-continuous',
  parameters: {
    value:          { type: 'INT',     default: 50 },
    obs_num:        { type: 'INT',     default: 1  },
    n_obs:          { type: 'INT',     default: 5  },
    true_mean:      { type: 'FLOAT',   default: 54 },
    true_std:       { type: 'FLOAT',   default: 10 },
    slider_default: { type: 'STRING',  default: 'none' },
    init_pos:       { type: 'INT',     default: 54 },
    show_value:     { type: 'BOOLEAN', default: true  },
  },
};

class TutorialObservationContinuousPlugin {
  constructor(jsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    if (document.activeElement && document.activeElement !== document.body) {
      document.activeElement.blur();
    }

    const { value, obs_num, n_obs, true_mean, true_std,
            slider_default, init_pos, show_value } = trial;
    const resolvedInitPos = typeof init_pos === 'function' ? init_pos() : (init_pos ?? 0);
    const unset = slider_default === 'none';

    const DIST_CAPTION = `This curve shows the true
      <span style="color:${DIST_COLOR};font-weight:bold;">distribution</span>. In the
      experiment, you will only see the individual
      <span style="color:${SAMPLE_COLOR};font-weight:bold;">numbers</span>.`;
    const BOX0 = `In this task, you'll see a sequence of
      <span style="color:${SAMPLE_COLOR};font-weight:bold;">numbers</span>. Each number is
      randomly drawn from a hidden
      <span style="color:${DIST_COLOR};font-weight:bold;">distribution</span>.`;
    const BOX1 = `Your <strong>goal</strong> is to estimate that distribution's
      <span style="color:${GOAL_COLOR};font-weight:bold;">mean</span>, based on
      all the numbers you've seen so far.`;
    const BOX2 = `<strong>Move</strong> the slider to show your estimate of the
      <span style="color:${GOAL_COLOR};font-weight:bold;">mean</span>.`;

    display_el.innerHTML = `
      <div class="tutorial-title">Tutorial</div>

      <div class="tutorial-wrap">

        <!-- TOP ROW: three equal panels -->
        <div class="tutorial-top-row">

          <!-- LEFT: goal text -->
          <div class="tutorial-panel">
            <p class="tutorial-info-block"><span>${BOX0}</span></p>
            <p class="tutorial-info-block"><span>${BOX1}</span></p>
            <p class="tutorial-info-block"><span>${BOX2}</span></p>
          </div>

          <!-- CENTRE: big number -->
          <div class="tutorial-panel tutorial-panel-centre">
            <div id="stimulus-display" class="stimulus-number"
              style="color:${SAMPLE_COLOR};opacity:0;">${value}</div>
          </div>

          <!-- RIGHT: distribution -->
          <div class="tutorial-panel tutorial-panel-right">
            <div id="dist-svg" class="dist-canvas" style="line-height:0;"></div>
            <p id="dist-caption" class="tutorial-info-block"
               style="margin-top:0.5rem;background:#fffbeb;border:1px solid #fbbf24;">
              <span>${DIST_CAPTION}</span>
            </p>
          </div>

        </div>

        <!-- BOTTOM: full-width slider + submit -->
        ${buildSliderHTML({ unset, initPos: resolvedInitPos, showValue: show_value })}
        <button id="submit-btn" class="jspsych-btn" disabled
          style="font-size:1.6rem;padding:1rem 3.5rem;min-width:200px;">
          Submit
        </button>

      </div>`;

    // Draw distribution — axis/dist/mean fully revealed (obs 2-5); the obs
    // marker always starts hidden regardless (see distribution-continuous.js)
    // and is revealed by the draw animation below.
    display_el.querySelector('#dist-svg').innerHTML =
      buildDistributionSVG(true_mean, true_std, value, true);

    const submitBtn = display_el.querySelector('#submit-btn');
    const svgRoot   = display_el.querySelector('#dist-svg svg');
    const centerEl  = display_el.querySelector('#stimulus-display');

    startContinuousDrawAnimation({
      svgRoot,
      centerEl,
      true_mean,
      true_std,
      obsNum: obs_num,
      onComplete: () => {
        if (!unset) submitBtn.disabled = false;
      },
    });

    // Wire immediately — DOM is synchronously ready right after innerHTML is
    // set. No timeout clock, so no deadline for jsPsych to race against;
    // no rAF/setTimeout deferral needed either (that pattern previously
    // caused an unclickable-button bug in the main observation plugins).
    initSlider(display_el, {
      unset,
      showValue: show_value,
      onFinish: () => {
        const response = parseInt(display_el.querySelector('#response-slider').value);
        this.jsPsych.finishTrial({ response, timed_out: false });
      },
    });
  }
}

TutorialObservationContinuousPlugin.info = info;
export default TutorialObservationContinuousPlugin;
