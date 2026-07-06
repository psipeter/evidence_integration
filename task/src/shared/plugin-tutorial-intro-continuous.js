/**
 * plugin-tutorial-intro-continuous.js
 *
 * Each box contains real content (invisible, for sizing) + absolute overlay.
 * SVG height matches left panel height via ResizeObserver.
 * Axis labels hidden until box 0 clicked; slider hidden until box 2 clicked.
 *
 * The distribution SVG itself is built by the shared distribution-continuous.js
 * (revealed=false here — group opacities start hidden, toggled to '1' below as
 * boxes are clicked), mirroring plugin-tutorial-observation-continuous.js which
 * uses the same builder with revealed=true. These used to be two separate,
 * drifting local implementations of nearly the same SVG.
 */

import { buildDistributionSVG } from './distribution-continuous.js';
import { buildSliderHTML, initSlider } from './slider-continuous.js';

const GOAL_COLOR   = '#2563eb';
const SAMPLE_COLOR = '#ef4444';
const DIST_COLOR   = '#16a34a';

const info = {
  name: 'tutorial-intro-continuous',
  parameters: {
    example_value: { type: 'INT',   default: 23 },
    true_mean:     { type: 'FLOAT', default: 20 },
    true_std:      { type: 'FLOAT', default: 20 },
  },
};

const makeBox = (id, realHTML, isActive) => `
  <p id="${id}" class="tutorial-info-block"
     style="${isActive ? 'cursor:pointer;' : ''}">
    <span id="${id}-placeholder"
          style="color:${isActive ? '#555' : '#ccc'};font-weight:bold;">
      ${isActive ? 'Click to reveal' : '· · ·'}
    </span>
    <span id="${id}-real" style="display:none;">${realHTML}</span>
  </p>`;

class TutorialIntroContinuousPlugin {
  constructor(jsPsych) { this.jsPsych = jsPsych; }

  trial(display_el, trial) {
    const self = this;
    document.body.style.backgroundColor = '#f5f5f5';
    const { example_value, true_mean, true_std } = trial;

    const BOX0 = `<span style="color:${SAMPLE_COLOR};font-weight:bold;">Numbers</span> are drawn one-at-a-time from a
      <span style="color:${DIST_COLOR};font-weight:bold;">hidden distribution</span>.`;
    const BOX1 = `Goal: estimate the
      <span style="color:${GOAL_COLOR};font-weight:bold;">true mean</span>
      of that distribution.`;
    const BOX2 = `After each <span style="color:${SAMPLE_COLOR};font-weight:bold;">observation</span>, move the slider to update your estimate.`;

    display_el.innerHTML = `
      <div class="tutorial-title">Tutorial</div>
      <div class="tutorial-wrap">
        <div class="tutorial-top-row">

          <div id="tut-left-panel" class="tutorial-panel">
            ${makeBox('tut-box-0', BOX0, true)}
            ${makeBox('tut-box-1', BOX1, false)}
            ${makeBox('tut-box-2', BOX2, false)}
          </div>

          <div class="tutorial-panel tutorial-panel-centre">
            <div id="tut-centre-number" class="stimulus-number"
                 style="color:${SAMPLE_COLOR};opacity:0;">${example_value}</div>
          </div>

          <div class="tutorial-panel tutorial-panel-right">
            <div id="dist-svg" class="dist-canvas" style="line-height:0;"></div>
          </div>

        </div>

        <div id="tut-slider-wrap" style="visibility:hidden;">
          ${buildSliderHTML({ unset: true, initPos: 0, showValue: true })}
          <button id="submit-btn" class="jspsych-btn" disabled
                  style="font-size:1.6rem;padding:1rem 3.5rem;min-width:200px;">
            Submit
          </button>
        </div>

      </div>`;

    // ── Draw SVG — scales to fill right panel via viewBox + width/height 100% ──
    display_el.querySelector('#dist-svg').innerHTML =
      buildDistributionSVG(true_mean, true_std, example_value, false);

    const sliderWrap = display_el.querySelector('#tut-slider-wrap');

    const jsPsych = self.jsPsych;
    const activateSlider = () => {
      sliderWrap.style.visibility = 'visible';
      initSlider(display_el, {
        unset: true,
        showValue: true,
        ghostPos: null,
        onFinish: () => {
          const slider   = display_el.querySelector('#response-slider');
          const response = parseInt(slider.value);
          jsPsych.finishTrial({ response, timed_out: false });
        },
      });
    };

    const revealBox = (id) => {
      display_el.querySelector(`#${id}-placeholder`).style.display = 'none';
      display_el.querySelector(`#${id}-real`).style.display = 'inline';
      display_el.querySelector(`#${id}`).style.cursor = 'default';
    };

    const activateBox = (id, onClickFn) => {
      const box = display_el.querySelector(`#${id}`);
      const ph  = display_el.querySelector(`#${id}-placeholder`);
      box.style.cursor = 'pointer';
      if (ph) { ph.style.display = 'inline'; ph.style.color = '#555'; ph.textContent = 'Click to reveal'; }
      box.addEventListener('click', onClickFn, { once: true });
    };

    const onBox0 = () => {
      revealBox('tut-box-0');
      // Axis labels + dist curve appear together
      const svg = display_el.querySelector('#dist-svg');
      svg.querySelector('#tut-svg-axis-labels').style.opacity = '1';
      svg.querySelector('#tut-svg-dist').style.opacity = '1';
      display_el.querySelector('#tut-centre-number').style.opacity = '1';
      activateBox('tut-box-1', onBox1);
    };
    const onBox1 = () => {
      revealBox('tut-box-1');
      display_el.querySelector('#dist-svg').querySelector('#tut-svg-mean').style.opacity = '1';
      activateBox('tut-box-2', onBox2);
    };
    const onBox2 = () => {
      revealBox('tut-box-2');
      const svg = display_el.querySelector('#dist-svg');
      svg.querySelector('#tut-svg-obs').style.opacity = '1';
      activateSlider();
    };

    activateBox('tut-box-0', onBox0);
  }
}

TutorialIntroContinuousPlugin.info = info;
export default TutorialIntroContinuousPlugin;
