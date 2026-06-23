import { buildBinarySliderHTML, initBinarySlider } from './slider-binary.js';
import { buildUrnSVG } from './urn-binary.js';
/**
 * plugin-tutorial-intro-binary.js
 * Obs 1 of the binary tutorial — progressive reveal via click.
 * Box 0 → dots visible; Box 1 → bar+??? visible; Box 2 → highlight + slider.
 */

const SAMPLE_BLUE = '#2563eb';
const SAMPLE_RED  = '#ef4444';
const DIST_COLOR  = '#16a34a';

const info = {
  name: 'tutorial-intro-binary',
  parameters: {
    example_value: { type: 'INT',   default: 1   },
    true_p:        { type: 'FLOAT', default: 0.7 },
    n_obs:         { type: 'INT',   default: 5   },
  },
};

const makeBox = (id, realHTML, isActive) => `
  <p id="${id}" class="practice-info-block"
     style="${isActive ? 'cursor:pointer;' : ''}">
    <span id="${id}-placeholder"
          style="color:${isActive ? '#555' : '#ccc'};font-weight:bold;">
      ${isActive ? 'Click to reveal' : '· · ·'}
    </span>
    <span id="${id}-real" style="display:none;">${realHTML}</span>
  </p>`;

class TutorialIntroBinaryPlugin {
  constructor(jsPsych) { this.jsPsych = jsPsych; }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    const { example_value, true_p } = trial;
    const ballCol = example_value === 1 ? SAMPLE_BLUE : SAMPLE_RED;

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
      <div class="practice-wrap">
        <div class="practice-top-row">
          <div id="tut-left-panel" class="practice-panel">
            ${makeBox('tut-box-0', BOX0, true)}
            ${makeBox('tut-box-1', BOX1, false)}
            ${makeBox('tut-box-2', BOX2, false)}
          </div>
          <div class="practice-panel practice-panel-centre">
            <div id="tut-ball" class="binary-circle"
                 style="background:${ballCol};opacity:0;"></div>
          </div>
          <div class="practice-panel practice-panel-right">
            <div id="urn-svg" class="dist-canvas" style="line-height:0;"></div>
          </div>
        </div>
        <div id="tut-slider-wrap" style="visibility:hidden;">
          ${buildBinarySliderHTML({ unset: true, initPos: 50, showValue: true })}
          <div style="text-align:center;margin-top:0.5rem;">
            <button id="submit-btn" class="jspsych-btn" disabled
                    style="font-size:1.6rem;padding:1rem 3.5rem;min-width:200px;">
              Submit
            </button>
          </div>
        </div>
      </div>`;

    // revealed=false: all groups hidden, revealed progressively
    display_el.querySelector('#urn-svg').innerHTML =
      buildUrnSVG(true_p, example_value, 1, false);

    const svg      = () => display_el.querySelector('#urn-svg');
    const jsPsych  = this.jsPsych;

    const activateSlider = () => {
      display_el.querySelector('#tut-slider-wrap').style.visibility = 'visible';
      initBinarySlider(display_el, {
        unset: true, showValue: true,
        onFinish: (response) => jsPsych.finishTrial({ response, timed_out: false }),
      });
    };

    const revealBox = (id) => {
      display_el.querySelector(`#${id}-placeholder`).style.display = 'none';
      display_el.querySelector(`#${id}-real`).style.display = 'inline';
      display_el.querySelector(`#${id}`).style.cursor = 'default';
    };

    const activateBox = (id, fn) => {
      const box = display_el.querySelector(`#${id}`);
      const ph  = display_el.querySelector(`#${id}-placeholder`);
      box.style.cursor = 'pointer';
      if (ph) { ph.style.display = 'inline'; ph.style.color = '#555'; ph.textContent = 'Click to reveal'; }
      box.addEventListener('click', fn, { once: true });
    };

    const onBox0 = () => {
      revealBox('tut-box-0');
      svg().querySelector('#tut-urn-dots').style.opacity = '1';
      svg().querySelector('#tut-urn-highlight').style.opacity = '1';
      display_el.querySelector('#tut-ball').style.opacity = '1';
      activateBox('tut-box-1', onBox1);
    };
    const onBox1 = () => {
      revealBox('tut-box-1');
      svg().querySelector('#tut-urn-label').style.opacity = '1';
      activateBox('tut-box-2', onBox2);
    };
    const onBox2 = () => {
      revealBox('tut-box-2');
      activateSlider();
    };

    activateBox('tut-box-0', onBox0);
  }
}

TutorialIntroBinaryPlugin.info = info;
export default TutorialIntroBinaryPlugin;
