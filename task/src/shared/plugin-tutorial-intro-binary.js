import { buildBinarySliderHTMLv2 as buildBinarySliderHTML, initBinarySliderV2 as initBinarySlider } from './slider-binary.js';
import { buildUrnSVG } from './urn-binary.js';
import { startBinaryDrawAnimation } from './binary-draw-animation.js';
/**
 * plugin-tutorial-intro-binary.js
 * Obs 1 of the binary tutorial — progressive reveal via click.
 * Box 0 (text) → image box (click to reveal bar + draw animation) →
 * Box 1 (goal text) → Box 2 (slider instructions) → slider.
 * The image reveal is its own step, separate from box 0's text, so
 * participants aren't reading and watching the animation at the same time.
 */

const SAMPLE_BLUE = '#2563eb';
const SAMPLE_RED  = '#ef4444';
const DIST_COLOR  = '#16a34a';

const info = {
  name: 'tutorial-intro-binary',
  parameters: {
    example_value: { type: 'INT',   default: 1   },
    true_p:        { type: 'FLOAT' },
    n_obs:         { type: 'INT',   default: 5   },
  },
};

const makeBox = (id, realHTML, isActive) => `
  <div id="${id}" class="tutorial-info-block" style="position:relative;${isActive ? 'cursor:pointer;' : ''}">
    <span id="${id}-placeholder"
          style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
                 font-weight:bold;color:${isActive ? '#555' : '#ccc'};white-space:nowrap;">
      ${isActive ? 'Click to reveal' : '· · ·'}
    </span>
    <span id="${id}-real" style="visibility:hidden;">${realHTML}</span>
  </div>`;

class TutorialIntroBinaryPlugin {
  constructor(jsPsych) { this.jsPsych = jsPsych; }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    const { example_value, true_p } = trial;

    const BOX0 = `In this task, you'll see a sequence of balls. Each ball is
      randomly colored
      <span style="color:${SAMPLE_RED};font-weight:bold;">red</span> or
      <span style="color:${SAMPLE_BLUE};font-weight:bold;">blue</span> according to a
      hidden <span style="color:${DIST_COLOR};font-weight:bold;">probability</span>.`;
    const BOX1 = `Your <strong>goal</strong> is to estimate that
      <span style="color:${DIST_COLOR};font-weight:bold;">probability</span>, based on
      all the balls you've seen in this sequence.`;
    const BOX2 = `<strong>Move</strong> the slider toward
      <span style="color:${SAMPLE_BLUE};font-weight:bold;">blue</span> or
      <span style="color:${SAMPLE_RED};font-weight:bold;">red</span> to show
      which you think is more likely, and by how much.`;
    const URN_CAPTION = `This bar shows the true
      <span style="color:${DIST_COLOR};font-weight:bold;">probability</span> of
      <span style="color:${SAMPLE_BLUE};font-weight:bold;">blue</span> vs
      <span style="color:${SAMPLE_RED};font-weight:bold;">red</span>. In the
      experiment, you will only see the colored balls.`;

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
            <div id="tut-ball" class="binary-circle" style="opacity:0;"></div>
          </div>
          <div class="tutorial-panel tutorial-panel-right">
            <div id="tut-image-box" style="position:relative;flex:1;">
              <div id="urn-svg" class="dist-canvas" style="line-height:0;"></div>
              <div id="tut-image-placeholder"
                   style="position:absolute;inset:0;display:flex;align-items:center;
                          justify-content:center;background:#fff;border-radius:6px;
                          cursor:default;color:#ccc;font-weight:bold;">
                · · ·
              </div>
            </div>
            <p id="urn-caption" class="tutorial-info-block"
               style="margin-top:0.5rem;opacity:0;background:#fffbeb;border:1px solid #fbbf24;">
              <span>${URN_CAPTION}</span>
            </p>
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

    display_el.querySelector('#urn-svg').innerHTML = buildUrnSVG(true_p, false);

    const svgRoot   = () => display_el.querySelector('#urn-svg svg');
    const centerEl  = () => display_el.querySelector('#tut-ball');
    const jsPsych = this.jsPsych;

    const activateSlider = () => {
      display_el.querySelector('#tut-slider-wrap').style.visibility = 'visible';
      initBinarySlider(display_el, {
        unset: true, showValue: true,
        onFinish: () => {
          const response = parseInt(display_el.querySelector('#response-slider').value);
          jsPsych.finishTrial({ response, timed_out: false });
        },
      });
    };

    const revealBox = (id) => {
      display_el.querySelector(`#${id}-placeholder`).style.display = 'none';
      display_el.querySelector(`#${id}-real`).style.visibility = 'visible';
      display_el.querySelector(`#${id}`).style.cursor = 'default';
    };

    const activateBox = (id, fn) => {
      const box = display_el.querySelector(`#${id}`);
      const ph  = display_el.querySelector(`#${id}-placeholder`);
      box.style.cursor = 'pointer';
      if (ph) { ph.style.color = '#555'; ph.textContent = 'Click to reveal'; }
      box.addEventListener('click', fn, { once: true });
    };

    const showBar = () => {
      const bar = svgRoot()?.querySelector('#tut-urn-bar');
      const bub = svgRoot()?.querySelector('#tut-urn-bubbles');
      if (bar) bar.style.opacity = '1';
      if (bub) bub.style.opacity = '1';
    };

    // Image box is its own click-to-reveal step, AFTER box 0's text and
    // BEFORE box 1's goal text — previously the image appeared automatically
    // alongside box 0, which split attention between reading and watching the
    // animation at the same moment.
    const activateImageBox = () => {
      const ph = display_el.querySelector('#tut-image-placeholder');
      if (!ph) return;
      ph.style.cursor = 'pointer';
      ph.style.color  = '#555';
      ph.textContent  = 'Click to reveal';
      ph.addEventListener('click', onImageBox, { once: true });
    };

    const onImageBox = () => {
      display_el.querySelector('#tut-image-placeholder')?.remove();
      showBar();
      startBinaryDrawAnimation({
        svgRoot:       svgRoot(),
        centerEl:      centerEl(),
        true_p,
        currentValue:  example_value,
        obsNum:        1,
      });
      activateBox('tut-box-1', onBox1);
    };

    const onBox0 = () => {
      revealBox('tut-box-0');
      activateImageBox();
    };
    const onBox1 = () => {
      revealBox('tut-box-1');
      const qmark = svgRoot()?.querySelector('#tut-urn-qmark');
      if (qmark) qmark.style.opacity = '1';
      const caption = display_el.querySelector('#urn-caption');
      if (caption) caption.style.opacity = '1';
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
