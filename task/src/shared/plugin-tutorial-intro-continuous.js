import { buildDistributionSVG } from './distribution-continuous.js';
import { buildSliderHTML, initSlider } from './slider-continuous.js';
import { startContinuousDrawAnimation, FADE_MS as DRAW_FADE_MS } from './continuous-draw-animation.js';
import { buildTrackerHTML } from './tutorial-tracker.js';
import { BOX0, BOX0B, BOX1, BOX2, SAMPLE_COLOR } from './tutorial-text-continuous.js';
/**
 * plugin-tutorial-intro-continuous.js
 * Obs 1 of the continuous tutorial — progressive reveal via click.
 * Box 0 (text, left panel) → image box (click to reveal distribution +
 * falling-bubble draw animation + Box 0b's text, which sits above the
 * distribution figure in the right panel and is revealed by THIS click, not
 * its own -- and the sequence-history tracker below the figure) → Box 1
 * (goal text) → Box 2 (slider instructions) → slider.
 * The image reveal is its own step, separate from box 0's text, so
 * participants aren't reading and watching the animation at the same time —
 * mirrors plugin-tutorial-intro-binary.js's structure exactly.
 * Box text is imported from tutorial-text-continuous.js, shared with
 * plugin-tutorial-observation-continuous.js -- never hardcode it here again
 * (see that module's own docstring for why).
 */

const info = {
  name: 'tutorial-intro-continuous',
  parameters: {
    example_value: { type: 'INT',   default: 23 },
    true_mean:     { type: 'FLOAT' },
    true_std:      { type: 'FLOAT' },
    n_obs:         { type: 'INT',    default: 5 },
    // Tracker -- see tutorial-tracker.js. Obs 1 has no history yet, so this
    // is just [example_value].
    values_so_far: { type: 'OBJECT', default: [] },
  },
};

const makeBox = (id, realHTML, isActive, extraStyle = '') => `
  <div id="${id}" class="tutorial-info-block" style="position:relative;${extraStyle}${isActive ? 'cursor:pointer;' : ''}">
    <span id="${id}-placeholder"
          style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
                 font-weight:bold;color:${isActive ? '#555' : '#ccc'};white-space:nowrap;">
      ${isActive ? 'Click to reveal' : '· · ·'}
    </span>
    <span id="${id}-real" style="visibility:hidden;">${realHTML}</span>
  </div>`;

class TutorialIntroContinuousPlugin {
  constructor(jsPsych) { this.jsPsych = jsPsych; }

  trial(display_el, trial) {
    const self = this;
    document.body.style.backgroundColor = '#f5f5f5';
    const { example_value, true_mean, true_std, n_obs, values_so_far } = trial;

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
            ${makeBox('tut-box-0b', BOX0B, false)}
            <div id="tut-image-box" style="position:relative;flex:1;">
              <div id="dist-svg" class="dist-canvas" style="line-height:0;"></div>
              <div id="tut-image-placeholder"
                   style="position:absolute;inset:0;display:flex;align-items:center;
                          justify-content:center;background:#fff;border-radius:6px;
                          cursor:default;color:#ccc;font-weight:bold;">
                · · ·
              </div>
            </div>
            <div id="tut-tracker" style="opacity:0;"></div>
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
    // Obs 1/intro has no PAST observations yet (history=[], the default) --
    // #tut-svg-history will just render empty, which is correct here.
    display_el.querySelector('#dist-svg').innerHTML =
      buildDistributionSVG(true_mean, true_std, example_value, false);

    // Tracker -- rendered now but kept at opacity:0 (see markup above) until
    // onImageBox reveals it alongside the distribution itself; showing an
    // all-empty-except-slot-1 tracker before the participant has even seen
    // the distribution would be showing progress through something they
    // don't know exists yet.
    display_el.querySelector('#tut-tracker').innerHTML = buildTrackerHTML({
      nObs: n_obs, obsNum: 1, values: values_so_far, color: SAMPLE_COLOR,
      revealCurrent: false,
    });

    const svgRoot    = () => display_el.querySelector('#dist-svg svg');
    const centerEl   = () => display_el.querySelector('#tut-centre-number');
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
      display_el.querySelector(`#${id}-real`).style.visibility = 'visible';
      display_el.querySelector(`#${id}`).style.cursor = 'default';
    };

    const activateBox = (id, onClickFn) => {
      const box = display_el.querySelector(`#${id}`);
      const ph  = display_el.querySelector(`#${id}-placeholder`);
      box.style.cursor = 'pointer';
      if (ph) { ph.style.color = '#555'; ph.textContent = 'Click to reveal'; }
      box.addEventListener('click', onClickFn, { once: true });
    };

    const showDist = () => {
      const svg = svgRoot();
      const axis = svg?.querySelector('#tut-svg-axis-labels');
      const dist = svg?.querySelector('#tut-svg-dist');
      if (axis) axis.style.opacity = '1';
      if (dist) dist.style.opacity = '1';
    };

    // Image box is its own click-to-reveal step, AFTER box 0's text and
    // BEFORE box 1's goal text — participants aren't reading and watching
    // the animation at the same moment.
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
      showDist();
      // Box 0b (second sentence, positioned above the distribution figure)
      // and the tracker (below it) are both revealed by THIS click, at the
      // same moment the distribution image itself appears -- not by box 0's
      // click. Neither has its own `activateBox()` call anywhere in this
      // file.
      revealBox('tut-box-0b');
      const tracker = display_el.querySelector('#tut-tracker');
      if (tracker) tracker.style.opacity = '1';
      startContinuousDrawAnimation({
        svgRoot:    svgRoot(),
        centerEl:   centerEl(),
        true_mean,
        true_std,
        obsNum:     1,
        onReveal: () => {
          const trackerNum = display_el.querySelector('#tut-tracker-current-num');
          if (trackerNum) {
            trackerNum.style.transition = `opacity ${DRAW_FADE_MS}ms ease`;
            trackerNum.style.opacity = '1';
          }
        },
      });
      activateBox('tut-box-1', onBox1);
    };

    const onBox0 = () => {
      revealBox('tut-box-0');
      activateImageBox();
    };
    const onBox1 = () => {
      revealBox('tut-box-1');
      const meanGroup = svgRoot()?.querySelector('#tut-svg-mean');
      if (meanGroup) meanGroup.style.opacity = '1';
      activateBox('tut-box-2', onBox2);
    };
    const onBox2 = () => {
      revealBox('tut-box-2');
      activateSlider();
    };

    activateBox('tut-box-0', onBox0);
  }
}

TutorialIntroContinuousPlugin.info = info;
export default TutorialIntroContinuousPlugin;
