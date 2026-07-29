import { buildCorrectAnswerColorsHTML, renderCorrectAnswerColors, FADE_MS } from './correct-answer-colors.js';
import { buildColorsSliderHTMLv2 as buildColorsSliderHTML, initColorsSliderV2 as initColorsSlider } from './slider-colors.js';
import { buildTrackerHTML } from './tutorial-tracker.js';
import { BOX0, BOX0B, BOX1, BOX2, SAMPLE_BLUE, SAMPLE_RED } from './tutorial-text-colors.js';
/**
 * plugin-tutorial-intro-colors.js
 * Obs 1 of the colors tutorial — progressive reveal via click, redesigned
 * this session to a simpler three-click progression (previously four
 * clicks, with a separate click just to reveal the correct-answer panel)
 * -- mirrors plugin-tutorial-intro-numbers.js's identical redesign:
 *
 *   Click 1 (left box 1, BOX0) -> reveals box 1's own text AND box 0b
 *     (top-right box) together, plus the centre example circle recoloring
 *     -- "you'll see balls, randomly drawn" (box 1) pairs naturally with
 *     actually showing one (the circle) and where it comes from (box 0b).
 *   Click 2 (left box 2, BOX1, the goal text) -> reveals box 2's own text
 *     AND the correct-answer panel (bar + dots), all at once -- no
 *     separate click-to-reveal step for the panel itself anymore. Tied to
 *     the goal text specifically: "estimate the percentage" pairs with
 *     actually showing what that percentage/answer looks like.
 *   Click 3 (left box 3, BOX2, the slider instructions) -> reveals box 3's
 *     own text AND activates the real response slider.
 *
 * The "Sequence history" tracker is DELIBERATELY never revealed anywhere
 * in this file -- it stays behind its "..." placeholder for this entire
 * observation. It first becomes visible starting at observation 2 (the
 * OBSERVATION plugin, which shows it unconditionally on every call) --
 * i.e. showing sequence history only starts making sense once there IS a
 * sequence longer than one value.
 *
 * The right panel's middle box used to show a blue/red bar + a bubbling-
 * then-reveal draw animation (tutorial-text-colors.js's own
 * SAMPLE_BLUE/SAMPLE_RED used to come via a separate urn-colors.js's
 * buildUrnSVG + colors-draw-animation.js, the latter deleted this
 * session -- still under task/ if this ever needs reverting), then
 * revealed via its OWN separate click step -- also removed this session,
 * per the three-click redesign above.
 *
 * Both right-column boxes show a "..." placeholder (matching the left
 * column's own locked-box convention) until their real content is
 * revealed -- the correct-answer box's white background/border
 * (.dist-canvas) is on the OUTER, ALWAYS-VISIBLE element, not inside the
 * hidden content wrapper (a real bug found and fixed this session in the
 * numbers version: putting the box's own background/border styling
 * inside the hidden wrapper made the whole box vanish into the page's
 * bare background before reveal, not just its content). The hidden
 * content wrapper also gets an explicit height:100% for the same reason
 * -- correct-answer-colors.js's own outer wrapper needs a REAL resolved
 * height to size itself against, not an auto-height ancestor.
 *
 * Box text is imported from tutorial-text-colors.js, shared with
 * plugin-tutorial-observation-colors.js -- never hardcode it here again
 * (see that module's own docstring for why).
 */

const info = {
  name: 'tutorial-intro-colors',
  parameters: {
    example_value: { type: 'INT',   default: 1   },
    true_p:        { type: 'FLOAT' },
    n_obs:         { type: 'INT',   default: 5   },
    // Tracker -- see tutorial-tracker.js. Obs 1 has no history yet, so
    // this is just [example_value].
    values_so_far: { type: 'OBJECT', default: [] },
  },
};

const makeBox = (id, realHTML, isActive, extraStyle = '', extraClass = '') => `
  <div id="${id}" class="tutorial-info-block colors-tutorial-box${extraClass ? ' ' + extraClass : ''}" style="position:relative;${extraStyle}${isActive ? 'cursor:pointer;' : ''}">
    <span id="${id}-placeholder"
          style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
                 font-weight:bold;color:${isActive ? '#555' : '#ccc'};white-space:nowrap;">
      ${isActive ? 'Click to reveal' : '· · ·'}
    </span>
    <span id="${id}-real" style="visibility:hidden;">${realHTML}</span>
  </div>`;

class TutorialIntroColorsPlugin {
  constructor(jsPsych) { this.jsPsych = jsPsych; }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    const { example_value, n_obs, values_so_far } = trial;

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
            <div id="tut-ball" class="colors-circle" style="opacity:0;"></div>
          </div>
          <div class="tutorial-panel tutorial-panel-right">
            ${makeBox('tut-box-0b', BOX0B, false, '', 'tutorial-right-top-box')}
            <div class="tutorial-right-image-box colors-tutorial-box dist-canvas" style="position:relative;">
              <span id="tut-cac-placeholder"
                    style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
                           font-weight:bold;color:#ccc;white-space:nowrap;">
                · · ·
              </span>
              <div id="tut-cac-content" style="visibility:hidden;height:100%;">
                <div class="tutorial-panel-caption tutorial-panel-caption-correct-answer-colors">Correct answer</div>
                ${buildCorrectAnswerColorsHTML()}
              </div>
            </div>
            <div class="tutorial-right-tracker-box colors-tutorial-box tutorial-tracker-highlight-white" style="position:relative;">
              <div id="tut-tracker-content" style="visibility:hidden;">
                <div class="tutorial-panel-caption">Sequence history</div>
                <div id="tut-tracker"></div>
              </div>
              <span style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
                           font-weight:bold;color:#ccc;white-space:nowrap;">
                · · ·
              </span>
            </div>
          </div>
        </div>
        <div id="tut-slider-wrap" style="visibility:hidden;">
          ${buildColorsSliderHTML({ unset: true, initPos: 50, showValue: true })}
          <div style="text-align:center;margin-top:0.5rem;">
            <button id="submit-btn" class="jspsych-btn" disabled
                    style="font-size:1.6rem;padding:1rem 3.5rem;min-width:200px;">
              Submit
            </button>
          </div>
        </div>
      </div>`;

    // Tracker is rendered now (values_so_far is just [example_value] at
    // this point) but its whole wrapper (#tut-tracker-content) stays
    // visibility:hidden -- see markup above -- for this ENTIRE
    // observation, behind a permanent "..." placeholder matching the
    // other locked boxes' own convention. Deliberately never revealed
    // anywhere in this file, see module docstring.
    display_el.querySelector('#tut-tracker').innerHTML = buildTrackerHTML({
      nObs: n_obs, obsNum: 1, values: values_so_far,
      color: (v) => v === 1 ? SAMPLE_BLUE : SAMPLE_RED,
      revealCurrent: false, renderDot: true,
    });

    const centerEl = () => display_el.querySelector('#tut-ball');
    const jsPsych = this.jsPsych;

    const activateSlider = () => {
      display_el.querySelector('#tut-slider-wrap').style.visibility = 'visible';
      initColorsSlider(display_el, {
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

    const onBox0 = () => {
      revealBox('tut-box-0');
      revealBox('tut-box-0b');
      const el = centerEl();
      const finalColor = example_value === 1 ? SAMPLE_BLUE : SAMPLE_RED;
      el.style.background = '#fff';
      el.style.border = '2px solid #ccc';
      el.style.opacity = '1';
      requestAnimationFrame(() => {
        el.style.transition = `background ${FADE_MS}ms ease, border-color ${FADE_MS}ms ease`;
        el.style.background = finalColor;
        el.style.borderColor = finalColor;
      });
      activateBox('tut-box-1', onBox1);
    };
    const onBox1 = () => {
      revealBox('tut-box-1');
      // history=[] (obs 1's only observation) -- fadeIn:true for the
      // one-time reveal. No separate click step for this panel anymore --
      // the box's own "..." placeholder is removed here, at the same
      // moment the real content (caption + bar/dots) becomes visible.
      display_el.querySelector('#tut-cac-placeholder')?.remove();
      display_el.querySelector('#tut-cac-content').style.visibility = 'visible';
      renderCorrectAnswerColors(display_el, { history: [], currentValue: example_value, fadeIn: true });
      activateBox('tut-box-2', onBox2);
    };
    const onBox2 = () => {
      revealBox('tut-box-2');
      activateSlider();
    };

    activateBox('tut-box-0', onBox0);
  }
}

TutorialIntroColorsPlugin.info = info;
export default TutorialIntroColorsPlugin;
