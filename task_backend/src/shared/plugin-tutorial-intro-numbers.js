import { buildCorrectAnswerHTML, renderCorrectAnswer, FADE_MS } from './correct-answer-numbers.js';
import { buildSliderHTML, initSlider } from './slider-numbers.js';
import { buildTrackerHTML } from './tutorial-tracker.js';
import { BOX0, BOX0B, BOX1, BOX2, SAMPLE_COLOR } from './tutorial-text-numbers.js';
/**
 * plugin-tutorial-intro-numbers.js
 * Obs 1 of the numbers tutorial — progressive reveal via click, redesigned
 * this session to a simpler three-click progression (previously four
 * clicks, with a separate click just to reveal the correct-answer panel):
 *
 *   Click 1 (left box 1, BOX0) -> reveals box 1's own text AND box 0b
 *     (top-right box) together, plus the centre example number fading in
 *     -- "you'll see numbers, randomly drawn" (box 1) pairs naturally with
 *     actually showing one (the number) and where it comes from (box 0b).
 *   Click 2 (left box 2, BOX1, the goal text) -> reveals box 2's own text
 *     AND the correct-answer panel (track + ticks + thumb), all at once --
 *     no separate click-to-reveal step for the panel itself anymore. Tied
 *     to the goal text specifically: "estimate the mean" pairs with
 *     actually showing what that mean/answer looks like.
 *   Click 3 (left box 3, BOX2, the slider instructions) -> reveals box 3's
 *     own text AND activates the real response slider.
 *
 * The "Sequence history" tracker is DELIBERATELY never revealed anywhere
 * in this file -- it stays at its initial opacity:0 for this entire
 * observation. It first becomes visible starting at observation 2 (the
 * OBSERVATION plugin, which shows it unconditionally on every call) --
 * i.e. showing sequence history only starts making sense once there IS a
 * sequence longer than one value.
 *
 * The right panel's middle box used to show a KDE curve + a bubbling-
 * then-reveal draw animation (distribution-numbers.js/numbers-draw-
 * animation.js, both deleted this session -- still under task/ if this
 * ever needs reverting), then a slider-style track (correct-answer-
 * numbers.js) revealed via its OWN separate click step -- also removed
 * this session, per the three-click redesign above.
 *
 * Box text is imported from tutorial-text-numbers.js, shared with
 * plugin-tutorial-observation-numbers.js -- never hardcode it here again
 * (see that module's own docstring for why).
 */

const info = {
  name: 'tutorial-intro-numbers',
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

const makeBox = (id, realHTML, isActive, extraStyle = '', extraClass = '') => `
  <div id="${id}" class="tutorial-info-block numbers-tutorial-box${extraClass ? ' ' + extraClass : ''}" style="position:relative;${extraStyle}${isActive ? 'cursor:pointer;' : ''}">
    <span id="${id}-placeholder"
          style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
                 font-weight:bold;color:${isActive ? '#555' : '#ccc'};white-space:nowrap;">
      ${isActive ? 'Click to reveal' : '· · ·'}
    </span>
    <span id="${id}-real" style="visibility:hidden;">${realHTML}</span>
  </div>`;

class TutorialIntroNumbersPlugin {
  constructor(jsPsych) { this.jsPsych = jsPsych; }

  trial(display_el, trial) {
    const self = this;
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
            <div id="tut-centre-number" class="stimulus-number"
                 style="color:${SAMPLE_COLOR};opacity:0;">${example_value}</div>
          </div>

          <div class="tutorial-panel tutorial-panel-right">
            ${makeBox('tut-box-0b', BOX0B, false, '', 'tutorial-right-top-box')}
            <div class="tutorial-right-image-box numbers-tutorial-box dist-canvas" style="position:relative;">
              <span id="tut-ca-placeholder"
                    style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
                           font-weight:bold;color:#ccc;white-space:nowrap;">
                · · ·
              </span>
              <div id="tut-ca-content" style="visibility:hidden;height:100%;">
                <div class="tutorial-panel-caption">Correct answer</div>
                ${buildCorrectAnswerHTML()}
              </div>
            </div>
            <div class="tutorial-right-tracker-box numbers-tutorial-box tutorial-tracker-highlight-white" style="position:relative;">
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
          ${buildSliderHTML({ unset: true, initPos: 0, showValue: true })}
          <button id="submit-btn" class="jspsych-btn" disabled
                  style="font-size:1.6rem;padding:1rem 3.5rem;min-width:200px;">
            Submit
          </button>
        </div>

      </div>`;

    // Tracker is rendered now (values_so_far is just [example_value] at
    // this point) but its whole wrapper (#tut-tracker-content) stays
    // visibility:hidden -- see markup above -- for this ENTIRE
    // observation, behind a permanent "..." placeholder matching the
    // other locked boxes' own convention. Deliberately never revealed
    // anywhere in this file, see module docstring.
    display_el.querySelector('#tut-tracker').innerHTML = buildTrackerHTML({
      nObs: n_obs, obsNum: 1, values: values_so_far, color: SAMPLE_COLOR,
      revealCurrent: false,
    });

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

    const onBox0 = () => {
      revealBox('tut-box-0');
      revealBox('tut-box-0b');
      const el = centerEl();
      el.style.opacity = '0';
      requestAnimationFrame(() => {
        el.style.transition = `opacity ${FADE_MS}ms ease`;
        el.style.opacity = '1';
      });
      activateBox('tut-box-1', onBox1);
    };
    const onBox1 = () => {
      revealBox('tut-box-1');
      // history=[] (obs 1's only observation) -- fadeIn:true for the
      // one-time reveal. No separate click step for this panel anymore --
      // the box's own "..." placeholder (matching the other locked boxes'
      // convention, added this session per explicit direction: the box
      // must show nothing but "..." before this click, not the caption
      // text or an empty track) is removed here, at the same moment the
      // real content (caption + track/ticks/thumb) becomes visible.
      display_el.querySelector('#tut-ca-placeholder')?.remove();
      display_el.querySelector('#tut-ca-content').style.visibility = 'visible';
      renderCorrectAnswer(display_el, { history: [], currentValue: example_value, fadeIn: true });
      activateBox('tut-box-2', onBox2);
    };
    const onBox2 = () => {
      revealBox('tut-box-2');
      activateSlider();
    };

    activateBox('tut-box-0', onBox0);
  }
}

TutorialIntroNumbersPlugin.info = info;
export default TutorialIntroNumbersPlugin;
