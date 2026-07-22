/**
 * plugin-tutorial-observation-binary.js
 * Obs 2-15 of the binary tutorial — same layout as obs 1 but fully revealed.
 *
 * No timeout clock, deliberately — participants need unhurried time to read
 * and think during the tutorial.
 *
 * Submit stays disabled until the participant actually interacts with the
 * slider (mousedown/input, via slider-binary.js's initBinarySlider) -- NOT
 * just once the draw animation completes. An earlier revision force-
 * enabled it as soon as the animation finished whenever slider_default was
 * 'last', which let a participant submit their PREVIOUS response unchanged
 * without ever touching the slider on this new observation -- removed
 * (chat history), matching the identical fix already applied to
 * plugin-tutorial-observation-continuous.js and plugin-observation-
 * continuous.js's own real-task behavior, which never had that shortcut.
 *
 * Box text is imported from tutorial-text-binary.js, shared with
 * plugin-tutorial-intro-binary.js -- never hardcode it here again (see that
 * module's own docstring for why).
 *
 * THREE PHASES (A, C, D -- mirrors plugin-tutorial-observation-
 * continuous.js's identically-named/ranged phases, chat history), by
 * obs_num -- each phase ONLY changes the top-right box's own text and
 * background color; nothing else moves or appears/disappears elsewhere on
 * screen:
 *   A (obs 1-5)   -- BOX0B's normal "each ball reflects a hidden
 *                    probability" text, white/plain. Obs 1 specifically is
 *                    the intro plugin, not this file -- it already shows
 *                    BOX0B the same way by default.
 *   C (obs 6-10)  -- "...ratio over all balls in the sequence" goal
 *                    reminder -- YELLOW (.tutorial-notify-yellow). The
 *                    history tracker below the urn figure is ALSO
 *                    highlighted in that same yellow during this phase
 *                    (.tutorial-tracker-highlight, NOT .tutorial-info-block
 *                    + .tutorial-notify-yellow directly -- see continuous's
 *                    own identical note on why that combination's padding
 *                    broke a tight tracker row's spacing) -- a highlight,
 *                    not a cover: the tracker's own content stays fully
 *                    visible inside it, unlike phase D below.
 *   D (obs 11-15) -- RECAP_TEXT_1/RECAP_TEXT_2 "use your memory" warning --
 *                    RED (.tutorial-notify-red). The urn figure and tracker
 *                    below it are SEPARATELY hidden behind an opaque RED
 *                    "empty placeholder" overlay (.tutorial-hide-wrap/
 *                    .tutorial-hidden-overlay) during this same phase --
 *                    the real elements still render underneath, completely
 *                    unmodified; only visually covered.
 *
 * TRACKER: unlike continuous's numeric tracker, binary's observations are
 * colors, not numbers to display -- there's no separate "number" to show
 * for a blue/red draw, the color IS the content. So this passes
 * renderDot:true and a per-value color FUNCTION (not a fixed color string)
 * to tutorial-tracker.js's buildTrackerHTML -- see that module's own
 * docstring for the shared settled/current/empty logic both modes use.
 * Also, deliberately NOT adding a "faded history of past draws" overlay
 * to the urn figure itself the way distribution-continuous.js's
 * #tut-svg-history does for the KDE curve: that continuous feature places
 * each past VALUE at its own x-position along a 0-100 axis, which the urn
 * bar has no equivalent of (a categorical blue/red split has no "position"
 * a past draw's color could sit at). The tracker alone carries the
 * "history" function for binary; no urn-binary.js changes were needed.
 */

import { buildBinarySliderHTMLv2 as buildBinarySliderHTML, initBinarySliderV2 as initBinarySlider } from './slider-binary.js';
import { buildUrnSVG, SAMPLE_BLUE, SAMPLE_RED } from './urn-binary.js';
import { startBinaryDrawAnimation, FADE_MS as DRAW_FADE_MS } from './binary-draw-animation.js';
import { buildTrackerHTML } from './tutorial-tracker.js';
import { BOX0, BOX0B, BOX1, BOX2, RECAP_TEXT_1, RECAP_TEXT_2 } from './tutorial-text-binary.js';

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
    // Tracker/history -- see tutorial-tracker.js's own docstring. Includes
    // the current value at the last index.
    values_so_far:  { type: 'OBJECT',  default: [] },
  },
};

/** See module docstring's "THREE PHASES" for the full rationale.
 * Returns { html, colorClass, phase } for the top-right box (BOX0B's slot). */
function rightTopBoxContent(obsNum) {
  if (obsNum >= 11 && obsNum <= 15) {
    return { html: `${RECAP_TEXT_1} ${RECAP_TEXT_2}`, colorClass: 'tutorial-notify-red', phase: 'D' };
  }
  if (obsNum >= 6 && obsNum <= 10) {
    return { html: 'Remember that your goal is to estimate the ratio over <strong>all</strong> balls in the sequence', colorClass: 'tutorial-notify-yellow', phase: 'C' };
  }
  return { html: BOX0B, colorClass: '', phase: 'A' };
}

class TutorialObservationBinaryPlugin {
  constructor(jsPsych) { this.jsPsych = jsPsych; }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    if (document.activeElement && document.activeElement !== document.body)
      document.activeElement.blur();

    const { value, obs_num, n_obs, true_p,
            slider_default, init_pos, show_value, values_so_far } = trial;
    const resolvedInitPos = typeof init_pos === 'function' ? init_pos() : (init_pos ?? 50);
    const unset = slider_default === 'none';
    const rightTop = rightTopBoxContent(obs_num);
    const hideGraphics = rightTop.phase === 'D';
    const hideOverlay = hideGraphics ? '<div class="tutorial-hidden-overlay"></div>' : '';
    // Tracker's own wrapper: phase D hides it (opaque red overlay, module
    // docstring); phase C instead HIGHLIGHTS it in the same yellow, via a
    // dedicated .tutorial-tracker-highlight class (not .tutorial-info-
    // block + .tutorial-notify-yellow directly -- see continuous's
    // identical note on why). Content stays fully visible inside it.
    const trackerWrapClass = hideGraphics ? 'tutorial-hide-wrap'
      : rightTop.phase === 'C' ? 'tutorial-tracker-highlight'
      : '';

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
            <p class="tutorial-info-block${rightTop.colorClass ? ' ' + rightTop.colorClass : ''}">
              <span>${rightTop.html}</span>
            </p>
            <div class="tutorial-hide-wrap" style="flex:1;">
              <div id="urn-svg" class="dist-canvas" style="line-height:0;"></div>
              ${hideOverlay}
            </div>
            <div class="${trackerWrapClass}">
              <div id="tut-tracker"></div>
              ${hideOverlay}
            </div>
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

    display_el.querySelector('#tut-tracker').innerHTML = buildTrackerHTML({
      nObs: n_obs, obsNum: obs_num, values: values_so_far,
      color: (v) => v === 1 ? SAMPLE_BLUE : SAMPLE_RED,
      revealCurrent: false, renderDot: true,
    });

    const svgRoot   = display_el.querySelector('#urn-svg svg');
    const centerEl  = display_el.querySelector('#tut-ball');

    startBinaryDrawAnimation({
      svgRoot,
      centerEl,
      true_p,
      currentValue: value,
      obsNum:       obs_num,
      onReveal: () => {
        const trackerDot = display_el.querySelector('#tut-tracker-current-num');
        if (trackerDot) {
          trackerDot.style.transition = `opacity ${DRAW_FADE_MS}ms ease`;
          trackerDot.style.opacity = '1';
        }
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
