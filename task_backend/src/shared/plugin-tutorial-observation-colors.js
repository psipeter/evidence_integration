/**
 * plugin-tutorial-observation-colors.js
 * Obs 2-15 of the colors tutorial — same layout as obs 1 but fully revealed.
 *
 * No timeout clock, deliberately — participants need unhurried time to read
 * and think during the tutorial.
 *
 * Submit stays disabled until the participant actually interacts with the
 * slider (mousedown/input, via slider-colors.js's initColorsSlider) -- NOT
 * just once the draw animation completes. An earlier revision force-
 * enabled it as soon as the animation finished whenever slider_default was
 * 'last', which let a participant submit their PREVIOUS response unchanged
 * without ever touching the slider on this new observation -- removed
 * (chat history), matching the identical fix already applied to
 * plugin-tutorial-observation-numbers.js and plugin-observation-
 * colors.js's own real-task behavior, which never had that shortcut.
 *
 * Box text is imported from tutorial-text-colors.js, shared with
 * plugin-tutorial-intro-colors.js -- never hardcode it here again (see that
 * module's own docstring for why).
 *
 * The right column's middle box used to show a blue/red bar + a bubbling-
 * then-reveal draw animation, teaching the FIXED true_p (a separate
 * urn-colors.js's buildUrnSVG + colors-draw-animation.js, the latter
 * deleted this session -- still under task/ if this ever needs
 * reverting; SAMPLE_BLUE/SAMPLE_RED now come from tutorial-text-
 * colors.js directly, which folded urn-colors.js's remaining color
 * constants into itself during a later cleanup pass, for symmetry with
 * numbers not having an equivalent separate file at all). Replaced
 * entirely with correct-answer-colors.js's bar+dots design: a bar that
 * SLIDES to the RUNNING blue-proportion's split point each time a new
 * observation arrives, with a dot per observation. Mirrors the identical
 * fix already applied to plugin-tutorial-observation-numbers.js: the
 * real task scores against the running proportion (config-base.js's
 * colors override, ERROR_MODE: 'running_p'), not the fixed true_p the
 * old figure taught -- and removes an artificial ~1s "wait for the
 * bubbles" delay that existed for every phase, not just the ones later
 * hidden behind an overlay.
 *
 * FIVE PHASES (A, B, C, D, E -- mirrors plugin-tutorial-observation-
 * numbers.js's identically-named/ranged phases), by obs_num -- each phase
 * ONLY changes the top-right box's own text/color and, for E only, the
 * correct-answer panel's contents and the tracker's visibility; nothing
 * else moves or appears/disappears elsewhere on screen:
 *   A (obs 1-3)   -- BOX0B's normal "each ball reflects a hidden
 *                    probability" text, white/plain. Obs 1 specifically is
 *                    the intro plugin, not this file -- it already shows
 *                    BOX0B the same way by default.
 *   B (obs 4-6)   -- SLIDER_REMINDER "the slider remembers your last
 *                    position" reminder -- BLUE (.tutorial-notify-blue).
 *   C (obs 7-9)  -- "...percentage over all balls in the sequence" goal
 *                    reminder -- YELLOW (.tutorial-notify-yellow). The
 *                    history tracker below the correct-answer panel is
 *                    ALSO highlighted in that same yellow during this
 *                    phase (.tutorial-tracker-highlight) -- a highlight,
 *                    not a cover: the tracker's own content stays fully
 *                    visible inside it, unlike phase D below.
 *   D (obs 10-12) -- RECAP_TEXT_1/RECAP_TEXT_2 "use your memory" warning --
 *                    RED (.tutorial-notify-red). The correct-answer panel
 *                    and tracker below it are SEPARATELY hidden behind an
 *                    opaque RED "empty placeholder" overlay
 *                    (.tutorial-hide-wrap/.tutorial-hidden-overlay) during
 *                    this same phase -- the real elements still render
 *                    underneath, completely unmodified; only visually
 *                    covered.
 *   E (obs 13-15) -- Clock demonstration -- YELLOW (.tutorial-notify-
 *                    yellow, matching phase C), text: "You have N seconds
 *                    to submit your response. The countdown clock looks
 *                    like this." (N derived from the real t_obs_ms, not
 *                    hardcoded). The correct-answer panel's contents are
 *                    REPLACED (not hidden behind an overlay, unlike phase
 *                    D) with a real countdown-clock canvas -- the same
 *                    observation-timeout-clock.js renderer the main
 *                    task's per-observation deadline uses -- and
 *                    recolored yellow (.dist-canvas-yellow) to match the
 *                    top box. The TRACKER, unlike the correct-answer
 *                    panel, is not replaced with anything -- it's covered
 *                    by an opaque YELLOW overlay
 *                    (.tutorial-hidden-overlay-yellow, same mechanism as
 *                    phase D's red one). A SECOND clock, visually
 *                    identical to (and started in sync with) the panel
 *                    one, also renders in the actual fixed top-right
 *                    corner position (.timeout-clock) that real trials
 *                    use. Both clocks are PURELY decorative here --
 *                    onTimeout is a no-op, since tutorial screens
 *                    deliberately have no response deadline -- and are
 *                    explicitly stopped once the participant submits,
 *                    matching plugin-observation-colors.js's own
 *                    `stopClock` cleanup.
 *
 * Captions ("Correct answer"/"Sequence history") only render when each
 * box's OWN real content is genuinely visible -- not during phase D (both
 * boxes hidden behind an overlay) or phase E (correct-answer box
 * replaced by the clock demo, tracker hidden behind its own overlay). See
 * showCaptions below -- mirrors plugin-tutorial-observation-numbers.js's
 * identical logic exactly.
 *
 * TRACKER: unlike numbers's numeric tracker, colors's observations are
 * colors, not numbers to display -- there's no separate "number" to show
 * for a blue/red draw, the color IS the content. So this passes
 * renderDot:true and a per-value color FUNCTION (not a fixed color string)
 * to tutorial-tracker.js's buildTrackerHTML -- see that module's own
 * docstring for the shared settled/current/empty logic both modes use.
 */

import { buildColorsSliderHTMLv2 as buildColorsSliderHTML, initColorsSliderV2 as initColorsSlider } from './slider-colors.js';
import { buildCorrectAnswerColorsHTML, renderCorrectAnswerColors, FADE_MS } from './correct-answer-colors.js';
import { startTimeoutClock } from './observation-timeout-clock.js';
import { buildTrackerHTML } from './tutorial-tracker.js';
import { BOX0, BOX0B, BOX1, BOX2, RECAP_TEXT_1, RECAP_TEXT_2, SLIDER_REMINDER, SAMPLE_BLUE, SAMPLE_RED } from './tutorial-text-colors.js';

const info = {
  name: 'tutorial-observation-colors',
  parameters: {
    value:          { type: 'INT',     default: 1      },
    obs_num:        { type: 'INT',     default: 1      },
    n_obs:          { type: 'INT',     default: 5      },
    true_p:         { type: 'FLOAT' },
    slider_default: { type: 'STRING',  default: 'none' },
    init_pos:       { type: 'INT',     default: 50     },
    show_value:     { type: 'BOOLEAN', default: true   },
    // Phase E only (see module docstring) -- duration for both demo
    // clocks. Mirrors numbers's identical param/default.
    t_obs_ms:       { type: 'INT',     default: 7000   },
    // Tracker/history -- see tutorial-tracker.js's own docstring. Includes
    // the current value at the last index.
    values_so_far:  { type: 'OBJECT',  default: [] },
  },
};

/** See module docstring's "FIVE PHASES" for the full rationale.
 * Returns { html, colorClass, phase } for the top-right box (BOX0B's slot). */
function rightTopBoxContent(obsNum, tObsMs) {
  if (obsNum >= 13 && obsNum <= 15) {
    const seconds = Math.round(tObsMs / 1000);
    return {
      html: `You have <strong>${seconds} seconds</strong> to submit your response. The countdown clock looks like this.`,
      colorClass: 'tutorial-notify-yellow',
      phase: 'E',
    };
  }
  if (obsNum >= 10 && obsNum <= 12) {
    return { html: `${RECAP_TEXT_1} ${RECAP_TEXT_2}`, colorClass: 'tutorial-notify-red', phase: 'D' };
  }
  if (obsNum >= 7 && obsNum <= 9) {
    return { html: 'Remember that your goal is to estimate the percentage over <strong>all</strong> balls in the sequence', colorClass: 'tutorial-notify-yellow', phase: 'C' };
  }
  if (obsNum >= 4 && obsNum <= 6) {
    return { html: SLIDER_REMINDER, colorClass: 'tutorial-notify-blue', phase: 'B' };
  }
  return { html: BOX0B, colorClass: '', phase: 'A' };
}

class TutorialObservationColorsPlugin {
  constructor(jsPsych) { this.jsPsych = jsPsych; }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    if (document.activeElement && document.activeElement !== document.body)
      document.activeElement.blur();

    const { value, obs_num, n_obs,
            slider_default, init_pos, show_value, values_so_far, t_obs_ms } = trial;
    const resolvedInitPos = typeof init_pos === 'function' ? init_pos() : (init_pos ?? 50);
    const unset = slider_default === 'none';
    const rightTop = rightTopBoxContent(obs_num, t_obs_ms);
    const hideGraphics = rightTop.phase === 'D';
    const showClock     = rightTop.phase === 'E';
    const kdeOverlay = hideGraphics ? '<div class="tutorial-hidden-overlay"></div>' : '';
    const trackerWrapClass = (hideGraphics || showClock) ? 'tutorial-hide-wrap'
      : rightTop.phase === 'C' ? 'tutorial-tracker-highlight'
      : 'tutorial-tracker-highlight-white';
    const trackerOverlay = hideGraphics ? '<div class="tutorial-hidden-overlay"></div>'
      : showClock ? '<div class="tutorial-hidden-overlay-yellow"></div>'
      : '';
    // Captions only render when each box's OWN real content is genuinely
    // visible -- not during phase D (both boxes hidden behind an
    // overlay) or phase E (correct-answer box replaced by the clock
    // demo, tracker hidden behind its own overlay). Mirrors
    // plugin-tutorial-observation-numbers.js's identical logic exactly.
    const showCaptions = !hideGraphics && !showClock;

    display_el.innerHTML = `
      <div class="tutorial-title">Tutorial</div>
      ${showClock ? '<canvas id="tut-corner-clock" class="timeout-clock" width="88" height="88"></canvas>' : ''}
      <div class="tutorial-wrap">
        <div class="tutorial-top-row">
          <div class="tutorial-panel">
            <p class="tutorial-info-block colors-tutorial-box"><span>${BOX0}</span></p>
            <p class="tutorial-info-block colors-tutorial-box"><span>${BOX1}</span></p>
            <p class="tutorial-info-block colors-tutorial-box"><span>${BOX2}</span></p>
          </div>
          <div class="tutorial-panel tutorial-panel-centre">
            <div id="tut-ball" class="colors-circle" style="opacity:0;"></div>
          </div>
          <div class="tutorial-panel tutorial-panel-right">
            <p class="tutorial-info-block tutorial-right-top-box colors-tutorial-box${rightTop.colorClass ? ' ' + rightTop.colorClass : ''}">
              <span>${rightTop.html}</span>
            </p>
            <div class="tutorial-hide-wrap tutorial-right-image-box colors-tutorial-box">
              ${showCaptions ? '<div class="tutorial-panel-caption tutorial-panel-caption-correct-answer-colors">Correct answer</div>' : ''}
              ${showClock
                ? '<div class="dist-canvas dist-canvas-yellow" style="display:flex;align-items:center;justify-content:center;"><canvas id="tut-obs-clock" width="120" height="120"></canvas></div>'
                : `<div class="dist-canvas">${buildCorrectAnswerColorsHTML()}</div>`}
              ${kdeOverlay}
            </div>
            <div class="tutorial-right-tracker-box colors-tutorial-box ${trackerWrapClass}">
              ${showCaptions ? '<div class="tutorial-panel-caption">Sequence history</div>' : ''}
              <div id="tut-tracker"></div>
              ${trackerOverlay}
            </div>
          </div>
        </div>
        ${buildColorsSliderHTML({ unset, initPos: resolvedInitPos, showValue: show_value })}
        <div style="text-align:center;margin-top:0.5rem;">
          <button id="submit-btn" class="jspsych-btn" disabled
            style="font-size:1.6rem;padding:1rem 3.5rem;min-width:200px;">
            Submit
          </button>
        </div>
      </div>`;

    const centerEl = display_el.querySelector('#tut-ball');
    let stopCornerClock = null;
    let stopKdeClock    = null;

    // Centre circle's own reveal -- a simple fade, uniform across EVERY
    // phase now (no more bubble-animation-linked timing at all). Runs
    // regardless of phase, including E (whose own clock demo below is
    // independent of this).
    const finalColor = value === 1 ? SAMPLE_BLUE : SAMPLE_RED;
    centerEl.style.background = '#fff';
    centerEl.style.border = '2px solid #ccc';
    centerEl.style.opacity = '1';
    requestAnimationFrame(() => {
      centerEl.style.transition = `background ${FADE_MS}ms ease, border-color ${FADE_MS}ms ease`;
      centerEl.style.background = finalColor;
      centerEl.style.borderColor = finalColor;
    });

    const revealTrackerDot = () => {
      const trackerDot = display_el.querySelector('#tut-tracker-current-num');
      if (trackerDot) {
        trackerDot.style.transition = `opacity ${FADE_MS}ms ease`;
        trackerDot.style.opacity = '1';
      }
    };

    if (showClock) {
      display_el.querySelector('#tut-tracker').innerHTML = buildTrackerHTML({
        nObs: n_obs, obsNum: obs_num, values: values_so_far,
        color: (v) => v === 1 ? SAMPLE_BLUE : SAMPLE_RED,
        revealCurrent: false, renderDot: true,
      });
      revealTrackerDot();

      // Both clocks are purely decorative -- onTimeout is a no-op, since
      // tutorial screens have no real deadline (module docstring). Both
      // are stopped on submit below, mirroring plugin-observation-
      // colors.js's own `stopClock` cleanup.
      const cornerCanvas = display_el.querySelector('#tut-corner-clock');
      const kdeCanvas    = display_el.querySelector('#tut-obs-clock');
      stopCornerClock = startTimeoutClock(cornerCanvas, t_obs_ms, () => {});
      stopKdeClock    = startTimeoutClock(kdeCanvas, t_obs_ms, () => {});
    } else {
      // history = all values BEFORE this one; renderCorrectAnswerColors
      // adds the new dot for `value` itself and SLIDES the bar from
      // history's own running blue-proportion to history+[value]'s.
      // Still rendered unconditionally even during phase D (hidden
      // behind the red overlay above) -- deliberately not skipped,
      // matching the correct-answer panel's whole point of representing
      // ground truth regardless of whether the participant can currently
      // see it.
      const history = values_so_far.slice(0, -1);
      renderCorrectAnswerColors(display_el, { history, currentValue: value });

      display_el.querySelector('#tut-tracker').innerHTML = buildTrackerHTML({
        nObs: n_obs, obsNum: obs_num, values: values_so_far,
        color: (v) => v === 1 ? SAMPLE_BLUE : SAMPLE_RED,
        revealCurrent: false, renderDot: true,
      });
      revealTrackerDot();
    }

    initColorsSlider(display_el, {
      unset, showValue: show_value,
      onFinish: () => {
        if (stopCornerClock) stopCornerClock();
        if (stopKdeClock)    stopKdeClock();
        const response = parseInt(display_el.querySelector('#response-slider').value);
        this.jsPsych.finishTrial({ response, timed_out: false });
      },
    });
  }
}

TutorialObservationColorsPlugin.info = info;
export default TutorialObservationColorsPlugin;
