/**
 * plugin-tutorial-observation-continuous.js
 * Obs 2-15 of the continuous tutorial. Three-column layout:
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
 * plugin-tutorial-observation-binary.js's startBinaryDrawAnimation. Submit
 * stays disabled until the participant actually interacts with the slider
 * (mousedown/input, via slider-continuous.js's initSlider) -- NOT just
 * once the animation completes -- see chat history for why an earlier
 * revision's shortcut here was removed.
 *
 * Box text is imported from tutorial-text-continuous.js, shared with
 * plugin-tutorial-intro-continuous.js -- never hardcode it here again (see
 * that module's own docstring for why).
 *
 * Centre column is the plain original layout (just the centered number) --
 * two prior revisions added, then removed, extra structure/popups there;
 * everything now lives in the top-right box instead (see below), so
 * there's no reason to touch the centre column at all anymore.
 *
 * THREE PHASES (A, C, D -- "B" removed, chat history: it was "the slider
 * remembers" reminder, now dropped entirely rather than renumbering the
 * rest), by obs_num -- each phase ONLY changes the top-right box's own
 * text and background color; nothing else moves or appears/disappears
 * elsewhere on screen (an earlier revision also had a separate box next
 * to the Submit button -- removed, "no more extra box near submit").
 * Explicit obs ranges, not an even split:
 *   A (obs 1-5)   -- BOX0B's normal "hidden distribution" text,
 *                    white/plain (no color class at all). Obs 1
 *                    specifically is the intro plugin (plugin-tutorial-
 *                    intro-continuous.js), not this file -- it already
 *                    shows BOX0B the same way by default, so obs 1-5 read
 *                    as one continuous phase across both files even
 *                    though only 2-5 are actually implemented here.
 *   C (obs 6-10)  -- "...mean of all numbers in this sequence" goal
 *                    reminder -- YELLOW (.tutorial-notify-yellow). The
 *                    history tracker below the KDE figure is ALSO
 *                    highlighted in that same yellow during this phase --
 *                    a highlight, not a cover: the tracker's own content
 *                    stays fully visible inside it, unlike phase D below.
 *                    Uses its OWN class (.tutorial-tracker-highlight), not
 *                    .tutorial-info-block + .tutorial-notify-yellow
 *                    directly, despite matching those colors exactly --
 *                    .tutorial-info-block's normal 0.6rem/0.8rem padding
 *                    ate into the already-tight 15-slot tracker row's
 *                    available width, and at this phase's enlarged font
 *                    size (chat history) that was JUST enough to make
 *                    adjacent slots' underlines touch and read as one
 *                    continuous line rather than 15 separate ones -- a
 *                    real bug, not a hypothetical risk. The dedicated
 *                    class uses a much smaller padding instead, so the
 *                    row keeps its full working width.
 *   D (obs 11-15) -- RECAP_TEXT_1/RECAP_TEXT_2 "use your memory" warning
 *                    -- RED (.tutorial-notify-red). The KDE figure and
 *                    tracker below it are SEPARATELY hidden behind an
 *                    opaque RED "empty placeholder" overlay
 *                    (.tutorial-hide-wrap/.tutorial-hidden-overlay,
 *                    matching .tutorial-notify-red's own colors -- all
 *                    three right-column boxes read as one consistent red
 *                    phase now, not two different colors) during this
 *                    same phase -- the real elements still render
 *                    underneath, completely unmodified; only visually
 *                    covered.
 */

import { buildDistributionSVG } from './distribution-continuous.js';
import { buildSliderHTML, initSlider } from './slider-continuous.js';
import { startContinuousDrawAnimation, FADE_MS as DRAW_FADE_MS } from './continuous-draw-animation.js';
import { buildTrackerHTML } from './tutorial-tracker.js';
import { BOX0, BOX0B, BOX1, BOX2, SAMPLE_COLOR, RECAP_TEXT_1, RECAP_TEXT_2 } from './tutorial-text-continuous.js';

const info = {
  name: 'tutorial-observation-continuous',
  parameters: {
    value:          { type: 'INT',     default: 50 },
    obs_num:        { type: 'INT',     default: 1  },
    n_obs:          { type: 'INT',     default: 5  },
    true_mean:      { type: 'FLOAT' },
    true_std:       { type: 'FLOAT' },
    slider_default: { type: 'STRING',  default: 'none' },
    init_pos:       { type: 'INT',     default: 54 },
    show_value:     { type: 'BOOLEAN', default: true  },
    // Tracker/history -- see tutorial-tracker.js and distribution-
    // continuous.js's own docstrings. Includes the current value at the
    // last index.
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
    return { html: 'Remember that your goal is to estimate the mean of <strong>all</strong> numbers in this sequence', colorClass: 'tutorial-notify-yellow', phase: 'C' };
  }
  return { html: BOX0B, colorClass: '', phase: 'A' };
}

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
            slider_default, init_pos, show_value, values_so_far } = trial;
    const resolvedInitPos = typeof init_pos === 'function' ? init_pos() : (init_pos ?? 0);
    const unset = slider_default === 'none';
    const rightTop = rightTopBoxContent(obs_num);
    const hideGraphics = rightTop.phase === 'D';
    const hideOverlay = hideGraphics ? '<div class="tutorial-hidden-overlay"></div>' : '';
    // Tracker's own wrapper: phase D hides it (opaque red overlay, module
    // docstring); phase C instead HIGHLIGHTS it in the same yellow, via a
    // dedicated .tutorial-tracker-highlight class (NOT .tutorial-info-
    // block + .tutorial-notify-yellow directly -- that combination's
    // normal padding broke the tracker's underlines, see module
    // docstring) -- content stays fully visible inside it.
    const trackerWrapClass = hideGraphics ? 'tutorial-hide-wrap'
      : rightTop.phase === 'C' ? 'tutorial-tracker-highlight'
      : '';

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

          <!-- CENTRE: plain original layout -->
          <div class="tutorial-panel tutorial-panel-centre">
            <div id="stimulus-display" class="stimulus-number"
              style="color:${SAMPLE_COLOR};opacity:0;">${value}</div>
          </div>

          <!-- RIGHT: top box's text/color varies by phase (module docstring);
               KDE + tracker each wrapped so phase D can overlay them
               opaquely without touching the JS that populates them below. -->
          <div class="tutorial-panel tutorial-panel-right">
            <p class="tutorial-info-block${rightTop.colorClass ? ' ' + rightTop.colorClass : ''}">
              <span>${rightTop.html}</span>
            </p>
            <div class="tutorial-hide-wrap" style="flex:1;">
              <div id="dist-svg" class="dist-canvas" style="line-height:0;"></div>
              ${hideOverlay}
            </div>
            <div class="${trackerWrapClass}">
              <div id="tut-tracker"></div>
              ${hideOverlay}
            </div>
          </div>

        </div>

        <!-- BOTTOM: full-width slider + submit -->
        ${buildSliderHTML({ unset, initPos: resolvedInitPos, showValue: show_value })}
        <button id="submit-btn" class="jspsych-btn" disabled
          style="font-size:1.6rem;padding:1rem 3.5rem;min-width:200px;">
          Submit
        </button>

      </div>`;

    // Draw distribution — axis/dist/mean fully revealed; the obs marker
    // always starts hidden regardless (see distribution-continuous.js) and
    // is revealed by the draw animation below. `history` (all values BEFORE
    // this one) draws the faded past-observation ticks. Still runs
    // unconditionally even during phase D (hidden behind the yellow
    // overlay above) -- deliberately not skipped, see module docstring.
    const history = values_so_far.slice(0, -1);
    display_el.querySelector('#dist-svg').innerHTML =
      buildDistributionSVG(true_mean, true_std, value, true, history);

    display_el.querySelector('#tut-tracker').innerHTML = buildTrackerHTML({
      nObs: n_obs, obsNum: obs_num, values: values_so_far, color: SAMPLE_COLOR,
      revealCurrent: false,
    });

    const svgRoot   = display_el.querySelector('#dist-svg svg');
    const centerEl  = display_el.querySelector('#stimulus-display');

    startContinuousDrawAnimation({
      svgRoot,
      centerEl,
      true_mean,
      true_std,
      obsNum: obs_num,
      onReveal: () => {
        const trackerNum = display_el.querySelector('#tut-tracker-current-num');
        if (trackerNum) {
          trackerNum.style.transition = `opacity ${DRAW_FADE_MS}ms ease`;
          trackerNum.style.opacity = '1';
        }
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
