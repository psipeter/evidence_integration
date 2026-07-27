/**
 * plugin-tutorial-observation-numbers.js
 * Obs 2-15 of the numbers tutorial. Three-column layout:
 *   Left  — goal/observation text
 *   Centre — identical to main task (number + slider + submit)
 *   Right  — distribution plot
 *
 * No timeout clock, deliberately — participants need unhurried time to read
 * and think during the tutorial. Wires the slider synchronously right after
 * setting innerHTML (Pattern A: no on_load, no async, no rAF/setTimeout
 * deferral) — same shape as plugin-tutorial-observation-colors.js.
 *
 * The distribution SVG is built by the shared distribution-numbers.js
 * (revealed=true here — axis/dist/mean fully shown; the #tut-svg-obs marker
 * always starts hidden regardless, and the falling-bubble draw animation
 * (numbers-draw-animation.js) reveals it — same auto-running pattern as
 * plugin-tutorial-observation-colors.js's startColorsDrawAnimation. Submit
 * stays disabled until the participant actually interacts with the slider
 * (mousedown/input, via slider-numbers.js's initSlider) -- NOT just
 * once the animation completes -- see chat history for why an earlier
 * revision's shortcut here was removed.
 *
 * Box text is imported from tutorial-text-numbers.js, shared with
 * plugin-tutorial-intro-numbers.js -- never hardcode it here again (see
 * that module's own docstring for why).
 *
 * Centre column is the plain original layout (just the centered number) --
 * two prior revisions added, then removed, extra structure/popups there;
 * everything now lives in the top-right box instead (see below), so
 * there's no reason to touch the centre column at all anymore.
 *
 * FIVE PHASES (A, B, C, D, E -- B reintroduced this session after being
 * designed-then-dropped in an earlier one; each phase is now exactly 3
 * observations, chat history), by obs_num -- each phase ONLY changes the
 * top-right box's own text/color and, for E only, the KDE box's contents
 * and the tracker's visibility; nothing else moves or appears/disappears
 * elsewhere on screen (an earlier revision also had a separate box next to
 * the Submit button -- removed, "no more extra box near submit").
 * Explicit obs ranges (3 each, 5 phases x 3 = 15 total, matching the
 * tutorial's full length):
 *   A (obs 1-3)   -- BOX0B's normal "hidden distribution" text,
 *                    white/plain (no color class at all). Obs 1
 *                    specifically is the intro plugin (plugin-tutorial-
 *                    intro-numbers.js), not this file -- it already
 *                    shows BOX0B the same way by default, so obs 1-3 read
 *                    as one numbers phase across both files even
 *                    though only 2-3 are actually implemented here.
 *   B (obs 4-6)   -- SLIDER_REMINDER "the slider remembers your last
 *                    position" reminder -- BLUE (.tutorial-notify-blue).
 *                    Reintroduced this session -- an earlier session
 *                    designed this exact phase, then dropped it entirely
 *                    per explicit direction rather than renumbering the
 *                    rest (obs 2-15 simply fell under phase A/C/D at the
 *                    time). Brought back now that phase E (below) also
 *                    needed a slot, evening every phase out to 3
 *                    observations apiece.
 *   C (obs 7-9)  -- "...mean of all numbers in this sequence" goal
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
 *                    adjacent slots' markers touch and read as one
 *                    numbers line rather than 15 separate ones -- a
 *                    real bug, not a hypothetical risk. The dedicated
 *                    class uses a much smaller padding instead, so the
 *                    row keeps its full working width.
 *   D (obs 10-12) -- RECAP_TEXT_1/RECAP_TEXT_2 "use your memory" warning
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
 *   E (obs 13-15) -- Clock demonstration (chat history, this session) --
 *                    YELLOW (.tutorial-notify-yellow, matching phase C --
 *                    both right-hand boxes get this same styling, chat
 *                    history), text: "You have N seconds to submit your
 *                    response. The countdown clock looks like this."
 *                    (N derived from the real t_obs_ms, not hardcoded --
 *                    see the "prefer no hardcoded literal" convention
 *                    elsewhere in this file). The KDE box's contents are
 *                    REPLACED (not hidden behind an overlay, unlike phase
 *                    D) with a real countdown-clock canvas -- the same
 *                    observation-timeout-clock.js renderer the main
 *                    task's per-observation deadline uses -- and recolored
 *                    yellow (.dist-canvas-yellow) to match the top box.
 *                    The TRACKER, unlike the KDE box, is not replaced with
 *                    anything -- it's covered by an opaque YELLOW overlay
 *                    (.tutorial-hidden-overlay-yellow, same mechanism as
 *                    phase D's red one, matching the other two right-
 *                    column boxes' color during this phase) that
 *                    preserves its layout space without showing its
 *                    content, since this phase has nothing to do with
 *                    sequence history. A SECOND clock,
 *                    visually identical to (and started in sync with) the
 *                    KDE-box one, also renders in the actual fixed
 *                    top-right corner position (.timeout-clock) that real
 *                    trials use -- so this phase shows the clock both
 *                    large/explained AND in its real on-screen position
 *                    at once. Both clocks are PURELY decorative here:
 *                    their onTimeout callback is a no-op, since tutorial
 *                    screens deliberately have no response deadline
 *                    (module docstring, above) -- reaching zero does
 *                    nothing, Submit stays available exactly as in every
 *                    other tutorial phase. Both are explicitly stopped
 *                    (rAF cancelled, visibilitychange listener removed)
 *                    once the participant submits, matching the real
 *                    observation plugins' own cleanup (see plugin-
 *                    observation-numbers.js's `stopClock`), so nothing
 *                    keeps running into subsequent screens.
 *
 * FIXED-HEIGHT TOP BOX (chat history, this session): the top-right box
 * above ALWAYS carries .tutorial-right-top-box (see style.css) regardless
 * of phase, fixing its height so its own (phase-dependent, variable-
 * length) text can no longer grow/shrink the whole right column -- and
 * therefore the slider's vertical position beneath it -- as obs_num moves
 * between phases.
 */

import { buildDistributionSVG } from './distribution-numbers.js';
import { buildSliderHTML, initSlider } from './slider-numbers.js';
import { startNumbersDrawAnimation, FADE_MS as DRAW_FADE_MS } from './numbers-draw-animation.js';
import { startTimeoutClock } from './observation-timeout-clock.js';
import { buildTrackerHTML } from './tutorial-tracker.js';
import { BOX0, BOX0B, BOX1, BOX2, SAMPLE_COLOR, RECAP_TEXT_1, RECAP_TEXT_2, SLIDER_REMINDER } from './tutorial-text-numbers.js';

const info = {
  name: 'tutorial-observation-numbers',
  parameters: {
    value:          { type: 'INT',     default: 50 },
    obs_num:        { type: 'INT',     default: 1  },
    n_obs:          { type: 'INT',     default: 5  },
    true_mean:      { type: 'FLOAT' },
    true_std:       { type: 'FLOAT' },
    slider_default: { type: 'STRING',  default: 'none' },
    init_pos:       { type: 'INT',     default: 54 },
    show_value:     { type: 'BOOLEAN', default: true  },
    // Phase E only (see module docstring) -- duration for both demo
    // clocks. Default mirrors config-base.js's DEFAULTS.T_OBS_MS (not
    // imported directly -- this shared/ file has no dependency on
    // config-base.js elsewhere, and every other default here is likewise
    // a plain literal, not threaded from config-base.js).
    t_obs_ms:       { type: 'INT',     default: 7000 },
    // Tracker/history -- see tutorial-tracker.js and distribution-
    // numbers.js's own docstrings. Includes the current value at the
    // last index.
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
    return { html: 'Remember that your goal is to estimate the mean of <strong>all</strong> numbers in this sequence', colorClass: 'tutorial-notify-yellow', phase: 'C' };
  }
  if (obsNum >= 4 && obsNum <= 6) {
    return { html: SLIDER_REMINDER, colorClass: 'tutorial-notify-blue', phase: 'B' };
  }
  return { html: BOX0B, colorClass: '', phase: 'A' };
}

class TutorialObservationNumbersPlugin {
  constructor(jsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    if (document.activeElement && document.activeElement !== document.body) {
      document.activeElement.blur();
    }

    const { value, obs_num, n_obs, true_mean, true_std,
            slider_default, init_pos, show_value, values_so_far, t_obs_ms } = trial;
    const resolvedInitPos = typeof init_pos === 'function' ? init_pos() : (init_pos ?? 0);
    const unset = slider_default === 'none';
    const rightTop = rightTopBoxContent(obs_num, t_obs_ms);
    const hideGraphics = rightTop.phase === 'D';
    const showClock     = rightTop.phase === 'E';
    const kdeOverlay = hideGraphics ? '<div class="tutorial-hidden-overlay"></div>' : '';
    // Tracker's own wrapper/overlay: phase D hides it (opaque red
    // overlay, module docstring); phase E ALSO hides it, but with a
    // YELLOW overlay instead (.tutorial-hidden-overlay-yellow) -- matches
    // the other two right-column boxes' color during this same phase
    // (chat history, this session -- an earlier revision used white/gray
    // here, which read as inconsistent next to the other two). Phase C
    // instead HIGHLIGHTS the tracker in the same yellow, via a dedicated
    // .tutorial-tracker-highlight class (NOT .tutorial-info-block +
    // .tutorial-notify-yellow directly -- that combination's normal
    // padding broke the tracker's markers, see module docstring) -- a
    // highlight, not a cover: content stays fully visible inside it,
    // unlike D/E. Phases A/B get the SAME highlight-box treatment, just
    // white instead of yellow (.tutorial-tracker-highlight-white, chat
    // history) -- there's no color reason to highlight them, but leaving
    // the tracker bare there (vs. C/D/E's boxed look) read as
    // inconsistent, so every phase now gets some version of this box.
    const trackerWrapClass = (hideGraphics || showClock) ? 'tutorial-hide-wrap'
      : rightTop.phase === 'C' ? 'tutorial-tracker-highlight'
      : 'tutorial-tracker-highlight-white';
    const trackerOverlay = hideGraphics ? '<div class="tutorial-hidden-overlay"></div>'
      : showClock ? '<div class="tutorial-hidden-overlay-yellow"></div>'
      : '';

    display_el.innerHTML = `
      <div class="tutorial-title">Tutorial</div>
      ${showClock ? '<canvas id="tut-corner-clock" class="timeout-clock" width="88" height="88"></canvas>' : ''}

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

          <!-- RIGHT: top box's text/color varies by phase (module docstring).
               KDE box and tracker box are each INDEPENDENTLY fixed-height
               (chat history, this session) via .tutorial-right-image-box/
               -tracker-box (see style.css) -- an earlier revision merged
               them into one shared-budget wrapper, which fixed cross-phase
               resizing but squeezed the KDE figure too small to read
               comfortably; un-merging lets each get its own generous,
               independently-tunable size while keeping the same
               anti-resize guarantee (flex:0 0 <clamp> + overflow:hidden on
               EACH box individually, not on a shared parent). Phase E
               replaces the KDE box's contents outright (a clock canvas,
               sized to comfortably fit this box's own floor value, not
               shrunk to survive a cramped shared budget the way an
               earlier revision did -- recolored yellow) rather than
               wrapping/overlaying it, but DOES cover the tracker box
               (white, not red -- see module docstring). Top box always
               carries .tutorial-right-top-box for a fixed height across
               every phase (module docstring). -->
          <div class="tutorial-panel tutorial-panel-right">
            <p class="tutorial-info-block tutorial-right-top-box${rightTop.colorClass ? ' ' + rightTop.colorClass : ''}">
              <span>${rightTop.html}</span>
            </p>
            <div class="tutorial-hide-wrap tutorial-right-image-box">
              ${showClock
                ? '<div class="dist-canvas dist-canvas-yellow" style="display:flex;align-items:center;justify-content:center;"><canvas id="tut-obs-clock" width="120" height="120"></canvas></div>'
                : '<div id="dist-svg" class="dist-canvas" style="line-height:0;"></div>'}
              ${kdeOverlay}
            </div>
            <div class="tutorial-right-tracker-box ${trackerWrapClass}">
              <div id="tut-tracker"></div>
              ${trackerOverlay}
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

    const centerEl = display_el.querySelector('#stimulus-display');
    let stopCornerClock = null;
    let stopKdeClock    = null;

    if (showClock) {
      // Phase E: no distribution SVG exists to bubble-animate around (see
      // module docstring), so reveal the centre number/tracker slot with a
      // simple fade instead -- the same fallback numbers-draw-
      // animation.js's showFinal() uses for prefers-reduced-motion, just
      // inlined here rather than routed through that module (which assumes
      // an svgRoot to query).
      display_el.querySelector('#tut-tracker').innerHTML = buildTrackerHTML({
        nObs: n_obs, obsNum: obs_num, values: values_so_far, color: SAMPLE_COLOR,
        revealCurrent: false,
      });
      centerEl.style.opacity = '0';
      requestAnimationFrame(() => {
        centerEl.style.transition = `opacity ${DRAW_FADE_MS}ms ease`;
        centerEl.style.opacity = '1';
      });
      const trackerNum = display_el.querySelector('#tut-tracker-current-num');
      if (trackerNum) {
        trackerNum.style.transition = `opacity ${DRAW_FADE_MS}ms ease`;
        trackerNum.style.opacity = '1';
      }

      // Both clocks are purely decorative -- onTimeout is a no-op, since
      // tutorial screens have no real deadline (module docstring). Both
      // are stopped on submit below, mirroring plugin-observation-
      // numbers.js's own `stopClock` cleanup.
      const cornerCanvas = display_el.querySelector('#tut-corner-clock');
      const kdeCanvas    = display_el.querySelector('#tut-obs-clock');
      stopCornerClock = startTimeoutClock(cornerCanvas, t_obs_ms, () => {});
      stopKdeClock    = startTimeoutClock(kdeCanvas, t_obs_ms, () => {});
    } else {
      // Draw distribution — axis/dist/mean fully revealed; the obs marker
      // always starts hidden regardless (see distribution-numbers.js) and
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

      const svgRoot = display_el.querySelector('#dist-svg svg');

      startNumbersDrawAnimation({
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
    }

    // Wire immediately — DOM is synchronously ready right after innerHTML is
    // set. No timeout clock GATING the trial, so no deadline for jsPsych to
    // race against; no rAF/setTimeout deferral needed either (that pattern
    // previously caused an unclickable-button bug in the main observation
    // plugins). Phase E's two demo clocks (above) run independently of this
    // and never gate Submit either -- they're stopped here, on submit,
    // purely for cleanup (see module docstring).
    initSlider(display_el, {
      unset,
      showValue: show_value,
      onFinish: () => {
        if (stopCornerClock) stopCornerClock();
        if (stopKdeClock)    stopKdeClock();
        const response = parseInt(display_el.querySelector('#response-slider').value);
        this.jsPsych.finishTrial({ response, timed_out: false });
      },
    });
  }
}

TutorialObservationNumbersPlugin.info = info;
export default TutorialObservationNumbersPlugin;
