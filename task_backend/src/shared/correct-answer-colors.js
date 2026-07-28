/**
 * correct-answer-colors.js
 * Replaces urn-colors.js's buildUrnSVG (tutorial usage only -- urn-colors.js
 * itself stays, its color constants are still used elsewhere) +
 * colors-draw-animation.js entirely (colors-draw-animation.js deleted this
 * session -- still under task/ if this ever needs reverting). The old bar
 * + bubbling-then-reveal animation taught the FIXED true_p (a value the
 * real task never scores against) with an artificial ~1s wait before
 * showing anything -- neither property was correct once the real task's
 * scored quantity was confirmed to be the RUNNING proportion of blue
 * draws (config-base.js's colors override, ERROR_MODE: 'running_p'), not
 * the fixed true_p. Mirrors correct-answer-numbers.js's identical fix for
 * the numbers task, adapted to colors' categorical (not continuous) data.
 *
 * Design: a thin blue/red bar, split at the RUNNING proportion of blue
 * draws so far (blue on the left, red on the right -- SLIDES smoothly to
 * its new split point each time a new observation arrives, mirroring the
 * numbers thumb's slide), with small dots accumulating ABOVE the bar: one
 * dot per draw, blue dots packed in from the LEFT edge, red dots packed
 * in from the RIGHT edge -- so their counts (not a continuous position,
 * unlike numbers' ticks) are what's visualized. No bubbling, no
 * artificial delay of any kind.
 *
 * jsPsych creates a FRESH DOM for every observation -- there's no
 * persistent element across calls to detect "was this already visible"
 * from, so the bar's slide START point is computed explicitly from
 * `history` (the running proportion of everything BEFORE this
 * observation), not inferred from any DOM state -- same reasoning as
 * correct-answer-numbers.js's own docstring.
 *
 * SHOW_EXACT_VALUE: whether the running blue-proportion's numeric value
 * is shown as text, or left as a visual-only bar split. Same flag/
 * rationale as correct-answer-numbers.js's identical constant.
 */
import { SAMPLE_BLUE, SAMPLE_RED } from './urn-colors.js';

export const SHOW_EXACT_VALUE = false;
export const FADE_MS = 1000;
export const MOVE_MS = 600;

// Max observations per trial (config-base.js's shared N_OBS) -- used to
// pick dot spacing that guarantees blue dots (packed in from the left)
// and red dots (packed in from the right) can never collide, even in
// the 15-observations-of-one-color extreme: spacing of 100/(N_OBS+1)
// leaves the LAST possible dot on either side short of the far edge, so
// the two sequences only ever meet, never overlap, regardless of the
// actual blue/red split. Duplicated here (not imported from config-
// base.js) for the same reason every other literal in this file's
// sibling modules is a plain constant, not threaded through config --
// see e.g. correct-answer-numbers.js's own t_obs_ms default note.
const N_OBS = 15;

function countBlueRed(values) {
  let blue = 0, red = 0;
  for (const v of values) { if (v === 1) blue++; else red++; }
  return { blue, red };
}

/** Returns the inner HTML for the panel -- a dot row above a blue/red
 * split bar. Starts fully transparent (opacity:0 on the bar);
 * renderCorrectAnswerColors() below reveals it the first time it's
 * called. */
export function buildCorrectAnswerColorsHTML() {
  return `
    <div class="correct-answer-colors-outer">
      <div id="tut-cac-dots" class="correct-answer-colors-dots"></div>
      <div id="tut-cac-bar" class="correct-answer-colors-bar" style="opacity:0;">
        <div id="tut-cac-bar-blue" class="correct-answer-colors-bar-blue" style="width:50%;"></div>
        <div id="tut-cac-bar-red" class="correct-answer-colors-bar-red" style="width:50%;"></div>
      </div>
      <span id="tut-cac-value" class="correct-answer-colors-value-label" style="opacity:0;"></span>
    </div>`;
}

/**
 * Renders one new dot for currentValue (packed in after however many
 * same-color dots already exist from `history`) plus every prior dot,
 * and SLIDES the bar from the running blue-proportion of `history` alone
 * to that of history+[currentValue]. If there's no history at all (the
 * very first observation), there's nothing to slide from -- the bar
 * just appears at that single draw's own all-or-nothing split (100%
 * blue or 100% red).
 *
 * @param {HTMLElement} display_el
 * @param {object} opts
 * @param {number[]} opts.history       +1/-1 values of every PAST observation
 * @param {number} opts.currentValue    this observation's own +1/-1 value
 * @param {boolean} [opts.showValue]    defaults to the module's SHOW_EXACT_VALUE flag
 * @param {boolean} [opts.fadeIn]       true for a one-time opacity fade-in
 *   (the intro plugin's one reveal moment); false (default) for the
 *   observation plugin, which is already visible on every call.
 */
export function renderCorrectAnswerColors(display_el, { history, currentValue, showValue = SHOW_EXACT_VALUE, fadeIn = false }) {
  const dotsLayer  = display_el.querySelector('#tut-cac-dots');
  const bar        = display_el.querySelector('#tut-cac-bar');
  const barBlue    = display_el.querySelector('#tut-cac-bar-blue');
  const barRed     = display_el.querySelector('#tut-cac-bar-red');
  const valueLabel = display_el.querySelector('#tut-cac-value');
  if (!dotsLayer || !bar || !barBlue || !barRed || !valueLabel) return;

  const allValues = [...history, currentValue];
  const { blue: nBlue, red: nRed } = countBlueRed(allValues);
  const spacing = 100 / (N_OBS + 1);

  const dots = [];
  for (let i = 1; i <= nBlue; i++) {
    const isCurrent = currentValue === 1 && i === nBlue;
    dots.push(`<div class="correct-answer-colors-dot${isCurrent ? ' correct-answer-colors-dot-current' : ' correct-answer-colors-dot-history'}" style="left:${(i * spacing).toFixed(2)}%;background:${SAMPLE_BLUE};"></div>`);
  }
  for (let j = 1; j <= nRed; j++) {
    const isCurrent = currentValue === -1 && j === nRed;
    dots.push(`<div class="correct-answer-colors-dot${isCurrent ? ' correct-answer-colors-dot-current' : ' correct-answer-colors-dot-history'}" style="left:${(100 - j * spacing).toFixed(2)}%;background:${SAMPLE_RED};"></div>`);
  }
  dotsLayer.innerHTML = dots.join('');

  const newBluePct = (nBlue / allValues.length) * 100;
  const { blue: prevBlue } = countBlueRed(history);
  const prevBluePct = history.length ? (prevBlue / history.length) * 100 : newBluePct; // nothing to slide from on the very first observation

  bar.style.transition = 'none';
  barBlue.style.width = `${prevBluePct}%`;
  barRed.style.width = `${100 - prevBluePct}%`;
  if (!fadeIn) bar.style.opacity = '1';

  if (showValue) {
    valueLabel.style.transition = 'none';
    valueLabel.textContent = `${prevBluePct.toFixed(0)}% blue`;
    if (!fadeIn) valueLabel.style.opacity = '1';
  } else {
    valueLabel.style.opacity = '0';
  }

  requestAnimationFrame(() => {
    const widthTransition = fadeIn
      ? `width ${MOVE_MS}ms ease, opacity ${FADE_MS}ms ease`
      : `width ${MOVE_MS}ms ease`;
    barBlue.style.transition = widthTransition;
    barRed.style.transition  = `width ${MOVE_MS}ms ease`;
    barBlue.style.width = `${newBluePct}%`;
    barRed.style.width  = `${100 - newBluePct}%`;
    bar.style.opacity = '1';

    if (showValue) {
      valueLabel.style.transition = fadeIn ? `opacity ${FADE_MS}ms ease` : 'none';
      valueLabel.style.opacity = '1';
      valueLabel.textContent = `${newBluePct.toFixed(0)}% blue`;
    }
  });
}
