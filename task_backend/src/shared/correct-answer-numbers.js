/**
 * correct-answer-numbers.js
 * Replaces distribution-numbers.js + numbers-draw-animation.js entirely
 * (both deleted this session -- still present under task/ if this ever
 * needs reverting). The old KDE curve + bubbling-then-reveal animation
 * taught the FIXED population mean (a value the real task never scores
 * against) with an artificial ~1s wait before showing anything -- neither
 * property was actually correct once the real task's scored quantity was
 * confirmed to be the RUNNING mean of observed values (config-base.js's
 * ERROR_MODE), not the fixed mean.
 *
 * Design: a small, non-interactive slider-style track. A tick per past
 * observation (faded) plus one for the current observation (bold), plus
 * two end-ticks at 0/100 with compact labels directly beneath them (not
 * a separate floating label row below the bar -- that cost real vertical
 * height the box didn't need to spend). A THUMB SLIDES (not pops, not
 * waits) to the running mean's position every time a new observation
 * arrives -- directly visualizing "this is what should move, and by how
 * much" rather than a static target. No bubbling, no artificial delay.
 *
 * jsPsych creates a FRESH DOM for every observation (a new trial() call
 * per plugin instance) -- there's no persistent thumb element across
 * calls to detect "was this already visible" from, so the slide's
 * START position is computed explicitly from `history` (the running mean
 * of everything BEFORE this observation), not inferred from any DOM
 * state.
 *
 * No separate "reveal the empty track first" step anymore (an earlier
 * version of this file had one, via a now-removed revealTrack() export,
 * used by the intro plugin's OLD two-click progression -- click the
 * image box to reveal the framework, click box 1 to reveal the answer
 * itself). The intro plugin's click progression was redesigned this
 * session to drop that separate image-box click entirely; the whole
 * panel (track + ticks + thumb) now appears in one step, so
 * renderCorrectAnswer() below reveals its own container directly -- a
 * single call is fully self-sufficient for both plugins now.
 *
 * SHOW_EXACT_VALUE: whether the running mean's numeric value is shown
 * next to the thumb, or left as a visual-only position. Deliberately a
 * flag, not a final decision -- kept here as ONE place to flip, pending a
 * later decision on which is better for production (chat history).
 * Currently false.
 */
export const SHOW_EXACT_VALUE = false;

// Reveal fade -- matches the centre-number/tracker-number fade duration
// elsewhere in the tutorial. Only actually used for the intro plugin's
// ONE-TIME reveal (fadeIn:true below) -- the observation plugin's calls
// are already-visible-every-time, no fade needed there, just the slide.
export const FADE_MS = 1000;
// Thumb slide-to-new-position duration -- the "watch it move" cue itself.
export const MOVE_MS = 600;

function meanOf(values) {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/** Returns the inner HTML for the panel -- a 0-100 track with end-ticks +
 * compact labels at 0/100, plus room for observation ticks/thumb/value
 * label. Starts fully transparent (opacity:0 on the inner track);
 * renderCorrectAnswer() below reveals it the first time it's called. */
export function buildCorrectAnswerHTML() {
  return `
    <div class="correct-answer-outer">
      <div id="tut-ca-inner" class="correct-answer-inner">
        <div class="correct-answer-line"></div>
        <div class="correct-answer-endtick" style="left:0%;"></div>
        <div class="correct-answer-endtick" style="left:100%;"></div>
        <span class="correct-answer-endlabel" style="left:0%;">0</span>
        <span class="correct-answer-endlabel" style="left:100%;">100</span>
        <div id="tut-ca-ticks"></div>
        <div id="tut-ca-thumb" class="correct-answer-thumb" style="opacity:0;"></div>
        <span id="tut-ca-value" class="correct-answer-value-label" style="opacity:0;"></span>
      </div>
    </div>`;
}

/**
 * Renders every history tick (faded) + one new bold tick for
 * currentValue, and SLIDES the thumb from the running mean of `history`
 * alone (its position just before this observation) to the running mean
 * of history+[currentValue] (its position now). If there's no history
 * at all (this is the very first observation), there's nothing to slide
 * from -- the thumb just appears at that single value's position.
 * Reveals its own container (#tut-ca-inner) unconditionally -- see
 * module docstring for why this makes the function fully self-sufficient
 * now, for both plugins.
 *
 * @param {HTMLElement} display_el
 * @param {object} opts
 * @param {number[]} opts.history       raw values of every PAST observation
 * @param {number} opts.currentValue    this observation's own raw value
 * @param {boolean} [opts.showValue]    defaults to the module's SHOW_EXACT_VALUE flag
 * @param {boolean} [opts.fadeIn]       true for a one-time opacity fade-in
 *   (the intro plugin's one reveal moment) in ADDITION to the slide;
 *   false (default) for the observation plugin, which is already visible
 *   on every call -- no repeated fade-flash across 14 observations.
 */
export function renderCorrectAnswer(display_el, { history, currentValue, showValue = SHOW_EXACT_VALUE, fadeIn = false }) {
  const container  = display_el.querySelector('#tut-ca-inner');
  const ticksLayer = display_el.querySelector('#tut-ca-ticks');
  const thumb      = display_el.querySelector('#tut-ca-thumb');
  const valueLabel = display_el.querySelector('#tut-ca-value');
  if (!container || !ticksLayer || !thumb || !valueLabel) return;

  container.style.opacity = '1';

  ticksLayer.innerHTML = history.map((v) => `
    <div class="correct-answer-tick correct-answer-tick-history" style="left:${v}%;"></div>
  `).join('') + `
    <div class="correct-answer-tick correct-answer-tick-current" style="left:${currentValue}%;"></div>`;

  const newMean  = meanOf([...history, currentValue]);
  const prevMean = history.length ? meanOf(history) : newMean; // nothing to slide from on the very first observation

  // Place the thumb/value label at the PREVIOUS position first, with no
  // transition -- then, next frame, turn transitions on and move to the
  // NEW position. This two-step (place, then animate) is what makes the
  // move visible; setting the final position directly would just show
  // it already there.
  thumb.style.transition = 'none';
  thumb.style.left = `${prevMean}%`;
  if (!fadeIn) thumb.style.opacity = '1';

  if (showValue) {
    valueLabel.style.transition = 'none';
    valueLabel.style.left = `${prevMean}%`;
    valueLabel.textContent = prevMean.toFixed(1);
    if (!fadeIn) valueLabel.style.opacity = '1';
  } else {
    valueLabel.style.opacity = '0';
  }

  requestAnimationFrame(() => {
    thumb.style.transition = fadeIn
      ? `left ${MOVE_MS}ms ease, opacity ${FADE_MS}ms ease`
      : `left ${MOVE_MS}ms ease`;
    thumb.style.left = `${newMean}%`;
    thumb.style.opacity = '1';

    if (showValue) {
      valueLabel.style.transition = fadeIn
        ? `left ${MOVE_MS}ms ease, opacity ${FADE_MS}ms ease`
        : `left ${MOVE_MS}ms ease`;
      valueLabel.style.left = `${newMean}%`;
      valueLabel.style.opacity = '1';
      valueLabel.textContent = newMean.toFixed(1);
    }
  });
}
