/**
 * tutorial-tracker.js
 * Row of N slots showing progress through the tutorial's illustrative
 * 15-observation sequence -- rendered as its own DOM row directly between
 * the distribution SVG and its caption (see plugin-tutorial-intro-
 * continuous.js / plugin-tutorial-observation-continuous.js), NOT inside
 * distribution-continuous.js's own SVG coordinate system, since this is a
 * separate flex row, not part of that plot.
 *
 * Added specifically to address a real pattern flagged from real pilot
 * data (chat history, chat #13): participants copying the raw stimulus
 * value as their response rather than tracking an accumulating estimate --
 * i.e. treating each screen as an independent "what number is this" task
 * rather than "one more piece of evidence in a sequence you're supposed to
 * be integrating." Existing tutorial text already says "based on all the
 * numbers you've seen" and "your evolving estimate", but nothing on screen
 * showed that accumulation happening. This does.
 *
 * Slot rendering (chat history -- REPLACED an earlier colored-circle
 * design; briefly tried a small black dot below every slot this session
 * to mark position, then REMOVED again per explicit direction -- "they
 * don't add anything useful" -- back to plain number/circle with no
 * marker at all beneath it): each slot is just its number (or, in
 * dot-mode, a colored circle) -- nothing else. Text/circle color is
 * ALWAYS `color` (red, by default) -- opacity is what distinguishes the
 * three states, not a color change:
 *   1..obsNum-1  (SETTLED) -- faded ${color} text/circle.
 *   obsNum       (CURRENT) -- full-opacity, bold ${color} text/circle,
 *                larger than settled -- echoes the SAME value shown big
 *                in the centre-panel stimulus. The number/circle itself
 *                starts hidden (`revealCurrent: false`) when the caller
 *                wants it to appear in sync with the bubbling-draw
 *                animation's own reveal rather than instantly on render --
 *                see plugin-tutorial-intro-continuous.js's onImageBox,
 *                which flips #tut-tracker-current-num's opacity from
 *                inside that animation's onReveal callback (fired the
 *                INSTANT the centre number begins its fade, not once it's
 *                already finished -- see continuous-draw-animation.js's
 *                own docstring for why onComplete was the wrong hook
 *                here).
 *   obsNum+1..nObs (EMPTY) -- invisible placeholder (transparent text/
 *                circle, same footprint) -- so the remaining count is
 *                still visible at a glance via the settled/current slots'
 *                own presence, and the row's height stays consistent
 *                without a visible marker for slots not yet reached.
 *
 * `color` is a parameter (not hardcoded) so this can be reused for binary's
 * blue/red draws -- continuous passes a fixed color string; binary
 * (chat history) passes a FUNCTION `(value) => colorString` instead
 * (values are +-1, not numbers to display), and sets `renderDot: true` to
 * switch every slot from number to a colored circle -- there's no
 * separate "number" to show for a binary draw, the color IS the content,
 * so text would be redundant here in a way it wasn't for continuous's
 * numeric values. Both modes share the exact same settled/current/empty
 * state logic and the same #tut-tracker-current-num id for the
 * reveal-timing wiring (continuous-draw-animation.js's onReveal /
 * binary-draw-animation.js's own equivalent).
 */

export function buildTrackerHTML({ nObs, obsNum, values, color = '#ef4444', revealCurrent = true, renderDot = false }) {
  const slots = [];
  const colorFor = (v) => typeof color === 'function' ? color(v) : color;

  for (let i = 0; i < nObs; i++) {
    const idx = i + 1; // 1-indexed slot position

    if (renderDot) {
      if (idx < obsNum) {
        slots.push(`
          <div class="tut-tracker-slot">
            <span class="tut-tracker-dot tut-tracker-dot-settled" style="background:${colorFor(values[i])};"></span>
          </div>`);
      } else if (idx === obsNum) {
        slots.push(`
          <div class="tut-tracker-slot">
            <span class="tut-tracker-dot tut-tracker-dot-current" id="tut-tracker-current-num"
                  style="background:${colorFor(values[i])};opacity:${revealCurrent ? 1 : 0};"></span>
          </div>`);
      } else {
        slots.push(`
          <div class="tut-tracker-slot">
            <span class="tut-tracker-dot tut-tracker-dot-empty"></span>
          </div>`);
      }
      continue;
    }

    if (idx < obsNum) {
      slots.push(`
        <div class="tut-tracker-slot">
          <span class="tut-tracker-num tut-tracker-num-settled" style="color:${colorFor(values[i])};">${values[i]}</span>
        </div>`);
    } else if (idx === obsNum) {
      slots.push(`
        <div class="tut-tracker-slot">
          <span class="tut-tracker-num tut-tracker-num-current" id="tut-tracker-current-num"
                style="color:${colorFor(values[i])};opacity:${revealCurrent ? 1 : 0};">${values[i]}</span>
        </div>`);
    } else {
      slots.push(`
        <div class="tut-tracker-slot">
          <span class="tut-tracker-num tut-tracker-num-empty">&nbsp;</span>
        </div>`);
    }
  }

  return `<div class="tut-tracker-row" id="tut-tracker-row">${slots.join('')}</div>`;
}
