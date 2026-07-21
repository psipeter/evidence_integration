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
 * design): each slot is just its number, colored, sitting on a black
 * underline ("underscore") -- no circle/background fill at all. Dropping
 * the circle freed up room for bigger, more legible numbers and removed a
 * layer of color that wasn't actually carrying information (the circle's
 * own fill color duplicated what the text color already said). Text color
 * is ALWAYS `color` (red, by default) -- opacity and underline weight are
 * what distinguish the three states, not a color change:
 *   1..obsNum-1  (SETTLED) -- faded ${color} text, thin underline.
 *   obsNum       (CURRENT) -- full-opacity, bold ${color} text, on a
 *                THICKER underline -- echoes the SAME value shown big in
 *                the centre-panel stimulus number. The number itself
 *                starts hidden (`revealCurrent: false`) when the caller
 *                wants it to appear in sync with the bubbling-draw
 *                animation's own reveal rather than instantly on render --
 *                see plugin-tutorial-intro-continuous.js's onImageBox,
 *                which flips #tut-tracker-current-num's opacity from
 *                inside that animation's onReveal callback (fired the
 *                INSTANT the centre number begins its fade, not once it's
 *                already finished -- see continuous-draw-animation.js's
 *                own docstring for why onComplete was the wrong hook
 *                here). The underline itself is NOT hidden/revealed --
 *                only the number text fades in; the underline is just
 *                framing, not "the answer".
 *   obsNum+1..nObs (EMPTY) -- a bare underline, no text -- so the
 *                remaining count is always visible at a glance without a
 *                separate textual counter. (The earlier design's separate
 *                "pointer dot" marking the current slot was dropped here
 *                too -- the bolded underline already does that job; having
 *                both was redundant.)
 *
 * `color` is a parameter (not hardcoded) so this can be reused for binary's
 * blue/red draws later without a rewrite -- continuous is the only caller
 * for now.
 */

export function buildTrackerHTML({ nObs, obsNum, values, color = '#ef4444', revealCurrent = true }) {
  const slots = [];

  for (let i = 0; i < nObs; i++) {
    const idx = i + 1; // 1-indexed slot position

    if (idx < obsNum) {
      slots.push(`
        <div class="tut-tracker-slot">
          <span class="tut-tracker-num tut-tracker-num-settled" style="color:${color};">${values[i]}</span>
        </div>`);
    } else if (idx === obsNum) {
      slots.push(`
        <div class="tut-tracker-slot">
          <span class="tut-tracker-num tut-tracker-num-current" id="tut-tracker-current-num"
                style="color:${color};opacity:${revealCurrent ? 1 : 0};">${values[i]}</span>
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
