/**
 * urn-binary.js
 * Shared probability-bar SVG builder for the binary tutorial.
 *
 * Used by:
 *   plugin-tutorial-intro-binary.js       (progressive reveal via group opacity)
 *   plugin-tutorial-observation-binary.js (fully revealed from the start)
 *
 * The grid-of-dots urn metaphor was removed (Jul 2026) — the task uses independent
 * Bernoulli draws, not sampling without replacement. The bar + bubbling animation
 * (binary-draw-animation.js) conveys the generative process instead.
 *
 * SVG groups / elements:
 *   #tut-urn-bar       — blue/red bar (no label). The two rects inside now
 *                        have their OWN ids (#tut-urn-bar-blue/-red, chat
 *                        history) -- binary-draw-animation.js's resolve
 *                        step dims whichever one does NOT match the
 *                        drawn outcome, by fill color, so it needs to
 *                        grab each independently rather than treating
 *                        the group as one unit.
 *   #tut-urn-qmark     — "???" label's opacity GROUP -- kept as an empty <g>
 *                        now (the actual text moved to HTML, see below),
 *                        purely so existing reveal code that toggles this
 *                        group's opacity by id keeps working unchanged.
 *   #tut-urn-bubbles   — ephemeral bubble <circle>s (filled by animation)
 *
 * DRAW OUTCOME IS SHOWN BY DIMMING THE LOSING BAR SEGMENT, NOT A
 * SEPARATE CIRCLE (chat history, this session): #tut-urn-draw -- a circle
 * that used to pop in ABOVE the bar and fade to the outcome color once
 * bubbling stopped -- is REMOVED entirely. Explicit feedback: that circle
 * sat in a spot with no visual connection to the bar's own proportions
 * below it, and the outcome it revealed was ALREADY shown a second and
 * third time (centre panel, tracker) -- redundant on top of disconnected.
 * The bubbles (which DO connect to the bar's proportions -- they spawn
 * proportionally to true_p, inside the bar) stay exactly as before; only
 * the RESOLVE step changed. An earlier attempt at that resolve step also
 * BRIGHTENED the winning segment (an HSL-derived lighter/more-saturated
 * variant) alongside dimming the loser -- dropped per explicit direction
 * ("doesn't look great"): the winning segment now keeps its EXACT
 * original SAMPLE_BLUE/SAMPLE_RED, untouched, and only the losing one
 * dims -- via plain alpha transparency (DIM_BLUE/DIM_RED below), not an
 * HSL-desaturated solid color. That's a deliberate reuse of the SAME
 * dimming technique this app's slider CSS already uses for its own
 * "last" (dimmed) mode (see style.css's input[type="range"].binary-
 * slider.slider-last rule, rgba(37,99,235,0.4)/rgba(239,68,68,0.4)) --
 * consistent with an existing convention rather than introducing a
 * second one. See binary-draw-animation.js's own docstring for the full
 * mechanism.
 *
 * TEXT LABEL IS HTML, NOT SVG (chat history): the "???" label used to be
 * an SVG <text> element, but preserveAspectRatio="none" (see LAYOUT's own
 * comment -- needed so the bar fills its box at any aspect ratio,
 * permanently) stretches EVERYTHING in the SVG's coordinate space non-
 * uniformly, including glyphs -- reported as "the ??? looks awkward"
 * (stretched/squashed text), mirrors distribution-continuous.js's
 * identical fix/rationale for its own "0"/"100"/"???" labels. Moving the
 * text out to a plain HTML <span>, absolutely positioned by PERCENTAGE
 * (not px) over the same wrapping container the <svg> sits in, sidesteps
 * that entirely -- the browser's normal text engine draws it, never
 * distorted, while its percentage-based position still tracks the exact
 * SVG-space coordinate it corresponds to regardless of the box's actual
 * rendered size. The label moved from BELOW the bar to ABOVE it, in the
 * vertical space #tut-urn-draw used to occupy (chat history, this
 * session) -- removing that circle freed real room up there, and putting
 * the label in it (rather than leaving it as unused margin) is what let
 * LAYOUT shrink overall (see LAYOUT's own comment) rather than just
 * leaving a blank gap where the circle used to be.
 * Consequently buildUrnSVG's return value is no longer a bare `<svg>` --
 * it's a `position:relative` wrapper DIV containing the <svg> PLUS the
 * label span as a sibling. `#urn-svg svg` (used elsewhere, e.g.
 * binary-draw-animation.js's svgRoot) still resolves correctly through
 * the extra nesting -- it's a descendant selector, not a direct-child
 * one. The label span's own opacity is NOT driven by the `vis`/`revealed`
 * flag inside the SVG's #tut-urn-qmark group anymore either -- it's
 * initialized to the SAME value at build time, but external reveal code
 * that used to flip that SVG group's opacity (plugin-tutorial-intro-
 * binary.js's onBox1()) now ALSO needs to flip the label span's opacity
 * by id (#tut-urn-qmark-label) -- see that file's own updated comments.
 *
 * Pass revealed=true to show the bar (and "???") immediately (obs 2–5).
 * Pass revealed=false to hide both initially (obs 1 intro, progressive reveal —
 * the intro plugin reveals the bar and "???" group independently).
 */

export const SAMPLE_BLUE = '#2563eb';
export const SAMPLE_RED  = '#ef4444';
export const DIST_COLOR  = '#16a34a';

// Dimmed variants of SAMPLE_BLUE/SAMPLE_RED (chat history, this session)
// -- used by binary-draw-animation.js's resolve step to dim whichever bar
// segment does NOT match the drawn outcome (the other keeps its exact
// original color, untouched -- see module docstring for why a brightened
// variant was tried and dropped). Plain rgba alpha, not an HSL-desaturated
// solid color -- these are the SAME rgb triples as SAMPLE_BLUE/RED
// (37,99,235 / 239,68,68) at the SAME 0.4 alpha already used for this
// exact purpose elsewhere in this app (style.css's input[type="range"]
// .binary-slider.slider-last track rule) -- reusing that existing
// "dimmed = same color, real transparency" convention rather than
// introducing a second, different-looking one.
export const DIM_BLUE = 'rgba(37, 99, 235, 0.4)';
export const DIM_RED  = 'rgba(239, 68, 68, 0.4)';

export const LAYOUT = {
  W:       194,
  // barY shrunk from 60 (chat history, this session): that margin used to
  // reserve room for #tut-urn-draw (a circle floating above the bar,
  // since removed -- see module docstring) plus a gap below it before the
  // bar itself. The "???" label now lives in roughly that same space
  // instead (see qY below), but a label needs far less dedicated room
  // than a circle-shaped element + its own clearance gap did, so barY
  // shrinks to reflect that smaller real requirement rather than leaving
  // the old margin as now-unused blank space.
  barY:    40,
  barH:    44,
  // H shrunk from 150 (chat history, this session) to match barY's own
  // reduction above, plus a smaller bottom margin (the bottom-of-bar
  // label that used to live below the bar is gone too -- moved above,
  // see module docstring) -- both freed spaces are removed rather than
  // left as unused padding, giving the bar itself a proportionally
  // larger share of the box once preserveAspectRatio="none" stretches
  // this viewBox to fill .tutorial-right-image-box (see that rule's own
  // comment in style.css). drawR/drawGap (the removed circle's own size/
  // gap constants) are gone from this object entirely -- not just unused,
  // structurally tied to a feature that no longer exists, unlike other
  // "kept for later" values elsewhere in this file.
  H:       104,
};

export const buildUrnSVG = (p, revealed = false) => {
  const { W, barY, barH, H } = LAYOUT;
  const midX = Math.round(p * W);
  const vis  = revealed ? '1' : '0';
  // "???" label sits above the bar (chat history, this session) -- gap
  // increased from an initial "-12" to "-22" per explicit direction ("a
  // little farther above the bar") -- plenty of clearance margin above
  // y=0 either way (checked against the shortest end of .tutorial-
  // right-image-box's clamp() range when this was first placed here, see
  // git history), so there's no clipping risk from moving it further.
  const qY   = barY - 22;

  // Convert SVG-viewBox-space coordinates to CSS percentages of the
  // wrapper's own box -- see module docstring's "TEXT LABEL IS HTML"
  // note for why, mirrors distribution-continuous.js's identical helper.
  const pctX = (x) => `${(x / W * 100).toFixed(2)}%`;
  const pctY = (y) => `${(y / H * 100).toFixed(2)}%`;

  return `<div style="position:relative;width:100%;height:100%;">
    <svg viewBox="0 0 ${W} ${H}" width="100%" height="100%"
      xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
      <defs>
        <clipPath id="tut-urn-bar-clip">
          <rect x="0" y="${barY}" width="${W}" height="${barH}" rx="3"/>
        </clipPath>
      </defs>
      <g id="tut-urn-bar" style="opacity:${vis};">
        <rect id="tut-urn-bar-blue" x="0" y="${barY}" width="${midX}" height="${barH}"
          fill="${SAMPLE_BLUE}" rx="3"/>
        <rect id="tut-urn-bar-red" x="${midX}" y="${barY}" width="${W - midX}" height="${barH}"
          fill="${SAMPLE_RED}" rx="3"/>
        <line x1="${midX}" y1="${barY}" x2="${midX}" y2="${barY + barH}"
          stroke="${DIST_COLOR}" stroke-width="5" stroke-linecap="round"/>
      </g>
      <g id="tut-urn-qmark" style="opacity:${vis};"></g>
      <g id="tut-urn-bubbles" clip-path="url(#tut-urn-bar-clip)"
         style="opacity:${vis};"></g>
    </svg>
    <span id="tut-urn-qmark-label"
      style="position:absolute;transform:translate(-50%,-50%);white-space:nowrap;
             font-family:Arial;font-size:2rem;font-weight:bold;color:${DIST_COLOR};
             left:${pctX(midX)};top:${pctY(qY)};opacity:${vis};">???</span>
  </div>`;
};
