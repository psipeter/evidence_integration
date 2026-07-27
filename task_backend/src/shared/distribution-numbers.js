/**
 * distribution-numbers.js
 * Shared distribution SVG builder for the numbers tutorial.
 *
 * Used by:
 *   plugin-tutorial-intro-numbers.js       (progressive reveal via group opacity)
 *   plugin-tutorial-observation-numbers.js (fully revealed from the start)
 *
 * Mirrors urn-colors.js's pattern for the colors tutorial: one shared builder
 * with a `revealed` flag, instead of two near-duplicate local implementations
 * (which is what existed here before — one plugin had progressive-reveal
 * opacity groups, the other didn't, and they drifted independently).
 *
 * The SVG contains these named groups/elements:
 *   #tut-svg-axis-labels — 0/100 axis line (labels moved to HTML, below)
 *   #tut-svg-dist        — filled distribution curve
 *   #tut-svg-mean        — true-mean tick ONLY (the "???" label moved to
 *                          HTML, below)
 *   #tut-svg-obs         — current-observation tick + value label. Always
 *                          starts at opacity 0 regardless of `revealed` —
 *                          its reveal is owned exclusively by
 *                          numbers-draw-animation.js's resolve step,
 *                          the same way urn-colors.js's bar segments are
 *                          recolored independently of the shared
 *                          `revealed` flag (see that file's own docstring
 *                          -- an earlier revision had a separate outcome
 *                          circle there instead, since removed).
 *   #tut-svg-history     — small, low-opacity ticks (no labels) for every
 *                          PAST observation in this tutorial sequence, at
 *                          the same axis position as #tut-svg-obs but
 *                          visually subordinate to it (shorter, thinner,
 *                          faded) -- the on-curve counterpart to
 *                          tutorial-tracker.js's slot row. Always visible
 *                          immediately (not gated by `revealed` or the draw
 *                          animation) since these are settled facts from
 *                          earlier screens, not something being revealed
 *                          now.
 *   #tut-svg-bubbles     — ephemeral bubble <circle>s (filled by the animation),
 *                          clipped to the plot area
 *
 * TEXT LABELS ARE HTML, NOT SVG (chat history, this session): the "0"/
 * "100" axis numbers and the "???" mean label used to be SVG <text>
 * elements, but preserveAspectRatio="none" (see LAYOUT's own comment --
 * needed so the plot fills its box at any aspect ratio, permanently)
 * stretches EVERYTHING in the SVG's coordinate space non-uniformly,
 * including glyphs -- reported as "the ??? and 0/100 look awkward"
 * (stretched/squashed text). Moving just the text out to plain HTML
 * <span>s, absolutely positioned by PERCENTAGE (not px) over the same
 * wrapping container the <svg> sits in, sidesteps that entirely: the
 * browser's normal text engine draws them, so they're never distorted,
 * while their percentage-based position still tracks the exact SVG-space
 * coordinate it corresponds to (via pctX/pctY below) regardless of the
 * box's actual rendered size. The curve/axis-line/mean-tick stay inside
 * the SVG and keep stretching to fill via "none" -- only the letterforms
 * needed to move.
 * Consequently buildDistributionSVG's return value is no longer a bare
 * `<svg>` -- it's a `position:relative` wrapper DIV containing the <svg>
 * PLUS the label spans as siblings. `#dist-svg svg` (used elsewhere, e.g.
 * numbers-draw-animation.js's svgRoot) still resolves correctly
 * through the extra nesting -- it's a descendant selector, not a direct-
 * child one. The three label spans' own opacity is NOT driven by the
 * `vis`/`revealed` flag directly inside the SVG anymore either -- they're
 * initialized to the SAME value at build time, but external reveal code
 * that used to flip an SVG group's opacity (plugin-tutorial-intro-
 * numbers.js's showDist()/onBox1()) now ALSO needs to flip the
 * matching label span's opacity by id (#tut-svg-axis-label-0/-100,
 * #tut-svg-mean-label) -- see that file's own updated comments.
 *
 * Pass revealed=true to show axis/dist/mean immediately (obs 2–5). #tut-svg-obs
 * still starts hidden either way — see above.
 * Pass revealed=false to hide axis/dist/mean initially (obs 1, progressive reveal).
 */
import { normalPDF } from './draw-performance-numbers.js';

const GOAL_COLOR   = '#2563eb';
const SAMPLE_COLOR = '#ef4444';
const DIST_COLOR   = '#16a34a';

export const LAYOUT = {
  W:    220,
  // H previously tuned to 220 (from an original 160) specifically to make
  // the viewBox's own aspect ratio (square) better MATCH
  // .tutorial-right-image-box's typical shape, since preserveAspectRatio
  // was "meet" at the time (fit-within, letterboxing whatever doesn't
  // match). That box's actual shape varies with viewport (its height is
  // vh-based, its width is vw-based via the column layout) -- no single
  // viewBox aspect ratio can match it at every window size, which is why
  // "lots of whitespace" kept reappearing no matter how this was tuned.
  // preserveAspectRatio is now "none" instead (see buildDistributionSVG
  // below) -- the SVG stretches independently in X/Y to exactly fill the
  // box at ANY aspect ratio, permanently, with no further tuning needed --
  // so H's specific value no longer matters for fill/whitespace at all;
  // it's kept at 220 purely because that's also what made plotH (and so
  // the curve itself) taller/more prominent, which is still wanted
  // independent of the aspect-ratio fix.
  H:    220,
  xMin: 0,
  xMax: 100,
  // pad.t increased from 26 (chat history) for the same reason as pad.b
  // above -- the "???" mean label sits just above the curve at pad.t-12,
  // and at its own doubled font size (2rem) that offset alone left too
  // little headroom above y=0 at the shortest end of .tutorial-right-
  // image-box's clamp() range, risking the label's top half clipping
  // against the box's own top edge (overflow:hidden). Same fix pattern:
  // grow the actual padding constant rather than nudge the offset.
  // pad.b increased from 44 (chat history) to guarantee real clearance
  // below the axis line for the "0"/"100" labels at their doubled font
  // size (2rem) -- a small offset bump alone risked the label's own
  // bottom half clipping against the box's edge (via .tutorial-right-
  // image-box's overflow:hidden) at the SHORTEST end of that box's
  // clamp() range, since half of a larger font's rendered height reaches
  // further down from its anchor point. Growing the actual padding
  // reserves guaranteed room regardless of box height, rather than
  // hoping a fixed offset happens to fit -- the same "adjust the layout
  // constant, not a one-off nudge" approach used elsewhere in this file.
  // Plot area (plotH below) shrinks somewhat as a result of both -- an
  // accepted, minor tradeoff for guaranteed no-clip label room at either
  // end of the box's real, viewport-dependent size range.
  pad:  { l: 20, r: 20, t: 38, b: 56 },
};

/**
 * @param {number} mu
 * @param {number} sigma
 * @param {number} currentValue
 * @param {boolean} [revealed]
 * @param {number[]} [history]  raw values of every PAST observation in this
 *   tutorial sequence (NOT including currentValue) -- e.g. for obs_num=4,
 *   this is tutorialValues[0..2]. Defaults to [] (obs 1/intro has no history
 *   yet, which is also correct there).
 */
export const buildDistributionSVG = (mu, sigma, currentValue, revealed = false, history = []) => {
  const { W, H, xMin, xMax, pad } = LAYOUT;
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;
  const axisY = pad.t + plotH;
  const tickH = 12;
  const historyTickH = 7; // shorter than the current-obs tick (tickH=12), so
                          // it's immediately readable as "lesser" even before
                          // opacity is considered
  const vis   = revealed ? '1' : '0';

  const xPos  = (x) => pad.l + (x - xMin) / (xMax - xMin) * plotW;
  const peakP = normalPDF(mu, mu, sigma);
  const yPos  = (p) => pad.t + plotH - (p / peakP) * plotH * 0.88;

  // Convert SVG-viewBox-space coordinates to CSS percentages of the
  // wrapper's own box -- see module docstring's "TEXT LABELS ARE HTML"
  // note for why. Using percentages (not px) is what keeps these
  // correctly positioned regardless of the box's actual rendered size --
  // the same reason the SVG itself uses width/height:100% + viewBox
  // rather than fixed pixel dimensions.
  const pctX = (x) => `${(x / W * 100).toFixed(2)}%`;
  const pctY = (y) => `${(y / H * 100).toFixed(2)}%`;

  const steps = 200;
  const curvePoints = Array.from({ length: steps + 1 }, (_, i) => {
    const x = xMin + (i / steps) * (xMax - xMin);
    return `${xPos(x).toFixed(2)},${yPos(normalPDF(x, mu, sigma)).toFixed(2)}`;
  }).join(' ');
  const fillPoints = curvePoints
    + ` ${xPos(xMax).toFixed(2)},${axisY} ${xPos(xMin).toFixed(2)},${axisY}`;

  // Shared style for all three HTML label spans -- centered on its own
  // (pctX,pctY) point via translate(-50%,-50%), same visual anchor a
  // text-anchor="middle" SVG <text> gave (the vertical anchor differs
  // slightly -- SVG y was a baseline, this centers instead -- an
  // intentional, invisible-at-this-scale simplification, not an oversight).
  const labelBase = 'position:absolute;transform:translate(-50%,-50%);white-space:nowrap;font-family:Arial;';

  return `<div style="position:relative;width:100%;height:100%;">
    <svg viewBox="0 0 ${W} ${H}" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
      <defs>
        <clipPath id="tut-svg-plot-clip">
          <rect x="${pad.l}" y="${pad.t}" width="${plotW}" height="${plotH}"/>
        </clipPath>
      </defs>
      <g id="tut-svg-axis-labels" style="opacity:${vis};">
        <line x1="${pad.l}" y1="${axisY}" x2="${pad.l + plotW}" y2="${axisY}"
          stroke="#bbb" stroke-width="1"/>
      </g>
      <g id="tut-svg-dist" style="opacity:${vis};">
        <polygon points="${fillPoints}" fill="rgba(22,163,74,0.15)" stroke="none"/>
        <polyline points="${curvePoints}"
          fill="none" stroke="${DIST_COLOR}" stroke-width="2" stroke-linejoin="round"/>
      </g>
      <g id="tut-svg-mean" style="opacity:${vis};">
        <line x1="${xPos(mu).toFixed(2)}" y1="${yPos(peakP).toFixed(2)}"
              x2="${xPos(mu).toFixed(2)}" y2="${axisY}"
              stroke="${GOAL_COLOR}" stroke-width="2"
              stroke-linecap="round"/>
      </g>
      <g id="tut-svg-bubbles" clip-path="url(#tut-svg-plot-clip)" style="opacity:1;"></g>
      <g id="tut-svg-history" style="opacity:1;">
        ${history.map(v => `
        <line x1="${xPos(v).toFixed(2)}" y1="${axisY}"
              x2="${xPos(v).toFixed(2)}" y2="${axisY + historyTickH}"
              stroke="${SAMPLE_COLOR}" stroke-width="2" stroke-linecap="round"
              opacity="0.35"/>`).join('')}
      </g>
      <g id="tut-svg-obs" style="opacity:0;">
        <line x1="${xPos(currentValue).toFixed(2)}" y1="${axisY}"
              x2="${xPos(currentValue).toFixed(2)}" y2="${axisY + tickH}"
              stroke="${SAMPLE_COLOR}" stroke-width="3" stroke-linecap="round"/>
      </g>
    </svg>
    <span id="tut-svg-axis-label-0"
      style="${labelBase}left:${pctX(pad.l)};top:${pctY(axisY + 20)};font-size:1.5rem;color:#999;opacity:${vis};">0</span>
    <span id="tut-svg-axis-label-100"
      style="${labelBase}left:${pctX(pad.l + plotW)};top:${pctY(axisY + 20)};font-size:1.5rem;color:#999;opacity:${vis};">100</span>
    <span id="tut-svg-mean-label"
      style="${labelBase}left:${pctX(xPos(mu))};top:${pctY(pad.t - 12)};font-size:2rem;font-weight:bold;color:${GOAL_COLOR};opacity:${vis};">???</span>
  </div>`;
};
