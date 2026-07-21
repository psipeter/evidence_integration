/**
 * distribution-continuous.js
 * Shared distribution SVG builder for the continuous tutorial.
 *
 * Used by:
 *   plugin-tutorial-intro-continuous.js       (progressive reveal via group opacity)
 *   plugin-tutorial-observation-continuous.js (fully revealed from the start)
 *
 * Mirrors urn-binary.js's pattern for the binary tutorial: one shared builder
 * with a `revealed` flag, instead of two near-duplicate local implementations
 * (which is what existed here before — one plugin had progressive-reveal
 * opacity groups, the other didn't, and they drifted independently).
 *
 * The SVG contains these named groups/elements:
 *   #tut-svg-axis-labels — 0/100 axis line + labels
 *   #tut-svg-dist        — filled distribution curve
 *   #tut-svg-mean        — true-mean tick + "???" label
 *   #tut-svg-obs         — current-observation tick + value label. Always
 *                          starts at opacity 0 regardless of `revealed` —
 *                          its reveal is owned exclusively by
 *                          continuous-draw-animation.js's resolve step, the
 *                          same way urn-binary.js's #tut-urn-draw circle is
 *                          managed independently of the shared `revealed` flag.
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
 * Pass revealed=true to show axis/dist/mean immediately (obs 2–5). #tut-svg-obs
 * still starts hidden either way — see above.
 * Pass revealed=false to hide axis/dist/mean initially (obs 1, progressive reveal).
 */
import { normalPDF } from './draw-performance-continuous.js';

const GOAL_COLOR   = '#2563eb';
const SAMPLE_COLOR = '#ef4444';
const DIST_COLOR   = '#16a34a';

export const LAYOUT = {
  W:    220,
  H:    160,
  xMin: 0,
  xMax: 100,
  pad:  { l: 20, r: 20, t: 26, b: 44 },
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

  const steps = 200;
  const curvePoints = Array.from({ length: steps + 1 }, (_, i) => {
    const x = xMin + (i / steps) * (xMax - xMin);
    return `${xPos(x).toFixed(2)},${yPos(normalPDF(x, mu, sigma)).toFixed(2)}`;
  }).join(' ');
  const fillPoints = curvePoints
    + ` ${xPos(xMax).toFixed(2)},${axisY} ${xPos(xMin).toFixed(2)},${axisY}`;

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
    <defs>
      <clipPath id="tut-svg-plot-clip">
        <rect x="${pad.l}" y="${pad.t}" width="${plotW}" height="${plotH}"/>
      </clipPath>
    </defs>
    <g id="tut-svg-axis-labels" style="opacity:${vis};">
      <line x1="${pad.l}" y1="${axisY}" x2="${pad.l + plotW}" y2="${axisY}"
        stroke="#bbb" stroke-width="1"/>
      <text x="${pad.l}" y="${axisY + 14}" text-anchor="middle"
        font-family="Arial" font-size="13" fill="#999">0</text>
      <text x="${pad.l + plotW}" y="${axisY + 14}" text-anchor="middle"
        font-family="Arial" font-size="13" fill="#999">100</text>
    </g>
    <g id="tut-svg-dist" style="opacity:${vis};">
      <polygon points="${fillPoints}" fill="rgba(22,163,74,0.15)" stroke="none"/>
      <polyline points="${curvePoints}"
        fill="none" stroke="${DIST_COLOR}" stroke-width="2" stroke-linejoin="round"/>
    </g>
    <g id="tut-svg-mean" style="opacity:${vis};">
      <line x1="${xPos(mu).toFixed(2)}" y1="${pad.t}"
            x2="${xPos(mu).toFixed(2)}" y2="${axisY}"
            stroke="${GOAL_COLOR}" stroke-width="2"
            stroke-linecap="round"/>
      <text x="${xPos(mu).toFixed(2)}" y="${pad.t - 12}"
            text-anchor="middle" font-family="Arial" font-size="14"
            font-weight="bold" fill="${GOAL_COLOR}">???</text>
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
  </svg>`;
};
