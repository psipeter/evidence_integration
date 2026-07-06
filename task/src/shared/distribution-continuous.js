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
 * The SVG contains four named groups:
 *   #tut-svg-axis-labels — 0/100 axis line + labels
 *   #tut-svg-dist        — filled distribution curve
 *   #tut-svg-mean        — true-mean tick + "???" label
 *   #tut-svg-obs         — current-observation tick + value label
 *
 * Pass revealed=true to show all groups immediately (obs 2-5).
 * Pass revealed=false to hide all groups initially (obs 1, progressive reveal).
 */
import { normalPDF } from './draw-performance-continuous.js';

const GOAL_COLOR   = '#2563eb';
const SAMPLE_COLOR = '#ef4444';
const DIST_COLOR   = '#16a34a';

export const buildDistributionSVG = (mu, sigma, currentValue, revealed = false) => {
  const W = 220, H = 160;
  const xMin = 0, xMax = 100;
  const pad  = { l: 20, r: 20, t: 26, b: 44 };
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;
  const axisY = pad.t + plotH;
  const tickH = 12;
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
    <g id="tut-svg-obs" style="opacity:${vis};">
      <line x1="${xPos(currentValue).toFixed(2)}" y1="${axisY}"
            x2="${xPos(currentValue).toFixed(2)}" y2="${axisY + tickH}"
            stroke="${SAMPLE_COLOR}" stroke-width="3" stroke-linecap="round"/>
      <text x="${xPos(currentValue).toFixed(2)}" y="${axisY + tickH + 12}"
            text-anchor="middle" font-family="Arial" font-size="14"
            font-weight="bold" fill="${SAMPLE_COLOR}">${currentValue}</text>
    </g>
  </svg>`;
};
