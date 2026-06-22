/**
 * bar-chart.js
 * SVG for binary trial/practice summary screens.
 *
 * Layout (top → bottom):
 *   Bar: blue|red filled bar with green true_p line and black estimate ticks above
 *   Row 1 (estimate ticks): labelled "estimates" on left, black vertical ticks
 *   Row 2 (obs circles):    labelled "observations" on left, blue/red dots
 *
 * Mirrors the continuous draw-performance.js layout for visual consistency.
 */

const SAMPLE_BLUE = '#2563eb';
const SAMPLE_RED  = '#ef4444';
const DIST_COLOR  = '#16a34a';

export const buildSummaryBarSVG = (true_p, values, responses) => {
  const W      = 520;
  const padL   = 88;   // left padding for row labels — matches draw-performance.js
  const padR   = 88;   // symmetric with padL for centred bar
  const padT   = 14;
  const barW   = W - padL - padR;
  const barH   = 32;
  const barY   = padT;

  // tick rows below the bar
  const dotR       = 7;
  const dotRowY    = barY + barH + 12;   // observation circles row (top)
  const tickH      = 14;
  const tickRowY   = dotRowY + dotR * 2 + 16;  // estimate ticks row (bottom)
  const H          = tickRowY + tickH + 12;

  // helpers
  const xPos = (v) => padL + Math.round((v / 100) * barW);
  const trueX = padL + Math.round(true_p * barW);

  // Bar segments
  const blueW = Math.round(true_p * barW);
  const redW  = barW - blueW;
  const bar = `
    <rect x="${padL}" y="${barY}" width="${blueW}" height="${barH}"
      fill="${SAMPLE_BLUE}" rx="3"/>
    <rect x="${trueX}" y="${barY}" width="${redW}" height="${barH}"
      fill="${SAMPLE_RED}" rx="3"/>
    <rect x="${padL}" y="${barY}" width="${barW}" height="${barH}"
      fill="none" stroke="#222" stroke-width="1.5" rx="3"/>
    <line x1="${trueX}" y1="${barY}" x2="${trueX}" y2="${barY + barH}"
      stroke="${DIST_COLOR}" stroke-width="6" stroke-linecap="round"/>`;

  // Estimate ticks — row below bar
  const estTicks = (responses || []).map(resp => {
    if (resp === null) return '';
    const x = xPos(resp);
    return `<line x1="${x}" y1="${tickRowY}" x2="${x}" y2="${tickRowY + tickH}"
      stroke="#222" stroke-width="2.5" stroke-linecap="round"/>`;
  }).join('');

  // Observation circles — row below estimate ticks
  const dotStep = dotR * 2 + 5;
  const dotCY   = dotRowY + dotR;
  const blueVals = (values || []).filter(v => v === 1);
  const redVals  = (values || []).filter(v => v === -1);

  const blueDots = blueVals.map((_, i) => {
    const cx = padL + i * dotStep + dotR;
    return `<circle cx="${cx}" cy="${dotCY}" r="${dotR}"
      fill="${SAMPLE_BLUE}" stroke="#fff" stroke-width="1.5"/>`;
  }).join('');

  const redDots = redVals.map((_, i) => {
    const cx = padL + barW - i * dotStep - dotR;
    return `<circle cx="${cx}" cy="${dotCY}" r="${dotR}"
      fill="${SAMPLE_RED}" stroke="#fff" stroke-width="1.5"/>`;
  }).join('');

  // Row labels — right-aligned in left padding, vertically centred on each row
  const estLabel = `
    <text x="${padL - 6}" y="${(tickRowY + tickH / 2 + 4).toFixed(1)}"
      font-family="Arial" font-size="11" font-weight="bold"
      fill="#222" text-anchor="end">estimates</text>`;
  const obsLabel = `
    <text x="${padL - 6}" y="${(dotCY + 4).toFixed(1)}"
      font-family="Arial" font-size="11" font-weight="bold"
      fill="#888" text-anchor="end">observations</text>`;

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
    ${bar}
    ${estLabel}${estTicks}
    ${obsLabel}${blueDots}${redDots}
  </svg>`;
};
