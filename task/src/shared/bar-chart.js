/**
 * bar-chart.js
 * SVG for binary trial/practice summary screens.
 *
 * Layout (top → bottom):
 *   Bar: blue|red filled bar with green true_p line
 *   "observations" row: blue/red dots (sequential, no overlap by design)
 *   "estimates" row:    black dot plot stacked upward (handles overlap)
 */

const SAMPLE_BLUE = '#2563eb';
const SAMPLE_RED  = '#ef4444';
const DIST_COLOR  = '#16a34a';

// Stack dots along a horizontal axis, stacking upward on overlap.
const stackDots = (values, xPosFn, r) => {
  const placed = [];
  return values.map(v => {
    const x = xPosFn(v);
    let row = 0;
    while (placed.some(p => p.row === row && Math.abs(p.x - x) < r * 2)) row++;
    placed.push({ x, row });
    return { x, row };
  });
};

export const buildBarOnly = (W = 344, barH = 20) => {
  const midX = Math.round(W / 2);
  return `<svg viewBox="0 0 ${W} ${barH}" width="100%" height="auto"
      xmlns="http://www.w3.org/2000/svg">
    <rect x="0" y="0" width="${midX}" height="${barH}"
      fill="${SAMPLE_BLUE}" rx="3"/>
    <rect x="${midX}" y="0" width="${W - midX}" height="${barH}"
      fill="${SAMPLE_RED}" rx="3"/>
    <rect x="0" y="0" width="${W}" height="${barH}"
      fill="none" stroke="#222" stroke-width="1.5" rx="3"/>
    <line x1="${midX}" y1="0" x2="${midX}" y2="${barH}"
      stroke="${DIST_COLOR}" stroke-width="5" stroke-linecap="round"/>
  </svg>`;
};

export const buildSummaryBarSVG = (true_p, values, responses) => {
  const W    = 520;
  const padL = 88;
  const padR = 88;
  const padT = 14;
  const barW = W - padL - padR;
  const barH = 32;
  const barY = padT;
  const trueX = padL + Math.round(true_p * barW);

  const r    = 3;
  const gap  = 1;
  const step = r * 2 + gap;

  const xPos = (v) => padL + Math.round((v / 100) * barW);

  // Observation circles: sequential (blue left→right, red right→left)
  const dotR    = r;
  const dotGap  = gap;
  const dotStep = dotR * 2 + dotGap;
  const obsGap  = 12;
  const obsBaseY = barY + barH + obsGap;
  const dotCY    = obsBaseY + dotR;

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

  // Estimate dot plot — stacked on continuous axis
  const validResps = (responses || []).filter(r => r !== null);
  const estStacked = stackDots(validResps, xPos, r);
  const estMaxRow  = estStacked.reduce((m, d) => Math.max(m, d.row), 0);
  const estGap     = 12;
  const estBaseY   = dotCY + dotR + estGap;
  const estDots    = estStacked.map(({ x, row }) => {
    const cy = estBaseY + r + row * step;
    return `<circle cx="${x}" cy="${cy.toFixed(1)}" r="${r}"
      fill="#222" stroke="#fff" stroke-width="1.2" opacity="0.85"/>`;
  }).join('');
  const estMaxH = (estMaxRow + 1) * step + r;

  const H = estBaseY + estMaxH + 14;

  // Row labels
  const obsLabelY = dotCY + 4;
  const estLabelY = estBaseY + r + (estMaxRow * step) / 2 + 4;
  const obsLabel  = `<text x="${padL - 8}" y="${obsLabelY.toFixed(1)}"
    font-family="Arial" font-size="11" font-weight="bold"
    fill="#888" text-anchor="end">observations</text>`;
  const estLabel  = `<text x="${padL - 8}" y="${estLabelY.toFixed(1)}"
    font-family="Arial" font-size="11" font-weight="bold"
    fill="#222" text-anchor="end">estimates</text>`;

  const bar = `
    <rect x="${padL}" y="${barY}" width="${Math.round(true_p * barW)}" height="${barH}"
      fill="${SAMPLE_BLUE}" rx="3"/>
    <rect x="${trueX}" y="${barY}" width="${barW - Math.round(true_p * barW)}" height="${barH}"
      fill="${SAMPLE_RED}" rx="3"/>
    <rect x="${padL}" y="${barY}" width="${barW}" height="${barH}"
      fill="none" stroke="#222" stroke-width="1.5" rx="3"/>
    <line x1="${trueX}" y1="${barY}" x2="${trueX}" y2="${barY + barH}"
      stroke="${DIST_COLOR}" stroke-width="6" stroke-linecap="round"/>`;

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
    ${bar}
    ${obsLabel}${blueDots}${redDots}
    ${estLabel}${estDots}
  </svg>`;
};
