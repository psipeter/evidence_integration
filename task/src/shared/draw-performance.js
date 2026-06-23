/**
 * draw-performance.js
 * Builds an SVG string for trial/practice summary screens.
 *
 * Layout (top → bottom):
 *   Green distribution curve + true mean dashed line
 *   Axis (0–100)
 *   "observations" row: red dot plot stacked upward from axis
 *   "estimates" row:    black dot plot stacked upward from a second baseline
 *
 * Dot plot stacking: circles of radius r are placed at the x position of each
 * value; if a circle would overlap an existing one at the same row it moves up
 * to the next row. This avoids any overlap while keeping exact x positions.
 */

export const normalPDF = (x, mu, sigma) =>
  Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));

// Stack dots: returns array of {x, row} for each value.
// row=0 is the bottom row; higher rows stack upward.
const stackDots = (values, xPos, r) => {
  const placed = []; // {x, row}
  return values.map(v => {
    const x = xPos(v);
    let row = 0;
    while (placed.some(p => p.row === row && Math.abs(p.x - x) < r * 2)) row++;
    placed.push({ x, row });
    return { x, row };
  });
};

export const buildPerformanceSVG = (mu, sigma, values, responses) => {
  const W = 520;
  const xMin = 0, xMax = 100;
  const pad  = { l: 88, r: 88, t: 22 };
  const plotW = W - pad.l - pad.r;

  const r      = 3;    // dot radius
  const gap    = 1;    // gap between stacked dots
  const step   = r * 2 + gap;

  const xPos = (x) => pad.l + (x - xMin) / (xMax - xMin) * plotW;

  // Stack obs and est dots
  const obsStacked = stackDots(values || [], xPos, r);
  const estStacked = stackDots((responses || []).filter(r => r !== null), xPos, r);
  const obsMaxRow  = obsStacked.reduce((m, d) => Math.max(m, d.row), 0);
  const estMaxRow  = estStacked.reduce((m, d) => Math.max(m, d.row), 0);

  // Heights: distribution area + axis + obs rows + gap + est rows + labels
  const distH    = 90;
  const axisY    = pad.t + distH;
  const obsH     = (obsMaxRow + 1) * step + r;
  const rowGap   = 12;   // gap between obs and est sections
  const estH     = (estMaxRow + 1) * step + r;
  const labelH   = 16;
  const H        = axisY + obsH + rowGap + estH + labelH;

  // Obs dots stack upward from axisY
  const obsDots = obsStacked.map(({ x, row }) => {
    const cy = axisY + r + row * step;
    return `<circle cx="${x.toFixed(1)}" cy="${cy.toFixed(1)}" r="${r}"
      fill="#ef4444" stroke="#fff" stroke-width="1.2" opacity="0.85"/>`;
  }).join('');

  // Est dots stack upward from estBaseY
  const estBaseY = axisY + obsH + rowGap;
  const estDots  = estStacked.map(({ x, row }) => {
    const cy = estBaseY + r + row * step;
    return `<circle cx="${x.toFixed(1)}" cy="${cy.toFixed(1)}" r="${r}"
      fill="#222" stroke="#fff" stroke-width="1.2" opacity="0.85"/>`;
  }).join('');

  // Row labels — right-aligned in left padding
  const obsLabelY = axisY + r + (obsMaxRow * step) / 2 + 4;
  const estLabelY = estBaseY + r + (estMaxRow * step) / 2 + 4;
  const obsLabel  = `<text x="${pad.l - 8}" y="${obsLabelY.toFixed(1)}"
    font-family="Arial" font-size="11" font-weight="bold"
    fill="#ef4444" text-anchor="end">observations</text>`;
  const estLabel  = `<text x="${pad.l - 8}" y="${estLabelY.toFixed(1)}"
    font-family="Arial" font-size="11" font-weight="bold"
    fill="#222" text-anchor="end">estimates</text>`;

  // Distribution curve
  const peakP = normalPDF(mu, mu, sigma);
  const yPos  = (p) => pad.t + distH - (p / peakP) * distH * 0.88;
  const steps = 300;
  const curvePts = Array.from({ length: steps + 1 }, (_, i) => {
    const x = xMin + (i / steps) * (xMax - xMin);
    return `${xPos(x).toFixed(2)},${yPos(normalPDF(x, mu, sigma)).toFixed(2)}`;
  }).join(' ');
  const fillPts = curvePts +
    ` ${xPos(xMax).toFixed(2)},${axisY} ${xPos(xMin).toFixed(2)},${axisY}`;

  const muX = xPos(mu);

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
    <polygon points="${fillPts}" fill="rgba(22,163,74,0.12)" stroke="none"/>
    <polyline points="${curvePts}"
      fill="none" stroke="#16a34a" stroke-width="1.5" stroke-linejoin="round"/>
    <line x1="${muX.toFixed(2)}" y1="${pad.t}"
          x2="${muX.toFixed(2)}" y2="${axisY}"
          stroke="#2563eb" stroke-width="2"
          stroke-dasharray="5 3" stroke-linecap="round"/>
    <line x1="${pad.l}" y1="${axisY}" x2="${pad.l + plotW}" y2="${axisY}"
      stroke="#ccc" stroke-width="1"/>
    <text x="${pad.l}" y="${axisY - 4}" text-anchor="middle"
      font-family="Arial" font-size="10" fill="#bbb">0</text>
    <text x="${pad.l + plotW}" y="${axisY - 4}" text-anchor="middle"
      font-family="Arial" font-size="10" fill="#bbb">100</text>
    ${obsLabel}${obsDots}
    ${estLabel}${estDots}
  </svg>`;
};
