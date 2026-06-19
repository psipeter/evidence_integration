/**
 * draw-performance.js
 * Builds an SVG string for trial/practice summary screens.
 * Green distribution, red observation ticks, black estimate ticks.
 */

export const normalPDF = (x, mu, sigma) =>
  Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));

export const buildPerformanceSVG = (mu, sigma, values, responses) => {
  const W = 520, H = 190;
  const xMin = 0, xMax = 100;
  const pad  = { l: 88, r: 88, t: 22, b: 38 };
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;
  const axisY = pad.t + plotH;
  const tickH = 14;
  const obsRowY  = axisY;        // red ticks: flush with axis
  const estRowY  = axisY + 18;   // black ticks: one row below

  const xPos  = (x) => pad.l + (x - xMin) / (xMax - xMin) * plotW;
  const peakP = normalPDF(mu, mu, sigma);
  const yPos  = (p) => pad.t + plotH - (p / peakP) * plotH * 0.88;

  // Curve points
  const steps = 300;
  const curvePoints = Array.from({ length: steps + 1 }, (_, i) => {
    const x = xMin + (i / steps) * (xMax - xMin);
    return `${xPos(x).toFixed(2)},${yPos(normalPDF(x, mu, sigma)).toFixed(2)}`;
  }).join(' ');

  const fillPoints = curvePoints
    + ` ${xPos(xMax).toFixed(2)},${axisY} ${xPos(xMin).toFixed(2)},${axisY}`;

  // Observation ticks (red) — top row
  const obsTicks = (values || []).map(v =>
    `<line x1="${xPos(v).toFixed(2)}" y1="${obsRowY}"
           x2="${xPos(v).toFixed(2)}" y2="${obsRowY + tickH}"
           stroke="#ef4444" stroke-width="2.5" stroke-linecap="round"/>`
  ).join('');

  // Estimate ticks (black) — bottom row
  const estTicks = (responses || []).map(r => {
    if (r === null) return '';
    return `<line x1="${xPos(r).toFixed(2)}" y1="${estRowY}"
                  x2="${xPos(r).toFixed(2)}" y2="${estRowY + tickH}"
                  stroke="#222" stroke-width="2.5" stroke-linecap="round"/>`;
  }).join('');

  // Row labels: right-aligned in left padding, vertically centred on each tick row
  const obsLegend = `
    <text x="${pad.l - 6}" y="${(obsRowY + tickH / 2 + 4).toFixed(1)}"
      font-family="Arial" font-size="11" font-weight="bold"
      fill="#ef4444" text-anchor="end">observations</text>`;
  const estLegend = `
    <text x="${pad.l - 6}" y="${(estRowY + tickH / 2 + 4).toFixed(1)}"
      font-family="Arial" font-size="11" font-weight="bold"
      fill="#222" text-anchor="end">estimates</text>`;

  const muX = xPos(mu);
  const meanLine = `
    <line x1="${muX.toFixed(2)}" y1="${pad.t}"
          x2="${muX.toFixed(2)}" y2="${axisY}"
          stroke="#2563eb" stroke-width="2"
          stroke-dasharray="5 3" stroke-linecap="round"/>`;

  return `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">
    <!-- Distribution fill -->
    <polygon points="${fillPoints}" fill="rgba(22,163,74,0.12)" stroke="none"/>
    <!-- Distribution outline -->
    <polyline points="${curvePoints}"
      fill="none" stroke="#16a34a" stroke-width="1.5" stroke-linejoin="round"/>
    <!-- True mean dashed line -->
    ${meanLine}
    <!-- Axis -->
    <line x1="${pad.l}" y1="${axisY}" x2="${pad.l + plotW}" y2="${axisY}"
      stroke="#ccc" stroke-width="1"/>
    <line x1="${pad.l}" y1="${axisY - 4}" x2="${pad.l}" y2="${axisY + 4}"
      stroke="#ccc" stroke-width="1"/>
    <text x="${pad.l}" y="${axisY - 7}" text-anchor="middle"
      font-family="Arial" font-size="10" fill="#bbb">0</text>
    <line x1="${pad.l + plotW}" y1="${axisY - 4}" x2="${pad.l + plotW}" y2="${axisY + 4}"
      stroke="#ccc" stroke-width="1"/>
    <text x="${pad.l + plotW}" y="${axisY - 7}" text-anchor="middle"
      font-family="Arial" font-size="10" fill="#bbb">100</text>

    ${obsTicks}
    ${estTicks}
    ${obsLegend}
    ${estLegend}
  </svg>`;
};
