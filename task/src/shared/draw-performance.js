/**
 * draw-performance.js
 * Builds an SVG string for continuous trial/practice summary screens.
 *
 * Layout (top → bottom):
 *   Row 0:   Full-width number line (0–100) with green Gaussian overlay
 *            and blue solid true-mean line. No title, no border.
 *   Rows 1–N: One thin row per observation, in order.
 *             Red obs thumb (thick) drawn first; black filled circle for estimate on top.
 *   Legend:  red thumb "Observations" | black circle "Your estimate"
 *            | blue line "True mean"
 */

export const normalPDF = (x, mu, sigma) =>
  Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));

const SAMPLE_RED  = '#ef4444';
const DIST_COLOR  = '#16a34a';

const BAR_H_TOP  = 40;   // height of the Gaussian row
const BAR_H_OBS  = 4;    // height of each obs row (thin)
const ROW_GAP    = 3;    // gap between obs rows
const TOP_GAP    = 6;    // gap between Gaussian row and first obs row
const ROW_STEP   = BAR_H_OBS + ROW_GAP;
const EST_R      = 2;    // radius of estimate circle
const PAD_T      = 8;
const PAD_B      = 24;
const PAD_L      = 8;
const PAD_R      = 16;
const W          = 480;
const BAR_W      = W - PAD_L - PAD_R;

const xPos = (v) => PAD_L + Math.round((v / 100) * BAR_W);

const _numberLine = (y, h, color = '#e5e7eb') =>
  `<rect x="${PAD_L}" y="${y}" width="${BAR_W}" height="${h}" fill="${color}" rx="1"/>`;

const _obsThumb = (x, y, h) =>
  `<line x1="${x}" y1="${y - 1}" x2="${x}" y2="${y + h + 1}"
     stroke="${SAMPLE_RED}" stroke-width="2.5" stroke-linecap="round"/>`;

const _estCircle = (x, y, h) =>
  `<circle cx="${x}" cy="${y + h / 2}" r="${EST_R}"
     fill="#222"/>`;

export const buildPerformanceSVG = (mu, sigma, values, responses) => {
  const vals  = values    || [];
  const resps = responses || [];
  const n     = Math.max(vals.length, resps.length);

  const H = PAD_T + BAR_H_TOP + TOP_GAP + n * ROW_STEP + PAD_B;

  const parts = [];

  // ── Row 0: Gaussian overlay ───────────────────────────────────────────────
  const topY = PAD_T;
  const muX  = xPos(mu);

  const peakP   = normalPDF(mu, mu, sigma);
  const curveYs = (v) => topY + BAR_H_TOP - (normalPDF(v, mu, sigma) / peakP) * BAR_H_TOP * 0.92;
  const steps   = 200;
  const pts     = Array.from({ length: steps + 1 }, (_, i) => {
    const v = (i / steps) * 100;
    return `${xPos(v).toFixed(1)},${curveYs(v).toFixed(1)}`;
  }).join(' ');
  const fillPts = pts +
    ` ${(PAD_L + BAR_W).toFixed(1)},${(topY + BAR_H_TOP).toFixed(1)}` +
    ` ${PAD_L.toFixed(1)},${(topY + BAR_H_TOP).toFixed(1)}`;

  parts.push(`<polygon points="${fillPts}" fill="rgba(22,163,74,0.15)"/>`);
  parts.push(`<polyline points="${pts}" fill="none" stroke="${DIST_COLOR}" stroke-width="1.8" stroke-linejoin="round"/>`);
  parts.push(`<line x1="${muX}" y1="${topY}" x2="${muX}" y2="${topY + BAR_H_TOP}"
    stroke="#2563eb" stroke-width="2.5" stroke-linecap="round"/>`);

  // ── Rows 1–n: per-observation rows ────────────────────────────────────────
  const obsStartY = PAD_T + BAR_H_TOP + TOP_GAP;

  for (let i = 0; i < n; i++) {
    const obs  = vals[i];
    const resp = resps[i];
    const y    = obsStartY + i * ROW_STEP;
    const midY = y + BAR_H_OBS / 2;

    parts.push(_numberLine(y, BAR_H_OBS));

    // Obs: red thumb underneath; estimate: black circle on top
    if (obs  != null) parts.push(_obsThumb(xPos(obs),  y, BAR_H_OBS));
    if (resp != null) {
      parts.push(_estCircle(xPos(resp), y, BAR_H_OBS));
    } else {
      parts.push(`<text x="${PAD_L + BAR_W / 2}" y="${midY + 3}"
        font-family="Arial" font-size="8" fill="#bbb" text-anchor="middle">—</text>`);
    }
  }

  // ── Legend ────────────────────────────────────────────────────────────────
  const legY  = H - PAD_B + 12;
  const legX0 = PAD_L;

  parts.push(`
    <line x1="${legX0 + 4}" y1="${legY - 5}" x2="${legX0 + 4}" y2="${legY + 5}"
      stroke="${SAMPLE_RED}" stroke-width="2.5" stroke-linecap="round"/>
    <text x="${legX0 + 10}" y="${legY + 4}"
      font-family="Arial" font-size="9" fill="#888">Observations</text>
    <circle cx="${legX0 + 96}" cy="${legY}" r="${EST_R}"
      fill="#222"/>
    <text x="${legX0 + 102}" y="${legY + 4}"
      font-family="Arial" font-size="9" fill="#555">Your estimate</text>
    <line x1="${legX0 + 200}" y1="${legY - 5}" x2="${legX0 + 200}" y2="${legY + 5}"
      stroke="#2563eb" stroke-width="2.5" stroke-linecap="round"/>
    <text x="${legX0 + 206}" y="${legY + 4}"
      font-family="Arial" font-size="9" fill="#2563eb">True mean</text>
  `);

  return `<svg viewBox="0 0 ${W} ${H}" width="100%"
    xmlns="http://www.w3.org/2000/svg">
    ${parts.join('\n    ')}
  </svg>`;
};
