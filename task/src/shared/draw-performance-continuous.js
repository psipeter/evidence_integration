/**
 * draw-performance-continuous.js
 * Builds an SVG string for continuous trial/tutorial summary screens.
 *
 * Layout (top → bottom):
 *   Rows 1–N: One thin row per observation, in order. Each row independently
 *             shows: red obs thumb (raw value), a blue mean tick, a green
 *             line connecting the black estimate circle to that tick
 *             (visualizing that specific response's error/distance from
 *             the reference), and the black estimate circle itself on top.
 *             What the blue tick actually IS depends on `errorMode` (chat
 *             history, see bonus-continuous.js's own docstring for the
 *             full true_mean/running_mean rationale):
 *               'true_mean'    (default) -- same x every row, the trial's
 *                               fixed generative mean.
 *               'running_mean' -- a DIFFERENT x each row: that row's own
 *                               cumulative mean of the raw values shown so
 *                               far (via bonus-continuous.js's
 *                               computeRunningMeans, imported here so this
 *                               file and the error-total calculation in
 *                               build-trial-timeline.js never compute
 *                               running means two different ways).
 *   Legend:  red thumb "Observations" | black circle "Your estimates"
 *            | blue tick "True mean" or "Running mean" (label follows
 *            errorMode). No "Error" or bonus/coin entries (chat history,
 *            REMOVED): the green error line itself is still drawn per row
 *            (still informative at a glance), but bonus is no longer a
 *            per-response/per-row concept at all -- see bonus-
 *            continuous.js, now a single per-TRIAL reward reported in its
 *            own box above this SVG (plugin-trial-summary-continuous.js /
 *            plugin-tutorial-summary-continuous.js), not inside the plot.
 *            Removing both legend entries also fixed a real crowding
 *            problem the 5-entry legend had -- see this file's own git
 *            history for the estimated-widths math that justified adding
 *            the 5th entry in the first place; that concern is moot now.
 *
 * PAD_L and PAD_R are EQUAL (bar sits centered in the row) -- previously
 * widened asymmetrically then symmetrically to fit per-row coins, which are
 * gone now; shrunk back down since there's nothing needing extra margin.
 */
import { computeRunningMeans } from './bonus-continuous.js';

export const normalPDF = (x, mu, sigma) =>
  Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));

export const SAMPLE_RED   = '#ef4444';
export const MEAN_BLUE    = '#2563eb';
export const ERROR_GREEN  = '#16a34a';
export const COIN_FILL    = '#f59e0b';
export const COIN_STROKE  = '#92400e';

const BAR_H_OBS  = 4;    // height of each obs row (thin)
const ROW_GAP    = 3;    // gap between obs rows
const ROW_STEP   = BAR_H_OBS + ROW_GAP;
const EST_R      = 2;    // radius of estimate circle
const PAD_T      = 8;
const PAD_B      = 24;
const PAD_L      = 16;
const PAD_R      = 16;
const W          = 480;
const BAR_W      = W - PAD_L - PAD_R;

const xPos = (v) => PAD_L + Math.round((v / 100) * BAR_W);

const _numberLine = (y, h, color = '#e5e7eb') =>
  `<rect x="${PAD_L}" y="${y}" width="${BAR_W}" height="${h}" fill="${color}" rx="1"/>`;

const _obsThumb = (x, y, h) =>
  `<line x1="${x}" y1="${y - 1}" x2="${x}" y2="${y + h + 1}"
     stroke="${SAMPLE_RED}" stroke-width="2.5" stroke-linecap="round"/>`;

const _meanTick = (x, y, h) =>
  `<line x1="${x}" y1="${y - 1}" x2="${x}" y2="${y + h + 1}"
     stroke="${MEAN_BLUE}" stroke-width="2.5" stroke-linecap="round"/>`;

const _errorLine = (xEst, xMean, y, h) =>
  `<line x1="${xEst}" y1="${y + h / 2}" x2="${xMean}" y2="${y + h / 2}"
     stroke="${ERROR_GREEN}" stroke-width="1.5" stroke-linecap="round" opacity="0.75"/>`;

const _estCircle = (x, y, h) =>
  `<circle cx="${x}" cy="${y + h / 2}" r="${EST_R}"
     fill="#222"/>`;

/**
 * A single stylized coin glyph, as a standalone inline SVG (not tied to
 * this file's row/legend coordinate system) -- used by the new total-
 * error/reward box on both summary screens (plugin-trial-summary-
 * continuous.js / plugin-tutorial-summary-continuous.js), NOT inside the
 * plot itself.
 */
export function coinGlyph(size = 14) {
  const r = size / 2 - 1;
  const c = size / 2;
  return `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}"
    style="vertical-align:middle;">
    <circle cx="${c}" cy="${c}" r="${r}" fill="${COIN_FILL}" stroke="${COIN_STROKE}" stroke-width="0.8"/>
    <circle cx="${c}" cy="${c}" r="${r * 0.55}" fill="none" stroke="${COIN_STROKE}" stroke-width="0.5" opacity="0.6"/>
  </svg>`;
}

export const buildPerformanceSVG = (mu, sigma, values, responses, errorMode = 'true_mean') => {
  const vals  = values    || [];
  const resps = responses || [];
  const n     = Math.max(vals.length, resps.length);

  const H = PAD_T + n * ROW_STEP + PAD_B;

  const parts = [];
  const muX = xPos(mu);
  const runningMeans = errorMode === 'running_mean' ? computeRunningMeans(vals) : null;
  const meanLabel = errorMode === 'running_mean' ? 'Running mean' : 'True mean';

  // ── Rows 1–n: per-observation rows ────────────────────────────────────────
  const obsStartY = PAD_T;

  for (let i = 0; i < n; i++) {
    const obs  = vals[i];
    const resp = resps[i];
    const y    = obsStartY + i * ROW_STEP;
    const midY = y + BAR_H_OBS / 2;
    // Per-row reference x: the running mean up to THIS row when in
    // running_mean mode, else the single fixed true-mean x for every row.
    const rowMeanX = (runningMeans && runningMeans[i] != null) ? xPos(runningMeans[i]) : muX;

    parts.push(_numberLine(y, BAR_H_OBS));

    // Error line drawn BEFORE the ticks/circle so it sits underneath them,
    // not obscuring the markers themselves.
    if (resp != null) parts.push(_errorLine(xPos(resp), rowMeanX, y, BAR_H_OBS));

    if (obs  != null) parts.push(_obsThumb(xPos(obs), y, BAR_H_OBS));
    parts.push(_meanTick(rowMeanX, y, BAR_H_OBS));

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
      font-family="Arial" font-size="9" fill="${SAMPLE_RED}">Observations</text>
    <circle cx="${legX0 + 96}" cy="${legY}" r="${EST_R}"
      fill="#222"/>
    <text x="${legX0 + 102}" y="${legY + 4}"
      font-family="Arial" font-size="9" fill="#555">Your estimates</text>
    <line x1="${legX0 + 200}" y1="${legY - 5}" x2="${legX0 + 200}" y2="${legY + 5}"
      stroke="${MEAN_BLUE}" stroke-width="2.5" stroke-linecap="round"/>
    <text x="${legX0 + 206}" y="${legY + 4}"
      font-family="Arial" font-size="9" fill="${MEAN_BLUE}">${meanLabel}</text>
  `);

  return `<svg viewBox="0 0 ${W} ${H}" width="100%"
    xmlns="http://www.w3.org/2000/svg">
    ${parts.join('\n    ')}
  </svg>`;
};
