/**
 * draw-performance-numbers.js
 * Builds an SVG string for numbers trial/tutorial summary screens.
 *
 * Layout (top → bottom):
 *   Rows 1–N: One thin row per observation, in order. Each row independently
 *             shows: red obs thumb (raw value), a blue mean tick, a green
 *             line connecting the black estimate circle to that tick
 *             (visualizing that specific response's error/distance from
 *             the reference), and the black estimate circle itself on top.
 *             What the blue tick actually IS depends on `errorMode` (chat
 *             history, see scoring.js's own docstring for the
 *             full true_mean/running_mean rationale):
 *               'true_mean'    (default) -- same x every row, the trial's
 *                               fixed generative mean.
 *               'running_mean' -- a DIFFERENT x each row: that row's own
 *                               cumulative mean of the raw values shown so
 *                               far (via scoring.js's
 *                               computeRunningMeans, imported here so this
 *                               file and the error-total calculation in
 *                               build-trial-timeline.js never compute
 *                               running means two different ways).
 *   Legend:  red thumb "Observations" | black circle "Your estimates"
 *            | blue tick "True average" or "Correct answer" (label follows
 *            errorMode). No "Error" or bonus/coin entries (chat history,
 *            REMOVED): the green error line itself is still drawn per row
 *            (still informative at a glance), but bonus is no longer a
 *            per-response/per-row concept at all -- see bonus-
 *            numbers.js, now a single per-TRIAL reward reported in its
 *            own box above this SVG (plugin-trial-summary-numbers.js /
 *            plugin-tutorial-summary-numbers.js), not inside the plot.
 *            Removing both legend entries also fixed a real crowding
 *            problem the 5-entry legend had -- see this file's own git
 *            history for the estimated-widths math that justified adding
 *            the 5th entry in the first place; that concern is moot now.
 *
 * PAD_L and PAD_R are EQUAL (bar sits centered in the row) -- previously
 * widened asymmetrically then symmetrically to fit per-row coins, which are
 * gone now; shrunk back down since there's nothing needing extra margin.
 */
import { computeRunningMeans } from './scoring.js';
import { RED, BLUE, GREEN } from './palette.js';

export const normalPDF = (x, mu, sigma) =>
  Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));

export const SAMPLE_RED   = RED;
export const MEAN_BLUE    = BLUE;
export const ERROR_GREEN  = GREEN;
export const COIN_FILL    = '#f59e0b';

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

export const buildPerformanceSVG = (mu, sigma, values, responses, errorMode = 'true_mean') => {
  const vals  = values    || [];
  const resps = responses || [];
  const n     = Math.max(vals.length, resps.length);

  const H = PAD_T + n * ROW_STEP + PAD_B;

  const parts = [];
  const muX = xPos(mu);
  const runningMeans = errorMode === 'running_mean' ? computeRunningMeans(vals) : null;
  const meanLabel = errorMode === 'running_mean' ? 'Correct answer' : 'True average';

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
