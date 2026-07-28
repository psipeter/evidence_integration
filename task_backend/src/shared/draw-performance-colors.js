/**
 * draw-performance-colors.js
 * SVG for colors trial/tutorial summary screens.
 *
 * Layout (top → bottom):
 *   Rows 1–N: Thin gray bars (chat history -- previously a blue/red split
 *             fill; removed, see below), no border. A black CIRCLE (chat
 *             history -- previously a vertical tick line; switched to a
 *             circle, matching draw-performance-numbers.js's own
 *             estimate marker, since two thin overlapping lines at nearly
 *             the same x read as one thicker line, while a circle sitting
 *             on a line stays visually distinct even when they nearly
 *             coincide) marks the estimate. All observation circles on
 *             the LEFT of the bar column, still colored by that draw's
 *             actual value (blue/red) -- unlike the bar fill, this IS
 *             genuinely informative (which ball was drawn), not just
 *             decorative. Each row ALSO gets its own green reference tick
 *             (mirrors draw-performance-numbers.js's redesign) and a
 *             violet line connecting the estimate circle to that
 *             reference tick, visualizing that specific response's
 *             error/distance. What the reference tick actually IS depends
 *             on `errorMode` (see scoring.js's own docstring for
 *             the full true_p/running_p rationale):
 *               'true_p'    (default) -- same x every row, the trial's
 *                           fixed generative probability.
 *               'running_p' -- a DIFFERENT x each row: that row's own
 *                           running percentage of blue draws so far (via
 *                           scoring.js's computeRunningRatios).
 *   Legend:  Half-blue/half-red circle "Observations" | black circle
 *            "Your estimates" | green tick "True probability" or
 *            "Running ratio" (label follows errorMode) | violet line
 *            "Error"
 *
 * TWO real bugs found and fixed together here (chat history):
 *   1. The blue/red split bar fill was fully opaque and drawn AFTER the
 *      error line in the per-row loop -- completely covering it. The
 *      error line was correctly computed and pushed into `parts`, it was
 *      just invisible underneath the bar's own rects. Removing the
 *      colored fill (replaced with a plain gray background, matching
 *      draw-performance-numbers.js's own _numberLine) fixes this as a
 *      side effect, not just cosmetically -- there's no longer an opaque
 *      shape drawn on top of it at all.
 *   2. (Separately, scoring.js) the reward formula itself was
 *      saturating at 0 for nearly all realistic responses -- see that
 *      file's own docstring for the full diagnosis; unrelated to this
 *      file, but discovered in the same conversation turn.
 *
 * Previously had a Row 0 with a full-height true-p bar spanning the whole
 * figure. Removed (chat history, mirrors numbers's identical change):
 * replaced with the per-row tick + error line instead, so every individual
 * response's accuracy is visible directly against its own row rather than
 * needing to compare against one shared reference row above everything
 * else. Error line uses VIOLET, not numbers's green -- green is
 * already taken here by the true-probability/running-ratio tick (an
 * existing convention predating this redesign), and blue/red are taken by
 * the ball colors, so a fourth, previously-unused color was needed.
 */
import { computeRunningRatios } from './scoring.js';
import { BLUE, RED, GREEN } from './palette.js';

const SAMPLE_BLUE  = BLUE;
const SAMPLE_RED   = RED;
const DIST_COLOR   = GREEN;
const ERROR_VIOLET = '#9333ea';

const OBS_R     = 5;    // observation circle radius
const BAR_H_OBS = 4;    // height of each obs/estimate bar
const ROW_GAP   = 4;    // vertical gap between obs bars
const ROW_STEP  = BAR_H_OBS + ROW_GAP;
const PAD_T     = 8;
const PAD_B     = 24;
const PAD_L     = 24;   // left margin — obs circles sit in left pad
const PAD_R     = 16;
const W         = 480;
const BAR_W     = W - PAD_L - PAD_R;

const OBS_CX    = PAD_L / 2;  // centre x for all obs circles (left of bar)

const respToX = (resp) => PAD_L + Math.round((resp / 100) * BAR_W);
// Single unified 0-100-scale-to-x function -- true_p arrives as a 0-1
// fraction and is converted to a 0-100 percentage ONCE, explicitly, at the
// call site below, rather than guessing which scale a given number is on
// by its magnitude (a real bug caught before it shipped: a running ratio
// of exactly 1 -- i.e. 1% -- would have been misread as "1.0 probability"
// = 100% by a magnitude-based heuristic).
const pctToX = (pct) => PAD_L + Math.round((pct / 100) * BAR_W);

const EST_R = 2;

const _numberLine = (y, h, color = '#e5e7eb') =>
  `<rect x="${PAD_L}" y="${y}" width="${BAR_W}" height="${h}" fill="${color}" rx="1"/>`;

const _estCircle = (x, y, h) =>
  `<circle cx="${x}" cy="${y + h / 2}" r="${EST_R}" fill="#222"/>`;

const _refTick = (x, y, h) =>
  `<line x1="${x}" y1="${y - 1}" x2="${x}" y2="${y + h + 1}"
     stroke="${DIST_COLOR}" stroke-width="2.5" stroke-linecap="round"/>`;

const _errorLine = (xEst, xRef, y, h) =>
  `<line x1="${xEst}" y1="${y + h / 2}" x2="${xRef}" y2="${y + h / 2}"
     stroke="${ERROR_VIOLET}" stroke-width="1.5" stroke-linecap="round" opacity="0.75"/>`;

// Half-blue / half-red circle for legend
const _splitCircle = (cx, cy, r) => `
  <clipPath id="lc-left"><rect x="${cx - r - 1}" y="${cy - r - 1}" width="${r + 1}" height="${r * 2 + 2}"/></clipPath>
  <clipPath id="lc-right"><rect x="${cx}" y="${cy - r - 1}" width="${r + 1}" height="${r * 2 + 2}"/></clipPath>
  <circle cx="${cx}" cy="${cy}" r="${r}" fill="${SAMPLE_BLUE}" clip-path="url(#lc-left)"/>
  <circle cx="${cx}" cy="${cy}" r="${r}" fill="${SAMPLE_RED}" clip-path="url(#lc-right)"/>
  <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#fff" stroke-width="1"/>`;

export const buildSummaryBarSVG = (true_p, values, responses, errorMode = 'true_p') => {
  const vals  = values    || [];
  const resps = responses || [];
  const n     = Math.max(vals.length, resps.length);

  const H = PAD_T + n * ROW_STEP + PAD_B;

  const parts = [];
  const trueX = pctToX(true_p * 100);
  const runningRatios = errorMode === 'running_p' ? computeRunningRatios(vals) : null;
  const refLabel = errorMode === 'running_p' ? 'Running ratio' : 'True probability';

  // ── Rows 1–n: per-observation bars ────────────────────────────────────────
  const obsStartY = PAD_T;

  for (let i = 0; i < n; i++) {
    const obs  = vals[i];
    const resp = resps[i];
    const y    = obsStartY + i * ROW_STEP;
    const midY = y + BAR_H_OBS / 2;
    // Per-row reference x: the running ratio up to THIS row when in
    // running_p mode, else the single fixed true-p x for every row.
    const rowRefX = (runningRatios && runningRatios[i] != null) ? pctToX(runningRatios[i]) : trueX;

    const splitX = resp != null ? respToX(resp) : PAD_L + Math.round(BAR_W / 2);

    parts.push(_numberLine(y, BAR_H_OBS));

    // Error line drawn on top of the (now plain, non-opaque-colored)
    // background -- correctly visible now, see module docstring bug #1.
    if (resp != null) parts.push(_errorLine(splitX, rowRefX, y, BAR_H_OBS));

    parts.push(_refTick(rowRefX, y, BAR_H_OBS));
    if (resp != null) parts.push(_estCircle(splitX, y, BAR_H_OBS));

    // All obs circles on the LEFT, coloured by value -- unlike the bar
    // fill (removed), this genuinely conveys information (which ball was
    // drawn), so it stays colored.
    const fill = obs === 1 ? SAMPLE_BLUE : obs === -1 ? SAMPLE_RED : '#aaa';
    parts.push(`<circle cx="${OBS_CX}" cy="${midY}" r="${OBS_R - 1}"
      fill="${fill}" stroke="#fff" stroke-width="1"/>`);

    // Timed-out: dash
    if (resp == null) {
      parts.push(`<text x="${PAD_L + BAR_W / 2}" y="${midY + 3}"
        font-family="Arial" font-size="8" fill="#bbb" text-anchor="middle">—</text>`);
    }
  }

  // ── Legend ────────────────────────────────────────────────────────────────
  const legY  = H - PAD_B + 12;
  const legR  = OBS_R - 1;
  const legX0 = PAD_L;

  parts.push(_splitCircle(legX0 + legR, legY, legR));
  parts.push(`<text x="${legX0 + legR * 2 + 4}" y="${legY + 4}"
    font-family="Arial" font-size="9" fill="#888">Observations</text>`);

  parts.push(`
    <circle cx="${legX0 + 100}" cy="${legY}" r="${EST_R}" fill="#222"/>
    <text x="${legX0 + 105}" y="${legY + 4}"
      font-family="Arial" font-size="9" fill="#555">Your estimates</text>
    <line x1="${legX0 + 210}" y1="${legY - 5}" x2="${legX0 + 210}" y2="${legY + 5}"
      stroke="${DIST_COLOR}" stroke-width="2.5" stroke-linecap="round"/>
    <text x="${legX0 + 215}" y="${legY + 4}"
      font-family="Arial" font-size="9" fill="${DIST_COLOR}">${refLabel}</text>
    <line x1="${legX0 + 320}" y1="${legY}" x2="${legX0 + 336}" y2="${legY}"
      stroke="${ERROR_VIOLET}" stroke-width="1.5" stroke-linecap="round"/>
    <text x="${legX0 + 342}" y="${legY + 4}"
      font-family="Arial" font-size="9" fill="${ERROR_VIOLET}">Error</text>`);

  return `<svg viewBox="0 0 ${W} ${H}" width="100%"
    xmlns="http://www.w3.org/2000/svg">
    ${parts.join('\n    ')}
  </svg>`;
};
