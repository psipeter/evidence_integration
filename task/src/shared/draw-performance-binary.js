/**
 * draw-performance-binary.js
 * SVG for binary trial/tutorial summary screens.
 *
 * Layout (top → bottom):
 *   Row 0:   True-p bar — full height, no title label
 *   Rows 1–N: Thin bars, no border. Split at estimate (vertical thumb).
 *             All observation circles on the LEFT of the bar column.
 *             Blue circle (+1) or red circle (−1).
 *   Legend:  Half-blue/half-red circle "Observations" | thumb "Your estimates"
 *            | green line "True probability"
 */

const SAMPLE_BLUE = '#2563eb';
const SAMPLE_RED  = '#ef4444';
const DIST_COLOR  = '#16a34a';

const OBS_R     = 5;    // observation circle radius
const BAR_H_TOP = 32;   // height of true-p bar
const BAR_H_OBS = 4;    // height of each obs/estimate bar
const ROW_GAP   = 4;    // vertical gap between obs bars
const TOP_GAP   = 6;    // gap between true-p bar and first obs bar
const ROW_STEP  = BAR_H_OBS + ROW_GAP;
const PAD_T     = 8;
const PAD_B     = 24;
const PAD_L     = 24;   // left margin — obs circles sit in left pad
const PAD_R     = 16;
const W         = 480;
const BAR_W     = W - PAD_L - PAD_R;

const OBS_CX    = PAD_L / 2;  // centre x for all obs circles (left of bar)

const respToX = (resp) => PAD_L + Math.round((resp / 100) * BAR_W);
const trueToX = (p)    => PAD_L + Math.round(p * BAR_W);

const _topBar = (y, trueX) => [
  `<rect x="${PAD_L}" y="${y}" width="${trueX - PAD_L}" height="${BAR_H_TOP}" fill="${SAMPLE_BLUE}" rx="3"/>`,
  `<rect x="${trueX}" y="${y}" width="${BAR_W - (trueX - PAD_L)}" height="${BAR_H_TOP}" fill="${SAMPLE_RED}" rx="3"/>`,
  `<line x1="${trueX}" y1="${y}" x2="${trueX}" y2="${y + BAR_H_TOP}"
     stroke="${DIST_COLOR}" stroke-width="5" stroke-linecap="round"/>`,
].join('');

const _obsBar = (y, splitX) => [
  `<rect x="${PAD_L}" y="${y}" width="${splitX - PAD_L}" height="${BAR_H_OBS}" fill="${SAMPLE_BLUE}" rx="1"/>`,
  `<rect x="${splitX}" y="${y}" width="${BAR_W - (splitX - PAD_L)}" height="${BAR_H_OBS}" fill="${SAMPLE_RED}" rx="1"/>`,
  `<line x1="${splitX}" y1="${y - 1}" x2="${splitX}" y2="${y + BAR_H_OBS + 1}"
     stroke="#222" stroke-width="2.5" stroke-linecap="round"/>`,
].join('');

// Half-blue / half-red circle for legend
const _splitCircle = (cx, cy, r) => `
  <clipPath id="lc-left"><rect x="${cx - r - 1}" y="${cy - r - 1}" width="${r + 1}" height="${r * 2 + 2}"/></clipPath>
  <clipPath id="lc-right"><rect x="${cx}" y="${cy - r - 1}" width="${r + 1}" height="${r * 2 + 2}"/></clipPath>
  <circle cx="${cx}" cy="${cy}" r="${r}" fill="${SAMPLE_BLUE}" clip-path="url(#lc-left)"/>
  <circle cx="${cx}" cy="${cy}" r="${r}" fill="${SAMPLE_RED}" clip-path="url(#lc-right)"/>
  <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#fff" stroke-width="1"/>`;

export const buildSummaryBarSVG = (true_p, values, responses) => {
  const vals  = values    || [];
  const resps = responses || [];
  const n     = Math.max(vals.length, resps.length);

  const H = PAD_T + BAR_H_TOP + TOP_GAP + n * ROW_STEP + PAD_B;

  const parts = [];

  // ── Row 0: true-p bar (no title) ─────────────────────────────────────────
  const topY  = PAD_T;
  const trueX = trueToX(true_p);
  parts.push(_topBar(topY, trueX));

  // ── Rows 1–n: per-observation bars ────────────────────────────────────────
  const obsStartY = PAD_T + BAR_H_TOP + TOP_GAP;

  for (let i = 0; i < n; i++) {
    const obs  = vals[i];
    const resp = resps[i];
    const y    = obsStartY + i * ROW_STEP;
    const midY = y + BAR_H_OBS / 2;

    const splitX = resp != null ? respToX(resp) : PAD_L + Math.round(BAR_W / 2);
    parts.push(_obsBar(y, splitX));

    // All obs circles on the LEFT, coloured by value
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
    <line x1="${legX0 + 100}" y1="${legY - 5}" x2="${legX0 + 100}" y2="${legY + 5}"
      stroke="#222" stroke-width="2.5" stroke-linecap="round"/>
    <text x="${legX0 + 105}" y="${legY + 4}"
      font-family="Arial" font-size="9" fill="#555">Your estimates</text>
    <line x1="${legX0 + 210}" y1="${legY - 5}" x2="${legX0 + 210}" y2="${legY + 5}"
      stroke="${DIST_COLOR}" stroke-width="3" stroke-linecap="round"/>
    <text x="${legX0 + 215}" y="${legY + 4}"
      font-family="Arial" font-size="9" fill="${DIST_COLOR}">True probability</text>`);

  return `<svg viewBox="0 0 ${W} ${H}" width="100%"
    xmlns="http://www.w3.org/2000/svg">
    ${parts.join('\n    ')}
  </svg>`;
};
