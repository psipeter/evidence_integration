/**
 * bar-chart.js
 * Shared SVG builder for binary trial/practice summary screens.
 */

const SAMPLE_BLUE = '#2563eb';
const SAMPLE_RED  = '#ef4444';
const DIST_COLOR  = '#16a34a';

export const buildSummaryBarSVG = (true_p, values, responses) => {
  const dotR   = 7;
  const dotGap = 5;
  const dotStep = dotR * 2 + dotGap;
  const W      = 520;
  const padX   = 20, padY = 14;
  const legendH = 32;
  const barW   = W - padX * 2;
  const barH   = 32;
  const barY   = padY + legendH;
  const dotsRowY = barY + barH + 16;
  const H      = dotsRowY + dotR * 2 + 12;

  const blueW  = Math.round(true_p * barW);
  const redW   = barW - blueW;
  const trueX  = padX + blueW;
  const lineTop = barY - 14;  // longer tick above bar
  const lineBot = barY;         // stops at bar top — does not go through bar

  const blueRect = `<rect x="${padX}" y="${barY}" width="${blueW}" height="${barH}"
    fill="${SAMPLE_BLUE}" rx="3"/>`;
  const redRect  = `<rect x="${trueX}" y="${barY}" width="${redW}" height="${barH}"
    fill="${SAMPLE_RED}" rx="3"/>`;
  const border   = `<rect x="${padX}" y="${barY}" width="${barW}" height="${barH}"
    fill="none" stroke="#222" stroke-width="1.5" rx="3"/>`;
  const greenLineTop = barY;
  const greenLineBot = barY + barH;
  const greenLine = `<line x1="${trueX}" y1="${greenLineTop}" x2="${trueX}" y2="${greenLineBot}"
    stroke="${DIST_COLOR}" stroke-width="6" stroke-linecap="round"/>`;

  const estimateLines = (responses || []).map(resp => {
    if (resp === null) return '';
    const estX = padX + Math.round((resp / 100) * barW);
    return `<line x1="${estX}" y1="${lineTop}" x2="${estX}" y2="${lineBot}"
      stroke="#222" stroke-width="2" stroke-linecap="round"/>`;
  }).join('');

  const blueVals = (values || []).filter(v => v === 1);   // +1 = blue
  const redVals  = (values || []).filter(v => v === -1);  // -1 = red
  const dotCY    = dotsRowY + dotR;

  const blueDots = blueVals.map((_, i) => {
    const cx = padX + i * dotStep + dotR;
    return `<circle cx="${cx}" cy="${dotCY}" r="${dotR}"
      fill="${SAMPLE_BLUE}" stroke="#fff" stroke-width="1.5"/>`;
  }).join('');

  const redDots = redVals.map((_, i) => {
    const cx = padX + barW - i * dotStep - dotR;
    return `<circle cx="${cx}" cy="${dotCY}" r="${dotR}"
      fill="${SAMPLE_RED}" stroke="#fff" stroke-width="1.5"/>`;
  }).join('');

  const legendY  = barY - legendH + 10;
  const legend   = `
    <text x="${padX}" y="${legendY}"
      font-family="Arial" font-size="14" font-weight="bold"
      fill="#222" text-anchor="start">estimates</text>`;

  return `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">
    ${blueRect}${redRect}${border}
    ${estimateLines}${greenLine}
    ${legend}
    ${blueDots}${redDots}
  </svg>`;
};
