/**
 * urn-binary.js
 * Shared urn SVG builder for the binary tutorial.
 *
 * Used by:
 *   plugin-tutorial-intro-binary.js   (progressive reveal via group opacity)
 *   plugin-practice-observation-binary.js (fully revealed from the start)
 *
 * The SVG contains three named groups:
 *   #tut-urn-dots      — dot grid + border rect
 *   #tut-urn-label     — bar with ??? label
 *   #tut-urn-highlight — ring around current observation dot
 *
 * Pass revealed=true to show all groups immediately (obs 2-5).
 * Pass revealed=false to hide all groups initially (obs 1, progressive reveal).
 */

const SAMPLE_BLUE = '#2563eb';
const SAMPLE_RED  = '#ef4444';
const DIST_COLOR  = '#16a34a';

export const buildUrnSVG = (p, currentValue, obsNum = 1, revealed = false) => {
  const COLS = 5, ROWS = 4, N = COLS * ROWS;
  const r    = 13, gap = 8, step = r * 2 + gap, pad = 16;
  const W     = COLS * step + pad * 2 - gap;
  const gridH = ROWS * step + pad * 2 - gap;

  const barGap = 20, barH = 18;
  const barY   = gridH + barGap;
  const H      = barY + barH + 6;
  const midX   = Math.round(p * W);
  const labelY = barY - 10;
  const vis    = revealed ? '1' : '0';

  const nBlue = Math.round(p * N);
  let colours = [
    ...Array(nBlue).fill(SAMPLE_BLUE),
    ...Array(N - nBlue).fill(SAMPLE_RED),
  ];
  let seed = 12345;
  const rnd = () => {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  };
  for (let i = colours.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [colours[i], colours[j]] = [colours[j], colours[i]];
  }

  const matchColor   = currentValue === 1 ? SAMPLE_BLUE : SAMPLE_RED;
  const matchIndices = colours.reduce((a, c, i) => { if (c === matchColor) a.push(i); return a; }, []);
  const matchIdx     = matchIndices[(obsNum - 1) % matchIndices.length];

  let dots = '', highlight = '';
  colours.forEach((col, i) => {
    const cx = pad + (i % COLS) * step + r;
    const cy = pad + Math.floor(i / COLS) * step + r;
    dots += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${col}" stroke="#fff" stroke-width="1.5"/>`;
    if (i === matchIdx) {
      highlight += `<circle cx="${cx}" cy="${cy}" r="${r + 3}"
        fill="none" stroke="#222" stroke-width="2.5"/>`;
    }
  });

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="100%"
    xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
    <g id="tut-urn-dots" style="opacity:${vis};">
      <rect x="8" y="8" width="178" height="144"
        fill="none" stroke="${DIST_COLOR}" stroke-width="2" rx="4"/>
      ${dots}
    </g>
    <g id="tut-urn-label" style="opacity:${vis};">
      <rect x="0" y="${barY}" width="${midX}" height="${barH}"
        fill="${SAMPLE_BLUE}" rx="3"/>
      <rect x="${midX}" y="${barY}" width="${W - midX}" height="${barH}"
        fill="${SAMPLE_RED}" rx="3"/>
      <rect x="0" y="${barY}" width="${W}" height="${barH}"
        fill="none" stroke="#222" stroke-width="1.5" rx="3"/>
      <line x1="${midX}" y1="${barY}" x2="${midX}" y2="${barY + barH}"
        stroke="${DIST_COLOR}" stroke-width="5" stroke-linecap="round"/>
      <text x="${midX}" y="${labelY}" text-anchor="middle"
            font-family="Arial" font-size="18" font-weight="bold"
            fill="#222">???</text>
    </g>
    <g id="tut-urn-highlight" style="opacity:${vis};">${highlight}</g>
  </svg>`;
};
