/**
 * urn-binary.js
 * Shared probability-bar SVG builder for the binary tutorial.
 *
 * Used by:
 *   plugin-tutorial-intro-binary.js       (progressive reveal via group opacity)
 *   plugin-tutorial-observation-binary.js (fully revealed from the start)
 *
 * The grid-of-dots urn metaphor was removed (Jul 2026) — the task uses independent
 * Bernoulli draws, not sampling without replacement. The bar + bubbling animation
 * (binary-draw-animation.js) conveys the generative process instead.
 *
 * SVG groups / elements:
 *   #tut-urn-bar      — blue/red bar (no label)
 *   #tut-urn-qmark    — "???" label, own opacity group so it can be revealed on
 *                       a different beat than the bar itself (intro screen reveals
 *                       it alongside box 2's text, not immediately with the bar)
 *   #tut-urn-bubbles  — ephemeral bubble <circle>s (filled by animation)
 *   #tut-urn-draw     — draw outcome circle above the bar
 *
 * Pass revealed=true to show the bar (and "???") immediately (obs 2–5).
 * Pass revealed=false to hide both initially (obs 1 intro, progressive reveal —
 * the intro plugin reveals the bar and "???" group independently).
 */

export const SAMPLE_BLUE = '#2563eb';
export const SAMPLE_RED  = '#ef4444';
export const DIST_COLOR  = '#16a34a';

export const LAYOUT = {
  W:       194,
  barY:    36,
  barH:    24,
  drawR:   6,
  drawGap: 8,
  H:       92,
};

const labelY = (barY, barH) => barY + barH + 26;

export const buildUrnSVG = (p, revealed = false) => {
  const { W, barY, barH, drawR, drawGap, H } = LAYOUT;
  const midX   = Math.round(p * W);
  const vis    = revealed ? '1' : '0';
  const drawCy = barY - drawGap - drawR;
  const qY     = labelY(barY, barH);

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="100%"
    xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
    <defs>
      <clipPath id="tut-urn-bar-clip">
        <rect x="0" y="${barY}" width="${W}" height="${barH}" rx="3"/>
      </clipPath>
    </defs>
    <g id="tut-urn-bar" style="opacity:${vis};">
      <rect x="0" y="${barY}" width="${midX}" height="${barH}"
        fill="${SAMPLE_BLUE}" rx="3"/>
      <rect x="${midX}" y="${barY}" width="${W - midX}" height="${barH}"
        fill="${SAMPLE_RED}" rx="3"/>
      <line x1="${midX}" y1="${barY}" x2="${midX}" y2="${barY + barH}"
        stroke="${DIST_COLOR}" stroke-width="5" stroke-linecap="round"/>
    </g>
    <g id="tut-urn-qmark" style="opacity:${vis};">
      <text x="${midX}" y="${qY}" text-anchor="middle"
            font-family="Arial" font-size="18" font-weight="bold"
            fill="${DIST_COLOR}">???</text>
    </g>
    <g id="tut-urn-bubbles" clip-path="url(#tut-urn-bar-clip)"
       style="opacity:${vis};"></g>
    <circle id="tut-urn-draw" cx="${(W / 2).toFixed(1)}" cy="${drawCy}"
      r="${drawR}" fill="#fff"
      style="opacity:0;"/>
  </svg>`;
};
