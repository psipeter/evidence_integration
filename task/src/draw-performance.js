/**
 * draw-performance.js
 * Green distribution, red observation ticks, black estimate ticks.
 * No true mean shown.
 */

export const normalPDF = (x, mu, sigma) =>
  Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));

export const drawPerformance = (canvas, mu, sigma, values, responses) => {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const xMin  = 10, xMax = 99;
  const pad   = { l: 36, r: 36, t: 12, b: 36 };
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;

  const xPx     = (x) => pad.l + (x - xMin) / (xMax - xMin) * plotW;
  const peakPDF = normalPDF(mu, mu, sigma);
  const yPx     = (p) => pad.t + plotH - (p / peakPDF) * plotH * 0.88;
  const axisY   = pad.t + plotH;
  const tickW   = 1.5;
  const tickH   = 8;

  // ── Green distribution fill ───────────────────────────────────────────
  ctx.beginPath();
  for (let px = 0; px <= plotW; px++) {
    const x = xMin + (px / plotW) * (xMax - xMin);
    const p = normalPDF(x, mu, sigma);
    px === 0 ? ctx.moveTo(pad.l + px, yPx(p)) : ctx.lineTo(pad.l + px, yPx(p));
  }
  ctx.lineTo(pad.l + plotW, axisY);
  ctx.lineTo(pad.l, axisY);
  ctx.closePath();
  ctx.fillStyle = 'rgba(22,163,74,0.12)'; ctx.fill();

  // ── Green distribution outline ────────────────────────────────────────
  ctx.beginPath();
  for (let px = 0; px <= plotW; px++) {
    const x = xMin + (px / plotW) * (xMax - xMin);
    const p = normalPDF(x, mu, sigma);
    px === 0 ? ctx.moveTo(pad.l + px, yPx(p)) : ctx.lineTo(pad.l + px, yPx(p));
  }
  ctx.strokeStyle = '#16a34a'; ctx.lineWidth = 1.5; ctx.stroke();

  // ── Axis ──────────────────────────────────────────────────────────────
  ctx.beginPath();
  ctx.moveTo(pad.l, axisY); ctx.lineTo(pad.l + plotW, axisY);
  ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1; ctx.stroke();
  ctx.fillStyle = '#999'; ctx.font = '10px Arial'; ctx.textAlign = 'center';
  ctx.fillText('10', pad.l,         axisY + 12);
  ctx.fillText('99', pad.l + plotW, axisY + 12);

  // ── Red observation ticks ─────────────────────────────────────────────
  values.forEach((v) => {
    ctx.beginPath();
    ctx.moveTo(xPx(v), axisY); ctx.lineTo(xPx(v), axisY + tickH);
    ctx.strokeStyle = '#ef4444'; ctx.lineWidth = tickW; ctx.stroke();
  });

  // ── Black estimate ticks ──────────────────────────────────────────────
  responses.forEach((r) => {
    if (r === null) return;
    ctx.beginPath();
    ctx.moveTo(xPx(r), axisY); ctx.lineTo(xPx(r), axisY + tickH);
    ctx.strokeStyle = '#222'; ctx.lineWidth = tickW; ctx.stroke();
  });

  // ── Legend ────────────────────────────────────────────────────────────
  const legY = axisY + tickH + 14;
  ctx.font = '10px Arial'; ctx.textAlign = 'left';

  // Red — observations
  ctx.strokeStyle = '#ef4444'; ctx.lineWidth = tickW;
  ctx.beginPath(); ctx.moveTo(pad.l, legY - 4); ctx.lineTo(pad.l + 12, legY - 4); ctx.stroke();
  ctx.fillStyle = '#555'; ctx.fillText('observations', pad.l + 16, legY);
  const obsW = ctx.measureText('observations').width;

  // Black — estimates
  const blkX = pad.l + 16 + obsW + 16;
  ctx.strokeStyle = '#222'; ctx.lineWidth = tickW;
  ctx.beginPath(); ctx.moveTo(blkX, legY - 4); ctx.lineTo(blkX + 12, legY - 4); ctx.stroke();
  ctx.fillStyle = '#555'; ctx.fillText('estimates', blkX + 16, legY);
};
