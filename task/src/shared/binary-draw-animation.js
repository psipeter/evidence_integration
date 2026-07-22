/**
 * binary-draw-animation.js
 * Bubbling generative animation for the binary tutorial.
 *
 * Sequence (all 5 tutorial observations):
 *   1. Centre observation circle is visible immediately as an empty ring
 *      (white background, gray border) throughout bubbling — a fixed visual
 *      anchor showing "this is where the outcome will land," without giving
 *      away the outcome itself.
 *   2. Small bubbles rise inside the blue/red probability bar (clipped). The
 *      draw circle above the bar stays fully invisible during this phase.
 *   3. Bubbling stops; the draw circle appears (pops in, white) at its
 *      already-known final position, and both it and the centre circle's
 *      ring fade white/gray → outcome colour simultaneously.
 *
 * Outcome colour is always fixed to currentValue (not a live Bernoulli sample).
 */

import { LAYOUT, SAMPLE_BLUE, SAMPLE_RED } from './urn-binary.js';

const BUBBLE_MS   = 1050;
export const FADE_MS = 380;
const SPAWN_EVERY = 45;   // ms between bubble spawns
const BUBBLE_R    = 3.5;
const BUBBLE_LIFE = 420;  // ms

const makeRng = (seed) => {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
};

const drawSeed = (trueP, obsNum) =>
  ((obsNum * 9973 + Math.round(trueP * 10000)) >>> 0) + 12345;

/**
 * @param {object} opts
 * @param {SVGElement} opts.svgRoot       root <svg> element
 * @param {HTMLElement} opts.centerEl   .binary-circle in centre panel
 * @param {number} opts.true_p
 * @param {number} opts.currentValue      1 = blue, -1 = red
 * @param {number} opts.obsNum            1-based observation index
 * @param {() => void} [opts.onReveal]  fired the INSTANT the centre circle/
 *   draw circle BEGIN fading to their outcome color (not once they finish)
 *   -- use this, not onComplete, for anything that should appear visually
 *   simultaneous with them (e.g. tutorial-tracker.js's current-slot dot).
 *   onComplete fires FADE_MS later, once they're already fully colored --
 *   wiring a simultaneous reveal to onComplete instead was a real,
 *   reported bug in continuous-draw-animation.js's own identical hook
 *   (chat history); this mirrors that same fix here rather than
 *   reintroducing it for binary.
 * @param {() => void} [opts.onComplete] fired once the fade has fully finished
 * @returns {() => void} cancel
 */
export function startBinaryDrawAnimation({
  svgRoot, centerEl, true_p, currentValue, obsNum, onReveal, onComplete,
}) {
  const isBlue      = currentValue === 1;
  const targetColor = isBlue ? SAMPLE_BLUE : SAMPLE_RED;
  const { W, barY, barH, drawR, drawGap } = LAYOUT;
  const midX        = true_p * W;
  const drawCy      = barY - drawGap - drawR;
  const rnd         = makeRng(drawSeed(true_p, obsNum));

  const bubbleLayer = svgRoot.querySelector('#tut-urn-bubbles');
  const drawCircle  = svgRoot.querySelector('#tut-urn-draw');

  // Final x position is determined now (currentValue is already known) so
  // the draw circle can appear at its real spot once it pops in after
  // bubbling, with no repositioning.
  const drawX = isBlue
    ? 10 + rnd() * Math.max(midX - 20, 20)
    : midX + 10 + rnd() * Math.max(W - midX - 20, 20);

  const showEmptyRing = () => {
    if (!centerEl) return;
    centerEl.style.background = '#fff';
    centerEl.style.border = '2px solid #ccc';
    centerEl.style.opacity = '1';
    centerEl.style.transition = '';
  };

  const showFinal = () => {
    if (bubbleLayer) bubbleLayer.innerHTML = '';
    if (drawCircle) {
      drawCircle.setAttribute('cx', drawX.toFixed(1));
      drawCircle.setAttribute('cy', String(drawCy));
      drawCircle.setAttribute('fill', targetColor);
      drawCircle.setAttribute('stroke', targetColor);
      drawCircle.setAttribute('stroke-width', '1.5');
      drawCircle.style.opacity = '1';
      drawCircle.style.transition = '';
    }
    if (centerEl) {
      centerEl.style.opacity = '1';
      centerEl.style.background = targetColor;
      centerEl.style.border = 'none';
      centerEl.style.transition = '';
    }
    onReveal?.();
    onComplete?.();
  };

  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    showFinal();
    return () => {};
  }

  if (!bubbleLayer || !drawCircle || !centerEl) {
    showFinal();
    return () => {};
  }

  let cancelled = false;
  let rafId     = null;
  let spawnTimer = null;
  let phaseTimer = null;
  const bubbles  = [];

  const cancel = () => {
    cancelled = true;
    if (rafId) cancelAnimationFrame(rafId);
    if (spawnTimer) clearInterval(spawnTimer);
    if (phaseTimer) clearTimeout(phaseTimer);
    bubbleLayer.innerHTML = '';
  };

  const spawnBubble = () => {
    const inBlue = rnd() < true_p;
    const xMin = inBlue ? 2 : midX + 2;
    const xMax = inBlue ? midX - 2 : W - 2;
    if (xMax <= xMin) return;
    const x  = xMin + rnd() * (xMax - xMin);
    const y  = barY + barH - BUBBLE_R - rnd() * (barH - BUBBLE_R * 2);
    const el = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    el.setAttribute('cx', x.toFixed(1));
    el.setAttribute('cy', y.toFixed(1));
    el.setAttribute('r', String(BUBBLE_R));
    el.setAttribute('fill', '#ffffff');
    el.setAttribute('opacity', '0.85');
    bubbleLayer.appendChild(el);
    bubbles.push({ el, born: performance.now(), x, y0: y });
  };

  const tick = (now) => {
    if (cancelled) return;
    for (let i = bubbles.length - 1; i >= 0; i--) {
      const b = bubbles[i];
      const age = now - b.born;
      if (age > BUBBLE_LIFE) {
        b.el.remove();
        bubbles.splice(i, 1);
        continue;
      }
      const t = age / BUBBLE_LIFE;
      const y = b.y0 - t * (barH * 0.55);
      const a = 0.85 * (1 - t);
      b.el.setAttribute('cy', y.toFixed(1));
      b.el.setAttribute('opacity', a.toFixed(2));
    }
    rafId = requestAnimationFrame(tick);
  };

  const resolveDraw = () => {
    if (cancelled) return;
    if (spawnTimer) clearInterval(spawnTimer);   // stop spawning new bubbles
    bubbleLayer.innerHTML = '';
    bubbles.length = 0;
    if (rafId) cancelAnimationFrame(rafId);

    // Draw circle pops in now (was fully invisible during bubbling), white,
    // at its already-known final position.
    drawCircle.setAttribute('cx', drawX.toFixed(1));
    drawCircle.setAttribute('cy', String(drawCy));
    drawCircle.setAttribute('fill', '#fff');
    drawCircle.setAttribute('stroke', '#fff');
    drawCircle.style.opacity = '1';
    drawCircle.style.transition = '';

    requestAnimationFrame(() => {
      if (cancelled) return;
      drawCircle.style.transition = `fill ${FADE_MS}ms ease, stroke ${FADE_MS}ms ease`;
      centerEl.style.transition   = `background ${FADE_MS}ms ease, border-color ${FADE_MS}ms ease`;
      drawCircle.setAttribute('fill', targetColor);
      drawCircle.setAttribute('stroke', targetColor);
      centerEl.style.background   = targetColor;
      centerEl.style.borderColor  = targetColor;
      onReveal?.();
      phaseTimer = setTimeout(() => {
        if (!cancelled) onComplete?.();
      }, FADE_MS);
    });
  };

  // centre visible as an empty ring throughout bubbling; draw circle stays
  // fully invisible until it pops in at resolve
  showEmptyRing();
  drawCircle.style.opacity = '0';

  spawnBubble();
  spawnTimer = setInterval(() => { if (!cancelled) spawnBubble(); }, SPAWN_EVERY);
  rafId = requestAnimationFrame(tick);
  phaseTimer = setTimeout(resolveDraw, BUBBLE_MS);

  return cancel;
}
