/**
 * continuous-draw-animation.js
 * Bubbling generative animation for the continuous tutorial — mirrors
 * binary-draw-animation.js's overall structure, but bubbles drift DOWNWARD
 * from under the Gaussian curve toward the x-axis (rather than rising
 * inside a horizontal bar), matching a "falling sample" metaphor for a
 * continuous draw instead of two discrete regions.
 *
 * Sequence (all 5 tutorial observations):
 *   1. Small bubbles spawn under the curve — x-position weighted by the
 *      Gaussian density via rejection sampling, so bubbles cluster near the
 *      mean, same as the real distribution — and drift straight down to the
 *      x-axis, fading out as they fall.
 *   2. Bubbling stops. The pre-built #tut-svg-obs marker (already positioned
 *      at the exact observed value by distribution-continuous.js — no
 *      randomness needed here, unlike the binary version, since the actual
 *      value is a specific number, not one of two regions) fades in, and the
 *      centre observation number fades in simultaneously.
 */

import { LAYOUT } from './distribution-continuous.js';
import { normalPDF } from './draw-performance-continuous.js';

export const FADE_MS = 1000;
const BUBBLE_MS   = 1050;
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

const drawSeed = (mu, sigma, obsNum) =>
  ((obsNum * 9973 + Math.round(mu * 100) + Math.round(sigma * 37)) >>> 0) + 54321;

/**
 * @param {object} opts
 * @param {SVGElement} opts.svgRoot     root <svg> element
 * @param {HTMLElement} opts.centerEl   .stimulus-number in centre panel
 * @param {number} opts.true_mean
 * @param {number} opts.true_std
 * @param {number} opts.obsNum          1-based observation index
 * @param {() => void} [opts.onReveal]  fired the INSTANT the centre number/
 *   obs-tick BEGIN fading in (not once they finish) -- use this, not
 *   onComplete, for anything that should appear visually simultaneous with
 *   them (e.g. tutorial-tracker.js's current-slot number). onComplete
 *   fires FADE_MS later, once they're already fully visible -- wiring a
 *   simultaneous reveal to onComplete instead was a real, reported bug
 *   (see tutorial-tracker.js/plugin-tutorial-*-continuous.js chat history):
 *   it made the tracker number visibly lag behind the centre number rather
 *   than appear together with it.
 * @param {() => void} [opts.onComplete] fired once the fade-in has fully finished
 * @returns {() => void} cancel
 */
export function startContinuousDrawAnimation({
  svgRoot, centerEl, true_mean, true_std, obsNum, onReveal, onComplete,
}) {
  const { W, H, xMin, xMax, pad } = LAYOUT;
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;
  const axisY = pad.t + plotH;
  const xPos  = (x) => pad.l + (x - xMin) / (xMax - xMin) * plotW;
  const peakP = normalPDF(true_mean, true_mean, true_std);
  const yPos  = (p) => pad.t + plotH - (p / peakP) * plotH * 0.88;

  const rnd = makeRng(drawSeed(true_mean, true_std, obsNum));

  const bubbleLayer = svgRoot.querySelector('#tut-svg-bubbles');
  const obsGroup    = svgRoot.querySelector('#tut-svg-obs');

  const showFinal = () => {
    if (bubbleLayer) bubbleLayer.innerHTML = '';
    if (obsGroup) { obsGroup.style.opacity = '1'; obsGroup.style.transition = ''; }
    if (centerEl) { centerEl.style.opacity = '1'; centerEl.style.transition = ''; }
    onReveal?.();
    onComplete?.();
  };

  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    showFinal();
    return () => {};
  }

  if (!bubbleLayer || !obsGroup || !centerEl) {
    showFinal();
    return () => {};
  }

  let cancelled  = false;
  let rafId      = null;
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

  // Rejection-sample an x weighted by the Gaussian density, so bubbles
  // cluster near the mean the same way real observations would.
  const sampleX = () => {
    for (let i = 0; i < 20; i++) {
      const x = xMin + rnd() * (xMax - xMin);
      const accept = normalPDF(x, true_mean, true_std) / peakP;
      if (rnd() < accept) return x;
    }
    return true_mean; // fallback if rejection sampling didn't converge
  };

  const spawnBubble = () => {
    const xVal = sampleX();
    const x    = xPos(xVal);
    const yTop = yPos(normalPDF(xVal, true_mean, true_std));
    if (yTop >= axisY) return; // no room to fall
    const el = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    el.setAttribute('cx', x.toFixed(1));
    el.setAttribute('cy', yTop.toFixed(1));
    el.setAttribute('r', String(BUBBLE_R));
    el.setAttribute('fill', '#ffffff');
    // A gray stroke so bubbles stay visible against the pale green fill
    // under the curve — plain white was low-contrast there.
    el.setAttribute('stroke', '#94a3b8');
    el.setAttribute('stroke-width', '1');
    el.setAttribute('opacity', '0.85');
    bubbleLayer.appendChild(el);
    bubbles.push({ el, born: performance.now(), x, y0: yTop });
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
      const y = b.y0 + t * (axisY - b.y0);   // drift DOWN to the x-axis
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

    requestAnimationFrame(() => {
      if (cancelled) return;
      obsGroup.style.transition  = `opacity ${FADE_MS}ms ease`;
      centerEl.style.transition  = `opacity ${FADE_MS}ms ease`;
      obsGroup.style.opacity = '1';
      centerEl.style.opacity = '1';
      onReveal?.();
      phaseTimer = setTimeout(() => {
        if (!cancelled) onComplete?.();
      }, FADE_MS);
    });
  };

  // centre number hidden during bubbling; obs marker already starts at
  // opacity 0 by default (see distribution-continuous.js)
  centerEl.style.opacity = '0';

  spawnBubble();
  spawnTimer = setInterval(() => { if (!cancelled) spawnBubble(); }, SPAWN_EVERY);
  rafId = requestAnimationFrame(tick);
  phaseTimer = setTimeout(resolveDraw, BUBBLE_MS);

  return cancel;
}
