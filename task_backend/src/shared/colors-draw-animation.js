/**
 * colors-draw-animation.js
 * Bubbling generative animation for the colors tutorial.
 *
 * Sequence (all 5 tutorial observations):
 *   1. Centre observation circle is visible immediately as an empty ring
 *      (white background, gray border) throughout bubbling — a fixed visual
 *      anchor showing "this is where the outcome will land," without giving
 *      away the outcome itself.
 *   2. Small bubbles rise inside the blue/red probability bar (clipped),
 *      proportionally to true_p -- more bubbles on whichever side is
 *      larger, so the bubbling itself visually connects to the bar's own
 *      proportions, not just decorative noise.
 *   3. Bubbling stops; the LOSING bar segment DIMS (fill -> DIM_BLUE/
 *      DIM_RED, plain alpha transparency) while the WINNING segment
 *      keeps its exact original color, untouched -- and the centre
 *      circle's ring fades white/gray -> outcome colour, simultaneously.
 *
 * REPLACED a separate "draw outcome" circle that used to pop in ABOVE the
 * bar and fade to the outcome color (chat history, this session) --
 * explicit feedback: that circle had no visual connection to the bar's
 * own proportions below it, and duplicated an outcome ALREADY shown twice
 * more (centre panel, tracker). Dimming the LOSING segment directly ties
 * the reveal to the actual mechanism being illustrated ("blue is drawn
 * with probability = blue's own share of this bar") instead of adding a
 * fourth, disconnected place to look. An earlier revision of this same
 * fix also BRIGHTENED the winning segment (an HSL-derived variant) --
 * dropped per explicit direction ("doesn't look great"); the winner is
 * now left completely alone, only the loser changes. See urn-colors.js's
 * own module docstring for the removed circle's history and the
 * DIM_BLUE/DIM_RED color definitions (plain rgba alpha, matching this
 * app's existing dimmed-slider convention, not a new one).
 *
 * Outcome colour is always fixed to currentValue (not a live Bernoulli sample).
 */

import { LAYOUT, SAMPLE_BLUE, SAMPLE_RED, DIM_BLUE, DIM_RED } from './urn-colors.js';

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
 * @param {HTMLElement} opts.centerEl   .colors-circle in centre panel
 * @param {number} opts.true_p
 * @param {number} opts.currentValue      1 = blue, -1 = red
 * @param {number} opts.obsNum            1-based observation index
 * @param {() => void} [opts.onReveal]  fired the INSTANT the centre circle/
 *   bar segments BEGIN fading to their outcome-reveal state (not once they
 *   finish) -- use this, not onComplete, for anything that should appear
 *   visually simultaneous with them (e.g. tutorial-tracker.js's
 *   current-slot dot). onComplete fires FADE_MS later, once they're
 *   already fully colored -- wiring a simultaneous reveal to onComplete
 *   instead was a real, reported bug in numbers-draw-animation.js's own
 *   identical hook (chat history); this mirrors that same fix here rather
 *   than reintroducing it for colors.
 * @param {() => void} [opts.onComplete] fired once the fade has fully finished
 * @returns {() => void} cancel
 */
export function startColorsDrawAnimation({
  svgRoot, centerEl, true_p, currentValue, obsNum, onReveal, onComplete,
}) {
  const isBlue      = currentValue === 1;
  const targetColor = isBlue ? SAMPLE_BLUE : SAMPLE_RED;
  // Only the LOSING segment gets a fill change (to its own DIM_* variant
  // -- see urn-colors.js's own comment); the winning one is left
  // completely untouched, keeping its exact original SAMPLE_BLUE/RED.
  const loserDim = isBlue ? DIM_RED : DIM_BLUE;
  const { W, barY, barH } = LAYOUT;
  const midX        = true_p * W;
  const rnd         = makeRng(drawSeed(true_p, obsNum));

  const bubbleLayer = svgRoot.querySelector('#tut-urn-bubbles');
  const barBlue      = svgRoot.querySelector('#tut-urn-bar-blue');
  const barRed       = svgRoot.querySelector('#tut-urn-bar-red');
  const loser        = isBlue ? barRed : barBlue;

  const showEmptyRing = () => {
    if (!centerEl) return;
    centerEl.style.background = '#fff';
    centerEl.style.border = '2px solid #ccc';
    centerEl.style.opacity = '1';
    centerEl.style.transition = '';
  };

  const showFinal = () => {
    if (bubbleLayer) bubbleLayer.innerHTML = '';
    if (loser) {
      loser.setAttribute('fill', loserDim);
      loser.style.transition = '';
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

  if (!bubbleLayer || !barBlue || !barRed || !centerEl) {
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

    requestAnimationFrame(() => {
      if (cancelled) return;
      // Only the loser's fill transitions -- the winner stays exactly as
      // it was built (see module docstring for why this replaced the
      // old draw-outcome circle, and why the winner is untouched rather
      // than also brightened).
      loser.style.transition = `fill ${FADE_MS}ms ease`;
      centerEl.style.transition = `background ${FADE_MS}ms ease, border-color ${FADE_MS}ms ease`;
      loser.setAttribute('fill', loserDim);
      centerEl.style.background   = targetColor;
      centerEl.style.borderColor  = targetColor;
      onReveal?.();
      phaseTimer = setTimeout(() => {
        if (!cancelled) onComplete?.();
      }, FADE_MS);
    });
  };

  // centre visible as an empty ring throughout bubbling; bar segments stay
  // at their base (unrecolored) fill until resolve
  showEmptyRing();

  spawnBubble();
  spawnTimer = setInterval(() => { if (!cancelled) spawnBubble(); }, SPAWN_EVERY);
  rafId = requestAnimationFrame(tick);
  phaseTimer = setTimeout(resolveDraw, BUBBLE_MS);

  return cancel;
}
