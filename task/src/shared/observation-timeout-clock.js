/**
 * observation-timeout-clock.js
 * Shared countdown-clock renderer, used by:
 *   - plugin-observation-continuous.js / plugin-observation-binary.js
 *     (real observation timeout, ties into finishTrial via onTimeout)
 *   - plugin-timeout-demo.js
 *     (tutorial-only demo of the same clock; onTimeout advances the demo's
 *     own internal screen state instead of calling finishTrial)
 *
 * Extracted because this logic was previously duplicated verbatim (just
 * variable-name differences) across all three — the kind of duplication
 * that makes it easy to fix a bug in one copy and forget the others.
 *
 * Draws a shrinking countdown ring on a square canvas, tints the page
 * background as time runs low, and calls onTimeout() once the deadline is
 * reached. Driven by requestAnimationFrame — NOT jsPsych.pluginAPI.setTimeout,
 * since it needs to redraw every frame, not just fire once.
 *
 * Returns a `stop()` function; callers must call it when the trial finishes
 * for any other reason (e.g. a real response), to cancel the pending rAF.
 */
export function startTimeoutClock(canvas, tObsMs, onTimeout) {
  const ctx  = canvas.getContext('2d');
  const size = canvas.width;
  const cx = size / 2, cy = size / 2;
  const R  = size / 2 - 5, SW = 4;
  const start = performance.now();

  let rafId = null;

  const draw = (now) => {
    const fraction = Math.min((now - start) / tObsMs, 1);

    if (fraction < 0.6) {
      document.body.style.backgroundColor = '#f5f5f5';
    } else {
      const t = (fraction - 0.6) / 0.4;
      document.body.style.backgroundColor =
        `rgb(${Math.round(245 + t * 9)},${Math.round(245 - t * 19)},${Math.round(245 - t * 19)})`;
    }

    const color = fraction < 0.6 ? '#aaa'
                : fraction < 0.85 ? '#f97316'
                : '#ef4444';

    ctx.clearRect(0, 0, size, size);
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, 2 * Math.PI);
    ctx.strokeStyle = '#e5e7eb'; ctx.lineWidth = SW; ctx.stroke();

    const remaining = 1 - fraction;
    if (remaining > 0) {
      ctx.beginPath();
      ctx.arc(cx, cy, R, -Math.PI / 2, -Math.PI / 2 + remaining * 2 * Math.PI);
      ctx.strokeStyle = color; ctx.lineWidth = SW; ctx.lineCap = 'round';
      ctx.stroke();
    }

    if (fraction < 1) {
      rafId = requestAnimationFrame(draw);
    } else {
      onTimeout();
    }
  };

  rafId = requestAnimationFrame(draw);

  return function stop() {
    if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
  };
}
