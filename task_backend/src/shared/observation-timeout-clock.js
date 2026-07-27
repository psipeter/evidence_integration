/**
 * observation-timeout-clock.js
 * Shared countdown-clock renderer, used by:
 *   - plugin-observation-numbers.js / plugin-observation-colors.js
 *     (real observation timeout, ties into finishTrial via onTimeout)
 *   - plugin-tutorial-observation-numbers.js / -colors.js
 *     (phase E's two decorative demo clocks -- onTimeout is a no-op there,
 *     since tutorial screens have no real deadline; see that file's own
 *     "phase E" docstring)
 *
 * Extracted because this logic was previously duplicated verbatim (just
 * variable-name differences) across the two real per-observation plugins
 * -- the kind of duplication that makes it easy to fix a bug in one copy
 * and forget the other. A third, tutorial-only standalone consumer
 * (plugin-timeout-demo.js) existed for a while but has since been removed
 * entirely (chat history) -- the phased per-observation hint progression's
 * own phase E now covers that same "show what the clock looks like" need
 * inline, without a dedicated screen/plugin.
 *
 * Draws a shrinking countdown ring on a square canvas, tints the page
 * background as time runs low, and calls onTimeout() once the deadline is
 * reached. Driven by requestAnimationFrame — NOT jsPsych.pluginAPI.setTimeout,
 * since it needs to redraw every frame, not just fire once.
 *
 * Also calls onTimeout() IMMEDIATELY if the tab becomes hidden
 * (document.visibilitychange), rather than leaving the countdown to sit
 * frozen until the tab regains focus. This isn't a cosmetic nicety —
 * requestAnimationFrame callbacks are fully SUSPENDED (not just throttled)
 * in a hidden tab, since rAF is tied to the paint cycle and there's nothing
 * to paint when hidden. Without this, switching tabs mid-observation acted
 * as an unintended, unbounded pause: the clock simply stopped advancing
 * until the tab became visible again, and any response given immediately
 * after would carry a badly inflated rt (performance.now() - trialStart
 * naively includes the away-time). Treating hidden-while-active as an
 * immediate timeout closes both the pause exploit and the rt-contamination
 * risk in one place — see plugin-iti-clock.js's timed_out branch for the
 * other half of this fix (requiring a manual click to proceed afterward,
 * so a single stray visibility change can cost at most one timeout, not an
 * unbounded cascade while the tab stays hidden).
 *
 * Returns a `stop()` function; callers must call it when the trial finishes
 * for any other reason (e.g. a real response), to cancel the pending rAF
 * AND remove the visibility listener — otherwise a later, unrelated tab
 * switch during a subsequent trial would have nothing left listening (the
 * listener is removed here), but leaving it un-removed would be a real
 * leak across many trials in a long session regardless.
 */
export function startTimeoutClock(canvas, tObsMs, onTimeout) {
  const ctx  = canvas.getContext('2d');
  const size = canvas.width;
  const cx = size / 2, cy = size / 2;
  const R  = size / 2 - 5, SW = 4;
  const start = performance.now();

  let rafId = null;
  let done  = false;

  const finish = () => {
    if (done) return;
    done = true;
    if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
    document.removeEventListener('visibilitychange', onVisibilityChange);
    onTimeout();
  };

  const onVisibilityChange = () => {
    if (document.hidden) finish();
  };
  document.addEventListener('visibilitychange', onVisibilityChange);

  const draw = (now) => {
    if (done) return;
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
      finish();
    }
  };

  rafId = requestAnimationFrame(draw);

  return function stop() {
    done = true;
    if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
    document.removeEventListener('visibilitychange', onVisibilityChange);
  };
}
