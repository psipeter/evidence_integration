/**
 * plugin-iti-clock.js
 * jsPsych 8 plugin — circular countdown clock between trials.
 * If timed_out=false: auto-advances once duration_ms elapses (normal ITI).
 * If timed_out=true: shows a "Too slow / X remaining" message, THEN a
 * manual "Repeat" button — deliberately does NOT auto-advance. See the
 * timed_out branch below for why (closes a tab-visibility exploit/failure
 * mode together with observation-timeout-clock.js).
 *
 * Used to also support a "popup distractor" mode (spawning random obs
 * popups during the ITI) plus an iti_condition-driven longer-ITI mode --
 * removed entirely (chat history): this study has no distractor
 * manipulation, so distractor_type/iti_condition/is_colors were all
 * dead weight once that mode could never actually fire. See git history
 * to restore if a future study needs it again.
 */

const info = {
  name: 'iti-clock',
  parameters: {
    duration_ms: { type: 'INT',     default: 1000 },
    color:       { type: 'STRING',  default: '#2563eb' },
    radius:      { type: 'INT',     default: 60 },
    timed_out:          { type: 'BOOLEAN', default: false },
    timeouts_remaining: { type: 'INT',     default: 3 },
  },
};

class ItiClockPlugin {
  constructor(jsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';

    const { duration_ms, color, radius, timed_out, timeouts_remaining } = trial;
    const strokeWidth = 5;
    const size = (radius + strokeWidth) * 2;
    const cx   = size / 2;
    const cy   = size / 2;

    // Guard: once the trial has finished, don't touch the DOM
    let active = true;
    let rafId  = null;
    let timeoutId = null;

    const finish = () => {
      if (!active) return;
      active = false;
      if (rafId)     { cancelAnimationFrame(rafId); rafId = null; }
      if (timeoutId) { clearTimeout(timeoutId); timeoutId = null; }
      this.jsPsych.finishTrial({ screen: 'iti', duration_ms, timed_out });
    };

    const showClock = () => {
      if (!active) return;
      display_el.innerHTML = `
        <div class="iti-wrap">
          <canvas id="iti-canvas" width="${size}" height="${size}"></canvas>
        </div>`;

      const canvas = display_el.querySelector('#iti-canvas');
      const ctx    = canvas.getContext('2d');
      const start  = performance.now();

      const draw = (now) => {
        if (!active) return;
        const fraction = Math.min((now - start) / duration_ms, 1);
        const endAngle = -Math.PI / 2 + fraction * 2 * Math.PI;

        ctx.clearRect(0, 0, size, size);

        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
        ctx.strokeStyle = '#ddd';
        ctx.lineWidth   = strokeWidth;
        ctx.stroke();

        if (fraction > 0) {
          ctx.beginPath();
          ctx.arc(cx, cy, radius, -Math.PI / 2, endAngle);
          ctx.strokeStyle = color;
          ctx.lineWidth   = strokeWidth;
          ctx.lineCap     = 'round';
          ctx.stroke();
        }

        if (fraction < 1) {
          rafId = requestAnimationFrame(draw);
        } else {
          finish();
        }
      };

      rafId = requestAnimationFrame(draw);
    };

    if (timed_out) {
      // Fade in(100ms) → out(1000ms) → in(2000ms) → hold, THEN a manual "Repeat"
      // button — deliberately NOT an auto-advancing clock like the normal ITI
      // below. If this screen auto-advanced, a stray tab-switch (which now
      // immediately consumes one timeout via observation-timeout-clock.js's
      // visibilitychange handling) could cascade into a second, third
      // automatic timeout while nobody's looking, forcing early-exit before
      // the participant even regains control. Requiring a click means the
      // WORST case from any single stray focus loss is exactly one consumed
      // timeout, then an indefinite, fully inert wait for the participant to
      // return and press Repeat — never more than that automatically.
      const FADE = 800;   // matches transition duration
      display_el.innerHTML = `
        <div class="iti-wrap" style="flex-direction:column;gap:1.2rem;">
          <span style="font-size:3rem;font-weight:bold;color:#ef4444;">
            Too slow
          </span>
          <span id="too-slow-pulse" data-timeouts-remaining="${timeouts_remaining}" style="
            font-size:2rem;font-style:italic;color:#555;
            opacity:0;transition:opacity ${FADE}ms ease;">
            ${timeouts_remaining} timeout${timeouts_remaining === 1 ? '' : 's'} remaining
          </span>
          <button id="repeat-btn" class="jspsych-btn" disabled style="
            font-size:1.4rem;padding:0.8rem 3rem;min-width:180px;
            opacity:0;transition:opacity ${FADE}ms ease;">
            Repeat
          </button>
        </div>`;
      const msgEl = display_el.querySelector('#too-slow-pulse');
      const btnEl = display_el.querySelector('#repeat-btn');
      if (btnEl) btnEl.addEventListener('click', finish);
      setTimeout(() => { if (msgEl && active) msgEl.style.opacity = '1'; }, 100);        // fade in
      setTimeout(() => { if (msgEl && active) msgEl.style.opacity = '0'; }, 100+FADE+200); // fade out
      setTimeout(() => {
        if (!active) return;
        if (msgEl) msgEl.style.opacity = '1';                 // fade in
        if (btnEl) { btnEl.style.opacity = '1'; btnEl.disabled = false; }
      }, 100+FADE*2+400);
    } else {
      showClock();
    }
  }
}

ItiClockPlugin.info = info;
export default ItiClockPlugin;
