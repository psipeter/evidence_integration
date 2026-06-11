/**
 * plugin-iti-clock.js
 * jsPsych 8 plugin — circular countdown clock that auto-advances.
 * If timed_out=true, briefly shows "too slow" before the clock starts.
 */

const info = {
  name: 'iti-clock',
  parameters: {
    duration_ms: { type: 'INT',     default: 1000 },
    color:       { type: 'STRING',  default: '#2563eb' },
    radius:      { type: 'INT',     default: 36 },
    timed_out:   { type: 'BOOLEAN', default: false },
    too_slow_ms: { type: 'INT',     default: 800 },
    is_repeat:   { type: 'BOOLEAN', default: false },
  },
};

class ItiClockPlugin {
  constructor(jsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';

    const { duration_ms, color, radius, timed_out, too_slow_ms } = trial;
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
      const message = trial.is_repeat ? 'too slow — try again' : 'too slow';
      display_el.innerHTML = `
        <div class="iti-wrap">
          <span class="too-slow-text">${message}</span>
        </div>`;
      timeoutId = setTimeout(showClock, too_slow_ms);
    } else {
      showClock();
    }
  }
}

ItiClockPlugin.info = info;
export default ItiClockPlugin;
