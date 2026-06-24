/**
 * plugin-observation.js
 * jsPsych 8 — one observation: stimulus number + slider + submit + timeout clock.
 */
import { buildSliderHTML, initSlider } from './slider.js';

const info = {
  name: 'observation',
  parameters: {
    value:          { type: 'INT',     default: 50 },
    trial_num:      { type: 'INT',     default: 1 },
    n_trials:       { type: 'INT',     default: 1 },
    t_obs_ms:       { type: 'INT',     default: 5000 },
    slider_default: { type: 'STRING',  default: 'none' },
    init_pos:       { type: 'INT',     default: 54 },
    show_value:     { type: 'BOOLEAN', default: true },
  },
};

class ObservationPlugin {
  constructor(jsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    if (document.activeElement && document.activeElement !== document.body) {
      document.activeElement.blur();
    }

    const { value, trial_num, n_trials, t_obs_ms,
            slider_default, init_pos, show_value } = trial;
    const trialStart = performance.now();

    // init_pos may be a lazy function — resolve it now
    const resolvedInitPos = typeof init_pos === 'function' ? init_pos() : (init_pos ?? 0);
    const unset = slider_default === 'none';

    display_el.innerHTML = `
      <canvas id="timeout-clock" class="timeout-clock" width="88" height="88"></canvas>
      <div class="obs-wrap">
        <div id="stimulus-display" class="stimulus-number" style="color:#ef4444;">${value}</div>
        ${buildSliderHTML({ unset, initPos: resolvedInitPos, showValue: show_value })}
      </div>
      <div style="text-align:center;">
        <button id="submit-btn" class="jspsych-btn" disabled
          style="font-size:1.6rem;padding:1rem 3.5rem;min-width:200px;">
          Submit
        </button>
      </div>`;

    // ── Timeout + finish ──────────────────────────────────────────────────────
    let rafId    = null;
    let finished = false;

    const finish = (timed_out) => {
      if (finished) return;
      finished = true;
      if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
      document.body.style.backgroundColor = '#f5f5f5';
      const slider   = display_el.querySelector('#response-slider');
      const hadUnset = slider.classList.contains('slider-unset');
      const response = (!timed_out && !hadUnset) ? parseInt(slider.value) : null;
      const rt       = timed_out ? null : Math.round(performance.now() - trialStart);
      this.jsPsych.finishTrial({ response, timed_out, rt });
    };

    // ── Slider ────────────────────────────────────────────────────────────────
    initSlider(display_el, {
      unset,
      showValue: show_value,
      onFinish:  () => finish(false),
    });

    // ── Timeout clock ─────────────────────────────────────────────────────────
    const canvas = display_el.querySelector('#timeout-clock');
    const ctx    = canvas.getContext('2d');
    const size   = canvas.width;
    const cx     = size / 2, cy = size / 2;
    const R      = size / 2 - 5;
    const SW     = 4;
    const start  = performance.now();

    const drawClock = (now) => {
      const fraction = Math.min((now - start) / t_obs_ms, 1);

      if (fraction < 0.6) {
        document.body.style.backgroundColor = '#f5f5f5';
      } else {
        const t = (fraction - 0.6) / 0.4;
        const r = Math.round(245 + t * (254 - 245));
        const g = Math.round(245 + t * (226 - 245));
        const b = Math.round(245 + t * (226 - 245));
        document.body.style.backgroundColor = `rgb(${r},${g},${b})`;
      }

      const color = fraction < 0.6 ? '#aaa'
                  : fraction < 0.85 ? '#f97316'
                  : '#ef4444';

      ctx.clearRect(0, 0, size, size);

      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, 2 * Math.PI);
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth   = SW;
      ctx.stroke();

      const remaining = 1 - fraction;
      if (remaining > 0) {
        ctx.beginPath();
        ctx.arc(cx, cy, R, -Math.PI / 2,
                -Math.PI / 2 + remaining * 2 * Math.PI);
        ctx.strokeStyle = color;
        ctx.lineWidth   = SW;
        ctx.lineCap     = 'round';
        ctx.stroke();
      }

      if (fraction < 1) {
        rafId = requestAnimationFrame(drawClock);
      } else {
        finish(true);
      }
    };

    rafId = requestAnimationFrame(drawClock);
  }
}

ObservationPlugin.info = info;
export default ObservationPlugin;
