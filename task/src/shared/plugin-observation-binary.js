import { buildBinarySliderHTML, initBinarySlider } from './slider-binary.js';
/**
 * plugin-observation-binary.js
 * Single binary observation: coloured circle + binary slider + timeout clock.
 * Mirrors plugin-observation.js but for binary (Bernoulli) trials.
 */

const SAMPLE_BLUE = '#2563eb';
const SAMPLE_RED  = '#ef4444';

const info = {
  name: 'observation-binary',
  parameters: {
    value:          { type: 'INT',     default: 1     }, // 1=blue, 0=red
    trial_num:      { type: 'INT',     default: 1     },
    n_trials:       { type: 'INT',     default: 1     },
    t_obs_ms:       { type: 'INT',     default: 5000  },
    slider_default: { type: 'STRING',  default: 'none'},
    init_pos:       { type: 'INT',     default: 50    },
    show_value:     { type: 'BOOLEAN', default: true  },
  },
};

class ObservationBinaryPlugin {
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
    const resolvedInitPos = typeof init_pos === 'function' ? init_pos() : (init_pos ?? 50);
    const unset = slider_default === 'none';
    const ballCol  = value === 1 ? SAMPLE_BLUE : SAMPLE_RED;  // +1=blue, -1=red

    display_el.innerHTML = `
      <canvas id="timeout-clock" class="timeout-clock" width="88" height="88"></canvas>
      <div class="obs-wrap">
        <div class="binary-circle" style="background:${ballCol};"></div>
        ${buildBinarySliderHTML({ unset, initPos: resolvedInitPos, showValue: show_value })}
      </div>
      <div style="text-align:center;">
        <button id="submit-btn" class="jspsych-btn"
          ${unset ? 'disabled' : ''}
          style="font-size:1.6rem;padding:1rem 3.5rem;min-width:200px;">
          Submit
        </button>
      </div>`;

    let rafId    = null;
    let finished = false;

    const finish = (timed_out) => {
      if (finished) return;
      finished = true;
      if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
      document.body.style.backgroundColor = '#f5f5f5';
      const slider   = display_el.querySelector('#response-slider');
      const hadUnset = slider.classList.contains('slider-unset') ||
                       slider.classList.contains('slider-last');
      const response = (!timed_out && !hadUnset) ? parseInt(slider.value) : null;
      const rt = timed_out ? null : Math.round(performance.now() - trialStart);
      this.jsPsych.finishTrial({ response, timed_out, rt });
    };

    initBinarySlider(display_el, {
      unset,
      showValue: show_value,
      onFinish: () => finish(false),
    });


    // Timeout clock
    const canvas = display_el.querySelector('#timeout-clock');
    const tCtx   = canvas.getContext('2d');
    const sz = canvas.width, cx = sz / 2, cy = sz / 2;
    const R = sz / 2 - 5, SW = 4;
    const start = performance.now();

    const drawClock = (now) => {
      const fraction = Math.min((now - start) / t_obs_ms, 1);
      if (fraction < 0.6) {
        document.body.style.backgroundColor = '#f5f5f5';
      } else {
        const t = (fraction - 0.6) / 0.4;
        document.body.style.backgroundColor =
          `rgb(${Math.round(245 + t * 9)},${Math.round(245 - t * 19)},${Math.round(245 - t * 19)})`;
      }
      const color = fraction < 0.6 ? '#aaa' : fraction < 0.85 ? '#f97316' : '#ef4444';
      tCtx.clearRect(0, 0, sz, sz);
      tCtx.beginPath(); tCtx.arc(cx, cy, R, 0, 2 * Math.PI);
      tCtx.strokeStyle = '#e5e7eb'; tCtx.lineWidth = SW; tCtx.stroke();
      const rem = 1 - fraction;
      if (rem > 0) {
        tCtx.beginPath();
        tCtx.arc(cx, cy, R, -Math.PI / 2, -Math.PI / 2 + rem * 2 * Math.PI);
        tCtx.strokeStyle = color; tCtx.lineWidth = SW; tCtx.lineCap = 'round';
        tCtx.stroke();
      }
      if (fraction < 1) { rafId = requestAnimationFrame(drawClock); }
      else               { finish(true); }
    };
    rafId = requestAnimationFrame(drawClock);
  }
}

ObservationBinaryPlugin.info = info;
export default ObservationBinaryPlugin;
