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

const updateSliderFill = (slider) => {
  const pct = (slider.value - slider.min) / (slider.max - slider.min) * 100;
  slider.style.setProperty('--pct', pct + '%');
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
    const unset    = slider_default === 'none';
    const ballCol  = value === 1 ? SAMPLE_BLUE : SAMPLE_RED;

    display_el.innerHTML = `
      <div class="trial-counter">Trial ${trial_num} / ${n_trials}</div>
      <canvas id="timeout-clock" class="timeout-clock" width="52" height="52"></canvas>
      <div class="obs-wrap">
        <div class="binary-circle" style="background:${ballCol};"></div>
        <div class="slider-section">
          <div class="slider-label-float-wrap">
            <div id="slider-float-label" class="slider-float-label"
              style="display:${unset ? 'none' : 'block'};">
              ${unset ? '' : init_pos + '%'}
            </div>
          </div>
          <div class="binary-slider-row">
            <div class="binary-ball binary-ball-blue"></div>
            <input type="range" id="response-slider"
              class="binary-slider ${unset ? 'slider-unset' : ''}"
              min="0" max="100" value="${init_pos}" step="1"
              style="--pct:${unset ? '50' : init_pos}%">
            <div class="binary-ball binary-ball-red"></div>
          </div>
        </div>
      </div>
      <div style="text-align:center;">
        <button id="submit-btn" class="jspsych-btn"
          ${unset ? 'disabled' : ''}
          style="font-size:1.1rem;padding:0.6rem 2.5rem;min-width:160px;">
          Submit
        </button>
      </div>`;

    const slider = display_el.querySelector('#response-slider');
    const lbl    = display_el.querySelector('#slider-float-label');
    const btn    = display_el.querySelector('#submit-btn');

    const updateLabel = () => {
      if (!show_value || !lbl) return;
      lbl.style.display = 'block';
      lbl.textContent   = slider.value + '%';
      const thumbR      = 3;
      const rect        = slider.getBoundingClientRect();
      const usable      = rect.width - 2 * thumbR;
      const pct         = (slider.value - slider.min) / (slider.max - slider.min);
      const posInSlider = thumbR + pct * usable;
      const sectionLeft = slider.parentElement.parentElement.getBoundingClientRect().left;
      lbl.style.left    = (rect.left - sectionLeft + posInSlider) + 'px';
    };

    let rafId    = null;
    let finished = false;

    const finish = (timed_out) => {
      if (finished) return;
      finished = true;
      if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
      document.body.style.backgroundColor = '#f5f5f5';
      document.removeEventListener('keydown', spaceHandler);
      const hadUnset = slider.classList.contains('slider-unset');
      const response = (!timed_out && !hadUnset) ? parseInt(slider.value) : null;
      this.jsPsych.finishTrial({ response, timed_out });
    };

    const attachListeners = () => {
      if (!unset) { updateLabel(); updateSliderFill(slider); }
      slider.addEventListener('pointerdown', () => {
        slider.classList.remove('slider-unset');
        btn.disabled = false;
        requestAnimationFrame(() => { updateLabel(); updateSliderFill(slider); });
      });
      slider.addEventListener('input', () => {
        slider.classList.remove('slider-unset');
        btn.disabled = false;
        updateLabel();
        updateSliderFill(slider);
      });
      btn.addEventListener('pointerdown', (e) => {
        if (!btn.disabled) { e.preventDefault(); finish(false); }
      });
    };

    requestAnimationFrame(() => requestAnimationFrame(attachListeners));

    const spaceHandler = (e) => {
      if (e.code === 'Space' && !btn.disabled) { e.preventDefault(); finish(false); }
    };
    document.addEventListener('keydown', spaceHandler);

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
