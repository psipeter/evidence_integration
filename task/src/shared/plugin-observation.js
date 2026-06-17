/**
 * plugin-observation.js
 * jsPsych 8 — one observation: stimulus number + slider + submit + timeout clock.
 */

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
    // Hard reset any state from previous screen
    document.body.style.backgroundColor = '#f5f5f5';

    // Blur the jsPsych display container — it grabs focus on every trial
    // start and can swallow early pointer events on the slider
    if (document.activeElement && document.activeElement !== document.body) {
      document.activeElement.blur();
    }

    const { value, trial_num, n_trials, t_obs_ms,
            slider_default, init_pos, show_value } = trial;
    const trialStart = performance.now();

    const unset = slider_default === 'none';

    display_el.innerHTML = `
      <div class="trial-counter">Trial ${trial_num} / ${n_trials}</div>
      <canvas id="timeout-clock" class="timeout-clock" width="52" height="52"></canvas>
      <div class="obs-wrap">
        <div id="stimulus-display" class="stimulus-number">${value}</div>
        <div class="slider-section">
          <div class="slider-label-float-wrap">
            <div id="slider-float-label" class="slider-float-label"
              style="display:${unset ? 'none' : 'block'};">
              ${unset ? '' : init_pos}
            </div>
          </div>
          <div class="slider-wrap">
            <span class="slider-label">10</span>
            <input type="range" id="response-slider"
              min="-100" max="100" value="${init_pos}" step="1"
              class="${unset ? 'slider-unset' : ''}">
            <span class="slider-label">100</span>
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

    // -----------------------------------------------------------------------
    // Label positioning
    // -----------------------------------------------------------------------
    const updateLabel = () => {
      if (!show_value || !lbl) return;
      lbl.style.display = 'block';
      lbl.textContent   = slider.value;
      const thumbR      = 3;
      const rect        = slider.getBoundingClientRect();
      const usable      = rect.width - 2 * thumbR;
      const pct         = (slider.value - slider.min) / (slider.max - slider.min);
      const posInSlider = thumbR + pct * usable;
      const sectionLeft = slider.parentElement.parentElement.getBoundingClientRect().left;
      lbl.style.left    = (rect.left - sectionLeft + posInSlider) + 'px';
    };

    // -----------------------------------------------------------------------
    // Single exit point
    // -----------------------------------------------------------------------
    let rafId    = null;
    let finished = false;

    const finish = (timed_out) => {
      if (finished) return;
      finished = true;
      if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
      document.body.style.backgroundColor = '#f5f5f5';
      const hadUnset = slider.classList.contains('slider-unset');
      const response = (!timed_out && !hadUnset) ? parseInt(slider.value) : null;
      const rt = timed_out ? null : Math.round(performance.now() - trialStart);
      this.jsPsych.finishTrial({ response, timed_out, rt });
    };

    // -----------------------------------------------------------------------
    // Attach slider + button listeners AFTER the browser has painted,
    // avoiding the focus-grab that jsPsych does at trial start
    // -----------------------------------------------------------------------
    // Double-rAF: wait for two frames so jsPsych's focus() call and
    // any resulting repaints have fully settled before attaching listeners.
    // Use pointerdown on button for fastest possible response registration.
    const attachListeners = () => {
      if (!unset) updateLabel();

      // 'input' fires on value change; 'pointerdown' fires on any click.
      // We need both: pointerdown to immediately enable submit and show the
      // thumb on first click (even if value doesn't change), input to update
      // the label as the user drags.
      slider.addEventListener('pointerdown', () => {
        slider.classList.remove('slider-unset');
        btn.disabled = false;
        // Update label on next frame so getBoundingClientRect is accurate
        requestAnimationFrame(updateLabel);
      });
      slider.addEventListener('input', () => {
        slider.classList.remove('slider-unset');
        btn.disabled = false;
        updateLabel();
      });

      btn.addEventListener('pointerdown', (e) => {
        if (!btn.disabled) { e.preventDefault(); finish(false); }
      });
    };

    requestAnimationFrame(() => requestAnimationFrame(attachListeners));


    // -----------------------------------------------------------------------
    // Timeout countdown clock
    // -----------------------------------------------------------------------
    const canvas = display_el.querySelector('#timeout-clock');
    const ctx    = canvas.getContext('2d');
    const size   = canvas.width;
    const cx     = size / 2;
    const cy     = size / 2;
    const R      = size / 2 - 5;
    const SW     = 4;
    const start  = performance.now();

    const drawClock = (now) => {
      const fraction = Math.min((now - start) / t_obs_ms, 1);

      // Background colour shift over the last 40%
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
