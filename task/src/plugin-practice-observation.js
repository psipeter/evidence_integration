/**
 * plugin-practice-observation.js
 * Three-column layout:
 *   Left  — goal/observation text
 *   Centre — identical to main task (number + slider + submit)
 *   Right  — distribution plot
 */

import { normalPDF } from './draw-performance.js';

const GOAL_COLOR   = '#2563eb';   // blue  — true mean / ???
const SAMPLE_COLOR = '#ef4444';   // red   — current observation
const DIST_COLOR   = '#16a34a';   // green — distribution curve

const info = {
  name: 'practice-observation',
  parameters: {
    value:          { type: 'INT',     default: 50 },
    obs_num:        { type: 'INT',     default: 1  },
    n_obs:          { type: 'INT',     default: 5  },
    true_mean:      { type: 'FLOAT',   default: 54 },
    true_std:       { type: 'FLOAT',   default: 10 },
    slider_default: { type: 'STRING',  default: 'none' },
    init_pos:       { type: 'INT',     default: 54 },
    show_value:     { type: 'BOOLEAN', default: true  },
  },
};

const drawDistribution = (canvas, mu, sigma, currentValue) => {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const xMin = 10, xMax = 99;
  const pad  = { l: 20, r: 20, t: 12, b: 36 };
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;

  const xPx     = (x) => pad.l + (x - xMin) / (xMax - xMin) * plotW;
  const peakPDF = normalPDF(mu, mu, sigma);
  const yPx     = (p) => pad.t + plotH - (p / peakPDF) * plotH * 0.88;
  const axisY   = pad.t + plotH;
  const tickH   = 7;

  // Axis
  ctx.beginPath();
  ctx.moveTo(pad.l, axisY); ctx.lineTo(pad.l + plotW, axisY);
  ctx.strokeStyle = '#bbb'; ctx.lineWidth = 1; ctx.stroke();

  // Green filled curve
  ctx.beginPath();
  for (let px = 0; px <= plotW; px++) {
    const x = xMin + (px / plotW) * (xMax - xMin);
    const p = normalPDF(x, mu, sigma);
    px === 0 ? ctx.moveTo(pad.l + px, yPx(p)) : ctx.lineTo(pad.l + px, yPx(p));
  }
  ctx.lineTo(pad.l + plotW, axisY);
  ctx.lineTo(pad.l, axisY);
  ctx.closePath();
  ctx.fillStyle = 'rgba(22,163,74,0.15)'; ctx.fill();

  // Green curve outline
  ctx.beginPath();
  for (let px = 0; px <= plotW; px++) {
    const x = xMin + (px / plotW) * (xMax - xMin);
    const p = normalPDF(x, mu, sigma);
    px === 0 ? ctx.moveTo(pad.l + px, yPx(p)) : ctx.lineTo(pad.l + px, yPx(p));
  }
  ctx.strokeStyle = DIST_COLOR; ctx.lineWidth = 2; ctx.stroke();

  // True mean: blue tick on axis + ??? label
  ctx.beginPath();
  ctx.moveTo(xPx(mu), axisY);
  ctx.lineTo(xPx(mu), axisY + tickH);
  ctx.strokeStyle = GOAL_COLOR; ctx.lineWidth = 2; ctx.stroke();
  ctx.fillStyle = GOAL_COLOR; ctx.font = 'bold 11px Arial'; ctx.textAlign = 'center';
  ctx.fillText('???', xPx(mu), axisY + tickH + 24);

  // Current observation: red tick on axis + value label
  if (currentValue !== null) {
    ctx.beginPath();
    ctx.moveTo(xPx(currentValue), axisY);
    ctx.lineTo(xPx(currentValue), axisY + tickH);
    ctx.strokeStyle = SAMPLE_COLOR; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = SAMPLE_COLOR; ctx.font = 'bold 11px Arial'; ctx.textAlign = 'center';
    ctx.fillText(String(currentValue), xPx(currentValue), axisY + tickH + 12);
  }
};

class PracticeObservationPlugin {
  constructor(jsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    if (document.activeElement && document.activeElement !== document.body) {
      document.activeElement.blur();
    }

    const { value, obs_num, n_obs, true_mean, true_std,
            slider_default, init_pos, show_value } = trial;
    const unset = slider_default === 'none';

    display_el.innerHTML = `
      <div class="practice-counter">
        <div>Practice trial</div>
        <div>Observation ${obs_num} / ${n_obs}</div>
      </div>

      <div class="practice-wrap">

        <!-- TOP ROW: three equal panels -->
        <div class="practice-top-row">

          <!-- LEFT: goal text -->
          <div class="practice-panel">
            <p class="practice-info-block">
              Numbers are drawn one-at-a-time from a
              <span style="color:${DIST_COLOR};font-weight:bold;">hidden distribution</span>.
            </p>
            <p class="practice-info-block">
              Goal: estimate the
              <span style="color:${GOAL_COLOR};font-weight:bold;">true mean</span>
              of that distribution.
            </p>
            <p class="practice-info-block">
              After each <span style="color:${SAMPLE_COLOR};font-weight:bold;">observation</span>,
              move the slider to update your estimate and press
              <strong>Submit</strong> <kbd class="key-badge">space</kbd>.
            </p>
          </div>

          <!-- CENTRE: big number -->
          <div class="practice-panel practice-panel-centre">
            <div id="stimulus-display" class="stimulus-number"
              style="color:${SAMPLE_COLOR};">${value}</div>
          </div>

          <!-- RIGHT: distribution -->
          <div class="practice-panel practice-panel-right">
            <canvas id="dist-canvas" class="dist-canvas" width="220" height="160"></canvas>
          </div>

        </div>

        <!-- BOTTOM: full-width slider + submit (same size as main task) -->
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
              min="10" max="99" value="${init_pos}" step="1"
              class="${unset ? 'slider-unset' : ''}">
            <span class="slider-label">99</span>
          </div>
        </div>

        <button id="submit-btn" class="jspsych-btn"
          ${unset ? 'disabled' : ''}
          style="font-size:1.1rem;padding:0.6rem 2.5rem;min-width:160px;">
          Submit
        </button>

      </div>`;

    // Draw distribution
    drawDistribution(
      display_el.querySelector('#dist-canvas'),
      true_mean, true_std, value
    );

    const slider = display_el.querySelector('#response-slider');
    const lbl    = display_el.querySelector('#slider-float-label');
    const btn    = display_el.querySelector('#submit-btn');

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

    let finished = false;

    const finish = () => {
      if (finished) return;
      finished = true;
      document.body.style.backgroundColor = '#f5f5f5';
      document.removeEventListener('keydown', spaceHandler);
      const hadUnset = slider.classList.contains('slider-unset');
      const response = !hadUnset ? parseInt(slider.value) : null;
      this.jsPsych.finishTrial({ response, timed_out: false });
    };

    const attachListeners = () => {
      if (!unset) updateLabel();
      slider.addEventListener('pointerdown', () => {
        slider.classList.remove('slider-unset');
        btn.disabled = false;
        requestAnimationFrame(updateLabel);
      });
      slider.addEventListener('input', () => {
        slider.classList.remove('slider-unset');
        btn.disabled = false;
        updateLabel();
      });
      btn.addEventListener('pointerdown', (e) => {
        if (!btn.disabled) { e.preventDefault(); finish(); }
      });
    };

    requestAnimationFrame(() => requestAnimationFrame(attachListeners));

    const spaceHandler = (e) => {
      if (e.code === 'Space' && !btn.disabled) {
        e.preventDefault();
        finish();
      }
    };
    document.addEventListener('keydown', spaceHandler);


  }
}

PracticeObservationPlugin.info = info;
export default PracticeObservationPlugin;
