/**
 * plugin-practice-observation.js
 * Three-column layout:
 *   Left  — goal/observation text
 *   Centre — identical to main task (number + slider + submit)
 *   Right  — distribution plot
 */

import { normalPDF } from './draw-performance.js';
import { buildSliderHTML, initSlider } from './slider.js';

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

const buildDistSVG = (mu, sigma, currentValue) => {
  const W = 220, H = 160;
  const xMin = 0, xMax = 100;
  const pad  = { l: 20, r: 20, t: 26, b: 44 };
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;
  const axisY = pad.t + plotH;
  const tickH = 12;

  const xPos  = (x) => pad.l + (x - xMin) / (xMax - xMin) * plotW;
  const peakP = normalPDF(mu, mu, sigma);
  const yPos  = (p) => pad.t + plotH - (p / peakP) * plotH * 0.88;

  const steps = 200;
  const curvePoints = Array.from({ length: steps + 1 }, (_, i) => {
    const x = xMin + (i / steps) * (xMax - xMin);
    return `${xPos(x).toFixed(2)},${yPos(normalPDF(x, mu, sigma)).toFixed(2)}`;
  }).join(' ');
  const fillPoints = curvePoints
    + ` ${xPos(xMax).toFixed(2)},${axisY} ${xPos(xMin).toFixed(2)},${axisY}`;

  const curveTopY = pad.t;
  const meanTick = `
    <line x1="${xPos(mu).toFixed(2)}" y1="${curveTopY}"
          x2="${xPos(mu).toFixed(2)}" y2="${axisY}"
          stroke="${GOAL_COLOR}" stroke-width="2"
          stroke-linecap="round"/>
    <text x="${xPos(mu).toFixed(2)}" y="${curveTopY - 12}"
          text-anchor="middle" font-family="Arial" font-size="14"
          font-weight="bold" fill="${GOAL_COLOR}">???</text>`;

  const obsTick = currentValue !== null ? `
    <line x1="${xPos(currentValue).toFixed(2)}" y1="${axisY}"
          x2="${xPos(currentValue).toFixed(2)}" y2="${axisY + tickH}"
          stroke="${SAMPLE_COLOR}" stroke-width="3" stroke-linecap="round"/>
    <text x="${xPos(currentValue).toFixed(2)}" y="${axisY + tickH + 12}"
          text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="${SAMPLE_COLOR}">${currentValue}</text>` : '';

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
    <line x1="${pad.l}" y1="${axisY}" x2="${pad.l + plotW}" y2="${axisY}"
      stroke="#bbb" stroke-width="1"/>
    <text x="${pad.l}" y="${axisY + 14}" text-anchor="middle"
      font-family="Arial" font-size="13" fill="#999">0</text>
    <text x="${pad.l + plotW}" y="${axisY + 14}" text-anchor="middle"
      font-family="Arial" font-size="13" fill="#999">100</text>
    <polygon points="${fillPoints}" fill="rgba(22,163,74,0.15)" stroke="none"/>
    <polyline points="${curvePoints}"
      fill="none" stroke="${DIST_COLOR}" stroke-width="2" stroke-linejoin="round"/>
    ${meanTick}
    ${obsTick}
  </svg>`;
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
    const resolvedInitPos = typeof init_pos === 'function' ? init_pos() : (init_pos ?? 0);
    const unset = slider_default === 'none';

    display_el.innerHTML = `
      <div class="tutorial-title">Tutorial</div>

      <div class="practice-wrap">

        <!-- TOP ROW: three equal panels -->
        <div class="practice-top-row">

          <!-- LEFT: goal text -->
          <div class="practice-panel">
            <p class="practice-info-block"><span>
              Numbers are drawn one-at-a-time from a
              <span style="color:${DIST_COLOR};font-weight:bold;">hidden distribution</span>.
            </span></p>
            <p class="practice-info-block"><span>
              Goal: estimate the
              <span style="color:${GOAL_COLOR};font-weight:bold;">true mean</span>
              of that distribution.
            </span></p>
            <p class="practice-info-block"><span>
              After each <span style="color:${SAMPLE_COLOR};font-weight:bold;">observation</span>, move the slider to update your estimate.
            </span></p>
          </div>

          <!-- CENTRE: big number -->
          <div class="practice-panel practice-panel-centre">
            <div id="stimulus-display" class="stimulus-number"
              style="color:${SAMPLE_COLOR};">${value}</div>
          </div>

          <!-- RIGHT: distribution -->
          <div class="practice-panel practice-panel-right">
            <div id="dist-svg" class="dist-canvas" style="line-height:0;"></div>
          </div>

        </div>

        <!-- BOTTOM: full-width slider + submit -->
        ${buildSliderHTML({ unset, initPos: resolvedInitPos, showValue: show_value })}
        <button id="submit-btn" class="jspsych-btn"
          ${unset ? 'disabled' : ''}
          style="font-size:1.6rem;padding:1rem 3.5rem;min-width:200px;">
          Submit
        </button>

      </div>`;

    // Draw distribution
    display_el.querySelector('#dist-svg').innerHTML = buildDistSVG(true_mean, true_std, value);

    const jsPsych = this.jsPsych;
    // Extra rAF ensures jsPsych has finished painting before slider measures layout
    requestAnimationFrame(() => {
      initSlider(display_el, {
        unset,
        showValue: show_value,
        onFinish: (response) => {
          jsPsych.finishTrial({ response, timed_out: false });
        },
      });
    });
  }
}

PracticeObservationPlugin.info = info;
export default PracticeObservationPlugin;
