/**
 * plugin-tutorial-intro-continuous.js
 *
 * Each box contains real content (invisible, for sizing) + absolute overlay.
 * SVG height matches left panel height via ResizeObserver.
 * Axis labels hidden until box 0 clicked; slider hidden until box 2 clicked.
 */

import { normalPDF } from './draw-performance.js';
import { buildSliderHTML, initSlider } from './slider.js';

const GOAL_COLOR   = '#2563eb';
const SAMPLE_COLOR = '#ef4444';
const DIST_COLOR   = '#16a34a';

const info = {
  name: 'tutorial-intro-continuous',
  parameters: {
    example_value: { type: 'INT',   default: 23 },
    true_mean:     { type: 'FLOAT', default: 20 },
    true_std:      { type: 'FLOAT', default: 20 },
  },
};

// Build SVG given pixel dimensions — normalises all coordinates to W x H.
const buildDistSVG = (mu, sigma, currentValue) => {
  const W = 220, H = 160;
  const xMin = 0, xMax = 100;
  const pad  = { l: 20, r: 20, t: 26, b: 44 };
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;
  const axisY = pad.t + plotH;
  const tickH = 12;
  const xPos  = x => pad.l + (x - xMin) / (xMax - xMin) * plotW;
  const peakP = normalPDF(mu, mu, sigma);
  const yPos  = p => pad.t + plotH - (p / peakP) * plotH * 0.88;
  const steps = 200;
  const curvePts = Array.from({ length: steps + 1 }, (_, i) => {
    const x = xMin + (i / steps) * (xMax - xMin);
    return `${xPos(x).toFixed(2)},${yPos(normalPDF(x, mu, sigma)).toFixed(2)}`;
  }).join(' ');
  const fillPts = curvePts
    + ` ${xPos(xMax).toFixed(2)},${axisY} ${xPos(xMin).toFixed(2)},${axisY}`;

  return `<svg id="tut-svg" viewBox="0 0 ${W} ${H}" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
    <g id="tut-svg-axis-labels" style="opacity:0;">
    <line x1="${pad.l}" y1="${axisY}" x2="${pad.l+plotW}" y2="${axisY}"
          stroke="#bbb" stroke-width="1"/>
      <text x="${pad.l}" y="${axisY+14}" text-anchor="middle"
            font-family="Arial" font-size="13" fill="#999">0</text>
      <text x="${pad.l+plotW}" y="${axisY+14}" text-anchor="middle"
            font-family="Arial" font-size="13" fill="#999">100</text>
    </g>
    <g id="tut-svg-dist" style="opacity:0;">
      <polygon points="${fillPts}" fill="rgba(22,163,74,0.15)" stroke="none"/>
      <polyline points="${curvePts}" fill="none" stroke="${DIST_COLOR}"
                stroke-width="2" stroke-linejoin="round"/>
    </g>
    <g id="tut-svg-mean" style="opacity:0;">
      <line x1="${xPos(mu).toFixed(2)}" y1="${pad.t}"
            x2="${xPos(mu).toFixed(2)}" y2="${axisY}"
            stroke="${GOAL_COLOR}" stroke-width="2"
            stroke-dasharray="5 3" stroke-linecap="round"/>
      <text x="${xPos(mu).toFixed(2)}" y="${pad.t-12}"
            text-anchor="middle" font-family="Arial" font-size="14"
            font-weight="bold" fill="${GOAL_COLOR}">???</text>
    </g>
    <g id="tut-svg-obs" style="opacity:0;">
      <line x1="${xPos(currentValue).toFixed(2)}" y1="${axisY}"
            x2="${xPos(currentValue).toFixed(2)}" y2="${axisY+tickH}"
            stroke="${SAMPLE_COLOR}" stroke-width="3" stroke-linecap="round"/>
      <text x="${xPos(currentValue).toFixed(2)}" y="${axisY+tickH+12}"
            text-anchor="middle" font-family="Arial" font-size="14"
            font-weight="bold" fill="${SAMPLE_COLOR}">${currentValue}</text>
    </g>
  </svg>`;
};

const makeBox = (id, realHTML, isActive) => `
  <p id="${id}" class="practice-info-block"
     style="position:relative;${isActive ? 'cursor:pointer;' : ''}">
    <span id="${id}-real" style="visibility:hidden;">${realHTML}</span>
    <span id="${id}-overlay" style="
      position:absolute;inset:0;
      display:${isActive ? 'flex' : 'none'};
      align-items:center;justify-content:center;
      color:#222;font-weight:bold;">
      Click to reveal
    </span>
  </p>`;

class TutorialIntroContinuousPlugin {
  constructor(jsPsych) { this.jsPsych = jsPsych; }

  trial(display_el, trial) {
    const self = this;
    document.body.style.backgroundColor = '#f5f5f5';
    const { example_value, true_mean, true_std } = trial;

    const BOX0 = `Numbers are drawn one-at-a-time from a
      <span style="color:${DIST_COLOR};font-weight:bold;">hidden distribution</span>.`;
    const BOX1 = `Goal: estimate the
      <span style="color:${GOAL_COLOR};font-weight:bold;">true mean</span>
      of that distribution.`;
    const BOX2 = `After each <span style="color:${SAMPLE_COLOR};font-weight:bold;">observation</span>, move the slider to update your estimate.`;

    display_el.innerHTML = `
      <div class="tutorial-title">Tutorial</div>
      <div id="tut-obs-circles" style="visibility:hidden;text-align:center;margin-bottom:0.3rem;">
        <span id="tut-circle-0" class="obs-circle obs-circle-filled">&#9679;</span>
        <span id="tut-circle-1" class="obs-circle">&#9675;</span>
        <span id="tut-circle-2" class="obs-circle">&#9675;</span>
        <span id="tut-circle-3" class="obs-circle">&#9675;</span>
        <span id="tut-circle-4" class="obs-circle">&#9675;</span>
      </div>
      <div class="practice-wrap">
        <div class="practice-top-row">

          <div id="tut-left-panel" class="practice-panel">
            ${makeBox('tut-box-0', BOX0, true)}
            ${makeBox('tut-box-1', BOX1, false)}
            ${makeBox('tut-box-2', BOX2, false)}
          </div>

          <div class="practice-panel practice-panel-centre">
            <div id="tut-centre-number" class="stimulus-number"
                 style="color:${SAMPLE_COLOR};opacity:0;">${example_value}</div>
          </div>

          <div class="practice-panel practice-panel-right">
            <div id="dist-svg" class="dist-canvas" style="line-height:0;"></div>
          </div>

        </div>

        <div id="tut-slider-wrap" style="visibility:hidden;">
          ${buildSliderHTML({ unset: true, initPos: 0, showValue: true })}
          <button id="submit-btn" class="jspsych-btn" disabled
                  style="font-size:1.1rem;padding:0.6rem 2.5rem;min-width:160px;">
            Submit
          </button>
        </div>

      </div>`;

    // ── Draw SVG — scales to fill right panel via viewBox + width/height 100% ──
    display_el.querySelector('#dist-svg').innerHTML =
      buildDistSVG(true_mean, true_std, example_value);

    const sliderWrap = display_el.querySelector('#tut-slider-wrap');

    const jsPsych = self.jsPsych;
    const activateSlider = () => {
      sliderWrap.style.visibility = 'visible';
      initSlider(display_el, {
        unset: true,
        showValue: true,
        ghostPos: null,
        onFinish: (response) => {
          jsPsych.finishTrial({ response, timed_out: false });
        },
      });
    };

    const revealBox = (id) => {
      display_el.querySelector(`#${id}-real`).style.visibility = 'visible';
      display_el.querySelector(`#${id}-overlay`).style.display = 'none';
      display_el.querySelector(`#${id}`).style.cursor = 'default';
    };

    const activateBox = (id, onClickFn) => {
      const overlay = display_el.querySelector(`#${id}-overlay`);
      overlay.style.display = 'flex';
      display_el.querySelector(`#${id}`).style.cursor = 'pointer';
      overlay.addEventListener('click', onClickFn, { once: true });
    };

    const onBox0 = () => {
      revealBox('tut-box-0');
      // Axis labels + dist curve appear together
      const svg = display_el.querySelector('#dist-svg');
      svg.querySelector('#tut-svg-axis-labels').style.opacity = '1';
      svg.querySelector('#tut-svg-dist').style.opacity = '1';
      activateBox('tut-box-1', onBox1);
    };
    const onBox1 = () => {
      revealBox('tut-box-1');
      display_el.querySelector('#dist-svg').querySelector('#tut-svg-mean').style.opacity = '1';
      activateBox('tut-box-2', onBox2);
    };
    const onBox2 = () => {
      revealBox('tut-box-2');
      const svg = display_el.querySelector('#dist-svg');
      svg.querySelector('#tut-svg-obs').style.opacity = '1';
      display_el.querySelector('#tut-centre-number').style.opacity = '1';
      display_el.querySelector('#tut-obs-circles').style.visibility = 'visible';
      activateSlider();
    };

    activateBox('tut-box-0', onBox0);
  }
}

TutorialIntroContinuousPlugin.info = info;
export default TutorialIntroContinuousPlugin;
