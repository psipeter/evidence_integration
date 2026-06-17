/**
 * plugin-tutorial-intro-binary.js
 *
 * Analogous to plugin-tutorial-intro-continuous.js for the binary task.
 * Three info boxes revealed progressively; each reveals corresponding SVG element.
 *
 * Box 0 click → reveal urn dots (hidden probability)
 * Box 1 click → reveal ??? probability label
 * Box 2 click → reveal highlighted ball + enable slider + show obs circles
 */

const GOAL_COLOR  = '#2563eb';
const SAMPLE_BLUE = '#2563eb';
const SAMPLE_RED  = '#ef4444';
const DIST_COLOR  = '#16a34a';

const info = {
  name: 'tutorial-intro-binary',
  parameters: {
    example_value: { type: 'INT',   default: 1   }, // 1=blue, -1=red
    true_p:        { type: 'FLOAT', default: 0.6 },
    n_obs:         { type: 'INT',   default: 5   },
  },
};

// Build urn SVG with named groups for progressive reveal.
// Groups: tut-urn-dots (box0), tut-urn-label (box1), tut-urn-highlight (box2)
const buildUrnSVG = (p, currentValue) => {
  const COLS = 5, ROWS = 4, N = COLS * ROWS;
  const r = 13, gap = 8, step = r * 2 + gap, pad = 16;
  const W = COLS * step + pad * 2 - gap;
  const H = ROWS * step + pad * 2 - gap + 30; // extra space for label

  const nBlue = Math.round(p * N);
  let colours = [
    ...Array(nBlue).fill(SAMPLE_BLUE),
    ...Array(N - nBlue).fill(SAMPLE_RED),
  ];
  let seed = 12345;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
  for (let i = colours.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [colours[i], colours[j]] = [colours[j], colours[i]];
  }

  const matchColor   = currentValue === 1 ? SAMPLE_BLUE : SAMPLE_RED;
  const matchIndices = colours.reduce((a, c, i) => { if (c === matchColor) a.push(i); return a; }, []);
  const matchIdx     = matchIndices[0];

  let dots = '', highlight = '';
  colours.forEach((col, i) => {
    const cx = pad + (i % COLS) * step + r;
    const cy = pad + Math.floor(i / COLS) * step + r;
    dots += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${col}" stroke="#fff" stroke-width="1.5"/>`;
    if (i === matchIdx) {
      highlight += `<circle cx="${cx}" cy="${cy}" r="${r + 3}"
        fill="none" stroke="#222" stroke-width="2.5"/>`;
    }
  });

  const labelY = ROWS * step + pad * 2 - gap + 20;
  const labelX = W / 2;

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="100%"
    xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
    <g id="tut-urn-dots" style="opacity:0;">
      <rect x="8" y="8" width="178" height="144" fill="none" stroke="#16a34a" stroke-width="2" rx="4"/>
      ${dots}
    </g>
    <g id="tut-urn-label" style="opacity:0;">
      <text x="${labelX - 36}" y="${labelY}" text-anchor="start"
            font-family="Arial" font-size="15" font-weight="bold"
            fill="#222">P(</text>
      <circle cx="${labelX - 14}" cy="${labelY - 4}" r="6" fill="${SAMPLE_BLUE}"/>
      <text x="${labelX - 5}" y="${labelY}" text-anchor="start"
            font-family="Arial" font-size="15" font-weight="bold"
            fill="#222">) = ???</text>
    </g>
    <g id="tut-urn-highlight" style="opacity:0;">${highlight}</g>
  </svg>`;
};

const updateSliderFill = (slider) => {
  const pct = (slider.value - slider.min) / (slider.max - slider.min) * 100;
  slider.style.setProperty('--pct', pct + '%');
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

class TutorialIntroBinaryPlugin {
  constructor(jsPsych) { this.jsPsych = jsPsych; }

  trial(display_el, trial) {
    const self = this;
    document.body.style.backgroundColor = '#f5f5f5';
    const { example_value, true_p, n_obs } = trial;
    const ballCol = example_value === 1 ? SAMPLE_BLUE : SAMPLE_RED;

    const BOX0 = `<span style="color:${SAMPLE_BLUE};font-weight:bold;">Blue</span> or
      <span style="color:${SAMPLE_RED};font-weight:bold;">red</span> balls
      are drawn one-at-a-time based on a<br>
      <span style="color:${DIST_COLOR};font-weight:bold;">hidden probability</span>.`;
    const BOX1 = `Goal: estimate the <strong>probability</strong>
      of drawing a <span style="color:${SAMPLE_BLUE};font-weight:bold;">blue</span> ball.`;
    const BOX2 = `After each <span style="color:#888;font-weight:bold;">observation</span>,
      move the slider to update your estimate.`;

    display_el.innerHTML = `
      <div class="tutorial-title">Tutorial</div>
      <div id="tut-obs-circles" style="visibility:hidden;text-align:center;margin-bottom:0.3rem;">
        <span class="obs-circle obs-circle-binary-filled">&#9679;</span>
        ${Array.from({length: n_obs - 1}, () =>
          '<span class="obs-circle">&#9675;</span>').join('')}
      </div>
      <div class="practice-wrap">
        <div class="practice-top-row">

          <div id="tut-left-panel" class="practice-panel">
            ${makeBox('tut-box-0', BOX0, true)}
            ${makeBox('tut-box-1', BOX1, false)}
            ${makeBox('tut-box-2', BOX2, false)}
          </div>

          <div class="practice-panel practice-panel-centre">
            <div id="tut-ball" class="binary-circle"
                 style="background:${ballCol};opacity:0;"></div>
          </div>

          <div class="practice-panel practice-panel-right">
            <div id="urn-svg" class="dist-canvas" style="line-height:0;"></div>
          </div>

        </div>

        <div id="tut-slider-wrap" style="visibility:hidden;">
          <div class="binary-slider-section">
            <div class="slider-label-float-wrap">
              <div id="slider-float-label" class="slider-float-label"
                   style="display:none;"></div>
            </div>
            <div class="binary-slider-row">
              <div class="binary-ball binary-ball-blue"></div>
              <input type="range" id="response-slider"
                     class="binary-slider slider-unset"
                     min="0" max="100" value="50" step="1" style="--pct:50%">
              <div class="binary-ball binary-ball-red"></div>
            </div>
          </div>
          <div style="text-align:center;margin-top:0.5rem;">
            <button id="submit-btn" class="jspsych-btn" disabled
                    style="font-size:1.1rem;padding:0.6rem 2.5rem;min-width:160px;">
              Submit
            </button>
          </div>
        </div>

      </div>`;

    display_el.querySelector('#urn-svg').innerHTML =
      buildUrnSVG(true_p, example_value);

    const slider     = display_el.querySelector('#response-slider');
    const lbl        = display_el.querySelector('#slider-float-label');
    const btn        = display_el.querySelector('#submit-btn');
    const sliderWrap = display_el.querySelector('#tut-slider-wrap');

    const updateLabel = () => {
      lbl.style.display = 'block';
      lbl.textContent   = slider.value + '%';
      const thumbR  = 3;
      const rect    = slider.getBoundingClientRect();
      const usable  = rect.width - 2 * thumbR;
      const pct     = (slider.value - slider.min) / (slider.max - slider.min);
      const pos     = thumbR + pct * usable;
      const secLeft = slider.parentElement.parentElement.getBoundingClientRect().left;
      lbl.style.left = (rect.left - secLeft + pos) + 'px';
    };

    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      const response = slider.classList.contains('slider-unset')
        ? null : parseInt(slider.value);
      self.jsPsych.finishTrial({ response, timed_out: false });
    };

    const activateSlider = () => {
      sliderWrap.style.visibility = 'visible';
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
      btn.addEventListener('pointerdown', e => {
        if (!btn.disabled) { e.preventDefault(); finish(); }
      });
    };

    const revealBox = (id) => {
      display_el.querySelector(`#${id}-real`).style.visibility = 'visible';
      display_el.querySelector(`#${id}-overlay`).style.display = 'none';
      display_el.querySelector(`#${id}`).style.cursor = 'default';
    };

    const activateBox = (id, fn) => {
      const ov = display_el.querySelector(`#${id}-overlay`);
      ov.style.display = 'flex';
      display_el.querySelector(`#${id}`).style.cursor = 'pointer';
      ov.addEventListener('click', fn, { once: true });
    };

    const svg = () => display_el.querySelector('#urn-svg');

    const onBox0 = () => {
      revealBox('tut-box-0');
      svg().querySelector('#tut-urn-dots').style.opacity = '1';
      activateBox('tut-box-1', onBox1);
    };
    const onBox1 = () => {
      revealBox('tut-box-1');
      svg().querySelector('#tut-urn-label').style.opacity = '1';
      activateBox('tut-box-2', onBox2);
    };
    const onBox2 = () => {
      revealBox('tut-box-2');
      svg().querySelector('#tut-urn-highlight').style.opacity = '1';
      display_el.querySelector('#tut-ball').style.opacity = '1';
      display_el.querySelector('#tut-obs-circles').style.visibility = 'visible';
      activateSlider();
    };

    activateBox('tut-box-0', onBox0);
  }
}

TutorialIntroBinaryPlugin.info = info;
export default TutorialIntroBinaryPlugin;
