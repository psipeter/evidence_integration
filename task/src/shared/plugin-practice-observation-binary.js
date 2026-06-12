/**
 * plugin-practice-observation-binary.js
 * Tutorial observation for the binary (Bernoulli) task.
 * Shows coloured circle instead of number, pie chart instead of distribution.
 * Slider has blue/red gradient fill and ball endpoint icons.
 */

const GOAL_COLOR   = '#2563eb';  // blue
const SAMPLE_BLUE  = '#2563eb';
const SAMPLE_RED   = '#ef4444';
const DIST_COLOR   = '#16a34a';  // green (pie border)

const info = {
  name: 'practice-observation-binary',
  parameters: {
    value:          { type: 'INT',     default: 1    }, // 1=blue, 0=red
    obs_num:        { type: 'INT',     default: 1    },
    n_obs:          { type: 'INT',     default: 5    },
    true_p:         { type: 'FLOAT',   default: 0.6  }, // true probability of blue
    slider_default: { type: 'STRING',  default: 'none' },
    init_pos:       { type: 'INT',     default: 50   },
    show_value:     { type: 'BOOLEAN', default: true  },
  },
};

// Build a 5x4 urn SVG: 20 dots coloured by probability p.
// One dot of the matching colour is highlighted with a ring to show current obs.
const buildUrnSVG = (p, currentValue, obsNum) => {
  const COLS = 5, ROWS = 4, N = COLS * ROWS;
  const r    = 13;   // dot radius
  const gap  = 8;    // gap between dots
  const step = r * 2 + gap;
  const pad  = 16;
  const W    = COLS * step + pad * 2 - gap;
  const H    = ROWS * step + pad * 2 - gap;

  // Deterministically assign colours: first round(p*N) dots blue, rest red.
  // Shuffle with a fixed seed so layout looks natural but is reproducible.
  const nBlue = Math.round(p * N);
  let colours = [
    ...Array(nBlue).fill(SAMPLE_BLUE),
    ...Array(N - nBlue).fill(SAMPLE_RED),
  ];
  // Fisher-Yates with fixed seed
  let seed = 12345;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
  for (let i = colours.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [colours[i], colours[j]] = [colours[j], colours[i]];
  }

  // Pick a different dot of the matching colour each observation
  // obsNum is passed in; we cycle through all matching indices
  const matchColor   = currentValue === 1 ? SAMPLE_BLUE : SAMPLE_RED;
  const matchIndices = colours.reduce((acc, c, i) => { if (c === matchColor) acc.push(i); return acc; }, []);
  const matchIdx     = matchIndices[(obsNum - 1) % matchIndices.length];

  let dots = '';
  colours.forEach((col, i) => {
    const row = Math.floor(i / COLS);
    const cx  = pad + (i % COLS) * step + r;
    const cy  = pad + row * step + r;
    const isHighlight = (i === matchIdx && currentValue !== null);

    dots += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${col}"
               stroke="#fff" stroke-width="1.5" />`;
    if (isHighlight) {
      dots += `<circle cx="${cx}" cy="${cy}" r="${r + 5}"
                 fill="none" stroke="#222" stroke-width="2.5" />`;
    }
  });

  return `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">${dots}</svg>`;
};

// Update the slider gradient fill
const updateSliderFill = (slider) => {
  const pct = (slider.value - slider.min) / (slider.max - slider.min) * 100;
  slider.style.setProperty('--pct', pct + '%');
};

class PracticeObservationBinaryPlugin {
  constructor(jsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    if (document.activeElement && document.activeElement !== document.body) {
      document.activeElement.blur();
    }

    const { value, obs_num, n_obs, true_p,
            slider_default, init_pos, show_value } = trial;
    const unset   = slider_default === 'none';
    const ballCol = value === 1 ? SAMPLE_BLUE : SAMPLE_RED;

    display_el.innerHTML = `
      <div class="tutorial-title">Tutorial &nbsp;·&nbsp; Observation ${obs_num} / ${n_obs}</div>

      <div class="practice-wrap">
        <div class="practice-top-row">

          <!-- LEFT -->
          <div class="practice-panel">
            <p class="practice-info-block">
              <span style="color:${SAMPLE_BLUE};font-weight:bold;">Blue</span> or
              <span style="color:${SAMPLE_RED};font-weight:bold;">red</span> balls
              are drawn one-at-a-time based on a
              <span style="color:${DIST_COLOR};font-weight:bold;">hidden probability</span>.
            </p>
            <p class="practice-info-block">
              Goal: estimate the
              <span style="color:${DIST_COLOR};font-weight:bold;">probability</span>
              of drawing a
              <span style="color:${SAMPLE_BLUE};font-weight:bold;">blue</span> ball.
            </p>
            <p class="practice-info-block">
              After each observation,
              move the slider to update your estimate and press
              <strong>Submit</strong> <kbd class="key-badge">space</kbd>.
            </p>
          </div>

          <!-- CENTRE: coloured circle -->
          <div class="practice-panel practice-panel-centre">
            <div id="stimulus-display" class="binary-circle"
              style="background:${ballCol};"></div>
          </div>

          <!-- RIGHT: pie chart -->
          <div class="practice-panel practice-panel-right">
            <div id="pie-svg" class="urn-canvas" style="line-height:0;"></div>
          </div>

        </div>

        <!-- Slider with blue/red gradient and ball icons -->
        <div class="binary-slider-section">
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

      <div style="text-align:center;margin-top:0.5rem;">
        <button id="submit-btn" class="jspsych-btn"
          ${unset ? 'disabled' : ''}
          style="font-size:1.1rem;padding:0.6rem 2.5rem;min-width:160px;">
          Submit
        </button>
      </div>`;

    display_el.querySelector('#pie-svg').innerHTML = buildUrnSVG(true_p, value, obs_num);

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
        if (!btn.disabled) { e.preventDefault(); finish(); }
      });
    };

    requestAnimationFrame(() => requestAnimationFrame(attachListeners));

    const spaceHandler = (e) => {
      if (e.code === 'Space' && !btn.disabled) { e.preventDefault(); finish(); }
    };
    document.addEventListener('keydown', spaceHandler);
  }
}

PracticeObservationBinaryPlugin.info = info;
export default PracticeObservationBinaryPlugin;
