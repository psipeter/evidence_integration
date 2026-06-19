import { buildBinarySliderHTML, initBinarySlider } from './slider-binary.js';
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
  const matchColor   = currentValue === 1 ? SAMPLE_BLUE : SAMPLE_RED;  // +1=blue, -1=red
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
      dots += `<circle cx="${cx}" cy="${cy}" r="${r + 3}"
                 fill="none" stroke="#222" stroke-width="2.5" />`;
    }
  });

  // Add label row below grid
  const H_full = H + 30;
  const labelY = H + 20;
  const labelX = W / 2;

  return `<svg viewBox="0 0 ${W} ${H_full}" width="100%" height="100%"
    xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
    <rect x="8" y="8" width="178" height="144" fill="none" stroke="#16a34a" stroke-width="2" rx="4"/>
    ${dots}
    <text x="${labelX - 36}" y="${labelY}" text-anchor="start"
          font-family="Arial" font-size="15" font-weight="bold"
          fill="#222">P(</text>
    <circle cx="${labelX - 14}" cy="${labelY - 4}" r="6" fill="${SAMPLE_BLUE}"/>
    <text x="${labelX - 5}" y="${labelY}" text-anchor="start"
          font-family="Arial" font-size="15" font-weight="bold"
          fill="#222">) = ???</text>
  </svg>`;
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
    const resolvedInitPos = typeof init_pos === 'function' ? init_pos() : (init_pos ?? 50);
    const unset = slider_default === 'none';
    const ballCol = value === 1 ? SAMPLE_BLUE : SAMPLE_RED;  // +1=blue, -1=red

    display_el.innerHTML = `
      <div class="tutorial-title">Tutorial</div>
      <div style="text-align:center;margin-bottom:0.3rem;">
        ${Array.from({length: n_obs}, (_, i) =>
          `<span class="obs-circle ${i < obs_num ? 'obs-circle-binary-filled' : ''}">${i < obs_num ? '&#9679;' : '&#9675;'}</span>`
        ).join('')}
      </div>

      <div class="practice-wrap">
        <div class="practice-top-row">

          <!-- LEFT -->
          <div class="practice-panel">
            <p class="practice-info-block">
              <span style="color:${SAMPLE_BLUE};font-weight:bold;">Blue</span> or
              <span style="color:${SAMPLE_RED};font-weight:bold;">red</span> balls
              are drawn one-at-a-time based on a<br>
              <span style="color:${DIST_COLOR};font-weight:bold;">hidden probability</span>.
            </p>
            <p class="practice-info-block">
              Goal: estimate the <strong>probability</strong>
              of drawing a
              <span style="color:${SAMPLE_BLUE};font-weight:bold;">blue</span> ball.
            </p>
            <p class="practice-info-block">
              After each <span style="color:#888;font-weight:bold;">observation</span>, move the slider to update your estimate.
            </p>
          </div>

          <!-- CENTRE: coloured circle -->
          <div class="practice-panel practice-panel-centre">
            <div id="stimulus-display" class="binary-circle"
              style="background:${ballCol};"></div>
          </div>

          <!-- RIGHT: urn chart -->
          <div class="practice-panel practice-panel-right">
            <div id="pie-svg" class="dist-canvas" style="line-height:0;"></div>
          </div>

        </div>

        ${buildBinarySliderHTML({ unset, initPos: resolvedInitPos, showValue: show_value })}
        <div style="text-align:center;margin-top:0.5rem;">
          <button id="submit-btn" class="jspsych-btn"
            ${unset ? 'disabled' : ''}
            style="font-size:1.1rem;padding:0.6rem 2.5rem;min-width:160px;">
            Submit
          </button>
        </div>
      </div>`;

    display_el.querySelector('#pie-svg').innerHTML = buildUrnSVG(true_p, value, obs_num);

    const jsPsych = this.jsPsych;
    requestAnimationFrame(() => {
      initBinarySlider(display_el, {
        unset,
        showValue: show_value,
        onFinish: (response) => {
          jsPsych.finishTrial({ response, timed_out: false });
        },
      });
    });

  }
}

PracticeObservationBinaryPlugin.info = info;
export default PracticeObservationBinaryPlugin;
