/**
 * plugin-timeout-demo.js
 * Three-screen timeout demo inserted after the tutorial summary.
 * Entirely self-contained — no side effects on the main timeline.
 *
 * Screen 1: observation layout, slider disabled, clock runs to zero.
 *           Fixed callout near the clock: "You must respond before the clock runs out"
 * Screen 2: "Too slow" screen with info boxes.
 * Screen 3: "Tutorial complete" + "Proceed to experiment" button.
 */

import { buildSliderHTML }       from './slider-continuous.js';
import { buildBinarySliderHTMLv2 as buildBinarySliderHTML } from './slider-binary.js';
import { startTimeoutClock }     from './observation-timeout-clock.js';

const SAMPLE_BLUE = '#2563eb';
const SAMPLE_RED  = '#ef4444';
const FADE_MS     = 1000; // matches plugin-observation-binary.js /
                          // plugin-observation-continuous.js / both
                          // tutorial draw-animation modules

const info = {
  name: 'timeout-demo',
  parameters: {
    is_binary:         { type: 'BOOLEAN', default: false },
    t_obs_ms:          { type: 'INT',     default: 7000  },
    max_timeouts:      { type: 'INT',     default: 3     },
    demo_value:        { type: 'INT',     default: 63    },
    demo_binary_value: { type: 'INT',     default: 1     },
  },
};

class TimeoutDemoPlugin {
  constructor(jsPsych) { this.jsPsych = jsPsych; }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    const { is_binary, t_obs_ms, max_timeouts, demo_value, demo_binary_value } = trial;
    const _tObs = t_obs_ms;

    // ── Screen 1 ─────────────────────────────────────────────────────────────
    const showScreen1 = () => {
      document.body.style.backgroundColor = '#f5f5f5';

      const sliderHTML = is_binary
        ? buildBinarySliderHTML({ unset: true, initPos: 50, showValue: true })
        : buildSliderHTML({ unset: true, initPos: 50, showValue: true });

      const stimulus = is_binary
        ? `<div id="demo-stimulus" class="binary-circle" style="background:#fff;"></div>`
        : `<div id="demo-stimulus" class="stimulus-number" style="color:#ef4444;opacity:0;">${demo_value}</div>`;

      // Callout injected into body so it renders near the fixed clock
      const note = document.createElement('div');
      note.id = 'timeout-demo-note';
      Object.assign(note.style, {
        position:   'fixed',
        top:        'calc(1.2rem + 88px + 0.75rem)',
        right:      'calc(1rem + 88px + 0.75rem)',
        background: '#fffbeb',
        border:     '1px solid #fbbf24',
        borderRadius: '8px',
        padding:    '0.5rem 0.9rem',
        fontSize:   '1.1rem',
        color:      '#92400e',
        lineHeight: '1.4',
        zIndex:     '200',
        whiteSpace: 'nowrap',
        boxShadow:  '0 2px 6px rgba(0,0,0,0.1)',
      });
      note.textContent = 'You must respond before the clock runs out ↗';
      document.body.appendChild(note);

      display_el.innerHTML = `
        <canvas id="demo-clock" class="timeout-clock" width="88" height="88"></canvas>
        <div class="obs-wrap">
          ${stimulus}
          <div style="opacity:0.4;pointer-events:none;width:100%;">${sliderHTML}</div>
        </div>
        <div style="text-align:center;">
          <button class="jspsych-btn" disabled
            style="font-size:1.6rem;padding:1rem 3.5rem;min-width:200px;opacity:0.4;">
            Submit
          </button>
        </div>`;

      const canvas = display_el.querySelector('#demo-clock');

      // Fade the stimulus in — purely cosmetic, matches the same fade used
      // for real observations and the tutorial's draw animations. Doesn't
      // gate anything: the clock below starts immediately regardless.
      const stimEl = display_el.querySelector('#demo-stimulus');
      requestAnimationFrame(() => {
        if (!stimEl) return;
        if (is_binary) {
          stimEl.style.transition = `background ${FADE_MS}ms ease`;
          stimEl.style.background = demo_binary_value === 1 ? SAMPLE_BLUE : SAMPLE_RED;
        } else {
          stimEl.style.transition = `opacity ${FADE_MS}ms ease`;
          stimEl.style.opacity = '1';
        }
      });

      startTimeoutClock(canvas, _tObs, () => {
        document.body.style.backgroundColor = '#f5f5f5';
        const n = document.getElementById('timeout-demo-note');
        if (n) n.parentNode.removeChild(n);
        showScreen2();
      });
    };

    // ── Screen 2 ─────────────────────────────────────────────────────────────
    const showScreen2 = () => {
      display_el.innerHTML = `
        <div style="display:flex;flex-direction:column;align-items:center;gap:1.2rem;padding-top:2rem;">
          <span style="font-size:3rem;font-weight:bold;color:#ef4444;">Too slow</span>
          <span style="font-size:2rem;font-style:italic;color:#555;">${max_timeouts} timeouts remaining</span>
          <div style="display:flex;flex-direction:column;align-items:center;gap:0.75rem;margin-top:0.5rem;">
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:8px;
                        padding:0.9rem 1.2rem;font-size:1.3rem;color:#333;text-align:center;">
              ↩ The observation will <strong>automatically repeat</strong>
            </div>
            <div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;
                        padding:0.9rem 1.2rem;font-size:1.3rem;color:#b91c1c;text-align:center;">
              &#9888; If you reach 0 timeouts in a trial, the experiment will end
              and you will receive partial compensation
            </div>
          </div>
          <button id="demo-next-btn" class="jspsych-btn"
            style="font-size:1.6rem;padding:1rem 3.5rem;margin-top:0.5rem;">
            Next
          </button>
        </div>`;

      display_el.querySelector('#demo-next-btn').addEventListener('pointerdown', (e) => {
        e.preventDefault();
        showScreen3();
      });
    };

    // ── Screen 3 ─────────────────────────────────────────────────────────────
    const showScreen3 = () => {
      document.body.style.backgroundColor = '#f5f5f5';
      display_el.innerHTML = `
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;min-height:50vh;gap:2rem;">
          <div style="font-size:2.2rem;font-weight:bold;color:#333;">Tutorial complete</div>
          <button id="demo-proceed-btn" class="jspsych-btn"
            style="font-size:1.6rem;padding:1rem 3.5rem;">
            Proceed to experiment
          </button>
        </div>`;

      display_el.querySelector('#demo-proceed-btn').addEventListener('pointerdown', (e) => {
        e.preventDefault();
        this.jsPsych.finishTrial({ screen: 'timeout_demo' });
      });
    };

    showScreen1();
  }
}

TimeoutDemoPlugin.info = info;
export default TimeoutDemoPlugin;
