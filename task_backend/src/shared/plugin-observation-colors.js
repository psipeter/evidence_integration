/**
 * plugin-observation-colors.js
 * Single colors observation: coloured circle + colors slider + timeout clock.
 *
 * IMPORTANT: trial() must NOT be declared `async`. jsPsych 8.2.3 advances the
 * timeline once a trial() method's returned Promise resolves — for an async
 * function with no internal `await`, that happens essentially synchronously
 * (right after registering the rAF timeout loop below), long before
 * finishTrial() is actually called. That caused a serious concurrency bug:
 * a new observation instance was spawned roughly every ITI cycle regardless
 * of whether the visible one had finished, with old instances' timeout loops
 * continuing to run invisibly in the background and calling finishTrial()
 * on trials jsPsych had already moved past. Completion must be signaled
 * exclusively via the explicit jsPsych.finishTrial() call below — trial()
 * itself just needs to render, call on_load(), and return.
 *
 * on_load() is called after innerHTML is set, signalling DOM ready.
 * initColorsSlider() attaches listeners directly — no setTimeout/rAF deferral.
 *
 * The countdown-clock rendering itself lives in observation-timeout-clock.js,
 * shared with plugin-observation-numbers.js (this logic used to be
 * duplicated verbatim in both files).
 */
import { buildColorsSliderHTMLv2 as buildColorsSliderHTML, initColorsSliderV2 as initColorsSlider } from './slider-colors.js';
import { startTimeoutClock } from './observation-timeout-clock.js';

const SAMPLE_BLUE = '#2563eb';
const SAMPLE_RED  = '#ef4444';
const FADE_MS     = 1000; // slower than the tutorial's 380ms fade — more
                          // noticeable in the main task where there's no
                          // preceding bubbling animation to build anticipation

const info = {
  name: 'observation-colors',
  parameters: {
    value:          { type: 'INT',     default: 1     },
    trial_num:      { type: 'INT',     default: 1     },
    n_trials:       { type: 'INT',     default: 1     },
    t_obs_ms:       { type: 'INT',     default: 5000  },
    slider_default: { type: 'STRING',  default: 'none'},
    init_pos:       { type: 'INT',     default: 50    },
    show_value:     { type: 'BOOLEAN', default: true  },
  },
};

class ObservationColorsPlugin {
  constructor(jsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_el, trial, on_load) {
    document.body.style.backgroundColor = '#f5f5f5';
    if (document.activeElement && document.activeElement !== document.body) {
      document.activeElement.blur();
    }

    const { value, trial_num, n_trials, t_obs_ms,
            slider_default, init_pos, show_value } = trial;
    const trialStart = performance.now();
    const resolvedInitPos = typeof init_pos === 'function' ? init_pos() : (init_pos ?? 50);
    const unset   = slider_default === 'none';
    const ballCol = value === 1 ? SAMPLE_BLUE : SAMPLE_RED;

    display_el.innerHTML = `
      <canvas id="timeout-clock" class="timeout-clock" width="88" height="88"></canvas>
      <div class="obs-wrap">
        <div id="obs-circle" class="colors-circle" style="background:#fff;"></div>
        ${buildColorsSliderHTML({ unset, initPos: resolvedInitPos, showValue: show_value })}
      </div>
      <div style="text-align:center;">
        <button id="submit-btn" class="jspsych-btn"
          ${unset ? 'disabled' : ''}
          style="font-size:1.6rem;padding:1rem 3.5rem;min-width:200px;">
          Submit
        </button>
      </div>`;

    // Signal DOM is ready
    on_load();

    // Fade the circle white → its actual color — purely cosmetic (mirrors the
    // tutorial's centre-circle fade in colors-draw-animation.js), doesn't gate
    // interactivity: the timeout clock and slider start immediately below
    // regardless of whether this transition has finished.
    const circle = display_el.querySelector('#obs-circle');
    requestAnimationFrame(() => {
      if (!circle) return;
      circle.style.transition = `background ${FADE_MS}ms ease`;
      circle.style.background = ballCol;
    });

    // ── Timeout + finish ──────────────────────────────────────────────────────
    let stopClock = null;
    let finished   = false;

    const finish = (timed_out) => {
      if (finished) return;
      finished = true;
      if (stopClock) stopClock();
      document.body.style.backgroundColor = '#f5f5f5';
      const slider   = display_el.querySelector('#response-slider');
      const hadUnset = slider?.classList.contains('slider-unset') ||
                       slider?.classList.contains('slider-last') || false;
      const response = (!timed_out && !hadUnset) ? parseInt(slider.value) : null;
      const rt       = timed_out ? null : Math.round(performance.now() - trialStart);
      this.jsPsych.finishTrial({ response, timed_out, rt });
    };

    // ── Slider — wire immediately, DOM is ready after on_load() ───────────────
    initColorsSlider(display_el, {
      unset,
      showValue: show_value,
      onFinish:  () => finish(false),
    });

    // ── Timeout clock ─────────────────────────────────────────────────────────
    const canvas = display_el.querySelector('#timeout-clock');
    stopClock = startTimeoutClock(canvas, t_obs_ms, () => finish(true));
  }
}

ObservationColorsPlugin.info = info;
export default ObservationColorsPlugin;
