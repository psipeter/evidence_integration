/**
 * plugin-observation-continuous.js
 * jsPsych 8 — one observation: stimulus number + slider + submit + timeout clock.
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
 * initSlider() attaches listeners directly — no setTimeout/rAF deferral.
 *
 * The countdown-clock rendering itself lives in observation-timeout-clock.js,
 * shared with plugin-observation-binary.js (this logic used to be duplicated
 * verbatim in both files).
 */
import { buildSliderHTML, initSlider } from './slider-continuous.js';
import { startTimeoutClock } from './observation-timeout-clock.js';

const FADE_MS = 1000; // mirrors the tutorial's continuous-draw-animation.js
                       // fade-in, for a consistent feel between tutorial and
                       // main task (same duration as plugin-observation-binary.js's
                       // circle fade too)

const info = {
  name: 'observation-continuous',
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

class ObservationContinuousPlugin {
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
    const resolvedInitPos = typeof init_pos === 'function' ? init_pos() : (init_pos ?? 0);
    const unset = slider_default === 'none';

    display_el.innerHTML = `
      <canvas id="timeout-clock" class="timeout-clock" width="88" height="88"></canvas>
      <div class="obs-wrap">
        <div id="stimulus-display" class="stimulus-number" style="color:#ef4444;opacity:0;">${value}</div>
        ${buildSliderHTML({ unset, initPos: resolvedInitPos, showValue: show_value })}
      </div>
      <div style="text-align:center;">
        <button id="submit-btn" class="jspsych-btn" disabled
          style="font-size:1.6rem;padding:1rem 3.5rem;min-width:200px;">
          Submit
        </button>
      </div>`;

    // Signal DOM is ready — jsPsych will mark trial as loaded
    on_load();

    // Fade the number in — purely cosmetic (mirrors the tutorial's centre-
    // number fade in continuous-draw-animation.js), doesn't gate
    // interactivity: the timeout clock and slider start immediately below
    // regardless of whether this transition has finished.
    const stimulus = display_el.querySelector('#stimulus-display');
    requestAnimationFrame(() => {
      if (!stimulus) return;
      stimulus.style.transition = `opacity ${FADE_MS}ms ease`;
      stimulus.style.opacity = '1';
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
      const hadUnset = slider?.classList.contains('slider-unset') ?? true;
      const response = (!timed_out && !hadUnset) ? parseInt(slider.value) : null;
      const rt       = timed_out ? null : Math.round(performance.now() - trialStart);
      this.jsPsych.finishTrial({ response, timed_out, rt });
    };

    // ── Slider — wire immediately, DOM is ready after on_load() ───────────────
    initSlider(display_el, {
      unset,
      showValue: show_value,
      onFinish:  () => finish(false),
    });

    // ── Timeout clock ─────────────────────────────────────────────────────────
    const canvas = display_el.querySelector('#timeout-clock');
    stopClock = startTimeoutClock(canvas, t_obs_ms, () => finish(true));
  }
}

ObservationContinuousPlugin.info = info;
export default ObservationContinuousPlugin;
