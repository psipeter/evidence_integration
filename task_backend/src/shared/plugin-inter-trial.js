/**
 * plugin-inter-trial.js
 * 5-second inter-trial reset screen.
 * Shows "Trial X / 40" and a pulsing "generating new sequence…" message
 * to provide a clear perceptual break between trials. Same wording for both
 * tasks — "sequence" describes the observable data stream and stays
 * accurate regardless of whether the hidden generative parameter
 * (probability / mean+std) happens to repeat from an earlier trial (the
 * task design reuses 6 unique parameter sets across 24 trials), whereas
 * "new distribution" would overclaim novelty it can't always guarantee.
 */

const info = {
  name: 'inter-trial',
  parameters: {
    trial_num:   { type: 'INT', default: 1  },
    n_trials:    { type: 'INT', default: 40 },
    duration_ms: { type: 'INT', default: 5000 },
    is_colors:   { type: 'BOOLEAN', default: false },
  },
};

class InterTrialPlugin {
  constructor(jsPsych) { this.jsPsych = jsPsych; }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    const { trial_num, n_trials, duration_ms } = trial;
    const label = 'generating new sequence…';

    display_el.innerHTML = `
      <div style="
        display:flex; flex-direction:column; align-items:center;
        justify-content:center; min-height:40vh; gap:1.2rem;">
        <div style="font-size:3rem; font-weight:bold; color:#555;">
          Trial ${trial_num} / ${n_trials}
        </div>
        <div id="gen-label" style="
          font-size:2rem; color:#888; font-style:italic;
          opacity:0; transition:opacity 0.8s ease;">
          ${label}
        </div>
      </div>`;

    // Fade the label in and out repeatedly
    const lbl = display_el.querySelector('#gen-label');
    let visible = false;
    let animId  = null;

    const pulse = () => {
      visible = !visible;
      lbl.style.opacity = visible ? '1' : '0';
      animId = setTimeout(pulse, 900);
    };
    // Small delay before first fade-in
    animId = setTimeout(pulse, 400);

    setTimeout(() => {
      clearTimeout(animId);
      this.jsPsych.finishTrial({ screen: 'inter_trial_reset', trial_num });
    }, duration_ms);
  }
}

InterTrialPlugin.info = info;
export default InterTrialPlugin;
