/**
 * plugin-trial-summary-continuous.js
 * Inter-trial summary — continuous task.
 * Identical layout to plugin-tutorial-summary-continuous.js. The one
 * difference: no blue banner here (this screen has never had one) -- the
 * total-error/reward box below sits at the very top instead of below one.
 *
 * total_error/reward (chat history) are computed by the caller (see
 * build-trial-timeline.js) and passed in as trial parameters purely for
 * DISPLAY. They're also attached directly to the object passed to
 * this.jsPsych.finishTrial() below -- that's the reliable way to get them
 * onto THIS trial's own stored data (a jsPsych trial's `data` merges with
 * whatever finishTrial() is called with), so they flow into JATOS via the
 * same per-trial on_trial_finish append every other trial already uses,
 * no separate append call needed. bonus-continuous.js's own docstring has
 * the full formula/rationale.
 */
import { buildPerformanceSVG, COIN_FILL, ERROR_GREEN } from './draw-performance-continuous.js';

const info = {
  name: 'trial-summary-continuous',
  parameters: {
    true_mean:        { type: 'FLOAT'   },
    true_std:         { type: 'FLOAT'   },
    values:           { type: 'OBJECT',  default: []    },
    responses:        { type: 'OBJECT',  default: []    },
    error_mode:       { type: 'STRING',  default: 'true_mean' },
    total_error:      { type: 'FLOAT',   default: 0     },
    reward:           { type: 'FLOAT',   default: 0     },
    show_performance: { type: 'BOOLEAN', default: false },
    is_last:          { type: 'BOOLEAN', default: false },
  },
};

class TrialSummaryContinuousPlugin {
  constructor(jsPsych) { this.jsPsych = jsPsych; }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';
    const { true_mean, true_std, values, responses, error_mode, total_error, reward,
            show_performance, is_last } = trial;

    display_el.innerHTML = `
      <div style="width:70vw;margin:0 auto;text-align:center;">
        ${show_performance ? `
        <div class="tutorial-info-block" style="margin-bottom:0.5rem;">
          <span>
            <span style="color:${ERROR_GREEN};font-weight:bold;">Total error: ${total_error.toFixed(1)}</span>
            &nbsp;&bull;&nbsp;
            <span style="color:${COIN_FILL};font-weight:bold;">Bonus: ${Math.round(reward)}¢</span>
          </span>
        </div>
        <div id="summary-svg"
          style="display:block;margin:0 auto 0.75rem;
                 width:100%;border:1px solid #e5e7eb;
                 border-radius:6px;background:#fff;padding:4px;">
        </div>` : ''}
        <button id="next-btn" class="jspsych-btn"
          style="font-size:1.6rem;padding:1rem 3.5rem;margin-top:1rem;">
          ${is_last ? 'Finish' : 'Next trial'}
        </button>
      </div>`;

    if (show_performance) {
      display_el.querySelector('#summary-svg').innerHTML =
        buildPerformanceSVG(true_mean, true_std, values, responses, error_mode);
    }

    display_el.querySelector('#next-btn').addEventListener('pointerdown', (e) => {
      e.preventDefault();
      this.jsPsych.finishTrial({ screen: 'inter_trial', total_error, reward });
    });
  }
}

TrialSummaryContinuousPlugin.info = info;
export default TrialSummaryContinuousPlugin;
