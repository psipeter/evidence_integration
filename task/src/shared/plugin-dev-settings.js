/**
 * plugin-dev-settings.js
 * Dev-only settings page (TEST_MODE=true only).
 * Renders a settings panel where the developer can override runtime params
 * before the experiment begins. Finishes with a devSettings object.
 */

export default class DevSettingsPlugin {
  constructor(jsPsych) { this.jsPsych = jsPsych; }

  static info = {
    name: 'dev-settings',
    parameters: {
      defaults: { type: 'OBJECT', default: {} },  // base config values
    },
  };

  trial(display_el, trial) {
    const d = trial.defaults;

    display_el.innerHTML = `
      <div class="screen-wrap" style="max-width:560px;margin:0 auto;text-align:left;">
        <div class="tutorial-title">Dev settings</div>

        <div style="display:flex;flex-direction:column;gap:1.2rem;margin-top:1rem;">

          <!-- Tutorial -->
          <div class="dev-row">
            <label class="dev-label">Tutorial</label>
            <div class="dev-options" id="opt-tutorial">
              <button class="dev-btn active" data-val="true">Show</button>
              <button class="dev-btn"        data-val="false">Skip</button>
            </div>
          </div>

          <!-- Number of trials -->
          <div class="dev-row">
            <label class="dev-label">Trials</label>
            <div class="dev-options" id="opt-trials">
              <button class="dev-btn active" data-val="5">5</button>
              <button class="dev-btn"        data-val="10">10</button>
              <button class="dev-btn"        data-val="20">20</button>
              <button class="dev-btn"        data-val="40">40</button>
            </div>
          </div>

          <!-- BTI duration -->
          <div class="dev-row">
            <label class="dev-label">BTI (ms)</label>
            <div class="dev-options" id="opt-bti">
              <button class="dev-btn active" data-val="500">500</button>
              <button class="dev-btn"        data-val="1000">1000</button>
              <button class="dev-btn"        data-val="5000">5000</button>
            </div>
          </div>

          <!-- ITI duration -->
          <div class="dev-row">
            <label class="dev-label">ITI (ms)</label>
            <div class="dev-options" id="opt-iti">
              <button class="dev-btn active" data-val="500">500</button>
              <button class="dev-btn"        data-val="1000">1000</button>
              <button class="dev-btn"        data-val="3000">3000</button>
            </div>
          </div>

          <!-- Distractor type -->
          <div class="dev-row">
            <label class="dev-label">Distractor</label>
            <div class="dev-options" id="opt-distractor">
              <button class="dev-btn"        data-val="none">None</button>
              <button class="dev-btn"        data-val="iti_length">ITI length</button>
              <button class="dev-btn active" data-val="popup">Popup</button>
            </div>
          </div>

        </div>

        <div style="text-align:center;margin-top:2.5rem;">
          <button id="dev-start-btn" class="jspsych-btn"
            style="font-size:1.6rem;padding:1rem 4rem;">
            Start experiment
          </button>
        </div>
      </div>`;

    // ── Toggle button groups ───────────────────────────────────────────────
    display_el.querySelectorAll('.dev-options').forEach(group => {
      group.querySelectorAll('.dev-btn').forEach(btn => {
        btn.addEventListener('pointerdown', () => {
          group.querySelectorAll('.dev-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
        });
      });
    });

    // ── Finish ─────────────────────────────────────────────────────────────
    display_el.querySelector('#dev-start-btn').addEventListener('pointerdown', () => {
      const pick = (id) => display_el.querySelector(`#${id} .dev-btn.active`)?.dataset.val;

      const settings = {
        showTutorial:   pick('opt-tutorial')   === 'true',
        nTrials:        parseInt(pick('opt-trials')  ?? '5', 10),
        btiMs:          parseInt(pick('opt-bti')     ?? '500', 10),
        itiMs:          parseInt(pick('opt-iti')     ?? '500', 10),
        distractorType: pick('opt-distractor') ?? 'popup',
      };

      this.jsPsych.finishTrial({ screen: 'dev_settings', settings });
    });
  }
}
