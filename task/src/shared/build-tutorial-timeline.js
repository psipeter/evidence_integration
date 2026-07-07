/**
 * build-tutorial-timeline.js
 * Builds the tutorial sub-timeline (intro → tutorial observations → tutorial
 * summary → timeout demo). Extracted from timeline-builder.js — pure
 * extraction, no behavior change.
 *
 * None of these screens have a response deadline — participants need
 * unhurried time to read and think during the tutorial. The timeout demo at
 * the end explains the real deadline before trial 1, but doesn't impose one
 * here.
 */
import ItiClockPlugin      from './plugin-iti-clock.js';
import TimeoutDemoPlugin   from './plugin-timeout-demo.js';

/**
 * @param {object} cfg
 * @param {boolean} cfg.isBinary
 * @param {number[]} cfg.tutorialValues
 * @param {number} cfg.tutorialMean
 * @param {number} cfg.tutorialStd
 * @param {string} cfg.sliderDefault
 * @param {number} cfg.defaultValue
 * @param {boolean} cfg.showSliderValue
 * @param {number} cfg.tObsMs
 * @param {number} cfg.maxTimeoutsPerTrial
 * @param {object} plugins
 * @param {*} plugins.TutorialIntroPlugin   (task-specific intro plugin, obs 0)
 * @param {*} plugins.TutorialObsPlugin     (task-specific tutorial obs plugin, obs 1..n)
 * @param {*} plugins.TutorialSummaryPlugin (task-specific tutorial summary plugin)
 * @returns {Array} tutorial timeline nodes
 */
export function buildTutorialTimeline(cfg, plugins) {
  const {
    isBinary, tutorialValues, tutorialMean, tutorialStd,
    sliderDefault, defaultValue, showSliderValue,
    tObsMs, maxTimeoutsPerTrial,
  } = cfg;
  const { TutorialIntroPlugin, TutorialObsPlugin, TutorialSummaryPlugin } = plugins;

  const tutorialTimeline = [];

  // ── Tutorial intro — handles obs 0 for both tasks ─────────────────────────
  const tutorialStartObs = 1;
  const introTrial = isBinary
    ? {
        type: TutorialIntroPlugin,
        example_value: tutorialValues[0],
        true_p:        tutorialMean,
        n_obs:         tutorialValues.length,
        data: { screen: 'tutorial_intro', observation: 0, value: tutorialValues[0] },
      }
    : {
        type: TutorialIntroPlugin,
        example_value: tutorialValues[0],
        true_mean:     tutorialMean,
        true_std:      tutorialStd,
        data: { screen: 'tutorial_intro', observation: 0, value: tutorialValues[0] },
      };

  // ── Tutorial observation state ────────────────────────────────────────────
  let tutorialLastResponse = defaultValue;
  let tutorialResponses    = [];

  introTrial.on_finish = (data) => {
    if (data.response !== null && data.response !== undefined) {
      tutorialLastResponse = data.response;
      tutorialResponses.push({ value: tutorialValues[0], response: data.response });
    }
  };
  tutorialTimeline.push(introTrial);

  for (let o = tutorialStartObs; o < tutorialValues.length; o++) {
    const _o     = o;
    const _value = tutorialValues[o];

    tutorialTimeline.push({
      timeline: [
        {
          type: ItiClockPlugin,
          duration_ms: 1000,  // tutorial ITI always 1s
          data: { screen: 'tutorial_iti', observation: _o },
        },
        {
          type: TutorialObsPlugin,
          value:          _value,
          obs_num:        _o + 1,
          n_obs:          tutorialValues.length,
          true_mean:      tutorialMean,
          true_std:       tutorialStd,
          true_p:         tutorialMean,
          slider_default: sliderDefault,
          init_pos:       () => tutorialLastResponse,
          show_value:     showSliderValue,
          data: { screen: 'tutorial_observation', observation: _o, value: _value },
          on_finish: (data) => {
            if (data.response !== null) {
              tutorialLastResponse = data.response;
              tutorialResponses.push({ value: _value, response: data.response });
            }
          },
        },
      ],
    });
  }

  tutorialTimeline.push({
    type: TutorialSummaryPlugin,
    // continuous params
    true_mean:  tutorialMean,
    true_std:   tutorialStd,
    // binary params
    true_p:     tutorialMean,
    values:     () => tutorialResponses.map(r => r.value),
    responses:  () => tutorialResponses.map(r => r.response),
    data: { screen: 'tutorial_summary' },
  });

  // Timeout demo — three screens explaining the timeout mechanism (this is
  // the one tutorial screen that intentionally shows a countdown, since its
  // whole purpose is to demonstrate what the real deadline looks like).
  tutorialTimeline.push({
    type:         TimeoutDemoPlugin,
    is_binary:    isBinary,
    t_obs_ms:     tObsMs,
    max_timeouts: maxTimeoutsPerTrial,
    data: { screen: 'timeout_demo' },
  });

  return tutorialTimeline;
}
