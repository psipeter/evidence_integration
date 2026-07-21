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
import { computeTrialReward, computeRunningMeans, refForObservation, computeResponseError } from './bonus-continuous.js';

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
 * @param {number} [cfg.itiShortMs]  tutorial between-observation ITI, ms (default 1000)
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
    tObsMs, maxTimeoutsPerTrial, itiShortMs = 1000, errorMode = 'true_mean',
  } = cfg;
  const { TutorialIntroPlugin, TutorialObsPlugin, TutorialSummaryPlugin } = plugins;

  // Same errorMode branch as build-trial-timeline.js -- see that file's own
  // comment for the full rationale; kept identical rather than duplicated
  // differently, since both ultimately call the same bonus-continuous.js
  // functions.
  const errorRefs = (values, trueMean) =>
    errorMode === 'running_mean' ? computeRunningMeans(values) : trueMean;
  // Computed ONCE (tutorialValues is fixed, not response-dependent) --
  // resolved per-observation via refForObservation() below, index 0 for
  // the intro trial and _o for each subsequent tutorial observation (both
  // index directly into tutorialValues, since intro handles index 0 of
  // that same array).
  const _refs = errorRefs(tutorialValues, tutorialMean);

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
        n_obs:         tutorialValues.length,
        // Tracker/history param -- see tutorial-tracker.js and
        // distribution-continuous.js's `history` param. Obs 1 has no
        // history yet; values_so_far is just its own value.
        values_so_far: tutorialValues.slice(0, 1),
        data: { screen: 'tutorial_intro', observation: 0, value: tutorialValues[0] },
      };

  // ── Tutorial observation state ────────────────────────────────────────────
  let tutorialLastResponse = defaultValue;
  let tutorialResponses    = [];

  introTrial.on_finish = (data) => {
    if (data.response !== null && data.response !== undefined) {
      tutorialLastResponse = data.response;
      const error = isBinary ? null : computeResponseError(data.response, refForObservation(_refs, 0));
      data.error = error;
      tutorialResponses.push({ value: tutorialValues[0], response: data.response, error });
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
          duration_ms: itiShortMs,
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
          // Tracker/history param -- see tutorial-tracker.js and
          // distribution-continuous.js's `history` param. Includes the
          // CURRENT value at the last index (tutorial-tracker.js's own
          // "current slot" logic expects that); distribution-continuous.js's
          // `history` is derived from this same slice minus the last entry
          // inside the plugin, not duplicated here.
          values_so_far:  tutorialValues.slice(0, _o + 1),
          data: { screen: 'tutorial_observation', observation: _o, value: _value },
          on_finish: (data) => {
            if (data.response !== null) {
              tutorialLastResponse = data.response;
              const error = isBinary ? null : computeResponseError(data.response, refForObservation(_refs, _o));
              data.error = error;
              tutorialResponses.push({ value: _value, response: data.response, error });
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
    error_mode: errorMode,
    // Per-TRIAL bonus preview (chat history, continuous only -- see
    // bonus-continuous.js) -- SUM of the already-stored per-observation
    // errors above, not recomputed independently. isBinary guard matches
    // build-trial-timeline.js's own (binary has no bonus scheme yet).
    total_error: () => isBinary ? 0 : tutorialResponses.reduce((sum, r) => sum + (r.error ?? 0), 0),
    reward:      () => isBinary ? 0 : computeTrialReward(tutorialResponses.reduce((sum, r) => sum + (r.error ?? 0), 0)),
    data: { screen: 'tutorial_summary' },
  });

  // Recap screen REMOVED (chat history) -- plugin-tutorial-recap-
  // continuous.js (now deleted) showed this same "you won't see these
  // graphics in the real experiment, rely on memory instead" message as
  // its own screen right after the summary. Obs 11-15 (see plugin-
  // tutorial-observation-continuous.js's buildHintHTML) now cover the same
  // ground DURING the tutorial observations themselves -- progressively
  // hiding the right column's graphics and showing the identical
  // RECAP_TEXT_1/RECAP_TEXT_2 message -- making a separate screen for it
  // afterward redundant.

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
