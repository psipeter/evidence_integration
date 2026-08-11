/**
 * build-tutorial-timeline.js
 * Builds the tutorial sub-timeline (intro → tutorial observations → tutorial
 * summary → tutorial complete).
 *
 * None of these screens have a response deadline — participants need
 * unhurried time to read and think during the tutorial.
 *
 * CHECKPOINTING (task_backend design -- see TODO.md): only tutorial_intro
 * and tutorial_observation carry response data and send a checkpoint
 * (phase='tutorial'); tutorial_iti/tutorial_summary/tutorial_complete need
 * no network call under the trial-boundary-resume design (a reload lands
 * back at the START of the tutorial regardless of which of these screens
 * it was on -- see progress-check's resume branch). trial_index stays the
 * -1 sentinel throughout (this is never a real trial); attempt is always
 * 0 -- unlike the real trial loop, the tutorial never retries an
 * observation (no timeout mechanism here at all).
 */
import jsPsychHtmlButtonResponse from '@jspsych/plugin-html-button-response';
import ItiClockPlugin      from './plugin-iti-clock.js';
import { computeTrialReward, computeResponseReward, computeErrorRefs, refForObservation, computeResponseError, NUMBERS_BONUS_DECAY, COLORS_BONUS_DECAY } from './scoring.js';
import { PHASES } from './phases.js';

/**
 * @param {object} cfg
 * @param {boolean} cfg.isColors
 * @param {number[]} cfg.tutorialValues
 * @param {number} cfg.tutorialMean
 * @param {number} cfg.tutorialStd
 * @param {string} cfg.sliderDefault
 * @param {number} cfg.defaultValue
 * @param {boolean} cfg.showSliderValue
 * @param {number} cfg.tObsMs
 * @param {number} [cfg.itiShortMs]  tutorial between-observation ITI, ms (default 1000)
 * @param {object} plugins
 * @param {*} plugins.TutorialIntroPlugin   (task-specific intro plugin, obs 0)
 * @param {*} plugins.TutorialObsPlugin     (task-specific tutorial obs plugin, obs 1..n)
 * @param {*} plugins.TutorialSummaryPlugin (task-specific tutorial summary plugin)
 * @param {Function} [sendCheckpoint]  see backend-client.js's createCheckpointSender
 * @returns {Array} tutorial timeline nodes
 */
export function buildTutorialTimeline(cfg, plugins, sendCheckpoint) {
  const {
    isColors, tutorialValues, tutorialMean, tutorialStd,
    sliderDefault, defaultValue, showSliderValue,
    tObsMs, itiShortMs = 1000, errorMode = 'true_mean',
  } = cfg;
  const { TutorialIntroPlugin, TutorialObsPlugin, TutorialSummaryPlugin } = plugins;
  const bonusDecay = isColors ? COLORS_BONUS_DECAY : NUMBERS_BONUS_DECAY;

  // tutorialMean holds true_p (0-1 scale) for colors and true_mean for
  // numbers -- same dual role it plays a few lines below.
  const _refs = computeErrorRefs(tutorialValues, tutorialMean, { isColors, errorMode });

  const tutorialTimeline = [];

  // ── Tutorial intro — handles obs 0 for both tasks ─────────────────────────
  const tutorialStartObs = 1;
  const introTrial = isColors
    ? {
        type: TutorialIntroPlugin,
        example_value: tutorialValues[0],
        true_p:        tutorialMean,
        n_obs:         tutorialValues.length,
        values_so_far: tutorialValues.slice(0, 1),
        data: { screen: 'tutorial_intro', observation: 0, value: tutorialValues[0] },
      }
    : {
        type: TutorialIntroPlugin,
        example_value: tutorialValues[0],
        true_mean:     tutorialMean,
        true_std:      tutorialStd,
        n_obs:         tutorialValues.length,
        values_so_far: tutorialValues.slice(0, 1),
        data: { screen: 'tutorial_intro', observation: 0, value: tutorialValues[0] },
      };

  // ── Tutorial observation state ────────────────────────────────────────────
  let tutorialLastResponse = defaultValue;
  let tutorialResponses    = [];

  introTrial.on_finish = (data) => {
    if (data.response !== null && data.response !== undefined) {
      tutorialLastResponse = data.response;
      const error = computeResponseError(data.response, refForObservation(_refs, 0));
      data.error = error;
      data.reward = computeResponseReward(error, bonusDecay);
      tutorialResponses.push({ value: tutorialValues[0], response: data.response, error });
    }
    sendCheckpoint?.({
      phase: PHASES.TUTORIAL, trialIndex: -1, observationIndex: 0, attempt: 0,
      response: data.response ?? null, timed_out: false, rt: data.rt ?? null,
      value: tutorialValues[0], true_mean: isColors ? null : tutorialMean,
      true_std: isColors ? null : tutorialStd, true_p: isColors ? tutorialMean : null,
      qid: null, error: data.error ?? null, reward: data.reward ?? 0,
    });
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
          t_obs_ms:       tObsMs,
          values_so_far:  tutorialValues.slice(0, _o + 1),
          data: { screen: 'tutorial_observation', observation: _o, value: _value },
          on_finish: (data) => {
            if (data.response !== null) {
              tutorialLastResponse = data.response;
              const error = computeResponseError(data.response, refForObservation(_refs, _o));
              data.error = error;
              data.reward = computeResponseReward(error, bonusDecay);
              tutorialResponses.push({ value: _value, response: data.response, error });
            }
            sendCheckpoint?.({
              phase: PHASES.TUTORIAL, trialIndex: -1, observationIndex: _o, attempt: 0,
              response: data.response ?? null, timed_out: false, rt: data.rt ?? null,
              value: _value, true_mean: isColors ? null : tutorialMean,
              true_std: isColors ? null : tutorialStd, true_p: isColors ? tutorialMean : null,
              qid: null, error: data.error ?? null, reward: data.reward ?? 0,
            });
          },
        },
      ],
    });
  }

  tutorialTimeline.push({
    type: TutorialSummaryPlugin,
    true_mean:  tutorialMean,
    true_std:   tutorialStd,
    true_p:     tutorialMean,
    values:     () => tutorialResponses.map(r => r.value),
    responses:  () => tutorialResponses.map(r => r.response),
    error_mode: errorMode,
    total_error: () => tutorialResponses.reduce((sum, r) => sum + (r.error ?? 0), 0),
    reward:      () => computeTrialReward(tutorialResponses.map(r => r.error), bonusDecay),
    data: { screen: 'tutorial_summary' },
  });

  // Tutorial complete — standalone transition screen between the tutorial
  // summary and the main experiment.
  tutorialTimeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div style="text-align:center;padding-top:8rem;">
        <div style="font-size:2.2rem;font-weight:bold;color:#333;margin-bottom:1.5rem;">Tutorial complete</div>
      </div>`,
    choices: ['Proceed to experiment'],
    button_html: (choice) =>
      `<button id="tutorial-complete-btn" class="jspsych-btn" style="font-size:1.6rem;padding:1rem 3.5rem;">${choice}</button>`,
    data: { screen: 'tutorial_complete' },
  });

  return tutorialTimeline;
}
