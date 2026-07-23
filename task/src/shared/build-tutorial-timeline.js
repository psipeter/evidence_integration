/**
 * build-tutorial-timeline.js
 * Builds the tutorial sub-timeline (intro → tutorial observations → tutorial
 * summary). Extracted from timeline-builder.js — pure extraction, no
 * behavior change (at extraction time).
 *
 * None of these screens have a response deadline — participants need
 * unhurried time to read and think during the tutorial.
 *
 * Timeout demo DISABLED (chat decision, this session): the 3-screen timeout
 * explanation that used to run after the tutorial summary and before trial 1
 * is commented out below, not deleted -- many participants found it
 * confusing, and the consent screen's own red warning box already discloses
 * the response deadline in text. A live clock demonstration was
 * reintroduced shortly after, but as part of the phase A-D hint
 * progression instead of a standalone screen -- see plugin-tutorial-
 * observation-continuous.js/-binary.js's own "phase E" for that.
 */
import jsPsychHtmlButtonResponse from '@jspsych/plugin-html-button-response';
import ItiClockPlugin      from './plugin-iti-clock.js';
import TimeoutDemoPlugin   from './plugin-timeout-demo.js';
import { computeTrialReward, computeResponseReward, computeRunningMeans, computeRunningRatios, refForObservation, computeResponseError } from './bonus-continuous.js';

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
  // differently. tutorialMean holds true_p (0-1 scale) for binary and
  // true_mean for continuous -- same dual role it already plays a few
  // lines below (true_p: tutorialMean / true_mean: tutorialMean).
  const errorRefs = (values, trueRef) => {
    if (isBinary) {
      return errorMode === 'running_p' ? computeRunningRatios(values) : trueRef * 100;
    }
    return errorMode === 'running_mean' ? computeRunningMeans(values) : trueRef;
  };
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
        // Tracker param -- see tutorial-tracker.js. Obs 1 has no history
        // yet; values_so_far is just its own value. Mirrors continuous's
        // identical param below.
        values_so_far: tutorialValues.slice(0, 1),
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
      const error = computeResponseError(data.response, refForObservation(_refs, 0));
      data.error = error;
      data.reward = computeResponseReward(error);
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
          // Phase E only (see plugin-tutorial-observation-continuous.js/
          // -binary.js's own docstring) -- duration for that phase's two
          // demo clocks. Threaded through from the same tObsMs the real
          // trial loop uses (build-trial-timeline.js), so the tutorial's
          // demo clock always matches the real deadline it's previewing.
          t_obs_ms:       tObsMs,
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
              const error = computeResponseError(data.response, refForObservation(_refs, _o));
              data.error = error;
              data.reward = computeResponseReward(error);
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
    // Per-TRIAL bonus preview (chat history, this session) -- SUM of the
    // already-stored per-observation REWARDS now, not one formula applied
    // to a pre-summed error -- see bonus-continuous.js's own "REWARD
    // FORMULA" docstring.
    total_error: () => tutorialResponses.reduce((sum, r) => sum + (r.error ?? 0), 0),
    reward:      () => computeTrialReward(tutorialResponses.map(r => r.error)),
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

  // Tutorial complete — standalone transition screen between the tutorial
  // summary and the main experiment (this session). This copy/button
  // previously lived as screen 3 of the now-disabled timeout demo (see
  // below) -- extracted here so the tutorial still ends on an explicit
  // "you're done, proceed to the real task" confirmation even with the
  // timeout mechanics no longer explained live. Uses the stock
  // html-button-response plugin (same pattern as build-welcome-screen.js)
  // rather than a dedicated custom plugin class, since there's no
  // interactivity here beyond the one button.
  //
  // NOTE on spacing: this plugin renders `choices`/`button_html` as a
  // SEPARATE sibling element AFTER the `stimulus` div, not inside it --
  // unlike the original demo's screen 3, which built title+button as one
  // hand-written flex column (so its min-height:50vh + justify-content:
  // center + gap:2rem correctly centered both together). Reusing that same
  // min-height/justify-content/gap recipe here centered ONLY the title
  // inside its own 50vh box, leaving the button to render right after that
  // already-tall box -- a large, unintended gap. Fixed by dropping the
  // vertical-centering trick entirely: simple top padding positions the
  // title, and margin-bottom on the title (not gap, since there's no shared
  // flex parent) sets the actual visual distance to the button.
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

  // Timeout demo — DISABLED (chat decision): many participants found this
  // 3-screen live demonstration confusing, and the consent screen already
  // discloses the response deadline via its own red warning box. Commented
  // out rather than deleted so it can be reintroduced easily. A live clock
  // demonstration WAS reintroduced (this session, shortly after this was
  // disabled) -- but as phase E of the per-observation hint progression in
  // plugin-tutorial-observation-continuous.js/-binary.js, not by
  // un-commenting this block. Its own screen 3 ("Tutorial complete" +
  // "Proceed to experiment") was separately extracted into the standalone
  // screen above, so re-enabling this block would currently show TWO
  // consecutive "tutorial complete"-style transitions back to back -- if
  // this demo is ever reintroduced anyway (e.g. for its specific 3-screen
  // explanatory framing, distinct from phase E's inline demonstration),
  // drop its own screen 3 (or the standalone screen above, whichever reads
  // better in sequence) rather than keeping both.
  // TimeoutDemoPlugin/its import, plugin-timeout-demo.js, and
  // timeline-builder.js's 'timeout_demo' progressLabel() case are all left
  // in place -- none of them need removing for this to take effect, only
  // this push() call actually wires the screen into the timeline.
  // tutorialTimeline.push({
  //   type:         TimeoutDemoPlugin,
  //   is_binary:    isBinary,
  //   t_obs_ms:     tObsMs,
  //   max_timeouts: maxTimeoutsPerTrial,
  //   data: { screen: 'timeout_demo' },
  // });

  return tutorialTimeline;
}
