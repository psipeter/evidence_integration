/**
 * build-trial-timeline.js
 * Pure JS — builds the trial loop portion of the jsPsych timeline.
 * No jsPsych imports, no DOM, no CSS.
 * Used by timeline-builder.js (production) and test_browser.mjs (E2E tests,
 * against the real dev server -- there's no isolated unit test for this
 * file specifically).
 *
 * Observation rows export value/true_mean/true_std/true_p/qid directly
 * (alongside trial/observation/response/timed_out/rt/time_elapsed) rather
 * than requiring parse_results.py to reconstruct them via a join against a
 * saved sequence file. That join-based design worked when every
 * participant shared one file (so (task, trial) alone determined `value`),
 * but broke once per-participant pool assignment was introduced -- (task,
 * trial) is no longer enough; you'd also need to know which pool member
 * that specific participant got. Recording the values directly instead
 * makes every participant's raw export fully self-contained: no lookup,
 * no join, no dependency on the pool files still existing/matching later.
 */
import { computeTrialReward, computeResponseReward, computeRunningMeans, computeRunningRatios, refForObservation, computeResponseError } from './bonus-continuous.js';

export function buildTrialTimeline(cfg, plugins, jsPsych, earlyExit) {
  const {
    sequences, sliderDefault, defaultValue, btiMs, tObsMs,
    showSliderValue, showTrialPerformance, MAX_TIMEOUTS_PER_TRIAL = 3,
    distractorType = 'iti_length', errorMode = 'true_mean',
  } = cfg;

  const { ItiClockPlugin, TrialObsPlugin, TrialSummaryPlugin,
          InterTrialPlugin, isBinary } = plugins;

  // ITI duration for distract condition
  // Only 'iti_length' extends the ITI; other types ('popup', etc.) use default
  const ITI_DISTRACT_MS = distractorType === 'iti_length' ? 5000 : null;

  // errorMode (chat history, see bonus-continuous.js's own docstring) --
  // picks what a response's error is measured AGAINST. Task-aware: binary
  // has its OWN errorMode values ('true_p'/'running_p', not continuous's
  // 'true_mean'/'running_mean'), and its own responses/refs live on a
  // 0-100 scale like continuous's, so true_p (a 0-1 fraction) needs
  // scaling by 100 to compare directly -- computeRunningRatios already
  // returns a 0-100 scale, so only the FIXED (non-running) branch needs
  // the explicit *100. `trueRef` is whichever of seq.true_mean/seq.true_p
  // the caller passes in (continuous sequences only populate true_mean,
  // binary ones only populate true_p).
  const errorRefs = (values, trueRef) => {
    if (isBinary) {
      return errorMode === 'running_p' ? computeRunningRatios(values) : trueRef * 100;
    }
    return errorMode === 'running_mean' ? computeRunningMeans(values) : trueRef;
  };

  let lastResponse = defaultValue, timedOut = false, trialTimeouts = 0;
  let exitFlag = false, lastTrialResponses = [];
  const timeline = [];

  for (let t = 0; t < sequences.length; t++) {
    const seq = sequences[t];
    const trialResponses = [];
    lastTrialResponses   = trialResponses;
    const trialTimeline  = [];
    // Computed ONCE per trial (values are known at build time, not
    // response-dependent) -- either a single scalar or a full
    // per-observation array, resolved per-observation via
    // refForObservation() below rather than recomputed per response.
    const _refs = errorRefs(seq.values, isBinary ? seq.true_p : seq.true_mean);

    for (let o = 0; o < seq.values.length; o++) {
      const _o = o, _val = seq.values[o];

      if (o > 0) {
        trialTimeline.push({
          timeline: [{ type: ItiClockPlugin,
            duration_ms: (seq.iti_condition === 'distract' && ITI_DISTRACT_MS !== null)
              ? ITI_DISTRACT_MS : (seq.iti_ms ?? 1000),
            iti_condition:   seq.iti_condition ?? 'control',
            distractor_type: distractorType,
            is_binary:       isBinary,
            data: { screen: 'iti', trial: t, observation: _o } }],
          conditional_function: () => !exitFlag,
        });
      }

      trialTimeline.push({
        timeline: [
          {
            type: TrialObsPlugin, value: _val, trial_num: t+1,
            n_trials: sequences.length, t_obs_ms: tObsMs,
            slider_default: sliderDefault, init_pos: () => lastResponse,
            show_value: showSliderValue,
            data: {
              screen: 'observation', trial: t, observation: _o,
              // Recorded directly (see module docstring) instead of being
              // left for parse_results.py to look up -- (task, trial)
              // alone no longer determines these once each participant can
              // get a different pool member.
              value: _val, qid: seq.qid,
              true_mean: seq.true_mean, true_std: seq.true_std,
              true_p: seq.true_p ?? null,
            },
            on_finish: (data) => {
              timedOut = data.timed_out;
              if (data.timed_out) {
                trialTimeouts++;
                jsPsych.data.addProperties({ trial_timeouts: trialTimeouts });
                if (trialTimeouts >= MAX_TIMEOUTS_PER_TRIAL) {
                  exitFlag = true;
                  setTimeout(earlyExit, 100);
                }
              } else {
                if (data.response !== null) lastResponse = data.response;
                // Computed and attached here (jsPsych's on_finish(data) can
                // mutate `data` and have it persist into the stored/global
                // DataCollection) so the value is already present on `data`
                // by the time the GLOBAL on_trial_finish -- timeline-
                // builder.js's per-trial JATOS append -- runs right after
                // this. "Saved as a separate column, appended in the usual
                // way" (chat history) needed no separate append call at
                // all; it just needed to exist on `data` before the
                // existing one fires. Summed at the trial-summary screen
                // (not recomputed independently there) so the two can never
                // drift apart -- see bonus-continuous.js's own docstring.
                // `data.reward` (chat history, this session) is the SAME
                // per-observation-then-summed pattern, now that reward
                // itself is computed per observation, not once per trial
                // from a pre-summed error -- see bonus-continuous.js's own
                // "REWARD FORMULA" docstring.
                const ref = refForObservation(_refs, _o);
                data.error = computeResponseError(data.response, ref);
                data.reward = computeResponseReward(data.error);
                trialResponses.push({ observation: _o, value: _val, response: data.response, error: data.error });
              }
            },
          },
          {
            timeline: [{ type: ItiClockPlugin, duration_ms: seq.iti_ms ?? 1000,
              timed_out: true, timeouts_remaining: () => MAX_TIMEOUTS_PER_TRIAL - trialTimeouts,
              data: { screen: 'iti_replay', trial: t, observation: _o } }],
            conditional_function: () => timedOut && trialTimeouts < MAX_TIMEOUTS_PER_TRIAL,
          },
        ],
        loop_function:        () => timedOut && trialTimeouts < MAX_TIMEOUTS_PER_TRIAL,
        conditional_function: () => !exitFlag,
      });
    }

    timeline.push({
      timeline: trialTimeline,
      conditional_function: () => !exitFlag,
      on_timeline_start: () => {
        lastResponse = defaultValue; timedOut = false;
        trialTimeouts = 0; trialResponses.length = 0;
      },
    });

    const _t = t, _values = [...seq.values], _resp = trialResponses;
    const _isLast = t === sequences.length - 1;

    if (!_isLast) {
      timeline.push({
        timeline: [{ type: TrialSummaryPlugin, true_mean: seq.true_mean, true_std: seq.true_std,
          true_p: seq.true_p ?? null, values: _values, responses: () => _resp.map(r => r.response),
          error_mode: errorMode,
          // Per-TRIAL bonus (chat history, this session) -- SUM of the
          // already-stored per-observation REWARDS now, not one formula
          // applied to a pre-summed error -- see bonus-continuous.js's own
          // "REWARD FORMULA" docstring. computeTrialReward takes the
          // per-observation ERRORS array and internally maps each through
          // computeResponseReward before summing, so this file itself
          // never duplicates that formula.
          total_error: () => _resp.reduce((sum, r) => sum + (r.error ?? 0), 0),
          reward:      () => computeTrialReward(_resp.map(r => r.error)),
          show_performance: showTrialPerformance, is_last: false,
          data: { screen: 'inter_trial', trial: _t } }],
        conditional_function: () => !exitFlag,
      });
      timeline.push({
        timeline: [{ type: InterTrialPlugin, trial_num: _t+2, n_trials: sequences.length,
          duration_ms: btiMs, is_binary: isBinary,
          data: { screen: 'inter_trial_reset', trial: _t } }],
        conditional_function: () => !exitFlag,
      });
    }
  }

  // Final summary
  const _seq = sequences[sequences.length - 1], _resp = lastTrialResponses;
  timeline.push({
    timeline: [{ type: TrialSummaryPlugin, true_mean: _seq.true_mean, true_std: _seq.true_std,
      true_p: _seq.true_p ?? null, values: [..._seq.values], responses: () => _resp.map(r => r.response),
      error_mode: errorMode,
      total_error: () => _resp.reduce((sum, r) => sum + (r.error ?? 0), 0),
      reward:      () => computeTrialReward(_resp.map(r => r.error)),
      show_performance: showTrialPerformance, is_last: true,
      data: { screen: 'inter_trial', trial: sequences.length - 1 } }],
    conditional_function: () => !exitFlag,
  });

  return { timeline, isExited: () => exitFlag };
}
