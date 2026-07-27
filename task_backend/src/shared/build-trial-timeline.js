/**
 * build-trial-timeline.js
 * Pure JS — builds the trial loop portion of the jsPsych timeline.
 * No jsPsych imports beyond the plugins passed in, no DOM, no CSS.
 *
 * Observation nodes carry value/true_mean/true_std/true_p/qid directly
 * (alongside trial/observation/response/timed_out/rt) rather than
 * requiring downstream analysis to reconstruct them via a join against a
 * saved sequence file -- every checkpoint row is self-contained.
 *
 * CHECKPOINTING (task_backend design -- see TODO.md): only the actual
 * `observation` node sends a checkpoint (phase='trial'); ITI/inter-trial/
 * summary nodes carry no response data and don't need one under the
 * trial-boundary-resume design (see the port review-pass inventory for
 * the full old-vs-new screen/phase mapping). `attempt` increments only
 * when a timeout-triggered retry replays the SAME (trial, observation)
 * pair -- tracked via a per-observation `attempt` counter that persists
 * across that observation's own loop_function re-runs (same JS closure),
 * resetting to 0 for the next observation.
 */
import { computeTrialReward, computeResponseReward, computeErrorRefs, refForObservation, computeResponseError } from './scoring.js';
import { PHASES } from './phases.js';

export function buildTrialTimeline(cfg, plugins, jsPsych, terminateSession, sendCheckpoint) {
  const {
    sequences, sliderDefault, defaultValue, btiMs, tObsMs,
    showSliderValue, showTrialPerformance, MAX_TIMEOUTS_PER_TRIAL = 3,
    distractorType = 'iti_length', errorMode = 'true_mean', startTrialIndex = 0,
  } = cfg;

  const { ItiClockPlugin, TrialObsPlugin, TrialSummaryPlugin,
          InterTrialPlugin, isColors } = plugins;

  // ITI duration for distract condition -- only 'iti_length' extends the
  // ITI; other types ('popup', etc.) use the default.
  const ITI_DISTRACT_MS = distractorType === 'iti_length' ? 5000 : null;

  let lastResponse = defaultValue, timedOut = false, trialTimeouts = 0;
  let exitFlag = false, lastTrialResponses = [];
  const timeline = [];

  for (let t = startTrialIndex; t < sequences.length; t++) {
    const seq = sequences[t];
    const trialResponses = [];
    lastTrialResponses   = trialResponses;
    const trialTimeline  = [];
    const _refs = computeErrorRefs(seq.values, isColors ? seq.true_p : seq.true_mean, { isColors, errorMode });

    for (let o = 0; o < seq.values.length; o++) {
      const _o = o, _val = seq.values[o];
      let attempt = 0; // resets per observation index; persists across THIS observation's own timeout retries

      if (o > 0) {
        trialTimeline.push({
          timeline: [{ type: ItiClockPlugin,
            duration_ms: (seq.iti_condition === 'distract' && ITI_DISTRACT_MS !== null)
              ? ITI_DISTRACT_MS : (seq.iti_ms ?? 1000),
            iti_condition:   seq.iti_condition ?? 'control',
            distractor_type: distractorType,
            is_colors:       isColors,
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
              value: _val, qid: seq.qid,
              true_mean: seq.true_mean, true_std: seq.true_std,
              true_p: seq.true_p ?? null,
            },
            on_finish: (data) => {
              timedOut = data.timed_out;
              const thisAttempt = attempt;

              if (data.timed_out) {
                attempt++; // next retry (if any) is a new attempt
                trialTimeouts++;
                jsPsych.data.addProperties({ trial_timeouts: trialTimeouts });
                data.error = null;
                data.reward = 0;
                sendCheckpoint?.({
                  phase: PHASES.TRIAL, trialIndex: t, observationIndex: _o, attempt: thisAttempt,
                  response: null, timed_out: true, rt: data.rt ?? null,
                  value: _val, true_mean: seq.true_mean, true_std: seq.true_std, true_p: seq.true_p ?? null,
                  qid: seq.qid, error: null, reward: 0,
                });
                if (trialTimeouts >= MAX_TIMEOUTS_PER_TRIAL) {
                  exitFlag = true;
                  setTimeout(terminateSession, 100);
                }
              } else {
                if (data.response !== null) lastResponse = data.response;
                const ref = refForObservation(_refs, _o);
                data.error = computeResponseError(data.response, ref);
                data.reward = computeResponseReward(data.error);
                trialResponses.push({ observation: _o, value: _val, response: data.response, error: data.error });
                sendCheckpoint?.({
                  phase: PHASES.TRIAL, trialIndex: t, observationIndex: _o, attempt: thisAttempt,
                  response: data.response, timed_out: false, rt: data.rt ?? null,
                  value: _val, true_mean: seq.true_mean, true_std: seq.true_std, true_p: seq.true_p ?? null,
                  qid: seq.qid, error: data.error, reward: data.reward,
                });
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
          total_error: () => _resp.reduce((sum, r) => sum + (r.error ?? 0), 0),
          reward:      () => computeTrialReward(_resp.map(r => r.error)),
          show_performance: showTrialPerformance, is_last: false,
          data: { screen: 'inter_trial', trial: _t } }],
        conditional_function: () => !exitFlag,
      });
      timeline.push({
        timeline: [{ type: InterTrialPlugin, trial_num: _t+2, n_trials: sequences.length,
          duration_ms: btiMs, is_colors: isColors,
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
