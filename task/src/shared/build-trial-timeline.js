/**
 * build-trial-timeline.js
 * Pure JS — builds the trial loop portion of the jsPsych timeline.
 * No jsPsych imports, no DOM, no CSS.
 * Used by timeline-builder.js (production) and test_timeline.mjs (tests).
 */
export function buildTrialTimeline(cfg, plugins, jsPsych, earlyExit) {
  const {
    sequences, sliderDefault, defaultValue, btiMs, tObsMs,
    showSliderValue, showTrialPerformance, MAX_TIMEOUTS_PER_TRIAL = 3,
    distractorType = 'iti_length',
    trialItiMs     = null,  // null = use seq.iti_ms; number = override
  } = cfg;

  const { ItiClockPlugin, TrialObsPlugin, TrialSummaryPlugin,
          InterTrialPlugin, isBinary } = plugins;

  // ITI duration for distract condition
  // Only 'iti_length' extends the ITI; other types ('popup', etc.) use default
  const ITI_DISTRACT_MS = distractorType === 'iti_length' ? 5000 : null;

  let lastResponse = defaultValue, timedOut = false, trialTimeouts = 0;
  let exitFlag = false, lastTrialResponses = [];
  const timeline = [];

  for (let t = 0; t < sequences.length; t++) {
    const seq = sequences[t];
    const trialResponses = [];
    lastTrialResponses   = trialResponses;
    const trialTimeline  = [];

    for (let o = 0; o < seq.values.length; o++) {
      const _o = o, _val = seq.values[o];

      if (o > 0) {
        trialTimeline.push({
          timeline: [{ type: ItiClockPlugin,
            duration_ms: (seq.iti_condition === 'distract' && ITI_DISTRACT_MS !== null)
              ? ITI_DISTRACT_MS : (trialItiMs ?? seq.iti_ms ?? 1000),
            iti_condition:   seq.iti_condition ?? 'control',
            distractor_type: distractorType,
            is_binary:       isBinary,
            data: { screen: 'iti', trial: t, observation: _o,
                    iti_ms: seq.iti_ms ?? 1000,
                    iti_condition:   seq.iti_condition ?? 'control',
                    distractor_type: distractorType } }],
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
            data: { screen: 'observation', trial: t, observation: _o, value: _val,
                    true_mean: seq.true_mean, true_p: seq.true_p ?? null, qid: seq.qid ?? null },
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
                trialResponses.push({ observation: _o, value: _val, response: data.response });
              }
            },
          },
          {
            timeline: [{ type: ItiClockPlugin, duration_ms: trialItiMs ?? seq.iti_ms ?? 1000,
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
      values: [..._seq.values], responses: () => _resp.map(r => r.response),
      show_performance: showTrialPerformance, is_last: true,
      data: { screen: 'inter_trial', trial: sequences.length - 1 } }],
    conditional_function: () => !exitFlag,
  });

  return { timeline, isExited: () => exitFlag };
}
