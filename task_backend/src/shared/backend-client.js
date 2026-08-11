/**
 * backend-client.js
 * Thin fetch wrapper around the three Supabase Edge Functions
 * (progress-check / progress-append / progress-finish). Deliberately
 * plain fetch, not supabase-js -- the client only ever needs to POST JSON
 * to three fixed endpoints, and supabase-js's default header behavior is
 * exactly what tripped up the new publishable-key format during backend
 * testing (see docs/HISTORY.md's task_backend section, "Env vars / key
 * format" note):
 * sending the key as `Authorization: Bearer` gets rejected as an invalid
 * JWT by the platform's verify_jwt gateway check. Plain fetch sends ONLY
 * the `apikey` header (matching what the Edge Functions themselves check
 * -- see supabase/functions/_shared/auth-check.ts), sidestepping the
 * question entirely rather than fighting a client library's defaults.
 *
 * VITE_SUPABASE_URL / VITE_SUPABASE_PUBLISHABLE_KEY are embedded at build
 * time (Vite's import.meta.env) -- safe to ship in the client bundle, same
 * as any Supabase publishable key; RLS (deny-all for anon/authenticated,
 * see docs/HISTORY.md's task_backend section, decision #1) plus this file
 * only ever hitting the three
 * Edge Function endpoints is what keeps the browser from touching the
 * database directly.
 */

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
const PUBLISHABLE_KEY = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY;
const FUNCTIONS_BASE = `${SUPABASE_URL}/functions/v1`;

async function callFunction(name, body) {
  const res = await fetch(`${FUNCTIONS_BASE}/${name}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', apikey: PUBLISHABLE_KEY },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${name} failed (${res.status}): ${text}`);
  }
  return res.json();
}

/**
 * Called once on page load. See supabase/functions/progress-check for the
 * exact response shape (status/phase/resumeTrialIndex/prolificCode/poolIndex)
 * and the four-way resume branch it implements.
 */
export function checkProgress({ prolificPid, task }) {
  return callFunction('progress-check', { prolific_pid: prolificPid, task });
}

/**
 * One checkpoint row per call -- idempotent upsert on the server side
 * (see supabase/functions/progress-append). `attempt` should only ever
 * increment for a timeout-triggered replay of the SAME trial/observation
 * index -- a plain network retry of one unconfirmed call should reuse the
 * same attempt number so the server-side upsert collapses it, not create
 * a second row.
 */
export function appendProgress({
  prolificPid, task, poolIndex, phase, trialIndex, observationIndex, attempt, ...rest
}) {
  return callFunction('progress-append', {
    prolific_pid: prolificPid, task, pool_index: poolIndex, phase,
    trial_index: trialIndex, observation_index: observationIndex, attempt, ...rest,
  });
}

/**
 * Called exactly once at session end (normal completion or early exit --
 * phase is 'finished' or 'terminated'). expectedTrialCount is only
 * required/checked when phase is 'finished' (see progress-finish).
 */
export function finishProgress({ prolificPid, task, poolIndex, phase, expectedTrialCount }) {
  return callFunction('progress-finish', {
    prolific_pid: prolificPid, task, pool_index: poolIndex, phase, expectedTrialCount,
  });
}

/**
 * createCheckpointSender -- wraps appendProgress with a short retry loop
 * plus consecutive-failure tracking. Checkpoints are still fire-and-
 * forget from the participant's perspective (same as the old JATOS
 * per-trial appends -- callers don't need to await this before letting
 * the timeline continue; the retry loop below lives entirely inside this
 * function and never blocks the caller) -- but unlike the old pipeline,
 * a RUN of failures becomes visible to the participant via onWarning
 * rather than vanishing silently, and now each individual checkpoint gets
 * more than one attempt before being counted as a failure at all -- this
 * is the direct fix for the exact incident this whole backend exists to
 * address (see docs/HISTORY.md's task_backend section, "Pilot #3" history:
 * per-trial saves failing with
 * zero visible symptom for an entire session), extended after a real
 * pilot session lost 11/480 observations to single-attempt fire-and-
 * forget calls that never got a second chance (see chat history).
 *
 * Retries are plain network retries of the SAME unconfirmed call, not a
 * participant-facing timeout-retry (that's a different concept entirely
 * -- see appendProgress's own docstring on `attempt`). `attempt` is NOT
 * incremented between retry attempts here, by design: the server-side
 * upsert already collapses duplicate submissions of the same
 * (pid, task, phase, trial_index, observation_index, attempt) into one
 * row, so retrying the identical payload is safe and can never create a
 * duplicate/extra row.
 *
 * @param {object} opts
 * @param {number} [opts.threshold]     consecutive failures (after retries exhausted) before onWarning fires (default 2)
 * @param {number} [opts.maxAttempts]   total attempts per checkpoint before giving up (default 3)
 * @param {number[]} [opts.retryDelaysMs] delay before each retry, ms (default [300, 800] -- one entry per retry, i.e. maxAttempts-1 entries)
 * @param {Function} opts.onWarning     called (no args) once threshold is hit
 * @param {Function} [opts.onRecovered] called once a checkpoint succeeds after a warning fired
 * @returns {Function} sendCheckpoint(payload) -- same payload shape as appendProgress
 */
export function createCheckpointSender({
  threshold = 2, maxAttempts = 3, retryDelaysMs = [300, 800], onWarning, onRecovered,
} = {}) {
  let consecutiveFailures = 0;
  let warningActive = false;

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  return async function sendCheckpoint(payload) {
    let lastErr;
    for (let attemptNum = 1; attemptNum <= maxAttempts; attemptNum++) {
      try {
        const result = await appendProgress(payload);
        consecutiveFailures = 0;
        if (warningActive) {
          warningActive = false;
          onRecovered?.();
        }
        return result;
      } catch (err) {
        lastErr = err;
        console.error(`checkpoint append failed (attempt ${attemptNum}/${maxAttempts})`, err);
        if (attemptNum < maxAttempts) {
          await sleep(retryDelaysMs[attemptNum - 1] ?? retryDelaysMs[retryDelaysMs.length - 1]);
        }
      }
    }
    // Every attempt failed -- NOW it counts as one real failure.
    consecutiveFailures++;
    if (consecutiveFailures >= threshold && !warningActive) {
      warningActive = true;
      onWarning?.();
    }
    throw lastErr;
  };
}
