/**
 * backend-client.js
 * Thin fetch wrapper around the three Supabase Edge Functions
 * (progress-check / progress-append / progress-finish). Deliberately
 * plain fetch, not supabase-js -- the client only ever needs to POST JSON
 * to three fixed endpoints, and supabase-js's default header behavior is
 * exactly what tripped up the new publishable-key format during backend
 * testing (see task_backend/TODO.md's "Env vars / key format" note):
 * sending the key as `Authorization: Bearer` gets rejected as an invalid
 * JWT by the platform's verify_jwt gateway check. Plain fetch sends ONLY
 * the `apikey` header (matching what the Edge Functions themselves check
 * -- see supabase/functions/_shared/auth-check.ts), sidestepping the
 * question entirely rather than fighting a client library's defaults.
 *
 * VITE_SUPABASE_URL / VITE_SUPABASE_PUBLISHABLE_KEY are embedded at build
 * time (Vite's import.meta.env) -- safe to ship in the client bundle, same
 * as any Supabase publishable key; RLS (deny-all for anon/authenticated,
 * see TODO.md decision #1) plus this file only ever hitting the three
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
 * createCheckpointSender -- wraps appendProgress with consecutive-failure
 * tracking. Checkpoints are fire-and-forget from the participant's
 * perspective (same as the old JATOS per-trial appends -- callers don't
 * need to await this before letting the timeline continue), but unlike
 * the old pipeline, a RUN of failures becomes visible to the participant
 * via onWarning rather than vanishing silently -- this is the direct fix
 * for the exact incident this whole backend exists to address (see
 * TODO.md's "Pilot #3" history: per-trial saves failing with zero visible
 * symptom for an entire session).
 *
 * @param {object} opts
 * @param {number} [opts.threshold]     consecutive failures before onWarning fires (default 2)
 * @param {Function} opts.onWarning     called (no args) once threshold is hit
 * @param {Function} [opts.onRecovered] called once a checkpoint succeeds after a warning fired
 * @returns {Function} sendCheckpoint(payload) -- same payload shape as appendProgress
 */
export function createCheckpointSender({ threshold = 2, onWarning, onRecovered } = {}) {
  let consecutiveFailures = 0;
  let warningActive = false;

  return async function sendCheckpoint(payload) {
    try {
      const result = await appendProgress(payload);
      consecutiveFailures = 0;
      if (warningActive) {
        warningActive = false;
        onRecovered?.();
      }
      return result;
    } catch (err) {
      consecutiveFailures++;
      console.error('checkpoint append failed', err);
      if (consecutiveFailures >= threshold && !warningActive) {
        warningActive = true;
        onWarning?.();
      }
      throw err;
    }
  };
}
