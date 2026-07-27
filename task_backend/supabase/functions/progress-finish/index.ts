// progress-finish/index.ts
//
// Called exactly once at the end of a session -- normal completion or a
// timeout-budget early exit (see finish-session.js / create-early-exit.js
// on the client, which is where the two paths both funnel through today).
// Writes the terminal 'finished'/'terminated' marker row and hands back
// the participant's Prolific code.
//
// For phase === 'finished' specifically, sanity-checks that the expected
// number of trial observations actually made it into `events` before
// responding. Per TODO.md: a mismatch is logged for manual reconciliation
// but does NOT block the code -- don't hold up a real participant's
// payment for our own bug. (Not checked for 'terminated' -- an incomplete
// trial is expected/correct there by definition.)
//
// Request body: {
//   prolific_pid: string,
//   task: 'numbers' | 'colors',
//   pool_index: number,
//   phase: 'finished' | 'terminated',
//   expectedTrialCount?: number   // required when phase === 'finished'
// }
// Response body: { prolificCode: string, dataComplete: boolean | null } | { error: string }

import { handleOptions, withCors } from '../_shared/cors.ts';
import { checkApiKey } from '../_shared/auth-check.ts';
import { supabaseAdmin } from '../_shared/supabase-admin.ts';
import { PROLIFIC_CODES } from '../_shared/prolific-codes.ts';
import { jsonResponse } from '../_shared/json.ts';

const OBSERVATIONS_PER_TRIAL = 15;

Deno.serve(async (req: Request) => {
  const preflight = handleOptions(req);
  if (preflight) return preflight;

  const unauthorized = checkApiKey(req);
  if (unauthorized) return withCors(unauthorized);

  if (req.method !== 'POST') {
    return withCors(jsonResponse({ error: 'method not allowed' }, 405));
  }

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return withCors(jsonResponse({ error: 'invalid JSON body' }, 400));
  }

  const { prolific_pid, task, pool_index, phase, expectedTrialCount } = body;

  if (typeof prolific_pid !== 'string' || !prolific_pid) {
    return withCors(jsonResponse({ error: 'prolific_pid is required' }, 400));
  }
  if (typeof task !== 'string' || !(task in PROLIFIC_CODES)) {
    return withCors(jsonResponse({ error: 'task must be numbers or colors' }, 400));
  }
  if (phase !== 'finished' && phase !== 'terminated') {
    return withCors(jsonResponse({ error: "phase must be 'finished' or 'terminated'" }, 400));
  }
  if (typeof pool_index !== 'number' || !Number.isInteger(pool_index)) {
    return withCors(jsonResponse({ error: 'pool_index must be an integer' }, 400));
  }

  let dataComplete: boolean | null = null;

  if (phase === 'finished') {
    if (typeof expectedTrialCount !== 'number' || !Number.isInteger(expectedTrialCount)) {
      return withCors(jsonResponse(
        { error: 'expectedTrialCount (integer) is required when phase is finished' }, 400,
      ));
    }

    const { data: rows, error: rowsErr } = await supabaseAdmin
      .from('events')
      .select('trial_index, observation_index')
      .eq('prolific_pid', prolific_pid)
      .eq('task', task)
      .eq('phase', 'trial');

    if (rowsErr) {
      console.error('progress-finish: row-count query failed', rowsErr);
      return withCors(jsonResponse({ error: 'database error' }, 500));
    }

    // Dedupe by (trial_index, observation_index) -- multiple `attempt`s at
    // the same checkpoint (timeout replays) are one logical observation
    // for this count, not several.
    const distinctPairs = new Set((rows ?? []).map((r) => `${r.trial_index}-${r.observation_index}`));
    const actualCount = distinctPairs.size;
    const expectedCount = expectedTrialCount * OBSERVATIONS_PER_TRIAL;
    dataComplete = actualCount === expectedCount;

    if (!dataComplete) {
      console.warn(
        `progress-finish: row-count mismatch for prolific_pid=${prolific_pid} task=${task}: ` +
        `expected ${expectedCount} trial-observation rows (${expectedTrialCount} trials x ${OBSERVATIONS_PER_TRIAL}), found ${actualCount}.`,
      );
    }
  }

  const { error: upsertErr } = await supabaseAdmin
    .from('events')
    .upsert({
      prolific_pid,
      task,
      pool_index,
      phase,
      trial_index: -1,
      observation_index: -1,
      attempt: 0,
      updated_at: new Date().toISOString(),
    }, {
      onConflict: 'prolific_pid,task,phase,trial_index,observation_index,attempt',
    });

  if (upsertErr) {
    console.error('progress-finish: terminal-marker upsert failed', upsertErr);
    return withCors(jsonResponse({ error: 'database error' }, 500));
  }

  const codes = PROLIFIC_CODES[task];
  const prolificCode = phase === 'finished' ? codes.completion : codes.earlyExit;

  return withCors(jsonResponse({ prolificCode, dataComplete }));
});
