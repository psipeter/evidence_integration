// progress-append/index.ts
//
// Idempotent upsert of one checkpoint row per call. Used for tutorial/trial
// observations AND welcome/consent bookkeeping events -- one endpoint, the
// `phase` field distinguishes them.
//
// IMPORTANT sentinel rule (see TODO.md decision log + supabase-admin.ts):
// Postgres unique constraints treat two NULLs as NOT equal, so a `null`
// sentinel would silently break idempotency. This function does NOT trust
// the client to send the right sentinel -- it derives trial_index /
// observation_index itself from `phase`, overriding whatever the client
// sent for indices that phase doesn't use:
//   welcome/consent/finished/terminated  -> trial_index = -1, observation_index = -1
//   tutorial                             -> trial_index = -1, observation_index = client's value
//   trial                                -> trial_index = client's value, observation_index = client's value
//
// Request body: {
//   prolific_pid, task, pool_index, phase,
//   trial_index?, observation_index?, attempt?,
//   response?, timed_out?, rt?, value?, true_mean?, true_std?, true_p?,
//   qid?, error?, reward?
// }
// Response body: { ok: true } | { error: string }

import { handleOptions, withCors } from '../_shared/cors.ts';
import { checkApiKey } from '../_shared/auth-check.ts';
import { supabaseAdmin } from '../_shared/supabase-admin.ts';
import { jsonResponse } from '../_shared/json.ts';

const VALID_PHASES = ['welcome', 'consent', 'tutorial', 'trial', 'finished', 'terminated'];
const VALID_TASKS = ['numbers', 'colors'];

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

  const { prolific_pid, task, phase, pool_index } = body;

  if (typeof prolific_pid !== 'string' || !prolific_pid) {
    return withCors(jsonResponse({ error: 'prolific_pid is required' }, 400));
  }
  if (typeof task !== 'string' || !VALID_TASKS.includes(task)) {
    return withCors(jsonResponse({ error: `task must be one of ${VALID_TASKS.join(', ')}` }, 400));
  }
  if (typeof phase !== 'string' || !VALID_PHASES.includes(phase)) {
    return withCors(jsonResponse({ error: `phase must be one of ${VALID_PHASES.join(', ')}` }, 400));
  }
  if (typeof pool_index !== 'number' || !Number.isInteger(pool_index)) {
    return withCors(jsonResponse({ error: 'pool_index must be an integer' }, 400));
  }

  let trial_index = -1;
  let observation_index = -1;

  if (phase === 'trial') {
    if (typeof body.trial_index !== 'number' || !Number.isInteger(body.trial_index)) {
      return withCors(jsonResponse({ error: 'trial_index must be an integer when phase is trial' }, 400));
    }
    if (typeof body.observation_index !== 'number' || !Number.isInteger(body.observation_index)) {
      return withCors(jsonResponse({ error: 'observation_index must be an integer when phase is trial' }, 400));
    }
    trial_index = body.trial_index;
    observation_index = body.observation_index;
  } else if (phase === 'tutorial') {
    if (typeof body.observation_index !== 'number' || !Number.isInteger(body.observation_index)) {
      return withCors(jsonResponse({ error: 'observation_index must be an integer when phase is tutorial' }, 400));
    }
    observation_index = body.observation_index;
    // trial_index stays -1 -- deliberately ignoring any trial_index the
    // client might send for a tutorial row.
  }
  // welcome/consent/finished/terminated: both stay -1, any client-sent
  // trial_index/observation_index for these phases is ignored.

  const attempt = typeof body.attempt === 'number' && Number.isInteger(body.attempt) ? body.attempt : 0;

  const row = {
    prolific_pid,
    task,
    pool_index,
    phase,
    trial_index,
    observation_index,
    attempt,
    response: numOrNull(body.response),
    timed_out: typeof body.timed_out === 'boolean' ? body.timed_out : null,
    rt: numOrNull(body.rt),
    value: numOrNull(body.value),
    true_mean: numOrNull(body.true_mean),
    true_std: numOrNull(body.true_std),
    true_p: numOrNull(body.true_p),
    qid: numOrNull(body.qid),
    error: numOrNull(body.error),
    reward: numOrNull(body.reward),
    updated_at: new Date().toISOString(),
  };

  const { error } = await supabaseAdmin
    .from('events')
    .upsert(row, {
      onConflict: 'prolific_pid,task,phase,trial_index,observation_index,attempt',
    });

  if (error) {
    console.error('progress-append: upsert failed', error);
    return withCors(jsonResponse({ error: 'database error' }, 500));
  }

  return withCors(jsonResponse({ ok: true }));
});

function numOrNull(value: unknown): number | null {
  return typeof value === 'number' ? value : null;
}
