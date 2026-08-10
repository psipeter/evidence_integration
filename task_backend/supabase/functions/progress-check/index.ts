// progress-check/index.ts
//
// Called on page load. Given { prolific_pid, task }, returns where this
// participant should resume -- per task_backend/TODO.md's four-way resume
// branch (trial-boundary resume, not exact-observation; see TODO.md for
// the full rationale, including why welcome/consent never block a fresh
// start).
//
// Request body:  { prolific_pid: string, task: 'numbers' | 'colors' }
//
// Response body: {
//   status: 'new' | 'in_progress' | 'finished' | 'terminated',
//   phase: 'tutorial' | 'trial' | 'finished' | 'terminated' | null,
//   resumeTrialIndex: number | null,  // start-of-trial index to resume at;
//                                      // null means "tutorial" or "full restart"
//   prolificCode: string | null,      // only set for 'finished'/'terminated'
//   poolIndex: number | null,
// }

import { handleOptions, withCors } from '../_shared/cors.ts';
import { checkApiKey } from '../_shared/auth-check.ts';
import { supabaseAdmin } from '../_shared/supabase-admin.ts';
import { PROLIFIC_CODES } from '../_shared/prolific-codes.ts';
import { jsonResponse } from '../_shared/json.ts';

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

  const prolific_pid = body.prolific_pid;
  const task = body.task;

  if (typeof prolific_pid !== 'string' || !prolific_pid) {
    return withCors(jsonResponse({ error: 'prolific_pid is required' }, 400));
  }
  if (typeof task !== 'string' || !(task in PROLIFIC_CODES)) {
    return withCors(jsonResponse({ error: 'task must be numbers or colors' }, 400));
  }

  // Check for an existing finished/terminated marker FIRST, independent
  // of whether it's the single latest row overall -- NOT the same check
  // as "latest.phase === finished" below would be. Once a terminal
  // marker is written, it must stay authoritative no matter what else
  // gets written afterward (e.g. finish-session.js's own catch-up resend
  // of missing observations writes trial-phase rows AFTER the finished
  // marker, which would otherwise get a higher id and make THIS query's
  // old "just look at the single latest row" logic think the session was
  // still in progress -- a real bug caught during a review pass, not
  // hypothetical; see chat history). This check is deliberately
  // independent of row recency for exactly that reason.
  const { data: terminal, error: terminalErr } = await supabaseAdmin
    .from('events')
    .select('phase, pool_index')
    .eq('prolific_pid', prolific_pid)
    .eq('task', task)
    .in('phase', ['finished', 'terminated'])
    .order('id', { ascending: false })
    .limit(1)
    .maybeSingle();

  if (terminalErr) {
    console.error('progress-check: terminal-marker query failed', terminalErr);
    return withCors(jsonResponse({ error: 'database error' }, 500));
  }

  if (terminal) {
    const codes = PROLIFIC_CODES[task];
    return withCors(jsonResponse({
      status: terminal.phase,
      phase: terminal.phase,
      resumeTrialIndex: null,
      prolificCode: terminal.phase === 'finished' ? codes.completion : codes.earlyExit,
      poolIndex: terminal.pool_index,
    }));
  }

  const { data: latest, error } = await supabaseAdmin
    .from('events')
    .select('phase, trial_index, observation_index, pool_index')
    .eq('prolific_pid', prolific_pid)
    .eq('task', task)
    .order('id', { ascending: false })
    .limit(1)
    .maybeSingle();

  if (error) {
    console.error('progress-check: latest-row query failed', error);
    return withCors(jsonResponse({ error: 'database error' }, 500));
  }

  // No rows at all, or only ever reached welcome -- full restart. Pool
  // index isn't preserved here deliberately: it's a deterministic hash of
  // prolific_pid (see poolIndexForParticipant), so the client recomputing
  // it gives the identical value with no coordination needed.
  if (!latest || latest.phase === 'welcome') {
    return withCors(jsonResponse({
      status: 'new',
      phase: null,
      resumeTrialIndex: null,
      prolificCode: null,
      poolIndex: null,
    }));
  }

  // Reached consent but never entered the tutorial -- resume straight into
  // the tutorial, don't replay welcome/consent.
  if (latest.phase === 'consent' || latest.phase === 'tutorial') {
    return withCors(jsonResponse({
      status: 'in_progress',
      phase: 'tutorial',
      resumeTrialIndex: null,
      prolificCode: null,
      poolIndex: latest.pool_index,
    }));
  }

  // phase === 'trial' -- check whether the current trial's last
  // observation (index 14) is already logged, to decide whether to resume
  // at this trial (incomplete) or the next one (complete).
  const { data: lastObsRow, error: lastObsErr } = await supabaseAdmin
    .from('events')
    .select('id')
    .eq('prolific_pid', prolific_pid)
    .eq('task', task)
    .eq('phase', 'trial')
    .eq('trial_index', latest.trial_index)
    .eq('observation_index', 14)
    .limit(1)
    .maybeSingle();

  if (lastObsErr) {
    console.error('progress-check: trial-completeness query failed', lastObsErr);
    return withCors(jsonResponse({ error: 'database error' }, 500));
  }

  const resumeTrialIndex = lastObsRow ? (latest.trial_index as number) + 1 : latest.trial_index;

  return withCors(jsonResponse({
    status: 'in_progress',
    phase: 'trial',
    resumeTrialIndex,
    prolificCode: null,
    poolIndex: latest.pool_index,
  }));
});
