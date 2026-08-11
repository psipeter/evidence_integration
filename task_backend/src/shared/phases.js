/**
 * phases.js
 * The six checkpoint phases our Supabase backend tracks (see
 * docs/HISTORY.md's task_backend section, schema DDL) -- deliberately
 * coarser than the old JATOS pipeline's per-screen tagging (17 distinct
 * `screen` values found during the port review pass). Only welcome/
 * consent/tutorial/trial ever need a checkpoint call; the rest of the old
 * screen taxonomy (ITI resets, summary screens, tutorial intro/complete
 * transitions) needs no network call at all under the trial-boundary-
 * resume design -- see that same section's decision #3.
 *
 * finished/terminated are the two session-ending outcomes -- kept as
 * separate, explicitly-named phases (not one generic "done" flag) since
 * they mean different things downstream: finished pays the full
 * completion code, terminated pays the partial/early-exit code. See
 * finish-session.js (finishSession) and create-terminate-session.js
 * (createTerminateSession) for where each actually gets used.
 */
export const PHASES = Object.freeze({
  WELCOME: 'welcome',
  CONSENT: 'consent',
  TUTORIAL: 'tutorial',
  TRIAL: 'trial',
  FINISHED: 'finished',
  TERMINATED: 'terminated',
});
