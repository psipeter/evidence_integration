# task_backend/TODO.md — Supabase backend build plan

**Read this in full before writing any code in task_backend/.** This is the
working plan for the Supabase-backed backend described in CLAUDE.md's
"Pilot #3 real-participant incidents..." and "Own-backend decision
(Supabase)" sections — read those first for the full incident history and
why this exists at all; this document is where the *design* was actually
settled (several details here supersede/refine what's sketched in
CLAUDE.md's "Architecture" and "Next steps" subsections, per a planning
conversation — CLAUDE.md itself should get a pointer to this file once
building starts).

The existing JATOS-based `task/` pipeline is NOT touched by any of this and
remains the live/fallback system throughout. Everything here is built fresh
in `task_backend/`.

---

## Why (one paragraph — see CLAUDE.md for full evidence)

Two real Prolific participants hit genuine JATOS-level failures during pilot
#3 (a reload-triggered session death mid-tutorial; a GeneralSingle
cookie/link-prefetch collision blocking a legitimate first attempt). A
follow-up empirical test confirmed a third, worse gap: per-trial saves can
fail silently for an entire session with zero participant-visible symptom.
Gorilla/Cognition.run/Labvanced were evaluated as hosted fixes and rejected
(see CLAUDE.md). Decision: a small backend on Supabase, keyed on
`prolific_pid` rather than a browser cookie, giving real reload/resume
instead of restart-or-block.

---

## Decisions settled (planning conversation, before any code)

These resolve the open flags raised when the plan in CLAUDE.md was first
reviewed. Treat these as settled unless something concrete surfaces during
implementation that contradicts them — don't silently re-open them.

1. **Access model: locked down, Edge-Functions-only.** There is no login
   system, so `prolific_pid` is never a verified identity claim — per-row
   RLS keyed on it would be security theater (anyone with the anon key
   could claim any `prolific_pid`). Instead: RLS is **enabled** on every
   table with **zero policies** for the `anon`/`authenticated` roles (deny
   all direct access, including read). The three Edge Functions are the
   *only* way in or out, using the service-role key server-side (which
   bypasses RLS by design). The browser never talks to the database
   directly, only to the Edge Function endpoints.

2. **Single table, not two.** Originally planned as separate `progress` +
   `events` tables. Collapsed into one `events` table: "current progress"
   is just the latest row per `(prolific_pid, task)`, queried directly —
   nothing to keep in sync, no dual-write transaction to get right, no way
   for the two to disagree. `progress-check` does one query
   (`ORDER BY id DESC LIMIT 1`), not a join.

3. **Resume granularity: trial-boundary, not exact-observation.** On
   reload, resume at the start of the current incomplete trial (or the
   tutorial, if still there) — not the exact observation. A trial/tutorial
   is only 15 observations, cheap to redo; this avoids reconstructing
   mid-trial UI state (slider position, in-flight animation) from a DB row,
   which is a lot of fragile surface area for very little participant time
   saved. Every observation is still logged individually for analysis —
   only the *resume point* is coarser than the *log*.

4. **`phase` covers welcome/consent too, for bookkeeping.** Originally
   planned to only log tutorial/trial/terminal events (welcome/consent
   didn't need to be *resumable*). Decided to log them anyway — cheap, and
   gives direct evidence of "did they open the link" / "did they actually
   consent" rather than inferring it from later rows existing.

5. **Timeout retries get their own attempt counter, not silent overwrite.**
   A genuine network-retry of one unconfirmed request should overwrite
   itself (that's the whole point of idempotent upsert). A timeout-
   triggered *replay* of the same observation is a different event and
   must not clobber the first attempt's row. Both are handled by an
   `attempt` column in the uniqueness key (see schema).

6. **Row shape mirrors the current JATOS lean-row design, not a JSON blob.**
   Stimulus/response fields are real flattened columns, not a `payload
   jsonb` bag — `SELECT * FROM events WHERE phase='trial'` should already
   look like a usable dataframe, close to a straight rewrite of
   `parse_results.py` rather than "unpack JSON, then reshape." Downstream
   analysis/plotting scripts get rewritten later against this shape — not
   a blocker for backend construction, just noted so the schema is chosen
   with that rewrite in mind.

---

## Schema

### `events`

One append-only table. Every row is either a checkpoint (tutorial/trial
observation) or a bookkeeping/terminal marker (welcome/consent/finished/
terminated).

| Column | Type | Notes |
|---|---|---|
| `id` | `bigserial primary key` | Use `ORDER BY id DESC LIMIT 1` to find "latest state" — don't rely on `created_at` ordering under retries. |
| `prolific_pid` | `text not null` | Real Prolific ID, or a dev/local fallback — **open item, see below** (the old `pilot_${jatos.workerId}` fallback was JATOS-specific and needs a replacement convention). |
| `task` | `text not null check (task in ('numbers','colors'))` | |
| `pool_index` | `int not null` | Deterministic hash of `prolific_pid` — mechanism unchanged from current `poolIndexForParticipant`, just stored per-row same as now for self-containment. |
| `phase` | `text not null check (phase in ('welcome','consent','tutorial','trial','finished','terminated'))` | |
| `trial_index` | `int` | 0–31 for `phase='trial'`; `-1` sentinel for `tutorial`/`welcome`/`consent`/`finished`/`terminated`. **Not `null`** -- Postgres unique constraints treat two `NULL`s as distinct, not equal, so a `null` sentinel would silently break idempotency for exactly the phase (tutorial) this backend exists to fix. Caught during Edge Function implementation; fixed in code (`progress-append` derives/overrides this from `phase` rather than trusting the client), noted here so the doc matches. |
| `observation_index` | `int` | 0–14 for `tutorial`/`trial`; `-1` sentinel for `welcome`/`consent`/`finished`/`terminated`. Same not-null reasoning as `trial_index` above. |
| `attempt` | `int not null default 0` | Increment on a timeout-triggered replay of the same `(trial_index, observation_index)`. Always `0` for non-observation phases. |
| `response` | `numeric` | nullable |
| `timed_out` | `boolean` | nullable |
| `rt` | `numeric` | nullable |
| `value` | `numeric` | nullable |
| `true_mean` | `numeric` | nullable, numbers only |
| `true_std` | `numeric` | nullable, numbers only |
| `true_p` | `numeric` | nullable, colors only |
| `qid` | `int` | nullable |
| `error` | `numeric` | nullable, per-observation bonus error |
| `reward` | `numeric` | nullable, per-observation bonus reward |
| `created_at` | `timestamptz not null default now()` | |
| `updated_at` | `timestamptz not null default now()` | bump on every upsert |

**Unique constraint** (the idempotency mechanism):
`unique (prolific_pid, task, phase, trial_index, observation_index, attempt)`

**Indexes**: at minimum `(prolific_pid, task, id desc)` to make the
progress-check "latest row" lookup fast.

```sql
create table events (
  id                 bigserial primary key,
  prolific_pid       text not null,
  task               text not null check (task in ('numbers','colors')),
  pool_index         int not null,
  phase              text not null check (phase in
                        ('welcome','consent','tutorial','trial','finished','terminated')),
  trial_index        int,
  observation_index  int,
  attempt            int not null default 0,
  response           numeric,
  timed_out          boolean,
  rt                 numeric,
  value              numeric,
  true_mean          numeric,
  true_std           numeric,
  true_p             numeric,
  qid                int,
  error              numeric,
  reward             numeric,
  created_at         timestamptz not null default now(),
  updated_at         timestamptz not null default now(),
  unique (prolific_pid, task, phase, trial_index, observation_index, attempt)
);

create index events_latest_idx on events (prolific_pid, task, id desc);

alter table events enable row level security;
-- Deliberately NO policies created for anon/authenticated — default-deny.
-- Only the service-role key (used server-side in Edge Functions) can
-- read/write. Do not add a permissive policy "to make testing easier."

-- IMPORTANT (found during step 3h testing): disabling "Automatically
-- expose new tables" in the project's Data API settings withholds base
-- table-level GRANTs from ALL Data API roles, including service_role --
-- not just anon/authenticated as decision #1 above assumed. service_role's
-- RLS-bypass is a separate property from having table privileges at all;
-- it still needs an explicit GRANT. anon/authenticated deliberately get
-- nothing (verified via curl in step 2); service_role needs exactly this:
grant select, insert, update on public.events to service_role;
grant usage, select on sequence public.events_id_seq to service_role;

-- SAME GAP, FOUND AGAIN LATER (building the test suite): none of the
-- Edge Functions ever issue a DELETE, so this went unnoticed until the
-- test suite's own cleanup step (a DIRECT DELETE against the REST API
-- using the secret key, for removing test rows -- see tests/helpers.mjs's
-- cleanupTestRows) hit a real 403. Same root cause as above, just a
-- privilege nothing had exercised yet:
grant delete on public.events to service_role;
```

---

## Resume logic (`progress-check`)

Query the latest row for `(prolific_pid, task)`. Branch on `phase`:

- `finished` / `terminated` → return that status + the participant's
  Prolific code (completion vs. earlyExit per `PROLIFIC_CODES`). Client
  skips the timeline entirely.
- `trial` → resume at the start of `trial_index` if its last observation
  (14) isn't logged yet, else `trial_index + 1`.
- `tutorial` → resume at the start of the tutorial.
- `consent` (reached but never entered tutorial) → resume at the start of
  the tutorial, skip welcome/consent again.
- `welcome` only, or no rows at all → full run from welcome.

---

## Open items to resolve during implementation (not blocking table creation)

- ~~**Dev/local fallback for `prolific_pid`.**~~ Resolved: falls back to
  `dev_${Date.now()}` (timestamped, so repeated local runs don't collide)
  when `?PROLIFIC_PID=` isn't present -- see `timeline-builder.js`'s
  `buildAndRun`. No `pilot_`-style prefix; there's no JATOS worker ID to
  fall back to anymore.
- ~~**Consecutive-failure threshold for the visible warning**~~ Resolved:
  `backend-client.js`'s `createCheckpointSender` defaults to a threshold of
  2 consecutive failures, then shows a non-blocking banner
  (`timeline-builder.js`'s `showSaveWarning`/`hideSaveWarning`) until a
  checkpoint succeeds again. Verified for real during the numbers/colors
  rename: a genuine mismatch between client and backend task strings
  produced real `progress-append` failures, and the warning path fired
  correctly rather than failing silently -- an unplanned but useful live
  test of this exact mechanism.
- ~~**`progress-finish` mismatch handling**~~ Resolved as planned: still
  delivers the code, logs a `console.warn` (visible in the Edge Function's
  dashboard logs) for manual reconciliation rather than blocking payment.
- ~~**CORS on Edge Functions**~~ — done in `_shared/cors.ts`, applied in all three functions.
- **Env vars / key format (resolved during implementation, July 2026):**
  the new publishable/secret keys are NOT JWTs, so the platform's default
  `verify_jwt` gateway check rejects them outright if sent as
  `Authorization: Bearer`. All three functions are deployed with
  `verify_jwt = false` (already the CLI's scaffolded default) and do their
  own lightweight check instead (`_shared/auth-check.ts`): the caller must
  send our project's publishable key on the `apikey` header specifically,
  not `Authorization`. Resolved on the client too: `backend-client.js`
  uses plain `fetch` with only the `apikey` header, deliberately not
  `supabase-js`, to sidestep its default header behavior entirely.
  Key values are read from the new JSON-dict env vars
  (`SUPABASE_SECRET_KEYS`/`SUPABASE_PUBLISHABLE_KEYS`, keyed `"default"`),
  falling back to the legacy `SUPABASE_SERVICE_ROLE_KEY`/`SUPABASE_ANON_KEY`
  vars if those aren't set — see `_shared/supabase-admin.ts` and
  `_shared/auth-check.ts`.
- The client should only ever call the three Edge Function endpoints,
  never the table REST endpoints directly — enforced by the deny-all RLS,
  not by convention alone (verified in step 2).

---

## Terminology: numbers/colors, not continuous/binary

The two tasks are referred to as **numbers** and **colors** throughout
`task_backend` (file names, directory names, the `task` column/parameter,
class names, CSS classes, the `isColors` flag) -- renamed from the
old JATOS-era `continuous`/`binary` naming for consistency with the
participant-facing labels already used on the welcome screen and Prolific
itself. This is a deliberate, `task_backend`-only rename: the untouched
`task/` (JATOS) pipeline still uses `continuous`/`binary` internally, and
the two codebases are not expected to share terminology.

One exception, worth remembering if this ever needs re-doing: `bonus-
continuous.js` did NOT mechanically become `bonus-numbers.js` -- it's
shared by both tasks, so it was renamed to the task-neutral `scoring.js`
instead. A blind find-and-replace will get this wrong; fix it by hand.

---

## Next steps

1. ~~Create the Supabase project; run the `events` DDL~~ — done, verified
   (deny-all RLS confirmed via curl, `service_role` grants added after
   catching the "Automatically expose new tables" gap -- see the DDL's
   own comments above).
2. ~~Write the three Edge Functions~~ — done, deployed, and tested directly
   (idempotent upsert confirmed via a real duplicate-call test: retrying
   the same checkpoint left exactly one row, not two).
3. ~~Build client-side resumability~~ — done (`timeline-builder.js`'s
   `buildAndRun`, now `async`, calls `progress-check` first and branches
   into new/resume-tutorial/resume-trial/already-done). Verified with a
   real headless-browser test: the same participant ID, reloaded, landed
   directly on `tutorial_intro` without re-showing welcome/consent.
4. ~~Swap the JATOS save calls for the new endpoints~~ — done. The old
   blanket `on_trial_finish` JATOS-append hook was dropped entirely rather
   than ported (see the port review-pass inventory in chat history for
   why: only 3-4 of the old pipeline's 17 screen tags ever needed a
   network call under the new schema). `jatos-shim.js` and `PROLIFIC_CODES`
   no longer exist client-side at all -- the server hands back the code
   directly from `progress-finish`.
5. ~~Set up the weekly backup process~~ -- DECIDED AGAINST: no scheduled/
   automated backup. Instead, manual backup on demand once real
   participant data actually exists in the table (ask Claude for the
   exact steps at that point -- a `pg_dump` against the project's
   connection string, or the dashboard's own backup/export feature,
   whichever is simpler at the time). Revisit this decision if data
   volume or collection duration ever makes "manual, occasional" feel
   inadequate.
6. ~~Test the three failure modes directly~~ — (a) and (b) now fully done;
   (c) still open, see below.
   - **(a) mid-session resume, including a real trial-boundary case**: DONE,
     verified with a real headless-browser test — completed the tutorial,
     fully completed trial 0, advanced to trial 1 within the same session,
     reloaded as the same participant, and landed exactly at trial 1
     observation 0 (`progress-check` returned `resumeTrialIndex: 1`,
     matching). Also added and verified a UX gap found while testing:
     resuming into the trial loop previously showed NO transition at all
     (silently dropping the participant into an observation) — now always
     shows the same "Trial X / 32 — generating new sequence…" screen used
     for normal between-trial transitions, with the correct trial number,
     whether the session is fresh or resuming
     (`timeline-builder.js` — the `InterTrialPlugin` push moved out of the
     `!skipToTrials` branch to run unconditionally before the trial loop).
   - **The `attempt` counter under a real timeout**: DONE, verified —
     forced three real observation timeouts in a row on the same
     `(trial_index=0, observation_index=0)`; confirmed via the actual
     outgoing request bodies that `attempt` incremented `0 → 1 → 2` with
     `timed_out=true` each time, the same observation replayed each time
     (never silently advancing), and the third timeout correctly
     triggered `terminateSession` (landed on the `terminated` screen).
     Also surfaced that `ItiClockPlugin`'s `timed_out=true` branch
     requires a manual "Repeat" click rather than auto-advancing — not a
     bug, a deliberate anti-tab-visibility-exploit design already in the
     ported code, just one the test needed to account for.
   - **(c) a completed/terminated participant re-visiting the link**:
     DONE, verified. Seeded a fully-complete trial loop and a `terminated`
     row directly via the real API (avoids re-running 480 real
     observations through the UI for something that's really a
     `progress-check` branch test), then loaded each as a fresh page
     load: a finished participant sees "You already completed this
     study" + their completion code; a terminated participant sees "This
     study session already ended" + their (different) earlyExit code.
     Neither ever ran the jsPsych timeline.

**Follow-on UX fix, found while verifying 6c**: the original redirect
for real Prolific participants (`window.location.href = ...`) fired
immediately on success with no on-screen fallback -- if that redirect
ever failed (network hiccup, popup blocker, Prolific-side issue), the
participant would have had no way to know their own code. Fixed via a
new shared `renderCompletionScreen` (`finish-session.js`), used by all
three session-ending paths (finish, terminate, returning-participant):
the code is always shown as visible, selectable text, with a "Continue
to Prolific" button AND a ~4s auto-redirect -- closing the tab or a
failed redirect can never lose the code. Non-Prolific (dev/pilot)
participants see the code too now, for local-testing convenience.
Verified for real: all three paths (finish/terminate/returning) show the
correct, DIFFERENT codes (completion vs. earlyExit) as visible text.

**Real bug caught and fixed while building the above test**: `backend-
client.js`'s `appendProgress` passes extra checkpoint fields through via
`...rest` verbatim — no camelCase-to-snake_case transform. But
`build-trial-timeline.js`/`build-tutorial-timeline.js` were calling
`sendCheckpoint` with `timedOut`/`trueMean`/`trueStd`/`truep` (camelCase),
while `progress-append`'s Edge Function only recognizes the snake_case
`timed_out`/`true_mean`/`true_std`/`true_p`. Every trial/tutorial
checkpoint written before this fix has `NULL` for those four columns,
regardless of the real values — a real, silent data-integrity gap, not a
cosmetic one. Fixed in both files (all four call sites); confirmed fixed
via the real request-body inspection in the test above. **Any rows written
before this fix are wrong on these four columns specifically** — safe to
ignore for pure test/smoke-test rows (already cleaned up), but worth
remembering if this ever needs a real data audit.

Work through the remaining items (dedicated test suite, weekly backup,
hosting deployment) — **sequentially, one at a time** — confirm each
before moving to the next, per how this project has been run so far.

---

## Hosting (a gap nobody noticed until asked about it directly)

**This was never part of the original plan.** CLAUDE.md's "Pilot #3"
investigation and the resulting plan (this doc) were entirely about *why
participant data was being silently lost* -- replacing JATOS's data-saving
job. Nobody -- not CLAUDE.md, not this doc, not several sessions of
building -- ever explicitly asked "who serves the HTML/JS files once
JATOS isn't in the picture?" JATOS/MindProbe did two jobs at once (serve
static files AND track participants/save data); only the second job was
ever discussed. This surfaced only when asked directly, well after the
client port was built and tested.

**Why it doesn't conflict with the original reasoning, though**: Gorilla/
Cognition.run/Labvanced were rejected for being full-stack platforms that
wanted to own task logic, participant tracking, AND data storage all at
once. A plain static host never sees a participant response and never
touches the database -- it's a dumb file server, not a rival to the
Supabase decision. Its failure mode is also qualitatively different from
what motivated this whole redesign: if a static host is down, the page
simply doesn't load, immediately and unmistakably, before a session ever
starts -- nothing like JATOS's actual failure mode (a session running
normally while responses silently vanish underneath it).

**Decision path** (options considered, in order):
1. Netlify / Vercel / Cloudflare Pages / GitHub Pages all evaluated as
   free-tier static hosts for the combined Vite build (see "Build output"
   below).
2. Initially leaned GitHub Pages (repo already on GitHub for version
   control -- zero new account needed) -- until discovering GitHub Pages
   is free only for PUBLIC repos; this repo was private at the time.
   Pivoted to Cloudflare Pages (free regardless of repo visibility, no
   GitHub-Actions-workflow-file to maintain, unlimited free-tier
   bandwidth, no Netlify-style "one project over quota pauses the whole
   account" risk).
3. Decided to make the repo public anyway (no participant data in it --
   verified via `git log --all --full-history` before flipping
   visibility, see below; will need to be public for code-sharing at
   publication regardless). That removes the original GitHub Pages
   blocker, so **reverted to GitHub Pages** -- it satisfies the original
   preference (no new service/account) and this project's traffic is
   nowhere near where Cloudflare's bandwidth/reliability edge would
   actually matter.

**Pre-publicity data-safety check** (done before flipping repo
visibility): `task/pilot3/{numbers,colors}.txt` contain REAL participant
data (raw Prolific worker IDs, full response logs -- the actual pilot #3
incident data CLAUDE.md's history refers to) and were sitting untracked
on disk. Confirmed via `git log --all --full-history -- task/pilot3` and
a content-pattern search that this was NEVER committed to git history.
Added to `task/.gitignore` (alongside the two `compute_bonus*_tmp.py`
scripts that read directly from it) before making the repo public --
see that file's own comments.

**Build output**: combined into ONE deployment, not two -- `vite.config.js`
builds both `index-numbers.html` and `index-colors.html` into a single
`dist/` (plain `npm run build`, no `--mode`; Vite's own code-splitting
even automatically factors the shared `timeline-builder.js` module into
its own chunk between the two entries, a nice side effect of combining
rather than a design goal). `base` is conditionally set to
`/evidence_integration/` in production (GitHub Pages serves this repo as
a project page at that path) vs `/` in local dev -- confirmed both work
via `npm run build` producing correctly-prefixed asset URLs, checked
directly in the built HTML.

**Mechanics** (`.github/workflows/deploy-task-backend.yml`): triggers on
push to `main`, path-filtered to `task_backend/**` so unrelated commits
(`task/`, docs, analysis scripts) don't trigger a rebuild. Needs two
repo secrets (`VITE_SUPABASE_URL`, `VITE_SUPABASE_PUBLISHABLE_KEY` -- see
`.env` for the actual values) and **Settings -> Pages -> Source: "GitHub
Actions"** set once in the GitHub UI -- neither is scriptable from this
repo, both are one-time manual steps.

**Verified live** (this session): the deployed site returns `200` at both
`https://psipeter.github.io/evidence_integration/index-numbers.html` and
`.../index-colors.html`, including the JS bundle itself (not a 404) --
confirming the `/evidence_integration/` base path resolved correctly at
build time. A real browser test against the live (not local-dev) site
confirmed the backend calls succeed too (the two secrets baked in
correctly, not `undefined`) and that reload/resume works end-to-end on
the actual deployed URL, not just localhost.

One real point of confusion hit during that verification, worth
recording since it'll recur: a close-tab-then-reopen test using the
**bare** URL (no `?PROLIFIC_PID=`) looked like a resume failure (landed
back on welcome) but wasn't one -- the dev/local fallback
(`dev_${Date.now()}`) deliberately mints a NEW, different participant ID
every page load with no query param present, specifically so repeated
local test runs don't collide with each other's rows. `progress-check`
correctly reported "new" for what was, from the server's perspective,
genuinely a brand-new participant both times. Testing resume on any
deployment (live or local) requires reusing the SAME explicit
`?PROLIFIC_PID=` value across both loads.

---

## Status note for future sessions

As of this note, the Supabase backend and the full `task_backend` client
port are built, deployed, and verified end-to-end against the real
database (not just unit-level) for: welcome/consent checkpointing,
tutorial resume, idempotent upsert, real trial-boundary resume (mid-
session, across a reload), the timeout-retry/`attempt`/terminate path,
and all three session-ending screens (finish/terminate/returning-
participant) showing the correct visible completion code. `CLAUDE.md`'s
"Own-backend decision (Supabase)" section already points here; no
further doc pointer needed.

Two things remain genuinely open, neither blocking the other:
1. ~~A persistent, re-runnable test suite.~~ DONE -- `task_backend/tests/`
   (`@playwright/test`), 5 spec files covering everything verified
   throughout this doc (basic flow, trial-boundary resume, timeout-retry/
   `attempt`, all three completion screens, a colors-task smoke check).
   Runs against the real deployed backend, not mocked. Self-cleaning via
   a local-only `.env.test` secret key (`SUPABASE_SECRET_KEY`) --
   surfaced one more instance of the same GRANT gap from the DDL section
   above (`DELETE` was never granted to `service_role` either, since no
   Edge Function had ever needed it until the suite's own cleanup step
   did); fixed the same way. All 7 tests pass.
2. ~~The weekly backup process~~ -- DECIDED AGAINST (see "Next steps" item
   5 above): manual, on-demand backup once real data exists, not a
   scheduled job. Nothing left to build here unless that decision changes.

Hosting (`Hosting` section above) is now fully done: repo public, GitHub
Pages live, verified against the real deployed URL (not just localhost).

**Everything from CLAUDE.md's original "Pilot #3" incident investigation
through this build-out is now done.** The only remaining open item in
the entire plan is the deferred manual-backup guidance above, which is
intentionally not something to build now -- it's something to ask for
when it's actually needed.

---

## Post-buildout review: deployment readiness (brainstorm session)

A step-back review turned up two real gaps neither of which had been
noticed before, plus a decision on a few more:

1. **Supabase free-tier 7-day auto-pause -- ACCEPTED, not fixed.** A
   project with zero DB activity for 7 days pauses until manually
   resumed; a real participant hitting a paused project during a gap
   would see total silence, arguably worse than the JATOS failure modes
   this backend was built to fix. Decided NOT to build a keep-alive ping
   for this: real data collection is expected to take only a few days
   once the study launches, well under the 7-day window. Revisit if
   collection ever stretches longer or happens in separated waves.
2. **No path from `events` back into the analysis pipeline (`fitting/`,
   `models/`) -- DEFERRED ON PURPOSE.** Confirmed via grep: zero existing
   script has any Supabase awareness. Decided to wait until the database
   is actually populated (even with fake/test data) before building this,
   and to do participant-data anonymization as PART OF that export step
   (distinct from the earlier repo-publicity anonymization, which was
   about `task/pilot3/`'s files, not the live database).
3. ~~**Bonus-payment CSV**~~ DONE -- `task_backend/compute_bonus.py`.
   Matches the original `task/compute_bonus_tmp.py`'s exact convention
   (only `phase='trial'` rows count toward bonus, mirroring the old
   schema's `screen=='observation'` exclusion of tutorial rows; reward in
   cents; $5.00 clip per participant). Verified against real data in the
   live table (paginated fetch, 1443 rows across 8 participants, correct
   per-participant sums and clipping).

   **Extended** to solve the Submission-ID-vs-Participant-ID mismatch
   found while building it: Prolific's dashboard bulk-bonus box wants
   `<submission_id>,<amount>` lines, not `prolific_pid`-keyed ones.
   `--prolific-export <path>` now takes Prolific's own demographic/
   submissions export (Submission ID + Participant ID side by side) and
   joins on `prolific_pid` automatically, printing ready-to-paste lines
   directly (the bulk-bonus box is copy-paste text, not a file upload --
   confirmed by the person testing it against the real UI). Nonzero-
   bonus participants NOT found in the export are excluded from the
   output and explicitly flagged for manual handling, rather than
   silently dropped or included with a blank ID that would fail the
   whole paste.

   Column matching reuses the exact fuzzy-substring-match-with-override
   pattern already proven in `task/reconcile_prolific_jatos.py`
   (`find_col`/`--prolific-id-col`) -- found this existing tool via a
   commit-message search *after* independently arriving at the same
   design, which is reassuring convergence rather than a wasted
   rediscovery. Added the same `--submission-id-col`/`--participant-id-col`
   override flags for consistency and resilience if Prolific's export
   wording ever shifts again (their own docs note it already has, more
   than once).

   Tested against a REAL Prolific demographic export (not synthetic) --
   confirmed correct parsing of the real header row
   (`Submission id,Participant id,Status,...`), correct join behavior in
   both directions (our test participants correctly reported "not found"
   in the real export; the real export's participants correctly reported
   "not found" against our test-only Supabase data), and the nonzero-
   bonus warning path firing correctly for real mismatched data.

4. **URGENT SAFETY GAP FOUND AND FIXED**: the real Prolific export used
   to test the above (containing real Participant IDs, real Submission
   IDs, and -- confirmed via a real `Completion code: C12FEFJU` value in
   one row -- tied to the actual live `colors` study) was placed directly
   in `task_backend/` with **zero gitignore protection**, in a repo
   that's now public. A `git add .` at that moment would have committed
   real participant PII. Fixed immediately: `task_backend/.gitignore`
   now excludes `prolific_*export*.csv` and, preemptively, `compute_bonus.py`'s
   own default output (`bonus_numbers.csv`/`bonus_colors.csv`/`bonus_*.csv`),
   which is exactly as sensitive (real IDs paired with real payment
   amounts) and would hit the identical gap the moment it's ever run for
   real. Confirmed via `git log --all --full-history` that nothing
   matching this was ever actually committed -- caught before any harm,
   not after. **Lesson for future sessions: any new file dropped into
   task_backend/ containing real participant identifiers needs an
   explicit gitignore entry checked immediately, not eventually** -- this
   is the second time in this project a real-data file has needed this
   exact reactive fix (the first was `task/pilot3/`); a proactive habit
   of checking before doing anything else with a newly-added file would
   be better than reacting each time.
5. **Desktop/laptop-only enrollment**: handled entirely on Prolific's
   side (their own device filters screen this out) -- no code needed.
6. **Researcher-facing monitoring and a kill-switch/maintenance-mode**
   -- discussed, deliberately NOT building either. Given the short
   (few-day) collection window: monitoring reduces to "glance at a couple
   of SQL queries during the first hour after launch," and a kill-switch
   is judged not worth building when the existing levers (pause the
   Prolific study to stop new starts; resume already means nobody loses
   progress if a fix requires a pause) are adequate at this scale.

---

## Small-sequence test variants (generate_sequences.py --name)

Added after a real design flaw was caught in an earlier attempt at a
full-session test: seeding most of a session directly via the API and
only driving a few observations for real ends up testing "does my seeded
data model of the app match the app" more than "does the app actually
work" -- exactly the kind of thing a real bug in the trial loop could
slip through undetected. The fix: generate a genuinely small (2-trial)
sequence file and drive the ENTIRE session through it for real, with
zero seeding anywhere. Same idea directly enables fast manual testing
too (locally or through a real Prolific preview), not just automated
tests.

**`generate_sequences.py --name <suffix>`**: builds a small variant
(`TEST_NUMBERS_N_PREFIX=2 x TEST_NUMBERS_N_REPEATS=1` = 2 trials for
numbers, `TEST_COLORS_N_TRIALS=2` for colors) using its OWN separate
constants, never the production `NUMBERS_*`/`COLORS_*` ones -- gated
behind the explicit `--name` flag specifically so there's no way to
accidentally produce a real-shaped production file with a tiny trial
count, or vice versa. Output: `sequences_<task>_<name>.json` -- omitting
`--name` leaves production behavior (`sequences_<task>.json`) completely
unaffected. Verified against real generated output: correct trial count,
correct distinct qids, correct schema.

**Client side** (`src/numbers|colors/config.js`): `VITE_SEQUENCES_VARIANT`
env var selects which file to load, via `import.meta.glob` (auto-
discovers ANY matching `sequences_<task>_*.json` file -- a new variant
never requires editing these files again, directly solving the "awkward
rerunning/copying/renaming" pain point from the old JATOS-era workflow).
Verified live in a real browser: with the variant set, the trial-
transition screen correctly showed "Trial 1 / 2", confirming the small
pool is genuinely loaded at runtime, not just that the build succeeded.

**Real tradeoff, deliberately accepted**: `import.meta.glob(..., {eager:
true})` bundles EVERY matching file at build time regardless of which
one gets selected at runtime -- confirmed directly (bundle size grew
~240KB just from the variant file existing on disk during a build).
Given this, and that the GitHub Actions deploy workflow does a fresh
checkout + build on every push (no "forgot to delete it" protection
possible in CI if a variant file were ever committed), **test-variant
files are gitignored** (`sequences_numbers_*.json` / `sequences_colors_
*.json` in `.gitignore`) -- generated locally/on-demand, never enter git
at all. `playwright.config.mjs` generates the `test2trial` variant
automatically if missing, so the test suite is self-sufficient on a
fresh checkout with no manual setup step.

**`tests/full-session-bonus.spec.mjs` rewritten** using this -- two
participants, each driven through a COMPLETE real session (welcome,
consent, tutorial, both real trials, the genuine `end` screen) with zero
seeding anywhere, using the `test2trial` variant. Captures the ACTUAL
`reward` the app computes per real observation (via the real outgoing
`progress-append` request bodies) rather than trying to hand-predict it
-- the slider has `step="1"` (confirmed directly), so any fractional
target response gets silently rounded by the browser itself, making
exact-value prediction fragile for no real benefit; capturing what the
app actually computed and verifying it survives intact through to
`compute_bonus.py`'s output is the property that actually matters.
Verified: all 10 tests in the full suite pass together (4.7 minutes
total), including this one and every pre-existing test now running
against the small variant instead of the 32-trial production pool.

**One coverage gap accepted, not fixed**: with only 2 trials (max 60
cents raw per participant), `compute_bonus.py`'s $5.00 clip can never
trigger in this test -- that branch was already verified manually
earlier this session against real over-$5 smoke-test data (see the
"Post-buildout review" section's bonus-CSV item above). This test's job
is full-pipeline integrity at a genuinely real, fast scale, not
exhaustive coverage of every reward magnitude.


---

## Tutorial redesign: "Correct answer" panel replaces the old KDE/urn
## figure (numbers + colors), pilot #3's comprehension problem

Motivated by a pilot #3 finding, brainstormed at length before any code:
a substantial fraction of participants appeared to just repeat the most
recently shown stimulus as their response, rather than integrating across
the sequence. Real-task-side interventions (mid-task nudges, always-on
scaffolding during actual trials) were explicitly rejected as too
intrusive / too likely to contaminate the measured behavior itself.
Decision: strengthen the TUTORIAL only, where added clarity can't
contaminate real trial data, and separately rely on post-hoc filtering
(an anchoring-score exclusion criterion) for whatever comprehension gap
remains -- not attempted yet, noted as a follow-up, not a blocker.

**A more fundamental problem found before touching the visuals**: the old
KDE curve (numbers) / urn bar (colors) visually taught the FIXED
population parameter (`true_mean`/`true_p`) as "the answer" -- but
`config-base.js`'s `ERROR_MODE` (`'running_mean'` for numbers, `'running_p'`
for colors) scores against the RUNNING statistic of what's been observed
so far, a materially different, evolving quantity, especially early in a
trial. The tutorial was pointing at the wrong target. Confirmed this
wasn't a stale/leftover config value before proceeding -- current,
intended scoring behavior for both tasks.

### The new "Correct answer" panel

`correct-answer-numbers.js` (numbers) / `correct-answer-colors.js`
(colors) -- both new files, replacing `distribution-numbers.js` +
`numbers-draw-animation.js` (numbers) and `colors-draw-animation.js`
(colors, `urn-colors.js` itself KEPT for its color constants, just no
longer used for the tutorial's own SVG bar) -- **all three deleted
entirely this session**; still present under `task/` if this ever needs
reverting.

- **Numbers**: a thin 0-100 slider-style track. A small BLUE CIRCLE thumb
  (not a bar -- deliberately small enough that the taller red ticks
  protrude visibly above/below it) SLIDES to the RUNNING MEAN's position
  every time a new observation arrives -- directly visualizing "this is
  what should move, and by how much." Red ticks mark every observation
  (faded for history, bold/taller for the current one).
- **Colors**: a thin blue/red bar, split at the RUNNING PROPORTION of
  blue draws so far (blue-left/red-right, SLIDES its split point the same
  way the numbers thumb slides). Small dots accumulate ABOVE the bar --
  one per draw, blue dots packed in from the LEFT edge, red dots packed
  in from the RIGHT edge, so their COUNTS (not a continuous position,
  unlike numbers -- colors' draws have no natural axis position) are
  what's shown. Dot spacing (`100/(N_OBS+1)`) guarantees the two
  sequences can never collide even in the all-one-color extreme.
- Both: NO bubbling animation, NO artificial delay of any kind (the old
  design's ~1s "wait for the bubbles before anything appears" -- found
  and removed for every phase, not just the ones later hidden behind an
  overlay). `SHOW_EXACT_VALUE` (both modules): whether the running
  statistic's exact number is shown as text next to the indicator, or
  left purely visual -- a flag, not a final decision; currently `false`.
  `renderCorrectAnswer(Colors)` reveals its own container and is fully
  self-sufficient for the observation plugin's every-call-already-visible
  usage; the intro plugin's one-time reveal passes `fadeIn:true`.

### Intro plugin (obs 1): three-click progression, redesigned from an
### earlier four-click version

Click 1 (left box 1) -> reveals box 1's own text AND the top-right box
(box 0b) together, plus the centre example number/circle fading in.
Click 2 (left box 2, the goal text) -> reveals box 2's own text AND the
whole correct-answer panel (no more separate click just for the panel).
Click 3 (left box 3, the slider instructions) -> reveals box 3's own text
AND activates the real response slider. The "Sequence history" tracker is
DELIBERATELY never revealed anywhere in either intro plugin file -- it
first becomes visible starting at observation 2 (the observation plugin,
which shows it unconditionally on every call), since sequence history
doesn't meaningfully apply yet with only one value.

Both right-column boxes show a "..." placeholder (matching the LEFT
column's own locked-box convention) until their real content is revealed.

### Box captions + sizing (numbers AND colors)

"Correct answer" / "Sequence history" captions: INSIDE each box,
top-center, absolutely positioned -- iterated through THREE placements
this session (above-as-flow-element, then an in-box overlay, back to
in-box) before landing here; kept for the record since the reasoning
mattered each time (see below). Only render when the box's OWN real
content is genuinely visible (`showCaptions = !hideGraphics && !showClock`
in both observation plugins) -- NOT during phase D (hidden behind an
overlay) or phase E (correct-answer replaced by the clock demo, tracker
hidden behind its own overlay) -- a caption reading "Correct answer" over
an opaque cover or a clock demo doesn't make sense, so the caller skips
rendering the caption's markup entirely rather than fighting stacking
order to hide something still in the DOM.

Six boxes total (3 left text boxes + 3 right boxes: top-box/correct-
answer/tracker), each `flex: 0 0 18vh` -- deliberately EQUAL on both
sides, not independently tuned/guessed per box (an explicit instruction,
after an earlier per-box-clamp() approach kept guessing wrong). Numbers
and colors use SEPARATE classes (`.numbers-tutorial-box` /
`.colors-tutorial-box`, identical properties) rather than one shared
class -- `.tutorial-info-block`/`.tutorial-right-top-box`/-image-box/
-tracker-box are ALL also used by the OTHER task's tutorial, and
coupling their box heights through one class would be a new exception to
this codebase's existing "independently designed tutorials" pattern for
no real benefit.

### Real bugs found and fixed while building this (all confirmed via
### direct DOM/geometry checks in a real browser, not visual guessing)

1. **Observation plugin never revealed the panel's own container.**
   `renderCorrectAnswer(Colors)`'s wrapper starts at CSS `opacity:0`
   (meant to be revealed by a staged-reveal helper); the intro plugin
   called that helper, the OBSERVATION plugin never did, since it has no
   staged reveal at all. Result: ticks/thumb were being positioned
   correctly the whole time (confirmed via computed style) but stayed
   invisible because the PARENT's opacity zeroed out everything inside it
   regardless -- a bug an earlier round of "check the child's own
   opacity" testing completely missed. Fixed by having the render
   function reveal its own container unconditionally.
2. **The box's own white background/border was placed INSIDE the
   hidden-until-revealed content wrapper.** Before reveal, the entire box
   frame vanished along with the content, leaving the bare page
   background showing through (reported directly as "an empty gray
   area"). Fixed by moving `.dist-canvas` (the class providing that
   background) onto the OUTER, always-visible box element, matching the
   left column's own `makeBox` convention (box chrome always visible;
   only the specific real-content span toggles).
3. **The hidden content wrapper had no explicit height**, so
   `correct-answer-outer`'s `height:100%` had nothing real to resolve
   against, collapsing/misplacing the indicator (reported as "a shrunk
   and overlapping box"). Fixed with an explicit `height:100%` on that
   wrapper, verified after the fix via exact pixel geometry (content
   rect height matches the box minus padding; the track's own vertical
   center matches the box's vertical center exactly).
4. **An early height-overflow diagnosis was wrong.** A "boxes taller than
   an 800px window" symptom was initially (incorrectly) attributed to two
   new caption elements; measured scroll-height was BYTE-IDENTICAL before
   and after removing that theory's own fix, proving captions were never
   the cause. The real, much larger factor turned out to be the LEFT
   column's own text boxes, which never had a `vh`-based height budget at
   all (`flex:1`, pure content-driven height, ~20-25vh per box measured
   directly) -- unlike every box in the right column. Surfaced only once
   asked to reason in `vh` terms consistently, per this project's own
   established sizing convention, rather than chasing absolute pixel
   numbers against one arbitrarily-chosen window size.

### Verification discipline

Every claim above was confirmed via real headless-browser checks (exact
`getComputedStyle`/`getBoundingClientRect` values, not just "the code
looks right") -- this was itself a lesson relearned mid-session: an
early screenshot misread ("the centre panel looks blank") turned out
wrong once cross-checked against the DOM directly, so subsequent checks
leaned on precise element geometry/computed style as the primary
evidence, with screenshots as a secondary, lower-confidence sanity check
rather than the main source of truth.


---

## Codebase cleanup pass (post-tutorial-redesign)

After the "Correct answer" panel redesign (previous section) left a lot of
now-inapplicable comments/exports scattered around, asked for a thorough
cleanup: remove dead code and stale comments referencing methods no longer
in use, then consider further consolidation/renaming. Two-part effort,
tackled in that order.

### Part 1: dead code removal -- DONE

Verified via a full sweep of every exported symbol against the WHOLE
`src/` tree (not just `shared/` -- an early version of this check was
wrongly scoped and produced false positives for `pickTutorialExample`/
`buildConfig`, which are used in `numbers/config.js`/`colors/config.js`).

Removed:
- `urn-colors.js`: `buildUrnSVG`, `LAYOUT`, `DIM_BLUE`, `DIM_RED` -- all
  only used by the tutorial's old bar-drawing code (deleted the session
  before this one). File was reduced to 3 color constants, then later
  folded away entirely (see Part 2).
- `tutorial-text-colors.js` / `tutorial-text-numbers.js`: `URN_CAPTION` /
  `DIST_CAPTION` (both dead, both had admitted as much in their own
  "pending a later pass" comments -- that pass never happened) and
  `WARNING_YELLOW` (dead in both files).
- `draw-performance-numbers.js`: `coinGlyph` (never called anywhere, not
  even internally -- confirmed only its own declaration existed) and
  `COIN_STROKE` (only used inside the now-deleted `coinGlyph`).
- Fixed every docstring caught pointing at deleted files/functions
  (`buildHintHTML`, `distribution-numbers.js`, etc.) or describing CSS
  values that no longer apply.
- CSS: found (and confirmed EMPIRICALLY via computed-style checks in a
  real browser, not just reasoning about specificity) that
  `.tutorial-right-image-box`/`.tutorial-right-tracker-box`/
  `.tutorial-right-top-box`'s own `flex`/`overflow` rules are now ALWAYS
  overridden by the newer `.numbers-tutorial-box`/`.colors-tutorial-box`
  classes (2-class selectors beat 1-class ones, and every current usage
  pairs them) -- trimmed the dead lines and the comments explaining
  reasoning that no longer applies. Fixed two stale comment references to
  classes renamed away earlier in the redesign
  (`.correct-answer-box-short`, `.tutorial-right-bottom-box`).

### Part 2: consolidation -- DONE (one significant finding), IN PROGRESS (broader pass)

**Found and fixed**: the same 3 hex colors (`#2563eb` blue / `#ef4444`
red / `#16a34a` green) were independently redeclared as "canonical"
named constants in 5+ different files -- `tutorial-text-numbers.js`'s
GOAL_COLOR/SAMPLE_COLOR/DIST_COLOR, the old `urn-colors.js`'s
SAMPLE_BLUE/SAMPLE_RED/DIST_COLOR, `draw-performance-numbers.js`'s
MEAN_BLUE/SAMPLE_RED/ERROR_GREEN, `draw-performance-colors.js`'s own
local SAMPLE_BLUE/SAMPLE_RED/DIST_COLOR, plus `plugin-observation-
numbers.js` hardcoding the red inline and `plugin-observation-colors.js`
redeclaring blue/red locally instead of importing its own task's
existing canonical source.

New `palette.js` is now the ONE underlying source (`BLUE`/`RED`/`GREEN`).
Every one of the files above now imports from it, keeping its own
semantically-named local alias (e.g. numbers's GOAL_COLOR vs colors's
SAMPLE_BLUE for the identical hex value) -- this fixes there being one
underlying value per color, not that every file must call it the same
thing. Deliberately did NOT extend this to the more incidental one-off
uses of the same hex values (`plugin-iti-clock.js`'s clock rendering,
`create-terminate-session.js`'s "Too slow" message, `slider-colors.js`'s
ruler bands, `observation-timeout-clock.js`'s warning color,
`tutorial-tracker.js`'s default parameter) -- those are standalone UI-
styling choices that happen to reuse the same brand colors, not
canonical named constants multiple files were each separately trying to
own; touching all of those too would be a much bigger, lower-value
refactor. Revisit only if a real inconsistency (not just a style
preference) surfaces there too.

Went one step further: `urn-colors.js`, once stripped to just 3 re-
exported constants, was folded directly INTO `tutorial-text-colors.js`
and deleted entirely -- numbers never had an equivalent separate color
file (its colors are defined directly in `tutorial-text-numbers.js`), so
this gives both tasks the exact same file structure. Redirected all 4
remaining importers (`correct-answer-colors.js`, `plugin-tutorial-intro-
colors.js`, `plugin-tutorial-observation-colors.js`, `plugin-tutorial-
summary-colors.js`), merging duplicate import lines where a file was
already importing separately from both. Verified: zero dangling
imports, every touched file syntax-checks clean.

**Outstanding from this pass** (left exactly where the previous session
ended, for whoever picks this back up):
1. **Naming question, not yet resolved**: `plugin-inter-trial.js` (the
   "Trial X / 40, generating new sequence…" between-trials reset screen)
   vs `plugin-iti-clock.js` -- "inter-trial" vs "ITI" (the standard
   abbreviation for the identical term) used inconsistently across two
   filenames. Had just re-read `plugin-inter-trial.js` (confirmed: the
   reset screen) but had NOT yet read `plugin-iti-clock.js` to determine
   whether it's a genuinely separate concept (names fine as-is) or the
   same thing named two different ways (one should be renamed for
   consistency) before running out of room in that session.
2. **`phases.js`** -- not yet opened/reviewed at all this pass.
3. **Broader naming-consistency pass** not yet done: whether `build-*.js`
   vs `plugin-*.js` naming holds up consistently throughout, and whether
   any OTHER file has the same kind of duplicated-canonical-constant
   pattern the color consolidation just fixed (the color case was found
   by deliberately checking; nothing guarantees it's the only instance).
4. **Nothing has been tested since this whole cleanup/consolidation pass
   began** -- explicitly deferred; testing happens only when asked for
   directly, not proactively during this pass. This includes the color
   consolidation, which touched 9 files and deleted one -- real risk
   surface that hasn't been exercised in a browser at all yet, despite
   every individual file syntax-checking clean.

