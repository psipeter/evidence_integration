# task-backend-schema

Use this skill when modifying the task_backend infrastructure itself —
the Supabase schema, Edge Functions, or deployment — as opposed to
routine UI/content work (client code conventions, scoring, tutorial
content live directly in `task_backend/CLAUDE.md` since those come up
far more often). This is settled, low-frequency-need reference.

---

## Backend schema (`events` table, Supabase)

One append-only table. Every row is either a checkpoint
(tutorial/trial observation) or a bookkeeping/terminal marker
(welcome/consent/finished/terminated).

| Column | Notes |
|---|---|
| `id` | `bigserial primary key` — `ORDER BY id DESC LIMIT 1` finds "latest state," never `created_at` (unreliable under retries) |
| `prolific_pid` | real Prolific ID, or `dev_${Date.now()}` fallback locally |
| `task` | `'numbers'` or `'colors'` |
| `pool_index` | deterministic hash of `prolific_pid` (see the task-backend-sequences skill) |
| `phase` | `'welcome'\|'consent'\|'tutorial'\|'trial'\|'finished'\|'terminated'` |
| `trial_index` | 0-31 for `phase='trial'`; `-1` sentinel otherwise (never `null` — two `NULL`s are distinct in Postgres, which would silently break idempotency) |
| `observation_index` | 0-14 for `tutorial`/`trial`; `-1` sentinel otherwise |
| `attempt` | increments only on a timeout-triggered replay of the same `(trial_index, observation_index)` |
| `response`, `timed_out`, `rt`, `value`, `true_mean`, `true_std`, `true_p`, `qid`, `error`, `reward` | nullable |

**Idempotency:** `unique (prolific_pid, task, phase, trial_index, observation_index, attempt)`.
**Access model:** RLS enabled, zero policies for `anon`/`authenticated`
(deny-all, including read) — the three Edge Functions below (service-role
key, server-side) are the only way in or out. The browser never talks to
the database directly.

## Edge Functions

- **`progress-check`** — called before building the jsPsych timeline.
  Queries the latest row for `(prolific_pid, task)`: `finished`/
  `terminated` → returns status + Prolific code, client skips the
  timeline; `trial` → resume at start of current/next trial; `tutorial`/
  `consent` → resume at tutorial start; `welcome`-only or no rows → full
  run from welcome.
- **`progress-append`** — replaces JATOS's `appendResultData`.
  Fire-and-forget is deliberately NOT used: the client tracks consecutive
  failures and surfaces a visible (non-blocking) warning banner after 2
  in a row.
- **`progress-finish`** — sanity-checks expected trial-row count before
  accepting a "finished" claim, then hands back the completion code as
  visible text (never only embedded in a redirect URL).

Resume granularity is **trial-boundary, not exact-observation** —
reloading mid-trial restarts that trial (cheap: 15 observations); every
observation is still logged individually for analysis.

## Hosting / deployment

Single combined `dist/` (both `index-numbers.html`/`index-colors.html`
built together via `npm run build`), deployed to GitHub Pages via
`.github/workflows/deploy-task-backend.yml` (path-filtered to
`task_backend/**`). Live at
`https://psipeter.github.io/evidence_integration/index-{numbers,colors}.html`.

**Prolific cutover: done.** Real pilot rounds have run directly against
task_backend (not JATOS), using Prolific's Study URL field pointed at the
URL above with `?PROLIFIC_PID={{%PROLIFIC_PID%}}`.
`supabase/functions/_shared/prolific-codes.ts` mirrors the old JATOS
completion/early-exit codes and is confirmed working end-to-end against
real Prolific submissions.
