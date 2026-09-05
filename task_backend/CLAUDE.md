# task_backend/CLAUDE.md — online task (numbers/colors)

Nested, directory-scoped conventions for the `task_backend/` online
experiment. Loads only when working in this directory. For why this
backend exists (replacing the retired JATOS pipeline) see
`docs/DECISIONS.md`; for the full build-out history see
`archive/HISTORY_task_backend_buildout.md`. This file covers current
state only — settled, not narrative.

**Status: settled infrastructure.** Prolific cutover is done, pilots have
run. Most sessions touching this project are modeling/figures on data
this backend produces, not edits to the backend itself.

---

## Terminology

**numbers**/**colors** throughout (file names, directory names, the
`task` column/parameter, class names, CSS classes) — NOT continuous/
binary, which is the retired `task/` pipeline's naming. The two codebases
don't share terminology. Exception: the shared scoring module is
`scoring.js` (task-neutral), not `bonus-numbers.js`, since both tasks use it.

---

## Backend schema, Edge Functions, hosting/deployment

See `.claude/skills/task-backend-schema/SKILL.md` for the `events` table
columns, idempotency/access model, the three Edge Functions, and
hosting/deployment status. Rarely-needed reference — not required for
routine app work.

## Sequences

See `.claude/skills/task-backend-sequences/SKILL.md` for generation
mechanics (`generate_sequences.py`) and downstream diagnostics
(`scripts/plot_sequences.py`).

## Scoring

`scoring.js` (shared by both tasks): `normError = rawError /
MAX_POSSIBLE_ERROR; reward = max(0, MAX_REWARD * (1 - bonusDecay *
normError))`, per observation, summed for trial/tutorial total.

- `MAX_REWARD = 2` cents, `MAX_POSSIBLE_ERROR = 100`.
- `bonusDecay` split per task (`NUMBERS_BONUS_DECAY = 25`,
  `COLORS_BONUS_DECAY = 15`) since numbers' std_fixed change made one
  shared value inflate rewards on numbers relative to colors. It's a
  REQUIRED argument to `computeResponseReward`/`computeTrialReward` (no
  default) so a call site can never silently use the wrong task's value.
- `ERROR_MODE` (`config-base.js`) is `'running_mean'` (numbers) /
  `'running_p'` (colors) — scores against the running statistic of
  observed values, not the fixed generative parameter. Deliberate
  methodological choice, not a placeholder.

## Tutorial

A "Correct answer" panel (`correct-answer-numbers.js`/`correct-answer-
colors.js`) replaces an earlier KDE-curve/urn-bar design, motivated by a
real pilot comprehension finding. Numbers: a thumb sliding to the running
mean's position on a 0-100 track, tick per observation. Colors: a
blue/red bar split at the running blue proportion, dots accumulating
above it. Plain HTML/CSS, no animation delay. Intro plugin: 3-click
progressive reveal. Tutorial observations: 5-phase (A-E) top-right hint
system keyed on observation number.

## Client code conventions

- **`build-*.js`** — plain builder functions, no jsPsych-plugin shape.
- **`plugin-*.js`** — real jsPsych plugins (`info` + `trial()`).
- **`create-*.js`** — hand-rolled, non-jsPsych-trial DOM/orchestration
  code.
- **jsPsych 8 plugins must never be `async`** — jsPsych 8.2.3 advances
  the timeline on Promise resolution, not `finishTrial()`; declaring
  `async` causes overlapping trial instances (a real, previously-shipped
  bug). Pattern A (no timeout clock: consent/tutorial/summary) vs
  Pattern B (has a timeout clock: real observation plugins) are the only
  two shapes.
- Parameters the app always supplies explicitly (`true_mean`, `true_std`,
  `true_p`) must have NO `default` key in `info.parameters` — this makes
  jsPsych fail loudly if ever missing, rather than silently rendering
  wrong data.

## Testing

4 Playwright spec files (`task_backend/tests/`), run against the real
deployed Supabase backend, not mocked:
- `happy-path.spec.mjs` — one full-session traversal per task; two-phase
  (UI-level, then a DB-only check gated on the first passing).
- `resume.spec.mjs` — reload mid-session resumes at the correct trial
  boundary.
- `timeout-retry.spec.mjs` — three real observation timeouts in a row:
  `attempt` increments, session terminates on the third.
- `completion-screens.spec.mjs` — all three session-ending paths show
  the visible completion code.

`playwright.config.mjs` uses dedicated ports (5183/5184) and a small
`test2trial` sequence variant so a full session completes in seconds.

```bash
cd task_backend
lsof -ti:5183 -ti:5184 | xargs -r kill -9   # clear stale test servers first
npx playwright test
```

**The person running this project runs tests themselves** — give exact
commands rather than running directly. If ever run programmatically, use
Bash's `run_in_background` rather than a blocking foreground call — a
full suite runs long enough that it needs the background form.

## Data pipeline out (Supabase → analysis)

`scripts/pull_soltani_data.py` (repo root, not in this directory) pulls
real, finished participant data from Supabase for an EXPLICIT list of
`prolific_pid`s per pilot round — different rounds are different people
with different generative parameters (e.g. numbers' `std_fixed` changed
between pilots), so merging them silently would break cross-pilot
comparison. `--list_candidates <task>` probes current status directly
from Supabase without building anything.

```bash
cd task_backend
npm install                    # first time only
npm run dev:numbers            # local dev server
npm run dev:colors
npm run build                  # production build -> dist/ (both tasks)
npm test                       # Playwright suite, see Testing above
```

## Participant exclusion and sequence diagnostics

Both moved to skills, since they're methodology/diagnostic reference
rather than routine backend work:
- Participant exclusion (`utils/participant_filters.py`, applied to this
  backend's pulled data) — `.claude/skills/data-pipeline/SKILL.md`.
- Sequence-generation diagnostics (`scripts/plot_sequences.py`) —
  `.claude/skills/task-backend-sequences/SKILL.md`.


