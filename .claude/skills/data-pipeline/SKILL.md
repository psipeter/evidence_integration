# data-pipeline

Use this skill when pulling real participant data from Supabase into
this repo's analysis pipeline (`scripts/pull_soltani_data.py`), or when
deciding which participant-exclusion method to apply to a pull. Not
needed for `carrabin`/`yoo` (static, already-built pkls) — soltani only.

---

## `scripts/pull_soltani_data.py` flags

- **`--complete_pairs`** — derives the pid set with a `finished` row in
  BOTH tasks (numbers and colors) live from Supabase. This is the only
  reproducible way to re-select the same people; no cohort pid lists are
  recorded anywhere in this repo.
- **`--exclusion_method {contingency,performance,baseline,non_integrator}`**
  — selects the participant-exclusion criterion. `non_integrator` is the
  current default; see "Participant exclusion" below for what it means
  and why it's the default over the other three.
- **`--per_task_exclusion`** — opts OUT of the default subject-level
  policy (`require_both_tasks`), applying exclusion separately per task
  instead. Only use this deliberately — the default exists because
  per-task exclusion silently degrades within-subject cross-task panels.
- **`--no_filter`** — skips `utils/participant_filters` entirely. For
  diagnosing how much the exclusion criteria change a result. NOT for
  published output; the script prints a warning when this is set.

## Pid registry

Integer pids come from the persistent registry
(`utils/pid_registry.py`), keyed on `prolific_pid` identity alone — NOT
computed fresh per call. Consequence: a filtered build and an unfiltered
build of the same data assign the SAME integer pid to the same real
person, and growing the pool with new participants never reassigns an
existing pid. The registry file (`data/pid_registry.json`) contains real
`prolific_pid`s and must NEVER be committed or pushed — always gitignored,
moved between machines by hand (scp/rsync), never through GitHub.

## Participant exclusion (`utils/participant_filters.py`)

Applied to task_backend's pulled data before any model fitting.

**`non_integrator` is the default criterion**: a participant for whom
observations before the most recent one make no reliable contribution to
predicting their response. Property of information, not accuracy or
weighting — retains inaccurate-but-genuine integrators, catches both
literal copiers and random/drifting responders with one test.
Operationalized as `response_t ~ 1 + value_t + mean(value_0..value_{t-1})`
per participant, with a trial-level cluster bootstrap (`n_boot=20000`,
`ci=95` defaults — verified stable across bootstrap seeds at this
n_boot). Retained if the prior-mean coefficient's CI excludes zero.

**`require_both_tasks` is the default** (opt out with
`--per_task_exclusion`): a participant failing in either task is dropped
from both — per-task exclusion silently degrades within-subject
cross-task panels by changing which people end up in the intersection,
not through power or reliability loss.

**Known gaps, deliberately not engineered around**: does not catch
integrating the wrong statistic, scale compression, or
anchored-with-a-nudge responding. The last is a real miss.

`performance`/`integration`/`contingency` remain as computed diagnostics
(not decisive) or are archived
(`archive/utils/archive_exclusion_criteria.py`). Full investigation and
why the alternatives lose: `docs/DECISIONS.md`,
`archive/HISTORY_modeling_2026.md`.

## Typical usage

```bash
# See who's currently eligible, without building anything
venv/bin/python scripts/pull_soltani_data.py --list_candidates numbers

# Pull the canonical, contamination-free dataset (default exclusion)
venv/bin/python scripts/pull_soltani_data.py --complete_pairs

# Diagnose exclusion impact (not for publication)
venv/bin/python scripts/pull_soltani_data.py --complete_pairs --no_filter
```

Output: `data/soltani_{numbers,colors}[_<datafile>].pkl`, via the same
shared filter/rescale/anonymize/save pipeline
(`build_model_inputs.py`'s `build_from_df()`) carrabin/yoo already use.
