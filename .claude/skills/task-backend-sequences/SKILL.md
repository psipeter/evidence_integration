# task-backend-sequences

Use this skill when generating the task_backend sequence pool
(`generate_sequences.py`) or diagnosing it against candidate models
(`plot_sequences.py`) — as opposed to routine app work.

---

## Generation (`task_backend/generate_sequences.py`)

**One consolidated script** — a single method per task (see
`docs/DECISIONS.md` for why not pure i.i.d. or moment-matching alone),
with its own `verify_numbers_trials`/`verify_colors_trials` asserts run
at generation time. Downstream tools (`scripts/plot_sequences.py`, below)
trust these rather than re-auditing.

- **`NUMBERS_STD_FIXED = 10`** (current value; reverted from 15 after a
  real pilot at std=15 showed a weak |Δresponse| decay signal). If a
  trial's achieved std misses tolerance even after retries, the whole
  prefix regenerates from scratch (up to 30 attempts) — a simpler
  single-mechanism replacement for an earlier two-mechanism repair,
  confirmed empirically redundant (6/6400 vs 5/6400 outliers).
- **Fixed tutorial sequences:** `choose_tutorial_sequences` (`--tutorial`
  flag) selects ONE trial per task from the real production pool as
  every participant's tutorial example — same trial for everyone.
  Written to `tutorial_sequence_{numbers,colors}.json` at repo root,
  imported directly by each task's `config.js`.
- **Files:** `sequences_numbers.json`/`sequences_colors.json`, each a
  plain JSON array of 200 independent pool members (no `.pkl`). Each
  member: 32 trial dicts (`qid, true_mean, true_std, true_p, values,
  prefix_length, iti_ms, iti_condition, trial`). Each participant gets
  ONE member via `poolIndexForParticipant` — a deterministic DJB2-style
  hash of `prolific_pid` — so there's no separate "production" file
  distinct from the pool. Same hash for both tasks, so one participant
  gets the same pool index in numbers and colors.
- **Test variants:** `generate_sequences.py --name <suffix>` builds a
  small variant (e.g. `test2trial`) using separate constants — gated
  behind `--name` so it can never produce a real-shaped production file
  by accident. `VITE_SEQUENCES_VARIANT` (client env var) selects which
  file to load; unset in every real build (production files are
  git-tracked, `_*`-suffixed variants are gitignored).
- **`iti_condition`/distractor system:** REMOVED — no distractor
  manipulation in this study. `iti_condition` is still generated (data
  layer unchanged) but consumed nowhere on the client — inert data, not
  inert code.

## Diagnostics (`scripts/plot_sequences.py`, repo root)

Plots candidate math/NEF models against this backend's real, deployed
sequence pool (`sequences_numbers.json`/`sequences_colors.json`) — NOT
`task/`'s old sequence files.

```bash
# across_models: fix sequences, vary the model
venv/bin/python scripts/plot_sequences.py across_models --alpha_0 1.0 --rl_lambda 0.5
venv/bin/python scripts/plot_sequences.py across_models --skip_nef   # math models only, fast

# across_pids: fix the model, vary the pid (per-pid spread)
venv/bin/python scripts/plot_sequences.py across_pids
```

Both default to `--pool_root task_backend`; both take `--n_pool N` to
cap the pool for a fast smoke test instead of the full 200 members.
Trusts `generate_sequences.py`'s own verification asserts (above) rather
than re-auditing the pool.
