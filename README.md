# Evidence Integration

This project studies **how people integrate sequential noisy evidence**,
using cognitive models and a biophysical spiking neural network (NEF) to
identify the computational and neural mechanisms underlying that process.

Central model: **α(t) = α₀ / t^λ** (power-law decaying learning rate). In
the NEF this emerges from spiking dynamics rather than being hardcoded.

The full scientific goals, current active thread, and figure-by-figure
results are tracked in **[`docs/SCIENCE.md`](docs/SCIENCE.md)**. Code
conventions, active models/datasets, and workflow rules are in
**[`CLAUDE.md`](CLAUDE.md)**. The reasoning behind past methodology and
platform decisions is in **[`docs/DECISIONS.md`](docs/DECISIONS.md)**.

---

## Tasks

| Name | N | Key features | Status |
|------|---|-------------|--------|
| carrabin | 21 | Binary inputs; 5 obs/trial; sequences repeat (qid); true_p known | Active |
| yoo | 38 | Continuous inputs; 30 obs/trial; no sequence repetition | Active |
| numbers | 46 | Continuous inputs; 15 obs/trial; Normal(mean, std); 8x4=32 trials, per-participant pool of 200 | Active |
| colors | 46 | Binary inputs (blue/red); 15 obs/trial; Bernoulli(p); 32 trials/participant, per-participant pool of 200 | Active |

numbers and colors were completed within-subject (same 46 participants
recruited via Prolific allowlist, after exclusion), unlocking cross-task
individual-differences analysis. See
**[`task_backend/CLAUDE.md`](task_backend/CLAUDE.md)** for the online
task's schema, deployment, and testing — data collection is complete and
these two datasets are now used throughout `paper/main.tex`.

---

## Repository structure

See `CLAUDE.md`'s "Repository structure" section for the full annotated
layout. In brief: `models/` (math + NEF), `fitting/` (Optuna RMSE/NLL
pipeline), `scripts/` (figures + analysis), `task_backend/` (online
experiment), `data/`, `docs/` (this project's living documentation),
`archive/` (retired code + frozen history).

## Legacy: task/ (retired)

The original JATOS/MindProbe-hosted online task. Superseded by
`task_backend/` above; fully retired and archived under
`archive/task/` (build artifacts and raw participant data deleted as
reproducible/recoverable, not preserved). Full design history:
`archive/HISTORY_task_legacy.md`.
