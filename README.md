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
| numbers | live | Continuous inputs; 15 obs/trial; Normal(mean, std); 8x4=32 trials, per-participant pool of 200 | **Piloting** (task_backend) |
| colors | live | Binary inputs (blue/red); 15 obs/trial; Bernoulli(p); 32 trials/participant, per-participant pool of 200 | **Piloting** (task_backend) |

numbers and colors are designed to be completed within-subject (same
participants recruited via Prolific allowlist), unlocking cross-task
individual-differences analysis. See
**[`task_backend/CLAUDE.md`](task_backend/CLAUDE.md)** for the online
task's schema, deployment, and testing — Prolific cutover is done, real
pilot rounds have run.

---

## Repository structure

See `CLAUDE.md`'s "Repository structure" section for the full annotated
layout. In brief: `models/` (math + NEF), `fitting/` (Optuna RMSE/NLL
pipeline), `scripts/` (figures + analysis), `task_backend/` (online
experiment), `data/`, `docs/` (this project's living documentation),
`archive/` (retired code + frozen history).

## Legacy: task/ (retired)

The original JATOS/MindProbe-hosted online task. Superseded by
`task_backend/` above; remains on disk for historical reference. Full
design history: `archive/HISTORY_task_legacy.md`.
