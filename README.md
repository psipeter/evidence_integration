# Evidence Integration

This repository contains code for modeling and analyzing individual variability in evidence integration across cognitive tasks in humans. It accompanies a manuscript currently under revision.

---

## Research Overview

To navigate uncertain environments, the brain must continuously integrate new information while weighing recent observations against longer-horizon outcomes. This project investigates the mechanisms underlying individual variability in this process using:

- **Mathematical models** — a prediction-error update rule scaled by the number of observations, capable of capturing a spectrum of integration strategies (recency-biased to temporally discounted)
- **Behavioral analysis** — applied to three cognitive tasks, capturing trial-by-trial estimation variability, action switching under incongruent social information, and decay in update magnitude
- **Biophysical neural network models** — linking cognitive parameters (representational noise, synaptic weight distributions) to observable behavioral heterogeneity

---

## Repository Structure

```
evidence_integration/
├── data/                  # Task data — only carrabin.pkl, jiang.pkl, yoo.pkl are tracked
├── models/                # Mathematical and neural network model implementations
├── fitting/               # Model fitting and parameter estimation routines
├── analysis/              # Analysis scripts and notebooks
├── figures/               # Generated figures (not tracked by git)
├── experiments/           # New experiment definitions and simulations
├── utils/                 # Shared utility functions
├── jobs/                  # SLURM job scripts for cluster runs
└── notebooks/             # Exploratory notebooks
```

## Data Schema

All three task dataframes share a common column schema:

| Column | Description |
|---|---|
| `pid` | Participant ID |
| `trial` | Trial index |
| `observation` | Sequential observation index within trial (all tasks) |
| `stage` | Social round index, jiang only (retained for analysis) |
| `value` | Stimulus input seen by participant or model |
| `response` | Participant output |

Jiang-specific columns: `network`, `who`, `degree`, `rd`, `stage`
Carrabin-specific columns: `qid` (unique stimulus sequence identifier)

> **Note:** This repo is a clean refactor. The original codebase is preserved in a separate private repository and should not be modified. Files are ported here selectively and rewritten for clarity and reproducibility.

---

## Reproducibility

### Environment

This project uses two coordinated environments to ensure identical reproduction locally and on the compute cluster:

```bash
# Conda environment (Python 3.11)
conda activate PY311

# Python virtual environment
source evidence_integration/venv/bin/activate
```

Both environments are kept in sync. `environment.yml` and `requirements.txt` are the sources of truth — update them whenever dependencies change.

### Running the Analysis

Each analysis step is designed to be run independently. Entry points are documented at the top of each script or notebook. A full pipeline description will be added here as the refactor progresses.

### Experiment Tracking

- All model fitting runs are logged with parameters, random seeds, and output paths
- New experiments are documented in `experiments/` with a corresponding design note (see `experiments/TEMPLATE.md`)
- Figures are regenerated from scripts — no manual edits to output files

---

## Workflow

This codebase is being actively refactored. The development workflow is:

1. **High-level discussion** between researcher and Claude (claude.ai) to scope changes or new experiments
2. **Cursor prompts** drafted by Claude and executed by Cursor in the local environment
3. **Review and commit** by the researcher before any changes are pushed

See `.cursorrules` for Cursor's operating instructions.

---

## Status

- [ ] Port and refactor core model code
- [ ] Port and refactor fitting routines
- [ ] Validate refactored code reproduces original results
- [ ] Design and implement new experiments
- [ ] Final analysis and figure generation