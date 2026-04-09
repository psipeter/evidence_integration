# Evidence Integration

This repository contains code for modeling and analyzing individual variability in evidence integration across cognitive tasks in humans. It accompanies a manuscript currently under revision.

---

## Research Overview

To navigate uncertain environments, the brain must continuously integrate new information while weighing recent observations against longer-horizon outcomes. This project investigates the mechanisms underlying individual variability in this process using:

- **Mathematical models** — a prediction-error update rule scaled by the number of observations, capturing a spectrum of integration strategies (recency-biased to temporally discounted)
- **Behavioral analysis** — applied to three cognitive tasks, capturing trial-by-trial estimation variability, action switching under incongruent social information, and decay in update magnitude
- **Biophysical neural network models** — linking cognitive parameters (representational noise, synaptic weight distributions) to observable behavioral heterogeneity, implemented in Nengo (Neural Engineering Framework)

---

## Tasks

| Name | Reference | n | Response type | Key measure |
|---|---|---|---|---|
| `carrabin` | Prat-Carrabin & Woodford (2024) | 21 | Continuous slider | Excursion (response variability per sequence) |
| `jiang` | Jiang et al. (2023) | 224 | Binary choice | Switch probability as function of social conflict |
| `yoo` | Yoo et al. (2025) | 38 | Continuous slider | Power-law decay of update magnitude |

---

## Repository Structure

```
evidence_integration/
├── data/                  # Task data — only carrabin.pkl, jiang.pkl, yoo.pkl are tracked
├── models/                # Mathematical and neural network model implementations
│   ├── math_models.py     # All math models: RL_n, B_n, DG_n, DG_z, RL_z, DG, RL_l, ADM
│   ├── synaptic.py        # NEF synaptic model (Nengo) — TODO
│   └── recurrent.py       # NEF recurrent model (Nengo) — TODO
├── fitting/               # Model fitting and parameter estimation
│   ├── losses.py          # Universal NLL loss function
│   └── fit.py             # Optuna fitting loop with k-fold CV
├── analysis/              # Analysis scripts and notebooks — TODO
├── experiments/           # New experiment definitions — TODO
├── utils/                 # Shared utilities
│   ├── paths.py           # Central path config — import data_path from here
│   └── uniform_encoders.py # Quasi-Monte Carlo encoder distributions for Nengo
├── jobs/                  # SLURM job scripts for cluster runs — TODO
│   ├── make_jobs.py
│   ├── submit_jobs.py
│   └── collect_jobs.py
└── notebooks/             # Exploratory notebooks
```

---

## Data Schema

All three task dataframes share a common column schema:

| Column | Description |
|---|---|
| `pid` | Participant ID |
| `trial` | Trial index |
| `observation` | Sequential observation index within trial (all tasks) |
| `stage` | Social round index — jiang only, retained for analysis |
| `value` | Stimulus input seen by participant or model |
| `response` | Participant or model output |

Jiang-specific columns: `network`, `who`, `degree`, `rd`, `stage`
Carrabin-specific columns: `qid` (unique stimulus sequence identifier)

---

## Models

### Mathematical models (`models/math_models.py`)

All models expose a unified interface:

```python
from models.math_models import run
estimates = run(params, save=False, trials=None)
```

`params` is a dict containing at minimum `"model_type"`, `"dataset"`, `"pid"`. Additional keys are model-specific.

| Dataset | Model | Parameters |
|---|---|---|
| carrabin | `RL_n` | alpha, sigma |
| carrabin | `B_n` | sigma |
| carrabin | `DG_n` | sigma |
| jiang | `DG_z` | z, beta |
| jiang | `RL_z` | alpha, z, beta |
| yoo | `DG` | sigma |
| yoo | `RL_l` | alpha, lambda, sigma |
| yoo | `ADM` | primacy, recency, nu, sigma |

### NEF models (`models/synaptic.py`, `models/recurrent.py`)
Biophysical Nengo models — not yet ported. Will expose the same `run(params)` interface.

---

## Fitting

Fitting uses Optuna with k-fold cross-validation (k=5 default). The objective function for each Optuna trial is the mean held-out NLL across folds. The universal loss is NLL with Gaussian noise for continuous tasks (carrabin, yoo) and a sigmoid decision rule for jiang.

```bash
python -m fitting.fit {dataset} {model_type} {pid}
```

Outputs saved per participant to `data/`:
- `{model_type}_{dataset}_{pid}_params.pkl`
- `{model_type}_{dataset}_{pid}_performance.pkl`
- `{model_type}_{dataset}_{pid}_cv_folds.pkl`

After fitting, run `rerun.py` to generate full model responses, then `collect.py` to aggregate into `{dataset}_models.pkl` for analysis. Both scripts are TODO.

Cluster parallelism is supported via MySQL Optuna storage — see `make_storage()` in `fitting/fit.py`.

---

## Environment

This project uses two coordinated environments to ensure identical reproduction locally and on the compute cluster:

```bash
conda activate PY311
source venv/bin/activate
```

Both environments are kept in sync. `environment.yml` and `requirements.txt` are the sources of truth.

---

## Workflow

1. **Discussion** between researcher and Claude (claude.ai) to scope changes or new experiments
2. **Cursor prompts** drafted by Claude and executed by Cursor in the local environment
3. **Review and commit** by the researcher before any changes are pushed

See `.cursorrules` for Cursor's operating instructions.

---

## Status

- [x] Port and refactor core utility code (`uniform_encoders.py`, `paths.py`)
- [x] Standardize data schema across all three task dataframes
- [x] Port and refactor mathematical models (`math_models.py`)
- [x] Implement universal NLL loss function (`losses.py`)
- [x] Implement Optuna fitting loop with k-fold CV (`fit.py`)
- [ ] Port NEF models (`synaptic.py`, `recurrent.py`)
- [ ] Implement `rerun.py` and `collect.py`
- [ ] Implement SLURM job scripts (`jobs/`)
- [ ] Validate refactored code reproduces original results
- [ ] Design and implement new experiments
- [ ] Final analysis and figure generation