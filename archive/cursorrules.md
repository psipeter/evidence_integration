# Cursor Rules — evidence_integration

## Project context

Computational and neural modeling of **sequential evidence integration** with a **power-law learning rate** \(\alpha(t) = \alpha_0 / t^{\lambda}\). **Active behavioral tasks:** **carrabin** and **yoo** only—individual differences in **primacy vs. recency** weighting of observations. Implementations span **`models/math_models.py`** (compact equations) and **`models/NEF.py`** (spiking **Nengo** networks via the **Neural Engineering Framework**).

---

## Active datasets

- **`carrabin`** — Prat-Carrabin & Woodford-style slider integration; **`qid`** links observations to question identity.
- **`yoo`** — Yoo et al.-style slider task; rich trial × observation trajectories for update dynamics.
# Diederen / NEF2d: archived. See archive/misc/cursorrules_diederen.md

**Jiang**, **usher**, and related experiment code are **archived under `archive/`**. Do **not** add dataset branches for them to active modules (`fitting/`, `models/`, `utils/`, `scripts/` outside archive). Do **not** `import` from **`archive/`** in production code paths.

---

## Workflow (Cursor)

1. Researcher and Claude agree on design.
2. Claude emits an explicit numbered **Cursor Prompt** (self-contained markdown).
3. Cursor implements that prompt only.

Add `# TODO: [decision needed]` when requirements are ambiguous and choose the conservative default.

### Cursor prompt format

- Title: `Cursor Prompt NNN — Short description`
- Body: one fenced markdown block
- Number prompts **sequentially** across the project (never reset)
- Paths relative to repo root; one prompt may span multiple labeled sections

---

## Code conventions

### Layout

| Concern | Location |
|---------|-----------|
| Math models | `models/math_models.py` |
| NEF build / simulate | `models/NEF.py` (+ `counting_*.py`) |
| Search spaces & NEF fixed scalars | `fitting/model_params.py` → **`MODEL_PARAMS`**, **`_NEF_FIXED`** |
| Training objective | `fitting/losses.py` → **`response_loss`** / **`compute_loss`** |
| Optuna orchestration | `fitting/fit.py` |
| Job enumeration / SLURM / `--local` | `fitting/submit.py` |
| Aggregating pickles | `fitting/collect.py` |
| Paths, palettes, SLURM templates | `utils/paths.py`, `utils/plot_style.py`, `utils/slurm.py` |
| Load merged params + **`trial_seed`** | `utils/run_params.py` |
| Regenerate NEF responses | `utils/save_responses.py` |
| Save spikes / activities | `utils/save_activities.py`, `utils/plot_spikes.py` |
| Publication figures | `scripts/figure_carrabin.py`, `scripts/figure_yoo.py` |

Prefer **new figure panels** inside existing **`figure_*.py`** scripts rather than spawning many one-off plotting scripts.

### Parameter naming

- Power-law / NEF fitted scalars: **`alpha_0`**, **`lambda_`** (trailing underscore on lambda).
- Plain RL: **`alpha`**.
- Carrabin NoisyCounting: **`mu`**, **`sigma_c`**, **`nu`**.
- Yoo ADM: **`phi`**, **`rho`** (see **`MODEL_PARAMS`** for ranges).
- Fixed NEF architecture (times, radii, neuron counts, **LMU** settings, **`pes_learning_rate`**, …): **`_NEF_FIXED`** merged into **`PARAM_DEFAULTS`** then overridden by fitted rows from pickles.

Merge order when rebuilding dicts for simulation: **`PARAM_DEFAULTS`** \< **`_NEF_FIXED`** \< **fitted Optuna params** (see **`utils.run_params.load_run_params`**).

### Seeds

Use **`trial_seed(base_seed, trial_number)`** from **`utils.run_params`** everywhere a per-trial RNG seed is needed. **`models/NEF.py`** and **`models/math_models.py`** import it as **`_trial_seed`**—do **not** reintroduce local definitions.

### Performance pickles

- **Current fits:** columns include **`loss`**, **`runtime`** (and identifiers **`model_type`**, **`dataset`**, **`pid`**).
- **Legacy folders:** may use **`cv_loss_mean`** instead of **`loss`**.

In **`scripts/figure_*.py`**, read loss via **`_get_loss(perf_df)`** (fallback **`cv_loss_mean`**), never hard-code **`["cv_loss_mean"]`** alone.

### Run folders

Pass **`run_folder`** as a **short name** resolved under **`RUNS_DIR`** (e.g. **`response`**, **`test_local`**)—that is what **`fitting.submit`** / **`fitting.collect`** expect.

Avoid passing **`data/runs/foo`** strings in hand-written commands; **`utils.paths.resolve_run_folder`** will normalize common mistakes, but short names prevent confusion.

Cluster submission expands to absolute paths internally; local **`python -m fitting.fit`** should still receive either a short name or an **absolute** path—avoid ambiguous **`data/runs/...`** relative strings unless you rely on **`resolve_run_folder`**.

### Fitting pipeline facts

- **`fitting.submit`** writes **`run_config.json`** listing jobs.
- **`fitting.fit`** creates **`study`** with TPESampler; objective = mean **k-fold** **`response_loss`** (RMSE).
- **No `loss_type` CLI or parameter**—only RMSE response matching.
- **NEF:** one simulation per Optuna trial covering all behavioral trials; CV splits by trial IDs held out from that cached dataframe.
- **Warm-start:** if **`RL_lambda_{dataset}_{pid}_params.pkl`** exists in the run folder before an **NEF** fit, **`alpha_0`** and **`lambda_`** from that row are enqueued as trial 0.

### Unified model API

```python
run(params: dict, save: bool = False, trials: list | None = None) -> pd.DataFrame
```

**`params`** must include **`model_type`**, **`dataset`**, **`pid`**, plus model-specific keys. Output columns always include **`model_type`**, **`pid`**, **`trial`**, **`observation`** (or task-appropriate indices), **`response`**.

### Local completion marker

Functions invoked under **`fitting.submit --local`** must print **`JOB_COMPLETE`** as the **final** stdout line.

---

## What NOT to do

- Do **not** wire **jiang / usher** back into **`MODEL_PARAMS`**, **`losses.response_loss`**, or **`collect`** without an explicit archival reversal plan.
- Do **not** add **`loss_type`**, **`shape_loss`**, **`joint_loss`**, softmax **`beta`**, **`alpha_bias_array`**, or **`rd`** bias hooks to active **NEF** / fitting code (historical variants live in **`archive/`**).
- Do **not** redefine **`_trial_seed`** locally.
- Do **not** read **`cv_loss_mean`** directly in figures—use **`_get_loss`**.
- Do **not** commit secrets, push git, or edit **`archive/`** unless the prompt explicitly requires it.

---

## Adding a new dataset

1. Add **`data/<dataset>.pkl`** with columns compatible with **`fitting.fit`** queries (at minimum **`pid`**, **`trial`**, **`observation`**, **`value`**, **`response`**).
2. Extend **`MODEL_PARAMS[<dataset>]`** in **`fitting/model_params.py`**.
3. Implement **`_run_<dataset>`** (or equivalent) in **`models/math_models.py`** and register in dataset / model validation.
4. Allow **`response_loss`** to accept the new **`params["dataset"]`** in **`fitting/losses.py`**.
5. Add **`fitting.collect`** awareness if new model types need special casing (often automatic via **`run_config.json`**).
6. Add **`scripts/figure_<dataset>.py`** mirroring structure of **`figure_yoo.py`** where possible.
7. Add / update **`DEFAULT_TIME_LIMITS`** / **`DEFAULT_MEM_LIMITS`** in **`utils/slurm.py`** for any new **`model_type`** strings.

---

## NEF implementation notes

- Build with **`build_network(obs_values, params, decoders)`** after **`_pretrain(params)`** for the counting subsystem.
- **`counting`** in **`_NEF_FIXED`**: **`"integrator"`** or **`"lmu"`**.
- **`base_seed`** (from params) stabilizes counting pretrain; **`trial_seed`** separates per-trial variability in the main network.
- Default scale: **`n_neurons=200`** (value / error), **`n_neurons_counting=1000`**, **`lmu_n_obs_max=30`** (max sequence length for LMU pretraining).
- Synaptic variant: **`NEF_synaptic`**; check **`PARAM_DEFAULTS`** / **`_NEF_FIXED`** for **`pes_learning_rate`** and time constants (**`tau_*`**, **`T_error`**, …).

---

## Key file relationships

- **`fitting/fit.py`** → **`models.NEF`**, **`models.math_models`**, **`fitting.losses`**, **`fitting.model_params`**, **`utils.paths`**, **`utils.save_responses`**
- **`utils/save_activities.py`** → **`models.NEF`**, **`utils.run_params`**, **`utils.paths`**
- **`scripts/figure_yoo.py`** / **`scripts/figure_carrabin.py`** → **`fitting.losses`** (diagnostics), **`utils.plot_style`**, **`utils.paths`**
- **`models/NEF.py`** → **`fitting.model_params`** (**`_NEF_FIXED`** only). **Keep this dependency shallow**—do not import arbitrary fitting modules from **`models/`**.

---

## Common commands

```bash
# One participant, local CLI (positional — no --n_trials flag)
python -m fitting.fit carrabin RL_lambda 1 500 5 response 42

# Submit one model × all PIDs
python -m fitting.submit carrabin RL_lambda --n_trials 500 --run_folder response

python -m fitting.collect response --type params
python -m fitting.collect response --type responses

python -m fitting.submit --resubmit params --run_folder response

python -m utils.save_activities carrabin NEF_recurrent 1 response error once_per_obs

python scripts/figure_carrabin.py --run_folder response
python scripts/figure_yoo.py --run_folder response --noise_folder yoo_response_noise
python scripts/dynamics_NEF.py --dataset carrabin --pid 1 --run_folder response

python scripts/check_jobs.py --cancel   # inspect / cleanup SLURM artifacts as documented in script
```

---

## Style & tooling

- **Python 3.11**; format with **`black`**; prefer **`pathlib`** via **`utils.paths`** (**`data_path`**, **`RUNS_DIR`**, **`resolve_run_folder`**, **`FIGURES_DIR`**).
- Figures: save **PNG** (300 dpi) + **PDF** under **`figures/`** from scripts; avoid **`plt.show()`** in batch figure scripts.
