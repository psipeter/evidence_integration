# fitting-pipeline

Use this skill when submitting, collecting, or reasoning about the exact
CLI mechanics of model-fitting jobs (`fitting.submit`/`fitting.fit`/
`fitting.collect`) — RMSE or NLL, any dataset. For which models exist and
why NLL/RMSE differ scientifically, see `docs/SCIENCE.md`/
`docs/DECISIONS.md`; for the `dataset` vs `--datafile` conceptual split,
see `CLAUDE.md`'s "Active datasets" section. This skill is the "how to
actually run it" reference.

---

## RMSE fitting

```bash
# Submit (cluster)
venv/bin/python -m fitting.submit carrabin NEF --n_trials 100 --run_folder carrabin --k 5
venv/bin/python -m fitting.submit yoo NEF --run_folder yoo --n_trials 100 --k 5

# Collect params and responses
venv/bin/python -m fitting.collect carrabin --type params
venv/bin/python -m fitting.collect carrabin --type responses

# Collect activities (after responses; needed for neural figures)
venv/bin/python -m fitting.collect yoo --type activities --ensembles error --timing once_per_obs
```

Run folders in current use: `data/runs/carrabin/`, `data/runs/yoo/`,
`data/runs/refit/`, `data/runs/rmse/`, `data/runs/nll/`,
`data/runs/nll_noise_only/`. `--nef_folder` in figure scripts redirects
NEF data to a separate folder from other models.

## soltani math-model fits

RMSE and NLL fits use two separate run folders (`data/runs/rmse/`,
`data/runs/nll/`) — NOT the older shared `data/runs/soltani/`, which
holds stale fits made against contaminated/smaller-pid-count data and is
read by no current figure (why the split happened:
`docs/DECISIONS.md`). Omit `--datafile` for the canonical, unsuffixed
`data/soltani_{numbers,colors}.pkl` (46 pids, contamination-free,
registry-stable — see the data-pipeline skill for how it's built).

`all` expands to every model including NEF with no skip flag — to fit
only math models, submit one at a time:

```bash
for m in Mean LeakyIntegrator PrimacyRecency RL_lambda; do
  venv/bin/python -m fitting.submit soltani_numbers $m --n_trials 300 --k 5 --run_folder rmse
done
venv/bin/python -m fitting.collect rmse --type params
venv/bin/python -m fitting.collect rmse --type responses
```

NLL fits (add `--loss nll`; every model needs its own `_resp_noise`
suffix — NEF's and `NoisyRL_lambda`'s own NLL branches are retired, see
`docs/DECISIONS.md`, so `--loss nll` only works on a `<model>_resp_noise`
name):

```bash
for m in Mean_resp_noise LeakyIntegrator_resp_noise PrimacyRecency_resp_noise RL_lambda_resp_noise; do
  venv/bin/python -m fitting.submit soltani_numbers $m --n_trials 300 --k 5 --run_folder nll --loss nll
done
venv/bin/python -m fitting.collect nll --type params
venv/bin/python -m fitting.collect nll --type responses
```

NEF's SLURM limits are 72h/32G (`utils/slurm.py`). `fitting.fit`'s CLI is
argparse-based (positional `dataset model_type pid`, then
`--n_trials/--k/--run_folder/--optuna_seed/--datafile`).

## NLL mechanics

- `fitting.fit(..., loss_fn="nll", n_sims=100)` dispatches to
  `math_models.add_noise` — the only ensemble source left active.
  Checked before the Optuna study is created, so an invalid model/loss
  combination fails immediately, not on the first trial.
- **Filenames**: NLL output files get a `_nll` suffix inserted before
  `{pid}` (`{model_type}_{stem}_nll_{pid}_*.pkl`), so an NLL fit can
  never silently overwrite an RMSE fit of the same model_type.
- **Default method — noise-only override**: NLL fits fix the base
  model's free parameters at their RMSE-fitted values and search ONLY
  `sigma_resp`, via `--override_from_folder <folder>`. Verified negligible
  loss/behavior difference vs. a full joint search in 11/12 tested combos
  — the one exception (RL_lambda on carrabin) is worth re-checking before
  relying on this for that specific cell.
  `data/runs/nll_noise_only/` is the canonical location for new fits of
  this kind; `data/runs/nll/` (full-joint) is kept as the verification
  baseline, not actively added to.
- **Before trusting any `--loss nll` fit on a new dataset/model
  combination**, run `scripts/verify_ensemble_invariant.py` (after
  touching `add_noise`, `_resp_noise_seed`, or
  `_validate_model_dataset`'s allowlists). There is no pytest suite in
  this project — this script is the closest equivalent for the NLL path.

## MLE fitting

Retired — see `docs/DECISIONS.md`. Was `NoisyCounting`-only (carrabin);
code archived at `archive/fitting/archive_fit_mle.py`.
