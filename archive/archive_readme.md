---
# Archive

This folder contains code for the jiang and usher tasks, removed from
the active codebase when the project focus shifted to the Diederen task.

## Contents

### archive/models/archive_math_models.py
Model classes extracted from models/math_models.py:
- `PopulationCoding`: Brezis, Bronfman & Usher (2018, Neural Computation)
  population coding model. Represents mean as population code; error
  scales with sequence mean magnitude and variance. Free parameter: gain.
- `PoissonCoding`: variant of PopulationCoding with Poisson noise. Best
  RMSE on usher (0.082) but flat variance sensitivity (seq_std slope 0.130
  vs human 0.276).
- `EmpiricalWeights`: fits per-pid serial position weights directly from
  data. Used as empirical upper bound on per-pid fits.
- `RL_lambda_boost`: RL_lambda with an additive boost to the weight at the
  final observation. Tested for usher obs10 spike; abandoned (RMSE=0.134,
  worst model; seq_std slope≈0; compression ratio=1.384).
- `RL_lambda_rd`: RL_lambda with network-degree-weighted observations for
  the jiang social network task.
- `DeGroot`: DeGroot opinion-averaging model for jiang.

### archive/fitting/archive_losses.py
Loss functions extracted from fitting/losses.py:
- `_observations_switch_conflict()`: switch/conflict metrics for jiang
  social network trials.
- `_apply_beta_sampling()`: converts continuous model responses to binary
  ±1 via sigmoid(beta * response) for jiang choice modelling.
- `_logistic_switch_slope()`: logistic regression of P(switch) vs true_rd
  per participant.
- `_compute_nd_coef_loss()`: OLS coefficient for ND-weighted observations
  predicting response sign (jiang shape loss).

### archive/fitting/archive_model_params.py
Parameter search spaces for jiang and usher tasks, extracted from
fitting/model_params.py. Includes _NEF_RANGES_JIANG (adds beta parameter
for the binary choice logistic link).

### archive/misc/archive_misc.py
Small code blocks from dataset-specific branches in models/NEF.py,
fitting/fit.py, fitting/submit.py, fitting/collect.py.

### archive/scripts/
- figure_usher.py: summary figure for the usher task (panels A–H).
- figure_jiang.py: summary figure for the jiang task.
- dynamics_NEF.py: NEF simulation dynamics (if jiang-specific).

## Tasks

### Jiang task
Rosenbaum et al. (2021, unpublished variant). N=97 participants, 81
trials, 10 observations per trial. Binary social network task: participants
reported a directional estimate after each stage of neighbour exchange.
Loss: negative log-likelihood of binary choices under sigmoid(beta *
model_expectation). Shape loss: ND-weighted OLS coefficient predicting
response sign.
Models: Bayes, DeGroot, RL, RL_lambda, RL_lambda_rd, NEF_recurrent,
NEF_synaptic.
Key finding: NEF per-pid power-law fitting captured individual differences
in social influence weighting. A rd-bias node in the NEF error ensemble
modelled network degree weighting.

### Usher task
Rosenbaum, Glickman & Usher (2021, Front Psychol,
DOI:10.3389/fpsyg.2021.693575). N=97 participants, 81 trials, 10
observations per trial, single continuous response at obs 10.
Models: Mean, RL, RL_lambda, PopulationCoding, PoissonCoding,
EmpiricalWeights, RL_lambda_boost, NEF_recurrent.
Key findings:
- PoissonCoding: best RMSE (0.082), flat variance sensitivity.
- NEF: moderate RMSE (0.109), best seq_std slope (0.151),
  anti-compressed responses (ratio 1.243).
- Human serial weights: flat obs1-9 (~0.10) with recency spike at
  obs10 (mean 0.213).
Key decision — obs10_boost abandoned: population-level boost of 0.113
produced RMSE=0.134 (worst), seq_std slope≈0.023, compression ratio=1.384.
Per-pid OLS weights too noisy (81 trials, 10 predictors) to justify
per-pid values.

## Why archived

Both tasks have a single end-of-sequence response, limiting
characterisation of trial-by-trial learning dynamics. The yoo task
(N=38, 30 observations, response after every observation) provides richer
data for fitting and validating the NEF model.

## How to restore

1. Copy archive/models/archive_math_models.py classes back to
   models/math_models.py
2. Merge archive/fitting/archive_losses.py functions back into
   fitting/losses.py
3. Merge archive/fitting/archive_model_params.py entries back into
   fitting/model_params.py
4. Re-insert archive/misc/archive_misc.py blocks into their original
   source files (provenance comments indicate origin)
5. Move archive/scripts/*.py back to scripts/
6. Copy archive/data/jiang.pkl and archive/data/usher.pkl back to data/

## Data files

Archived task pickles live in ``archive/data/``:

- ``jiang.pkl`` — moved from ``data/jiang.pkl`` (social network task).
- ``usher.pkl`` — moved from ``data/usher.pkl`` (Rosenbaum, Glickman & Usher 2021).
- ``jiang_networks.npy`` — moved from ``data/jiang_networks.npy`` (shape 7×7×43
  adjacency matrices for jiang trials). No ``jiang_networks.pkl`` was present in
  the repo; the network supplement file uses the ``.npy`` extension. Social
  network graph structures (node degrees, adjacency matrices) for jiang task
  trials. Originally used by ``save_activities.py`` to inject ``rd`` (network
  degree) values into the NEF error ensemble via ``alpha_bias_array``, and by
  the jiang Bayes model in ``archive/models/archive_math_models.py``. Archived
  alongside ``jiang.pkl`` when the jiang task was removed from the active codebase.

Active datasets remain in ``data/``: ``carrabin.pkl``, ``yoo.pkl``, ``diederen.pkl``.

---
