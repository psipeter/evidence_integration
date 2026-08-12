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

## Sequence-generation diagnostic scripts (superseded by scripts/plot_sequences.py)

Moved when `scripts/inspect_sequences.py` and `scripts/inspect_iid_sequences.py`
were consolidated into `scripts/plot_sequences.py` (two branches --
`across_models` and `across_pids` -- both reading exclusively from
task_backend's real, deployed sequence pool). That consolidation surfaced
four more scripts that were entirely in service of the OLD, now-inactive
pipeline and had no remaining purpose once it was retired:

### archive/scripts/test_sequences.py
Produced `figures/test_sequences.pdf` (3-row x 7-col cross-task RL_lambda/
NEF-vs-Human comparison, quartile-split panels). Read
`task/sequences/{task}_sequences.pkl` -- the OLD task/ pipeline's single
shared reference file -- via columns (`trial_type`, `std_condition`) that
don't exist in ANY current schema, task/'s current one or task_backend's;
this script's own main()/figure pipeline was already stale relative to
current production sequences before this move, not just redundant.
Three of its functions (`fit_lambda_mid`, `split_half_lambda`,
`compute_abs_delta`) were genuine, still-needed utilities with NO
dependency on that stale schema -- confirmed directly, not assumed --
and were inlined into `plot_sequences.py` before this file was archived,
rather than lost.

### archive/scripts/run_nef_sequences.py, submit_nef_sequences.py, collect_nef_sequences.py
A SLURM job-submission/collection trio that existed purely to feed
test_sequences.py's own `test_sequences_responses.pkl` pipeline with a
lambda-swept NEF simulation (one job per lambda value, run on the
cluster). Reads the same stale `task/sequences/*.pkl` file layout.
Once test_sequences.py's own figure was retired, this trio had no
remaining consumer -- `plot_sequences.py`'s own NEF path
(`simulate_nef_task`/`load_or_simulate_nef`) runs directly and
synchronously against ONE representative task_backend pool member, with
no job-submission step needed at all.

### Not moved, but flagged for awareness: scripts/pilot_overview.py
Still in `scripts/` -- NOT moved here, since it's a different concern
from sequence-generation diagnostics (it compares REAL collected pilot
participant data against models, not simulated agents against generated
sequences) and wasn't redundant with `plot_sequences.py` under the
criterion this cleanup pass used. It IS, however, likely superseded by
`scripts/figure_soltani_{performance,temporal,variability}.py`, which
cover the same real-data-vs-model comparison via the project's normal
PTN-figure convention (properly Optuna-fitted models, participant-
filtered `data/task_{continuous,binary}.pkl` inputs) rather than
pilot_overview.py's fixed/hand-tuned model parameters and ad-hoc trial-
based join against `task/sequences/*.json`. Left for a separate,
deliberate decision rather than archived under this pass's criterion.

### How to restore
Copy the four files back to `scripts/`; `plot_sequences.py`'s own inlined
copies of `fit_lambda_mid`/`split_half_lambda`/`compute_abs_delta` would
then be a duplicate of test_sequences.py's originals (harmless, but worth
de-duplicating back to a single cross-file import at that point).

---

## archive/task_backend/ -- production sequence snapshots

### sequences_numbers_std15.json, tutorial_sequence_numbers_std15.json
Backed up when `task_backend/generate_sequences.py`'s `NUMBERS_STD_FIXED`
was reverted 15 -> 10 (a real pilot with std=15 showed little to no clear
|delta response| decay signal in real participants -- see
figure_soltani_temporal.py and chat history -- so the PI wanted to test
whether std=15 was simply too noisy a task). This is the SECOND
reversal of this constant (10 -> 15 -> 10, see generate_sequences.py's
own comment on `NUMBERS_STD_FIXED` for the full history of both). These
two files are exactly what was live in production immediately before the
second reversal: the full 200-member std=15 numbers pool, and the fixed
tutorial example chosen from it. Also recoverable via git history
(commit `443fe1e`), but kept here too as an explicit, easy-to-find
snapshot rather than relying solely on git log.

### How to restore
Copy both files back to `task_backend/` (dropping the `_std15` suffix),
set `NUMBERS_STD_FIXED = 15.0` back in `generate_sequences.py`, and
rebuild the frontend (`npm run build`) to confirm. No need to re-run
`--tutorial` if restoring these exact files -- they already ARE that
selection's output.

---

## archive/utils/archive_participant_filters_legacy.py

The original (pre-Cohen's-f²-reframe) version of `utils/participant_
filters.py`'s exclusion criteria. Archived when the PI raised a direct
concern about defensibility: the original version used three different
statistical objects (a tolerance-based literal-copy fraction, a binomial
test, a raw Pearson correlation, and a partial-correlation-with-a-hand-
picked-r=0.10-cutoff), each individually defensible but collectively
reading as an ad hoc patchwork rather than one principled measurement.
The replacement applies ONE consistent statistical framework (nested
regression + Cohen's f² effect size, f²=0.02 "small effect" convention)
to the same three underlying questions.

The archived `flag_recency_only` (r=0.10 partial-correlation version) has
a real, if narrow, blind spot the current f²-based version doesn't: it
missed 2 real participants that the ORIGINAL `flag_no_integration`
(literal-copy check, also archived here) had caught, because Cohen's
f²=0.02 convention for a regression's added-predictor effect size
corresponds to roughly r=0.14, not r=0.10 -- confirmed directly against
real data, not assumed. Nothing here still runs as part of the active
pipeline -- see `utils/participant_filters.py`'s own module docstring and
`CLAUDE.md`'s "Participant exclusion criteria" section for the current,
settled account.

### How to restore
Copy `flag_no_integration`, `_direction_and_magnitude_stats`,
`flag_noncontingent_sign`, `flag_noncontingent_magnitude`,
`flag_recency_only`, and `compute_exclusion_report` from this file back
into `utils/participant_filters.py` (renaming the current f²-based
`compute_exclusion_report` to something else first, e.g.
`compute_exclusion_report_f2`, to avoid a name collision). The shared
helpers this file imports from the active module
(`_compute_updates`/`_compute_recency_features`) would need copying back
in too if the active module's own versions of them ever diverge.

---
