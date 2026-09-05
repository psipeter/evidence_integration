# archive/HISTORY_modeling_2026.md — participant exclusion, model development, NEF fitting history

Archived, frozen narrative. For current model/fitting conventions see
CLAUDE.md; for the decisions here, see docs/DECISIONS.md and
docs/SCIENCE.md (current scientific state).

---


Started from the hypothesis that the existing filter was TOO AGGRESSIVE (it
excluded 33/60 = 55% of complete_pairs as a union). That hypothesis turned out to
be wrong, but chasing it produced a better criterion and, more usefully, a
definition to justify it. Recording the progression because most of the dead ends
are ones that would be re-attempted otherwise.

### The four candidates

| method | basis | numbers | colors |
|--------|-------|---------|--------|
| `contingency` | three Cohen's f² tests (recency_only, noncontingent sign/magnitude) | 25/60 (42%) | 19/60 (32%) |
| `performance` | carrabin's rule: mean abs error >N SD above the retained mean | 9/60 (15%) | 9/60 (15%) |
| `integration` | skill score vs "copy the latest observation" | 36/61 (59%) | ~1 |
| **`non_integrator`** | **prior observations make no RELIABLE contribution** | **19/61 (31%)** | **17/61 (28%)** |

`non_integrator` is now the default. `performance` and `integration` moved to
`archive/utils/archive_exclusion_criteria.py`. `contingency` retained as a
computed DIAGNOSTIC (it no longer decides anything) because recency_only tests the
same construct as non_integrator by a different method, and their agreement --
23/25 and 18/19 -- is what validates the exclusions.

### Published precedent, which is what prompted the whole investigation

carrabin excluded 4/25 (16%) on ONE model-free quantity: mean |p̂ − p|, with the
excluded group at .263 (SD .0298) against .176 (SD .0132), a >6 SD separation.
yoo excluded 8/46 (17%), of which SEVEN were fMRI-technical (1 structural
abnormality, 6 head motion >3 mm) and exactly ONE was behavioural -- a
post-experiment questionnaire in which the subject said they tracked pairwise
differences rather than the average. Neither used a model-based contingency test.
Applying our own criteria back to their data: carrabin's 21 retained subjects all
score skill 1.00-1.51 (0 would fail ours), but 7 of yoo's 38 (18%) WOULD fail --
so our criterion is stricter than field norms, which is worth stating.

### Why the high rate is real, not an artefact

The `integration` criterion is model-free, shares no quantity with the temporal
panels, and independently reproduced 23/25 and 18/19 of the contingency
exclusions. Roughly half of numbers participants perform worse than reporting
ONLY the latest observation (mean error 8.35 vs an optimal 3.28). The original
"too aggressive" hypothesis is disconfirmed.

### What the bad participants are actually DOING

Assigning each participant the candidate strategy that best predicts their
responses. Zero good participants are best fit by `last_value`; zero bad ones by
`running_mean`. But there are TWO distinct failure modes, not one:

- **Literal copying** (23 of 37 bad numbers participants). Exact-copy rates of
  0.63-1.00, best-fit error down to 0.0-1.6 points. Transcribing the stimulus.
- **Drifting** (11 of 37). Best fit by their OWN PREVIOUS RESPONSE, moving the
  slider constantly (within-trial SD 7.7-27.1 vs 4.4 for good participants) with
  no relation to the evidence. Not a sticky slider -- they move MORE than good
  participants.

That second mode is why a single-axis filter cannot work: copiers sit at one
extreme of any weighting measure and drifters sit in the middle, alongside genuine
integrators.

Contributing context: they are NOT rushing (median inter-observation latency 4.2 s
vs 3.4 s for retained -- SLOWER), not timing out, not leaving the slider still,
and passed the tutorial first try. Comprehension failure, not inattention. Which
explains why five rounds of instruction and bonus tweaks did not move it. The
tutorial teaches but has NO pass/fail gate; gating it on demonstrated performance
is the highest-value fix available.

### Measured and rejected

- **`frac_copy_value`** as a metric. Confounded by `true_std`: at std=10, copying
  the latest value is nearly correct. Superseded by skill/`g_lag0`, which
  normalise per participant.
- **The skill score** (`integration`). Its threshold was defensible -- a 0.29-wide
  empirical void, so any cut in (0.041, 0.334) gave the identical partition -- but
  the METRIC is not monotone in integration depth. On synthetic leaky integrators
  it PEAKS at α=0.20 (+0.745), above a near-optimal α=0.10 (+0.603), because with
  15 observations mild recency overweighting tracks the running mean better than a
  sluggish filter. A genuine α=0.70 integrator with realistic noise scores +0.115,
  a hair above its own 0.10 threshold. It discards inaccurate integrators.
- **`g_lag0`**, the serial-position weight on the latest observation from
  regressing each response on ALL prior observations. Recovers α almost exactly
  (0.100/0.200/0.350/0.494/0.687/0.959 for true α 0.10-1.00) and is nearly immune
  to response noise -- so it IS the right MEASURE of integration depth. But it is
  continuous with no natural cutoff (largest gap 0.076 across 0.03-1.00) and
  CANNOT catch random responders, whose diffuse weights score ~0.12,
  indistinguishable from optimal. Any weight-based test is blind to the "nothing
  predicts them" mode. Report it descriptively; do not filter on it.
- **`gain`** (b_current + b_prior ≈ 1 as a validity check). Catches random and
  frozen responders cleanly (0.00 vs 0.94-1.00) but was rejected as a filter
  component in favour of a single test.
- **A one-sided version** of the final criterion, to catch scale inversion
  (reporting % red for % blue). 1 of 61 numbers and 0 of 61 colors are reliably
  negative, and that one is marginal (b=-0.074, CI [-0.171,-0.013]).
- **Stability across session halves.** 26% of retained numbers participants pass
  pooled but not both halves -- and the asymmetry runs the WRONG way for fatigue:
  12 integrate only in the SECOND half against 4 only in the first. Mostly late
  LEARNING, consistent with error falling 19% from the first 8 to the last 8
  trials. Would penalise a slow start.
- **A trials 8-31 burn-in.** Moves retention by ONE participant per task (numbers
  42→41, colors 43→44), retained sets indistinguishable in accuracy on the same
  late trials (4.90 vs 4.79; 6.94 vs 7.03). Use all 32.

### Fatigue: there is none

Within task, error DECREASES with trial index (numbers -0.075/trial p=0.001;
colors -0.066 p=0.004), first 8 → last 8 trials 9.63 → 7.76 (-19.4%). Across
tasks the second task is if anything better (skill 0.262 → 0.380, p=0.45), with
order well counterbalanced (32 numbers-first, 28 colors-first). Only 3 numbers and
1 colors participant worsen beyond their own noise (per-pid slope t>2); the 22
with positive slopes are the upper half of a null distribution, and 22/60 is BELOW
the 30/60 expected by chance. The slope IS a reliable individual trait (split-half
r=0.758 numbers) -- but the reliable variation is in how much people IMPROVE.
Low-skill participants improve FASTEST (corr(skill, slope) = +0.297, p=0.021),
which is the opposite of "the disengaged fatigue".

Trial order is randomised per participant (54 and 53 distinct orders across 60),
and adjusting each trial's error for that trial's optimal-agent error leaves the
slope unchanged (-0.0751 → -0.0741), so the improvement is not a sequence artefact.

### `require_both_tasks` became the default

A participant failing in either task is dropped from BOTH. Found by following up a
collapse in the within-subject cross-task panels: under per-task exclusion with
`integration`, numbers retained 29 and colors 36, and the 26-pid intersection was
a differently-selected group -- cross-task λ correlation fell to r=0.331 (p=0.099)
from r=0.587 (p=0.0013). NOT power (cp_perf at n=44 gives r=0.572), NOT reliability
(λ split-half was HIGHER, colors 0.836 vs 0.796; attenuation ceilings 0.791 vs
0.780), NOT range restriction (λ SD and range unchanged). Purely the composition
of the intersection.

### Effect on the results

| build | n both | cross-task λ r | col 3 p | col 2 decay | λ numbers |
|-------|--------|----------------|---------|-------------|-----------|
| contingency (`complete_pairs`) | 27 | 0.587 (p=.0013) | 0.00043 | 3.09x | 0.433 |
| **non_integrator (`cp_ni`)** | **35** | **0.508 (p=.0018)** | **0.00051** | **2.46x** | **0.390** |
| integration | 24 | 0.338 (p=.107) | 0.00000 | 2.90x | 0.474 |
| performance | 44 | 0.572 (p<.0001) | 0.033 | 1.23x | 0.266 |
| no filter | 60 | 0.656 (p<.0001) | 0.011 | 1.13x | 0.240 |

`non_integrator` gives 8 MORE participants than the contingency filter with every
result intact -- the only criterion that improves the sample without weakening
anything.

Two asymmetries worth carrying into any write-up. The DECAY results (cols 2-3)
need a filter and weaken monotonically as it loosens. The CROSS-TASK λ correlation
does NOT -- it is strongest with everyone included (r=0.656, n=60), so that finding
depends on no exclusion at all. Different panels have different sensitivity to
exclusion; report the asymmetry rather than smoothing it over.

### Honest limitations of the chosen criterion

Not threshold-free, though I described it that way at one point. It removes the
arbitrary MAGNITUDE threshold and replaces it with a conventional significance
level, whose sensitivity is: ci=90/95/99 → 16/17/24 flagged (numbers), 17/17/20
(colors), so ci=99 adds 7 (+41%). Bootstrap seed moved 2-3 participants at
n_boot=2000, hence the default n_boot=20000 (verified: seeds 0/1/2 identical, ~10 s
per task via a Gram-matrix bootstrap). The PREDICTOR SET is the largest source of
variation -- last-3-lags + older mean gives numbers 23 (churn +10/-4) and colors 15
(+3/-5) -- but `prior_mean` is right on principle: it asks the definitional
question as ONE test, whereas the full-lag version widens every CI (power loss →
more flagged) while giving four uncorrected chances at significance (multiplicity →
fewer flagged); those errors moving in opposite directions across tasks is the
signature of an ill-posed test.

Known gaps, deliberately not engineered around: it retains anyone whose responses
reliably use history, so it does not catch integrating the WRONG STATISTIC (running
sum, max, a subset), SCALE COMPRESSION (right direction, only 40-60 of the slider),
or ANCHORED-WITH-A-NUDGE. The first two are arguably correct to retain; the third
is a real miss. And being a significance test it is POWER-DEPENDENT: the ~30% rate
is tied to this design's 32 trials.

## NoisyRL_lambda: response noise reconciles the fitted/descriptive lambda gap (this session)

Started from a worry that RL_lambda was fitting badly: most fitted `lambda_` above
0.75 (9/35 pinned at exactly 1.0) against a descriptive lambda -- measured from the
decay of |Δresponse| -- below 0.5. Chasing it produced a resolution, a new model,
and three retracted hypotheses of mine along the way. Recording the whole
progression because the dead ends were each plausible.

### The fits were not bad; the comparison was mis-specified

Two things were measured before drawing any conclusion:
- **Parameter recovery works.** Fitting noiseless RL_lambda to noisy RL_lambda data
  recovers lambda almost exactly at every noise level tested (true 0.20 -> 0.192,
  0.40 -> 0.367, 0.60 -> 0.592, 0.80 -> 0.808, even at noise SD 0.10, double the
  human level). So misspecification-via-noise does NOT bias lambda upward.
- **lambda is strongly identified, not weakly.** The RMSE profile along lambda is
  steep and monotone (pid 1: 0.1032 at lambda=0 -> 0.0484 at 0.8). No flat region,
  no boundary drift. The optimiser finds a real minimum.

So the high fitted lambda is a finding about response LEVELS, not an artefact. The
two lambdas measure different things: fitted lambda answers "what weighting
reproduces where the slider ends up", descriptive lambda answers "how fast does the
amount of movement shrink".

### The mechanism, and what "noise" means here

Human |Δresponse| PLATEAUS (~0.06 on [-1,1]) rather than decaying to zero. A
deterministic RL_lambda has |Δ| -> 0 by construction, since |Δ| = alpha(t)*|PE| and
alpha(t) -> 0. Sequence variation keeps |PE| alive but cannot keep |Δ| alive -- the
gain shrinks regardless. This was a genuine confusion worth resolving explicitly:
the relevant noise is RESPONSE noise (slider imprecision, lapses, tick rounding),
NOT sequence variation. The two are separable because within a qid group the stimuli
are IDENTICAL, so within-qid residual SD measures response variation with sequences
held fixed. Measured: humans ~0.055; every math model exactly 0.000e+00.

Decisive test: adding each pid's OWN measured response noise to RL_lambda's fitted
output moves its descriptive lambda from 0.921 to 0.369 against a human 0.294 --
paired gap +0.008, p=0.668, indistinguishable -- with the plateau also matching
(gap -0.0044, p=0.599). For colors the deterministic model was already close and
noise overshoots, so this is largely a numbers-task phenomenon.

### PrimacyRecency does not show the discrepancy, and the reason is structural

At median fitted parameters, the normalised weight on the NEWEST observation:

| t | PrimacyRecency | RL_lambda alpha(t) |
|---|----------------|--------------------|
| 1 | 1.000 | 0.947 |
| 5 | 0.373 | 0.305 |
| 15 | 0.327 | 0.141 |

PR asymptotes to a CONSTANT (~0.33) because for `o=t` the recency factor is
`eps_r^1` at every t, so the newest observation always retains weight. RL_lambda's
alpha decays without bound. Hence PR's |Δ| plateaus by construction and RL_lambda's
cannot. Late-observation plateau, numbers: human 0.0633, PR 0.0445, RL_lambda
0.0223, Mean 0.0131. Descriptive-lambda gap ordered identically: PR +0.219,
RL_lambda +0.396, Mean +0.765. LeakyIntegrator overshoots the other way (plateau
0.0992, gap -0.232) because fixed gamma means constant weight forever. So the three
models BRACKET the human -- a coherent story about wanting a decaying-but-floored
learning rate.

### Retracted along the way

- **"Add an asymptotic term to alpha(t)."** Wrong. Response noise produces an
  |Δ| floor by itself (`E|Δ| -> 1.128*sigma` even when the systematic Δ -> 0), so
  an alpha floor would fit noise with a systematic parameter. Tested per pid: 13/35
  prefer a floored power law, but the fitted floor is statistically
  indistinguishable from the floor PREDICTED by each pid's own response noise
  (numbers median 0.0455 vs 0.0625 predicted, p=0.377, correlated r=0.661 across
  pids). Note this also invalidates the descriptive-lambda level we had been
  quoting: fitting the floored form gives lambda median 1.642 vs 0.294 for the pure
  form, so the noise floor drags the pure-form exponent down.
- **"LeakyIntegrator's `v=0.0` init handicaps it."** Wrong, and checked against the
  task rather than argued: `DEFAULT_VALUE = 50` and `lastResponse` is reset at every
  trial start, so each trial genuinely begins with the slider at the midpoint =
  exactly 0.0 on [-1,1]. The init IS the task's initial condition; initialising at
  x_0 would ignore where the slider actually was.
- **"PrimacyRecency's recency exponent disagrees with its docstring."** Wrong.
  Verified numerically identical under the docstring's stated 1-indexed convention;
  I had misread `eps_r**(n-o)` as giving `eps_r^0` on the last observation when it
  gives `eps_r^1`.

### The model, and what it establishes

`NoisyRL_lambda` = RL_lambda + `sigma_state` (perturbs the estimate, compounds ->
variance growth and autocorrelation, i.e. temporal cols 3-4) + `sigma_resp`
(perturbs only the report, i.i.d. -> a plateau, no autocorrelation). Reduces to
RL_lambda exactly at sigma=0.

RMSE cannot identify either sigma (both collapse to ~0; 24-25 of 35 exactly zero),
so both carry nonzero LOWER BOUNDS chosen by matching human prefix variability and
RMSE-vs-running-mean. With floors in place essentially every pid sits at them, so
the fitted values are not evidence about the noise level.

What it buys, with `alpha_0`/`lambda_` barely moved (numbers lambda 0.704 -> 0.662,
r=0.964):

| numbers | ratio first/last | plateau | descriptive lambda | gap vs human |
|---------|------------------|---------|--------------------|--------------|
| HUMAN | 2.46 | 0.0633 | 0.294 | -- |
| RL_lambda | 7.24 | 0.0223 | 0.921 | +0.382, p<0.0001 |
| NoisyRL_lambda | **2.50** | 0.0537 | 0.405 | **+0.035, p=0.62** |

Colors: plateau becomes exact (0.0853 vs 0.0854) and the per-pid lambda correlation
improves 0.782 -> 0.894.

**Circularity, stated honestly.** sigma_resp's floor was calibrated to the measured
within-qid residual SD, and the plateau is largely a function of that quantity -- so
the plateau match is partly by construction. NOT circular: nothing tied sigma to the
DECAY RATIO or the descriptive lambda, and both landed on target. Two independent
quantities from one calibrated input.

### Still open

- Identical noise for all pids gives human-scale variability but not human
  individual DIFFERENCES: NoisyRL_lambda's prefix-variability distribution is a
  narrow spike (~0.04-0.05) vs the human's broad 0.2-0.5, split-half reliability is
  weaker (numbers r=0.49** vs 0.81****), and the numbers lambda correlation actually
  DROPS 0.644 -> 0.524 even as the level matches. Per-participant sigma_resp fixed
  at each pid's measured value is the obvious next step; MODEL_PARAMS supports a
  `fixed` dict but not per-pid values.
- The observation-0 variability profile cannot be matched by this model family
  (human 0.0093 -> 0.0515 step; model always highest at observation 0). Probably
  task structure rather than a noise process.
- A distributional (NLL) loss remains the only route that would actually FIT the
  noise rather than calibrate it. Tabled: `compute_sim_db_loss` keys cells on the
  FULL sequence tuple, which suits carrabin's repeated sequence pool but not
  soltani's mostly-unique per-participant sequences, and `build_sim_db`
  hand-duplicates each model's implementation (seeding by simulation index rather
  than by trial), so adding a third pair doubles the drift risk.

## RNN as a conditional-mean estimator: works for carrabin, fails for soltani (this session)

Goal was a distributional fit metric to complement RMSE, which cannot see variance
(demonstrated repeatedly this session: NoisyRL_lambda's noise parameters collapse to
their bounds under RMSE). The proposed route -- from earlier discussion -- was to
fit an RNN per participant as a "best possible" conditional-mean estimator, then use
(a) the residual SD as a response-variability metric covering all 15 observations
rather than only the shared prefix, and (b) the RNN prediction as a DENOISED target
for a distributional loss.

Conclusion: the premise is DATASET-DEPENDENT. It holds for carrabin and fails for
soltani, for a structural reason rather than a tuning one. Use the RNN for
carrabin; for soltani keep qid-grouped response std.

### The decisive test, and what it showed

The test is whether the RNN beats simple models on HELD-OUT data. If a 2-parameter
model out-predicts it, it is not a best-possible conditional mean, and its residual
is contaminated with its own prediction error rather than being response noise.

| | trials/pid | obs/trial | sequences | held-out RMSE |
|---|---|---|---|---|
| carrabin | 200 | 5 | repeating pool | RNN 0.1225 BEATS NoisyCounting 0.1324 (15/21 pids), and every other model 21/21 |
| soltani numbers | 32 | 15 | mostly unique | RNN 0.0626 LOSES to RL_lambda 0.0526 (0/4 pids) and to the parameter-free running mean 0.0545 |

Carrabin gives the GRU 6x more trials AND repeating sequences, so a held-out trial
has often been seen in training -- interpolation. Soltani's 32 sequences are unique,
so a held-out trial is genuinely novel -- extrapolation, which is the regime where a
101-parameter model loses to a 2-parameter delta rule.

Worth noting the carrabin numbers came from files already on disk
(data/runs/carrabin/RNN_carrabin_performance.pkl), whose saved `loss` IS
`cv_rmse` -- genuinely out-of-fold at k=5, n_hidden=4.

### Two of my hypotheses were wrong

Asked why the same setup succeeded on carrabin and failed on soltani, I proposed
(a) the old carrabin result was in-sample and (b) it used a different n_hidden.
BOTH wrong: the saved loss is held-out cv_rmse, and n_hidden was 4 in both cases.
The difference is training data and sequence structure, which neither hypothesis
touched.

### n_hidden matters but does not rescue it

Sweep on soltani_numbers, 4 pids, k=8 (28 of 32 trials per fit), held-out RMSE:

| n_hidden | RMSE | vs RL_lambda |
|---|---|---|
| 1 | 0.1751 | +233% |
| 2 | 0.0911 | +73% |
| **3** | **0.0626** | **+19%** |
| 4 | 0.0701 | +33% |
| 5 | 0.0722 | +37% |

A clean U-shape with an interior optimum -- 1 underfits, 4-5 overfit -- so the
default of 4 WAS mistuned, by ~19 percentage points. But even at n_hidden=3
RL_lambda wins on 4/4 pids and the zero-parameter running mean still wins too.

### Consequences for the two applications

- **sigma_RNN cannot replace prefix variability for soltani.** At the best setting
  it is 0.0626 against the qid-repeat estimate of ~0.055 -- only 14% inflated,
  which is tempting -- but the inflation is the GRU's own prediction error, and
  RL_lambda's residual on the same rows would give a lower estimate still. At the
  original settings (k=5, n_hidden=4) sigma_RNN was 0.18, more than 3x the qid
  estimate.
- **The RNN prediction cannot be a denoised target for a distributional loss on
  soltani**, because it is LESS accurate than the models being evaluated. Scoring
  NoisyRL_lambda against a target that RL_lambda predicts better would be perverse.

An NLL needs no conditional-mean estimator anyway: score the observed y under the
model's simulated predictive distribution. That penalises mean AND variance
mismatch together as a proper scoring rule -- the quadratic term punishes a wrong
mean and understated variance, log(sigma) punishes overstated variance. The RNN's
remaining possible roles were (i) a denoised target for the mean term and (ii) a
noise ceiling for normalising NLL across participants; (i) is now ruled out for
soltani, and (ii) would inherit the same contamination.

Note the NLL is UNDEFINED for deterministic models (sigma_m = 0 gives infinite
NLL; compute_sim_db_loss clamps at 1e-3, silently turning it into scaled squared
error with an arbitrary scale). So it applies to stochastic models only and
complements RMSE rather than replacing it.

### Two bugs fixed, both of which made models/RNN.py unusable on soltani

- `build_trial_tensors` derived observations-per-trial from `max(observation)`,
  silently assuming 1-INDEXED data. On soltani (0..14) that gave n_obs=14 while
  every trial has 15 rows, so the `len(td) != n_obs` guard dropped EVERY trial and
  the function failed on an empty stack. Now uses the modal row count
  (index-agnostic) and raises with a clear message if nothing matches. Same class
  of 0-indexing bug as the ones fixed earlier in the activity keying and the lambda
  estimator.
- `generate_rnn_responses` emitted `observation = oi + 1` over `range(n_obs)`,
  hardcoding 1-indexing: on soltani it mislabelled every row and dropped
  observation 0. Now uses each trial's own observation labels.

Also added `cross_validated_predictions()`, stitching OUT-OF-FOLD predictions across
folds so every observation gets a prediction from a model that did not see it, with
nothing discarded. Motivation: in-sample residuals are systematically too small
because the fit absorbs noise, and on soltani that is severe -- in-sample sigma
~0.046-0.056 against 0.18 out-of-fold at k=5. In-sample sigma happens to MATCH the
qid estimate (~0.055), but only because the GRU has memorised each trial, so the
agreement is coincidental rather than validating. The function still uses the
held-out fold for early stopping, making its predictions mildly optimistic; a
nested split was judged not worth a third partition of 32 trials.

### Caveats on the soltani conclusion

4 pids, numbers only, one seed per setting. 0.0626 vs 0.0526 on n=4 is not a strong
separation and a firm decision would want ~10 pids. The ordering was consistent
across all four pids at every n_hidden, which is why the thread was stopped here
rather than powered up.

## NLL fitting infrastructure; response noise split into two mechanisms (this session)

Continuation of the NoisyRL_lambda thread. Goal: a loss function that can
actually IDENTIFY a noise parameter (RMSE cannot -- it collapses every noise
parameter to its floor, since squared error is minimised by the conditional
mean regardless of variance), and a design that lets the noise MECHANISM be
compared rather than just its presence.

### The NLL loss

`fitting.losses.compute_nll` / `nll_from_ensemble`: Gaussian NLL of the single
observed human response at each (pid, trial, observation) under the model's
simulated predictive distribution (mean + SD from n_sims independent draws). A
proper scoring rule -- the quadratic term punishes a wrong mean AND an
understated variance, log(sigma) punishes an overstated one -- so unlike RMSE it
has a genuine interior optimum. Verified directly and unconstrained (floor
0.001, i.e. effectively no floor): NLL fell from 389 at sigma_resp=0.001 to
-2.46 at the optimum (~0.04-0.05) and rose again beyond it. A real U-shape.

`models.math_models.simulate_ensemble(params, n_sims, return_index=False)`: for
a genuinely stochastic model, n_sims realisations without re-simulating from
scratch per (trial, observation) the way run() does (which would cost ~48k
pandas queries per Optuna trial at n_sims=100 on soltani -- not viable). Each
(trial, sim) is ONE forward pass; seeded `_trial_seed(sim, trial)` so
`simulate_ensemble(params, n)[i] == run({**params, "seed": i}).response`
exactly (verified to floating point, <=3.3e-16, across all four datasets).

Wired into `fitting.fit` via `--loss {rmse,nll}` and `--n_sims`. n_sims=100
verified stable (5 reseeded reps of a sigma sweep all picked the identical
argmin; n_sims=25 already agreed) at ~0.45s/eval, ~2.3 min per 300-trial fit.
NLL output files get a `_nll` suffix before `{pid}` so they can never silently
overwrite an RMSE fit of the same model_type -- the two loss scales differ (NLL
can be negative) and a silent overwrite would be a correctness hazard, not a
naming inconvenience.

### The noise mechanism was split into two, at the user's suggestion

Original NoisyRL_lambda had both sigma_state (compounds into the estimate) and
sigma_resp (i.i.d. on the report). Splitting was proposed to avoid depending on
a prior RMSE fit for the deterministic-model comparison, and to isolate the
noise MECHANISM (compounding vs i.i.d.) at EQUAL parameter count rather than one
model simply having more parameters than another:

  NoisyRL_lambda            RL_lambda + sigma_state ONLY (compounding)
  <model>_resp_noise        {Mean,LeakyIntegrator,PrimacyRecency,RL_lambda}
                            + sigma_resp ONLY (i.i.d.), via a NEW generic
                            add_noise() wrapper

`add_noise(params, n_sims, sigma_resp, return_index=False)`: calls the base
model's run() ONCE for its deterministic mean trajectory, then draws n_sims
i.i.d. Gaussian perturbations on top, clipped to [-1,1]. No per-observation loop
needed (i.i.d. noise has no sequential structure to replay), so it is cheaper
than simulate_ensemble's state-noise loop and is entirely generic -- it never
touches per-model branches in math_models.py, so it wraps any of the four
deterministic models without new code per model. Accepts either the bare base
name ("RL_lambda") or the fitting-time suffixed name ("RL_lambda_resp_noise")
identically (base_model_of() strips the suffix), since fit.py's objective
passes the suffixed name.

Verified: reduces EXACTLY to run()'s output at sigma_resp=0 (0.00e+00, all
four datasets x four base models); empirical mean/SD track the requested
values away from the +-1 clipping boundary; bare and suffixed names produce
identical output.

Registered for ALL FOUR DATASETS (carrabin, yoo, soltani_numbers,
soltani_colors) in MODEL_PARAMS, not soltani-only as NoisyRL_lambda originally
was. Floor 0.001 on every noise parameter -- TECHNICAL only, not calibrated,
since NLL was shown to find its own interior optimum unconstrained.

### Extending to carrabin/yoo surfaced two real bugs, both invisible to
### py_compile and to exercising individual branches in isolation

1. **Triplicated code, dormant until now.** Adding NoisyRL_lambda earlier this
   session used an unguarded string-replace on the anchor
   `if model_type == "RL_lambda":`, which appears once in EACH of
   `_run_carrabin`, `_run_yoo`, `_run_soltani_common` (RL_lambda is valid for
   all three). With no occurrence limit, the replace silently duplicated the
   entire branch into all three instead of the intended one. Harmless while
   unreachable (carrabin/yoo never had the model registered); activated the
   moment this request registered it there.

2. **Deduplicating the triplication introduced a WORSE bug.** Refactoring the
   three copies into one shared helper by inserting a top-level `def` string at
   the text position of the FIRST occurrence -- which sat inside
   `_run_carrabin`'s indented body -- caused the bare `def` to dedent out of
   that function. Everything textually after it, including `_run_carrabin`'s
   remaining RL_lambda/LeakyIntegrator/PrimacyRecency branches, became
   unreachable dead code swallowed into the new helper's body.
   `_run_carrabin(..., model_type="RL_lambda", ...)` would have returned None.
   Caught only by testing every carrabin branch directly, not by py_compile.
   Fixed by locating the exact corrupted text and reconstructing both pieces
   (the standalone helper, and _run_carrabin's restored tail) explicitly.

3. **simulate_ensemble labelled columns wrong for 1-indexed datasets.** It used
   a synthetic `range(n_obs)` for the observation index rather than the
   dataset's real values. Coincidentally correct for soltani (0-indexed,
   0..14) but WRONG for carrabin (1-indexed, 1..5): the carrabin Laplace-
   shrinkage formula (t = observation + 1) then received the wrong t, biasing
   the ensemble vs run() by up to 0.167 -- caught only by the direct
   equivalence check, not by any per-branch test.

All three fixed; `scripts/verify_ensemble_invariant.py` (see below) passes
clean on all combinations after the fixes.

### A fourth bug, in the FIX for the model-params update

Adding the split MODEL_PARAMS entries for all four datasets used a
find-and-replace loop that recomputed `s.find('"NoisyRL_lambda": {')` from
scratch after each insertion. The replacement text itself CONTAINS that exact
substring (it is inserting a dict literal whose key is that name), so the loop
kept re-finding the text it had just inserted and never advanced -- a genuine
infinite loop, not a slow computation (confirmed via `timeout`, exit code
124). Fixed by locating all four match positions on the ORIGINAL string before
any insertion, then replacing right-to-left so earlier offsets stay valid.

### scripts/verify_ensemble_invariant.py (new)

Not a pytest suite -- this project has none, and an earlier docstring falsely
claimed one existed in `tests/`; corrected. Run manually after touching
simulate_ensemble, add_noise, any _run_* dispatcher, or
_validate_model_dataset's allowlists, and before trusting a --loss nll fit on
a dataset/model combination not previously checked. Two check families:

- simulate_ensemble vs run(seed=i), for _STOCHASTIC_ENSEMBLE_MODELS.
- add_noise vs run(): sigma=0 exact equality; empirical mean/SD near the
  requested values AWAY FROM THE +-1 BOUNDARY (clip(mu + noise, -1, 1) is
  CORRECTLY biased near the boundary -- confirmed directly on soltani_colors'
  Mean model, which legitimately outputs exactly +-1 on 15.6% of rows: mean
  gap 0.0035 away from the boundary vs 0.0257 on boundary rows, and the
  boundary-row gap does NOT shrink with more sims, confirming bias rather than
  Monte Carlo noise -- so the check correctly excludes those rows rather than
  loosening its tolerance globally); and bare-name vs suffixed-name identity.

All checks currently pass (exit 0) locally across all dataset x model
combinations. NOT yet run on the cluster -- required before trusting any
cluster --loss nll fit, given how many of the above were invisible until
directly tested.

### Not yet done

- No jobs submitted to the cluster.
- The actual NoisyRL_lambda vs RL_lambda_resp_noise NLL comparison -- the
  scientific payoff of this whole thread -- has not been run at real
  n_trials/n_sims on any dataset.
- Response variability (qid-based) as the parallel individual-differences
  metric for carrabin/numbers/colors (yoo has no qid repeats) is planned but
  not implemented as a figure or comparison.

---

## Persistent pid registry; pull_soltani_data.py rename; pilot-4 contamination found and removed (this session)

### The pid-instability bug

`build_model_inputs.py`'s `build_from_df()` used to compute the integer
`pid` fresh on every call:

    all_pids = sorted(df["prolific_pid"].unique())
    pid_map = {p: i + 1 for i, p in enumerate(all_pids)}

That recomputes the mapping from scratch, by alphabetically sorting
whichever `prolific_pid`s are in THAT call's data, every time. The moment
the participant pool changes size, inserting new `prolific_pid` strings
into that sort generally shifts the alphabetical rank of most of the
EXISTING participants too -- not just appends new ones at the end.
Confirmed as a live, not hypothetical, bug this session: `data/
soltani_numbers.pkl` grew from 35 to 45 pids between two builds, and the
model-fit response files (still only pids 1-35, from the older build)
could no longer be safely joined on `pid` against the current human data
-- pid=5 in one file and pid=5 in the other were very likely different
real people. This is exactly what made a new `lambda_model_correlation`
figure's colors/numbers panels come back with near-zero, non-significant
correlations despite `make_response_change`'s own median curves showing
models tracking human decay closely: a MEDIAN is a population statistic,
invariant to which label is attached to which value, so it stayed correct
under the mislabeling; a per-pid CORRELATION depends entirely on the
labels being right, so it was destroyed by it.

**Fix: `utils/pid_registry.py`**, a persistent, append-only
`prolific_pid -> pid` mapping stored at `data/pid_registry.json`. Loads
the existing registry, keeps every known `prolific_pid`'s integer
unchanged, assigns new ones only to genuinely new `prolific_pid`s (sorted
for determinism, continuing from `max(existing) + 1`), then saves the
updated registry back before returning. `build_from_df()` now calls
`get_or_assign_pids()` instead of the from-scratch enumeration. Verified
with a synthetic test that deliberately inserted new IDs alphabetically
BETWEEN existing ones, confirming they don't get reshuffled. Side benefit
noted but not separately tested for: a filtered and an unfiltered build
now assign the SAME pid to the same person, which was also false before
(the old mapping depended on who else was in that specific call's batch).

The registry file contains REAL Prolific participant IDs, unlike every
canonical `data/*.pkl` file, which only ever gets the anonymized integer
-- it must never be committed, and specifically must never go through
GitHub even now that the canonical soltani files do (see below). It's
gitignored explicitly (on top of already being covered by the wholesale
`data/` rule), and syncing it to another machine (e.g. the cluster) is a
manual, non-git responsibility -- copy the one file directly.

### `build_task_backend_inputs.py` renamed to `pull_soltani_data.py`

The old name sounded like it BUILDS INPUTS FOR task_backend (i.e.
configures the task); it does the opposite -- pulls results OUT of
task_backend/Supabase. Renamed to say what it does, and its own module
docstring now explicitly walks through the pipeline's steps 1-3 (pull
from Supabase -> filter to an explicit/derived pid list -> resolve via
the persistent registry), with step 4 (rescale/anonymize/save) staying in
`build_model_inputs.py`'s `build_from_df`, called into rather than
duplicated. Every cross-reference updated (`build_model_inputs.py`,
`figure_soltani_performance.py`, `figure_soltani_variability.py`,
`CLAUDE.md`, `README.md`, `.gitignore`) -- a plain rename, no logic
change in this step.

### `--complete_pairs` was pulling in a stale pilot round

Supabase's `events` table is append-only, so an OLDER pilot round's
participants are STILL present with a perfectly genuine `'finished'` row
long after that round ended. `--complete_pairs`, as originally written,
had no date cutoff and no check that a participant's session used the
CURRENT generative parameters -- it just intersected "everyone finished in
both tasks," for all time. Confirmed directly: an actual pull returned 51
pids, 5 of which had `true_std=15` (pilot 4's fixed numbers-task std) sitting
alongside 46 with the current `true_std=10` -- exactly the failure mode the
module's own docstring already warned about for the explicit-pid-list path,
but never actually guarded against for `--complete_pairs`. Those 5 pids
(`670bd903349d5d24bc92dcb0`, `69163607e65df2b5dbe294fa`,
`697b8bbd3f4ddf0f4e102d42`, `69af34e771ce9d065c0d9d80`,
`6a11c6a18ea3cad18626f8b4`) are literally the same ones the module's own
usage example under `--pilot pilot4` already named -- independent
confirmation this was genuinely pilot 4, not some other anomaly.

Checked whether colors has an equivalent risk: it does not. Every pid's
`true_p` range (including all 5 pilot-4 pids) is identical,
`[0.1333, 0.8667]`, regardless of round -- colors' generative parameters
have apparently never changed, unlike numbers' (`NUMBERS_STD_FIXED`
history: 10 -> 15 -> 10, per `task_backend/generate_sequences.py`'s own
comment).

**Fixed at two levels, deliberately not just one:**

1. Code: `--complete_pairs` now excludes any pid whose numbers session used
   a `true_std` other than `CURRENT_NUMBERS_STD_FIXED` (10, duplicated from
   `generate_sequences.py`'s own constant rather than imported --
   `task_backend/` is a JS/Vite app with that script as its one standalone
   Python utility, not a package this analysis pipeline otherwise reaches
   into; must be kept in sync by hand if that constant ever changes again).
   Intentionally `--complete_pairs`-only, not applied to the explicit
   `--numbers_pids`/`--colors_pids` path, since that path is how a SPECIFIC
   pilot round gets rebuilt on purpose. This is a safety net for the NEXT
   time a generative parameter changes and an old round lingers in the same
   table, not what actually removed the contamination this time (see below).
2. Data: the 5 pilot-4 prolific_pids' rows were deleted directly from
   Supabase's `events` table (both tasks, all phases) after a preview
   `SELECT` confirmed the exact rows affected -- a full, complete session
   each (consent -> tutorial -> ~480 trial rows -> finished), nothing
   unexpected caught. Safe to do irreversibly because pilot 4's data was
   already separately archived (`data/task_results_pilot4.pkl`,
   `data/soltani_{numbers,colors}_pilot4.pkl`) before this session, so
   nothing unique was lost. Re-running `--complete_pairs` afterward found
   nothing to exclude (Supabase itself is clean now), producing the final
   46-pid canonical files directly.

Final state: `data/soltani_numbers.pkl`/`soltani_colors.pkl`, 46 pids
(`complete_pairs` + `non_integrator` exclusion), pid range 1-51 with
`{11, 14, 19, 34, 48}` now absent -- confirmed to be exactly the 5 removed
pilot-4 pids, and confirmed every one of the other 46 kept their pre-
existing registry pid (46 already known, 0 newly assigned).

### Canonical soltani files now tracked in git

`data/soltani_numbers.pkl`/`soltani_colors.pkl` were gitignored; now
tracked, matching `carrabin.pkl`/`yoo.pkl`'s own existing treatment --
GitHub is now the sync channel for these two files specifically between
this machine and the cluster. `data/pid_registry.json` is the deliberate
exception, per above.

**A real git gotcha surfaced doing this, worth remembering.** Per
`gitignore(5)`: "It is not possible to re-include a file if a parent
directory of that file is excluded." `data/` (trailing slash) excludes the
whole directory as a unit, so git never even looks inside it to evaluate a
later `!data/soltani_numbers.pkl` override -- for any file not ALREADY
tracked, that negation line is a silent no-op. `carrabin.pkl`/`yoo.pkl`'s
own identical-looking negation lines were never actually doing anything
either -- those files were already in the git index from before this
wholesale rule existed, and an already-tracked file is immune to
`.gitignore` regardless of any rule. Confirmed directly: `git check-ignore
-v` returns clean (exit 1, untracked-and-not-ignored) for `carrabin.pkl`
but reports the wholesale `data/` rule as still matching for the brand-new
`soltani_numbers.pkl`, and a plain `git add` refuses it with "ignored by
one of your .gitignore files." Fixed with a one-time `git add -f`; after
that, exactly like `carrabin.pkl`, both files are permanently tracked and
immune to the `data/` rule for every future `git add`, no `-f` needed
again.

---

## Soltani math-model RMSE/NLL fits; presentation deck finished; NEF refit planned next (this session)

### Fresh math-model fits against the corrected 46-pid data

With the pid-registry and pilot-4 fixes above in place, refit Mean/
LeakyIntegrator/PrimacyRecency/RL_lambda for soltani_numbers/soltani_colors
under RMSE (`data/runs/rmse/`, `--n_trials 300 --k 5`), and
Mean_resp_noise/LeakyIntegrator_resp_noise/PrimacyRecency_resp_noise/
NoisyRL_lambda under NLL (`data/runs/nll/`, same n_trials/k, `--loss nll`)
-- exact commands now in `CLAUDE.md`'s own "soltani math-model fits"
section. These are the FIRST fits against the contamination-free 46-pid
canonical data; every earlier soltani fit in `data/runs/soltani/` was
made against progressively stale data (35, then 45, pids) and is no
longer read by anything.

### carrabin's NEF fit found to be incomplete, mid-session

While reviewing `presentations/make_figures.py`'s own "Model Performance"
figure, the balls (carrabin) panel had silently lost its significance
bars relative to an earlier render the person had on hand. Traced
directly (not guessed): `NEF_carrabin_performance.pkl` (the plain
RMSE-fitted file that figure reads) has only 16 of 21 pids, though
NEF_carrabin_responses_mle.pkl (the MLE-fitted variant other panels
read) has all 21 -- confirmed via the file's own pid column, and via
direct comparison against an uploaded copy of the earlier SVG (its own
embedded `<dc:date>` metadata dates it to earlier the same day, before
the current carrabin fit files' own mtime). Something regenerated
carrabin's NEF fit, incompletely, sometime that day, before this
session started -- not caused by anything done in this session, since
the carrabin path resolvers were never touched.

Given a genuine refit was already the acknowledged next step (see
below), this was treated as a KNOWN, temporary gap rather than
re-running the old fit: `presentations/make_figures.py`'s "Model
Performance" figure now shows `RL_lambda` in balls' "our model" slot
instead of NEF (RL_lambda has a complete, real 21-pid carrabin fit on
disk) -- documented explicitly in that file's own `TASK_PANELS` comment
as a stand-in pending the NEF refit, not a permanent substitution the
way RL_lambda already stands in for colors/numbers (where NEF has never
been fit at all).

### Presentation deck: response-noise (sigma) figures added, alongside real bugs found and fixed

The Dartmouth talk gained a full sigma (response-noise) track mirroring
the existing lambda track, plus corrections to figures already in the
deck:

- **`sigma_sanity_human.svg`** ("Consistent Across Trials & Tasks (sigma)")
  and **`sigma_model_correlation.svg`** ("Models Capture sigma
  Differences") -- split-half reliability and human-vs-model correlation
  for response noise, mirroring the existing lambda_sanity_human/
  lambda_model_correlation figures.
- **A real bug found and fixed while building the split-half sigma
  figure**: an early version applied balls' own median-split,
  min_trials=3 convention (figure_carrabin_variability.py's own
  `half_split_std`) uniformly to colors/numbers too, on the wrong
  assumption that lambda's own uniform odd/even convention generalized
  directly to sigma. It does not -- figure_soltani_variability.py's own
  split-half convention for THIS metric uses odd/even splitting with NO
  min_trials threshold at all. Stacked together, the wrong convention
  destroyed the colors sample: 46 pids (full data) down to just 14 with
  both halves defined, several of them degenerate exact zeros, versus
  45/46 under the real script's own convention. Confirmed against both
  real scripts' actual code before fixing, not guessed. Fixed by
  dispatching per task on each real script's own convention rather than
  applying one uniformly.
- **`sigma_model_correlation.svg` was extended past its own first version**:
  initially built with only the ONE genuinely-stochastic model per task
  (NEF/NoisyRL_lambda), on the reasoning that every OTHER model's sigma
  is exactly zero under RMSE fitting. That reasoning holds for the
  bare-name RMSE fits but not for Mean/LeakyIntegrator/PrimacyRecency's
  own "_resp_noise" NLL fits, which add a genuine per-pid sigma_resp term
  -- confirmed directly (e.g. numbers' Mean_resp_noise ranges 0.028-0.459
  across 46 pids, not pinned at a floor) before adding all three. Result:
  NoisyRL_lambda/NEF track human sigma more consistently across all three
  tasks (r=0.67-0.95, all p<0.001) than the resp_noise models do (some
  non-significant, e.g. balls' Mean r=0.28 ns) -- a genuinely interesting,
  clean finding for the talk.
- **Shared axis ranges added to both model-correlation figures**, per
  instruction: lambda_model_correlation already had a shared [0,1.5]
  x/y range across its 3 panels; sigma_model_correlation's own per-panel
  autoscaling was replaced with one shared [0,0.6] range. Both figures'
  y-axis label is now drawn on the leftmost panel only.
- **`variance_autocorr_human.svg`/`variance_autocorr_models.svg` refactored**
  to share one data-loader and one panel-drawing function, so the two
  images are now identical apart from the added model data and legend
  (per instruction) -- including a synchronized y-axis range across the
  two separately-saved SVGs (a throwaway probe pass reconstructs the
  same range the models figure would produce, mirroring
  make_response_change's own established two-pass pattern, since two
  separate top-level functions have no shared Axes object to sharey=True
  together). Model roster dropped RL_lambda (per instruction, matching
  model_performance_nll's own 4-model roster) in favor of
  Mean_resp_noise/LeakyIntegrator_resp_noise/PrimacyRecency_resp_noise +
  NoisyRL_lambda (colored as RL_lambda, labeled "RL_lambda*" in the
  legend, full model names throughout).
- **Conclusions slide fixed to show all bullets at once** rather than
  stepping in one at a time -- wrapped each column's list in a
  `.nonincremental` div, overriding the deck's global `incremental: true`
  default for just that one slide.

### Presentation: DONE for now. Next planned step: NEF refit

The deck is considered complete as of this session. The person's own
stated next step, to be picked up in a NEW chat (a prompt for that chat
was written directly into the person's own message/response at the end
of this session): refit NEF for soltani_numbers/soltani_colors under
BOTH losses, matching the math-model fits above --

- **RMSE pass**: a highly accurate network (large `n_neurons`) purely for
  behavioral fidelity -- faithfully reproducing RL_lambda's own
  input-output behavior, not yet concerned with noise sources.
- **NLL pass**: `n_neurons` itself as an OPTUNA-FIT parameter (not fixed),
  since n_neurons is the architectural lever that controls how much
  genuine spiking noise (response noise, PE noise, etc.) the network
  produces -- letting Optuna choose it directly targets fitting the
  amount of variance the noise sources should produce, rather than
  guessing a fixed value and hoping it lands in a reasonable regime.

Once both fits land, the plan is to (a) add NEF into every existing
presentation figure that currently substitutes RL_lambda/NoisyRL_lambda
in its place (model_performance, response_change, lambda/sigma
correlation, variance_autocorr), and (b) revisit the neural-comparison
figures (figure_carrabin_neural.py/figure_yoo_neural.py's own N-group
panels), not touched in some time, extending them to soltani where that
makes sense.


---

## Shared-database/cross-pid Optuna sim_db: revisited and re-confirmed tabled; NEF multi-seed activity files + NLL ensemble branch built (this session)

### Shared simulation database: why it stays tabled

Revisited the `fit_mle.py`/`build_sim_db.py` shared-database architecture
(cross-pid Optuna cross-reporting, see this file's earlier session on that
topic) as a candidate for NEF's NLL fitting. Re-confirmed it does not transfer:
the cross-pid caching benefit it exists for depends on SEQUENCES REPEATING
across pids/trials, true for carrabin's small repeating pool but false for
yoo/soltani's mostly-unique per-participant sequences -- exactly the earlier
"Tabled" note above, re-derived from first principles before remembering it was
already written down. Still live for NoisyCounting/carrabin only; not adopted
for NEF or for the production RMSE/NLL pipeline (`fitting/fit.py`), which has
no shared database and never will for this reason.

### n_sims calibration via cheap math-model proxies (`scripts/calibrate_nll_nsims.py`)

Built to answer "how many sims does NEF's NLL ensemble need" WITHOUT paying
NEF's own Nengo cost -- runs on `NoisyRL_lambda`/`RL_lambda_resp_noise` only
(seconds, not hours), against REAL human data (no synthetic ground truth,
matching `fitting/fit.py`'s own established validation convention: check
whether independent Monte Carlo reps AGREE, not whether they recover a known
truth). Two views: NLL-value convergence at a fixed (alpha_0, lambda_, sigma)
point (mean +/- spread across reps vs a large-n_sims reference), and
argmin-sigma recovery stability (does repeated fitting land on the same
answer).

**Real finding, not assumed**: the noise MECHANISM changes how many sims are
needed. On soltani_numbers pid 13, `RL_lambda_resp_noise` (i.i.d. response
noise) gave a perfectly stable argmin from n_sims=10; `NoisyRL_lambda`
(compounding state noise) needed n_sims=40 to stabilise, having genuinely
wobbled at n_sims=10-20. NEF's recurrent value ensemble means a sim's
tuning-curve idiosyncrasies persist and compound through a trial -- structurally
closer to the state-noise mechanism -- so `RL_lambda_resp_noise`'s cheaper
number would be a falsely optimistic stand-in for NEF; `NoisyRL_lambda`'s is
the honest one.

**Also real: required n_sims is NOT uniform across pids.** Ran 4 soltani_numbers
pids (10, 13, 16, 22) through `NoisyRL_lambda`. 3 of 4 were stable from
n_sims=10 already; pid 13 -- the one that also needed the LARGEST recovered
sigma (0.25 vs 0.10 for the other three) -- needed n_sims=40 to stabilise and
n_sims=160-320 to fully settle. Not a coincidence: a noisier fit needs more
samples to pin down its own variance precisely. Settled on **n_sims=50** as a
working default for NEF -- covers the typical case with margin, short of the
worst case's full 320 -- explicitly a ballpark to raise later once a direct NEF
measurement exists, not a validated number the way NoisyRL_lambda's own
n_sims=100 (measured against REAL NEF-adjacent data, in `fitting/fit.py`'s own
docstring) is.

One early pitfall worth recording: the first `argmin_stability` run pinned at
the EDGE of the tested sigma grid (0.02-0.1) with zero spread at every n_sims --
not genuine stability, just the search hitting a wall because the fixed
(alpha_0, lambda_) point (RL_lambda's own RMSE optimum, held fixed rather than
jointly refit under NLL) needed more residual variance explained than a narrow
grid could supply. Widening the grid iteratively (to 0.05-0.8, then a properly
resolved 0.1-0.4) is what produced the genuine interior optimum reported above.
A boundary-pinned argmin with zero cross-rep spread is a red flag for "grid too
narrow," not evidence of stability.

### Multi-seed counting-activity files: the actual disk-cost realization

Initial framing (activity files could be reused across trials for different
sims, since the Gram matrix depends only on n_neurons_counting/seed, not on
pid or trial) was WRONG, caught by the person before any code was written: NEF's
network (error, value, AND the counting subnetwork) is seeded once per
simulated trial, so reusing a seed across DIFFERENT trials would give both
trials the literal same neural substrate -- correlating, not independently
sampling, whatever idiosyncratic bias that seed's tuning curves carry. A
genuine ensemble needs `n_trials * n_sims` distinct seeds, not `n_sims`.

Measured (not estimated) the resulting disk-cost model directly against every
activity file already on disk before trusting it: size is driven almost
entirely by `n_neurons_counting^2 * n_trials_precomputed`, and NOT by
`n_neurons` at all (the file only stores the counting subnetwork's `memory`
ensemble). Consequence: carrabin (200 trial-seeds) is ~5-7x more expensive per
unit of `n_neurons_counting` than yoo (30) or soltani (40) at the same size --
an `n_neurons_counting=2000` file is ~1-1.3GB for yoo/soltani but would be
~6.4GB for carrabin, BEFORE any `n_sims>1` multiplication (which scales
linearly on top of that). See CLAUDE.md's "NEF architecture" section for the
full cost table and the resulting recommendation to size carrabin's own NLL
activity files smaller than yoo/soltani's.

### Architecture built: `sim`-aware seeding, resumable precompute, NEF's own ensemble

Three pieces, each backward compatible (verified directly, not just reasoned
about):

1. **`counting_integrator.activity_key_for_trial(dataset, trial, sim=1)`** --
   `sim` defaults to 1 and reproduces the exact original key/seed for every
   existing caller. For `sim>1`, offsets by a full dataset-sized block:
   `(sim-1)*_DATASET_N_TRIALS[dataset] + base`. Verified directly: sim=1 output
   matches the pre-change function exactly for all 4 datasets; 5 sims x every
   real trial for all 4 datasets produces zero key collisions.

2. **`counting_integrator.precompute_activities(..., n_sims=1)`** -- generates
   `n_trials*n_sims` keys instead of just `n_trials`. RESUMABLE: loads any
   existing file first and only simulates missing keys, so growing an
   `n_sims=1` file up to `n_sims=50` does not re-pay for the keys it already
   has. Verified directly (Nengo calls stubbed, real function exercised
   end-to-end): fresh generation produces the right key set; a second call
   with a larger `n_sims` adds exactly the new keys and leaves the old ones'
   values byte-identical; a third call with nothing new to do is a correct
   no-op.

3. **`models.NEF.simulate_ensemble(params, n_sims, return_index=False)`** --
   the NEF analogue of `math_models.simulate_ensemble`: same `(n_sims, n_rows)`
   shape, same row order, so `fitting/fit.py`'s NLL dispatch (`_cross_validate_nll`)
   works identically regardless of which one produced the ensemble. Applies
   `run()`'s own post-processing (`nef_response_to_model_scale`, then
   `apply_binary_transform`) ONCE on the full stacked frame rather than
   re-deriving carrabin's Laplace-shrinkage formula by hand (the risk
   `math_models.simulate_ensemble`'s own docstring flags). Verified directly
   (Nengo call stubbed): ensemble shape and row-ordering are correct; different
   sims produce different values for the same row; the transform applied
   inside `simulate_ensemble` matches `apply_binary_transform` called by hand
   on the same raw output, exactly.

   The required-activity-file check (`_require_activity_map`) was factored out
   of `run()` into a shared helper used by both `run()` (n_sims=1) and
   `simulate_ensemble()` (n_sims=n_sims), so the two paths' requirements can't
   drift apart the way a second hand-written copy eventually would.

`fitting/fit.py` now accepts `NEF` for `--loss nll`: validation extended
(`is_nef_model = model_type == "NEF"`), and the objective's ensemble dispatch
gets a third branch (`model_type == "NEF" -> NEF.simulate_ensemble`, ahead of
the existing `_STOCHASTIC_ENSEMBLE_MODELS`/`_resp_noise` branches). Verified
directly: `fit("carrabin", "NEF", ..., loss_fn="nll")`'s validation now passes
where it used to be rejected; `fit("carrabin", "Mean", ..., loss_fn="nll")`
still correctly raises, with an updated message listing NEF among the valid
alternatives.

`models.NEF.NEF_DEFAULT_N_SIMS = 50` added as a documented, discoverable
constant -- deliberately NOT wired into `fitting/fit.py`'s own shared
`--n_sims` CLI default (100), which stays at its own validated value for
NoisyRL_lambda/`_resp_noise`; pass `--n_sims 50` explicitly for NEF fits.

**Known gap, stated explicitly rather than left implicit**:
`scripts/verify_ensemble_invariant.py` (the existing real-Nengo-free equivalence
check for math models' ensemble paths) does not cover `NEF.simulate_ensemble` --
it only imports `models.math_models`. The stubbed-Nengo checks done this
session are a real but lesser guarantee (structure/transform correctness, not a
true Nengo-level comparison against `run()` at matching seeds). A proper NEF
equivalent -- comparing `simulate_ensemble`'s per-sim output against `run()`
called once per seed, with REAL Nengo -- has not been built yet and should be
before the first real NEF NLL fit is trusted, the same way the math-model side
required exactly this check to catch two real bugs when NoisyRL_lambda was
extended to new datasets.


---

## NEF RMSE fits: n_neurons/n_neurons_counting bumped for all 4 datasets (yoo/soltani to 500/2000, carrabin to 500/500); weekend submit planned (this session)

### The decision

Settled on large `n_neurons=500` for all 4 datasets, explicitly erring on the
side of too many neurons rather than a precisely-justified number. But NOT a
uniform `n_neurons_counting` across all 4: yoo/soltani_numbers/soltani_colors
went to 2000, while carrabin went to 500 (not 2000) -- because activity-file
size scales with `n_neurons_counting^2 * trial-seeds precomputed`, and
carrabin precomputes 200 trial-seeds against yoo's 30/soltani's 40, so
`nc=2000` there would cost ~6.4GB against ~1-1.3GB for the other three at the
same `nc`. `nc=500` for carrabin also meant reusing a file already on disk
from an earlier session, rather than generating anything new.

`fitting/model_params.py`'s NEF `fixed` dicts were updated accordingly
(previously carrabin ran at 100/100 via `_NEF_FIXED`'s own untouched
defaults; yoo/soltani ran at 200/1000). This is the ONLY mechanism that
controls submit-time size -- `fitting.fit`/`fitting.submit` have no CLI
override for it -- so this file change IS the actual decision, not a
convenience default sitting alongside a real override.

This also settles something left open earlier in the session: carrabin and
yoo's NEF fits are being redone FRESH at these new sizes, not just soltani's.
Their existing fits (`data/runs/carrabin/`, `data/runs/yoo/`, `data/runs/
refit/`) are now the OLD, smaller-`n_neurons` baseline -- every current figure
script still defaults to reading those folders, so nothing breaks
automatically; repointing figures at the new fits is a deliberate later step,
not automatic.

### What's actually ready, and what isn't

Generated locally and verified this session: `counting_activities_n500_
nc2000_{yoo,soltani_numbers,soltani_colors}.pkl` -- correct key counts/ranges
for each dataset's own trial count, soltani_numbers/soltani_colors confirmed
byte-identical via checksum (as the disk-cost model predicts, since they
share radius_c/timing). Carrabin needed NO new generation at all --
`counting_activities_n500_nc500_carrabin.pkl` already existed on disk from an
earlier session's n==nc scan, and was re-verified this session (200/200 keys,
correct MtM shape).

Two things NOT confirmed before a weekend-long unattended submit, flagged
explicitly rather than assumed at the time:

1. Whether these files have actually been scp'd to the cluster. UPDATE
   (later same session): confirmed done -- the files (regenerated with
   `--n_sims 2` per the resumable scheme, adding the ensemble-mechanism's
   extra keys without touching the sim=1 entries already relied on for
   RMSE) were generated locally then copied to the cluster.
2. Real per-trial Nengo timing at the NEW sizes, for ANY dataset. The only
   real timing measured this session was carrabin at the OLD 100/100 size
   (~2s/trial, via `scripts/check_NEF_pipeline.py`'s real Nengo run). A
   relevant prior exists from a much earlier session (the shared-database/
   MLE exploration): NEF's per-point cost there was found to be dominated by
   fixed Nengo build/simulate overhead rather than scaling much with
   `n_neurons` across 50-500 -- but that was a DIFFERENT code path (the
   fit_mle.py loop, which also includes database-scan overhead), so it's a
   prior worth weighing, not a substitute for measuring this pipeline
   directly at 500/500 (carrabin) or 500/2000 (yoo/soltani). UPDATE (later
   same session): the person decided to proceed WITHOUT measuring this,
   fully informed -- worth recording exactly what was weighed. `fitting.
   fit`'s `study.optimize()` runs with no persistent Optuna storage (the
   CLI never passes `--storage`), and every output file is written ONLY
   after `study.optimize()` returns, i.e. only once all `n_trials` complete.
   A job killed at the 72h SLURM wall-clock limit produces NOTHING, not a
   partial result -- the 72h ceiling bounds wasted TIME, not wasted OUTPUT.
   `fitting.submit` submits one job per pid, so this run is 21 (carrabin) +
   38 (yoo) + 46 (soltani_numbers) + 46 (soltani_colors) = 151 separate
   jobs, each independently exposed to this risk. The person's call, made
   on the strength of the priors above -- not a gap I'm papering over.

### Submit plan (n_trials=200, per fitting.submit's own default)

    for ds in carrabin yoo soltani_numbers soltani_colors; do
      venv/bin/python -m fitting.submit $ds NEF --n_trials 200 --k 5 --run_folder rmse --dry_run
    done
    # inspect jobs/*.sh, then drop --dry_run to actually submit:
    for ds in carrabin yoo soltani_numbers soltani_colors; do
      venv/bin/python -m fitting.submit $ds NEF --n_trials 200 --k 5 --run_folder rmse
    done
    venv/bin/python -m fitting.collect rmse --type params
    venv/bin/python -m fitting.collect rmse --type responses

`run_folder rmse` reused deliberately (already holds soltani's math-model RMSE
fits) rather than inventing a new folder name -- `dataset_stem` keeps every
dataset's filenames distinct within it, so there's no collision risk, and
carrabin/yoo's NEF fits landing there for the first time doesn't touch
anything already using `data/runs/carrabin/`, `data/runs/yoo/`, or `data/runs/
refit/`.

---

## scripts/make_paper_figures.py consolidation, and a new consolidated neural predictions figure (Acts 1-5) (this session)

### The bigger picture: one script for every main-paper figure

`scripts/make_paper_figures.py` (new this session, copied from presentations/
make_figures.py and substantially diverged since) is now the single script
building every main-paper figure -- RMSE/NLL model performance, best-fit
fraction, response-change decay, and the lambda/sigma/neural "giant"
combined figures (`lambda_giant`, `sigma_giant`, `neural_giant`), among
others. Each giant figure stacks several existing figures' own panels into
one combined figure by calling their EXACT SAME underlying panel-drawing
helpers -- never duplicating panel logic -- so the standalone figures stay
available and unaffected. This entry focuses on the neural predictions work
specifically (the newest, least-precedented piece); the other figures'
many individual styling iterations (axis ranges, label wording, legend
placement, significance-bar conventions, etc.) aren't itemized here.

### Neural predictions figure: motivation and 5-act structure

See CLAUDE.md's own `## Neural predictions figure (Acts 1-5)` section for
the full, current-state version of this -- summarized here with the
reasoning behind the key decisions.

The theory's central mechanism is a weighted prediction error (PE) updating
an internal estimate. The person's framing: many of the behavioural
phenomena already shown in the P/T figures are hypothesised to be driven by
the dynamics of the neural population that represents this PE, and the
NEF's own simulated error population is hypothesised to resemble a real
neural population somewhere in PFC (or possibly striatum/VTA) -- so every
claim in this figure is meant to be a concrete, testable prediction for a
real neuroimaging experiment, not just a description of model internals.

Five acts, escalating in strength of claim:
1. Toy/illustrative population dynamics (no fitting, no behavioural data) --
   raster + decoded PE demo; error-neuron activity vs observation across a
   few lambda values; decoded PE vs time-within-observation across an
   alpha_0 x n_neurons grid. Claims: alpha_0 controls the PE upswing;
   n_neurons controls its noise level; lambda controls the rate error
   neurons go quiescent.
2. Behaviour <-> PE representation, BOTH axes measurable (no model fitting
   needed on either axis): sigma vs PE variability; delta-R(early-late) vs
   delta-A(early-late).
3. Both X and Y jointly controlled by the same underlying parameter: sigma
   AND PE variability vs fitted alpha_0; fitted lambda vs delta-R-decay AND
   delta-A-decay (twin axes) -- same underlying data as Act 2, replotted.
4. Validation via ablation/statistical control (not yet built) -- partial
   correlations plus, where feasible, mechanistic ablations, matching
   yoo's own existing lambda=0 ablation precedent.
5. Optional -- synaptic vs working-memory implementation comparison under
   an ITI manipulation (not started; separate downstream scope, explicitly
   deferred rather than folded into this figure).

**Task choice, and why it's a real simplification over the old N1-N8
table**: the OLD per-task neural figures (figure_carrabin_neural.py,
figure_yoo_neural.py) each only ever showed HALF the story, because
carrabin has a real fitted sigma but no fitted lambda, while yoo has a real
fitted lambda but no fitted sigma. Both soltani tasks (colors, numbers)
have real fits for BOTH, discovered only once NEF was actually fit for all
4 datasets this session -- so Acts 1-3 all run on ONE task (`soltani_
numbers`, picked arbitrarily over colors) instead of splitting the
argument across two tasks that each only cover half of it.

### scripts/neural_experiments.py (new)

Generalizes scripts/extras_carrabin.py's own pattern (param-grid sweeps,
probe simulations) to an arbitrary `--task`, since none of it is actually
carrabin-specific under the hood -- models/NEF.py's build_network/_pretrain
are always built on counting_integrator regardless of dataset (confirmed
directly before assuming this generalized). Three experiments:

- `raster_demo` -- one trial, arbitrary (alpha_0, n_neurons, lambda_), full
  per-timestep trace of the error population's raw neuron output (for
  utils/plot_spikes.py's spike-raster machinery) plus decoded value/PE.
- `sweep` -- vary ONE or TWO of {alpha_0, n_neurons, lambda_} (a cross
  product if two) across arbitrary values, others at fixed base values,
  over several seeds, at full per-timestep resolution. Two-parameter mode
  was added after the person reflected on an initial single-parameter
  design and preferred the original reference figure's own two-parameter
  (alpha_0 x n_neurons) convention instead -- required a real code change,
  not just a replotting, since two single-parameter sweep calls at
  different base values would otherwise silently overwrite the same output
  file (the old filename didn't encode which base value the OTHER
  parameter was held at).
- `probe` -- full per-timestep simulation at a pid's own fitted params
  across their real trials, for Act 2/3's within-repeat variability numbers
  (sigma/PE-variability). Has a `--mode run/submit/collect` lifecycle,
  reusing utils/slurm.py's existing generic job-script/submit machinery
  (the same one fitting/submit.py itself uses) rather than inventing new
  submission plumbing -- this is the expensive piece, cluster-bound,
  matching carrabin's own 2.36GB probe_pids_carrabin.pkl at only 21 pids
  (numbers has 46, longer sequences).

Output: data/runs/neural_experiments/. A genuine environment-mismatch bug
was caught and fixed while first writing this file: the `create_file` tool
writes to a different (local sandbox) filesystem than `filesystem:write_file`/
`shell:run_command`/`view` operate on (the actual remote project host) --
the SAME failure mode as `bash_tool` vs `shell:run_command` earlier in this
session. Caught immediately (the script simply didn't exist when run), not
silently.

### Act 1: done and iterated on substantially

All three panels (raster+PE demo, lambda sweep, alpha_0 x n_neurons sweep)
were built, then iterated on heavily against two hand-drawn reference SVGs
the person uploaded (PE_dynamics.svg, lambda_drives_discounting.svg) --
rasterized locally via `wkhtmltoimage` after `cairosvg`/`rsvg-convert` were
unavailable, so the actual intended visual style could be inspected rather
than guessed from raw SVG XML. Real bugs found and fixed along the way,
worth recording:

- The raster's own "active neuron" selection was silently a no-op:
  utils/plot_spikes.py's `preprocess_spikes` convenience wrapper defaults
  to `sample_size=200`, which exceeded the raster's own `n_neurons=100`, so
  its variance-based active-neuron filter never actually filtered anything,
  and its final `merge` step then block-averaged neurons into synthetic
  composites. Fixed by calling `sample_by_variance`/`cluster` directly with
  `num=50` (well under 100) and skipping `merge` -- confirmed by checking
  check_NEF_pipeline.py and its archived predecessor, the only other
  NEF-dynamics spike-raster code in the repo, that neither does anything
  more than the same no-op wrapper call.
- Swapping the raster/PE-line twin axes to opposite sides (PE on the left,
  raster's own neuron-count scale on the right, once its label was removed
  per instruction) required setting `ax`'s own tick-right AFTER calling
  `ax.twinx()`, not before -- twinx() was found (by rendering) to reset it
  back to the left otherwise, stacking both axes' tick numbers on top of
  each other.

Current Act 1 parameter values (arbitrary, chosen for visual separation
after a few rounds of "does this show a real difference" iteration --
lambda 0.1 vs 0.5 at alpha_0=0.3/n=500 barely separated at all, so this is a
genuine finding worth remembering, not just a styling choice):
- Raster demo: alpha_0=0.8, n_neurons=100, lambda_=0.7, n_obs=15.
- Lambda sweep: lambda_ in {0.1, 0.6}, base alpha_0=1.0, base n_neurons=100,
  n_obs=15, n_seeds=10.
- Alpha_0 x n_neurons sweep: alpha_0 in {0.1, 0.3}, n_neurons in {30, 300},
  base alpha_0=0.1, base n_neurons=30, base lambda_=0.5, n_obs=15,
  n_seeds=10.

Final panel-level formatting settled this session: panel 1 (raster) x-axis
zoomed to 5 observations (10s), y-range 0-50 (no tick labels) for the
raster and 0.0-0.8 for decoded PE; panel 2 (alpha_0 x n_neurons) y-range
0.0-0.3 (ticks every 0.1), dashed "PE/Response measured at" markers hidden
by default (`show_markers=False`, a real parameter, not just removed);
panel 3 (lambda sweep) x-range 0-15 (ticks every 5), y-range 62-82 (ticks
every 2).

### Act 2/3: data sources identified, neither run yet

Two data sources, both already fully supported by EXISTING infrastructure
-- no new code needed for either:
1. Probe simulation, via neural_experiments.py's own `probe` command.
2. Per-observation ensemble activities/encoders, via utils/save_activities.py
   (already dataset-generic -- the same mechanism that produced yoo's own
   activities_error_yoo.pkl), invoked through fitting.submit's existing
   `--resubmit activities` mode. This was a real discovery this session --
   initially assumed this would need new code in neural_experiments.py,
   until tracing fitting/collect.py's own `_collect_activities` back
   through fitting/submit.py's `--resubmit activities` branch to
   utils/save_activities.py, which turned out to already handle any
   dataset generically. NEF_soltani_numbers_responses.pkl (needed for the
   delta-R-decay half) already exists from the original RMSE fit -- no
   regeneration needed there either.

Neither has actually been run yet. Commands are recorded in CLAUDE.md's own
`### Neural predictions figure` simulation-pipeline recipe.

### Act 4/5: not started, deliberately

Per instruction: "we'll explore 4 and 5 once we have 1-3 in hand."

### Follow-up: separating neural output from behavioural fits (rmse/nll)

Caught before it caused a real problem, not after: `fitting.submit
--resubmit activities` (the mechanism identified above for Act 2/3's
activities data source) used the SAME `--run_folder` for both READING
fitted params and WRITING activity/encoder output. Run against
`--run_folder rmse` as originally planned, this would have written neural
simulation output directly into `data/runs/rmse/`, mixing it with pure
behavioural fitting results (params/performance/responses) -- exactly what
the person wants that folder reserved for.

Fix: `fitting/submit.py`'s `_resubmit()` and `utils/save_activities.py`'s
`run()` both gained an optional `out_folder` (CLI: `--out_folder`),
defaulting to `run_folder` when omitted -- verified directly (via
`--dry_run`) that omitting it produces byte-identical job-script commands
to before the change, and that passing `--out_folder neural_experiments`
correctly reads params from `rmse/` while writing activities/encoders to
`neural_experiments/`. No existing caller's behavior changes; this was a
genuine, small generalization (not a workaround), matching the pattern
`extras_carrabin.py` already used elsewhere for the same read/write split.

---

## Noise-only NLL fitting adopted as the new default (this session)

### Motivation

The person noticed model-performance RANKING under NLL and RMSE looked
nearly identical, and asked a sharper question than "are the losses
similar": are the FITTED PARAMETERS themselves substantially different
under NLL vs RMSE? If not, NLL fitting is basically just fitting the added
noise on top of an RMSE-shaped fit, and the joint search could be
simplified -- directly relevant to NEF, where a joint alpha_0/lambda_/
n_neurons search under NLL is expensive (real Nengo simulations per trial),
and fixing alpha_0/lambda_ at their RMSE values while only searching
n_neurons would need far fewer Optuna iterations to land somewhere good.

### First check: how much do RMSE-fitted and full-joint-NLL-fitted
parameters actually differ?

Compared LeakyIntegrator (gamma), PrimacyRecency (eps_p, eps_r), and
RL_lambda (alpha_0, lambda_) across all 4 datasets, RMSE fit vs the
EXISTING full-joint NLL fit (data/runs/nll/):
- LeakyIntegrator: r=0.93-1.00, drift 0-4% everywhere -- hypothesis holds
  strongly.
- RL_lambda: r=0.71-0.98 everywhere, but drift magnitude varies a lot by
  task -- carrabin/yoo 17-26%, both soltani tasks 0-5%.
- PrimacyRecency: r=0.38-0.78, drift up to 38% -- hypothesis does NOT hold
  for this model; its parameters genuinely move under NLL.

This was suggestive but indirect -- correlation/drift in the PARAMETERS
doesn't directly answer whether performance or behaviour actually differ.

### The real test: a noise-only fitting branch

Built a genuine "fix the base params, search only noise" fitting mode,
rather than just reasoning from the parameter comparison above:

- `fitting/fit.py`: `_suggest_params()` gained a `fixed_override: dict`
  parameter -- pins any listed parameter to an explicit value instead of
  Optuna-suggesting it, whether or not that parameter is normally free in
  `MODEL_PARAMS[dataset][model_type]` (replaces a normally-searched range,
  OR adds a parameter not in that spec at all -- the latter needed for a
  future NEF variant whose own spec would only list n_neurons, with
  alpha_0/lambda_ supplied entirely via the override).
- `fit()` gained `override_from_folder` (CLI: `--override_from_folder`):
  reads `{base_model}_{stem}_{pid}_params.pkl` from that folder (base
  model via `base_model_of()`, stripping "_resp_noise"), falls back to the
  combined `{base_model}_{stem}_params.pkl` filtered by pid if no per-pid
  file exists (found directly: carrabin's own RMSE folder only has the
  combined file, not per-pid ones -- the fallback exists because of this,
  not preemptively), and builds `fixed_override` from whichever of the
  base model's own free parameters are found there. Also had to fix a real
  bug caught by this: `best_params` (saved at the end of `fit()`) is built
  from `study.best_trial.params`, which NEVER contains override-pinned
  values (Optuna only records what it actually suggested) -- without
  merging `fixed_override` back in, the saved params/responses would have
  been missing alpha_0/lambda_/gamma/etc entirely.
- `fitting/submit.py`: `--override_from_folder` threaded through
  `_resolve_jobs`/`_submit_job`/`_run_local`, matching how `--datafile`/
  `--n_sims` already get threaded through.

Smoke-tested directly before any real campaign: RL_lambda_resp_noise on
soltani_numbers pid 1 (confirmed only sigma_resp varied across trials, and
the saved alpha_0/lambda_ EXACTLY matched that pid's real RMSE fit) and
LeakyIntegrator_resp_noise on carrabin pid 1 (confirmed the combined-file
fallback path also produces an exact match) before trusting either code
path at scale.

### Running it: n_trials=100, all 3 models x all 4 datasets, into data/runs/nll_noise_only/

First attempt ran locally (`--local`) in the background; killed after
observing ~1 pid/minute (far slower than a 20-trial smoke test suggested,
likely per-study Optuna/TPESampler overhead dominating at this trial
count) -- at that rate the full 453-fit campaign (3 models x 151 pids)
would have taken 7-8 hours blocking. Resubmitted to the cluster instead
(one SLURM job per pid, running in parallel, using the SAME
DEFAULT_TIME_LIMITS/DEFAULT_MEM_LIMITS as every other non-NEF fitting job
-- nothing NEW requested).

**A real bug in the one-line submit command, caught by checking rather
than assuming**: the per-dataset RMSE-folder lookup used a chained
`A && B || C && D || E` bash expression to pick carrabin/yoo/rmse per
dataset. Confirmed directly (by echoing each branch rather than trusting
mental tracing of bash operator precedence) that for `ds=carrabin`
specifically, BOTH the "carrabin" and "yoo" branches fired -- a successful
`||` short-circuit still leaves exit status 0, which the FOLLOWING `&&`
then sees as "proceed" -- producing a malformed multi-value
`--override_from_folder` argument. This is exactly why carrabin's jobs
never appeared in `jobs/` at all (fitting.submit errored out before
writing any job script). yoo/soltani_colors/soltani_numbers were
unaffected (confirmed their own branches resolve correctly). Fixed by
resubmitting carrabin alone with an explicit if/elif/else instead of the
chained expression.

### Result: the hypothesis holds, with one real exception

Compared noise_only against the existing full-joint NLL fit
(data/runs/nll/), on BOTH loss and actual simulated behaviour, across all
12 model x dataset combinations:
- Loss: full-joint is statistically significantly better in 11/12 combos
  (Wilcoxon; large n makes even a tiny shift detectable) but the actual
  magnitude is negligible in all but one -- 0.001-0.028 NLL units against
  medians of -0.5 to -2.2 (under 2% of the loss scale).
- Behaviour: response correlation between the two fits' simulated
  trajectories is r=0.995-1.000 everywhere; RMSE between them is
  0.009-0.058 on the [-1,1] response scale.
- The one real exception: RL_lambda on carrabin, diff=0.116 NLL units,
  r=0.995 -- still high in absolute terms, but 4-6x every other cell's own
  diff and the weakest correlation of the 12. Consistent with the
  parameter-drift comparison earlier in this same entry: RL_lambda's own
  alpha_0/lambda_ move the MOST under full-joint NLL specifically on
  carrabin/yoo, so fixing them costs the most exactly there. The soltani
  tasks -- where the neural work actually runs -- show no such issue.

### Decision

Adopted as the default NLL fitting method going forward for these 3
models: data/runs/nll_noise_only/ is now the CANONICAL location for new
NLL fits; data/runs/nll/ (the old full-joint search) is kept as the
verification baseline this comparison was run against, not deleted, not
being added to further. See CLAUDE.md's own "Default NLL fitting method"
section for the current-state summary and exact recipe.

Applying the same approach to NEF (fix alpha_0/lambda_, search only
n_neurons) remains the motivating end goal but is NOT YET BUILT -- blocked
on a real prerequisite found while scoping it: every candidate n_neurons
value needs its own precomputed counting-activity file, and currently only
the single production value (500) has one, for any of the 4 datasets.
Generating a real candidate set is a genuine disk/compute cost not yet
scoped or approved.

---

## From NEF NLL n_neurons search to synthetic forward simulation for Acts 2/3 (this session)

### The NEF joint-search attempt: built, smoke-tested, abandoned

Following on directly from the entry above: implemented `fitting.fit`'s own
`search_n_neurons` parameter -- a genuine joint (alpha_0, lambda_, n_neurons)
NLL search for NEF, promoting n_neurons out of `MODEL_PARAMS`' `"fixed"`
dict and into an Optuna-searched categorical, mirroring n_neurons into
n_neurons_counting on every trial (the two must move together -- confirmed
directly that `models.counting_integrator.build_network` reads BOTH from
params). Threaded through `fitting/submit.py` identically to how
`override_from_folder` already was.

Two real infrastructure problems surfaced immediately, unrelated to the
search design itself:
- `losses.nll_from_ensemble` computes `ens.std(axis=0, ddof=1)` -- this is
  UNDEFINED at n_sims=1 (zero degrees of freedom), confirmed directly by a
  real failed trial ("Degrees of freedom <= 0", NaN loss). The precompute
  used for search_n_neurons's own candidate n_neurons values therefore
  needed n_sims>=2, which multiplies file size roughly linearly -- at
  n_sims=50 (matching NEF_DEFAULT_N_SIMS elsewhere) this would have run
  ~67.5GB across a 10-value grid (100-1000 step 100), confirmed by
  measuring the actual n_sims=1 baseline sizes and scaling -- too large
  for the cluster's home-directory quota (Discovery: 50GB). Narrowed to 3
  values (200/500/1000) at n_sims=20 (~9.2GB) as a compromise.
- A cluster-side symptom (`ReqNodeNotAvail, Reserved for maintenance`) led
  to two SEPARATE, permanent fixes, kept regardless of what happened to
  the search itself: `utils/slurm.py` gained `SINGLE_PASS_TIME_LIMIT`
  (2h) for `--resubmit activities`/`responses` jobs, which are one
  forward pass and were absurdly requesting the same 72h walltime sized
  for a full 200-trial fit; and `sbatch --export=ALL` was made explicit
  rather than relying on the cluster's own default for environment
  inheritance.

A `NEF_ACTIVITY_SCRATCH_DIR` env-var fallback was added to
`models.counting_integrator.load_activities` (to let the inflated
n_sims=20 files live on Discovery's `/scratch` instead of the 50GB home
quota) and then FULLY REVERTED once the search itself was abandoned --
confirmed no trace of it remains in `load_activities`, `_require_activity_
map`'s error message, or anywhere else, keeping `sbatch --export=ALL`
(genuinely useful on its own merits) but nothing scratch-specific.

**Why abandoned**: reconsidering the original motivation (checking whether
RMSE-fitted alpha_0 might be masking a real alpha_0-n_neurons interaction
by compensating for noise) against what panel 7 (sigma vs alpha_0) had
already shown with REAL fitted params -- soltani_numbers' own alpha_0 is
too tightly clustered near ceiling (median 0.994) to reveal ANY alpha_0
effect regardless of whether n_neurons is also searched. The person
reconsidered and asked for all three parameters (alpha_0, lambda_,
n_neurons) to vary TOGETHER so their individual contributions could be
teased apart via partial correlations, rather than fixing any one of them
-- which is exactly what search_n_neurons was building toward. But it was
never run at scale: the person then proposed synthetic data instead (see
below), which sidesteps the entire NLL-fitting apparatus.

`search_n_neurons` was then removed entirely from `fitting/fit.py` and
`fitting/submit.py` (both the parameter and its CLI flag) -- confirmed
zero remaining references except the docstring note explaining why it's
gone. `override_from_folder` (the math-model noise-only work, still
active) was confirmed unaffected by a real dry-run smoke test after the
removal.

### The pivot: synthetic forward simulation instead of NLL fitting

The person's reframing of what Acts 2/3 actually need to claim: "our model
predicts that behavioral and neural quantities vary together... this is a
behavioral/neural prediction we expect to be validated in empirical
experiments if they are ever run... we can explain the origins of these
effects through neural parameters." Since this is a prediction for FUTURE
studies to test, not a fit to existing behavioural data, artificial data is
exactly as good as real-pid data -- freeing the design from needing NLL
fitting, real fitted params, or even real trial sequences at all.

Design, settled across several exchanges:
- N=200 "virtual pids" (more than the real 46, since nothing ties this to
  actual participants).
- Each virtual pid = one independently-generated trial sequence (NOT a
  real participant's) paired with ONE randomly-drawn (alpha_0, lambda_,
  n_neurons) -- one parameterized NEF model per virtual pid, not shared
  across several.
- alpha_0, lambda_ ~ Uniform(0, 1) (chosen after reviewing real RMSE-fit
  summary statistics across all 4 datasets together and finding raw
  mean+/-2SD bounds routinely fell outside each parameter's actual [0,1]
  range -- simpler to just sample the full range directly, especially
  since the whole point is to escape numbers' own narrow real alpha_0
  distribution).
- n_neurons ~ uniform choice over {100, 200, ..., 1000} -- narrowed from
  an initial 3-value plan (200/500/1000, inherited from the abandoned
  search) back to the full 10-value grid once n_sims=1 was confirmed
  sufficient (see below), since disk was no longer the constraint it was
  under the NLL-search plan.

**Trial sequences**: `task_backend/generate_sequences.py --task numbers
--n_pool N` already generates independent pool members with the exact
same repeated-prefix/qid structure real participants get (confirmed
directly: 200 members, 32 trials each, 8 qids x 4 repeats, 15 observations
per trial, matching soltani_numbers' own real structure exactly) --
written to `data/synthetic_pool/`, never touching real experimental data.
No new generation code needed; this existing script IS the pool-generation
mechanism, just called with a larger --n_pool than task_backend itself
ever uses in production.

### A real, previously-uncaught bug found and fixed along the way

While designing the new pipeline, checked how `NEF.run()` (the validated,
heavily-used fitting pathway) pairs each trial with its own network seed,
and found `neural_experiments.py`'s OWN `_probe_worker` (used for the
original fitted-pid Acts 2/3 data, built earlier this session) did it
differently and wrongly: `activity_map.get(int(trial))` and `p = {**params,
"seed": int(trial)}`, using the raw 0-indexed trial number directly for
both the activity-map lookup AND the simulation seed. `NEF.run()` instead
uses `models.counting_integrator.activity_key_for_trial(dataset, trial)`
for BOTH -- a shared helper that exists specifically because soltani
trials are 0-indexed while activity keys start at 1, and because key k's
decoders are ONLY valid for a network built with seed=k (confirmed
directly in `build_network`: `net.error = nengo.Ensemble(..., seed=seed,
...)` -- the error ensemble's own encoders are seeded by the exact same
value). The bare-trial-number version either missed the activity map for
trial 0 entirely (falling back to the much slower, seed-mismatched
_pretrain path) or, for every other trial, silently paired that trial's
simulation with a DIFFERENT trial's own tuning curves -- plausible-looking
but wrong output, not an error.

Fixed by switching `_probe_worker` to use `activity_key_for_trial` for
both the lookup and the seed, matching `NEF.run()` exactly. The affected
output (`probe_soltani_numbers_pid*.pkl`, `probe_soltani_numbers.pkl` --
everything the original fitted-pid Acts 2/3 panels were built from) was
deleted rather than kept, since it's unknown how much any given trial's
result was actually affected without regenerating. `probe` remains a real,
working experiment for other uses; it's just no longer what Acts 2/3 draw
from.

This also raised a question about the SAME per-trial-seed convention for
the new synthetic pipeline: if each trial's network gets a genuinely
different seed, its error-ensemble encoders differ too (same code path
confirmed above) -- so identifying which neurons are weight-tuned (needed
for the activity-decay panels) can't reuse one pid-level encoders file the
way `utils/save_activities.py`'s own (seed-never-varies-per-trial)
convention does. Resolved by saving encoders PER TRIAL for the new
pipeline, not per virtual pid -- confirmed this is a real requirement, not
just caution, directly in `build_network`'s own code.

### The new pipeline: scripts/neural_experiments.py's `synthetic` experiment

One new experiment, same `--mode run/submit/collect` shape as `probe`:
- `_synthetic_params(virtual_pid)`: deterministic RandomState(seed=
  virtual_pid) draw, so re-running `--mode run` for the same pid always
  reproduces the same draw.
- `_load_synthetic_trials`: reads that virtual pid's own pool member from
  `data/synthetic_pool/sequences_numbers.json` (task_backend's own bare
  task name, not this project's "soltani_"-prefixed key -- handled via a
  small path-mapping helper).
- `_simulate_trial_full`: one build+run per trial, extracting response and
  decoded PE at their usual readout points AND per-neuron error-population
  activity at the SAME points (tau_probe-filtered, matching `utils/save_
  activities.py`'s own convention for this quantity -- deliberately NOT
  the raw synapse=None probe `_simulate_full` uses for the spike raster,
  a different use case), plus that trial's own encoders.
- `_synthetic_worker`: loops a virtual pid's ~32 trials, handles the
  activity-map lookup (falling back to `_pretrain` if a given seed isn't
  precomputed), assembles four DataFrames (probe/activity/encoders/params)
  from ONE simulation pass -- satisfying the requirement that no further
  commands are needed afterward.

Verified directly before trusting this at scale:
- `_simulate_trial_full` in isolation: correct shapes (activity (n_neurons,
  ), encoders (n_neurons, 2)) for a 3-observation toy trial. Caught and
  fixed one bug in the process -- `value_trace` needed `.squeeze()` (shape
  (T,1) since net.value has dimensions=1), the same convention `_simulate_
  full` already used elsewhere in this file but that this new function
  had omitted.
- Full `_synthetic_worker` for virtual_pid=6 (drawn n_neurons=100, the
  cheapest case, for a fast smoke test): 480 probe rows and 480 activity
  rows (32 trials x 15 observations, exact), 3200 encoder rows (32 x 100,
  exact), qid repeat counts exactly 4 for all 8 qids, params matching the
  deterministic draw exactly. This run's own output was KEPT (not deleted
  as a throwaway) since virtual_pid=6 is legitimately part of the eventual
  200 -- `--mode submit`'s own existing-file check will skip it
  automatically when the full campaign runs.

### Counting-activity files: regenerated at a clean, uniform n_sims=1

The abandoned search had left three of the ten files (n=200/500/1000)
inflated to n_sims=20; the rest were already at n_sims=1 from the original
precompute. Since `synthetic`'s own probe-style variability comes from
repeated qids across trials (not an ensemble average), n_sims=1 is
sufficient -- confirmed this matches the SAME mechanism `_probe_worker`
already used successfully. Deleted and regenerated all 10 (100-1000 step
100) at n_sims=1 uniformly, bringing total disk back down from the
inflated set's multi-GB sizes to the original ~1.35GB total.

### Status

Data generation is running (200 virtual pids, cluster-submitted); the
figure itself (`make_neural_giant()`'s row 2 panels, currently the
`_load_neural_probe_variability`/`_load_neural_decay_metrics` functions in
`scripts/make_paper_figures.py`) has NOT yet been rewired to read from the
new `synthetic_soltani_numbers_*.pkl` output -- still pointed at the old
(now-deleted) fitted-pid `probe_soltani_numbers.pkl` and `NEF_soltani_
numbers_responses.pkl`/`activities_error_soltani_numbers_*.pkl`. That
rewiring, plus the weight-tuned-neuron filtering logic (needs updating to
use the new PER-TRIAL encoders rather than the old per-pid ones), is the
next concrete step once the cluster run finishes.

---

## Rewiring neural_giant to synthetic data, two real bugs, sampling bounds, and neural_main (this session)

### Rewiring make_paper_figures.py to the synthetic output

`_load_neural_probe_variability`/`_load_neural_decay_metrics` were
rewritten to read `synthetic_soltani_numbers_{probe,activity,encoders,
params}.pkl` instead of the old fitted-pid `probe`/`activities_error_*`/
`NEF_soltani_numbers_responses.pkl` files. The weight-tuned-neuron
filtering for activity decay had to change shape, not just source: since
encoders are now saved PER TRIAL (confirmed earlier this session that a
trial's own seed determines its own encoders), identifying weight-tuned
neurons is now a groupby-apply over (virtual_pid, trial) pairs rather
than a single per-pid lookup. Column names (`resp_std`, `pe_std`,
`act_decay`, `resp_decay`, `alpha_0`, `lambda_`) were kept identical to
the old loaders' own output, so none of the four downstream plotting
functions needed any changes at all -- confirmed by checking their source
directly rather than assuming.

### First real result, and immediate concern

Initial run (old Uniform(0,1)/{100,...,1000} bounds): sigma-vs-PE-
variability improved (r=0.41->0.55) but the decay-related relationships
collapsed -- activity-decay-vs-response-decay went from r=0.83 (original
real-fit figure) to r=-0.13 (null), and lambda-vs-decay from r=0.74/0.84
to r=-0.02/0.08 (null). The person immediately flagged this as suspicious
rather than accepting it, specifically suspecting the seed-handling
change introduced a bug -- which led directly to finding bug #1 below
(though, as it turned out, bug #2 was the dominant cause of THIS specific
collapse).

### Bug #1: activity_key_for_trial missing from the new pipeline

Checking whether `scripts/check_NEF_pipeline.py` could detect the
suspected bug (it couldn't -- confirmed directly that its own checks
never call anything in neural_experiments.py, only validating consistency
within models/NEF.py itself) led to comparing `_simulate_trial_full`
directly against the canonical `models.NEF._simulate_trial` instead. This
surfaced a REAL, separate bug in the OLDER `_probe_worker` (built earlier
this session, feeding the ORIGINAL fitted-pid Acts 2/3 data): it used
`activity_map.get(int(trial))` and `seed=int(trial)` directly, instead of
`models.counting_integrator.activity_key_for_trial(dataset, trial)` (the
shared helper `NEF.run()` itself uses, existing specifically because
soltani trials are 0-indexed while activity keys start at 1, and because
a network's own encoders are seeded identically to its counting
subnetwork -- confirmed directly in `build_network`: `net.error =
nengo.Ensemble(..., seed=seed, ...)`). Fixed by switching to
`activity_key_for_trial` for both the lookup and the seed; the affected
old `probe_soltani_numbers*.pkl` files were deleted rather than kept,
since it's unknown how much any given trial was actually affected without
regenerating.

### Bug #2: raw vs rescaled observation values (the dominant cause)

Directly comparing `_simulate_trial_full` against `_simulate_trial` on
identical inputs (same params/seed/obs_values) found the two agreed to
floating-point precision ONCE a smaller, separate discrepancy in the
response-readout formula was fixed (`_simulate_trial_full` used a single-
point lookup; `models.NEF._extract_responses` averages over a small
window, `|t - readout_time| < dt*3` -- fixed to match exactly). This
confirmed the SIMULATION machinery itself was correct, which reframed the
question: why did the person's own observation of "|Delta R| decay values
5-20x larger than the original figure" still not add up?

The actual answer surfaced from a different direction: the person asked
for a check-NEF-pipeline comparison at a real (alpha_0, n_neurons=1000)
combination against RL_lambda, which came back in a sane ballpark (RMSE
~0.07, growing across the trial as expected) -- ruling out a broken core
pipeline. Then, comparing an actual virtual pid's own NEF response
sequence against what RL_lambda would predict for the identical trial
revealed the real problem directly: `_synthetic_worker` was feeding RAW
0-100-scale pool values (e.g. "16, 14, 21, ...") straight into NEF. The
pool JSON (`task_backend/generate_sequences.py`'s own output) is on the
raw scale; `data/soltani_numbers.pkl`'s own "value" column (what the
older, correctly-working `_probe_worker` reads) had ALREADY been rescaled
(x/50-1) by `scripts/build_model_inputs.py`'s `build_from_df()` before
ever reaching NEF. Feeding raw values saturates NEF's ensembles
(radius_e=1.5, radius_v=1.0) almost immediately -- confirmed directly: a
sample trial's NEF response sequence was `[0.22, 0.42, 0.60, 0.76, 0.95,
0.99, 0.99, ...]`, saturating near +1 rather than tracking a running
mean, exactly the "plausible-looking but meaningless output" failure mode
`nef_obs_values()`'s own docstring warns about. Fixed by applying the
exact same x/50-1 rescale to the pool's raw values before simulating,
task-aware (numbers needs it; colors' own pool values are already +-1 and
must NOT be rescaled again). Confirmed directly after the fix: the same
sample trial's rescaled inputs (all negative, -0.68 to -0.98) produced a
smooth, sensible NEF response trajectory (-0.32 to -0.76), not saturation.

Regenerating with BOTH fixes recovered activity-decay-vs-response-decay
to r=0.86 (matching the original r=0.83) and brought lambda-vs-decay from
null back to real and significant (r=0.16/0.38, still weaker than the
original r=0.74/0.84 -- which motivated the bounds investigation below).
Sigma-vs-PE-variability landed at r=0.43, and a new, unexpectedly strong
finding appeared: alpha_0-vs-PE-variability at r=0.67 (mechanistically
sensible: PE = alpha(t) x (obs-value), so alpha_0 directly scales the PE
product's own variance).

### Sampling bounds investigation

The person asked what happens restricting to n_neurons>=500: sigma-vs-PE-
variability jumped to r=0.81 and alpha_0-vs-sigma(response) went from
null (r=0.11) to real (r=0.50) -- but the decay-related relationships
barely moved. This split makes mechanistic sense: sigma is DIRECTLY a
noise-driven quantity, so it should be most sensitive to n_neurons' own
role controlling spiking-noise magnitude; decay reflects a systematic
drift (alpha(t) discounting), a much less noise-dependent signal.

The person was still surprised lambda-vs-response-decay stayed weak, and
asked to check n_neurons>500 AND lambda_>0.1 AND alpha_0>0.2 together.
Checking each restriction separately (not just the combination) found
alpha_0>0.2 ALONE did almost all the work (lambda-vs-decay jumping to
r=0.36/0.54), while n_neurons>500 alone gave a smaller boost and
lambda_>0.1 alone actually WEAKENED the correlation slightly (trimming
only the low end of lambda's own range costs power to detect lambda's own
slope). This has a clean, direct mechanistic explanation, not just an
empirical pattern: alpha(t) = alpha_0/t^lambda, and at very low alpha_0,
alpha(t) is near zero for EVERY t regardless of lambda -- there's no
updating for lambda to modulate in the first place, a genuine floor
effect in the equation itself, not a noise problem more neurons could
ever fix.

Checking the REAL fitted alpha_0 distribution for soltani_numbers
confirmed this restriction doesn't exclude real behaviour: 0 of 46 real
pids have alpha_0<0.2, and the true observed minimum is 0.384 (5th
percentile 0.481) -- so >=0.2 was already safe, just looser than
necessary; a tighter floor around 0.4-ish would track the real population
more precisely, though the real distribution is heavily skewed toward 1
regardless (median 0.994), which a uniform sampling range never fully
matches.

**Final bounds adopted**: alpha_0 ~ Uniform(0.5, 1), lambda_ ~
Uniform(0.1, 1), n_neurons ~ uniform choice over {500, 600, ..., 1500}
(raised from {100,...,1000} for the same n_neurons>=500 reasoning).
Regenerating with these bounds: EVERY relationship in the figure came out
real and significant with no post-hoc filtering needed -- sigma-vs-PE-
variability r=0.80, alpha_0-vs-sigma(response) r=0.31, activity-vs-
response-decay r=0.67, lambda-vs-decay r=0.52/0.64 (both up substantially
from the original wide-bounds r=0.16/0.38). The counting-activity
precompute grid was correspondingly widened to 500-1500 (11 values,
n_neurons=n_neurons_counting, n_sims=1 -- confirmed sufficient, since
`synthetic`'s own variability comes from repeated qids across trials, not
an ensemble average needing n_sims>=2).

### neural_giant layout restructure

Per instruction: row 2 (sigma_R/sigma_PE) and row 3 (DeltaR/DeltaA-decay)
each now lead with their own scatter-vs-scatter panel, then three twin-
axis panels (one per alpha_0/lambda_/n_neurons) via a new shared
`_plot_neural_dual_vs_param` helper -- a genuine breakdown of how much
each parameter individually contributes, not a substitute for the still-
pending multivariate regression. Y-axes (including twin axes) are shared
across each full row, applied manually after plotting since twin axes
aren't reachable via `plt.subplots`' own `sharey` (they're created per-
panel, not at subplot-creation time). Point size reduced substantially
and alpha lowered throughout, emphasizing the regression line and its CI
band over individual points. Axis labels: "Fitted " prefix dropped,
"n_neurons" -> "neurons", 0 included as an x-tick where the parameter is
bounded and 0 is a meaningful reference (alpha_0/lambda_ panels and both
scatter panels; NOT n_neurons, where real data starts at 500 and forcing
the axis to 0 would waste half the panel on empty space).

### neural_main: a second figure, the oddball paradigm

The person wanted a different angle: isolate each of alpha_0/lambda_/
n_neurons's own causal contribution to the error population's response to
a SURPRISING observation, rather than the giant's own random-covariation
design. Design, settled iteratively:

- 3 observations clustered tightly around a center, then one "oddball"
  observation deviating from it by a fixed amount -- after the 3
  clustered observations, the value population should have converged
  near the center, so the error population's response represents
  |oddball - center|.
- The person's own prediction, stated explicitly: for a FIXED deviation
  MAGNITUDE, the response (in abs(PE)) should be roughly independent of
  which center it's centered on. Rather than assume this, it's checked
  directly via a dedicated panel (see below) -- built in per instruction
  ("it can be a 3rd column... removed if it shows a trivial result"),
  since a trivial (fully-overlapping) result is itself the thing that
  justifies aggregating across centers elsewhere in the figure, not
  wasted verification.
- Simulated across a full grid: cluster_centers x oddball_deviations x
  one swept parameter (the other two held fixed at explicit --base_*
  values), using abs(decoded PE) throughout. One cluster job per grid
  cell (`--mode run/submit/collect`, same lifecycle as `synthetic`/
  `probe` -- a real timing check found even a modest single-context run
  (5 values x 20 seeds) exceeds a reasonable single local call, well
  before the full grid multiplies that further).
- Values are on the raw 0-100 scale (matching how a person would
  describe them, e.g. "60, 59, 61" then "70") -- rescaled via the exact
  same x/50-1 transform bug #2 above required, applied task-aware inside
  `_oddball_worker` this time from the start.
- Deviation magnitude was narrowed from an initial +-15/+-10 to just
  +-10 (per instruction), halving the grid from 80 to 40 jobs for the
  first real run.

**Layout** (3x3, one row per parameter): col 1 = |decoded PE| vs time at
one representative (center, deviation) context, 3 representative values
of that row's parameter; col 2 = that parameter (x) vs max |decoded PE|
AND % decrease by trial end, twin axes, mean +- SEM aggregated across the
WHOLE grid; col 3 = the center-invariance check itself, one line per
cluster_center at a fixed deviation and (near-)base parameter value.

**Verification before trusting any real run**: built a synthetic grid
file with KNOWN properties (PE amplitude deliberately made proportional
to the swept parameter, decay rate deliberately made parameter-
independent, response deliberately made center-independent) and confirmed
all three panel functions reproduce exactly what those known properties
predict -- max PE scaling with the parameter, a flat % decrease line, and
perfect overlap across centers in the invariance panel. This is a code-
correctness check, not a result -- the real prediction (whether alpha_0
genuinely produces bigger AND more-attenuated responses, and whether
centers genuinely don't matter) is still unconfirmed pending the actual
cluster run.

**Status**: row 1 (alpha_0) built and code-verified; data submitted to
the cluster (`--cluster_centers 20 40 60 80 --cluster_spread 1
--oddball_deviations -10 10 --sweep_values 0.2 0.4 0.6 0.8 1.0
--base_alpha_0 0.7 --base_lambda_ 0.7 --base_n_neurons 500`, n_neurons=
500/nc=2000 held fixed -- the RMSE production default) but not yet
collected. Rows 2 (lambda_) and 3 (n_neurons) are not yet built.

### A real infrastructure caveat, encountered and resolved mid-session

During this session, `view` (filesystem MCP) and bash-tool file reads both
briefly showed STALE content immediately after a real, successful
`edit_file` write (confirmed by the tool's own diff output) -- direct
bash writes were confirmed NOT to persist to the shared file at all,
while bash READS eventually caught up to a successful edit_file write
after a short lag, rather than staying permanently stale. Treated as a
transient consistency delay rather than a split-brain issue going
forward: `edit_file`/`str_replace` remain the only reliable way to
persist code changes; bash reads used for verification should be treated
with a little skepticism immediately after a write, re-checked if a
result looks unexpectedly stale.

### neural_main's param_scan: a degenerate-input bug, then a full redesign

A later session revisited `neural_main`'s rows 2/3 (lambda_/n_neurons,
the `param_scan` experiment) after the oddball/param_scan debugging above
had settled. Several separable things happened, in order:

**1. A real bug: constant input gives the model nothing to integrate.**
The original `param_scan` (see above) simulated ONE arbitrary trial with
`obs_values=ones(n_obs)` -- appropriate for the ORIGINAL neural_giant's
own illustrative Act 1.2 lambda panel, carried over unexamined into a
NEW use (a quantitative decay-vs-parameter regression). A flat,
unchanging input gives the value ensemble nothing to actually integrate,
so any activity trend across observations was an artefact of the
degenerate input, not a real lambda-driven signature. Caught when the
person noticed the resulting deltaA decay metric's sign looked flipped
relative to the ORIGINAL giant's own (real-trial-based, via `synthetic`)
equivalent panel. Direct comparison confirmed the two decay formulas were
IDENTICAL in sign convention (both "start minus end") -- the discrepancy
was the generative design, not a missing minus sign.

**2. Fix: move to real trials, per-pid.** `param_scan` was rewritten to
run on REAL soltani_numbers trials -- each replicate is one real pid,
using that pid's own 32 real trials, with `activity_key_for_trial`
providing both the activity-map lookup and simulation seed for each
trial (the same convention `_probe_worker` already uses). Every swept
parameter value initially still saw every real pid (a (sweep_value, pid)
full cross-product, ~46 pids x N values = job count multiplying
quickly).

**3. A design problem, then ANOTHER fix: per-replicate random draws.**
Once real data was flowing, the person noticed the decay-vs-param
regression panel (col 2) had an unnatural, discrete-strip appearance --
every pid landed on the exact same handful of x-positions (the shared
explicit grid), rather than a genuinely continuous parameter axis. Fixed
by moving to ONE random draw per replicate: each real pid (or, later,
each synthetic virtual_pid) gets its OWN single value of the swept
parameter, drawn uniformly from `[--sweep_low, --sweep_high]`,
deterministically seeded per (trial_source, sweep_param, pid) via
`zlib.crc32` on an explicit byte string. Using Python's own built-in
`hash()` for this was considered and explicitly rejected mid-implementation
before it ever shipped: `hash()` on `str` is randomized per-process
(`PYTHONHASHSEED`), so seeding from it would have silently drawn a
DIFFERENT value for the same pid on every separate cluster job -- a
reproducibility bug that would have been very hard to notice after the
fact (each individual run looks fine; only a cross-run comparison would
reveal the drift). Job granularity dropped from (sweep_value, pid) to
(pid) alone as a direct consequence -- one job per replicate, not one
per (value, replicate).

**4. Raising N: the synthetic 200-member pool.** With per-pid random
draws working, the person asked to increase N past the ~46 real pids
available by running the same design on the pre-generated synthetic
pool (`data/synthetic_pool/sequences_numbers.json`, 200 members x 32
trials each, same structure `synthetic`'s own virtual pids use).
Confirmed directly before implementing: pool JSON values are on the RAW
0-100 scale (e.g. `[16, 14, 21, 16, 24]`), NOT canonical [-1,1] -- same
rescale bug class as bug #2 far above, this time avoided proactively by
checking `_synthetic_worker`'s own docstring first rather than
discovering it via saturated ensembles. `--trial_source {real,synthetic}`
added as an explicit choice on `param_scan`'s own CLI, rather than one
silently replacing the other.

**5. Col 1's own redesign: threshold groups, not representative values.**
With every replicate now having its OWN random value rather than sharing
a fixed grid, `_plot_neural_main_activity_vs_obs` (row 2/3 col 1) could
no longer pick "the low/high value" the way the original grid-based
version did. First fix: pick whichever single pid's own draw landed
NEAREST each target (0.1, 0.7) and plot just that one pid's own curve.
Superseded almost immediately, per instruction, by a group-based version:
every pid with `sweep_param<=low_thresh` (line 1) and every pid with
`sweep_param>=high_thresh` (line 2) -- letting EVERY matching replicate
contribute, not just the single nearest one. Combined with a separate
instruction to actually use seaborn's own automatic mean+CI aggregation
here (matching `_plot_oddball_pe_trace`'s already-established convention
-- hand seaborn a long-format frame at the correct unit-of-independence,
let it aggregate, rather than manually computing mean/SEM +
`fill_between`): each matching pid's own trials are pre-folded to ONE row
per (pid, observation) first (raw per-timestep samples within a trial
aren't independent draws and would otherwise make seaborn's own CI
falsely tight), THEN that per-pid frame is handed to
`sns.lineplot(hue="group")`, which computes the mean and CI band across
pids itself. N per group now appears directly in its own legend entry.

**6. Two new DV-vs-DV panels (col 3), one per row.** The ORIGINAL
neural_giant already pairs a "two dependent variables plotted directly
against each other" panel with every row (panel 5: sigma_R vs sigma_PE;
panel 9: DeltaR-decay vs DeltaA-decay) -- `neural_main` had no
analogue until this session. Added `_plot_oddball_dv_scatter` (row 1:
max |decoded PE| vs decrease, one point per grid cell) and
`_plot_param_scan_dv_scatter` (row 2: reusing `_param_scan_decay_metrics`
directly, no new data-loading code) -- both matching the ORIGINAL
giant's own established house style exactly (flat color, small
low-alpha points, `sns.regplot` fit + CI band, pearsonr r + significance
stars) rather than color-coding by the swept parameter, since that
convention was already established elsewhere in this exact file and a
new one wasn't asked for. Figure grid grew from 3x2 to 3x3 as a direct
consequence.

**A separate infrastructure discovery, unrelated to any of the above**:
the machine Claude's own filesystem/shell MCP tools reach (`hydra`) and
the machine the person's own cluster jobs actually run on/from
(`discovery-01`, under a different username there) are SEPARATE
filesystems. Confirmed directly: Claude deleted a truncated collected-
data file via its own tools, confirmed it gone via a follow-up `ls` --
but the person's own attempt to rerun the exact same regenerate command
on `discovery-01` reported the file still present ("Already exists --
skipping"). This means Claude cannot verify ANY cluster-side file state
(existence, size, truncation, job queue status) directly going forward
-- every such claim Claude makes from its own tool calls reflects
`hydra` only, and needs the person's own terminal output as the real
ground truth. Recurred at least twice more the same session (a second
truncated file, from a job that appears to have been killed mid-write
on the actual compute node) -- diagnosed each time from file SIZE alone
(one file at ~1/6th the size of every sibling file, all others
identical to each other), without needing to inspect cluster job logs
directly.

### neural_main row 3 (n_neurons): SNR measure exploration

Before building row 3's real data-generation panels, a `run_n_neurons_
convergence` diagnostic (`neural_experiments.py`, now archived --
see `archive/scripts/archive_n_neurons_convergence_exploration.py`)
walked through several candidate SNR measures on the SAME oddball trial
structure `oddball` already uses, across three (n_neurons, n_neurons_
counting) pairs at a 4x ratio (50:200, 100:400, 200:800).

**1. Convergence hypothesis, tested directly, found NOT to hold
cleanly.** The motivating idea: 3 pre-observations clustered within +-1
unit of the oddball's own cluster_center should let the network's
running VALUE estimate settle to nearly the same level across seeds
before the oddball hits, so cross-seed variance in PE/response AT the
oddball would then directly reflect momentary error-population SNR
rather than accumulated drift. Measured directly (value/PE mean and
variance at several windows -- pre-obs1 baseline, after obs1, the ITI
right before the oddball, the oddball's own presentation, and the final
decision) with BOTH a within-seed (across-time) and an across-seed (of
each seed's own window mean) variance decomposition for every window.
Result: `value_iti_before_oddball`'s across-seed variance was 13-56x its
own within-seed variance across all three pairs -- most of the cross-
seed spread by the oddball is DRIFT, not momentary noise, even with
tightly clustered pre-observations. `value_decision`'s across-seed
variance (0.034 at n=50) was nearly identical to `value_iti_before_
oddball`'s (0.029) -- confirming most of the FINAL decision's own spread
was already baked in before the oddball ever arrived. PE fared better
but not perfectly: across/within ratio ~1.1-1.5x at observation 1
(before any drift could accumulate) vs ~2.2-3.5x at the oddball -- real
but much smaller drift contamination than the value-based measures.
Kept the oddball paradigm anyway (rather than falling back to
observation-1 measures) since real human data shows little variance at
the first observation -- the settled-expectation-then-surprise
manipulation is the intended design, not a workaround.

**2. Response variability (sigma_response) vs. PE-within-seed-variance,
both decrease with n_neurons, different slopes, same underlying cause.**
Comparing relative decline from n=50 to n=200: response variability drops
~15x; PE-within-seed variance (during the oddball's own 400-600ms window,
centered on the established ~0.5s peak-response latency) drops only
~6.8x. Resolved as expected, not a discrepancy: response variance is a
DOWNSTREAM, integrated/amplified consequence of the same per-instant
noise source, compounded across every observation that fed into the
current estimate, while PE-within-seed-variance measures that noise
source at one instant -- same sign, different magnitude, connected by
the integration process.

**3. A purely-neural (decoder-free) complement was requested, to see
what's plausibly measurable from real spike data without needing a
trained decoder.** Raw per-seed error-population spike arrays (BOTH the
obs1 and oddball 400-600ms windows, every neuron, plus that trial's own
encoders) were saved to a TEMPORARY folder specifically so multiple
candidate measures could be tried without rerunning any simulation.

  - **Within-trial Fano factor (tried, abandoned)**: bin spike counts,
    compute per-neuron variance/mean ACROSS BINS within one trial,
    average across neurons. Result: flat/noisy across n_neurons
    (mean~0.12-0.16, sd~0.03-0.07 at every n_neurons value) -- NO trend,
    unlike every decoded measure. Diagnosed as a genuine conceptual
    mismatch, not a power/binning issue: Fano factor is a SINGLE-NEURON
    statistic (how variable is one unit's own count, relative to its own
    mean) with no mechanism to capture the POPULATION-AVERAGING benefit
    that decoding gets from combining many independent noisy units
    (decoded SNR improves roughly ~1/sqrt(n) from averaging; a single
    neuron's own regularity doesn't have to change at all as n grows).

  - **Split-half population reliability (worked)**: bin spike counts
    (20ms bins), randomly split a subpopulation into two halves (50
    random splits, averaged), pool (sum) each half's own counts per bin,
    correlate the two halves' pooled time series -- all WITHIN one
    trial. Unlike Fano factor, this DOES capture population-averaging:
    each half gets more neurons too as n_neurons grows, so each half's
    own pooled signal becomes a cleaner average of the shared
    (stimulus-driven) component. Confirmed on weight-tuned-only neurons
    first: r rose from 0.48-0.64 (n=50) to ~0.83 (n=200), with across-
    seed SD shrinking from ~0.17-0.20 to ~0.04-0.05 over the same range --
    the cleanest n_neurons effect found from any purely-neural measure.

  - **Population choice matters for interpretation, not just magnitude**:
    compared weight-tuned-only vs ALL neurons vs non-weight-tuned
    (PE-dimension-tuned) neurons. ALL neurons gave higher r (0.81-0.94)
    and non-weight-tuned alone gave intermediate r (0.71-0.92) than
    weight-tuned-only (0.48-0.83) -- but this is confounded by population
    SIZE (non-weight-tuned neurons are ~2x the weight-tuned count in this
    network, and ALL neurons is the full union), not necessarily by
    population IDENTITY; a fair size-controlled comparison was flagged
    as the correct way to test identity specifically, but not run, since
    non-weight-tuned was chosen for a different, practical reason
    instead (see next point). Also: the obs1-vs-oddball asymmetry visible
    with weight-tuned-only at n=50 (0.64 vs 0.48) shrank close to zero
    with the larger non-weight-tuned/all-neuron populations -- consistent
    with that asymmetry being a small-population-size artefact rather
    than something specific to the weight dimension.

  - **Settled on non-weight-tuned neurons specifically**, per instruction
    -- "weight-tuned" is a model-internal concept (which dimension a
    neuron's own encoder points toward) that doesn't map onto anything an
    experimentalist could identify from real data, while "PE-dimension-
    tuned" (non-weight-tuned, in this network's own 2D error ensemble) is
    easier to explain and could plausibly be operationalized empirically
    (e.g. neurons whose firing tracks surprise/error magnitude rather
    than the integration weight itself).

  - **Population identity, confirmed directly from models/NEF.py**: all
    of this spike data is from the ERROR population (`net.error.neurons`,
    2D: weight dimension + raw-PE dimension) -- NOT the value population
    (`net.value`, a separate ensemble holding the running decoded
    estimate), which none of this touched.

**Settled on two DVs for panel 2, both restricted to the SAME 400-600ms
window within the oddball's own presentation**: (1) decoded PE within-
seed variance, (2) split-half spike-population reliability on non-
weight-tuned neurons. `run_n_neurons_convergence` was renamed to
`run_n_neurons_snr` and simplified to compute ONLY these two, ENTIRELY IN
MEMORY -- the temporary raw-spike-saving step (and every other measure
tried above) was archived rather than kept live, since the exploratory
flexibility it existed for is no longer needed once the measure is fixed,
and persisting raw spikes at real-panel scale (many more n_neurons
values x more seeds) would be needlessly large. The old
data/runs/neural_experiments/tmp_spike_arrays/ folder and the old
n_neurons_convergence_soltani_numbers.pkl output are stale and should be
deleted.

### neural_main row 3: n_neurons_snr grid expansion + cluster job-splitting

Once the two SNR DVs were settled, the person asked to expand `n_neurons_
snr` from a single fixed (cluster_center, oddball_deviation) combo to a
full grid, matching row 1's own oddball design: the SAME cluster_centers
row 1's alpha_0 sweep already used (20, 35, 50, 65, 80), oddball_
deviations of both signs (-10, 10, not just +10), and two more (n_neurons,
n_neurons_counting) pairs at the established 4x ratio (150:600, 250:1000
-- confirmed directly, NOT 250:800 as first guessed from an ambiguous
phrasing, since 250x4=1000). Aggregation across this grid isn't decided
yet, so every cell keeps its own point rather than being pre-averaged.

Asked directly whether the FIRST oddball experiment (row 1) needed
cluster job-splitting at this same scale, rather than assuming --
confirmed directly from `run_oddball`'s own code: yes, `oddball`'s
`--mode submit` already loops over the full (cluster_center, oddball_
deviation, sweep_value) cross-product, one job per cell (exactly how the
90-cell alpha_0 grid was run earlier in this same session). `n_neurons_
snr` was brought in line with that exact precedent: extracted the
per-cell computation into `_n_neurons_snr_worker`, added the same
--mode run/submit/collect lifecycle, one job per (cluster_center,
oddball_deviation, n_neurons_pair) cell (50 jobs for the current grid --
5 centers x 2 deviations x 5 pairs, 10 seeds each), rather than one long
sequential local call (500 simulated trials, well past what a single
local run should attempt, matching the same reasoning `oddball`'s own
docstring already gives for its own grid).

## Boundary-clipping correction for sigma-growth negative control (this session)

### Motivation: filling in `make_sigma_main`'s blank row 2

`make_sigma_main` had a row 2 left deliberately BLANK (see the earlier
"3x3 kept ... with row 2 left literally blank" note this section now
replaces). The person wanted a second, more intuitive companion to row
3's autocorrelation panel for the same underlying claim -- that human/
NEF response variability under REPEATED, identical stimuli (a qid/
quasi-qid prefix) should GROW across observations within that shared
prefix, because spiking/state noise gets folded into the represented
state and compounds forward, whereas the math models' own noise
mechanism (`models/math_models.py`'s `add_noise`, i.i.d. Gaussian added
AFTER each deterministic update) has no such persistence and should stay
flat. This is T5 (residual variance growth) from the PTN taxonomy,
computed the same way `_resid_variance_growth` already did for
carrabin/soltani's own established figures -- that function existed in
`make_paper_figures.py` already, unused, apparently left over from an
earlier draft of this exact row.

### Bug #1, found first: every `_resp_noise` pid/model shared one seed

Initial diagnostic (raw `_resid_variance_growth`, Human + the four
`_resp_noise` math models + NEF, on balls/colors/numbers) showed a clean
dichotomy on balls (Human/NEF grow, math models mildly decline) but NOT
on colors/numbers, where several math models grew almost as much as
Human -- and, suspiciously, in near-lockstep with each other despite
having unrelated functional forms (Mean/LeakyIntegrator/PrimacyRecency/
RL_lambda).

Traced to `models/math_models.py`'s `add_noise`: its noise draw was
`RandomState(int(params.get("seed", 0)))` -- and `fitting.fit` never set
a `"seed"` key anywhere it called `add_noise` (the Optuna-search-time
ensemble draw, or the final single-realization save draw). Every pid,
every wrapped model, defaulted to the SAME seed, so the underlying
z-draw sequence was IDENTICAL across pids, just rescaled by that pid's
own fitted `sigma_resp` -- not independent noise. Any incidental
position-dependent pattern in that one fixed draw would show up
identically, after rescaling, in every pid and every model: exactly the
synchronized "growth" observed.

Fixed via `_resp_noise_seed(pid, model_type)` (new in `math_models.py`),
deriving a seed unique per (pid, base model) through
`utils.run_params.trial_seed` (int/tuple-of-int hashing only -- Python's
str hashing is randomized per-process via `PYTHONHASHSEED`, so a naive
`hash((pid, model_type_string))` would NOT have been reproducible across
the separate process invocations `fitting.fit` runs per pid). `add_noise`
now uses this whenever `params` has no explicit `"seed"`; an explicit
seed still overrides it (kept for any caller wanting several independent
realizations of the same pid/model).

All 604 previously-run `_resp_noise` NLL fits (Mean/LeakyIntegrator/
PrimacyRecency/RL_lambda x carrabin/yoo/soltani_numbers/soltani_colors)
were deleted and resubmitted with the identical job spec recorded in
`data/runs/nll/run_config.json` (`n_trials=200, k=5, optuna_seed=42,
n_sims=100`, no `override_from_folder`) -- cheap to redo since these are
closed-form math models, no NEF/cluster-scale simulation involved. Fixed
the growth pattern almost everywhere: balls and numbers went fully flat
(ratios ~0.94-1.06, from ~0.85-1.66 before), matching Human/NEF's own
clean growth (~1.3-2.8) with no confound left to explain.

### Bug #2 (not a bug): colors' remaining growth is real boundary clipping

Colors still showed real growth for Mean (x1.54) and PrimacyRecency
(x1.37) even after the seed fix -- LeakyIntegrator/RL_lambda were fine
(~0.97/1.18). Ruled out an imperfect-qid-match explanation directly: the
deterministic `mu` component is bit-identical (max std 0.0 across 5
pids' worth of qid groups, all three tasks) within a qid group, so no
real signal is leaking in via mismatched stimuli.

The actual cause: `add_noise`'s `clip(mu + N(0,sigma_resp), -1, 1)`.
Colors is binary evidence, so a running-mean-like estimator's `mu` after
ONE observation is exactly the observation itself -- exactly +-1, sitting
right on the clip boundary. Confirmed directly: Mean's own
distance-to-boundary went 0.0 (obs 0) -> 0.375 -> 0.438 -> 0.531 -> 0.6
(obs 4) for one pid. Near the boundary, roughly half of any added noise
draw gets truncated away (censored, not resampled), so the OBSERVED
variance is well below the true `sigma_resp^2`; as `mu` moves inward
across observations, less gets clipped and the observed variance climbs
back toward the true value -- pure measurement artefact, zero change in
the actual (constant, per-pid) noise parameter. This also explains WHY
only Mean/PrimacyRecency show it: both put a large, unshrunk weight on
the very first observation (Mean IS that observation; PrimacyRecency
weights it heavily), while LeakyIntegrator/RL_lambda scale the first
observation by `(1-gamma)`/`alpha_0` respectively, so their own obs-0 `mu`
is rarely close enough to +-1 for clipping to matter.

Quantitatively confirmed via the closed-form variance of a clipped
normal, `Var(clip(X,-1,1))` for `X ~ N(mu, sigma^2)` (folded/censored-
normal moments, standard derivation via `E[X^2*1(a<X<b)]`'s decomposition
into truncated first/second moments). Computed the PREDICTED growth
curve from nothing but each pid's already-fitted `mu` trajectory and
`sigma_resp` (no new fitting) and compared to the ACTUAL observed curve:

| task | model | observed ratio | predicted ratio |
|---|---|---|---|
| balls | all four | 0.94-0.97 | 0.99-1.02 |
| colors | Mean | 1.54 | 1.35 |
| colors | LeakyIntegrator | 0.97 | 0.93 |
| colors | PrimacyRecency | 1.37 | 1.35 |
| colors | RL_lambda | 1.18 | 1.13 |
| numbers | all four | 0.95-1.06 | 0.99-1.01 |

The match (especially colors' two affected models, both landing within
0.02-0.19 of the real ratio using ZERO free parameters beyond what was
already fit) leaves no real, unexplained mechanism to hunt for.
Cross-referenced against a HUMAN-side twin of this exact issue already on
record: colors' first observation is independently known to be
ceiling-pinned for real participants too ("78.7% of colors prefix
responses are pinned at an extreme vs 0.3% for numbers" -- see this
file's temporal-figure section), and dropping that observation from
cols 3-4 was tried and REJECTED there because losing a fitting point
cost more than the contamination. Same tradeoff, different figure.

### Considered and rejected: removing clipping, or excluding data

Three alternatives to a figure-level correction were weighed before
settling on one:

- **Remove clipping from `add_noise` entirely.** Rejected: the clip is
  part of the actual fitted model, not a plotting artefact -- removing
  it means refitting all 604 `_resp_noise` jobs against a genuinely
  different likelihood (the SAME fits also feed
  `make_sigma_model_correlation` and `make_model_best_fit`'s NLL panel,
  so the blast radius is wider than this one figure), and an unclipped
  model can predict responses outside the response scale's own bounds --
  not a fix, a different (worse) model.
- **Uniform boundary-distance exclusion** (drop any (pid, obs, qid)
  group whose qid-conditional mean sits within some threshold of +-1,
  applied identically to every source). Tested directly at thresholds
  0.8-1.0: DOES flatten the math models (e.g. Mean's ratio -> 1.01 at
  threshold 0.9), but WRECKS Human's own curve at the exact same
  threshold -- Human's colors obs-0 responses are pinned even harder
  than the models' (as expected, matching the ceiling-effect finding
  above), so the filter shrinks that observation's human sample to a
  tiny, unstable remainder (one test run: ratio flipped to 0.40 with a
  spurious spike at obs 0). Rejected -- fixes the wrong side of the
  comparison.
- **Drop observation 0 as a whole column, colors only, uniformly.**
  Better than the soft filter (no per-group sample shrinkage) but only a
  PARTIAL fix: the clip effect is a gradual function of distance-to-
  boundary across ALL four colors observations, not just obs 0 (matches
  the predicted-vs-observed table above, which already differs from 1.0
  well past the first point) -- Mean's ratio only came down to 1.30 with
  the whole column dropped, still a visible, confusing residual trend.
  Rejected as insufficient on its own.

### The fix actually shipped: analytic boundary correction, colors' Mean/PrimacyRecency only

Inverted the same closed-form relationship: given each pid's own fitted
`sigma_resp` and known `mu` trajectory, solve (via `scipy.optimize.
brentq`, monotonic in sigma for fixed mu so the root is unique) for the
IMPLIED pre-clip sigma at each observation that would produce the
ACTUALLY OBSERVED (clipped) variance. This is a real correction -- the
standard treatment for censored-variance estimation (the same idea
behind a Tobit correction) -- not a data exclusion or a model change.
Flattens both affected curves close to 1.0:

| model | raw ratio | corrected ratio |
|---|---|---|
| Mean | 1.54 | 1.15 |
| PrimacyRecency | 1.37 | 0.98 |

Applied ONLY to `("colors", "Mean")` and `("colors", "PrimacyRecency")`
(`SIGMA_GROWTH_BOUNDARY_CORRECTED` in `make_paper_figures.py`) --
LeakyIntegrator/RL_lambda on colors and every model on balls/numbers
were already flat under the raw metric once the seed bug was fixed, so
correcting those too would only add `brentq`-inversion noise for no
benefit. Implemented as `_clipped_normal_var` (the closed-form moments),
`_implied_sigma` (the inversion), `_resp_noise_params_path` (needed
because the correction must re-run the deterministic base model from its
fitted params to get `mu` -- the saved `_responses.pkl` file only has the
already-noisy realization), `_resid_variance_growth_corrected` (the
per-pid/per-observation implied-sigma curve, same `[observation, mean,
std, se]` shape as the raw `_resid_variance_growth` so it drops into the
same drawing code), and `_growth_stats_for_source` (the dispatcher that
swaps in the corrected version only for the flagged pairs).

**This is a genuine asymmetry, by design, not an oversight**: Human and
NEF stay on the raw empirical metric throughout, because there is no
equivalent parametric correction available for either -- Human's own
boundary effect is a real behavioural ceiling response (not a known-form
clip on a fitted noise parameter to invert), and NEF's noise is not
simple additive Gaussian. Each source gets the estimator that's actually
right for what it is, rather than forcing one uniform (and, for Human,
demonstrably worse -- see the rejected boundary-filter test above)
treatment onto all of them.

### Row 2, as originally shipped (superseded -- see "Row 2/3 normalization and cleanup" below)

`make_sigma_main`'s row 2, AS ORIGINALLY WIRED IN: Human (solid grey) +
`NLL_RESP_NOISE_MODELS` (solid, except colors' Mean/PrimacyRecency which
were DASHED) + NEF (solid, its own color) -- mean +/- SEM per
observation, RAW (absolute SD) scale. The figure-level legend marked any
model corrected on ANY task with a trailing `"*"` and a legend title
("* colors only: boundary-corrected (see docs/HISTORY.md)") rather than
per-task marking, since the legend itself was one entry per model
across the whole figure. `NEF` was a genuine addition to this row's
roster relative to row 3 (which at the time only ever plotted
`NLL_RESP_NOISE_MODELS`) -- it is the other real positive case for this
metric, not just a negative control.

One scale mismatch was noticed while rendering: NEF's curve on
colors/numbers sat at a much smaller absolute scale (~0.01-0.03) than
Human/the math models (~0.07-0.33), because it reads the plain RMSE-fit
`_responses.pkl` (a single point-estimate realization, less noisy in
absolute terms) rather than an MLE/ensemble variant. RESOLVED below --
not via the twin-axis treatment first suggested here, but by
normalizing every curve to its own baseline, which makes absolute-scale
differences stop mattering at all.

## Model-performance cleanup and lambda/sigma giant retirement (this session)

### model_performance: RMSE-only again, tighter sig-bar headroom

`make_model_performance` briefly carried a second (NLL) row,
consolidating both metrics into one figure. That row was removed, per
instruction -- NLL reporting for the same roster/tasks lives entirely in
the pre-existing `make_model_performance_nll` (its own 1x4 figure),
which already duplicated that exact content, so nothing needed a new
home. Figure is back to 1x4, RMSE only.

Three more targeted tightenings, all per instruction, all scoped to
this one figure (the shared `draw_sig_line`/`_draw_metric_boxplot`
conventions used elsewhere in this file are untouched):
- Sig-bar headroom: `dy_step` fraction cut from the shared 0.07
  convention to 0.045, with smaller lead-in/trail padding, so annotation
  margin doesn't dominate the panel.
- Outliers hidden (`showfliers=False`, new parameter on
  `_draw_metric_boxplot`, default `True` so the one other/future caller
  is unaffected) and the y-axis top pinned to the ACTUAL max whisker
  value via a new `_whisker_top()` helper (seaborn/matplotlib's own
  whis=1.5 definition), not matplotlib's autoscaled ylim -- autoscale
  still carries its own ~5% margin above whatever's topmost, which was
  exactly the leftover whitespace being removed.
- Legend pulled in tight under the x-axis: constrained-layout `h_pad`
  dropped from the shared 0.25 convention to 0.03, plus
  `borderaxespad=0.2` on the legend itself.

### make_lambda_giant retired, split into six

`make_lambda_giant` (a 4x4 mega-figure stacking `make_response_change`'s
own 4 panels, `make_lambda_overview`'s own 2x4 content, and
`make_lambda_model_correlation`'s own 3 panels, unmodified, into one
combined layout) was retired per instruction and split into six pieces.
Full code preserved at `archive/scripts/archive_lambda_giant.py` (NOT
standalone-runnable as archived -- it references module-level state
still living in `make_paper_figures.py`; see that file's own header for
the complete list). Removed from the `FIGURES` dict; `lambda_giant` as a
CLI argument now fails with a clear "invalid choice" listing every
valid name.

- `make_lambda_main` -- 2x3, rows 1-2 cols 2-4 of the giant
  (snacks/colors/numbers only; balls/demo column dropped).
- `make_lambda_metric` -- 1 panel, the giant's own row 2 col 1 (the
  "lambda definition" demo, INLINED regplot-with-binning version, not
  the plainer `_plot_lambda_demo` helper) -- pulled out standalone so it
  can be hand-composited into `lambda_main` as an Inkscape inset from
  its own saved SVG. Made SQUARE (figsize (3.7, 3.7)) later in the same
  session, matching the average dimension of one `lambda_main` panel
  (~3.53in wide x ~3.88in tall, 2x3 grid over `(FIGURE_SIZE[0],
  FIGURE_SIZE[1]*1.9*0.75)`).
- `make_lambda_balls` -- 1 panel, the giant's own row 1 col 1
  (balls-task response-change panel) -- moved to supplementary since it
  doesn't show the expected trend.
- `make_lambda_reliability` -- 1x3, the giant's own row 3 cols 2-4
  (split-half reliability), titles RESTORED (no longer sitting under a
  row that already names each task the way the giant's row 1 did).
- `make_lambda_humanvmodel` -- 1x3, the giant's own row 4 cols 2-4
  (model-vs-human lambda correlation), titles restored. NOTE: currently
  IDENTICAL in content to the pre-existing `make_lambda_model_correlation`
  (same panels, same helper, same roster) -- built as its own
  function/output per instruction rather than reusing that name, but the
  two now duplicate each other; not yet reconciled.
- `make_lambda_sigma_crosstask` -- NOT a piece of the original giant.
  Pairs the giant's own row 3 col 1 (lambda colors-vs-numbers crosstask)
  with the analogous panel from `make_sigma_giant`'s own row 2 col 1
  (sigma crosstask). Originally built 2x1 (stacked), changed to 1x2
  (side by side) later in the same session per instruction --
  `figsize=(FIGURE_SIZE[0]*2/3, FIGURE_SIZE[1]*0.75)`, swapped from the
  stacked version's `(FIGURE_SIZE[0]/3, FIGURE_SIZE[1]*1.5)` to keep each
  individual panel roughly the same size.

### make_sigma_giant retired, split into two

Same treatment, same session: `make_sigma_giant` (a 3x4 mega-figure:
rows 1-2 = `make_sigma_overview`'s own 2x4 content, row 3 = the
autocorrelation panels) retired and split, per instruction. Archived at
`archive/scripts/archive_sigma_giant.py`.

- `make_sigma_main` -- 3x3, column 1 (the "sigma definition"/"rho
  definition" schematics plus the crosstask panel -- previously column 1
  of rows 1/2/3) removed entirely:
  - Row 1: variability KDE panels, unchanged, real titles.
  - Row 2: left BLANK at this point in the session (see "Boundary-
    clipping correction" section above for how it got filled in, and
    "Row 2/3 normalization and cleanup" below for how it was refined
    further afterward).
  - Row 3: autocorrelation panels, TITLES CLEARED (redundant with row
    1's, two rows up).
  3x3 was kept rather than collapsing to 2x3, per instruction, despite
  row 2 being blank at the time.
- `make_sigma_reliability` -- 1x3 supplementary: the giant's own row 2
  cols 2-4 (split-half reliability of sigma), titles restored.

## Row 2/3 normalization and cleanup (this session, after row 2 was filled in)

Following the boundary-clipping-correction work above (which filled
row 2 in with the raw residual-variance-growth metric), three more
rounds of refinement, all per instruction, in the same session:

**Row 2 normalized to a relative scale.** `_draw_variance_growth_panel`
now divides each source's (Human, each `NLL_RESP_NOISE_MODELS` model,
NEF) own `mean`/`se` by that SAME source's own value at its own first
available observation, so every curve starts at 1.0 and the axis reads
as "how many times its own baseline sigma has grown by observation k" --
directly matching how this row's own results were already being talked
about narratively (e.g. "grows x1.31", "x2.02") well before the plot
itself was normalized to show that ratio at every point along x, not
just the endpoints. A dashed guide line at y=1 marks the baseline.
Resolves the NEF absolute-scale mismatch flagged in "Row 2, as
originally shipped" above with no separate twin-axis treatment needed.
Ylabel: `$\sigma_R$ (normalized)`.

**Row 2's own legend removed.** Per instruction -- row 3 now carries the
only legend for both rows (same Human/model/NEF color roster). This also
removes the `"*"`/dashed-line explanation that used to live in row 2's
legend title, marking colors' Mean/PrimacyRecency as boundary-corrected
-- the dashed line styling itself (`SIGMA_GROWTH_BOUNDARY_CORRECTED`) is
untouched, only the auto-generated legend note explaining it is gone.
Worth covering in a caption or the hand-drawn inset if that distinction
still needs surfacing to a reader.

**Row 3: NEF added.** `_load_variance_autocorr_data` gained an
`include_nef=True` parameter (default `False`, so
`make_variance_autocorr_human`/`models`, which never pass it, are
unaffected) that loads NEF's own residual autocorrelation using the SAME
special per-task path convention row 2's own NEF handling already used
(`_variability_model_path`'s MLE variant for balls, `_delta_responses_
path`'s RMSE variant for colors/numbers -- NEF is never in
`NLL_RESP_NOISE_MODELS`/`NLL_MODEL_ORDER`, so it can't go through the
normal `responses_path_fn` mechanism the math models use). Confirmed
directly: NEF's own autocorrelation shows the same decaying-from-well-
above-zero pattern as Human/NoisyRL_lambda at every lag, all three tasks
(e.g. numbers: 0.69 -> 0.39 -> 0.06). Row 3's legend now includes a NEF
entry. Ylabel changed to a placeholder shorthand, `$\rho_\varepsilon$
(autocorrelation)` (epsilon for "residual") -- meant to be spelled out
in a hand-drawn inset rather than the axis label itself, per
instruction.

### Autocorrelation pairing definition: pooled (t, t+k) vs first-obs-only

Question raised: does `_resid_autocorr` pool EVERY valid (t, t+k) pair
within a trial into one correlation per (pid, lag), or does it only use
the window's first observation paired with (first+k)? Answer: the
former, already -- confirmed directly by reading the implementation.

Added a `pool_all_pairs: bool = True` parameter to `_resid_autocorr`
(threaded through `_load_variance_autocorr_data` too) to make the
simpler alternative available for direct comparison, default `True`
preserving existing behavior for every caller. Built a throwaway
comparison (`scripts/_tmp_autocorr_compare.py`, numeric only, deleted
after use; then `scripts/_tmp_autocorr_variant.py`, a full row-3-style
plot with `pool_all_pairs=False`, rendered and inspected, then deleted).

**Result** (Human + NEF, all three tasks; at each task's own largest
lag the two necessarily coincide, since only one (t,t+k) pair remains to
pool there -- a useful sanity check that both branches agree):

| Task | Source | Pooled (k=1,2,3[,4]) | First-obs-only |
|---|---|---|---|
| Balls | Human | 0.62 / 0.41 / 0.27 | 0.52 / 0.35 / 0.26 |
| Balls | NEF | 0.88 / 0.76 / 0.64 | 0.82 / 0.70 / 0.61 |
| Colors | Human | 0.47 / 0.35 / 0.27 / 0.16 | 0.61 / 0.37 / 0.25 / 0.16 |
| Colors | NEF | 0.70 / 0.44 / 0.19 / 0.01 | 0.34 / 0.17 / 0.10 / 0.01 |
| Numbers | Human | 0.40 / 0.23 / 0.15 | 0.29 / 0.26 / 0.15 |
| Numbers | NEF | 0.69 / 0.39 / 0.06 | 0.35 / 0.14 / 0.06 |

Balls stays close either way. Colors/numbers Human shifts modestly and
inconsistently (sometimes up, sometimes down at k=1). Colors/numbers
NEF -- the case this row most needs to preserve -- roughly HALVES under
the simplified version (0.70->0.34, 0.69->0.35 at k=1). The simplified
version also loses statistical power: pooling combines multiple (t,t+k)
pairs per trial, so more pids clear the "n>=3 pairs" inclusion
threshold than under the one-pair-per-trial simplified version (e.g.
colors Human k=1: only 13 pids usable under first-obs-only).

**Decision: keep `pool_all_pairs=True`** (already the default; no code
change needed to `make_sigma_main` itself, which never passed the
parameter). The simplified version isn't just noisier -- it materially
attenuates the NEF signal specifically. The parameter itself stays in
the code (on both `_resid_autocorr` and `_load_variance_autocorr_data`)
for any future revisit.


## make_neural_giant retired; neural_main now authoritative (this session)

### Why

`make_neural_giant` (the original "Acts 1-3" figure: toy/illustrative
population dynamics at arbitrary params, then sigma_R/sigma_PE and
DeltaR/DeltaA-decay each plotted against alpha_0/lambda_/n_neurons via
RANDOM-virtual-pid covariation across all three parameters at once) and
neural_main (parameter-by-parameter: oddball for alpha_0, param_scan
for lambda_/n_neurons, isolating each parameter's own causal
contribution one row/column at a time) had been developed side by side
for several sessions, described as complementary -- the giant showing
random covariation, neural_main showing controlled per-parameter
sweeps. Per instruction this session: neural_main is now the sole,
authoritative figure for presenting the impact of neural parameters on
behavior and activity. The giant's own random-covariation design added
second-order value (showing the relationships hold under naturalistic
parameter covariation, not just controlled sweeps) but that value no
longer justified maintaining two full figures' worth of data pipeline
and plotting code side by side, especially with neural_main still
actively evolving (new n_neurons row, new DV-scatter columns, real vs
synthetic trial sources -- see its own CLAUDE.md section for the full,
current structure).

### What was archived

Confirmed via repo-wide grep before archiving that nothing else calls
any of the following -- all exclusive to `make_neural_giant`'s own
dependency tree (`_plot_neural_dual_vs_param`, `NEURAL_EXP_DIR`,
`NEURAL_READOUT_OFFSET`, and `_fold_observation_time` are all SHARED
with `make_neural_main` and were left in place):

- `make_neural_giant` itself (the 3x4 figure function).
- `_plot_neural_raster_demo`, `_plot_neural_lambda_activity`,
  `_plot_neural_pe_dynamics` (row 1's three toy/illustrative panels).
- `_load_neural_probe_variability`, `_load_neural_decay_metrics` (rows
  2-3's own data loaders, reading neural_experiments.py's `synthetic`
  experiment output).
- `_plot_neural_sigma_vs_pe_variability`, `_plot_neural_resp_vs_act_decay`
  (rows 2-3's own leftmost DV-vs-DV panels).
- `_neural_weight_on_cols` -- had ZERO callers even before this
  archiving (confirmed by grep); `_load_neural_decay_metrics` duplicates
  its logic inline instead of calling it. Archived alongside the rest
  since it's exclusively neural_giant-era code, not because it was ever
  load-bearing.
- `NEURAL_ENCODER_THRESHOLD` -- this file's own local copy of the
  same-valued constant `scripts/neural_experiments.py` independently
  defines for its own weight-tuned-neuron identification; NOT a shared
  import, so removing this copy doesn't affect that file.

Full code preserved at `archive/scripts/archive_neural_giant.py` (NOT
standalone-runnable as archived -- see that file's own header for the
full list of module-level state it still depends on). Also removed: the
now-fully-unused `from utils.plot_spikes import plot_spikes,
preprocess_spikes, sample_by_variance, cluster` import (only
`_plot_neural_raster_demo` used any of these, and `preprocess_spikes`
was already unused dead weight even within that function -- see its own
docstring, which explains why `sample_by_variance`+`cluster` were used
directly instead).

Removed from the `FIGURES` dict; `neural_giant` as a CLI argument now
fails with a clear "invalid choice" listing every remaining valid name.

### What was NOT touched (explicitly out of scope)

- `scripts/neural_experiments.py`'s own `raster_demo`/`sweep`/`probe`/
  `synthetic` subcommands -- these generated the now-archived figure's
  own data and have no other caller either, but the instruction was
  specifically about "the old neural_giant functions" (the plotting
  side, in `make_paper_figures.py`), not the simulation/data-generation
  pipeline. They're likely dead weight now too, but left alone rather
  than assumed in scope -- worth a follow-up pass if confirmed unwanted.
- The generated data files themselves under `data/runs/
  neural_experiments/` (`raster_demo_*.pkl`, `sweep_*.pkl`,
  `synthetic_*.pkl`, `probe_*.pkl`) -- left on disk, not deleted.
- The full "Acts 1-3" status/bug-fix narrative that used to live in
  CLAUDE.md's own "Neural predictions figure" section (the N=200
  synthetic-virtual-pid pipeline, its final sampling bounds, and the two
  real bugs found building it -- a probe-worker activity-key/seed
  mismatch, and a raw-vs-canonical observation-scale mismatch that
  saturated NEF's ensembles) -- NOT deleted, just no longer repeated in
  CLAUDE.md now that it describes a retired pipeline; the full narrative
  already lives permanently in this file's own history (see the
  "Neural predictions figure (Acts 1-5)" era entries earlier in
  docs/HISTORY.md if the exact diagnostic trail is ever needed again).

### Docs restructuring

CLAUDE.md's `## Neural predictions figure (Acts 1-5)` section was
restructured, per instruction ("the act 1-5 framework needs revisiting/
removing"):
- Renamed to `## Neural predictions figure` (dropped the "Acts 1-5"
  framing entirely). Motivation kept (still accurate), rewritten to
  point at neural_main's own per-parameter design instead of the old
  5-act narrative.
- "Structure -- 5 acts" subsection REMOVED (folded into neural_main's
  own already-detailed structure section, promoted below).
- "Implementation"/"Status" subsections (describing the now-retired
  pipeline in detail) REMOVED from CLAUDE.md -- that history stays
  intact in this file (HISTORY.md), per CLAUDE.md's own stated scope
  ("current state only"; HISTORY.md is "the full design history").
- The two forward-looking ideas that used to be "Act 4" (validation via
  ablation/partial-correlation control) and "Act 5" (a synaptic-vs-
  working-memory implementation comparison) were KEPT, per instruction,
  as a new "Future extensions" list -- decoupled from the retired
  figure's own numbering, so they read as standing todos for whichever
  figure eventually takes them on, not artifacts of a dead naming
  scheme.
- `### neural_main -- a second, parameter-by-parameter figure`
  (previously nested as a `###` subsection UNDER the Acts-1-5 heading,
  despite its own text already saying "a separate figure from Acts
  1-5") promoted to its own top-level `## neural_main` section, with
  a new RETIREMENT NOTE at the top explaining the change and its own
  "complementary to the giant... not a replacement for it" framing
  corrected (that's no longer true).
- The `## Simulation pipeline` section's own `### Neural predictions
  figure (Acts 1-3 ...)` command-reference subsection was relabeled
  RETIRED (commands kept for provenance -- they document how the
  archived figure's own data was generated -- but no longer read as an
  active workflow), and its stale `## Neural predictions figure (Acts
  1-5)` cross-reference and its `neural_giant` build-command line were
  both fixed/removed.
