# Evidence Integration

## Scientific overview

This project studies individual differences in sequential evidence integration,
using cognitive models and a biophysical spiking neural network (NEF) to
identify computational and neural mechanisms.

Four goals:
1. Honest comparative benchmarking (NEF competitive with cognitive models on RMSE)
2. Cross-task generalisability (same NEF architecture across tasks)
3. Breadth of predictions (temporal patterns, noise, state persistence, individual differences)
4. Novel testable predictions (spiking noise = state-persistent variability, distinguishable
   from response noise even when RMSE is matched)

Central model: alpha(t) = alpha_0 / t^lambda (power-law decaying learning rate).
In the NEF this emerges from spiking dynamics rather than being hardcoded.

---

## Tasks

| Name | N | Key features |
|------|---|-------------|
| carrabin | 21 | Binary inputs; 5 obs/trial; sequences repeat (qid); true_p known |
| yoo | 38 | Continuous inputs; 30 obs/trial; no sequence repetition |

Proposed new task: repeated sequences (carrabin) + long sequences + continuous
values (yoo). Unlocks all PVTBN metrics simultaneously.

---

## Metric taxonomy (PVTBN)

One figure per group per task (combinable if needed). Figures save PDF only.

### P — Performance
| Code | Metric |
|------|--------|
| P1 | Estimation error: RMSE to hidden probability (true_p); human + all models |
| P2 | Model fit: RMSE to human responses; model comparison |

### V — Variance
| Code | Metric |
|------|--------|
| V1 | Distributional fit (MLE): model captures full response distribution |
| V2 | Response variability for identical inputs: std(response|obs,qid) per pid |
| V3 | Test-retest reliability: variability stable across session halves (r=0.88****) |
| V4 | State vs response noise: T3/T4 patterns, NEF vs NoisyCounting comparison |

### T — Temporal
| Code | Metric |
|------|--------|
| T1 | Task performance vs observation position |
| T2 | Response change vs observation: update magnitude pattern |
| T3 | Residual variance growth across obs (state noise accumulation) |
| T4 | Within-trial residual autocorrelation decay (state persistence) |

### B — Bias
| Code | Metric |
|------|--------|
| B1 | Weight profile: flat/primacy/recency/U-shaped temporal weighting |
| B2 | Surprise sensitivity and confirmation bias |

### N — Neural (NEF predictions; testable in future experiments)
| Code | Metric |
|------|--------|
| N1 | Decoded PE timecourse within observation window |
| N2 | Response and PE variability vs n_neurons |
| N3 | State persistence from spiking noise (mechanistic account of T3/T4) |
| N4 | (Future) Neural population geometry |

---

## Models

| Dataset | Model | Role | Free params |
|---------|-------|------|-------------|
| carrabin | Mean | Optimal baseline | none |
| carrabin | LeakyIntegrator | Leaky integrator | gamma |
| carrabin | PrimacyRecency | Temporal weighting | eps_p, eps_r |
| carrabin | NoisyCounting | Task-specific (Prat-Carrabin 2024) | mu, sigma_c, nu |
| carrabin | RL_lambda | Power-law delta rule | alpha_0, lambda_ |
| carrabin | NEF | Spiking NEF integrator | alpha_0, lambda_ |
| yoo | (same set minus NoisyCounting) | | |

NoisyCounting has two fitted versions:
- RMSE-fitted: sigma_c collapses to ~0 (response-noise artefact)
- MLE-fitted (fit_mle.py): recovers sigma_c ~0.03-0.08, nu ~0.08-0.21

---

## Current figure inventory

### figure_carrabin_performance.py (P group, 1x3)
A: task schematic | B: P1 estimation error | C: P2 model fit

### figure_carrabin_variability.py (V group, 1x3)
A: V2 KDE with per-pid lines | B: V2 RMSE regplot | C: V3 test-retest

### figure_carrabin.py (combined overview, 2x4)
A: schematic | B: P2 RMSE | C: V2 KDE | D: V2 regplot |
E: N2/N3 n_neurons scan | F: T4 autocorrelation | G: T3 variance growth | H: pending

### figure_yoo.py (combined overview, 2x4)
A: schematic | B: P2 RMSE | C: T2 response change | D: B1 weight profile |
E-F: pending | G: T1 task error by obs | H: pending

---

## Fitting pipelines

### RMSE fitting
    python -m fitting.submit carrabin NEF --n_trials 100 --run_folder carrabin
    python -m fitting.collect carrabin --type params
    python scripts/figure_carrabin.py --run_folder carrabin --extra_models NoisyCounting

### MLE fitting (NoisyCounting, carrabin)
    bash jobs/submit_mle_fit.sh NoisyCounting carrabin 500 100
    # 21 pids x 500 fits x 100 sims; ~8 min wall time; cross-pid sharing

---

## Environment

Always use: /home/psipeter/evidence_integration/venv/bin/python
Cluster: /dartfs-hpc/rc/home/n/f007qzn/
SLURM: use pwd -P and export EVIDENCE_INTEGRATION_ROOT=${ROOT}

---

## Archive

Older models/data (diederen, jiang, usher) in archive/. Do not reactivate.
