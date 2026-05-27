# Diederen dataset — archival notes

Archived after analysis revealed:
- Only 3 sessions per pid, ~28 obs per distribution per session
- EV_A = -EV_B always: carryover analyses not interpretable
- Per-pid power law fits significant in only 38-42% of individual fits
- All models perform equivalently (RMSE differences < 0.02, within noise)
- NEF2d architecture validated but insufficient data for meaningful
  model comparison or individual-level parameter estimation

Raw data retained at data/diederen.pkl.
A new experiment with better design is planned.

Second-pass archival moves:
- scripts/compare_nef2d_sweep.py -> archive/scripts/compare_nef2d_sweep.py
- scripts/build_diederen.py -> archive/scripts/build_diederen.py
- jobs/submit_nef2d_sweep.sh -> archive/jobs/submit_nef2d_sweep.sh
- models/math_models.py diederen sections -> archive/misc/math_models_diederen.py
