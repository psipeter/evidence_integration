"""
Generate stimulus sequences for the evidence integration task.

Structure:
  - 100 trials, each with 15 observations
  - Each trial has a randomly chosen mean (30–79) and std (5–15)
  - Observations drawn from Normal(mean, std), rounded to integers,
    clipped to [10, 99] — redraw any out-of-range values
  - Output: sequences.pkl (DataFrame) and sequences.json (for jsPsych)

DataFrame columns:
  trial        : int, 0-indexed trial number
  observation  : int, 0-indexed observation within trial (0–14)
  value        : int, stimulus value shown to participant
  true_mean    : float, the generative mean for that trial
  true_std     : float, the generative std for that trial
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_TRIALS      = 100
N_OBS         = 15
VALUE_MIN     = 10
VALUE_MAX     = 99
MEAN_RANGE    = (30, 79)   # generative mean uniformly drawn from this range
STD_RANGE     = (5, 15)    # generative std uniformly drawn from this range
SEED          = 42
MAX_REDRAWS   = 1000       # safety limit per observation

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Generate sequences
# ---------------------------------------------------------------------------
records = []

for trial in range(N_TRIALS):
    true_mean = rng.uniform(*MEAN_RANGE)
    true_std  = rng.uniform(*STD_RANGE)

    observations = []
    for obs in range(N_OBS):
        # Redraw until value is in [VALUE_MIN, VALUE_MAX]
        for _ in range(MAX_REDRAWS):
            v = int(np.round(rng.normal(true_mean, true_std)))
            if VALUE_MIN <= v <= VALUE_MAX:
                break
        else:
            # Fallback: clip (should never happen with chosen ranges)
            v = int(np.clip(np.round(rng.normal(true_mean, true_std)),
                            VALUE_MIN, VALUE_MAX))

        observations.append(v)
        records.append({
            'trial':       trial,
            'observation': obs,
            'value':       v,
            'true_mean':   round(true_mean, 4),
            'true_std':    round(true_std, 4),
        })

df = pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
assert len(df) == N_TRIALS * N_OBS, "Row count mismatch"
assert df['value'].between(VALUE_MIN, VALUE_MAX).all(), "Out-of-range values"
print(f"Generated {N_TRIALS} trials × {N_OBS} observations = {len(df)} rows")
print(f"Value range: {df['value'].min()} – {df['value'].max()}")
print(f"Mean of true_means: {df['true_mean'].mean():.1f}")
print(f"Mean of true_stds:  {df['true_std'].mean():.1f}")
print()
print(df.head(20).to_string(index=False))

# ---------------------------------------------------------------------------
# Save as .pkl
# ---------------------------------------------------------------------------
out_dir = Path(__file__).parent / 'sequences'
out_dir.mkdir(exist_ok=True)

pkl_path = out_dir / 'sequences.pkl'
df.to_pickle(pkl_path)
print(f"\nSaved: {pkl_path}")

# ---------------------------------------------------------------------------
# Also save as JSON for jsPsych (nested: list of trials, each a list of values)
# ---------------------------------------------------------------------------
# Format: [ { trial, true_mean, true_std, values: [v0, v1, ...] }, ... ]
json_data = []
for trial in range(N_TRIALS):
    trial_df = df[df['trial'] == trial].sort_values('observation')
    json_data.append({
        'trial':     trial,
        'true_mean': float(trial_df['true_mean'].iloc[0]),
        'true_std':  float(trial_df['true_std'].iloc[0]),
        'values':    trial_df['value'].tolist(),
    })

json_path = out_dir / 'sequences.json'
with open(json_path, 'w') as f:
    json.dump(json_data, f, indent=2)
print(f"Saved: {json_path}")
