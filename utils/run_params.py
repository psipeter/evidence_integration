"""
Shared utility for loading fitted parameters from a run folder,
merging with MODEL_PARAMS fixed values and NEF PARAM_DEFAULTS.

Used by save_responses, save_activities, dynamics_NEF, and
iti_perturbation to avoid duplicating the same loading pattern.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fitting.model_params import MODEL_PARAMS
from utils.paths import resolve_run_folder


def trial_seed(base_seed: int, trial_number: int) -> int:
    """Derive a reproducible per-trial seed from base_seed and trial."""
    return abs(hash((int(base_seed), int(trial_number)))) % (2**31)


def load_run_params(
    pid: int,
    dataset: str,
    model_type: str,
    run_folder: str | Path,
) -> dict:
    """
    Load best-fit params for one pid from a run folder, merge with
    MODEL_PARAMS fixed values and PARAM_DEFAULTS.

    Returns a fully-populated params dict ready to pass to NEF.run()
    or math_models.run().
    """
    from models.NEF import PARAM_DEFAULTS

    run_folder = resolve_run_folder(run_folder)
    params_path = run_folder / f"{model_type}_{dataset}_{pid}_params.pkl"
    params = pd.read_pickle(params_path).iloc[0].to_dict()
    fixed = MODEL_PARAMS.get(dataset, {}).get(model_type, {}).get("fixed", {})
    merged = {**PARAM_DEFAULTS, **fixed, **params}
    merged["dataset"] = dataset
    merged["model_type"] = model_type
    merged["pid"] = int(pid)
    if "recurrent" in model_type:
        merged["nef_type"] = "recurrent"
    elif "synaptic" in model_type:
        merged["nef_type"] = "synaptic"
    return merged
