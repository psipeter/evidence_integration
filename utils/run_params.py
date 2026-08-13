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
from utils.paths import dataset_stem, resolve_run_folder


def trial_seed(base_seed: int, trial_number: int) -> int:
    """Derive a reproducible per-trial seed from base_seed and trial."""
    return abs(hash((int(base_seed), int(trial_number)))) % (2**31)


def load_run_params(
    pid: int,
    dataset: str,
    model_type: str,
    run_folder: str | Path,
    datafile: str | None = None,
) -> dict:
    """
    Load best-fit params for one pid from a run folder, merge with
    MODEL_PARAMS fixed values and PARAM_DEFAULTS.

    Returns a fully-populated params dict ready to pass to NEF.run()
    or math_models.run().

    `datafile` is the data-version suffix (see utils.paths.dataset_stem).
    Fitted params live in {model_type}_{stem}_{pid}_params.pkl, so it is needed
    to locate the file at all. Defaults to None, which reproduces the previous
    unsuffixed behaviour exactly -- existing carrabin and yoo run folders are
    unaffected.

    If not passed explicitly, it falls back to the `datafile` value stored
    INSIDE the params pkl (fitting.fit records it as a column). That lets
    downstream callers keep working without plumbing the suffix through, while
    an explicit argument still wins. Note MODEL_PARAMS is keyed on the dataset
    FAMILY, so `dataset` -- never the stem -- is used for the fixed-param lookup.
    """
    from models.NEF import PARAM_DEFAULTS

    run_folder = resolve_run_folder(run_folder)
    stem = dataset_stem(dataset, datafile)
    params_path = run_folder / f"{model_type}_{stem}_{pid}_params.pkl"
    params = pd.read_pickle(params_path).iloc[0].to_dict()
    fixed = MODEL_PARAMS.get(dataset, {}).get(model_type, {}).get("fixed", {})
    merged = {**PARAM_DEFAULTS, **fixed, **params}
    merged["dataset"] = dataset
    merged["datafile"] = (
        datafile if datafile is not None else params.get("datafile")
    )
    merged["model_type"] = model_type
    merged["pid"] = int(pid)
    if "recurrent" in model_type:
        merged["nef_type"] = "recurrent"
    elif "synaptic" in model_type:
        merged["nef_type"] = "synaptic"
    return merged
