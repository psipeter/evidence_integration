"""
Filesystem layout for this project: all paths are ``pathlib.Path`` values
anchored at the project root.

This module is the single source of truth for those paths. Any code that
reads or writes files should import constants and helpers from here instead
of building ``Path`` objects or string paths locally, so layout stays
consistent and easy to change in one place.

The project root can be overridden via the ``EVIDENCE_INTEGRATION_ROOT``
environment variable — useful on HPC clusters where symlinks may cause
``Path(__file__).resolve()`` to return a different path than ``pwd``.
"""

import os
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ENV_ROOT = os.environ.get("EVIDENCE_INTEGRATION_ROOT")
PROJECT_ROOT: Path = Path(_ENV_ROOT).resolve() if _ENV_ROOT else _THIS_DIR.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RUNS_DIR: Path = DATA_DIR / "runs"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"


def resolve_run_folder(run_folder: Path | str) -> Path:
    """
    Return the run-folder path under ``RUNS_DIR`` (or absolute), creating it if needed.

    Relative ``test_local`` → ``RUNS_DIR / test_local``.
    """
    path = (
        RUNS_DIR / run_folder
        if not Path(run_folder).is_absolute()
        else Path(run_folder)
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_path(filename: str) -> Path:
    """Return ``DATA_DIR / filename``."""
    return DATA_DIR / filename


def dataset_stem(dataset: str, datafile: str | None = None) -> str:
    """Combine a dataset FAMILY name with an optional data-version suffix.

    ``dataset`` is the family key that selects model behaviour -- it indexes
    ``fitting.model_params.MODEL_PARAMS``, picks the branch in
    ``models.math_models``, keys the response transforms in
    ``utils.binary_transform``, and names the NEF counting-activity files.
    ``datafile`` selects WHICH BUILD of that family's human data to use.

    Keeping them separate means a new round of data needs no new model
    plumbing (no MODEL_PARAMS entry, no math_models branch, no regenerated
    counting-activity file) -- only a new pkl:

        dataset_stem("soltani_numbers")           -> "soltani_numbers"
        dataset_stem("soltani_numbers", "pilot5") -> "soltani_numbers_pilot5"

    Used for BOTH the input pkl (``data/{stem}.pkl``) and every fit output
    (``{model_type}_{stem}_{pid}_*.pkl``), so a fit can never be silently
    paired with human data it wasn't fit against. That pairing failure was
    real, not hypothetical: ``data/runs/soltani_math_v1`` held fits made
    against JATOS-era data under the same unsuffixed dataset name as a later,
    different pkl, with non-corresponding pids -- the figures merged them on
    ``pid`` and plotted different people's fits against each other's data.

    Mirrors the ``--datafile`` suffix convention the figure scripts already
    use, deliberately: it is a plain filename suffix, not a pilot-specific
    concept, so it works unchanged for a production dataset.
    """
    return f"{dataset}_{datafile}" if datafile else dataset
