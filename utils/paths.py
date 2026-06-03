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
