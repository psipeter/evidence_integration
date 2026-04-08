"""
Filesystem layout for this project: all paths are ``pathlib.Path`` values
anchored at the project root.

This module is the single source of truth for those paths. Any code that
reads or writes files should import constants and helpers from here instead
of building ``Path`` objects or string paths locally, so layout stays
consistent and easy to change in one place.
"""

from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT: Path = _THIS_DIR.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
FITTING_DIR: Path = PROJECT_ROOT / "fitting"
ANALYSIS_DIR: Path = PROJECT_ROOT / "analysis"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
JOBS_DIR: Path = PROJECT_ROOT / "jobs"


def data_path(filename: str) -> Path:
    """Return ``DATA_DIR / filename``."""
    return DATA_DIR / filename
