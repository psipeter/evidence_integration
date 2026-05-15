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
RUNS_DIR: Path = DATA_DIR / "runs"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"


def resolve_run_folder(run_folder: str | Path) -> Path:
    """
    Return an absolute run-folder path without duplicating ``data/runs``.

    Relative ``test_local`` → ``RUNS_DIR / test_local``. Relative
    ``data/runs/test_local`` (from project root) → that directory, not
    ``RUNS_DIR / data/runs/test_local``.
    """
    p = Path(run_folder)
    if p.is_absolute():
        return p.resolve()
    anchored = (PROJECT_ROOT / p).resolve()
    runs_resolved = RUNS_DIR.resolve()
    try:
        anchored.relative_to(runs_resolved)
        return anchored
    except ValueError:
        pass
    return (runs_resolved / p).resolve()


def data_path(filename: str) -> Path:
    """Return ``DATA_DIR / filename``."""
    return DATA_DIR / filename
