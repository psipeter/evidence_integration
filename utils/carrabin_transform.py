"""
utils/carrabin_transform.py

Backwards-compatibility shim. The transform logic has moved to
utils/binary_transform.py, which handles carrabin, soltani_numbers,
and soltani_colors datasets under a unified interface.

This module re-exports apply_carrabin_transform as a thin wrapper so
existing code that imports from here continues to work unchanged.
"""
from utils.binary_transform import apply_binary_transform


def apply_carrabin_transform(df, dataset):
    """Wrapper around apply_binary_transform for backwards compatibility."""
    return apply_binary_transform(df, dataset)
