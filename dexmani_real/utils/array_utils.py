"""Array construction utilities."""

from __future__ import annotations

import numpy as np

__all__ = ["nan_array"]


def nan_array(shape: int | tuple[int, ...], dtype: type = np.float64) -> np.ndarray:
    """Create an array filled with NaN.

    Centralized factory for the ``np.full(shape, np.nan, dtype=np.float64)``
    pattern repeated across the codebase.  Ensures consistent dtype and NaN fill.
    """
    return np.full(shape, np.nan, dtype=dtype)
