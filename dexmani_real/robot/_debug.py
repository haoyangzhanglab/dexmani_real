"""Shared debug helpers for robot hardware drivers."""

from __future__ import annotations

from typing import Any

import numpy as np


def print_state(state: dict[str, Any]) -> None:
    """Pretty-print a state dictionary, rounding numpy arrays for readability."""
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape={value.shape}, value={np.round(value, 6)}")
        else:
            print(f"{key}: {value}")
