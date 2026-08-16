"""Shared joint-limit hierarchy validation for the XHand device.

The XHand has a three-level joint-limit hierarchy::

    rated (hardware hard-stop envelope) ⊇ mechanical (model envelope) ⊇ command (operational)

Runtime config may narrow but never widen the rated envelope, and command
bounds must stay inside mechanical bounds.  This single helper keeps the three
layers from drifting apart.
"""

from __future__ import annotations

import numpy as np

from dexmani_real.utils.schema import HAND_JOINT_SHAPE


def validate_hand_limit_nesting(
    command_lower: object,
    command_upper: object,
    mechanical_lower: object,
    mechanical_upper: object,
    rated_lower: object,
    rated_upper: object,
    *,
    label: str = "hand",
) -> None:
    """Validate ``rated ⊇ mechanical ⊇ command`` for the 12-DoF hand.

    Raises ``ValueError`` on any shape, finiteness, ordering, or nesting
    violation.  ``label`` prefixes the error message so callers retain their
    subsystem identity in diagnostics.
    """
    bounds = {
        "command_lower": np.asarray(command_lower, dtype=np.float64),
        "command_upper": np.asarray(command_upper, dtype=np.float64),
        "mechanical_lower": np.asarray(mechanical_lower, dtype=np.float64),
        "mechanical_upper": np.asarray(mechanical_upper, dtype=np.float64),
        "rated_lower": np.asarray(rated_lower, dtype=np.float64),
        "rated_upper": np.asarray(rated_upper, dtype=np.float64),
    }
    for name, array in bounds.items():
        if array.shape != HAND_JOINT_SHAPE:
            raise ValueError(f"{label} {name} must have shape {HAND_JOINT_SHAPE}")
    if not np.all(np.isfinite(np.concatenate(tuple(bounds.values())))):
        raise ValueError(f"{label} joint limits must be finite")
    if np.any(bounds["command_lower"] > bounds["command_upper"]):
        raise ValueError(f"{label} command limits must be ordered")
    if np.any(bounds["mechanical_lower"] > bounds["mechanical_upper"]):
        raise ValueError(f"{label} mechanical limits must be ordered")
    if np.any(bounds["mechanical_lower"] < bounds["rated_lower"]) or np.any(
        bounds["mechanical_upper"] > bounds["rated_upper"]
    ):
        raise ValueError(f"{label} mechanical limits cannot exceed the rated device envelope")
    if np.any(bounds["command_lower"] < bounds["mechanical_lower"]) or np.any(
        bounds["command_upper"] > bounds["mechanical_upper"]
    ):
        raise ValueError(f"{label} command limits must be inside mechanical limits")
