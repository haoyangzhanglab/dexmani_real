"""Shared joint-limit hierarchy validation for the XHand device.

The XHand has a three-level joint-limit hierarchy::

    rated (hardware hard-stop envelope) ⊇ mechanical (model envelope) ⊇ command (operational)

Runtime config may narrow but never widen the rated envelope, and command
bounds must stay inside mechanical bounds.  This single helper keeps the three
layers from drifting apart.
"""

from __future__ import annotations

import numpy as np

from dexmani_real.robot_spec import HAND_JOINT_SHAPE

# Learned policy checkpoints run their forward/normalizer arithmetic in
# float32, while the command path intentionally stores float64.  This narrow
# allowance is only for canonicalizing that representation roundoff to the
# operational endpoint; it is never applied to the mechanical/rated envelope.
POLICY_HAND_ENDPOINT_ROUNDOFF_TOLERANCE_RAD = 1e-6


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
        raise ValueError(
            f"{label} mechanical limits cannot exceed the rated device envelope"
        )
    if np.any(bounds["command_lower"] < bounds["mechanical_lower"]) or np.any(
        bounds["command_upper"] > bounds["mechanical_upper"]
    ):
        raise ValueError(f"{label} command limits must be inside mechanical limits")


def validate_hand_command_bounds(
    hand_cmd: object,
    operational_lower: object,
    operational_upper: object,
    mechanical_lower: object,
    mechanical_upper: object,
    rated_lower: object,
    rated_upper: object,
) -> np.ndarray:
    """Reject one hand endpoint outside its nested command/mechanical limits."""
    command = np.asarray(hand_cmd, dtype=np.float64)
    op_lower = np.asarray(operational_lower, dtype=np.float64)
    op_upper = np.asarray(operational_upper, dtype=np.float64)
    mech_lower = np.asarray(mechanical_lower, dtype=np.float64)
    mech_upper = np.asarray(mechanical_upper, dtype=np.float64)
    rated_low = np.asarray(rated_lower, dtype=np.float64)
    rated_high = np.asarray(rated_upper, dtype=np.float64)
    if command.shape != HAND_JOINT_SHAPE:
        raise ValueError(
            f"hand command must have shape {HAND_JOINT_SHAPE}, got {command.shape}"
        )
    if not np.all(np.isfinite(command)):
        raise ValueError("hand command must be finite")
    validate_hand_limit_nesting(
        op_lower,
        op_upper,
        mech_lower,
        mech_upper,
        rated_low,
        rated_high,
        label="hand command",
    )
    if np.any(command < op_lower - 1e-12) or np.any(command > op_upper + 1e-12):
        raise ValueError("hand command violates operational joint limits")
    if np.any(command < mech_lower - 1e-12) or np.any(command > mech_upper + 1e-12):
        raise ValueError("hand command violates rated mechanical joint limits")
    return command.copy()


def canonicalize_policy_hand_endpoint_roundoff(
    hand_cmd: object,
    operational_lower: object,
    operational_upper: object,
    mechanical_lower: object,
    mechanical_upper: object,
    rated_lower: object,
    rated_upper: object,
) -> tuple[np.ndarray, bool]:
    """Canonicalize only a learned policy's tiny operational-bound roundoff.

    The learned-policy runtime converts float32 network outputs to float64
    before its safety checks.  A value one or a few float32 ULPs beyond a
    *narrower operational* limit is not a meaningful physical command.  This
    helper maps it to the exact operational boundary, but first keeps the
    mechanical/rated envelope strict.  Manual, teleoperation, and replay paths
    remain reject-only through :func:`validate_hand_command_bounds`.
    """
    command = np.asarray(hand_cmd, dtype=np.float64)
    op_lower = np.asarray(operational_lower, dtype=np.float64)
    op_upper = np.asarray(operational_upper, dtype=np.float64)
    mech_lower = np.asarray(mechanical_lower, dtype=np.float64)
    mech_upper = np.asarray(mechanical_upper, dtype=np.float64)
    rated_low = np.asarray(rated_lower, dtype=np.float64)
    rated_high = np.asarray(rated_upper, dtype=np.float64)
    if command.shape != HAND_JOINT_SHAPE:
        raise ValueError(
            f"hand policy endpoint must have shape {HAND_JOINT_SHAPE}, got {command.shape}"
        )
    if not np.all(np.isfinite(command)):
        raise ValueError("hand policy endpoint must be finite")
    validate_hand_limit_nesting(
        op_lower,
        op_upper,
        mech_lower,
        mech_upper,
        rated_low,
        rated_high,
        label="hand policy endpoint",
    )
    if np.any(command < mech_lower - 1e-12) or np.any(command > mech_upper + 1e-12):
        raise ValueError("hand policy endpoint violates rated mechanical joint limits")
    below = np.flatnonzero(
        command < op_lower - POLICY_HAND_ENDPOINT_ROUNDOFF_TOLERANCE_RAD
    )
    above = np.flatnonzero(
        command > op_upper + POLICY_HAND_ENDPOINT_ROUNDOFF_TOLERANCE_RAD
    )
    if below.size or above.size:
        detail: list[str] = []
        detail.extend(
            (
                f"j{index}: target={command[index]:.17g}, "
                f"lower={op_lower[index]:.17g}, "
                f"delta={command[index] - op_lower[index]:+.3e}"
            )
            for index in below
        )
        detail.extend(
            (
                f"j{index}: target={command[index]:.17g}, "
                f"upper={op_upper[index]:.17g}, "
                f"delta={command[index] - op_upper[index]:+.3e}"
            )
            for index in above
        )
        raise ValueError(
            "hand policy endpoint violates operational joint limits: "
            + ", ".join(detail)
        )
    canonical = np.clip(command, op_lower, op_upper)
    return canonical, not np.array_equal(canonical, command)
