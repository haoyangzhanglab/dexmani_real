"""Pre-send validation — centralized safety gate for teleop actions.

Extracted from RobotInterface.validate_action() to remove the circular
dependency on teleop.control.safety (now deleted).

Ref: ManiUniCon validate_action pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Protocol

import numpy as np

from dexmani_real.robot.types import _ARM_TORQUE_LIMIT_NM, RobotAction
from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    pass

# Per-finger tactile force limit (Newtons, L2 norm across fx/fy/fz).
# Set generously (30 N) to avoid false positives during normal teleop —
# typical grasping forces are 1–6 N; this gate only catches genuinely
# excessive forces (e.g., crushing, collision with rigid environment).
# Individual fingers can be lowered via this array.
_HAND_TACTILE_FORCE_LIMIT_N = np.full(5, 30.0, dtype=np.float64)

logger = get_logger(__name__)


# ── Protocol for static type-checking on the safety-critical robot parameter ──


class SupportsValidation(Protocol):
    """Minimal interface for the ``robot`` parameter of :func:`validate_action`.

    Avoids circular import of ``RobotInterface`` while giving mypy enough
    information to catch typos in attribute access.
    """

    arm: Any  # duck-typed: .is_error(), .is_connected()
    hand: Any  # duck-typed: .connected_flag, .error_state


# ── Individual safety gates (independently testable) ──


def _check_hardware_error(robot: SupportsValidation) -> Optional[str]:
    """Reject if arm or hand reports an SDK error state.

    Hand errors gate only when the hand is connected (``connected_flag``),
    so arm-only / degraded mode is not blocked by an absent hand reporting
    ``is_error() == True`` (xhand.py:395).
    """
    if robot.arm.is_error():
        return "arm error state"
    if robot.hand.connected_flag and robot.hand.error_state:
        return "hand error state"
    return None


def _check_arm_connected(robot: SupportsValidation) -> Optional[str]:
    """Reject if arm is not connected."""
    if not robot.arm.is_connected():
        return "arm not connected"
    return None


def _check_action_nan(action: RobotAction) -> Optional[str]:
    """Reject if action contains NaN/Inf joint commands."""
    if not np.all(np.isfinite(action.arm_qpos_cmd)):
        return "action arm_qpos_cmd NaN/Inf"
    if not np.all(np.isfinite(action.hand_qpos_cmd)):
        return "action hand_qpos_cmd NaN/Inf"
    return None


def _validate_sensor_array(
    data: np.ndarray | None, expected_shape: tuple[int, ...], name: str
) -> Optional[str]:
    """Shared shape+finite check for optional sensor arrays.

    Returns a rejection reason string, or ``None`` if the data is valid.
    ``data is None`` is treated as "skip" (returns ``None``).
    """
    if data is None:
        return None
    arr = np.asarray(data, dtype=np.float64)
    if not (arr.shape == expected_shape and np.all(np.isfinite(arr))):
        return f"{name} data invalid"
    return None


def _check_torque(actual_arm_tau: np.ndarray | None) -> Optional[str]:
    """Reject if any joint torque exceeds per-joint limits."""
    if actual_arm_tau is None:
        return None
    reason = _validate_sensor_array(actual_arm_tau, (7,), "torque")
    if reason is not None:
        return reason
    tau = np.asarray(actual_arm_tau, dtype=np.float64)
    over_idx = np.where(np.abs(tau) > _ARM_TORQUE_LIMIT_NM)[0]
    if len(over_idx) > 0:
        return f"torque limit exceeded: joints={over_idx.tolist()}"
    return None


def _check_velocity_nan(actual_arm_qvel: np.ndarray | None) -> Optional[str]:
    """Reject if arm velocity data is NaN/Inf or wrong shape."""
    if actual_arm_qvel is None:
        return None
    reason = _validate_sensor_array(actual_arm_qvel, (7,), "arm velocity")
    if reason is not None:
        return reason
    return None


def _check_tactile_force(
    actual_hand_tactile_sum: np.ndarray | None,
    hand_connected: bool,
    hand_is_error: bool,
) -> Optional[str]:
    """Reject if any finger force exceeds the tactile limit.

    Gated behind ``hand_connected`` so arm-only sessions are unaffected.
    """
    if actual_hand_tactile_sum is None or not hand_connected or hand_is_error:
        return None
    reason = _validate_sensor_array(actual_hand_tactile_sum, (5, 3), "tactile force")
    if reason is not None:
        logger.warning("Tactile force data shape mismatch or contains NaN — skipping gate")
        return None  # fail-open: skip gate on bad data (asymmetric with torque)
    force = np.asarray(actual_hand_tactile_sum, dtype=np.float64)
    force_mag = np.linalg.norm(force, axis=1)  # (5,) per-finger L2 norm
    over_idx = np.where(force_mag > _HAND_TACTILE_FORCE_LIMIT_N)[0]
    if len(over_idx) > 0:
        return f"tactile force limit exceeded: fingers={over_idx.tolist()}"
    return None


# ── Public API ──


def validate_action(
    robot: SupportsValidation,
    action: RobotAction,
    *,
    actual_arm_qvel: np.ndarray | None = None,
    actual_arm_tau: np.ndarray | None = None,
    actual_hand_tactile_sum: np.ndarray | None = None,
) -> tuple[bool, str]:
    """Centralized pre-send validation.

    Checks (fail-fast order):
      1. SDK error state (arm + hand is_error)
      2. Arm connection
      3. Action NaN guard (arm + hand joint commands)
      4. Torque gating — per-joint threshold check
      5. Arm velocity NaN guard
      6. Tactile force gating — per-finger threshold check

    Removed from this gate (covered elsewhere):
      - Workspace clamp → TeleopPipeline Stage 3 (before IK)
      - Hand joint-limit clip → XHand.send_action() internal _limit_joint_range
      - Self-collision check → IK Stage (_check_teleop_collision_gate) + firmware error 22
      - Env collision check → IK Stage (_check_teleop_collision_gate)
      - Hand current gating → firmware tor_max (320mA, xhand.py:180) is primary
        protection; software gate was disabled due to J3 false positives.

    All sensor parameters are optional — ``None`` skips the check
    (backward compatible).

    Freshness contract: ``get_state()`` must be called before
    ``validate_action()`` in the control loop.  HandSHMAdapter caches
    ``connected_flag``/``error_state`` from the SHM ring; ``is_connected()``
    /``is_error()`` are cheap local-flag queries that do NOT themselves
    refresh.
    """
    for gate in (
        lambda: _check_hardware_error(robot),
        lambda: _check_arm_connected(robot),
        lambda: _check_action_nan(action),
        lambda: _check_torque(actual_arm_tau),
        lambda: _check_velocity_nan(actual_arm_qvel),
        lambda: _check_tactile_force(
            actual_hand_tactile_sum, robot.hand.connected_flag, robot.hand.error_state
        ),
    ):
        reason = gate()
        if reason is not None:
            return False, reason

    return True, "ok"
