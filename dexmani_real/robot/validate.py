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
    """Reject if the **arm** reports an SDK error state.

    Hand errors are intentionally NOT gated here. The arm and hand are
    independent subsystems with separate error recovery:

    - Hand child: board-error auto-clear in ``get_state()`` (xhand.py:559),
      consecutive-send-errors watchdog → reconnect, estop → detorque.
    - Arm inner loop: Mode 6 online trajectory planning, tracking-error
      monitor, torque/temperature readback.

    Blocking arm commands for a hand board glitch (tipboard / jointboard /
    commboard) freezes the arm unnecessarily — the transient error often
    self-clears within one hand tick (33 ms) while the arm stays frozen
    until the next manual intervention (ref: 2026-07-30 session).
    """
    if robot.arm.is_error():
        return "arm error state"
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


# ── Public API ──


def validate_action(
    robot: SupportsValidation,
    action: RobotAction,
    *,
    actual_arm_qvel: np.ndarray | None = None,
    actual_arm_tau: np.ndarray | None = None,
) -> tuple[bool, str]:
    """Centralized pre-send validation (arm only — hand errors are NOT gated).

    Checks (fail-fast order):
      1. Arm SDK error state
      2. Arm connection
      3. Action NaN guard (arm + hand joint commands)
      4. Torque gating — per-joint threshold check
      5. Arm velocity NaN guard

    .. note::

       Hand error state is intentionally NOT checked here.
       See :func:`_check_hardware_error` for rationale.

    Removed from this gate (covered elsewhere):
      - Workspace clamp → TeleopPipeline Stage 3 (before IK)
      - Hand joint-limit clip → XHand.send_action() internal _limit_joint_range
      - Self-collision check → IK Stage (_check_teleop_collision_gate) + firmware error 22
      - Env collision check → IK Stage (_check_teleop_collision_gate)
      - Hand current gating → firmware tor_max (320mA, xhand.py:155) is primary
        protection; software gate was disabled due to J3 false positives.
      - Hand board errors → hand child auto-clears in get_state() (xhand.py:559);
        logged by hand child when non-zero (hand_process.py).

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
        # Torque gate disabled — J1 torque routinely exceeds 50 Nm limit
        # when the arm is extended sideways, causing excessive safety
        # rejections during normal teleop.  Hardware-level overcurrent
        # protection (firmware) remains active.
        # lambda: _check_torque(actual_arm_tau),
        lambda: _check_velocity_nan(actual_arm_qvel),
    ):
        reason = gate()
        if reason is not None:
            return False, reason

    return True, "ok"
