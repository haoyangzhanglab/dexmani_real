"""Cartesian jog proposals, input-bound epochs and guarded command references."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.planning.kinematics.pose import Pose, quat_multiply
from dexmani_real.robot.commands import DispatchError, DispatchResult, DispatchStatus, RobotCommand
from dexmani_real.runtime.state import RunEndReason, RuntimeState, SafetyState

if TYPE_CHECKING:
    from dexmani_real.robot.robot import DexManiRobot
    from dexmani_real.runtime.operator_input import KeyboardInput

CARTESIAN_JOG_KEYS = frozenset(
    {"w", "s", "a", "d", "up", "down", "left", "right", "i", "k", "j", "l"}
)


def any_jog_key_held(active_keys: Iterable[str]) -> bool:
    """Check physical jog keys even when opposite increments cancel out."""
    return not CARTESIAN_JOG_KEYS.isdisjoint(active_keys)


def jog_motion_requested(active_keys: Iterable[str]) -> bool:
    keys = frozenset(active_keys)
    if keys.intersection({"r", "q", "esc"}):
        return False
    dx, drpy = compute_cartesian_jog_delta(keys, 1.0, 1.0)
    return bool(np.any(dx) or np.any(drpy))


def submit_jog_target(
    keys: KeyboardInput,
    generation: int,
    runtime_state: RuntimeState,
    robot: DexManiRobot,
    target: np.ndarray,
    epoch: int,
    *,
    valid_until_ns: int,
    execute_until_ns: int,
    commit_reference: Callable[[], None],
) -> DispatchResult | None:
    """Bind input to an epoch, submit without input locks, then commit local references.

    Every jog generation change revokes the bound epoch. A late enqueue therefore
    cannot restore old authority, even if input changes while IPC is blocked.
    commit_reference must only assign already constructed local command references.
    """
    def reserve() -> int | None:
        with runtime_state.motion_lock:
            if runtime_state.run_id != epoch or not runtime_state.requests_allow_motion():
                return None
            if time.monotonic_ns() >= min(valid_until_ns, execute_until_ns):
                runtime_state.revoke_motion(expected_run_id=epoch)
                raise DispatchError(
                    "jog proposal deadline expired", DispatchResult(),
                    revoked=True, cause="deadline_expired",
                )
            if runtime_state.safety_state == SafetyState.ARMED:
                if runtime_state.begin_motion() is None:
                    return None
            elif runtime_state.safety_state != SafetyState.RUNNING:
                return None
            return runtime_state.run_id

    run_id = keys.commit_if_current(generation, reserve)
    if run_id is None:
        return None
    try:
        result = robot.send_action(
            RobotCommand(run_id, target),
            valid_until_ns=valid_until_ns,
            execute_until_ns=execute_until_ns,
        )
        if result.arm != DispatchStatus.SUBMITTED or result.hand != DispatchStatus.NOT_CALLED:
            raise DispatchError("incomplete arm-only dispatch", result)

        def finish() -> bool:
            with runtime_state.motion_lock:
                if not runtime_state.command_may_cross_sdk(run_id=run_id):
                    return False
                if time.monotonic_ns() >= min(valid_until_ns, execute_until_ns):
                    return False
                commit_reference()
                return True

        if not keys.commit_if_current(generation, finish):
            # SUBMITTED remains a fact; this only declines advancing the jog reference.
            expired = time.monotonic_ns() >= min(valid_until_ns, execute_until_ns)
            raise DispatchError(
                "jog proposal expired or revoked during submission",
                result,
                revoked=True,
                cause="deadline_expired" if expired else "authority_revoked",
            )
        return result
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            reason = RunEndReason.ESTOP
        elif isinstance(exc, DispatchError) and not exc.revoked:
            reason = RunEndReason.HARDWARE_FAULT
        else:
            reason = RunEndReason.EXECUTOR_BOUNDARY
        runtime_state.revoke_motion(reason=reason, expected_run_id=run_id)
        raise


def propose_cartesian_jog_pose(
    command_pose: Pose,
    dx: np.ndarray,
    drpy: np.ndarray,
    position_lower: np.ndarray,
    position_upper: np.ndarray,
) -> tuple[Pose, np.ndarray, bool]:
    """Project a jog proposal without advancing the dispatched target reference."""
    desired_position = command_pose.p + dx
    position = np.clip(desired_position, position_lower, position_upper)
    rotating = bool(np.any(drpy != 0.0))
    changed = rotating or not np.array_equal(position, command_pose.p)
    quaternion = command_pose.q.copy()
    if rotating:
        delta_quaternion = Rotation.from_euler("xyz", drpy).as_quat()[[3, 0, 1, 2]]
        quaternion = quat_multiply(delta_quaternion, quaternion)
    return Pose(p=position, q=quaternion), desired_position, changed


def compute_cartesian_jog_delta(
    pressed_keys: Iterable[str],
    delta_pos: float,
    delta_rpy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map held keys to EEF position and rotation increments."""
    keys = frozenset(pressed_keys)
    dx = np.zeros(3, dtype=np.float64)
    if "w" in keys:
        dx[0] += delta_pos
    if "s" in keys:
        dx[0] -= delta_pos
    if "a" in keys:
        dx[1] -= delta_pos
    if "d" in keys:
        dx[1] += delta_pos
    if "up" in keys:
        dx[2] += delta_pos
    if "down" in keys:
        dx[2] -= delta_pos
    dx_norm = float(np.linalg.norm(dx))
    if dx_norm > delta_pos:
        dx *= delta_pos / dx_norm

    drpy = np.zeros(3, dtype=np.float64)
    if "left" in keys:
        drpy[0] += delta_rpy
    if "right" in keys:
        drpy[0] -= delta_rpy
    if "i" in keys:
        drpy[1] += delta_rpy
    if "k" in keys:
        drpy[1] -= delta_rpy
    if "j" in keys:
        drpy[2] -= delta_rpy
    if "l" in keys:
        drpy[2] += delta_rpy
    drpy_norm = float(np.linalg.norm(drpy))
    if drpy_norm > delta_rpy:
        drpy *= delta_rpy / drpy_norm

    return dx, drpy
