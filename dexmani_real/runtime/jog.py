"""持续伺服会话中的笛卡尔 jog；松键减速保持，换向更新目标。"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from typing import cast

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.control.ik import IKReferenceError, IKResult
from dexmani_real.geometry.pose import Pose, quat_multiply
from dexmani_real.robot.commands import (
    OperationCancelled,
    RobotCommand,
)
from dexmani_real.runtime.ik_reference import IKReference, prepare_reference
from dexmani_real.runtime.keyboard import JOG_KEY_PAIRS, JOG_MOTION_KEYS
from dexmani_real.runtime.observation import check_device_feedback

logger = logging.getLogger(__name__)

JOG_CONTROL_KEYS = JOG_MOTION_KEYS | {
    "q",
    "r",
    "esc",
}


class CartesianJog:
    """持有提交参考；下一次增量从已提交解的 FK 出发。"""

    def __init__(self, solver, config, bounds, *, timing=None):
        self.solver = solver
        self.config = config
        self.position_lower, self.position_upper = bounds
        self.ik_reference = None
        self.rejection_reason = None
        self.timing = timing
        self.target_rejection: Callable[[np.ndarray], str | None] | None = None

    def clear_reference(self):
        self.ik_reference = None

    def step(self, robot, keys, state, runtime, *, scheduled_ns=None):
        """发布点动目标或减速保持；故障由会话边界记录、撤权并收尾。"""
        timing = self.timing
        trace_control_ns = time.monotonic_ns() if timing is not None else None
        pressed_keys = keys.snapshot() & JOG_CONTROL_KEYS
        with state.motion_lock:
            if not (state.requests_allow_motion() and not state.stop_signal.is_set()):
                raise OperationCancelled("jog authority revoked")

        row = robot.read_state()
        if timing is not None:
            timing.add(
                "control_tick", timestamp_ns=trace_control_ns, scheduled_ns=scheduled_ns,
                observation_ns=time.monotonic_ns(), arm_feedback_ns=int(row.arm["timestamp_ns"][0]),
                keys=sorted(pressed_keys),
            )
        issues = check_device_feedback(row, runtime)
        if issues:
            raise OperationCancelled("jog feedback rejected: " + "; ".join(issues))
        target = None
        rejection_reason = None
        if jog_motion_requested(pressed_keys):
            compute_ns = time.monotonic_ns()
            try:
                result, anchor = self.propose(
                    pressed_keys,
                    row.arm["qpos"][0],
                    control_ns=compute_ns,
                    trace_control_ns=trace_control_ns,
                )
            except IKReferenceError as exc:
                if timing is not None:
                    timing.add(
                        "ik", timestamp_ns=compute_ns, control_ns=trace_control_ns,
                        finished_ns=time.monotonic_ns(), rejection_reason=str(exc),
                    )
                raise OperationCancelled(str(exc)) from exc
            if result.qpos is None:
                rejection_reason = result.reason
            elif keys.snapshot() & JOG_CONTROL_KEYS == pressed_keys:
                target = result.qpos
        if rejection_reason is not None and rejection_reason != self.rejection_reason:
            logger.warning(
                "Jog target rejected; braking and holding: %s", rejection_reason
            )
        self.rejection_reason = rejection_reason
        if target is not None:
            issues = check_device_feedback(row, runtime)
            if issues:
                raise OperationCancelled(
                    "jog mapping sources expired: " + "; ".join(issues)
                )
        command = RobotCommand(arm_qpos=target, arm_hold=target is None)
        submit_start_ns = time.monotonic_ns()
        arm_published_ns = None
        try:
            arm_published_ns = robot.send_action(
                command,
                expires_ns=submit_start_ns + int(self.config.target_max_age_s * 1e9),
            )
        finally:
            if timing is not None:
                timing.add(
                    "arm_submit", timestamp_ns=submit_start_ns, control_ns=trace_control_ns,
                    finished_ns=time.monotonic_ns(), arm_published_ns=arm_published_ns,
                    goal_qpos=command.arm_qpos, hold=command.arm_hold,
                    accepted=arm_published_ns is not None, rejection_reason=rejection_reason,
                )
        if target is not None:
            self.ik_reference = IKReference(
                cast(np.ndarray, command.arm_qpos), anchor, cast(int, arm_published_ns)
            )
        else:
            self.clear_reference()

    def propose(
        self, pressed_keys, measured_qpos, *, control_ns, trace_control_ns=None
    ) -> tuple[IKResult, np.ndarray]:
        measured, command, anchor, h = prepare_reference(
            self.solver,
            self.ik_reference,
            measured_qpos,
            control_ns=control_ns,
            period_s=1 / self.config.control_hz,
        )
        baseline_pose = self.solver.compute_eef_pose_base(command)
        dx, drpy = jog_delta(
            pressed_keys, self.config.delta_pos_m, self.config.delta_rpy_rad
        )
        pose = jog_pose(
            baseline_pose,
            dx,
            drpy,
            self.position_lower,
            self.position_upper,
        )
        result = self.solver.solve(pose, command, anchor, h=h)
        if result.qpos is not None and self.target_rejection is not None:
            reason = self.target_rejection(result.qpos)
            if reason is not None:
                result = IKResult(qpos=None, reason=reason)
        if self.timing is not None:
            self.timing.add(
                "ik", timestamp_ns=control_ns, control_ns=trace_control_ns,
                finished_ns=time.monotonic_ns(), desired_position_m=pose.p, desired_quat_wxyz=pose.q,
                h_s=h, reference_qpos=command, measured_qpos=measured, goal_qpos=result.qpos,
                rejection_reason=None if result.qpos is not None else result.reason,
            )
        return result, anchor


def jog_motion_requested(active_keys: Iterable[str]) -> bool:
    keys = frozenset(active_keys)
    if keys.intersection({"r", "q", "esc"}):
        return False
    return any(
        (positive in keys) != (negative in keys) for positive, negative in JOG_KEY_PAIRS
    )


def jog_pose(
    command_pose: Pose,
    dx: np.ndarray,
    drpy: np.ndarray,
    position_lower: np.ndarray,
    position_upper: np.ndarray,
) -> Pose:
    p = command_pose.p
    # 允许从内缩范围外逐步退回；无平移输入的轴保持原位。
    lower = np.minimum(position_lower, p)
    upper = np.maximum(position_upper, p)
    position = np.clip(p + dx, lower, upper)
    rotating = bool(np.any(drpy != 0.0))
    quaternion = command_pose.q
    if rotating:
        delta_quaternion = Rotation.from_euler("xyz", drpy).as_quat()[[3, 0, 1, 2]]
        quaternion = quat_multiply(delta_quaternion, quaternion)
    return Pose(p=position, q=quaternion)


def jog_delta(
    pressed_keys: Iterable[str],
    delta_pos: float,
    delta_rpy: float,
) -> tuple[np.ndarray, np.ndarray]:
    keys = frozenset(pressed_keys)
    directions = np.array(
        [
            int(positive in keys) - int(negative in keys)
            for positive, negative in JOG_KEY_PAIRS
        ],
        dtype=np.float64,
    )
    dx, drpy = directions[:3], directions[3:]
    dx *= delta_pos / max(1.0, float(np.linalg.norm(dx)))
    drpy *= delta_rpy / max(1.0, float(np.linalg.norm(drpy)))
    return dx, drpy
