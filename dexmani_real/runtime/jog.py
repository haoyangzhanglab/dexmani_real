"""持续伺服会话中的笛卡尔 jog；松键减速保持，换向更新目标。"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import cast

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.control.constraints import workspace_start_rejection
from dexmani_real.control.ik import IKReferenceError, IKResult
from dexmani_real.geometry.pose import Pose, quat_multiply
from dexmani_real.robot.commands import (
    OperationCancelled,
    RobotCommand,
    arm_reference_from_feedback,
)
from dexmani_real.runtime.ik_reference import IKReference, prepare_reference
from dexmani_real.runtime.keyboard import JOG_KEY_PAIRS
from dexmani_real.runtime.observation import (
    check_device_feedback,
    read_observation,
)

logger = logging.getLogger(__name__)

JOG_CONTROL_KEYS = frozenset(key for pair in JOG_KEY_PAIRS for key in pair) | {
    "q",
    "r",
    "esc",
}


class CartesianJog:
    """持有提交参考；下一次增量从已提交解的 FK 出发。"""

    def __init__(self, solver, config, bounds):
        self.solver = solver
        self.config = config
        self.position_lower, self.position_upper = bounds
        self.ik_reference = None
        self.rejection_reason = None

    def clear_reference(self):
        self.ik_reference = None

    def step(self, robot, keys, state, runtime):
        """发布点动目标或减速保持；故障由会话边界记录、撤权并收尾。"""
        pressed_keys = keys.snapshot() & JOG_CONTROL_KEYS
        with state.motion_lock:
            if not (state.requests_allow_motion() and not state.stop_signal.is_set()):
                raise OperationCancelled("jog authority revoked")

        row = read_observation(None, robot)
        issues = check_device_feedback(row, runtime)
        if issues:
            raise OperationCancelled("jog feedback rejected: " + "; ".join(issues))
        target = None
        rejection_reason = None
        if jog_motion_requested(pressed_keys):
            try:
                result = self.propose(
                    pressed_keys,
                    row.arm["qpos"][0],
                    feedback_ns=int(row.arm["timestamp_ns"][0]),
                    control_ns=time.monotonic_ns(),
                    hand_qpos=(
                        row.hand["qpos"][0]
                        if row.hand is not None
                        else np.deg2rad(runtime.hand.home_qpos_deg)
                    ),
                )
            except IKReferenceError as exc:
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
        robot.send_action(
            command,
            expires_ns=time.monotonic_ns() + int(self.config.target_max_age_s * 1e9),
        )
        submitted_ns = time.monotonic_ns()
        if target is not None:
            anchor = (
                arm_reference_from_feedback(row.arm["qpos"][0], *self.solver.limits.T)
                if self.ik_reference is None
                else self.ik_reference.anchor
            )
            self.ik_reference = IKReference(
                cast(np.ndarray, command.arm_qpos), anchor, submitted_ns
            )
        else:
            self.clear_reference()

    def propose(
        self, pressed_keys, measured_qpos, *, hand_qpos, feedback_ns, control_ns
    ) -> IKResult:
        measured, previous, anchor, h = prepare_reference(
            self.solver,
            self.ik_reference,
            measured_qpos,
            feedback_ns=feedback_ns,
            control_ns=control_ns,
            period_s=1 / self.config.control_hz,
        )
        baseline_pose = self.solver.compute_eef_pose_base(previous)
        if self.ik_reference is None:
            workspace = self.solver.constraints.workspace_with_tolerance
            reason = workspace_start_rejection(
                baseline_pose.p,
                workspace[:, 0],
                workspace[:, 1],
            )
            if reason is not None:
                return IKResult(qpos=None, reason=reason)
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
        return self.solver.solve(
            pose, measured, previous, anchor, h=h, hand_qpos=hand_qpos
        )


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
