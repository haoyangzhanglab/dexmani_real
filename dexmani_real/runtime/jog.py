"""持续伺服会话中的笛卡尔 jog；松键减速保持，换向更新目标。"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.control.ik import IKResult
from dexmani_real.geometry.pose import Pose, quat_multiply
from dexmani_real.robot.commands import OperationCancelled, RobotCommand
from dexmani_real.runtime.keyboard import JOG_KEY_PAIRS
from dexmani_real.runtime.observation import (
    check_device_feedback,
    read_observation,
)

logger = logging.getLogger(__name__)

JOG_CONTROL_KEYS = frozenset(key for pair in JOG_KEY_PAIRS for key in pair) | {"q", "r", "esc"}


class CartesianJog:
    """持有成功提交的目标参考；跟踪误差不改变操作者目标。"""

    def __init__(self, solver, config, bounds):
        self.solver = solver
        self.config = config
        self.position_lower, self.position_upper = bounds
        self.command_pose = self.command_qpos = None
        self.rejection_reason = None

    def clear_reference(self):
        self.command_pose = self.command_qpos = None

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
            pose, result = self.propose(
                pressed_keys, row.arm["qpos"][0],
                hand_qpos=(row.hand["qpos"][0] if row.hand is not None
                           else np.deg2rad(runtime.hand.home_qpos_deg)),
            )
            if result.qpos is None:
                rejection_reason = result.reason
            elif keys.snapshot() & JOG_CONTROL_KEYS == pressed_keys:
                target = result.qpos
        if rejection_reason is not None and rejection_reason != self.rejection_reason:
            logger.warning("IK rejected target; braking and holding: %s", rejection_reason)
        self.rejection_reason = rejection_reason
        if target is not None:
            issues = check_device_feedback(row, runtime)
            if issues:
                raise OperationCancelled("jog mapping sources expired: " + "; ".join(issues))
        command = RobotCommand(arm_qpos=target, arm_hold=target is None)
        robot.send_action(
            command,
            expires_ns=time.monotonic_ns() + int(self.config.target_max_age_s * 1e9),
        )
        if target is not None:
            self.command_pose, self.command_qpos = pose.copy(), command.arm_qpos
        else:
            self.clear_reference()

    def propose(self, pressed_keys, measured_qpos, *, hand_qpos) -> tuple[Pose, IKResult]:
        baseline_qpos, baseline_pose = self.command_qpos, self.command_pose
        if baseline_pose is None:
            baseline_qpos = measured_qpos
            baseline_pose = self.solver.compute_eef_pose_base(measured_qpos)
        dx, drpy = jog_delta(
            pressed_keys, self.config.delta_pos_m, self.config.delta_rpy_rad
        )
        pose, changed = jog_pose(
            baseline_pose, dx, drpy,
            self.position_lower, self.position_upper,
        )
        if changed:
            result = self.solver.solve(
                pose, measured_qpos, baseline_qpos,
                hand_qpos=hand_qpos,
            )
        else:
            # 饱和仍是持续输入；保留已接受的目标，同时检查变化后的手部几何。
            reason = self.solver.constraints.rejection(
                baseline_qpos, collision_hand_qpos=hand_qpos,
            )
            result = IKResult(qpos=baseline_qpos if reason is None else None, reason=reason or "")
        return pose, result


def jog_motion_requested(active_keys: Iterable[str]) -> bool:
    keys = frozenset(active_keys)
    if keys.intersection({"r", "q", "esc"}):
        return False
    return any(
        (positive in keys) != (negative in keys)
        for positive, negative in JOG_KEY_PAIRS
    )


def jog_pose(
    command_pose: Pose,
    dx: np.ndarray,
    drpy: np.ndarray,
    position_lower: np.ndarray,
    position_upper: np.ndarray,
) -> tuple[Pose, bool]:
    position = np.clip(command_pose.p + dx, position_lower, position_upper)
    rotating = bool(np.any(drpy != 0.0))
    changed = rotating or not np.array_equal(position, command_pose.p)
    quaternion = command_pose.q.copy()
    if rotating:
        delta_quaternion = Rotation.from_euler("xyz", drpy).as_quat()[[3, 0, 1, 2]]
        quaternion = quat_multiply(delta_quaternion, quaternion)
    return Pose(p=position, q=quaternion), changed


def jog_delta(
    pressed_keys: Iterable[str],
    delta_pos: float,
    delta_rpy: float,
) -> tuple[np.ndarray, np.ndarray]:
    keys = frozenset(pressed_keys)
    directions = np.array([
        int(positive in keys) - int(negative in keys)
        for positive, negative in JOG_KEY_PAIRS
    ], dtype=np.float64)
    dx, drpy = directions[:3], directions[3:]
    dx *= delta_pos / max(1.0, float(np.linalg.norm(dx)))
    drpy *= delta_rpy / max(1.0, float(np.linalg.norm(drpy)))
    return dx, drpy
