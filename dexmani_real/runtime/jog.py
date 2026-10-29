"""持续伺服会话中的笛卡尔 jog；松键减速保持，换向更新目标。"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.geometry.pose import Pose, quat_multiply
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.runtime.keyboard import CARTESIAN_JOG_KEYS, JOG_KEY_PAIRS
from dexmani_real.runtime.observation import (
    ObservationNeeds,
    check_observation,
    read_observation,
)
from dexmani_real.runtime.state import SafetyState

if TYPE_CHECKING:
    from dexmani_real.control.ik import IKResult


logger = logging.getLogger(__name__)


class CartesianJog:
    """持有成功提交的目标参考；跟踪误差不改变操作者目标。"""

    def __init__(self, solver, config, bounds):
        self.solver = solver
        self.config = config
        self.position_lower, self.position_upper = bounds
        self.command_pose = self.command_qpos = None
        self.blocked_until_release = False

    def clear_reference(self):
        self.command_pose = self.command_qpos = None

    def stop(self, robot, state):
        revoked = state.revoke_motion()
        try:
            robot.stop()
        finally:
            self.clear_reference()
        return revoked

    def pause(self, robot, state):
        try:
            return self.stop(robot, state)
        finally:
            self.blocked_until_release = True

    def prepare(self, robot, state):
        """仅在进入会话或真实停止后准备；普通按键变化不重新准备。"""
        self.clear_reference()
        robot.prepare_streaming(state.run_id)
        if state.begin_motion() is None:
            raise OperationCancelled("jog cancelled during preparation")

    def step(self, robot, keys, state, runtime):
        """发布点动目标或减速保持；故障由会话边界记录、撤权并收尾。"""
        pressed_keys = keys.snapshot()
        if CARTESIAN_JOG_KEYS.isdisjoint(pressed_keys):
            self.blocked_until_release = False
        with state.motion_lock:
            safety_state, epoch = state.safety_state, state.run_id
            allowed = state.requests_allow_motion() and safety_state in (
                SafetyState.ARMED, SafetyState.RUNNING,
            )
        if not allowed:
            self.stop(robot, state)
            return
        if safety_state == SafetyState.ARMED:
            if not self.blocked_until_release:
                self.prepare(robot, state)
            # 准备是独立轮次；下一轮重新读取按键与反馈，不补交准备前的输入。
            return

        row = read_observation(None, robot)
        issues, feedback_until_ns = check_observation(
            row, runtime, ObservationNeeds(), require_hand=runtime.hand.enabled,
        )
        if issues:
            raise RobotError("jog feedback rejected: " + "; ".join(issues))
        target = None
        if jog_motion_requested(pressed_keys) and not self.blocked_until_release:
            pose, result = self.propose(
                pressed_keys, row.arm["qpos"][0],
                hand_qpos=(row.hand["qpos"][0] if row.hand is not None
                           else np.deg2rad(runtime.hand.home_qpos_deg)),
            )
            if result is not None:
                if result.qpos is None:
                    logger.warning("IK rejected target: %s", result.reason)
                    self.blocked_until_release = True
                elif keys.snapshot() == pressed_keys:
                    target = result.qpos
        if target is None:
            self.clear_reference()
        command = RobotCommand(epoch, arm_qpos=target, arm_hold=target is None)
        source_ns = int(row.arm["timestamp_ns"][0])
        robot.send_action(
            command,
            valid_until_ns=feedback_until_ns,
            execute_until_ns=source_ns + int(self.config.target_max_age_s * 1e9),
        )
        if target is not None:
            self.command_pose, self.command_qpos = pose.copy(), command.arm_qpos

    def propose(self, pressed_keys, measured_qpos, *, hand_qpos) -> tuple[Pose, IKResult | None]:
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
        result = None
        if changed:
            result = self.solver.solve(
                pose, measured_qpos, baseline_qpos,
                hand_qpos=hand_qpos,
            )
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
