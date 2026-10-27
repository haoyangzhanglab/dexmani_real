"""笛卡尔 jog 提案、输入代际与正常提交后的目标参考。"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.geometry.pose import Pose, quat_multiply
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.runtime.keyboard import CARTESIAN_JOG_KEYS
from dexmani_real.runtime.state import RunEndReason, SafetyState, SessionState

if TYPE_CHECKING:
    from dexmani_real.robot.system import Robot
    from dexmani_real.runtime.keyboard import Keyboard


class JogController:
    """持有成功提交的目标参考；跟踪误差不改变操作者目标。"""

    def __init__(self, solver, config, workspace):
        self.solver = solver
        self.config = config
        self.position_lower = workspace[:, 0] + config.workspace_command_margin_m
        self.position_upper = workspace[:, 1] - config.workspace_command_margin_m
        self.command_pose = self.command_qpos = None
        self.blocked_until_release = False

    def clear_reference(self):
        self.command_pose = self.command_qpos = None

    def release_if_idle(self, pressed_keys):
        if not any_jog_key_held(pressed_keys):
            self.blocked_until_release = False

    def propose(self, pressed_keys, measured_qpos, *, hand_qpos):
        baseline_qpos, baseline_pose = self.command_qpos, self.command_pose
        if baseline_pose is None:
            baseline_qpos = measured_qpos
            baseline_pose = self.solver.compute_eef_pose_base(measured_qpos)
        dx, drpy = compute_cartesian_jog_delta(
            pressed_keys, self.config.delta_pos_m, self.config.delta_rpy_rad
        )
        pose, desired_position, changed = propose_cartesian_jog_pose(
            baseline_pose, dx, drpy,
            self.position_lower, self.position_upper,
        )
        result = None
        if changed:
            result = self.solver.solve(
                pose, measured_qpos, baseline_qpos,
                hand_qpos=hand_qpos,
            )
        if result is None or result.qpos is None:
            self.blocked_until_release = True
        return pose, desired_position, result

    def submit(
        self, pose: Pose, target: np.ndarray, keys: Keyboard, generation: int,
        runtime_state: SessionState, robot: Robot, epoch: int, *,
        valid_until_ns: int, execute_until_ns: int,
    ) -> bool:
        """约束输入代际与运行身份，仅在正常提交后更新目标参考。"""
        deadline_ns = min(valid_until_ns, execute_until_ns)

        def reserve() -> int | None:
            with runtime_state.motion_lock:
                if runtime_state.run_id != epoch or not runtime_state.requests_allow_motion():
                    return None
                if time.monotonic_ns() >= deadline_ns:
                    runtime_state.revoke_motion(expected_run_id=epoch)
                    raise TimeoutError("jog proposal deadline expired")
                if runtime_state.safety_state == SafetyState.ARMED:
                    if runtime_state.begin_motion() is None:
                        return None
                elif runtime_state.safety_state != SafetyState.RUNNING:
                    return None
                return runtime_state.run_id

        run_id = keys.commit_if_current(generation, reserve)
        if not run_id:
            return False
        try:
            command = RobotCommand(run_id, target, input_generation=generation)
            next_pose = pose.copy()
            # IPC 消费者在首次发送前再次核对输入代际。
            robot.send_action(
                command,
                valid_until_ns=valid_until_ns,
                execute_until_ns=execute_until_ns,
            )

            def finish() -> bool:
                with runtime_state.motion_lock:
                    if not runtime_state.can_move(run_id=run_id):
                        return False
                    if time.monotonic_ns() >= deadline_ns:
                        return False
                    self.command_pose, self.command_qpos = next_pose, command.arm_qpos
                    return True

            if not keys.commit_if_current(generation, finish):
                if time.monotonic_ns() >= deadline_ns:
                    raise TimeoutError("jog proposal expired during submission")
                raise OperationCancelled("jog proposal changed during submission")
            return True
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                reason = RunEndReason.ESTOP
            elif isinstance(exc, RobotError):
                reason = RunEndReason.HARDWARE_FAULT
            else:
                reason = RunEndReason.EXECUTOR_BOUNDARY
            runtime_state.revoke_motion(reason=reason, expected_run_id=run_id)
            raise


def any_jog_key_held(active_keys: Iterable[str]) -> bool:
    """相反方向相互抵消时，仍需等待所有 jog 键释放。"""
    return not CARTESIAN_JOG_KEYS.isdisjoint(active_keys)


def jog_motion_requested(active_keys: Iterable[str]) -> bool:
    keys = frozenset(active_keys)
    if keys.intersection({"r", "q", "esc"}):
        return False
    return any(
        (positive in keys) != (negative in keys)
        for positive, negative in (
            ("w", "s"), ("d", "a"), ("up", "down"),
            ("left", "right"), ("i", "k"), ("j", "l"),
        )
    )


def propose_cartesian_jog_pose(
    command_pose: Pose,
    dx: np.ndarray,
    drpy: np.ndarray,
    position_lower: np.ndarray,
    position_upper: np.ndarray,
) -> tuple[Pose, np.ndarray, bool]:
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
