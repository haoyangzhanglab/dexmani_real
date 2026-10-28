"""笛卡尔 jog 提案、输入代际与正常提交后的目标参考。"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.geometry.pose import Pose, quat_multiply
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.runtime.keyboard import CARTESIAN_JOG_KEYS, JOG_KEY_PAIRS
from dexmani_real.runtime.observation import (
    Observation,
    ObservationNeeds,
    check_observation,
    read_observation,
)
from dexmani_real.runtime.state import RunEndReason, SafetyState, SessionState

if TYPE_CHECKING:
    from dexmani_real.control.ik import IKResult
    from dexmani_real.robot.system import Robot
    from dexmani_real.runtime.keyboard import Keyboard


logger = logging.getLogger(__name__)


@dataclass
class JogStep:
    observation: Observation | None = None
    pose: Pose | None = None
    desired_position: np.ndarray | None = None
    ik_result: IKResult | None = None
    committed: bool = False
    issues: tuple[str, ...] = ()


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

    def release_if_idle(self, pressed_keys):
        if CARTESIAN_JOG_KEYS.isdisjoint(pressed_keys):
            self.blocked_until_release = False

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

    def step(self, robot, keys, state, runtime, *, deadline_source):
        """准备后采样一次，输入代际与两类期限共同约束提交。"""
        step = JogStep()
        pressed_keys, generation = keys.snapshot()
        self.release_if_idle(pressed_keys)
        moving = jog_motion_requested(pressed_keys) and not self.blocked_until_release
        with state.motion_lock:
            safety_state, epoch = state.safety_state, state.run_id
            allowed = state.requests_allow_motion() and safety_state in (
                SafetyState.ARMED, SafetyState.RUNNING,
            )
        if not allowed:
            self.stop(robot, state)
            step.issues = ("jog_motion_revoked",)
            return step

        if not moving:
            if robot.stop_required:
                self.stop(robot, state)
            self.clear_reference()
        elif safety_state == SafetyState.ARMED:
            self.clear_reference()
            if robot.stop_required:
                robot.stop()
            try:
                robot.prepare_streaming(epoch)
                if keys.snapshot()[1] != generation or state.run_id != epoch:
                    raise OperationCancelled("jog changed during preparation")
            except (OperationCancelled, TimeoutError) as exc:
                logger.info("jog preparation interrupted: %s", exc)
                self.stop(robot, state)
                return step

        row = step.observation = read_observation(None, robot)
        issues, feedback_until_ns = check_observation(
            row, runtime, ObservationNeeds(), require_hand=runtime.hand.enabled,
        )
        if issues:
            self.clear_reference()
            if robot.stop_required:
                self.stop(robot, state)
            step.issues = tuple(issues)
            return step
        if not moving:
            return step

        step.pose, step.desired_position, step.ik_result = self.propose(
            pressed_keys, row.arm["qpos"][0],
            hand_qpos=(row.hand["qpos"][0] if row.hand is not None
                       else np.deg2rad(runtime.hand.home_qpos_deg)),
        )
        if step.ik_result is None or step.ik_result.qpos is None:
            self.pause(robot, state)
            return step
        source_ns = (int(row.arm["timestamp_ns"][0]) if deadline_source == "arm"
                     else row.observation_timestamp_ns)
        try:
            step.committed = self.submit(
                step.pose, step.ik_result.qpos, keys, generation, state, robot, epoch,
                valid_until_ns=feedback_until_ns,
                execute_until_ns=source_ns + int(self.config.target_max_age_s * 1e9),
            )
        except (OperationCancelled, TimeoutError) as exc:
            logger.info("jog submission interrupted: %s", exc)
        if not step.committed:
            self.stop(robot, state)
        return step

    def propose(self, pressed_keys, measured_qpos, *, hand_qpos) -> tuple[Pose, np.ndarray, IKResult | None]:
        baseline_qpos, baseline_pose = self.command_qpos, self.command_pose
        if baseline_pose is None:
            baseline_qpos = measured_qpos
            baseline_pose = self.solver.compute_eef_pose_base(measured_qpos)
        dx, drpy = jog_delta(
            pressed_keys, self.config.delta_pos_m, self.config.delta_rpy_rad
        )
        pose, desired_position, changed = jog_pose(
            baseline_pose, dx, drpy,
            self.position_lower, self.position_upper,
        )
        result = None
        if changed:
            result = self.solver.solve(
                pose, measured_qpos, baseline_qpos,
                hand_qpos=hand_qpos,
            )
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
