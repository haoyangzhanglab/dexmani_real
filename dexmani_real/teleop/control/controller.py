"""Current-row teleop mapping, absolute target dispatch and raw recording."""

from typing import NamedTuple

import numpy as np

from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
from dexmani_real.planning.kinematics.ik import IKFailureKind
from dexmani_real.planning.kinematics.pose import quat_wxyz_to_rot6d, rot6d_to_quat_wxyz
from dexmani_real.recording.frame import build_episode_frame
from dexmani_real.robot.action import ActionIntent
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.robot import DispatchError, DispatchInterrupted
from dexmani_real.runtime.observation import feedback_deadline_ns
from dexmani_real.runtime.safety import RunEndReason, revoke_motion_if_run_id
from dexmani_real.teleop.control.action_proposal import compute_target_eef_pose
from dexmani_real.teleop.control.hand_retargeting import (
    HandRetargetObservationCache,
    compute_hand_command,
    reset_hand_retargeter,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class TeleopController:
    def __init__(self, realizer, arm_mapper, runtime, hand_retargeter):
        self.realizer, self.arm_mapper, self.runtime = (
            realizer,
            arm_mapper,
            runtime,
        )
        self.hand_retargeter = hand_retargeter
        self.arm_fk = make_arm_fk()
        self.hand_observation_cache = HandRetargetObservationCache()
        self.previous_arm_command = None
        self.clear_reference()

    def clear_reference(self):
        self.previous_arm_command = None
        self.arm_mapper.clear()
        self.smoothed_eef_position = self.smoothed_eef_quaternion = None
        self.hand_observation_cache.reset()
        reset_hand_retargeter(self.hand_retargeter)

    def reset_reference(self, row):
        self.clear_reference()
        self.previous_arm_command = row.arm["qpos"][0].copy()
        pos, rot = self.arm_fk.compute(self.previous_arm_command)
        self.arm_mapper.reset(
            wrist_pos=row.vr["wrist_pos"],
            wrist_quat_wxyz=row.vr["wrist_quat_wxyz"],
            eef_pos=pos,
            eef_quat_wxyz=rot6d_to_quat_wxyz(rot),
        )
        if row.hand is not None:
            reset_hand_retargeter(self.hand_retargeter, row.hand["qpos"][0])
        return self.arm_mapper.is_ready()

    def compute_target(self, row):
        cfg = self.runtime
        mapped = self.arm_mapper.map(row.vr["wrist_pos"], row.vr["wrist_quat_wxyz"])
        if mapped is None:
            return None
        target = compute_target_eef_pose(
            mapped["pos"],
            mapped["quat_wxyz"],
            previous_position_world_m=self.smoothed_eef_position,
            previous_quat_world_wxyz=self.smoothed_eef_quaternion,
            ema_alpha_position=cfg.policy.ema.alpha_pos,
            ema_alpha_rotation=cfg.policy.ema.alpha_rot,
        )
        intent = np.concatenate(
            (target.position_world_m, quat_wxyz_to_rot6d(target.quat_world_wxyz))
        )
        hand = None
        if cfg.policy.hand_enabled:
            proposal = compute_hand_command(
                self.hand_retargeter, row.vr, self.hand_observation_cache
            )
            if proposal is None:
                return None
            hand = proposal
        realized = self.realizer.realize(
            ActionIntent("eef", intent, hand), row.arm["qpos"][0], self.previous_arm_command
        )
        solution = realized.ik_result
        if solution is not None and solution.failure_kind == IKFailureKind.INVALID_OUTPUT:
            raise RuntimeError(f"online IK technical failure: {solution.reason}")
        return realized

    def commit_dispatched_target(self, realization):
        self.previous_arm_command = realization.arm_qpos.copy()
        self.smoothed_eef_position = realization.eef_pose[:3].copy()
        self.smoothed_eef_quaternion = rot6d_to_quat_wxyz(realization.eef_pose[3:])


class ControlStepResult(NamedTuple):
    target_dispatched: bool
    dispatch: object
    interrupted: bool


def execute_control_step(
    controller, shared, robot, row, recorder=None, *, termination_details=None
):
    epoch = int(shared.run_id.value)
    realization = controller.compute_target(row)
    control_ok = realization is not None and realization.arm_qpos is not None
    target = (
        RobotCommand(epoch, realization.arm_qpos, realization.hand_qpos) if control_ok else None
    )
    if realization is not None and realization.rejection_reason:
        logger.debug("teleop target rejected: %s", realization.rejection_reason)
    if recorder is not None:
        recorder.check_error()
    result = failure = stop_error = None
    interrupted = False
    try:
        if target is not None:
            try:
                result = robot.send_action(
                    target,
                    valid_until_ns=feedback_deadline_ns(row, controller.runtime, include_vr=True),
                )
            except (DispatchError, DispatchInterrupted) as exc:
                result, failure = exc.result, exc
                cancelled = isinstance(exc, DispatchInterrupted)
                if cancelled:
                    shared.estop_request.value = True
                revoke_motion_if_run_id(
                    shared,
                    epoch,
                    reason=RunEndReason.ESTOP
                    if cancelled
                    else RunEndReason.EXECUTOR_BOUNDARY
                    if exc.revoked
                    else RunEndReason.HARDWARE_FAULT,
                )
        interrupted = failure is not None or int(shared.run_id.value) != epoch
        if interrupted:
            try:
                robot.stop()
            except Exception as exc:
                stop_error = exc
                if termination_details is not None:
                    termination_details.append(
                        dict(
                            stage="dispatch_stop",
                            exception_type=type(exc).__name__,
                            message=str(exc),
                        )
                    )
                if failure is not None:
                    logger.exception("stop after dispatch also failed")
        elif result is not None:
            controller.commit_dispatched_target(realization)
    finally:
        if recorder is not None and (interrupted or recorder.accepting_frames):
            try:
                recorder.add_frame(build_episode_frame(row, target, result))
            except Exception as exc:
                if failure is not None:
                    logger.exception("recording after dispatch also failed")
                    raise failure from exc
                raise
    if isinstance(failure, DispatchInterrupted) or (failure is not None and not failure.revoked):
        raise failure
    if stop_error is not None:
        raise stop_error
    return ControlStepResult(control_ok and not interrupted, result, interrupted)
