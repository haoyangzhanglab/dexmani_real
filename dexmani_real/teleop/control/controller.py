"""Current-row teleop mapping, absolute target dispatch and raw recording."""

import logging
import time

from typing import NamedTuple

import numpy as np

from dexmani_real.recording.recorder import RecordingError
from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
from dexmani_real.planning.kinematics.ik import IKFailureKind
from dexmani_real.planning.kinematics.pose import quat_wxyz_to_rot6d, rot6d_to_quat_wxyz
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


logger = logging.getLogger(__name__)


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
            previous_position_xarm_base_m=self.smoothed_eef_position,
            previous_quat_xarm_base_wxyz=self.smoothed_eef_quaternion,
            ema_alpha_position=cfg.policy.ema.alpha_pos,
            ema_alpha_rotation=cfg.policy.ema.alpha_rot,
        )
        intent = np.concatenate(
            (target.position_xarm_base_m, quat_wxyz_to_rot6d(target.quat_xarm_base_wxyz))
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
    controller, shared, robot, row, recorder=None, *, action_deadline_ns, termination_details=None
):
    epoch = int(shared.run_id)
    realization = target = result = failure = stop_error = None
    control_ok = interrupted = False
    try:
        try:
            realization = controller.compute_target(row)
            control_ok = realization is not None and realization.arm_qpos is not None
            if control_ok:
                target = RobotCommand(epoch, realization.arm_qpos, realization.hand_qpos)
            if recorder is not None:
                recorder.check_error()
            if target is not None:
                result = robot.send_action(
                    target,
                    valid_until_ns=min(
                        action_deadline_ns,
                        feedback_deadline_ns(row, controller.runtime, include_vr=True),
                    ),
                )
        except BaseException as exc:
            failure = exc
            if isinstance(exc, (DispatchError, DispatchInterrupted)):
                result = exc.result
            cancelled = isinstance(exc, KeyboardInterrupt)
            cause = getattr(exc, "cause", None)
            if termination_details is not None:
                termination_details.append(dict(
                    stage="control", exception_type=type(exc).__name__, message=str(exc), cause=cause,
                ))
            if cancelled:
                shared.estop_request = True
            revoke_motion_if_run_id(
                shared, epoch,
                reason=RunEndReason.ESTOP if cancelled else
                RunEndReason.RECORDING_FAILURE if isinstance(exc, RecordingError) else
                RunEndReason.EXECUTOR_BOUNDARY if getattr(exc, "revoked", False) else
                RunEndReason.HARDWARE_FAULT if isinstance(exc, DispatchError) else
                RunEndReason.POLICY_FAILURE,
            )
        if not control_ok and recorder is not None:
            if termination_details is not None:
                termination_details.append(dict(
                    stage="control", reason="target_unavailable",
                    rejection=getattr(realization, "rejection_reason", None),
                ))
            revoke_motion_if_run_id(shared, epoch, reason=RunEndReason.EXECUTOR_BOUNDARY)
        expired = time.monotonic_ns() >= action_deadline_ns
        if expired:
            if termination_details is not None:
                termination_details.append(dict(
                    stage="dispatch", reason="action_deadline_expired",
                    deadline_ns=action_deadline_ns, returned_ns=time.monotonic_ns(),
                ))
            # A blocking SDK may have accepted a target; retain its actual status.
            revoke_motion_if_run_id(shared, epoch, reason=RunEndReason.EXECUTOR_BOUNDARY)
        interrupted = expired or failure is not None or int(shared.run_id) != epoch
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
                recorder.add_frame(row, target, result)
            except Exception as exc:
                if failure is not None:
                    logger.exception("recording after dispatch also failed")
                    raise failure from exc
                raise
    if failure is not None and not (
        isinstance(failure, DispatchError) and failure.revoked
        and failure.cause in {"authority_revoked", "deadline_expired"}
    ):
        raise failure
    if stop_error is not None:
        raise stop_error
    return ControlStepResult(control_ok and not interrupted, result, interrupted)
