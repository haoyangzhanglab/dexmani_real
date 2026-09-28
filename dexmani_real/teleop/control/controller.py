"""Current-row teleop mapping, absolute target dispatch and raw recording."""

import numpy as np

from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
from dexmani_real.planning.kinematics.ik import IKFailureKind
from dexmani_real.planning.kinematics.pose import quat_wxyz_to_rot6d, rot6d_to_quat_wxyz
from dexmani_real.recording.frame import build_episode_frame
from dexmani_real.robot.action import ActionIntent
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.robot import DispatchError
from dexmani_real.runtime.safety import revoke_motion
from dexmani_real.teleop.control.action_proposal import compute_target_eef_pose
from dexmani_real.teleop.control.hand_retargeting import (
    HandRetargetObservationCache,
    compute_hand_command,
    reset_hand_retargeter,
)


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

    def compute_command(self, row, run_id):
        cfg = self.runtime
        mapped = self.arm_mapper.map(row.vr["wrist_pos"], row.vr["wrist_quat_wxyz"])
        if mapped is None:
            return None, False, None
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
            try:
                proposal = compute_hand_command(
                    self.hand_retargeter, row.vr, self.hand_observation_cache
                )
                if proposal is None:
                    return None, False, intent
                hand = proposal
            except (ValueError, RuntimeError):
                return None, False, intent
        realized = self.realizer.realize(
            ActionIntent("eef", intent, hand), row.arm["qpos"][0], self.previous_arm_command
        )
        solution = realized.ik_result
        if solution.failure_kind == IKFailureKind.INVALID_OUTPUT:
            raise RuntimeError(f"online IK technical failure: {solution.reason}")
        if not solution.success:
            return None, False, intent
        return RobotCommand(run_id, realized.arm_qpos, realized.hand_qpos), True, realized.eef_pose


def execute_control_step(controller, shared, robot, row, recorder=None):
    epoch = int(shared.run_id.value)
    target, control_ok, intent = controller.compute_command(row, epoch)
    if recorder is not None:
        recorder.check_error()
    result = None
    if target is not None:
        try:
            result = robot.send_action(target)
        except DispatchError:
            if recorder is not None:
                recorder.mark_discard("dispatch_rejected")
            revoke_motion(shared)
            try:
                robot.stop()
            except Exception:
                from dexmani_real.utils.log import get_logger

                get_logger(__name__).exception("stop after teleop dispatch failure also failed")
            raise
    stamp = result.timestamp_ns if result is not None else 0
    if int(shared.run_id.value) != epoch:
        if recorder is not None:
            recorder.mark_discard("dispatch_rejected")
        return control_ok
    if result is not None:
        controller.previous_arm_command = target.arm_qpos.copy()
        controller.smoothed_eef_position = intent[:3].copy()
        controller.smoothed_eef_quaternion = rot6d_to_quat_wxyz(intent[3:])
    if recorder is not None:
        if not control_ok or result is None:
            recorder.mark_discard("control_failure")
        if recorder.accepting_frames:
            recorder.add_frame(build_episode_frame(row, target, result), step_timestamp_ns=stamp)
    return control_ok
