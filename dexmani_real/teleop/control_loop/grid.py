"""Current-row teleop mapping, absolute target publication and raw recording."""
import numpy as np
from dexmani_real.planning import Pose
from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
from dexmani_real.planning.kinematics.pose import rot6d_to_quat_wxyz, quat_wxyz_to_rot6d
from dexmani_real.robot.commands import RobotCommand, publish_command
from dexmani_real.robot.projection import project_arm_command, project_hand_command
from dexmani_real.recording.frame import build_episode_frame
from dexmani_real.recording.storage.schema import FRAME_OK, FRAME_RETARGET_FAIL, FRAME_IK_FAIL
from dexmani_real.teleop.control_loop.action_proposal import compute_target_eef_pose
from dexmani_real.teleop.control_loop.hand_control import HandRetargetObservationCache, compute_hand_command, reset_hand_retargeter


class TeleopController:
    def __init__(self, planner, arm_mapper, runtime, hand_retargeter):
        self.planner, self.arm_mapper, self.runtime = planner, arm_mapper, runtime
        self.hand_retargeter = hand_retargeter
        self.arm_fk = make_arm_fk()
        self.cache = HandRetargetObservationCache()
        self.prev_qpos_cmd = None
        self.clear_reference()

    def clear_reference(self):
        self.arm_mapper.clear()
        self.ema_pos = self.ema_quat = None
        self.cache.reset()
        reset_hand_retargeter(self.hand_retargeter)

    def reset_reference(self, row):
        self.clear_reference()
        self.prev_qpos_cmd = row.arm["qpos"][0].copy()
        pos, rot = self.arm_fk.compute(self.prev_qpos_cmd)
        self.arm_mapper.reset(wrist_pos=row.vr["wrist_pos"], wrist_quat_wxyz=row.vr["wrist_quat_wxyz"],
                              eef_pos=pos, eef_quat_wxyz=rot6d_to_quat_wxyz(rot))
        if row.hand is not None:
            reset_hand_retargeter(self.hand_retargeter, row.hand["qpos"][0])
        return self.arm_mapper.is_ready()

    def compute(self, row, run_id):
        cfg = self.runtime
        mapped = self.arm_mapper.map(row.vr["wrist_pos"], row.vr["wrist_quat_wxyz"])
        if mapped is None:
            return None, FRAME_IK_FAIL, None
        target = compute_target_eef_pose(mapped["pos"], mapped["quat_wxyz"],
            previous_position_world_m=self.ema_pos, previous_quat_world_wxyz=self.ema_quat,
            workspace_bounds_world_m=cfg.policy.workspace.as_array(),
            ema_alpha_position=cfg.policy.ema.alpha_pos, ema_alpha_rotation=cfg.policy.ema.alpha_rot)
        intent = np.concatenate((target.position_world_m, quat_wxyz_to_rot6d(target.quat_world_wxyz)))
        hand = None
        if cfg.policy.hand_enabled:
            try:
                proposal = compute_hand_command(self.hand_retargeter, row.vr, self.cache)
                if proposal is None:
                    return None, FRAME_RETARGET_FAIL, intent
                hand = project_hand_command(proposal, qpos_min_rad=cfg.hand.qpos_min_rad,
                                            qpos_max_rad=cfg.hand.qpos_max_rad)
            except (ValueError, RuntimeError):
                return None, FRAME_RETARGET_FAIL, intent
        solution = self.planner.solve_teleop_ik(Pose(p=target.position_world_m, q=target.quat_world_wxyz),
                                               row.arm["qpos"][0], self.prev_qpos_cmd)
        if not solution.success:
            return None, FRAME_IK_FAIL, intent
        arm = project_arm_command(solution.qpos, row.arm["qpos"][0],
            joint_lower_rad=cfg.arm.joint_limit_lower, joint_upper_rad=cfg.arm.joint_limit_upper)
        self.ema_pos, self.ema_quat = target.position_world_m, target.quat_world_wxyz
        return RobotCommand(run_id, arm, hand), FRAME_OK, intent


def run_control_grid_tick(controller, shared, row, recorder=None):
    epoch = int(shared.run_id.value)
    target, status, intent = controller.compute(row, epoch)
    stamp = publish_command(shared, target) if target is not None else 0
    if int(shared.run_id.value) != epoch or (target is not None and not stamp):
        return status
    if stamp:
        controller.prev_qpos_cmd = target.arm_qpos.copy()
    if recorder is not None and recorder.is_recording:
        recorder.add_frame(build_episode_frame(row, target, action_timestamp_ns=stamp,
                                              frame_status=status, arm_eef_intent=intent))
    return status
