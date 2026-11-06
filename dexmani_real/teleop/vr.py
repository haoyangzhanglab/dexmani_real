"""VR 锚点映射、手部重定向与提交后的位姿平滑。"""

from scipy.spatial.transform import Rotation

from dexmani_real.control.actions import ActionResult, StartPoseRejected
from dexmani_real.control.constraints import workspace_start_rejection
from dexmani_real.geometry.pose import (
    WXYZ_TO_XYZW,
    XYZW_TO_WXYZ,
    Pose,
)
from dexmani_real.robot.commands import arm_reference_from_feedback
from dexmani_real.runtime.ik_reference import IKReference, prepare_reference


class VRController:
    def __init__(self, action_mapper, hand_retargeter, config, vr_to_robot_rot):
        self.action_mapper = action_mapper
        self.hand_retargeter = hand_retargeter
        self.config = config
        self.rotation = Rotation.from_matrix(vr_to_robot_rot)
        self.clear()

    def clear(self):
        self.anchor = None
        self.ik_reference = None
        self.previous_eef_target = None

    def reset(self, row):
        self.clear()
        arm_qpos = self.action_mapper.solver.measured_reference(row.arm["qpos"][0])
        pose = self.action_mapper.solver.compute_eef_pose_base(arm_qpos)
        workspace = self.action_mapper.solver.constraints.workspace
        reason = workspace_start_rejection(pose.p, workspace[:, 0] - 1e-9, workspace[:, 1] + 1e-9)
        if reason is not None:
            raise StartPoseRejected(reason)
        eef_rotation = Rotation.from_quat(pose.q[WXYZ_TO_XYZW])
        wrist_rotation = Rotation.from_quat(row.vr["wrist_quat_wxyz"][WXYZ_TO_XYZW])
        # R_target = A W (A W0)^-1 E0，初始腕姿态映射到实测 EEF 姿态。
        rotation_right = (self.rotation * wrist_rotation).inv() * eef_rotation
        if self.hand_retargeter is not None:
            self.hand_retargeter.reset(
                initial_qpos=row.hand["qpos"][0], timestamp_ns=row.observation_timestamp_ns,
            )

        # 所有初始化成功后才发布新锚点；首帧 EMA 从同一个实测 FK 位姿开始。
        self.anchor = row.vr["wrist_pos"].copy(), pose.p, rotation_right
        self.previous_eef_target = pose

    def _wrist_target(self, wrist_pos, wrist_quat_wxyz):
        """锚点相对映射与最短旋转 EMA；平滑参考只在提交后更新。"""
        previous = self.previous_eef_target
        if self.anchor is None or previous is None:
            raise RuntimeError("VR wrist mapping requires a reset anchor")
        wrist_origin, eef_origin, rotation_right = self.anchor
        position = eef_origin + self.config.vr_pos_scale * self.rotation.apply(wrist_pos - wrist_origin)
        rotation = self.rotation * Rotation.from_quat(wrist_quat_wxyz[WXYZ_TO_XYZW]) * rotation_right
        previous_rotation = Rotation.from_quat(previous.q[WXYZ_TO_XYZW])
        relative = previous_rotation.inv() * rotation
        smoothed = previous_rotation * Rotation.from_rotvec(self.config.ema_alpha_rot * relative.as_rotvec())
        alpha_pos = self.config.ema_alpha_pos
        return Pose(alpha_pos * position + (1.0 - alpha_pos) * previous.p, smoothed.as_quat()[XYZW_TO_WXYZ])

    def compute_target(self, row, *, control_ns) -> ActionResult:
        reference = prepare_reference(
            self.action_mapper.solver, self.ik_reference, row.arm["qpos"][0],
            feedback_ns=int(row.arm["timestamp_ns"][0]), control_ns=control_ns,
            period_s=1 / self.config.control_hz,
        )
        target = self._wrist_target(row.vr["wrist_pos"], row.vr["wrist_quat_wxyz"])
        hand = None
        if self.hand_retargeter is not None:
            hand = self.hand_retargeter.retarget(
                row.vr["landmarks"], timestamp_ns=control_ns,
            )
            if hand is None:
                # 无 arm 求解时仍检查参考，禁止失配目标进入普通重发。
                self.action_mapper.solver.bounds(*reference)
                return ActionResult(None, None, self.hand_retargeter.rejection_reason)
        return self.action_mapper.map_eef(
            target, hand, reference
        )

    def commit(self, command, *, eef_pose, submit_end_ns, measured_qpos):
        if self.hand_retargeter is not None:
            self.hand_retargeter.commit(command.hand_qpos, timestamp_ns=submit_end_ns)
        anchor = (arm_reference_from_feedback(measured_qpos, *self.action_mapper.solver.limits.T)
                  if self.ik_reference is None else self.ik_reference.anchor)
        self.ik_reference = IKReference(command.arm_qpos, anchor, submit_end_ns)
        # map_eef 交出的临时目标此后归控制器所有。
        self.previous_eef_target = eef_pose
