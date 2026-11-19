"""VR 锚点映射、手部重定向与提交后的位姿平滑。"""

import sys
import time
from typing import cast

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.control.actions import ActionResult, StartPoseRejected
from dexmani_real.geometry.pose import (
    WXYZ_TO_XYZW,
    XYZW_TO_WXYZ,
    Pose,
)
from dexmani_real.runtime.ik_reference import IKReference, prepare_reference


class VRController:
    def __init__(self, action_mapper, hand_retargeter, config, vr_to_robot_rot, *, timing=None):
        self.action_mapper = action_mapper
        self.hand_retargeter = hand_retargeter
        self.config = config
        self.rotation = Rotation.from_matrix(vr_to_robot_rot)
        self.timing = timing
        self.anchor: tuple[np.ndarray, np.ndarray, Rotation] | None
        self.clear()

    def clear(self):
        self.anchor = None
        self.ik_reference = None
        self.previous_eef_target = None

    def reset(self, row):
        self.clear()
        arm_qpos = self.action_mapper.solver.measured_reference(row.arm["qpos"][0])
        pose = self.action_mapper.solver.compute_eef_pose_base(arm_qpos)
        workspace = self.action_mapper.solver.workspace
        if (pose.p < workspace[:, 0] - 1e-9).any() or (pose.p > workspace[:, 1] + 1e-9).any():
            raise StartPoseRejected("实测末端超出允许的起点范围；请手动就位或调整范围后重试")
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

    def _wrist_target(self, wrist_pos, wrist_quat_wxyz, *, control_ns) -> tuple[Pose, np.ndarray]:
        """位置先投影再做 EMA，并返回本拍超出工作空间的位移。"""
        previous = self.previous_eef_target
        if self.anchor is None or previous is None:
            raise RuntimeError("VR wrist mapping requires a reset anchor")
        wrist_origin, eef_origin, rotation_right = self.anchor
        desired_position = eef_origin + self.config.vr_pos_scale * self.rotation.apply(wrist_pos - wrist_origin)
        if not np.isfinite(desired_position).all():
            raise ValueError("VR mapped position must be finite")
        workspace = self.action_mapper.solver.workspace
        position = np.clip(desired_position, workspace[:, 0], workspace[:, 1])
        rotation = self.rotation * Rotation.from_quat(wrist_quat_wxyz[WXYZ_TO_XYZW]) * rotation_right
        previous_rotation = Rotation.from_quat(previous.q[WXYZ_TO_XYZW])
        relative = previous_rotation.inv() * rotation
        smoothed = previous_rotation * Rotation.from_rotvec(self.config.ema_alpha_rot * relative.as_rotvec())
        alpha_pos = self.config.ema_alpha_pos
        target = Pose(alpha_pos * position + (1.0 - alpha_pos) * previous.p, smoothed.as_quat()[XYZW_TO_WXYZ])
        if self.timing is not None:
            self.timing.add(
                "eef_intent",
                control_ns=control_ns,
                desired_position_m=desired_position,
                bounded_position_m=position,
                desired_quat_wxyz=rotation.as_quat()[XYZW_TO_WXYZ],
                filtered_position_m=target.p,
                filtered_quat_wxyz=target.q,
            )
        return target, desired_position - position

    def compute_target(self, row, *, control_ns) -> tuple[ActionResult, np.ndarray, np.ndarray]:
        timing = self.timing
        reference = prepare_reference(
            self.action_mapper.solver, self.ik_reference, row.arm["qpos"][0],
            control_ns=control_ns,
            period_s=1 / self.config.control_hz,
        )
        measured, command, anchor, h = reference
        target, discarded_position_m = self._wrist_target(
            row.vr["wrist_pos"], row.vr["wrist_quat_wxyz"],
            control_ns=row.control_timestamp_ns,
        )
        hand = None
        retargeter = self.hand_retargeter
        if retargeter is not None:
            hand_started_ns = time.monotonic_ns() if timing is not None else None
            try:
                hand = retargeter.retarget(row.vr["landmarks"], timestamp_ns=control_ns)
            finally:
                if timing is not None:
                    timing.add(
                        "hand_retarget",
                        timestamp_ns=hand_started_ns,
                        finished_ns=time.monotonic_ns(),
                        control_ns=row.control_timestamp_ns,
                        succeeded=hand is not None,
                        rejection_reason=retargeter.rejection_reason if hand is None else None,
                    )
            if hand is None:
                return ActionResult(None, None, retargeter.rejection_reason), anchor, discarded_position_m
        ik_started_ns = time.monotonic_ns() if timing is not None else None
        result = None
        try:
            result = self.action_mapper.map_eef(target, hand, reference)
            return result, anchor, discarded_position_m
        finally:
            if timing is not None:
                timing.add(
                    "ik",
                    timestamp_ns=ik_started_ns,
                    finished_ns=time.monotonic_ns(),
                    control_ns=row.control_timestamp_ns,
                    desired_position_m=target.p,
                    desired_quat_wxyz=target.q,
                    h_s=h,
                    reference_qpos=command,
                    measured_qpos=measured,
                    goal_qpos=result.arm_qpos if result is not None else None,
                    rejection_reason=result.rejection_reason if result is not None else str(sys.exc_info()[1]),
                )

    def commit(self, command, *, ik_anchor, arm_published_ns, submit_end_ns, eef_pose, discarded_position_m):
        if self.hand_retargeter is not None:
            self.hand_retargeter.commit(command.hand_qpos, timestamp_ns=submit_end_ns)
        self.ik_reference = IKReference(command.arm_qpos, ik_anchor, arm_published_ns)
        if eef_pose is not None:
            # 忘掉新目标被裁掉的平移，手势反向时从工作空间边沿立即响应。
            wrist_origin, eef_origin, rotation_right = cast(
                tuple[np.ndarray, np.ndarray, Rotation], self.anchor,
            )
            self.anchor = wrist_origin, eef_origin - discarded_position_m, rotation_right
            self.previous_eef_target = eef_pose
