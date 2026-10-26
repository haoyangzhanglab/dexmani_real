"""关节等价角、连续性与运动目标约束。"""

import numpy as np

from dexmani_real.geometry.kinematics.arm_fk import ArmFK
from dexmani_real.robot.model import XARM7_EQUIVALENT_JOINT_MASK

JUMP_LIMIT_RAD = np.deg2rad((30, 30, 30, 35, 40, 40, 40))
EQUIVALENT_JOINTS = np.asarray(XARM7_EQUIVALENT_JOINT_MASK, dtype=bool)


def joint_target_rejection(q, current, previous, limits):
    if q.shape != (7,) or not np.isfinite(q).all():
        return "invalid_target"
    lower, upper = limits.T
    if np.any(q < lower) or np.any(q > upper):
        return "operational_limits"
    # 硬件接收绝对角度，跳变检查不能使用周期最短距离。
    if np.any(np.abs(q - previous) > JUMP_LIMIT_RAD):
        return "jump"
    if np.max(np.abs(q - current)) > np.deg2rad(150):
        return "hw_dist"
    return None


def wrap_nearest_equivalent(qpos, reference, joint_limit_lower, joint_limit_upper):
    result = np.asarray(qpos, dtype=np.float64).copy()
    lower = np.asarray(joint_limit_lower)[EQUIVALENT_JOINTS]
    upper = np.asarray(joint_limit_upper)[EQUIVALENT_JOINTS]
    q = result[EQUIVALENT_JOINTS]
    ref = np.asarray(reference)[EQUIVALENT_JOINTS]
    period = 2 * np.pi
    k_min = np.ceil((lower - q) / period)
    k_max = np.floor((upper - q) / period)
    nearest = np.clip(np.rint((ref - q) / period), k_min, k_max)
    # 无可用等价分支时保留原值，由调用方裁剪或拒绝。
    result[EQUIVALENT_JOINTS] = np.where(k_min <= k_max, q + nearest * period, q)
    return result


def equivalent_joint_delta(qpos, reference_qpos):
    delta = qpos - reference_qpos
    delta[EQUIVALENT_JOINTS] = (delta[EQUIVALENT_JOINTS] + np.pi) % (2 * np.pi) - np.pi
    return delta


class MotionConstraints:
    """检查目标端点；HOME 路径由 HomePlanner 单独检查。"""

    def __init__(
        self,
        collision_model,
        *,
        joint_lower_rad,
        joint_upper_rad,
        workspace,
        workspace_margin_m,
    ):
        self.collision_model = collision_model
        self.target_fk = ArmFK()
        self.limits = np.column_stack((joint_lower_rad, joint_upper_rad))
        self.workspace = np.array(workspace, dtype=np.float64, copy=True)
        self.margin = workspace_margin_m

    def joint_rejection(self, target, current, previous):
        return joint_target_rejection(target, current, previous, self.limits)

    def rejection(self, target, current, previous, *, collision_hand_qpos):
        reason = self.joint_rejection(target, current, previous)
        if reason is not None:
            return reason
        self.collision_model.set_hand_qpos(collision_hand_qpos)
        return self.endpoint_rejection(target)

    def endpoint_rejection(self, target, *, position=None):
        if position is None:
            position, _ = self.target_fk.compute(target)
        if np.any(position < self.workspace[:, 0] - self.margin) or np.any(
            position > self.workspace[:, 1] + self.margin
        ):
            return "final_eef_workspace"
        return self.collision_model.rejection(target)
