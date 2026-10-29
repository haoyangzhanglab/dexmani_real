"""关节等价角、运行限位与运动目标端点约束。"""

import numpy as np

from dexmani_real.control.collision import CollisionModel
from dexmani_real.geometry.kinematics import ArmFK
from dexmani_real.robot.model import XARM7_EQUIVALENT_JOINT_MASK

EQUIVALENT_JOINTS = np.asarray(XARM7_EQUIVALENT_JOINT_MASK, dtype=bool)


def joint_target_rejection(q, limits):
    lower, upper = limits.T
    if np.any(q < lower) or np.any(q > upper):
        return "operational_limits"
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


class MotionConstraints:
    """检查目标端点；HOME 路径由 HomePlanner 单独检查。"""

    def __init__(self, arm, environment):
        self.collision_model = CollisionModel(static_boxes=environment.static_boxes)
        self.target_fk = ArmFK()
        self.limits = np.column_stack((arm.joint_limit_lower, arm.joint_limit_upper))
        self.workspace = environment.workspace.as_array()
        self.margin = environment.eef_workspace_margin_m

    def rejection(self, target, *, collision_hand_qpos):
        reason = joint_target_rejection(target, self.limits)
        if reason is not None:
            return reason
        self.collision_model.set_hand_qpos(collision_hand_qpos)
        return self.endpoint_rejection(target)

    def endpoint_rejection(self, target, *, position=None):
        if position is None:
            position, _ = self.target_fk.forward(target)
        if np.any(position < self.workspace[:, 0] - self.margin) or np.any(
            position > self.workspace[:, 1] + self.margin
        ):
            return "final_eef_workspace"
        return self.collision_model.rejection(target)
