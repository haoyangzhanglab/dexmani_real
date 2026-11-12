"""运行限位与运动目标端点约束。"""

import numpy as np


def workspace_start_rejection(position, lower, upper):
    """检查调用方的起点接纳范围，不修改实测位置。"""
    if np.any(position < lower) or np.any(position > upper):
        return "实测末端超出允许的起点范围；请手动就位或调整范围后重试"
    return None


class MotionConstraints:
    """EEF/jog 目标端点的几何检查。"""

    def __init__(self, arm, environment):
        self.limits = np.column_stack((arm.joint_limit_lower, arm.joint_limit_upper))
        self.workspace = environment.workspace.as_array()
        tolerance = environment.eef_workspace_tolerance_m
        self.workspace_with_tolerance = self.workspace + (-tolerance, tolerance)

    def endpoint_rejection(self, *, position):
        if np.any(position < self.workspace_with_tolerance[:, 0]) or np.any(
            position > self.workspace_with_tolerance[:, 1]
        ):
            return "final_eef_workspace"
        return None
