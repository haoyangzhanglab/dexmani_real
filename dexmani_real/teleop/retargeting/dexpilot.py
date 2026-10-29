"""DexPilot 指尖向量拟合，输出 SDK 顺序的关节角。"""

import numpy as np
from dex_retargeting.optimizer import DexPilotOptimizer
from dex_retargeting.optimizer_utils import LPFilter
from dex_retargeting.robot_wrapper import RobotWrapper

from dexmani_real.config.teleop import DexPilotConfig
from dexmani_real.robot.model import (
    HAND_JOINT_SHAPE,
    XHAND_FINGERTIP_LINK_NAMES,
    XHAND_RIGHT_URDF_PATH,
    XHAND_SDK_JOINT_NAMES,
)
from dexmani_real.teleop.retargeting.landmarks import prepare_landmarks


class DexPilotRetargeter:
    def __init__(self, config: DexPilotConfig):
        config.validate()
        self.config = config
        robot = RobotWrapper(str(XHAND_RIGHT_URDF_PATH))
        self.optimizer = DexPilotOptimizer(
            robot, list(XHAND_SDK_JOINT_NAMES),
            finger_tip_link_names=list(XHAND_FINGERTIP_LINK_NAMES),
            wrist_link_name="right_hand_link",
            scaling=config.scaling_factor,
            project_dist=config.project_dist_m,
            escape_dist=config.escape_dist_m,
        )
        self.limits = robot.joint_limits[self.optimizer.idx_pin2target]
        self.optimizer.set_joint_limit(self.limits)
        self.landmark_pairs = self.optimizer.target_link_human_indices
        self.fixed_qpos = np.empty(0, dtype=np.float32)
        self.filter = LPFilter(config.low_pass_alpha)
        self.reset()

    def reset(self, initial_qpos=None):
        qpos = self.limits.mean(axis=1) if initial_qpos is None else initial_qpos
        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.shape != HAND_JOINT_SHAPE or not np.isfinite(qpos).all():
            raise ValueError("DexPilot initial_qpos must be a finite (12,) array")
        self.previous_solution = qpos.copy()
        self.filter.reset()
        self.filter.next(qpos.astype(np.float64))
        self.optimizer.projected[:] = False

    def retarget(self, landmarks):
        mano = prepare_landmarks(
            landmarks, pinky_scale=self.config.pinky_scale,
            pinky_palm_scale=self.config.pinky_palm_scale,
        )
        if mano is None:
            return None
        reference = (mano[self.landmark_pairs[1]] - mano[self.landmark_pairs[0]]).astype(np.float32)
        seed = np.clip(self.previous_solution, self.limits[:, 0], self.limits[:, 1])

        # 直接求解，避免上游 retarget() 捕获 RuntimeError 后返回旧目标。
        objective = self.optimizer.get_objective_function(
            reference, self.fixed_qpos, seed.astype(np.float32),
        )
        self.optimizer.opt.set_min_objective(objective)
        qpos = np.asarray(self.optimizer.opt.optimize(seed), dtype=np.float32)
        if qpos.shape != HAND_JOINT_SHAPE or not np.isfinite(qpos).all():
            raise ValueError("DexPilot solver must return a finite (12,) array")
        self.previous_solution = qpos
        return self.filter.next(qpos.astype(np.float64))
