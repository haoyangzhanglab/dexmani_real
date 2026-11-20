"""有界 XHand 重定向，使用实测或已确认提交的关节参考。"""

import nlopt
import numpy as np

from dexmani_real.geometry.kinematics import HandFK
from dexmani_real.robot.commands import validate_hand_target
from dexmani_real.robot.model import (
    HAND_DOF,
    XHAND_FINGERTIP_LINK_NAMES,
    XHAND_SDK_JOINT_NAMES,
)
from dexmani_real.teleop.retargeting.base import HandRetargeterBase
from dexmani_real.teleop.retargeting.landmarks import HandGeometry, project_direction
from dexmani_real.teleop.retargeting.objective import HandObjective

LOSS_TOLERANCE = 1e-7

FINGER_FRAME_NAMES = (
    ("right_hand_thumb_rota_link1", "right_hand_thumb_rota_link2"),
    ("right_hand_index_rota_link1", "right_hand_index_rota_link2"),
    ("right_hand_mid_link1", "right_hand_mid_link2"),
    ("right_hand_ring_link1", "right_hand_ring_link2"),
    ("right_hand_pinky_link1", "right_hand_pinky_link2"),
)


class HandRetargeter(HandRetargeterBase):
    def __init__(self, config, hand_config):
        super().__init__(config, hand_config)
        self.fk = HandFK()
        model = self.fk.model
        names = [
            name for pair, tip in zip(FINGER_FRAME_NAMES, XHAND_FINGERTIP_LINK_NAMES)
            for name in (*pair, tip)
        ]
        self.frame_ids = [model.getFrameId(name) for name in names]
        if max(self.frame_ids) >= model.nframes:
            raise ValueError("XHand retargeting frames are missing from the URDF")
        neutral = self.fk.forward(np.zeros(HAND_DOF), self.frame_ids)
        positions = neutral[:, :3, 3].reshape(5, 3, 3)
        segments = np.linalg.norm(np.diff(positions, axis=1), axis=2)
        origins, axes = self.fk.neutral_joint_geometry()
        # 目标 Frame 的父关节就是屈曲关节；沿 URDF 链取基部，避免维护第二份关节表。
        flex_joints = [
            [model.frames[fid].parent for fid in self.frame_ids[start:start + 2]]
            for start in range(0, len(self.frame_ids), 3)
        ]
        flex_indices = np.array([
            [XHAND_SDK_JOINT_NAMES.index(model.names[jid]) for jid in pair]
            for pair in flex_joints
        ])
        flex_axes = axes[flex_indices]
        normals = flex_axes[:, 0]
        if not np.allclose(np.cross(normals, flex_axes[:, 1]), 0, atol=1e-9):
            raise ValueError("XHand flexion axes must be parallel")
        if np.any(segments <= 0) or not np.allclose(
            np.sum(np.diff(positions, axis=1) * normals[:, None], axis=2), 0, atol=1e-9,
        ):
            raise ValueError("XHand target segments must lie in their flexion planes")
        base_indices = [
            XHAND_SDK_JOINT_NAMES.index(model.names[model.parents[jid]])
            for jid, _ in flex_joints[:2]
        ]
        base_axes, base_origins = axes[base_indices], origins[base_indices]
        base_vectors = np.array([
            positions[0, 0] - base_origins[0], positions[1, 1] - positions[1, 0],
        ])
        base_vectors /= np.linalg.norm(base_vectors, axis=1)[:, None]
        self.geometry = HandGeometry(
            positions=positions, segments=segments, base_axes=base_axes,
            base_indices=np.array(base_indices), flex_indices=flex_indices,
            flex_axes=flex_axes, limits=self.limits,
            base_directions=project_direction(base_vectors, base_axes),
        )
        self.optimizer = nlopt.opt(nlopt.LD_SLSQP, HAND_DOF)
        self.optimizer.set_maxeval(config.max_evaluations)
        self.optimizer.set_maxtime(config.max_solve_time_s)
        # 优化器的绝对目标收敛容差。
        self.optimizer.set_ftol_abs(LOSS_TOLERANCE)

    def _retarget(self, points: np.ndarray, qpos: np.ndarray, dt_s: float, alpha: float) -> np.ndarray | None:
        targets = self.calibration.targets(
            points, self.geometry, self.config, fk=self.fk, frame_ids=self.frame_ids,
        )
        objective = HandObjective(
            self.fk, self.frame_ids, targets, qpos, dt_s, self.config,
        )
        # 搜索的是滤波前候选；补偿 alpha 后，最终命令仍可达到 v_max。
        # 若候选先限为 v_max*dt 再滤波，高频时实际速率上限会降为 alpha*v_max。
        step = self.config.max_joint_speed_rad_s * dt_s / alpha
        lower = np.maximum(self.limits[:, 0], qpos - step)
        upper = np.minimum(self.limits[:, 1], qpos + step)
        self.optimizer.set_lower_bounds(lower)
        self.optimizer.set_upper_bounds(upper)
        self.optimizer.set_min_objective(objective)
        seed = qpos.copy()
        try:
            solution = np.asarray(self.optimizer.optimize(seed), dtype=np.float64)
        except nlopt.RoundoffLimited as exc:
            self.rejection_reason = f"hand solver failed: {type(exc).__name__}: {exc}"
            return None
        status = int(self.optimizer.last_optimize_result())
        if status <= 0:
            raise RuntimeError(f"hand solver returned unsuccessful status={status}")
        if not np.isfinite(self.optimizer.last_optimum_value()):
            raise ValueError("hand solver objective is non-finite")
        validate_hand_target(solution, lower, upper)
        # 实测起点或最后提交参考同时作为 warm start、阻尼中心和滤波状态。
        return solution if alpha == 1.0 else qpos + alpha * (solution - qpos)
