"""有界 XHand 重定向，使用实测或已确认提交的关节参考。"""

from math import expm1

import nlopt
import numpy as np

from dexmani_real.control.actions import StartPoseRejected
from dexmani_real.geometry.kinematics import HandFK
from dexmani_real.robot.commands import (
    HandTargetRejected,
    hand_reference_from_feedback,
    validate_hand_target,
)
from dexmani_real.robot.model import (
    HAND_DOF,
    XHAND_FINGERTIP_LINK_NAMES,
    XHAND_SDK_JOINT_NAMES,
)
from dexmani_real.teleop.retargeting.landmarks import (
    HandCalibration,
    HandGeometry,
    HandGeometryRejected,
    prepare_landmarks,
    project_direction,
)
from dexmani_real.teleop.retargeting.objective import HandObjective

LOSS_TOLERANCE = 1e-7

FINGER_FRAME_NAMES = (
    ("right_hand_thumb_rota_link1", "right_hand_thumb_rota_link2"),
    ("right_hand_index_rota_link1", "right_hand_index_rota_link2"),
    ("right_hand_mid_link1", "right_hand_mid_link2"),
    ("right_hand_ring_link1", "right_hand_ring_link2"),
    ("right_hand_pinky_link1", "right_hand_pinky_link2"),
)


class HandRetargeter:
    def __init__(self, config, hand_config):
        config.validate()
        self.config = config
        self.fk = HandFK()
        model = self.fk.model
        self.limits = np.column_stack((hand_config.qpos_min_rad, hand_config.qpos_max_rad))
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
        self.calibration = None
        self.reference: tuple[np.ndarray, int] | None = None
        self.rejection_reason = None

    def calibrate(self, measurements):
        self.calibration = HandCalibration.from_measurements(measurements, self.config)

    def reset(self, initial_qpos: np.ndarray, *, timestamp_ns: int) -> None:
        if timestamp_ns <= 0:
            raise ValueError("hand reset timestamp must be positive")
        try:
            qpos = hand_reference_from_feedback(initial_qpos, *self.limits.T)
        except HandTargetRejected as exc:
            raise StartPoseRejected(f"手部实测参考不满足要求，请先就位后再次按 B。{exc}") from exc
        self.reference = qpos, int(timestamp_ns)
        self.rejection_reason = None

    def retarget(self, landmarks: np.ndarray, *, timestamp_ns: int) -> np.ndarray | None:
        reference = self.reference
        if self.calibration is None or reference is None:
            raise RuntimeError("hand retargeting requires calibration and a measured reset pose")
        qpos, reference_ns = reference
        if timestamp_ns < reference_ns:
            raise ValueError("hand computation time moved backwards")
        dt_s = (timestamp_ns - reference_ns) * 1e-9
        self.rejection_reason = None
        if dt_s == 0:
            return qpos.copy()
        try:
            points = prepare_landmarks(landmarks)
            targets = self.calibration.targets(
                points, self.geometry, self.config, fk=self.fk, frame_ids=self.frame_ids,
            )
        except HandGeometryRejected as exc:
            self.rejection_reason = str(exc)
            return None
        tau = self.config.smoothing_tau_s
        alpha = 1.0 if tau == 0 else -expm1(-dt_s / tau)
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
        try:
            validate_hand_target(solution, lower, upper)
            # 实测起点或最后提交参考同时作为 warm start、阻尼中心和滤波状态。
            command = solution if alpha == 1.0 else qpos + alpha * (solution - qpos)
            return validate_hand_target(command, *self.limits.T)
        except ValueError as exc:
            self.rejection_reason = str(exc)
            return None

    def commit(self, qpos: np.ndarray, *, timestamp_ns: int) -> None:
        # 控制线程仅在发送正常返回后更新最后提交参考。
        if self.reference is None or timestamp_ns < self.reference[1]:
            raise ValueError("committed hand time moved backwards or reset is missing")
        self.reference = qpos.copy(), int(timestamp_ns)
