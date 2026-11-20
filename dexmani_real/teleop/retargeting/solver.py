"""基于官方 DexPilot 向量目标的有界 XHand 重定向。"""

from collections.abc import Callable, Sequence
from math import expm1

import nlopt
import numpy as np

from dexmani_real.config.teleop import HandRetargetingConfig
from dexmani_real.config.xhand import HandConfig
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
    XHAND_RIGHT_URDF_PATH,
    XHAND_SDK_JOINT_NAMES,
)
from dexmani_real.teleop.retargeting.landmarks import (
    FINGER_CHAINS,
    HandCalibration,
    HandGeometry,
    HandGeometryRejected,
    prepare_landmarks,
    project_direction,
)

FINGER_FRAME_NAMES = (
    ("right_hand_thumb_rota_link1", "right_hand_thumb_rota_link2"),
    ("right_hand_index_rota_link1", "right_hand_index_rota_link2"),
    ("right_hand_mid_link1", "right_hand_mid_link2"),
    ("right_hand_ring_link1", "right_hand_ring_link2"),
    ("right_hand_pinky_link1", "right_hand_pinky_link2"),
)


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


class HandRetargeter:
    def __init__(self, config: HandRetargetingConfig, hand_config: HandConfig):
        import torch
        from dex_retargeting.optimizer import DexPilotOptimizer
        from dex_retargeting.robot_wrapper import RobotWrapper

        config.validate()
        self.config = config
        self.limits = np.column_stack((hand_config.qpos_min_rad, hand_config.qpos_max_rad))
        self.calibration: HandCalibration | None = None
        self.reference: tuple[np.ndarray, int] | None = None
        self.rejection_reason: str | None = None
        self.fk = HandFK()
        model = self.fk.model
        names = [
            name for pair, tip in zip(FINGER_FRAME_NAMES, XHAND_FINGERTIP_LINK_NAMES)
            for name in (*pair, tip)
        ]
        frame_ids = [model.getFrameId(name) for name in names]
        if max(frame_ids) >= model.nframes:
            raise ValueError("XHand retargeting frames are missing from the URDF")
        neutral = self.fk.forward(np.zeros(HAND_DOF), frame_ids)
        positions = neutral[:, :3, 3].reshape(5, 3, 3)
        segments = np.diff(positions, axis=1)
        segment_lengths = np.linalg.norm(segments, axis=2)
        origins, axes = self.fk.neutral_joint_geometry()
        # 目标 Frame 的父关节就是屈曲关节；沿 URDF 链取基部，避免维护第二份关节表。
        flex_joints = [
            [model.frames[fid].parent for fid in frame_ids[start:start + 2]]
            for start in range(0, len(frame_ids), 3)
        ]
        flex_indices = np.array([
            [XHAND_SDK_JOINT_NAMES.index(model.names[jid]) for jid in pair]
            for pair in flex_joints
        ])
        flex_axes = axes[flex_indices]
        normals = flex_axes[:, 0]
        if not np.allclose(np.cross(normals, flex_axes[:, 1]), 0, atol=1e-9):
            raise ValueError("XHand flexion axes must be parallel")
        if np.any(segment_lengths <= 0) or not np.allclose(
            np.sum(segments * normals[:, None], axis=2), 0, atol=1e-9,
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
            neutral_directions=segments / segment_lengths[:, :, None], base_axes=base_axes,
            base_indices=np.array(base_indices), flex_indices=flex_indices,
            flex_axes=flex_axes,
            flex_coupling=np.sum(flex_axes[:, 0] * flex_axes[:, 1], axis=1),
            limit_centers=self.limits.mean(axis=1),
            base_directions=project_direction(base_vectors, base_axes),
        )
        self._dexpilot = DexPilotOptimizer(
            RobotWrapper(str(XHAND_RIGHT_URDF_PATH)), list(XHAND_SDK_JOINT_NAMES),
            list(XHAND_FINGERTIP_LINK_NAMES), "right_hand_link",
            huber_delta=config.huber_delta_m, norm_delta=0.0, project_dist=0.0, escape_dist=0.0,
        )
        # 上游按腕点以及每根手指的末点编号，行顺序为拇指对指、其他指尖配对、腕到各指尖。
        self._origin, self._task = self._dexpilot.target_link_human_indices // FINGER_CHAINS.shape[1]
        # 连续权重直接乘在上游逐行 Huber 损失上，继续由上游计算向量梯度。
        huber_loss = self._dexpilot.huber_loss
        row_scales = torch.ones(len(self._task), dtype=torch.float64)
        self._dexpilot.huber_loss = lambda distance, zero: huber_loss(distance, zero) * row_scales
        self._contact_scales = row_scales.numpy()[:10]

        self.optimizer = self._dexpilot.opt
        self.optimizer.set_maxeval(config.max_evaluations)
        self.optimizer.set_maxtime(config.max_solve_time_s)
        self.optimizer.set_ftol_abs(1e-7)

    def calibrate(self, measurements: Sequence[np.ndarray] | np.ndarray) -> None:
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
        calibration, reference = self.calibration, self.reference
        if calibration is None or reference is None:
            raise RuntimeError("hand retargeting requires calibration and a measured reset pose")
        qpos, reference_ns = reference
        if timestamp_ns < reference_ns:
            raise ValueError("hand computation time moved backwards")
        dt_s = (timestamp_ns - reference_ns) * 1e-9
        self.rejection_reason = None
        if dt_s == 0:
            return qpos.copy()
        tau = self.config.smoothing_tau_s
        alpha = 1.0 if tau == 0 else -expm1(-dt_s / tau)
        try:
            points = prepare_landmarks(landmarks)
            objective = self._objective(points, calibration, qpos, dt_s)
            # 在滤波前放宽速度界，使滤波后的目标仍可达到配置的最大关节速度。
            step = self.config.max_joint_speed_rad_s * dt_s / alpha
            lower = np.maximum(self.limits[:, 0], qpos - step)
            upper = np.minimum(self.limits[:, 1], qpos + step)
            self.optimizer.set_lower_bounds(lower)
            self.optimizer.set_upper_bounds(upper)
            self.optimizer.set_min_objective(objective)
            solution = self.optimizer.optimize(qpos.copy())
            if not np.isfinite(self.optimizer.last_optimum_value()):
                raise ValueError("hand solver objective is non-finite")
            solution = validate_hand_target(solution, lower, upper)
            if alpha == 1.0:
                return solution
            return validate_hand_target(qpos + alpha * (solution - qpos), *self.limits.T)
        except nlopt.RoundoffLimited as exc:
            self.rejection_reason = f"hand solver failed: {type(exc).__name__}: {exc}"
        except (HandGeometryRejected, HandTargetRejected) as exc:
            self.rejection_reason = str(exc)
        return None

    def commit(self, qpos: np.ndarray, *, timestamp_ns: int) -> None:
        # 控制线程在发送正常返回后更新最后提交参考。
        if self.reference is None or timestamp_ns < self.reference[1]:
            raise ValueError("committed hand time moved backwards or reset is missing")
        self.reference = qpos.copy(), int(timestamp_ns)

    def _objective(
        self, points: np.ndarray, calibration: HandCalibration, qpos: np.ndarray, dt_s: float,
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        projected = np.zeros((6, 3))
        projected[1:] = calibration.project_fingertips(
            points, self.geometry, fk=self.fk, tolerance=self.config.bone_length_tolerance,
        )
        target_vectors = projected[self._task] - projected[self._origin]
        human_origin, human_task = self._dexpilot.target_link_human_indices[:, :10]
        human_vectors = points[human_task] - points[human_origin]
        gaps = np.linalg.norm(human_vectors, axis=1)
        full = self.config.pinch_full_m
        start = self.config.pinch_start_m
        gains = _smoothstep((start - gaps) / (start - full))
        thumb_gaps = gaps[:4]
        thumb_gains = gains[:4]
        # 留给无名指主导动作的空间随领先幅度连续变化。
        thumb_gains *= _smoothstep((thumb_gaps[2] - thumb_gaps) / 0.005)
        pair_gains = gains[4:]
        pair_gains[:] = np.minimum(
            pair_gains, np.minimum(thumb_gains[self._origin[4:10] - 2], thumb_gains[self._task[4:10] - 2]),
        )
        target_thumb_vectors = target_vectors[:4]
        projected_gaps = np.linalg.norm(target_thumb_vectors, axis=1)
        shrink = np.minimum(1.0, np.maximum(thumb_gaps, full) / np.maximum(projected_gaps, 1e-9))
        target_thumb_vectors *= (1.0 + thumb_gains * (shrink - 1.0))[:, None]
        target_vectors[4:10] += pair_gains[:, None] * (human_vectors[4:] - target_vectors[4:10])
        self._contact_scales[:4] = 1.0 + 15.0 * thumb_gains
        self._contact_scales[4:] = 1.0 + 5.0 * pair_gains
        dexpilot_objective = self._dexpilot.get_objective_function(target_vectors, np.empty(0), qpos)
        damping = self.config.damping_s / dt_s

        def objective(candidate: np.ndarray, gradient: np.ndarray) -> float:
            value = dexpilot_objective(candidate, gradient)
            difference = candidate - qpos
            if gradient.size:
                gradient[:] = 2.0 * gradient + damping * difference
            return 2.0 * value + 0.5 * damping * np.dot(difference, difference)

        return objective
