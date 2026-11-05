"""有界 XHand 重定向；实测起点建立参考，提交尝试正常返回后推进参考。"""

from math import expm1

import nlopt
import numpy as np

from dexmani_real.control.actions import StartPoseRejected
from dexmani_real.geometry.kinematics import HandFK
from dexmani_real.robot.commands import hand_reference_from_feedback
from dexmani_real.robot.model import (
    HAND_JOINT_SHAPE,
    XHAND_FINGERTIP_LINK_NAMES,
    XHAND_SDK_JOINT_NAMES,
)
from dexmani_real.teleop.retargeting.landmarks import (
    HandCalibration,
    HandGeometryRejected,
    prepare_landmarks,
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
        self.limits = np.column_stack((hand_config.qpos_min_rad, hand_config.qpos_max_rad))
        model_limits = np.column_stack((
            self.fk.model.lowerPositionLimit, self.fk.model.upperPositionLimit,
        ))[self.fk.urdf_to_sdk]
        if np.any(self.limits[:, 0] < model_limits[:, 0] - 1e-6) or np.any(
            self.limits[:, 1] > model_limits[:, 1] + 1e-6,
        ):
            raise ValueError("hand command limits exceed the retargeting model limits")
        names = [
            name for pair, tip in zip(FINGER_FRAME_NAMES, XHAND_FINGERTIP_LINK_NAMES)
            for name in (*pair, tip)
        ]
        self.frame_ids = [self.fk.model.getFrameId(name) for name in names]
        thumb_cmc_id = self.fk.model.getFrameId("right_hand_thumb_bend_link")
        if max(*self.frame_ids, thumb_cmc_id) >= self.fk.model.nframes:
            raise ValueError("XHand retargeting frames are missing from the URDF")
        neutral = self.fk.forward(np.zeros(12), [*self.frame_ids, thumb_cmc_id])
        self.robot_points = neutral[:15, :3, 3].reshape(5, 3, 3)
        self.robot_segments = np.linalg.norm(np.diff(self.robot_points, axis=1), axis=2)
        self.robot_palm_width = np.linalg.norm(self.robot_points[1, 0] - self.robot_points[4, 0])
        self.thumb_base_xy = neutral[15, :2, 3]
        self.thumb_base_radius = np.linalg.norm(self.robot_points[0, 0, :2] - self.thumb_base_xy)
        self.optimizer = nlopt.opt(nlopt.LD_SLSQP, 12)
        self.optimizer.set_maxeval(config.max_evaluations)
        self.optimizer.set_maxtime(config.max_solve_time_s)
        # 优化器的绝对目标收敛容差。
        self.optimizer.set_ftol_abs(LOSS_TOLERANCE)
        self.calibration = None
        self.reference_qpos = None
        self.reference_timestamp_ns = None
        self.rejection_reason = None

    def calibrate(self, measurements):
        self.calibration = HandCalibration.from_measurements(measurements, self.config)

    def reset(self, initial_qpos, *, timestamp_ns):
        qpos = np.asarray(initial_qpos, dtype=np.float64)
        if timestamp_ns <= 0:
            raise ValueError("hand reset timestamp must be positive")
        try:
            reference = hand_reference_from_feedback(qpos, *self.limits.T)
        except ValueError as exc:
            if qpos.shape != HAND_JOINT_SHAPE or not np.isfinite(qpos).all():
                raise
            outside = (qpos < self.limits[:, 0]) | (qpos > self.limits[:, 1])
            joints = "; ".join(
                f"{XHAND_SDK_JOINT_NAMES[i]}={np.rad2deg(qpos[i]):.2f}° "
                f"(要求 {np.rad2deg(self.limits[i, 0]):.2f}–"
                f"{np.rad2deg(self.limits[i, 1]):.2f}°)"
                for i in np.flatnonzero(outside)
            )
            raise StartPoseRejected(f"本段未开始：手部起点超出操作范围：{joints}；请先就位后再次按 B。") from exc
        self.reference_qpos = reference
        self.reference_timestamp_ns = int(timestamp_ns)
        self.rejection_reason = None

    def retarget(self, landmarks, *, timestamp_ns):
        if self.calibration is None or self.reference_qpos is None:
            raise RuntimeError("hand retargeting requires calibration and a measured reset pose")
        if timestamp_ns < self.reference_timestamp_ns:
            raise ValueError("hand computation time moved backwards")
        dt_s = (timestamp_ns - self.reference_timestamp_ns) * 1e-9
        self.rejection_reason = None
        if dt_s == 0:
            return self.reference_qpos.copy()
        try:
            points = prepare_landmarks(landmarks)
            targets = self.calibration.targets(
                points, self.robot_points, self.config,
                robot_segments=self.robot_segments, robot_palm_width=self.robot_palm_width,
                thumb_base_xy=self.thumb_base_xy, thumb_base_radius=self.thumb_base_radius,
            )
        except HandGeometryRejected as exc:
            self.rejection_reason = str(exc)
            return None
        tau = self.config.smoothing_tau_s
        alpha = 1.0 if tau == 0 else -expm1(-dt_s / tau)
        objective = HandObjective(
            self.fk, self.frame_ids, targets, self.reference_qpos, dt_s, self.config,
        )
        # 搜索的是滤波前候选；补偿 alpha 后，最终命令仍可达到 v_max。
        # 若候选先限为 v_max*dt 再滤波，高频时实际速率上限会降为 alpha*v_max。
        step = self.config.max_joint_speed_rad_s * dt_s / alpha
        lower = np.maximum(self.limits[:, 0], self.reference_qpos - step)
        upper = np.minimum(self.limits[:, 1], self.reference_qpos + step)
        self.optimizer.set_lower_bounds(lower)
        self.optimizer.set_upper_bounds(upper)
        self.optimizer.set_min_objective(objective)
        seed = self.reference_qpos.copy()
        try:
            solution = np.asarray(self.optimizer.optimize(seed), dtype=np.float64)
        except nlopt.RoundoffLimited as exc:
            self.rejection_reason = f"hand solver failed: {type(exc).__name__}: {exc}"
            return None
        status = int(self.optimizer.last_optimize_result())
        if status <= 0:
            raise RuntimeError(f"hand solver returned unsuccessful status={status}")
        if (
            solution.shape != HAND_JOINT_SHAPE
            or not np.isfinite(solution).all()
            or np.any(solution < lower - 1e-9) or np.any(solution > upper + 1e-9)
        ):
            raise ValueError(f"hand solver returned an invalid bounded solution (status={status})")
        solution = np.clip(solution, lower, upper)
        if not np.isfinite(self.optimizer.last_optimum_value()):
            raise ValueError("hand solver objective is non-finite")
        # 实测起点或最后提交参考同时作为 warm start、阻尼中心和唯一滤波状态。
        command = self.reference_qpos + alpha * (solution - self.reference_qpos)
        # 凸组合理论上不越限，但浮点减加可在限位上产生舍入量级的越界。
        # 消除舍入误差，不放宽关节或最终速率限值。
        return np.clip(command, self.limits[:, 0], self.limits[:, 1])

    def commit(self, qpos, *, timestamp_ns):
        # 控制线程仅在发送正常返回后更新最后提交参考。
        if self.reference_timestamp_ns is None or timestamp_ns < self.reference_timestamp_ns:
            raise ValueError("committed hand time moved backwards or reset is missing")
        self.reference_qpos = qpos.copy()
        self.reference_timestamp_ns = int(timestamp_ns)
