"""有界、可诊断的 XHand 在线重定向；只有提交成功才推进命令状态。"""

import time
from math import expm1

import nlopt
import numpy as np

from dexmani_real.geometry.kinematics import HandFK
from dexmani_real.robot.model import (
    HAND_JOINT_SHAPE,
    XHAND_FINGERTIP_LINK_NAMES,
)
from dexmani_real.teleop.retargeting.landmarks import HandCalibration, prepare_landmarks
from dexmani_real.teleop.retargeting.objective import HandObjective

FINGER_FRAME_NAMES = (
    ("right_hand_thumb_rota_link1", "right_hand_thumb_rota_link2"),
    ("right_hand_index_rota_link1", "right_hand_index_rota_link2"),
    ("right_hand_mid_link1", "right_hand_mid_link2"),
    ("right_hand_ring_link1", "right_hand_ring_link2"),
    ("right_hand_pinky_link1", "right_hand_pinky_link2"),
)


class AdaptiveHandRetargeter:
    def __init__(self, config, hand_config, *, control_hz):
        config.validate()
        hand_config.validate_home_pose()
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("retargeting control_hz must be finite and positive")
        self.config, self.nominal_dt_s = config, 1.0 / control_hz
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
        # 同一 loss 单位的容差也用于下方 seed 改善接受，保持既有双重语义。
        self.optimizer.set_ftol_abs(config.ftol_abs)
        self.calibration = None
        self.previous_command = None
        self.command_timestamp_ns = None
        self.rejection_reason = None
        self.last_diagnostics = {}

    def calibrate(self, measurements):
        self.calibration = HandCalibration.from_measurements(measurements, self.config)

    def calibration_metadata(self):
        if self.calibration is None:
            raise RuntimeError("hand retargeting requires morphology calibration")
        return dict(
            sample_count=self.calibration.sample_count,
            human_bone_lengths_m=self.calibration.bone_lengths.tolist(),
            human_palm_width_m=self.calibration.palm_width,
        )

    def reset(self, initial_qpos, *, timestamp_ns):
        qpos = np.asarray(initial_qpos, dtype=np.float64)
        if qpos.shape != HAND_JOINT_SHAPE or not np.isfinite(qpos).all():
            raise ValueError("hand initial_qpos must be a finite (12,) array")
        if timestamp_ns <= 0:
            raise ValueError("hand reset timestamp must be positive")
        self.previous_command = np.clip(qpos, self.limits[:, 0], self.limits[:, 1])
        self.command_timestamp_ns = int(timestamp_ns)
        self.rejection_reason = None
        self.last_diagnostics = {}

    def retarget(self, landmarks, *, timestamp_ns):
        if self.calibration is None or self.previous_command is None:
            raise RuntimeError("hand retargeting requires calibration and a measured reset pose")
        if timestamp_ns < self.command_timestamp_ns:
            raise ValueError("hand observation time moved backwards")
        # 准备阶段 reset 后用同一观测预计算，零间隔使用名义控制周期。
        dt_s = (timestamp_ns - self.command_timestamp_ns) * 1e-9 or self.nominal_dt_s
        self.rejection_reason = None
        self.last_diagnostics = {}
        try:
            points = prepare_landmarks(landmarks)
            targets = self.calibration.targets(
                points, self.robot_points, self.config,
                robot_segments=self.robot_segments, robot_palm_width=self.robot_palm_width,
                thumb_base_xy=self.thumb_base_xy, thumb_base_radius=self.thumb_base_radius,
            )
        except ValueError as exc:
            self.rejection_reason = str(exc)
            return None
        tau = self.config.smoothing_tau_s
        alpha = 1.0 if tau == 0 else -expm1(-dt_s / tau)
        objective = HandObjective(
            self.fk, self.frame_ids, targets, self.previous_command, dt_s, self.config,
        )
        # 搜索的是滤波前候选；补偿 alpha 后，最终命令仍可达到 v_max。
        # 若候选先限为 v_max*dt 再滤波，高频时实际速率上限会降为 alpha*v_max。
        step = self.config.max_joint_speed_rad_s * dt_s / alpha
        lower = np.maximum(self.limits[:, 0], self.previous_command - step)
        upper = np.minimum(self.limits[:, 1], self.previous_command + step)
        self.optimizer.set_lower_bounds(lower)
        self.optimizer.set_upper_bounds(upper)
        self.optimizer.set_min_objective(objective)
        seed = self.previous_command.copy()
        initial_loss = objective(seed, np.empty(0))
        if not np.isfinite(initial_loss):
            self.rejection_reason = "hand solver seed loss is non-finite"
            self.last_diagnostics = dict(initial_loss=None, failure_reason=self.rejection_reason)
            return None
        started = time.perf_counter()
        try:
            solution = np.asarray(self.optimizer.optimize(seed), dtype=np.float64)
        except (RuntimeError, nlopt.RoundoffLimited) as exc:
            self.rejection_reason = f"hand solver failed: {type(exc).__name__}: {exc}"
            self.last_diagnostics = dict(
                status=int(self.optimizer.last_optimize_result()),
                solve_ms=(time.perf_counter() - started) * 1000,
            )
            return None
        status = int(self.optimizer.last_optimize_result())
        solve_ms = (time.perf_counter() - started) * 1000
        if (
            status <= 0 or solution.shape != HAND_JOINT_SHAPE
            or not np.isfinite(solution).all()
            or np.any(solution < lower - 1e-9) or np.any(solution > upper + 1e-9)
        ):
            self.rejection_reason = f"hand solver returned an invalid bounded solution (status={status})"
            self.last_diagnostics = dict(
                status=status, evaluations=self.optimizer.get_numevals(), solve_ms=solve_ms,
                initial_loss=initial_loss, dt_s=dt_s, failure_reason=self.rejection_reason,
            )
            return None
        solution = np.clip(solution, lower, upper)
        final_loss = objective(solution, np.empty(0))
        if not np.isfinite(final_loss) or final_loss > initial_loss + self.config.ftol_abs:
            self.rejection_reason = "hand solver did not improve the committed-command seed"
            self.last_diagnostics = dict(
                status=status, evaluations=self.optimizer.get_numevals(), solve_ms=solve_ms,
                initial_loss=initial_loss, dt_s=dt_s,
                final_loss=final_loss if np.isfinite(final_loss) else None,
                failure_reason="non-finite final loss" if not np.isfinite(final_loss) else self.rejection_reason,
            )
            return None
        # 同一个已提交目标同时作为 warm start、阻尼中心和唯一滤波状态。
        command = self.previous_command + alpha * (solution - self.previous_command)
        # 凸组合理论上不越限，但浮点减加可在限位上产生舍入量级的越界。
        # 消除舍入误差后仍由 commit 严格校验，不放宽关节或最终速率限值。
        return np.clip(command, self.limits[:, 0], self.limits[:, 1])

    def commit(self, qpos, *, timestamp_ns):
        qpos = np.asarray(qpos, dtype=np.float64)
        if (
            qpos.shape != HAND_JOINT_SHAPE or not np.isfinite(qpos).all()
            or np.any(qpos < self.limits[:, 0]) or np.any(qpos > self.limits[:, 1])
        ):
            raise ValueError("committed hand command must be finite and within operating limits")
        if self.command_timestamp_ns is None or timestamp_ns < self.command_timestamp_ns:
            raise ValueError("committed hand time moved backwards or reset is missing")
        self.previous_command = qpos.copy()
        self.command_timestamp_ns = int(timestamp_ns)
