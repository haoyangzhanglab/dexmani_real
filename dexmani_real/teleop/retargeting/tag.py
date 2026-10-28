"""TAG：指尖位置拟合与捏合细化。"""

import nlopt
import numpy as np
import pinocchio as pin

from dexmani_real.geometry.kinematics import HandFK
from dexmani_real.robot.model import HAND_JOINT_SHAPE, XHAND_RIGHT_URDF_PATH
from dexmani_real.teleop.config import TAGConfig
from dexmani_real.teleop.retargeting.landmarks import prepare_landmarks

ROBOT_FINGER_LENGTHS = np.array([0.161, 0.208, 0.206, 0.204, 0.145])
PINCH_SKIP_THRESHOLD = 0.01
FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20])


class TAGRetargeter:
    def __init__(self, config: TAGConfig):
        config.validate()
        self.config = config
        self.fk = HandFK(urdf_path=XHAND_RIGHT_URDF_PATH)
        self.urdf_to_sdk = np.argsort(self.fk.sdk_to_urdf)
        self.lower = self.fk.model.lowerPositionLimit
        self.upper = self.fk.model.upperPositionLimit
        self.dof = self.fk.model.nv
        self.finger_scale = (
            ROBOT_FINGER_LENGTHS / config.human_finger_lengths * config.finger_scale_boost
        )[:, None]

        self.position_optimizer = nlopt.opt(nlopt.LD_LBFGS, self.dof)
        self.pinch_optimizer = nlopt.opt(nlopt.LD_SLSQP, self.dof)
        for optimizer, objective, tolerance, maxeval in (
            (self.position_optimizer, self.position_objective, 1e-4, 80),
            (self.pinch_optimizer, self.pinch_objective, 1e-6, 100),
        ):
            optimizer.set_lower_bounds(self.lower)
            optimizer.set_upper_bounds(self.upper)
            optimizer.set_ftol_abs(tolerance)
            optimizer.set_maxeval(maxeval)
            optimizer.set_min_objective(objective)
        self.reset()

    def reset(self, initial_qpos=None):
        if initial_qpos is None:
            self.previous_solution = (self.lower + self.upper) / 2
        else:
            qpos = np.asarray(initial_qpos, dtype=np.float64)
            if qpos.shape != HAND_JOINT_SHAPE or not np.isfinite(qpos).all():
                raise ValueError("TAG initial_qpos must be a finite (12,) array")
            self.previous_solution = np.clip(qpos[self.fk.sdk_to_urdf], self.lower, self.upper)
        self.pinch_factors = np.zeros(4)

    def retarget(self, landmarks):
        mano = prepare_landmarks(
            landmarks,
            pinky_scale=self.config.pinky_scale,
            pinky_palm_scale=self.config.pinky_palm_scale,
        )
        if mano is None:
            return None
        tips = mano[FINGERTIP_INDICES] - mano[0]
        return self.solve(tips)[self.urdf_to_sdk]

    def solve(self, fingertips: np.ndarray) -> np.ndarray:
        self.target_fingertips = fingertips * self.finger_scale
        position = self.bounded_solution(self.position_optimizer.optimize(self.previous_solution))

        distances = np.linalg.norm(fingertips[1:] - fingertips[0], axis=1)
        factors = np.clip(
            (self.config.pinch_start_dist_m - distances)
            / (self.config.pinch_start_dist_m - self.config.pinch_full_dist_m),
            0,
            1,
        )
        alpha = self.config.pinch_ema_alpha
        self.pinch_factors = (1 - alpha) * self.pinch_factors + alpha * factors

        solution = position
        if self.pinch_factors.max() >= PINCH_SKIP_THRESHOLD:
            self.position_solution = position
            try:
                solution = self.bounded_solution(self.pinch_optimizer.optimize(position))
            except nlopt.RoundoffLimited:
                # 第一阶段已通过有限性与限位处理，仍可作为本帧解及下一帧初值。
                solution = position
        self.previous_solution = solution
        return solution

    def bounded_solution(self, qpos) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != HAND_JOINT_SHAPE or not np.isfinite(qpos).all():
            raise ValueError("TAG solver must return a finite (12,) array")
        return np.clip(qpos, self.lower, self.upper)

    def update_kinematics(self, qpos, with_jacobian):
        if with_jacobian:
            pin.computeJointJacobians(self.fk.model, self.fk.data, qpos)
        else:
            pin.forwardKinematics(self.fk.model, self.fk.data, qpos)
        pin.updateFramePlacements(self.fk.model, self.fk.data)

    def position_objective(self, qpos, grad):
        """Σ‖p_tip(q) − target‖² + smooth_weight·‖q − q_prev‖²。"""
        self.update_kinematics(qpos, bool(grad.size))
        error = qpos - self.previous_solution
        value = self.config.smooth_weight * np.dot(error, error)
        if grad.size:
            grad[:] = 2 * self.config.smooth_weight * error
        for index, frame in enumerate(self.fk.fingertip_frame_ids):
            error = self.fk.data.oMf[frame].translation - self.target_fingertips[index]
            value += np.dot(error, error)
            if grad.size:
                jacobian = pin.getFrameJacobian(
                    self.fk.model,
                    self.fk.data,
                    frame,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )[:3]
                grad[:] += 2 * (error @ jacobian)
        return float(value)

    def pinch_objective(self, qpos, grad):
        """两项关节正则 + Σ pinch_weight·factor²·‖p_finger(q) − p_thumb(q)‖²。"""
        self.update_kinematics(qpos, bool(grad.size))
        value = 0.0
        if grad.size:
            grad[:] = 0
        for weight, reference in (
            (self.config.reg_stage1_weight, self.position_solution),
            (self.config.reg_last_weight, self.previous_solution),
        ):
            error = qpos - reference
            value += weight * np.dot(error, error)
            if grad.size:
                grad[:] += 2 * weight * error

        thumb, *fingers = self.fk.fingertip_frame_ids
        thumb_position = self.fk.data.oMf[thumb].translation
        if grad.size:
            thumb_jacobian = pin.getFrameJacobian(
                self.fk.model,
                self.fk.data,
                thumb,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3]
        for frame, factor in zip(fingers, self.pinch_factors):
            if factor < 1e-4:
                continue
            error = self.fk.data.oMf[frame].translation - thumb_position
            weight = self.config.pinch_base_weight * factor**2
            value += weight * np.dot(error, error)
            if grad.size:
                jacobian = pin.getFrameJacobian(
                    self.fk.model,
                    self.fk.data,
                    frame,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )[:3]
                grad[:] += 2 * weight * (error @ (jacobian - thumb_jacobian))
        return float(value)
