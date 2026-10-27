"""TAG：指尖位置拟合与捏合细化。"""

import nlopt
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from dexmani_real.geometry.kinematics import HandFK, SDK_TO_URDF_IDX
from dexmani_real.robot.model import HAND_JOINT_SHAPE
from dexmani_real.teleop.config import TAGConfig
from dexmani_real.teleop.retargeting.landmarks import prepare_landmarks

FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20])
URDF_TO_SDK_IDX = np.argsort(SDK_TO_URDF_IDX)


class TAGRetargeter:
    def __init__(self, config: TAGConfig):
        config.validate()
        self.config = config
        self.fk = HandFK()
        self.lower = self.fk.model.lowerPositionLimit
        self.upper = self.fk.model.upperPositionLimit
        self.dof = self.fk.model.nv
        self.finger_scale = (
            np.asarray(config.robot_finger_lengths) / config.human_finger_lengths
            * config.finger_scale_boost
        )[:, None]
        self.mano_to_urdf = Rotation.from_euler("xyz", config.mano_to_urdf_euler).as_matrix()

        self.position_optimizer = nlopt.opt(nlopt.LD_LBFGS, self.dof)
        self.pinch_optimizer = nlopt.opt(nlopt.LD_SLSQP, self.dof)
        for optimizer, objective, tolerance, maxeval in (
            (self.position_optimizer, self.position_objective, config.ftol_abs_s1, config.maxeval_s1),
            (self.pinch_optimizer, self.pinch_objective, config.ftol_abs_s2, config.maxeval_s2),
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
            self.previous_solution = np.clip(qpos[SDK_TO_URDF_IDX], self.lower, self.upper)
        self.position_solution = None
        self.target_fingertips = None
        self.pinch_factors = np.zeros(4)

    def retarget(self, landmarks):
        mano = prepare_landmarks(
            landmarks, pinky_scale=self.config.pinky_scale,
            pinky_palm_scale=self.config.pinky_palm_scale,
        )
        if mano is None:
            return None
        tips = (mano[FINGERTIP_INDICES] - mano[0]) @ self.mano_to_urdf.T
        return self.solve(tips)[URDF_TO_SDK_IDX]

    def solve(self, fingertips):
        self.target_fingertips = fingertips * self.finger_scale
        if not np.isfinite(self.target_fingertips).all():
            raise ValueError("scaled fingertip positions must be finite")
        position = self.bounded_solution(self.position_optimizer.optimize(self.previous_solution))

        distances = np.linalg.norm(fingertips[1:] - fingertips[0], axis=1)
        factors = np.clip(
            (self.config.pinch_start_dist_m - distances)
            / (self.config.pinch_start_dist_m - self.config.pinch_full_dist_m),
            0, 1,
        )
        alpha = self.config.pinch_ema_alpha
        self.pinch_factors = (1 - alpha) * self.pinch_factors + alpha * factors

        solution = position
        if self.pinch_factors.max() >= self.config.pinch_skip_threshold:
            self.position_solution = position
            solution = self.bounded_solution(self.pinch_optimizer.optimize(position))
        self.previous_solution = solution
        return solution

    def bounded_solution(self, qpos):
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != HAND_JOINT_SHAPE or not np.isfinite(qpos).all():
            raise ValueError("TAG solver must return a finite (12,) array")
        return np.clip(qpos, self.lower, self.upper)

    def update_kinematics(self, qpos):
        pin.computeJointJacobians(self.fk.model, self.fk.data, qpos)
        pin.updateFramePlacements(self.fk.model, self.fk.data)

    def position_objective(self, qpos, grad):
        """Σ‖p_tip(q) − target‖² + smooth_weight·‖q − q_prev‖²。"""
        self.update_kinematics(qpos)
        error = qpos - self.previous_solution
        value = self.config.smooth_weight * np.dot(error, error)
        gradient = 2 * self.config.smooth_weight * error
        for index, frame in enumerate(self.fk.fingertip_frame_ids):
            jacobian = pin.getFrameJacobian(
                self.fk.model, self.fk.data, frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3]
            error = self.fk.data.oMf[frame].translation - self.target_fingertips[index]
            value += np.dot(error, error)
            gradient += 2 * (error @ jacobian)
        if grad.size:
            grad[:] = gradient
        return float(value)

    def pinch_objective(self, qpos, grad):
        """两项关节正则 + Σ pinch_weight·factor²·‖p_finger(q) − p_thumb(q)‖²。"""
        self.update_kinematics(qpos)
        value = 0.0
        gradient = np.zeros(self.dof)
        for weight, reference in (
            (self.config.reg_stage1_weight, self.position_solution),
            (self.config.reg_last_weight, self.previous_solution),
        ):
            error = qpos - reference
            value += weight * np.dot(error, error)
            gradient += 2 * weight * error

        thumb, *fingers = self.fk.fingertip_frame_ids
        thumb_position = self.fk.data.oMf[thumb].translation
        thumb_jacobian = pin.getFrameJacobian(
            self.fk.model, self.fk.data, thumb, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[:3]
        for frame, factor in zip(fingers, self.pinch_factors):
            if factor < 1e-4:
                continue
            error = self.fk.data.oMf[frame].translation - thumb_position
            jacobian = pin.getFrameJacobian(
                self.fk.model, self.fk.data, frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3]
            weight = self.config.pinch_base_weight * factor**2
            value += weight * np.dot(error, error)
            gradient += 2 * weight * (error @ (jacobian - thumb_jacobian))
        if grad.size:
            grad[:] = gradient
        return float(value)
