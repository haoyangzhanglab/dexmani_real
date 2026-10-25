"""TAG position fitting and pinch refinement with fixed-base hand kinematics."""

from __future__ import annotations

import logging

import nlopt
import numpy as np

from dexmani_real.robot.model import HAND_JOINT_SHAPE, XHAND_SDK_JOINT_NAMES
from dexmani_real.teleop.config import TAGRetargetingParams
from dexmani_real.teleop.retargeting.fingertip_objective import (
    FingertipObjective,
    joint_regularization,
)
from dexmani_real.teleop.retargeting.landmarks import prepare_landmarks
from dexmani_real.utils.log import ThrottledWarner

logger = logging.getLogger(__name__)


class TAGOptimizer:
    """Two-stage NLopt fitting; joints use model-order radians and fingertips use metres."""

    def __init__(
        self,
        *,
        fingertip_objective: FingertipObjective,
        joint_limits_lower: np.ndarray,
        joint_limits_upper: np.ndarray,
        config: TAGRetargetingParams,
    ) -> None:
        self.fingertip_objective = fingertip_objective
        self.dof = fingertip_objective.dof
        self.num_fingertips = len(fingertip_objective.fingertip_frame_ids)

        lower = np.asarray(joint_limits_lower, dtype=np.float64)
        upper = np.asarray(joint_limits_upper, dtype=np.float64)
        if lower.shape != (self.dof,) or upper.shape != (self.dof,):
            raise ValueError(f"joint limits must both have shape ({self.dof},)")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("joint limits must be finite")
        if np.any(lower > upper):
            raise ValueError("joint lower bounds must not exceed upper bounds")
        self.joint_limits_lower = lower.copy()
        self.joint_limits_upper = upper.copy()

        robot_lengths = np.asarray(config.robot_finger_lengths, dtype=np.float64)
        human_lengths = np.asarray(config.human_finger_lengths, dtype=np.float64)
        ratio = robot_lengths / human_lengths
        self.finger_scale: np.ndarray = ratio * config.finger_scale_boost

        self.position_optimizer = nlopt.opt(nlopt.LD_LBFGS, self.dof)
        self.position_optimizer.set_lower_bounds(self.joint_limits_lower.tolist())
        self.position_optimizer.set_upper_bounds(self.joint_limits_upper.tolist())
        self.position_optimizer.set_ftol_abs(config.ftol_abs_s1)
        self.position_optimizer.set_maxeval(config.maxeval_s1)
        self.position_optimizer.set_min_objective(self.position_objective)
        self.smooth_weight = config.smooth_weight

        self.pinch_optimizer = nlopt.opt(nlopt.LD_SLSQP, self.dof)
        self.pinch_optimizer.set_lower_bounds(self.joint_limits_lower.tolist())
        self.pinch_optimizer.set_upper_bounds(self.joint_limits_upper.tolist())
        self.pinch_optimizer.set_ftol_abs(config.ftol_abs_s2)
        self.pinch_optimizer.set_maxeval(config.maxeval_s2)
        self.pinch_optimizer.set_min_objective(self.pinch_objective)

        self.pinch_base_weight = config.pinch_base_weight
        self.pinch_start_dist = config.pinch_start_dist_m
        self.pinch_full_dist = config.pinch_full_dist_m
        self.pinch_ema_alpha = config.pinch_ema_alpha
        self.pinch_skip_threshold = config.pinch_skip_threshold
        self.position_reference_weight = config.reg_stage1_weight
        self.previous_solution_weight = config.reg_last_weight

        self.default_qpos = (self.joint_limits_lower + self.joint_limits_upper) / 2.0
        self.previous_solution = self.default_qpos.copy()
        self.position_solution: np.ndarray | None = None
        self.pinch_factors = np.zeros(self.num_fingertips - 1, dtype=np.float64)
        self.target_fingertips: np.ndarray | None = None
        self.position_warn = ThrottledWarner(logger=logger)
        self.pinch_warn = ThrottledWarner(logger=logger)
        self.bounds_warn = ThrottledWarner(logger=logger)

    def solve(self, fingertip_positions: np.ndarray) -> np.ndarray | None:
        """Fit wrist-relative URDF-frame tips; Stage-1 roundoff produces no joint target."""
        fingertip_positions = np.asarray(fingertip_positions, dtype=np.float64)
        if fingertip_positions.shape != (self.num_fingertips, 3):
            raise ValueError(f"fingertip_positions must have shape ({self.num_fingertips}, 3)")
        self.target_fingertips = fingertip_positions * self.finger_scale[:, np.newaxis]
        # Check the scaled target once before NLopt repeatedly evaluates its gradients.
        if not np.all(np.isfinite(self.target_fingertips)):
            raise ValueError("scaled fingertip positions must be finite")

        try:
            position_solution = self.position_optimizer.optimize(self.previous_solution.copy())
        except nlopt.RoundoffLimited:
            self.position_warn(
                "TAGOptimizer: Stage 1 reached the NLopt roundoff limit — no target produced",
                exc_info=True,
            )
            return None
        position_solution = self.bounded_qpos(position_solution, "Stage 1 result")

        thumb_finger_vectors = fingertip_positions[1:] - fingertip_positions[0]
        pinch_distances = np.linalg.norm(thumb_finger_vectors, axis=1)
        target_factors = np.clip(
            (self.pinch_start_dist - pinch_distances)
            / (self.pinch_start_dist - self.pinch_full_dist),
            0.0,
            1.0,
        )
        self.pinch_factors = (
            (1.0 - self.pinch_ema_alpha) * self.pinch_factors
            + self.pinch_ema_alpha * target_factors
        )

        if float(np.max(self.pinch_factors)) < self.pinch_skip_threshold:
            self.previous_solution = position_solution.copy()
            return position_solution

        self.position_solution = position_solution
        try:
            solution = self.pinch_optimizer.optimize(position_solution)
        except nlopt.RoundoffLimited:
            self.pinch_warn(
                "TAGOptimizer: Stage 2 reached the NLopt roundoff limit — falling back to Stage 1",
                exc_info=True,
            )
            solution = position_solution.copy()
        else:
            solution = self.bounded_qpos(solution, "Stage 2 result")

        self.previous_solution = solution.copy()
        return solution

    def reset(self, qpos: np.ndarray | None = None) -> None:
        """Reset temporal state, optionally warm-starting from (dof,) model-order angles."""
        if qpos is not None:
            self.previous_solution = self.bounded_qpos(qpos, "reset warm start", warn_on_clip=False)
        else:
            self.previous_solution = self.default_qpos.copy()
        self.pinch_factors.fill(0.0)
        self.position_solution = None

    def bounded_qpos(
        self, qpos: np.ndarray, label: str, *, warn_on_clip: bool = True
    ) -> np.ndarray:
        """Validate and project one NLopt state into the configured box bounds."""
        qpos_array = np.asarray(qpos, dtype=np.float64)
        if qpos_array.shape != (self.dof,) or not np.all(np.isfinite(qpos_array)):
            raise ValueError(f"{label} must be a finite ({self.dof},) array")
        bounded = np.clip(qpos_array, self.joint_limits_lower, self.joint_limits_upper)
        if warn_on_clip and np.any(np.abs(bounded - qpos_array) > 1e-9):
            self.bounds_warn("TAGOptimizer: projected %s into NLopt bounds", label)
        return bounded

    def position_objective(self, qpos: np.ndarray, grad: np.ndarray) -> float:
        position_value, position_gradient = self.fingertip_objective.position_value_gradient(
            qpos, self.target_fingertips
        )
        regularization_value, regularization_gradient = joint_regularization(
            qpos, self.previous_solution, self.smooth_weight
        )
        if grad.size > 0:
            grad[:] = position_gradient + regularization_gradient
        return float(position_value + regularization_value)

    def pinch_objective(self, qpos: np.ndarray, grad: np.ndarray) -> float:
        value = 0.0
        gradient = np.zeros(self.dof, dtype=np.float64)
        for weight, reference in (
            (self.position_reference_weight, self.position_solution),
            (self.previous_solution_weight, self.previous_solution),
        ):
            regularization_value, regularization_gradient = joint_regularization(
                qpos, reference, weight
            )
            value += regularization_value
            gradient += regularization_gradient
        pinch_value, pinch_gradient = self.fingertip_objective.pinch_value_gradient(
            qpos, self.pinch_factors, self.pinch_base_weight
        )
        if grad.size > 0:
            grad[:] = gradient + pinch_gradient
        return float(value + pinch_value)


FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=np.intp)


class TAGHandRetargeter:
    """Fit wrist-relative right-hand MANO tips and return SDK-order joint angles."""

    def __init__(
        self,
        fingertip_link_names: tuple[str, ...],
        tag_config: TAGRetargetingParams,
        urdf_path: str,
    ) -> None:
        from scipy.spatial.transform import Rotation

        tag_config.validate()
        resolved_urdf_path = str(urdf_path)
        fingertip_objective = FingertipObjective(resolved_urdf_path, fingertip_link_names)
        model = fingertip_objective.model

        model_names = list(model.names[1:])
        self.model_to_sdk_indices = np.array(
            [model_names.index(name) for name in XHAND_SDK_JOINT_NAMES], dtype=np.intp
        )
        self.sdk_to_model_indices = np.argsort(self.model_to_sdk_indices)

        self.optimizer = TAGOptimizer(
            fingertip_objective=fingertip_objective,
            joint_limits_lower=model.lowerPositionLimit,
            joint_limits_upper=model.upperPositionLimit,
            config=tag_config,
        )

        self.R_urdf_from_mano = Rotation.from_euler(
            "xyz", tag_config.mano_to_urdf_euler
        ).as_matrix()
        self.pinky_scale = float(tag_config.pinky_scale)
        self.pinky_palm_scale = float(tag_config.pinky_palm_scale)

        logger.info(
            "TAGHandRetargeter ready (urdf=%s, mano→urdf=%s)",
            resolved_urdf_path,
            tag_config.mano_to_urdf_euler,
        )

    def retarget(self, landmarks: np.ndarray | None) -> np.ndarray | None:
        """Invalid geometry/Stage-1 roundoff returns None; Stage-2 roundoff retains Stage 1."""
        mano = prepare_landmarks(
            landmarks,
            pinky_scale=self.pinky_scale,
            pinky_palm_scale=self.pinky_palm_scale,
        )
        if mano is None:
            return None

        wrist_relative_tips = mano[FINGERTIP_INDICES] - mano[0]
        tips_urdf = wrist_relative_tips @ self.R_urdf_from_mano.T

        qpos_model = self.optimizer.solve(tips_urdf)

        if qpos_model is None:
            return None

        return qpos_model[self.model_to_sdk_indices]

    def reset(self, initial_qpos: np.ndarray | None = None) -> None:
        """Reset optimizer state and optional SDK-order warm start."""
        if (
            initial_qpos is not None
            and initial_qpos.shape == HAND_JOINT_SHAPE
            and np.all(np.isfinite(initial_qpos))
        ):
            qpos_model = initial_qpos[self.sdk_to_model_indices]
            self.optimizer.reset(qpos_model)
        else:
            self.optimizer.reset(None)
