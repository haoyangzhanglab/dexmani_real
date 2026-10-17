"""TAG optimizer ported from TAG/Retargeting/Hand_Retargeting/New_method/opt_xhand.py.
Fixed-identity FreeFlyer: L-BFGS fits fingertips with temporal smoothing; SLSQP refines near-thumb pinches.
"""

from __future__ import annotations

import logging

import nlopt
import numpy as np
import pinocchio as pin

from dexmani_real.config.control import TAGRetargetingParams
from dexmani_real.robot.model import HAND_JOINT_SHAPE, XHAND_SDK_JOINT_NAMES
from dexmani_real.teleop.retargeting.pin_grad import PinGrad
from dexmani_real.teleop.retargeting.retargeter import prepare_landmarks
from dexmani_real.utils.log import ThrottledWarner

logger = logging.getLogger(__name__)


class HandOptimizer:
    """Fit fingertips with analytic gradients; joint limits use radians and finger lengths use meters.
    Scale lengths by robot/human × finger_scale_boost; stage-specific tolerances/budgets configure NLopt.
    Stage 2 applies the distance-ramped pinch attraction, EMA/skip threshold, and Stage-1/previous-frame anchors.
    The injected PinGrad and its mutable Data belong to this optimizer.
    """

    def __init__(
        self,
        *,
        pin_grad: PinGrad,
        joint_limits_lower: np.ndarray,
        joint_limits_upper: np.ndarray,
        config: TAGRetargetingParams,
    ) -> None:
        self.pin_grad = pin_grad
        self.dof: int = self.pin_grad.dof
        self.finger_num: int = len(self.pin_grad.tip_frame_ids)

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

        self.qpos_floating = np.zeros(3 + 4 + self.dof, dtype=np.float64)
        self.qpos_floating[6] = 1.0  # FreeFlyer q uses xyzw; index 6 is identity w.

        self.opt_s1 = nlopt.opt(nlopt.LD_LBFGS, self.dof)
        self.opt_s1.set_lower_bounds(self.joint_limits_lower.tolist())
        self.opt_s1.set_upper_bounds(self.joint_limits_upper.tolist())
        self.opt_s1.set_ftol_abs(config.ftol_abs_s1)
        self.opt_s1.set_maxeval(config.maxeval_s1)
        self.opt_s1.set_min_objective(self._obj_s1)
        self._smooth_weight = config.smooth_weight

        self.opt_s2 = nlopt.opt(nlopt.LD_SLSQP, self.dof)
        self.opt_s2.set_lower_bounds(self.joint_limits_lower.tolist())
        self.opt_s2.set_upper_bounds(self.joint_limits_upper.tolist())
        self.opt_s2.set_ftol_abs(config.ftol_abs_s2)
        self.opt_s2.set_maxeval(config.maxeval_s2)
        self.opt_s2.set_min_objective(self._obj_s2)

        self.pinch_base_weight = config.pinch_base_weight
        self.pinch_start_dist = config.pinch_start_dist_m
        self.pinch_full_dist = config.pinch_full_dist_m
        self.pinch_ema_alpha = config.pinch_ema_alpha
        self.pinch_skip_threshold = config.pinch_skip_threshold
        self.reg_s1_weight = config.reg_stage1_weight
        self.reg_last_weight = config.reg_last_weight

        self._default_qpos = (self.joint_limits_lower + self.joint_limits_upper) / 2.0
        self.last_qpos: np.ndarray = self._default_qpos.copy()
        self.qpos_stage1: np.ndarray | None = None
        self.pinch_factors: np.ndarray = np.zeros(self.finger_num, dtype=np.float64)
        self._current_target: np.ndarray | None = None
        self._stage1_warn = ThrottledWarner(logger=logger)
        self._stage2_warn = ThrottledWarner(logger=logger)
        self._bounds_warn = ThrottledWarner(logger=logger)

    def solve(self, fingertip_positions: np.ndarray) -> np.ndarray | None:
        """Solve wrist-relative URDF-frame tips (finger_num,3); return model-order qpos or None on Stage-1 roundoff."""
        fingertip_positions = np.asarray(fingertip_positions, dtype=np.float64)
        if fingertip_positions.shape != (self.finger_num, 3):
            raise ValueError(f"fingertip_positions must have shape ({self.finger_num}, 3)")
        self._current_target = fingertip_positions * self.finger_scale[:, np.newaxis]
        # Check the scaled target once before NLopt repeatedly evaluates its gradients.
        if not np.all(np.isfinite(self._current_target)):
            raise ValueError("scaled fingertip positions must be finite")

        warm_start = self._bounded_qpos(self.last_qpos, "Stage 1 warm start")
        self.last_qpos = warm_start.copy()
        try:
            q_s1 = self.opt_s1.optimize(warm_start)
        except nlopt.RoundoffLimited:
            self._stage1_warn(
                "HandOptimizer: Stage 1 reached the NLopt roundoff limit — no target produced",
                exc_info=True,
            )
            return None
        q_s1 = self._bounded_qpos(q_s1, "Stage 1 result")

        vecs = fingertip_positions[1:] - fingertip_positions[0]
        dists = np.linalg.norm(vecs, axis=1)
        target_factors = np.clip(
            (self.pinch_start_dist - dists) / (self.pinch_start_dist - self.pinch_full_dist),
            0.0,
            1.0,
        )
        self.pinch_factors = (
            1.0 - self.pinch_ema_alpha
        ) * self.pinch_factors + self.pinch_ema_alpha * np.concatenate(
            [np.zeros(1, dtype=np.float64), target_factors]
        )

        if float(np.max(self.pinch_factors)) < self.pinch_skip_threshold:
            self.last_qpos = q_s1.copy()
            return q_s1

        self.qpos_stage1 = q_s1
        try:
            q_s2 = self.opt_s2.optimize(q_s1)
        except nlopt.RoundoffLimited:
            self._stage2_warn(
                "HandOptimizer: Stage 2 reached the NLopt roundoff limit — falling back to Stage 1",
                exc_info=True,
            )
            q_s2 = q_s1.copy()
        q_s2 = self._bounded_qpos(q_s2, "Stage 2 result")

        self.last_qpos = q_s2.copy()
        return q_s2

    def reset(self, qpos: np.ndarray | None = None) -> None:
        """Reset temporal state, optionally warm-starting from (dof,) model-order angles."""
        if qpos is not None:
            self.last_qpos = self._bounded_qpos(qpos, "reset warm start", warn_on_clip=False)
        else:
            self.last_qpos = self._default_qpos.copy()
        self.pinch_factors = np.zeros(self.finger_num, dtype=np.float64)
        self.qpos_stage1 = None

    def _bounded_qpos(
        self, qpos: np.ndarray, label: str, *, warn_on_clip: bool = True
    ) -> np.ndarray:
        """Validate and project one NLopt state into the configured box bounds."""
        qpos_array = np.asarray(qpos, dtype=np.float64)
        if qpos_array.shape != (self.dof,) or not np.all(np.isfinite(qpos_array)):
            raise ValueError(f"{label} must be a finite ({self.dof},) array")
        bounded = np.clip(qpos_array, self.joint_limits_lower, self.joint_limits_upper)
        if warn_on_clip and np.any(np.abs(bounded - qpos_array) > 1e-9):
            self._bounds_warn("HandOptimizer: projected %s into NLopt bounds", label)
        return bounded

    def _obj_s1(self, qpos: np.ndarray, grad: np.ndarray) -> float:
        """Stage 1 objective: position error + temporal smoothness."""
        self.qpos_floating[7:] = qpos
        # solve() sets _current_target before NLopt callbacks.
        g_pos, loss_pos = self.pin_grad.compute_position_gradient(
            self.qpos_floating,
            self._current_target,  # type: ignore[arg-type]
        )
        g_smooth, loss_smooth = PinGrad.compute_smoothness_gradient(
            qpos, self.last_qpos, self._smooth_weight
        )
        if grad.size > 0:
            grad[:] = g_pos + g_smooth
        return float(loss_pos + loss_smooth)

    def _obj_s2(self, qpos: np.ndarray, grad: np.ndarray) -> float:
        """Stage 2 objective: pinch refinement with regularization."""
        self.qpos_floating[7:] = qpos
        self.pin_grad.update_kinematics(self.qpos_floating)

        total_loss = 0.0
        total_grad = np.zeros(self.dof, dtype=np.float64)

        for weight, ref in (
            (self.reg_s1_weight, self.qpos_stage1),
            (self.reg_last_weight, self.last_qpos),
        ):
            g_smooth, loss_smooth = PinGrad.compute_smoothness_gradient(qpos, ref, weight)
            total_loss += loss_smooth
            total_grad += g_smooth

        thumb_fid = self.pin_grad.tip_frame_ids[0]
        J_thumb_v = pin.getFrameJacobian(
            self.pin_grad.model,
            self.pin_grad.data,
            thumb_fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[:3, 6:]
        p_thumb = self.pin_grad.data.oMf[thumb_fid].translation

        for i in range(1, self.finger_num):
            factor = self.pinch_factors[i]
            if factor < 1e-4:
                continue

            fid = self.pin_grad.tip_frame_ids[i]
            J_finger_v = pin.getFrameJacobian(
                self.pin_grad.model, self.pin_grad.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )[:3, 6:]
            p_finger = self.pin_grad.data.oMf[fid].translation
            diff = p_finger - p_thumb
            weight = self.pinch_base_weight * (factor * factor)

            total_loss += float(weight * np.sum(diff * diff))
            total_grad += weight * 2.0 * (diff @ (J_finger_v - J_thumb_v))

        if grad.size > 0:
            grad[:] = total_grad
        return float(total_loss)


_FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=np.intp)


class TAGHandRetargeter:
    """Map right-hand operator landmarks through wrist-relative MANO tips to TAG, then SDK-order qpos."""

    def __init__(
        self,
        fingertip_link_names: tuple[str, ...],
        tag_config: TAGRetargetingParams,
        urdf_path: str,
    ) -> None:
        from scipy.spatial.transform import Rotation

        tag_config.validate()
        resolved_urdf_path = str(urdf_path)
        pin_grad = PinGrad(resolved_urdf_path, list(fingertip_link_names))
        model = pin_grad.model
        joint_lo = model.lowerPositionLimit[7:].copy()
        joint_hi = model.upperPositionLimit[7:].copy()

        model_names = list(model.names[2:])  # skip universe and FreeFlyer root
        self._mapping_model_to_sdk = np.array(
            [model_names.index(name) for name in XHAND_SDK_JOINT_NAMES], dtype=np.intp
        )
        self._mapping_sdk_to_model = np.argsort(self._mapping_model_to_sdk)

        self._optimizer = HandOptimizer(
            pin_grad=pin_grad,
            joint_limits_lower=joint_lo,
            joint_limits_upper=joint_hi,
            config=tag_config,
        )

        self._R_mano_to_urdf: np.ndarray = Rotation.from_euler(
            "xyz", tag_config.mano_to_urdf_euler
        ).as_matrix()
        self._pinky_scale = float(tag_config.pinky_scale)
        self._pinky_palm_scale = float(tag_config.pinky_palm_scale)

        logger.info(
            "TAGHandRetargeter ready (urdf=%s, mano→urdf=%s)",
            resolved_urdf_path,
            tag_config.mano_to_urdf_euler,
        )

    def retarget(self, landmarks: np.ndarray | None) -> np.ndarray | None:
        """Map operator-frame landmarks to SDK qpos; invalid geometry/Stage-1 roundoff returns None.
        Stage-2 roundoff retains the Stage-1 result; technical errors propagate.
        """
        mano = prepare_landmarks(
            landmarks, pinky_scale=self._pinky_scale, pinky_palm_scale=self._pinky_palm_scale,
        )
        if mano is None:
            return None

        tips = mano[_FINGERTIP_INDICES].copy()
        tips -= mano[0]  # MANO wrist
        tips_urdf = tips @ self._R_mano_to_urdf.T

        qpos_model = self._optimizer.solve(tips_urdf)

        if qpos_model is None:
            return None

        return qpos_model[self._mapping_model_to_sdk]

    def reset(self, initial_qpos: np.ndarray | None = None) -> None:
        """Reset optimizer state and optional SDK-order warm start."""
        if (
            initial_qpos is not None
            and initial_qpos.shape == HAND_JOINT_SHAPE
            and np.all(np.isfinite(initial_qpos))
        ):
            qpos_model = initial_qpos[self._mapping_sdk_to_model]
            self._optimizer.reset(qpos_model)
        else:
            self._optimizer.reset(None)
