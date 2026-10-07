"""TAG two-stage NLopt hand retargeting optimizer.

Ported from TAG/Retargeting/Hand_Retargeting/New_method/opt_xhand.py.
Adapted for DexMani: all parameters constructor-injected, no glove FK dependency,
FreeFlyer base fixed at identity (frame alignment via external rotation).

Stage 1 (L-BFGS): global fingertip position matching + temporal smoothness.
Stage 2 (SLSQP): pinch refinement — conditional on finger-to-thumb proximity.
"""

from __future__ import annotations

import time
from typing import Any

import nlopt
import numpy as np
import pinocchio as pin

from dexmani_real.robot.model import HAND_JOINT_SHAPE, XHAND_SDK_JOINT_NAMES
from dexmani_real.teleop.retargeting.pin_grad import PinGrad
from dexmani_real.teleop.retargeting.retargeter import (
    _OPERATOR2MANO_RIGHT,
    _estimate_palm_frame,
    adaptive_retargeting_xhand,
    validate_landmarks,
)
from dexmani_real.utils.log import ThrottledWarner, get_logger

logger = get_logger(__name__)


class HandOptimizer:
    """Minimize fingertip squared error with analytic Pinocchio gradients.

    Joint limits are (dof,) radians; finger lengths are (finger_num,) meters,
    scaled by robot/human * finger_scale_boost. ftol_abs_s* and maxeval_s*
    set each stage's NLopt tolerance and evaluation budget.

    Stage 1 adds temporal smoothness. Stage 2 adds thumb-to-finger attraction
    with pinch_base_weight and a start/full distance ramp in meters.
    pinch_ema_alpha spans frozen (0) to instant (1); Stage 2 is skipped when
    max(pinch_factors) < pinch_skip_threshold. reg_stage1_weight and
    reg_last_weight anchor Stage 2 to Stage 1 and the previous frame.

    """

    def __init__(
        self,
        *,
        urdf_path: str,
        fingertip_frame_names: list[str],
        joint_limits_lower: np.ndarray,
        joint_limits_upper: np.ndarray,
        finger_lengths_robot: np.ndarray,
        finger_lengths_human: np.ndarray,
        finger_scale_boost: float,
        smooth_weight: float = 0.02,
        ftol_abs_s1: float = 1e-4,
        maxeval_s1: int = 80,
        ftol_abs_s2: float = 1e-6,
        maxeval_s2: int = 100,
        pinch_base_weight: float = 2000.0,
        pinch_start_dist_m: float = 0.030,
        pinch_full_dist_m: float = 0.008,
        pinch_ema_alpha: float = 0.75,
        pinch_skip_threshold: float = 0.01,
        reg_stage1_weight: float = 1.0,
        reg_last_weight: float = 0.8,
    ) -> None:
        self.pin_grad = PinGrad(urdf_path, fingertip_frame_names)
        self.dof: int = self.pin_grad.dof
        self.finger_num: int = len(self.pin_grad.tip_frame_ids)
        if self.finger_num != 5:
            raise ValueError(
                f"TAG optimizer requires exactly five fingertip frames, got {self.finger_num}"
            )

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

        robot_lengths = np.asarray(finger_lengths_robot, dtype=np.float64)
        human_lengths = np.asarray(finger_lengths_human, dtype=np.float64)
        if (
            robot_lengths.shape != (self.finger_num,)
            or human_lengths.shape != (self.finger_num,)
            or not np.all(np.isfinite(robot_lengths))
            or not np.all(np.isfinite(human_lengths))
            or np.any(robot_lengths <= 0)
            or np.any(human_lengths <= 0)
            or not np.isfinite(finger_scale_boost)
            or finger_scale_boost <= 0
        ):
            raise ValueError(
                "finger lengths and scale boost must be finite, positive five-finger values"
            )
        ratio = robot_lengths / human_lengths
        self.finger_scale: np.ndarray = ratio * finger_scale_boost  # (finger_num,)

        self.qpos_floating = np.zeros(3 + 4 + self.dof, dtype=np.float64)
        self.qpos_floating[6] = 1.0  # w=1 → identity quaternion (x, y, z, w)

        self.opt_s1 = nlopt.opt(nlopt.LD_LBFGS, self.dof)
        self.opt_s1.set_lower_bounds(self.joint_limits_lower.tolist())
        self.opt_s1.set_upper_bounds(self.joint_limits_upper.tolist())
        self.opt_s1.set_ftol_abs(ftol_abs_s1)
        self.opt_s1.set_maxeval(maxeval_s1)
        self.opt_s1.set_min_objective(self._obj_s1)
        self._smooth_weight = smooth_weight

        self.opt_s2 = nlopt.opt(nlopt.LD_SLSQP, self.dof)
        self.opt_s2.set_lower_bounds(self.joint_limits_lower.tolist())
        self.opt_s2.set_upper_bounds(self.joint_limits_upper.tolist())
        self.opt_s2.set_ftol_abs(ftol_abs_s2)
        self.opt_s2.set_maxeval(maxeval_s2)
        self.opt_s2.set_min_objective(self._obj_s2)

        self.pinch_base_weight = pinch_base_weight
        self.pinch_start_dist = pinch_start_dist_m
        self.pinch_full_dist = pinch_full_dist_m
        self.pinch_ema_alpha = pinch_ema_alpha
        self.pinch_skip_threshold = pinch_skip_threshold
        self.reg_s1_weight = reg_stage1_weight
        self.reg_last_weight = reg_last_weight

        self._default_qpos = (self.joint_limits_lower + self.joint_limits_upper) / 2.0
        self.last_qpos: np.ndarray = self._default_qpos.copy()
        self.qpos_stage1: np.ndarray | None = None
        self.pinch_factors: np.ndarray = np.zeros(self.finger_num, dtype=np.float64)
        self._current_target: np.ndarray | None = None  # (finger_num, 3) scaled targets
        self._stage1_warn = ThrottledWarner(interval_s=5.0, logger=logger)
        self._stage2_warn = ThrottledWarner(interval_s=5.0, logger=logger)
        self._bounds_warn = ThrottledWarner(interval_s=5.0, logger=logger)

    def solve(self, fingertip_positions: np.ndarray) -> np.ndarray | None:
        """Solve wrist-centered fingertips (finger_num, 3) in the URDF frame.

        Returns model-order angles, or None on Stage 1 failure.
        """
        fingertip_positions = np.asarray(fingertip_positions, dtype=np.float64)
        if fingertip_positions.shape != (self.finger_num, 3):
            raise ValueError(f"fingertip_positions must have shape ({self.finger_num}, 3)")
        if not np.all(np.isfinite(fingertip_positions)):
            raise ValueError("fingertip_positions must be finite")

        self._current_target = fingertip_positions * self.finger_scale[:, np.newaxis]

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

        vecs = fingertip_positions[1:] - fingertip_positions[0]  # (finger_num-1, 3)
        dists = np.linalg.norm(vecs, axis=1)  # (finger_num-1,)
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
        if qpos is not None and np.asarray(qpos).shape == (self.dof,) and np.all(np.isfinite(qpos)):
            qpos_array = np.asarray(qpos, dtype=np.float64)
            bounded = self._bounded_qpos(qpos_array, "reset warm start", warn_on_clip=False)
            self.last_qpos = bounded
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

        for weight, ref, label in [
            (self.reg_s1_weight, self.qpos_stage1, "stage1"),
            (self.reg_last_weight, self.last_qpos, "last"),
        ]:
            if ref is None:
                logger.warning(
                    "HandOptimizer: _obj_s2 called without %s reference — skipping anchor", label
                )
                continue
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
    """Retarget VR to XHand with TAG's two-stage NLopt optimizer.

    Pipeline: (21, 3) operator landmarks -> palm/MANO frame -> pinky scaling
    -> tips [4, 8, 12, 16, 20] minus wrist[0] -> R_mano_to_urdf -> solve
    -> remap model joints to (12,) SDK order.

    Uses right-hand geometry and shares retarget()/reset() with
    DexPilotHandRetargeter. debug enables per-frame timing logs.
    """

    def __init__(
        self,
        fingertip_link_names: tuple[str, ...],
        tag_config: Any,
        urdf_path: str,
        debug: bool = False,
    ) -> None:
        import pinocchio as pin
        from scipy.spatial.transform import Rotation

        from dexmani_real.teleop.retargeting.pin_grad import validate_fingertip_frame_names

        tag_config.validate()
        resolved_urdf_path = str(urdf_path)
        resolved_tip_names = validate_fingertip_frame_names(fingertip_link_names)
        model = pin.buildModelFromUrdf(resolved_urdf_path, pin.JointModelFreeFlyer())
        joint_lo = model.lowerPositionLimit[7:].copy()
        joint_hi = model.upperPositionLimit[7:].copy()

        # Pinocchio and SDK use different joint orders.
        model_names = list(model.names[2:])  # skip "universe" and "root_joint"
        self.sdk_joint_names = XHAND_SDK_JOINT_NAMES
        self._mapping_model_to_sdk = np.array(
            [model_names.index(name) for name in self.sdk_joint_names], dtype=np.intp
        )
        self._mapping_sdk_to_model = np.argsort(self._mapping_model_to_sdk)

        self._optimizer = HandOptimizer(
            urdf_path=resolved_urdf_path,
            fingertip_frame_names=list(resolved_tip_names),
            joint_limits_lower=joint_lo,
            joint_limits_upper=joint_hi,
            finger_lengths_robot=np.array(tag_config.robot_finger_lengths, dtype=np.float64),
            finger_lengths_human=np.array(tag_config.human_finger_lengths, dtype=np.float64),
            finger_scale_boost=tag_config.finger_scale_boost,
            smooth_weight=tag_config.smooth_weight,
            ftol_abs_s1=tag_config.ftol_abs_s1,
            maxeval_s1=tag_config.maxeval_s1,
            ftol_abs_s2=tag_config.ftol_abs_s2,
            maxeval_s2=tag_config.maxeval_s2,
            pinch_base_weight=tag_config.pinch_base_weight,
            pinch_start_dist_m=tag_config.pinch_start_dist_m,
            pinch_full_dist_m=tag_config.pinch_full_dist_m,
            pinch_ema_alpha=tag_config.pinch_ema_alpha,
            pinch_skip_threshold=tag_config.pinch_skip_threshold,
            reg_stage1_weight=tag_config.reg_stage1_weight,
            reg_last_weight=tag_config.reg_last_weight,
        )

        self._R_mano_to_urdf: np.ndarray = Rotation.from_euler(
            "xyz", tag_config.mano_to_urdf_euler
        ).as_matrix()
        self._pinky_scale = float(tag_config.pinky_scale)
        self._pinky_palm_scale = float(tag_config.pinky_palm_scale)

        self.debug = bool(debug)

        logger.info(
            "TAGHandRetargeter ready (urdf=%s, mano→urdf=%s)",
            resolved_urdf_path,
            tag_config.mano_to_urdf_euler,
        )

    def retarget(self, landmarks: np.ndarray | None) -> np.ndarray | None:
        """Retarget VR landmarks (operator-frame, 21×3) to XHand joint qpos (12,).

        Expected invalid input or Stage 1 roundoff returns ``None``. Unexpected
        optimizer errors propagate to the session owner.
        """
        if landmarks is None:
            return None
        valid, reason = validate_landmarks(landmarks)
        if not valid:
            logger.warning(
                "TAGHandRetargeter: landmarks rejected (%s) — no target produced", reason
            )
            return None

        t0 = time.perf_counter() if self.debug else 0.0

        try:
            wrist_rot = _estimate_palm_frame(landmarks)
            mano = landmarks @ wrist_rot @ _OPERATOR2MANO_RIGHT
        except (ValueError, np.linalg.LinAlgError):
            logger.warning("TAGHandRetargeter: coordinate transform failed — no target produced")
            return None

        mano = adaptive_retargeting_xhand(
            mano,
            scale=self._pinky_scale,
            palm_scale=self._pinky_palm_scale,
        )

        tips = mano[_FINGERTIP_INDICES].copy()  # (5, 3) in MANO frame
        tips -= mano[0]  # center at wrist
        tips_urdf = tips @ self._R_mano_to_urdf.T  # (5, 3) in URDF frame

        qpos_model = self._optimizer.solve(tips_urdf)  # (12,) in Pinocchio model order

        if qpos_model is None:
            return None

        qpos_sdk = qpos_model[self._mapping_model_to_sdk]

        if self.debug:
            dt_ms = 1000.0 * (time.perf_counter() - t0)
            logger.info("TAGHandRetargeter: retarget %.2f ms", dt_ms)

        return qpos_sdk

    def reset(self, initial_qpos: np.ndarray | None = None) -> None:
        """Reset optimizer state and optional SDK-order warm start."""
        if (
            initial_qpos is not None
            and initial_qpos.shape == HAND_JOINT_SHAPE
            and np.all(np.isfinite(initial_qpos))
        ):
            # SDK order → model order for optimizer warm-start
            qpos_model = initial_qpos[self._mapping_sdk_to_model]
            self._optimizer.reset(qpos_model)
        else:
            self._optimizer.reset(None)
