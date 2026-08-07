"""TAG two-stage NLopt hand retargeting optimizer.

Ported from TAG/Retargeting/Hand_Retargeting/New_method/opt_xhand.py.
Adapted for DexMani: all parameters constructor-injected, no glove FK dependency,
FreeFlyer base fixed at identity (frame alignment via external rotation).

Stage 1 (L-BFGS): global fingertip position matching + temporal smoothness.
Stage 2 (SLSQP): pinch refinement — conditional on finger-to-thumb proximity.
"""

from __future__ import annotations

import nlopt
import numpy as np
import pinocchio as pin

from dexmani_real.teleop.tag_retargeting.pin_grad import PinGrad
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class HandOptimizer:
    """Two-stage NLopt hand retargeting optimizer.

    Minimizes ||FK(q) - target||² via analytic Pinocchio gradients.

    Parameters
    ----------
    urdf_path:
        Absolute path to XHand URDF file.
    fingertip_frame_names:
        URDF frame names for the 5 fingertip links.
    joint_limits_lower / joint_limits_upper:
        Joint bounds (rad), shape (dof,).
    finger_lengths_robot / finger_lengths_human:
        Finger lengths (m), shape (finger_num,).  Scale = robot/human * boost.
    finger_scale_boost:
        Multiplier on the robot/human length ratio (1.2 = slight over-extension).
    smooth_weight:
        Temporal smoothness weight in Stage 1 objective.
    ftol_abs_s1 / maxeval_s1:
        Stage 1 NLopt convergence tolerance / max evaluations.
    ftol_abs_s2 / maxeval_s2:
        Stage 2 NLopt convergence tolerance / max evaluations.
    pinch_base_weight:
        Base weight for thumb-to-finger attraction in Stage 2.
    pinch_start_dist_m / pinch_full_dist_m:
        Distance thresholds (m) for pinch activation ramp.
    pinch_ema_alpha:
        EMA smoothing factor for pinch activation (0.0 = frozen, 1.0 = instant).
    pinch_skip_threshold:
        Skip Stage 2 when max(pinch_factors) < this value.
    reg_stage1_weight / reg_last_weight:
        Stage 2 regularization: anchor to Stage 1 solution / previous frame.
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
        pinch_ema_alpha: float = 0.4,
        pinch_skip_threshold: float = 0.01,
        reg_stage1_weight: float = 1.0,
        reg_last_weight: float = 0.8,
    ) -> None:
        # ── Kinematics ──
        self.pin_grad = PinGrad(urdf_path, fingertip_frame_names)
        self.dof: int = self.pin_grad.dof
        self.finger_num: int = len(self.pin_grad.tip_frame_ids)

        # ── Finger length scaling ──
        ratio = np.asarray(finger_lengths_robot, dtype=np.float64) / np.asarray(
            finger_lengths_human, dtype=np.float64
        )
        self.finger_scale: np.ndarray = ratio * finger_scale_boost  # (finger_num,)

        # ── FreeFlyer buffer: base translation always zero, rotation always identity ──
        # Frame alignment is handled externally via R_mano_to_urdf on target positions.
        self.qpos_floating = np.zeros(3 + 4 + self.dof, dtype=np.float64)
        self.qpos_floating[6] = 1.0  # w=1 → identity quaternion (x, y, z, w)

        # ── NLopt Stage 1: L-BFGS ──
        self.opt_s1 = nlopt.opt(nlopt.LD_LBFGS, self.dof)
        self.opt_s1.set_lower_bounds(joint_limits_lower.tolist())
        self.opt_s1.set_upper_bounds(joint_limits_upper.tolist())
        self.opt_s1.set_ftol_abs(ftol_abs_s1)
        self.opt_s1.set_maxeval(maxeval_s1)
        self.opt_s1.set_min_objective(self._obj_s1)
        self._smooth_weight = smooth_weight

        # ── NLopt Stage 2: SLSQP (pinch refinement) ──
        self.opt_s2 = nlopt.opt(nlopt.LD_SLSQP, self.dof)
        self.opt_s2.set_lower_bounds(joint_limits_lower.tolist())
        self.opt_s2.set_upper_bounds(joint_limits_upper.tolist())
        self.opt_s2.set_ftol_abs(ftol_abs_s2)
        self.opt_s2.set_maxeval(maxeval_s2)
        self.opt_s2.set_min_objective(self._obj_s2)

        # ── Pinch parameters ──
        self.pinch_base_weight = pinch_base_weight
        self.pinch_start_dist = pinch_start_dist_m
        self.pinch_full_dist = pinch_full_dist_m
        self.pinch_ema_alpha = pinch_ema_alpha
        self.pinch_skip_threshold = pinch_skip_threshold
        self.reg_s1_weight = reg_stage1_weight
        self.reg_last_weight = reg_last_weight

        # ── State ──
        init = (joint_limits_lower + joint_limits_upper) / 2.0
        self.last_qpos: np.ndarray = init.copy()
        self.qpos_stage1: np.ndarray | None = None
        self.pinch_factors: np.ndarray = np.zeros(self.finger_num, dtype=np.float64)
        self._current_target: np.ndarray | None = None  # (finger_num, 3) scaled targets

    # ── Public API ──────────────────────────────────────────────

    def solve(self, fingertip_positions: np.ndarray) -> np.ndarray:
        """Solve one frame: fingertip positions → joint angles.

        Args:
            fingertip_positions: (finger_num, 3) target positions in URDF frame,
                centered at wrist origin.

        Returns:
            (dof,) joint angles in Pinocchio model order.
        """
        # Scale targets (1.2× boost discourages over-flexing)
        self._current_target = fingertip_positions * self.finger_scale[:, np.newaxis]

        # ── Stage 1: Global position matching ──
        try:
            q_s1 = self.opt_s1.optimize(self.last_qpos)
        except Exception:
            logger.warning("HandOptimizer: Stage 1 NLopt failed — holding last_qpos", exc_info=True)
            q_s1 = self.last_qpos.copy()

        # ── Pinch detection (on UNscaled targets) ──
        vecs = fingertip_positions[1:] - fingertip_positions[0]  # (finger_num-1, 3)
        dists = np.linalg.norm(vecs, axis=1)  # (finger_num-1,)
        target_factors = np.clip(
            (self.pinch_start_dist - dists) / (self.pinch_start_dist - self.pinch_full_dist),
            0.0,
            1.0,
        )
        # EMA update: smooths pinch transition across frames
        self.pinch_factors = (1.0 - self.pinch_ema_alpha) * self.pinch_factors + self.pinch_ema_alpha * np.concatenate(
            [np.zeros(1, dtype=np.float64), target_factors]
        )

        if float(np.max(self.pinch_factors)) < self.pinch_skip_threshold:
            self.last_qpos = q_s1.copy()
            return q_s1

        # ── Stage 2: Pinch refinement ──
        self.qpos_stage1 = q_s1
        try:
            q_s2 = self.opt_s2.optimize(q_s1)
        except Exception:
            logger.warning("HandOptimizer: Stage 2 NLopt failed — falling back to Stage 1", exc_info=True)
            q_s2 = q_s1.copy()

        self.last_qpos = q_s2.copy()
        return q_s2

    def reset(self, qpos: np.ndarray | None = None) -> None:
        """Reset temporal state for a clean episode start.

        Args:
            qpos: Optional (dof,) joint angles to warm-start from
                  (in Pinocchio model order).
        """
        if qpos is not None and qpos.shape == (self.dof,) and np.all(np.isfinite(qpos)):
            self.last_qpos = qpos.copy()
        self.pinch_factors = np.zeros(self.finger_num, dtype=np.float64)
        self.qpos_stage1 = None

    # ── NLopt callbacks ─────────────────────────────────────────

    def _obj_s1(self, qpos: np.ndarray, grad: np.ndarray) -> float:
        """Stage 1 objective: position error + temporal smoothness."""
        self.qpos_floating[7:] = qpos
        # _current_target is set by solve() before any NLopt callback fires;
        # mypy sees np.ndarray|None but runtime is always np.ndarray here.
        g_pos, loss_pos = self.pin_grad.compute_position_gradient(self.qpos_floating, self._current_target)  # type: ignore[arg-type]
        g_smooth, loss_smooth = PinGrad.compute_smoothness_gradient(qpos, self.last_qpos, self._smooth_weight)
        if grad.size > 0:
            grad[:] = g_pos + g_smooth
        return float(loss_pos + loss_smooth)

    def _obj_s2(self, qpos: np.ndarray, grad: np.ndarray) -> float:
        """Stage 2 objective: pinch refinement with regularization."""
        self.qpos_floating[7:] = qpos
        self.pin_grad.update_kinematics(self.qpos_floating)

        total_loss = 0.0
        total_grad = np.zeros(self.dof, dtype=np.float64)

        # Regularization: anchor to Stage 1 + temporal consistency
        for weight, ref, label in [
            (self.reg_s1_weight, self.qpos_stage1, "stage1"),
            (self.reg_last_weight, self.last_qpos, "last"),
        ]:
            if ref is None:
                logger.warning("HandOptimizer: _obj_s2 called without %s reference — skipping anchor", label)
                continue
            g, l = PinGrad.compute_smoothness_gradient(qpos, ref, weight)
            total_loss += l
            total_grad += g

        # Per-finger pinch penalty: attract each fingertip to thumb
        thumb_fid = self.pin_grad.tip_frame_ids[0]
        J_thumb_v = pin.getFrameJacobian(
            self.pin_grad.model, self.pin_grad.data, thumb_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
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
