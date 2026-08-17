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
from dexmani_real.utils.log import ThrottledWarner, get_logger

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
    prior_weight:
        Human-flexion prior weight γ: pulls the solved joints toward the
        per-frame human flexion reference (``q_prior`` passed to ``solve``) in
        both stages, masked to the 10 flexion joints by ``prior_mask``. 0.0
        disables the prior.
    prior_mask:
        (dof,) 0/1 weight mask selecting which joints the prior applies to
        (1 = flexion joints, 0 = unmapped joints such as thumb_rota1 /
        index_bend).  Required when ``prior_weight > 0``.
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
        prior_weight: float = 0.0,
        prior_mask: np.ndarray | None = None,
    ) -> None:
        # ── Kinematics ──
        self.pin_grad = PinGrad(urdf_path, fingertip_frame_names)
        self.dof: int = self.pin_grad.dof
        self.finger_num: int = len(self.pin_grad.tip_frame_ids)
        if self.finger_num != 5:
            raise ValueError(f"TAG optimizer requires exactly five fingertip frames, got {self.finger_num}")

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

        # ── Finger length scaling ──
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
            raise ValueError("finger lengths and scale boost must be finite, positive five-finger values")
        ratio = robot_lengths / human_lengths
        self.finger_scale: np.ndarray = ratio * finger_scale_boost  # (finger_num,)

        # ── FreeFlyer buffer: base translation always zero, rotation always identity ──
        # Frame alignment is handled externally via R_mano_to_urdf on target positions.
        self.qpos_floating = np.zeros(3 + 4 + self.dof, dtype=np.float64)
        self.qpos_floating[6] = 1.0  # w=1 → identity quaternion (x, y, z, w)

        # ── NLopt Stage 1: L-BFGS ──
        self.opt_s1 = nlopt.opt(nlopt.LD_LBFGS, self.dof)
        self.opt_s1.set_lower_bounds(self.joint_limits_lower.tolist())
        self.opt_s1.set_upper_bounds(self.joint_limits_upper.tolist())
        self.opt_s1.set_ftol_abs(ftol_abs_s1)
        self.opt_s1.set_maxeval(maxeval_s1)
        self.opt_s1.set_min_objective(self._obj_s1)
        self._smooth_weight = smooth_weight

        # ── NLopt Stage 2: SLSQP (pinch refinement) ──
        self.opt_s2 = nlopt.opt(nlopt.LD_SLSQP, self.dof)
        self.opt_s2.set_lower_bounds(self.joint_limits_lower.tolist())
        self.opt_s2.set_upper_bounds(self.joint_limits_upper.tolist())
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

        # ── Human-flexion prior ──
        self.prior_weight = float(prior_weight)
        if not np.isfinite(self.prior_weight) or self.prior_weight < 0:
            raise ValueError("prior_weight must be finite and non-negative")
        if prior_mask is not None:
            prior_mask = np.asarray(prior_mask, dtype=np.float64)
            if prior_mask.shape != (self.dof,):
                raise ValueError(f"prior_mask must have shape ({self.dof},)")
            if not np.all(np.isfinite(prior_mask)):
                raise ValueError("prior_mask must be finite")
        self.prior_mask = prior_mask

        # ── State ──
        self._default_qpos = (self.joint_limits_lower + self.joint_limits_upper) / 2.0
        self.last_qpos: np.ndarray = self._default_qpos.copy()
        self.qpos_stage1: np.ndarray | None = None
        self.pinch_factors: np.ndarray = np.zeros(self.finger_num, dtype=np.float64)
        self._current_target: np.ndarray | None = None  # (finger_num, 3) scaled targets
        self._current_q_prior: np.ndarray | None = None  # (dof,) per-frame human flexion reference
        self._stage1_warn = ThrottledWarner(interval_s=5.0, logger=logger)
        self._stage2_warn = ThrottledWarner(interval_s=5.0, logger=logger)
        self._bounds_warn = ThrottledWarner(interval_s=5.0, logger=logger)

    # ── Public API ──────────────────────────────────────────────

    def solve(self, fingertip_positions: np.ndarray, q_prior: np.ndarray | None = None) -> np.ndarray | None:
        """Solve one frame: fingertip positions → joint angles.

        Args:
            fingertip_positions: (finger_num, 3) target positions in URDF frame,
                centered at wrist origin.
            q_prior: Optional (dof,) per-frame human-flexion reference in
                Pinocchio model order.  Only used when ``prior_weight > 0``.

        Returns:
            (dof,) joint angles in Pinocchio model order, or ``None`` when
            Stage 1 fails. The caller must hold its last valid command.
        """
        fingertip_positions = np.asarray(fingertip_positions, dtype=np.float64)
        if fingertip_positions.shape != (self.finger_num, 3):
            raise ValueError(f"fingertip_positions must have shape ({self.finger_num}, 3)")
        if not np.all(np.isfinite(fingertip_positions)):
            raise ValueError("fingertip_positions must be finite")

        if q_prior is not None:
            q_prior = np.asarray(q_prior, dtype=np.float64)
            if q_prior.shape != (self.dof,):
                raise ValueError(f"q_prior must have shape ({self.dof},)")
            if not np.all(np.isfinite(q_prior)):
                raise ValueError("q_prior must be finite")
        self._current_q_prior = q_prior

        # Scale targets (1.2× boost discourages over-flexing)
        self._current_target = fingertip_positions * self.finger_scale[:, np.newaxis]

        # ── Stage 1: Global position matching ──
        warm_start = self._bounded_qpos(self.last_qpos, "Stage 1 warm start")
        self.last_qpos = warm_start.copy()
        try:
            q_s1 = self.opt_s1.optimize(warm_start)
        except Exception:
            self._stage1_warn(
                "HandOptimizer: Stage 1 NLopt failed — caller will hold last command",
                exc_info=True,
            )
            return None
        try:
            q_s1 = self._bounded_qpos(q_s1, "Stage 1 result")
        except ValueError:
            self._stage1_warn(
                "HandOptimizer: Stage 1 returned invalid qpos — caller will hold last command",
                exc_info=True,
            )
            return None

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
            self._stage2_warn("HandOptimizer: Stage 2 NLopt failed — falling back to Stage 1", exc_info=True)
            q_s2 = q_s1.copy()
        try:
            q_s2 = self._bounded_qpos(q_s2, "Stage 2 result")
        except ValueError:
            self._stage2_warn(
                "HandOptimizer: Stage 2 returned invalid qpos — falling back to Stage 1",
                exc_info=True,
            )
            q_s2 = q_s1.copy()

        self.last_qpos = q_s2.copy()
        return q_s2

    def reset(self, qpos: np.ndarray | None = None) -> None:
        """Reset temporal state for a clean episode start.

        Args:
            qpos: Optional (dof,) joint angles to warm-start from
                  (in Pinocchio model order).
        """
        if qpos is not None and np.asarray(qpos).shape == (self.dof,) and np.all(np.isfinite(qpos)):
            qpos_array = np.asarray(qpos, dtype=np.float64)
            bounded = self._bounded_qpos(qpos_array, "reset warm start", warn_on_clip=False)
            self.last_qpos = bounded
        else:
            self.last_qpos = self._default_qpos.copy()
        self.pinch_factors = np.zeros(self.finger_num, dtype=np.float64)
        self.qpos_stage1 = None

    def _bounded_qpos(self, qpos: np.ndarray, label: str, *, warn_on_clip: bool = True) -> np.ndarray:
        """Validate and project one NLopt state into the configured box bounds."""
        qpos_array = np.asarray(qpos, dtype=np.float64)
        if qpos_array.shape != (self.dof,) or not np.all(np.isfinite(qpos_array)):
            raise ValueError(f"{label} must be a finite ({self.dof},) array")
        bounded = np.clip(qpos_array, self.joint_limits_lower, self.joint_limits_upper)
        if warn_on_clip and np.any(np.abs(bounded - qpos_array) > 1e-9):
            self._bounds_warn("HandOptimizer: projected %s into NLopt bounds", label)
        return bounded

    # ── NLopt callbacks ─────────────────────────────────────────

    def _compute_prior_gradient(self, qpos: np.ndarray) -> tuple[np.ndarray, float]:
        """Human-flexion prior gradient/loss (zeros when disabled)."""
        if self._current_q_prior is None or self.prior_mask is None or self.prior_weight <= 0:
            return np.zeros(self.dof, dtype=np.float64), 0.0
        diff = qpos - self._current_q_prior
        w = self.prior_weight * self.prior_mask
        return 2.0 * w * diff, float(np.sum(w * diff * diff))

    def _obj_s1(self, qpos: np.ndarray, grad: np.ndarray) -> float:
        """Stage 1 objective: position error + temporal smoothness + natural-hand prior."""
        self.qpos_floating[7:] = qpos
        # _current_target is set by solve() before any NLopt callback fires;
        # mypy sees np.ndarray|None but runtime is always np.ndarray here.
        g_pos, loss_pos = self.pin_grad.compute_position_gradient(self.qpos_floating, self._current_target)  # type: ignore[arg-type]
        g_smooth, loss_smooth = PinGrad.compute_smoothness_gradient(qpos, self.last_qpos, self._smooth_weight)
        g_prior, loss_prior = self._compute_prior_gradient(qpos)
        if grad.size > 0:
            grad[:] = g_pos + g_smooth + g_prior
        return float(loss_pos + loss_smooth + loss_prior)

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

        # Natural-hand prior
        g_prior, loss_prior = self._compute_prior_gradient(qpos)
        total_loss += loss_prior
        total_grad += g_prior

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
