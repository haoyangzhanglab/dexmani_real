"""VR-to-XHand retargeting via dex_retargeting + adaptive finger scaling.

Two landmark-space adaptations compensate human-robot kinematic mismatches:
1. adaptive_retargeting_thumb — scales thumb_tip to match XHand mechanical thumb length
2. adaptive_retargeting_xhand — scales pinky chain (MCP→PIP→DIP→TIP) by extension state
"""

from __future__ import annotations

__all__ = ["XHandRetargeter", "TAGHandRetargeter", "adaptive_retargeting_xhand"]

import time

import numpy as np
import yaml
from dex_retargeting.retargeting_config import RetargetingConfig

from dexmani_real import ASSET_DIR
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_palm_fallback_warn = None  # lazy ThrottledWarner — initialized on first fallback

# ── Pinky landmark indices (MediaPipe convention) ──
_PINKY_MCP = 17
_PINKY_PIP = 18
_PINKY_DIP = 19
_PINKY_TIP = 20

# Pinky adaptive scaling — 1:1 match with LeFranX reference values.
# Ref: LeFranX vr_hand_detector_adapter.py:27-84

_PINKY_MIN_EXTENSION = 0.03  # fully curled pinky MCP→TIP distance
_PINKY_MAX_EXTENSION = 0.10  # fully extended pinky MCP→TIP distance
_PINKY_BASE_SCALE = 1.2  # minimum scaling for curled state
_PINKY_MAX_SCALE = 2.2  # maximum scaling for extended state

# ── Operator-to-MANO coordinate transform ──
# (originally from deleted utils/hand_utils.py — sole caller is retarget() below)

# Operator→MANO coordinate transform (right hand).
# det = -1: improper rotation includes a reflection for left→right hand chirality flip.
_OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)


def _estimate_palm_frame(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """Estimate a palm coordinate frame (3×3 rotation matrix) from 21 hand landmarks.

    Uses wrist + index MCP + middle MCP to fit a palm plane via SVD, then
    constructs a right-handed orthonormal frame with x pointing from middle
    MCP to wrist.  Returns the identity on any numerical failure.

    Ref: LeFranX vr_hand_detector_adapter.py:293-342
    """
    keypoint_3d_array = np.asarray(keypoint_3d_array, dtype=np.float64)
    if keypoint_3d_array.shape != (21, 3):
        raise ValueError(f"keypoint_3d_array must have shape (21, 3), got {keypoint_3d_array.shape}")

    eps = 1e-8
    points = keypoint_3d_array[[0, 5, 9], :].copy()

    if not np.all(np.isfinite(points)):
        global _palm_fallback_warn
        if _palm_fallback_warn is None:
            from dexmani_real.utils.log import ThrottledWarner
            _palm_fallback_warn = ThrottledWarner(interval_s=5.0)
        _palm_fallback_warn("_estimate_palm_frame: NaN/Inf in wrist+MCP points — falling back to identity")
        return np.eye(3, dtype=np.float64)

    x_vector = points[0] - points[2]  # wrist → middle MCP
    points_centered = points - np.mean(points, axis=0, keepdims=True)

    try:
        _, _, v = np.linalg.svd(points_centered)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64)

    normal = v[2, :]
    normal_norm = np.linalg.norm(normal)
    if normal_norm < eps:
        return np.eye(3, dtype=np.float64)
    normal = normal / normal_norm

    # Gram-Schmidt
    x = x_vector - np.sum(x_vector * normal) * normal
    x_norm = np.linalg.norm(x)
    if x_norm < eps:
        return np.eye(3, dtype=np.float64)
    x = x / x_norm

    z = np.cross(x, normal)
    z_norm = np.linalg.norm(z)
    if z_norm < eps:
        return np.eye(3, dtype=np.float64)
    z = z / z_norm

    # LeFranX: use index_mcp → middle_mcp as lateral reference
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1.0
        z *= -1.0

    return np.stack([x, normal, z], axis=1)


def adaptive_retargeting_xhand(landmarks: np.ndarray) -> np.ndarray:
    """Apply adaptive pinky scaling for XHand robot (LeFranX approach).

    Compensates for human-to-robot finger length differences by adaptively
    scaling pinky chain segments based on finger extension state.
    Scales more when extended (for reaching), less when curled (for fist-making).

    Operates on MANO-space landmarks — modifies pinky PIP/DIP/TIP positions
    in-place on a copy. The scaled landmarks are then used directly by the
    existing retargeting pipeline.

    Ref: LeFranX vr_hand_detector_adapter.py:27-84

    Args:
        landmarks: (21, 3) array in MANO coordinate space.

    Returns:
        (21, 3) array with pinky chain scaled (new copy, input unchanged).
    """
    landmarks = landmarks.copy()

    # Finger extension: distance from MCP to TIP
    pinky_extension = float(np.linalg.norm(landmarks[_PINKY_TIP] - landmarks[_PINKY_MCP]))

    # Normalize extension ratio (0.0 = fully curled, 1.0 = fully extended)
    extension_ratio = np.clip(
        (pinky_extension - _PINKY_MIN_EXTENSION) / (_PINKY_MAX_EXTENSION - _PINKY_MIN_EXTENSION),
        0.0,
        1.0,
    )

    # Adaptive scaling: more when extended, less when curled
    adaptive_scale = _PINKY_BASE_SCALE + (_PINKY_MAX_SCALE - _PINKY_BASE_SCALE) * extension_ratio

    # Apply progressive scaling along the kinematic chain (MCP→PIP→DIP→TIP).
    # Each segment is extended from the (possibly modified) parent joint.
    mcp_to_pip = landmarks[_PINKY_PIP] - landmarks[_PINKY_MCP]
    landmarks[_PINKY_PIP] = landmarks[_PINKY_MCP] + mcp_to_pip * adaptive_scale

    pip_to_dip = landmarks[_PINKY_DIP] - landmarks[_PINKY_PIP]
    landmarks[_PINKY_DIP] = landmarks[_PINKY_PIP] + pip_to_dip * adaptive_scale

    dip_to_tip = landmarks[_PINKY_TIP] - landmarks[_PINKY_DIP]
    landmarks[_PINKY_TIP] = landmarks[_PINKY_DIP] + dip_to_tip * adaptive_scale

    return landmarks


class XHandRetargeter:
    def __init__(
        self,
        fixed_joint_values: np.ndarray | None = None,
        hand_type: str = "right",
        retargeting_type: str = "dexpilot",
        debug_adapters: bool = False,
        smoothing_alpha: float | None = None,
    ):
        self.hand_type = hand_type
        self.retargeting_type = retargeting_type
        self.fixed_joint_values = np.array([]) if fixed_joint_values is None else np.array(fixed_joint_values)
        self.debug_adapters = bool(debug_adapters)
        self.last_debug: dict[str, float | str] = {}
        self._smoothing_alpha: float | None = float(np.clip(smoothing_alpha, 0.0, 1.0)) if smoothing_alpha is not None else None
        self._hand_ema_state: np.ndarray | None = None

        self.urdf_joint_names = [
            "right_hand_thumb_bend_joint",
            "right_hand_thumb_rota_joint1",
            "right_hand_thumb_rota_joint2",
            "right_hand_index_bend_joint",
            "right_hand_index_joint1",
            "right_hand_index_joint2",
            "right_hand_mid_joint1",
            "right_hand_mid_joint2",
            "right_hand_ring_joint1",
            "right_hand_ring_joint2",
            "right_hand_pinky_joint1",
            "right_hand_pinky_joint2",
        ]

        self.load_retargeter()

    def load_retargeter(self):
        config_path = ASSET_DIR / "retargeting" / f"xhand_{self.hand_type}_{self.retargeting_type}.yml"

        # Read YAML — extract custom smoothing_alpha before passing to RetargetingConfig
        with open(str(config_path), "r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        cfg = yaml_config["retargeting"]

        # Teleoperator-level EMA (our custom field, not a RetargetingConfig param)
        if self._smoothing_alpha is None:
            self._smoothing_alpha = float(cfg.pop("smoothing_alpha", 0.3))

        # Standard build (same path as LeFranX)
        RetargetingConfig.set_default_urdf_dir(str(ASSET_DIR / "robots"))
        self.retargeter = RetargetingConfig.from_dict(cfg).build()

        self.indices = self.retargeter.optimizer.target_link_human_indices

        retargeter_joint_names = self.retargeter.optimizer.robot.dof_joint_names
        self.retargeted_joint_order = np.array(
            [retargeter_joint_names.index(name) for name in self.urdf_joint_names]
        ).astype(int)

    @property
    def low_pass_alpha(self) -> float:
        """Current LPFilter alpha (new-value weight: 1.0 = pass-through, →0 = freeze).

        Default from YAML config (low_pass_alpha: 0.6)."""
        return float(self.retargeter.filter.alpha)

    @low_pass_alpha.setter
    def low_pass_alpha(self, value: float) -> None:
        """Tune the LPFilter smoothing strength at runtime (default from YAML: 0.6)."""
        self.retargeter.filter.alpha = float(value)

    def _build_ref_value(self, hand_joint_pos: np.ndarray) -> np.ndarray:
        """Build reference value from hand landmarks for retargeting.

        Applies adaptive_retargeting_xhand (LeFranX pinky chain scaling)
        before computing origin→task difference vectors.
        """
        if self.retargeting_type == "position":
            return hand_joint_pos[self.indices, :]

        # LeFranX: scale pinky chain on landmarks before computing ref vectors.
        # This directly modifies PIP/DIP/TIP positions along the MCP→TIP chain,
        # compensating for human-robot finger length differences.
        scaled_landmarks = adaptive_retargeting_xhand(hand_joint_pos)

        origin_indices = self.indices[0, :]
        task_indices = self.indices[1, :]

        ref_value = scaled_landmarks[task_indices, :] - scaled_landmarks[origin_indices, :]
        return ref_value

    def retarget(self, landmarks: np.ndarray | None) -> np.ndarray | None:
        """Retarget raw VR landmarks (operator-frame, 21x3) to XHand joint qpos.

        Handles coordinate transform (operator → MANO), input validation,
        and NLP optimization.  Returns None on any failure; caller falls
        back to the previous hand command.
        """
        if landmarks is None:
            return None
        if landmarks.shape != (21, 3):
            return None
        if not np.all(np.isfinite(landmarks)):
            logger.warning("VR landmarks contain NaN/Inf — holding hand position")
            return None

        # ── Coordinate transform: operator → MANO ──
        try:
            wrist_rot = _estimate_palm_frame(landmarks)
            mano_landmarks = landmarks @ wrist_rot @ _OPERATOR2MANO_RIGHT
        except (ValueError, TypeError, np.linalg.LinAlgError):
            logger.warning("Coordinate transform failed — holding hand position")
            return None

        start_time = time.time()

        ref_value = self._build_ref_value(mano_landmarks)
        qpos = self.retargeter.retarget(ref_value, fixed_qpos=self.fixed_joint_values)

        if qpos is None:
            logger.warning("Retargeting returned None.")
            return None

        # ── Teleoperator-level EMA smoothing (ref: LeFranX smoothing_alpha=0.3) ──
        # Applied AFTER SeqRetargeting LPFilter for two-layer smoothing.
        # Early-exit when alpha >= 1.0 (default config: pass-through, disabled).
        qpos_arr = np.asarray(qpos, dtype=float)
        assert self._smoothing_alpha is not None  # set by load_retargeter()
        if self._smoothing_alpha < 1.0:
            if self._hand_ema_state is not None:
                qpos_arr = self._smoothing_alpha * qpos_arr + (1.0 - self._smoothing_alpha) * self._hand_ema_state
            self._hand_ema_state = qpos_arr.copy()
        else:
            self._hand_ema_state = None

        # Joint order remap
        qpos_arr = qpos_arr[self.retargeted_joint_order]

        if self.debug_adapters:
            self.last_debug = {
                "retarget_ms": 1000 * (time.time() - start_time),
                "adaptives": "pinky(LeFranX chain scaling)",
            }
            logger.info("retarget_debug: %s", self.last_debug)

        return qpos_arr

    def reset(self, initial_qpos: np.ndarray | None = None) -> None:
        """Reset retargeter state for a clean episode start.

        Resets the SLSQP warm-start seed, the LPFilter EMA accumulator,
        the teleoperator EMA state, and the DexPilot projection indicators.

        Args:
            initial_qpos: Optional (12,) array of current hand joint positions.
                When provided, seeds the SLSQP warm-start with the actual hardware
                pose instead of joint_limits.mean(1) (neutral).  This eliminates
                the first-frame NLP convergence cost — the seed is already near
                the optimum, so SLSQP converges in 1-2 iterations instead of
                potentially dozens.  Reduces between-session timing variance from
                ~2.4× to near-zero.
        """
        self.retargeter.reset()  # SeqRetargeting: resets last_qpos + counters
        if self.retargeter.filter is not None:
            self.retargeter.filter.reset()  # LPFilter: clears EMA accumulator
        self.retargeter.optimizer.projected[:] = False  # DexPilot: clears projection state
        self._hand_ema_state = None  # Teleoperator EMA: clear for fresh episode

        # ── Smart warm-start: seed with actual hardware pose ──
        # Without this, last_qpos is joint_limits.mean(1) — a neutral pose that
        # can be far from the operator's current hand shape.  SLSQP must then
        # converge from neutral → actual, costing more iterations (and wall time)
        # on the first frame.  Seeding with the real hardware position makes the
        # first-frame optimization nearly trivial.
        if initial_qpos is not None and initial_qpos.shape == (12,):
            qpos = np.asarray(initial_qpos, dtype=np.float32)
            if np.all(np.isfinite(qpos)):
                # Remap from hardware joint order (urdf_joint_names) to
                # retargeter internal order before subsetting by pin2target.
                qpos_retargeter = qpos[self.retargeted_joint_order]
                idx = self.retargeter.optimizer.idx_pin2target
                self.retargeter.last_qpos = qpos_retargeter[idx]
            else:
                logger.warning("initial_qpos contains NaN/Inf — falling back to neutral seed")


# ═══════════════════════════════════════════════════════════════════════════════
# MediaPipe fingertip landmark indices
# ═══════════════════════════════════════════════════════════════════════════════

_FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=np.intp)
"""MediaPipe hand landmark indices for fingertips: thumb, index, middle, ring, pinky."""


# ═══════════════════════════════════════════════════════════════════════════════
# TAGHandRetargeter — TAG two-stage NLopt hand retargeting backend
# ═══════════════════════════════════════════════════════════════════════════════


class TAGHandRetargeter:
    """VR-to-XHand retargeting via TAG's two-stage NLopt optimization.

    Same protocol as ``XHandRetargeter`` (``__init__``, ``retarget()``, ``reset()``)
    so the policy loop can use either class without code changes.

    Pipeline::

        VR landmarks (21,3) operator frame
            → _estimate_palm_frame + _OPERATOR2MANO_RIGHT  (MANO frame)
            → adaptive_retargeting_xhand                    (pinky chain scaling)
            → extract fingertips [4,8,12,16,20] - wrist[0]  (wrist-centered)
            → rotate by R_mano_to_urdf                      (align to URDF frame)
            → HandOptimizer.solve()                         (two-stage NLopt)
            → model→SDK joint order remap
            → EMA post-filter
            → (12,) SDK-order qpos

    Parameters
    ----------
    hand_type:
        ``"right"`` (default).
    smoothing_alpha:
        Teleoperator-level EMA smoothing factor (0.0 = freeze, 1.0 = pass-through).
        Default 0.3 provides moderate smoothing at 16 Hz.
    debug:
        If True, log per-frame retargeting timing.
    """

    def __init__(
        self,
        hand_type: str = "right",
        smoothing_alpha: float = 0.3,
        debug: bool = False,
    ) -> None:
        from scipy.spatial.transform import Rotation

        from dexmani_real.config.defaults import hand as hand_d, tag_retargeting as tag_cfg
        from dexmani_real.teleop.tag_retargeting.optimizer import HandOptimizer

        # ── Load URDF, read joint limits ──
        urdf_path = str(ASSET_DIR / "robots" / "xhand" / f"xhand_{hand_type}.urdf")
        model = pin_loading(urdf_path)
        joint_lo = model.lowerPositionLimit[7:].copy()
        joint_hi = model.upperPositionLimit[7:].copy()

        # ── Joint order mapping ──────────────────────────────────────────
        # Pinocchio parses URDF joints in a different order than the canonical
        # SDK order used by XHand driver.  We build two mapping arrays:
        #
        #   _mapping_model_to_sdk:  for each SDK joint i, its Pinocchio model index.
        #     Usage: qpos_model[_mapping_model_to_sdk] → SDK-order qpos (retarget output).
        #   _mapping_sdk_to_model:  for each model joint i, its SDK index.
        #     Usage: qpos_sdk[_mapping_sdk_to_model] → model-order qpos (reset warm-start).
        #
        # Pinocchio model order:  index_bend, index_j1, index_j2, mid_j1, mid_j2,
        #   pinky_j1, pinky_j2, ring_j1, ring_j2, thumb_bend, thumb_rota_j1, thumb_rota_j2
        # Canonical SDK order:    thumb_bend, thumb_rota_j1, thumb_rota_j2, index_bend, …
        model_names = list(model.names[2:])  # skip "universe" and "root_joint"
        self.urdf_joint_names = [
            "right_hand_thumb_bend_joint",
            "right_hand_thumb_rota_joint1",
            "right_hand_thumb_rota_joint2",
            "right_hand_index_bend_joint",
            "right_hand_index_joint1",
            "right_hand_index_joint2",
            "right_hand_mid_joint1",
            "right_hand_mid_joint2",
            "right_hand_ring_joint1",
            "right_hand_ring_joint2",
            "right_hand_pinky_joint1",
            "right_hand_pinky_joint2",
        ]
        self._mapping_model_to_sdk = np.array(
            [model_names.index(n) for n in self.urdf_joint_names], dtype=np.intp
        )
        self._mapping_sdk_to_model = np.argsort(self._mapping_model_to_sdk)

        # ── Merge driver-enforced joint limits ───────────────────────────
        # URDF limits are used as NLopt bounds, but the XHand driver silently
        # clamps qpos to its own [qpos_min, qpos_max] (anti-clogging margins).
        # For 5 distal joints the driver lower bound is more restrictive than
        # the URDF (thumb_rota_j2: +10°, index/mid/ring/pinky_j2: +5°).
        # Passing the intersection to the optimizer prevents it from proposing
        # unreachable angles that would cause FK-predicted ≠ actual fingertip positions.
        from dexmani_real.robot.xhand import XHandConfig as _XHandConfig

        _driver_cfg = _XHandConfig()
        _driver_lo_sdk = np.asarray(_driver_cfg.qpos_min, dtype=np.float64)  # rad, SDK order
        _driver_hi_sdk = np.asarray(_driver_cfg.qpos_max, dtype=np.float64)  # rad, SDK order
        _driver_lo_model = _driver_lo_sdk[self._mapping_sdk_to_model]
        _driver_hi_model = _driver_hi_sdk[self._mapping_sdk_to_model]
        joint_lo = np.maximum(joint_lo, _driver_lo_model)
        joint_hi = np.minimum(joint_hi, _driver_hi_model)

        # ── Optimizer ──
        self._optimizer = HandOptimizer(
            urdf_path=urdf_path,
            fingertip_frame_names=list(hand_d.fingertip_link_names),
            joint_limits_lower=joint_lo,
            joint_limits_upper=joint_hi,
            finger_lengths_robot=np.array(tag_cfg.robot_finger_lengths, dtype=np.float64),
            finger_lengths_human=np.array(tag_cfg.human_finger_lengths, dtype=np.float64),
            finger_scale_boost=tag_cfg.finger_scale_boost,
            smooth_weight=tag_cfg.smooth_weight,
            ftol_abs_s1=tag_cfg.ftol_abs_s1,
            maxeval_s1=tag_cfg.maxeval_s1,
            ftol_abs_s2=tag_cfg.ftol_abs_s2,
            maxeval_s2=tag_cfg.maxeval_s2,
            pinch_base_weight=tag_cfg.pinch_base_weight,
            pinch_start_dist_m=tag_cfg.pinch_start_dist_m,
            pinch_full_dist_m=tag_cfg.pinch_full_dist_m,
            pinch_ema_alpha=tag_cfg.pinch_ema_alpha,
            pinch_skip_threshold=tag_cfg.pinch_skip_threshold,
            reg_stage1_weight=tag_cfg.reg_stage1_weight,
            reg_last_weight=tag_cfg.reg_last_weight,
        )

        # ── Pre-computed transforms (avoid per-frame allocation) ──
        self._R_mano_to_urdf: np.ndarray = Rotation.from_euler("xyz", tag_cfg.mano_to_urdf_euler).as_matrix()
        # Identity: MANO +Z (finger extension) → URDF +Z (finger extension). Verified 2026-08-07.

        # ── Smoothing & debug ──
        self._smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self._ema_state: np.ndarray | None = None
        self.debug = bool(debug)

        logger.info(
            "TAGHandRetargeter ready (urdf=%s, smoothing_alpha=%.2f, mano→urdf=%s)",
            urdf_path,
            self._smoothing_alpha,
            tag_cfg.mano_to_urdf_euler,
        )

    # ── Public API (compatible with XHandRetargeter) ────────────

    def retarget(self, landmarks: np.ndarray | None) -> np.ndarray | None:
        """Retarget VR landmarks (operator-frame, 21×3) to XHand joint qpos (12,).

        Returns None on any failure — caller falls back to previous hand command.
        """
        if landmarks is None:
            return None
        if landmarks.shape != (21, 3):
            return None
        if not np.all(np.isfinite(landmarks)):
            logger.warning("TAGHandRetargeter: landmarks contain NaN/Inf — holding position")
            return None

        t0 = time.perf_counter() if self.debug else 0.0

        # 1. Coordinate transform: operator → MANO (reuse existing helpers)
        try:
            wrist_rot = _estimate_palm_frame(landmarks)
            mano = landmarks @ wrist_rot @ _OPERATOR2MANO_RIGHT
        except (ValueError, np.linalg.LinAlgError):
            logger.warning("TAGHandRetargeter: coordinate transform failed — holding position")
            return None

        # 2. Adaptive pinky chain scaling (reuse existing)
        mano = adaptive_retargeting_xhand(mano)

        # 3. Extract 5 fingertip positions → wrist-centered → rotate to URDF frame
        tips = mano[_FINGERTIP_INDICES].copy()  # (5, 3) in MANO frame
        tips -= mano[0]  # center at wrist
        tips_urdf = tips @ self._R_mano_to_urdf.T  # (5, 3) in URDF frame

        # 4. Two-stage NLopt optimization
        try:
            qpos_model = self._optimizer.solve(tips_urdf)  # (12,) in Pinocchio model order
        except Exception:
            logger.warning("TAGHandRetargeter: optimizer.solve() crashed — holding position", exc_info=True)
            return None

        if qpos_model is None:
            return None

        # 5. Joint order remap: model order → canonical SDK order
        qpos_sdk = qpos_model[self._mapping_model_to_sdk]

        # 6. Teleoperator-level EMA post-filter
        if self._smoothing_alpha < 1.0:
            if self._ema_state is not None:
                qpos_sdk = self._smoothing_alpha * qpos_sdk + (1.0 - self._smoothing_alpha) * self._ema_state
            self._ema_state = qpos_sdk.copy()
        else:
            self._ema_state = None

        if self.debug:
            dt_ms = 1000.0 * (time.perf_counter() - t0)
            logger.info("TAGHandRetargeter: retarget %.2f ms", dt_ms)

        return qpos_sdk

    def reset(self, initial_qpos: np.ndarray | None = None) -> None:
        """Reset retargeter state for a clean episode start.

        Args:
            initial_qpos: Optional (12,) current hand joint positions in SDK order.
                When provided, seeds the NLopt warm-start from the actual hardware
                pose so the first-frame optimization converges from near-optimum.
        """
        # Clear teleoperator EMA state
        self._ema_state = None

        if initial_qpos is not None and initial_qpos.shape == (12,) and np.all(np.isfinite(initial_qpos)):
            # SDK order → model order for optimizer warm-start
            qpos_model = initial_qpos[self._mapping_sdk_to_model]
            self._optimizer.reset(qpos_model)
        else:
            self._optimizer.reset(None)

    @property
    def low_pass_alpha(self) -> float:
        """Current EMA smoothing alpha (1.0 = pass-through)."""
        return self._smoothing_alpha

    @low_pass_alpha.setter
    def low_pass_alpha(self, value: float) -> None:
        """Tune the EMA smoothing strength at runtime."""
        self._smoothing_alpha = float(np.clip(value, 0.0, 1.0))


# ── Lazy Pinocchio URDF loader (avoids import at module level) ──


def pin_loading(urdf_path: str):
    """Load a URDF with Pinocchio FreeFlyer model.

    Extracted so TAGHandRetargeter.__init__ doesn't need a module-level
    pinocchio import (consistent with the project's lazy-SDK-import pattern).
    """
    import pinocchio as pin

    return pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
