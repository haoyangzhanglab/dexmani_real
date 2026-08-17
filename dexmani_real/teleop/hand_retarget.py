"""VR-to-XHand retargeting: TAG (in-repo NLopt) and DexPilot (dex_retargeting).

One landmark-space adaptation compensates the human-robot kinematic mismatch:
``adaptive_retargeting_xhand`` scales the pinky chain (MCP→PIP→DIP→TIP) by a
constant per-backend ``pinky_scale`` (plus an optional palm-baseline offset).
"""

from __future__ import annotations

__all__ = ["XHandRetargeter", "TAGHandRetargeter", "adaptive_retargeting_xhand", "validate_landmarks"]

import time
from typing import Any

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.utils.schema import HAND_JOINT_SHAPE, XHAND_SDK_JOINT_NAMES
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_palm_fallback_warn = None  # lazy ThrottledWarner — initialized on first fallback


class _TAGRuntimeOverrides:
    """Private policy-to-TAG wiring that leaves the public constructor stable."""

    def __init__(self, config: Any | None, urdf_path: str) -> None:
        self.config = config
        self.urdf_path = str(urdf_path)


def _tag_config_with_urdf(config: Any | None, urdf_path: str) -> _TAGRuntimeOverrides:
    return _TAGRuntimeOverrides(config, urdf_path)


# ── Pinky landmark indices (MediaPipe convention) ──
_PINKY_MCP = 17
_PINKY_PIP = 18
_PINKY_DIP = 19
_PINKY_TIP = 20

# Pinky scale.  Production injects the recalibrated per-backend ``pinky_scale``
# (see config defaults); this fallback matches the DexPilot default so a no-config
# ``XHandRetargeter`` stays stable.  The original LeFranX extension-dependent
# interpolation (a 1.2→2.2 range keyed on the pinky extension ratio) is removed.
# Ref: LeFranX vr_hand_detector_adapter.py:27-84

_PINKY_SCALE_FALLBACK = 1.15

_CONTIGUOUS_BONES = tuple(
    (parent, child)
    for chain in (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20),
    )
    for parent, child in zip(chain, chain[1:])
)

_FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)

# Human-flexion → robot-joint mapping (feature_index → SDK joint index).  Mirrors
# hand_retarget_eval._FLEXION_FEATURE_JOINT_PAIRS: the human CMC/MCP/PIP+DIP
# flexion angles map onto the robot flexion joints.  thumb_rota1 and index_bend
# have no landmark-flexion mapping and are excluded from the prior.
_HUMAN_FLEXION_JOINT_PAIRS = (
    (0, 0), (1, 2),  # thumb CMC→bend, MCP+IP→rota2
    (2, 4), (3, 5),  # index MCP→j1, PIP+DIP→j2
    (4, 6), (5, 7),  # middle
    (6, 8), (7, 9),  # ring
    (8, 10), (9, 11),  # pinky
)
_SDK_FLEXION_MASK = np.zeros(12, dtype=np.float64)
for _feature_index, _sdk_index in _HUMAN_FLEXION_JOINT_PAIRS:
    _SDK_FLEXION_MASK[_sdk_index] = 1.0


def _human_flexion_rad(landmarks: np.ndarray) -> np.ndarray:
    """Human flexion reference (10,) from one (21, 3) landmark frame.

    Rotation-invariant per-finger angles: thumb CMC + MCP+IP, then per finger
    (index/mid/ring/pinky) MCP + PIP+DIP — the same feature order as
    ``_HUMAN_FLEXION_JOINT_PAIRS`` and ``hand_retarget_eval.extract_hand_features``.
    """
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.shape != (21, 3):
        raise ValueError(f"landmarks must have shape (21, 3), got {pts.shape}")
    flex = np.empty(10, dtype=np.float64)
    for finger_index, chain in enumerate(_FINGER_CHAINS):
        bones = [pts[chain[index + 1]] - pts[chain[index]] for index in range(4)]
        ang = np.empty(3, dtype=np.float64)
        for joint_index, (first, second) in enumerate(zip(bones, bones[1:])):
            denominator = np.linalg.norm(first) * np.linalg.norm(second)
            if denominator <= 1e-12:
                raise ValueError("landmarks contain a degenerate hand bone")
            ang[joint_index] = np.arccos(np.clip(np.sum(first * second) / denominator, -1.0, 1.0))
        if finger_index == 0:
            flex[0] = ang[0]
            flex[1] = ang[1] + ang[2]
        else:
            base = 2 * finger_index
            flex[base] = ang[0]
            flex[base + 1] = ang[1] + ang[2]
    return flex


def _human_flexion_sdk_reference(landmarks: np.ndarray) -> np.ndarray:
    """(12,) SDK-order prior reference: human flexion at the 10 flexion joints, 0 elsewhere."""
    flex = _human_flexion_rad(landmarks)
    reference = np.zeros(12, dtype=np.float64)
    for feature_index, sdk_index in _HUMAN_FLEXION_JOINT_PAIRS:
        reference[sdk_index] = flex[feature_index]
    return reference


# Operator→MANO coordinate transform (right hand).
# det = +1: this is a proper rotation.  Unity left-handed → FLU chirality conversion
# has already happened once in sensor/vr_receiver_process.py.
_OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)


def validate_landmarks(keypoint_3d_array: np.ndarray) -> tuple[bool, str]:
    """Apply the fail-closed geometric gate before touching temporal state."""
    points = np.asarray(keypoint_3d_array, dtype=np.float64)
    if points.shape != (21, 3):
        return False, f"shape {points.shape} != (21, 3)"
    if not np.all(np.isfinite(points)):
        return False, "contains NaN/Inf"
    index_basis = points[5] - points[0]
    pinky_basis = points[17] - points[0]
    index_length = float(np.linalg.norm(index_basis))
    pinky_length = float(np.linalg.norm(pinky_basis))
    if index_length < 0.01 or pinky_length < 0.01:
        return False, "wrist-to-index/pinky MCP baseline is shorter than 1 cm"
    palm_sine = float(np.linalg.norm(np.cross(index_basis, pinky_basis)) / (index_length * pinky_length))
    if palm_sine < 0.1:
        return False, "palm basis is collinear"
    shortest_bone = min(float(np.linalg.norm(points[child] - points[parent])) for parent, child in _CONTIGUOUS_BONES)
    if shortest_bone < 0.002:
        return False, "a retargeting bone is shorter than 2 mm"
    return True, ""


def _estimate_palm_frame(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """Estimate a palm coordinate frame (3×3 rotation matrix) from 21 hand landmarks.

    Uses wrist + index MCP + middle MCP to fit a palm plane via SVD, then
    constructs a right-handed orthonormal frame with x pointing from middle
    MCP to wrist. The caller must first run ``validate_landmarks()``.

    Ref: LeFranX vr_hand_detector_adapter.py:293-342
    """
    keypoint_3d_array = np.asarray(keypoint_3d_array, dtype=np.float64)
    if keypoint_3d_array.shape != (21, 3):
        raise ValueError(f"keypoint_3d_array must have shape (21, 3), got {keypoint_3d_array.shape}")

    eps = 1e-8
    points = keypoint_3d_array[[0, 5, 9], :].copy()

    x_vector = points[0] - points[2]  # middle MCP → wrist
    points_centered = points - np.mean(points, axis=0, keepdims=True)

    try:
        _, _, v = np.linalg.svd(points_centered)
    except np.linalg.LinAlgError as exc:
        raise ValueError("palm SVD failed") from exc

    normal = v[2, :]
    normal_norm = np.linalg.norm(normal)
    if normal_norm < eps:
        raise ValueError("palm normal is degenerate")
    normal = normal / normal_norm

    # Gram-Schmidt
    x = x_vector - np.sum(x_vector * normal) * normal
    x_norm = np.linalg.norm(x)
    if x_norm < eps:
        raise ValueError("palm longitudinal axis is degenerate")
    x = x / x_norm

    z = np.cross(x, normal)
    z_norm = np.linalg.norm(z)
    if z_norm < eps:
        raise ValueError("palm lateral axis is degenerate")
    z = z / z_norm

    # LeFranX: use index_mcp → middle_mcp as lateral reference
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1.0
        z *= -1.0

    return np.stack([x, normal, z], axis=1)


def adaptive_retargeting_xhand(
    landmarks: np.ndarray,
    *,
    scale: float = _PINKY_SCALE_FALLBACK,
    palm_scale: float = 1.0,
) -> np.ndarray:
    """Apply constant pinky-chain scaling for the XHand (LeFranX-derived).

    Compensates the human-to-robot finger length mismatch by scaling the pinky
    chain (MCP→PIP→DIP→TIP) by ``scale`` and, when ``palm_scale != 1.0``, the
    pinky wrist→MCP baseline independently.  The original LeFranX
    extension-dependent interpolation (a 1.2→2.2 range keyed on the pinky
    extension ratio) has been superseded by a constant per-backend
    ``pinky_scale``; ``scale`` is applied uniformly.

    Operates on MANO-space landmarks — modifies pinky PIP/DIP/TIP positions
    in-place on a copy. The scaled landmarks are then used directly by the
    existing retargeting pipeline.

    Ref: LeFranX vr_hand_detector_adapter.py:27-84

    Args:
        landmarks: (21, 3) array in MANO coordinate space.
        scale: Pinky-chain scale.  The backends inject their resolved
            ``pinky_scale`` (a constant); the module constant is only the
            no-config fallback.
        palm_scale: Scale on the pinky wrist→MCP baseline (1.0 = no-op).  The
            backends inject their resolved ``pinky_palm_scale``; the module
            fallback is 1.0.

    Returns:
        (21, 3) array with pinky chain scaled (new copy, input unchanged).
    """
    landmarks = np.asarray(landmarks, dtype=np.float64).copy()
    if landmarks.shape != (21, 3) or not np.all(np.isfinite(landmarks)):
        raise ValueError("landmarks must be a finite (21, 3) array")
    # Preserve every parent→child vector from the unmodified input.  Computing
    # distal segments after moving their parent compounds the translation and
    # is not a kinematic scaling of the original pinky.
    raw = landmarks.copy()

    # Palm offset: scale the wrist→MCP baseline independently of the finger chain.
    # The robot pinky MCP is proportionally farther from the wrist than the human's
    # (URDF pinky_joint1 origin ≈0.1085 m vs human ≈0.0865 m ≈ 1.25×), which the
    # distal-only scale cannot express.  Moving the MCP landmark radially from the
    # wrist lengthens the wrist→tip reference so the solver need not over-flex the
    # MCP to shorten reach.  palm_scale == 1.0 is a no-op.
    if palm_scale != 1.0:
        landmarks[_PINKY_MCP] = landmarks[0] + (raw[_PINKY_MCP] - landmarks[0]) * palm_scale

    # Apply uniform scaling along the kinematic chain (MCP→PIP→DIP→TIP).
    # Each segment is extended from the (possibly modified) parent joint.
    mcp_to_pip = raw[_PINKY_PIP] - raw[_PINKY_MCP]
    landmarks[_PINKY_PIP] = landmarks[_PINKY_MCP] + mcp_to_pip * scale

    pip_to_dip = raw[_PINKY_DIP] - raw[_PINKY_PIP]
    landmarks[_PINKY_DIP] = landmarks[_PINKY_PIP] + pip_to_dip * scale

    dip_to_tip = raw[_PINKY_TIP] - raw[_PINKY_DIP]
    landmarks[_PINKY_TIP] = landmarks[_PINKY_DIP] + dip_to_tip * scale

    return landmarks


class XHandRetargeter:
    def __init__(
        self,
        fixed_joint_values: np.ndarray | None = None,
        hand_type: str = "right",
        retargeting_type: str = "dexpilot",
        debug_adapters: bool = False,
        dexpilot_config: Any | None = None,
    ):
        self.hand_type = hand_type
        self.retargeting_type = retargeting_type
        self.fixed_joint_values = np.array([]) if fixed_joint_values is None else np.array(fixed_joint_values)
        self.debug_adapters = bool(debug_adapters)
        self._dexpilot_config = dexpilot_config
        if dexpilot_config is not None:
            self._pinky_scale = float(dexpilot_config.pinky_scale)
            self._pinky_palm_scale = float(dexpilot_config.pinky_palm_scale)
        else:
            self._pinky_scale = _PINKY_SCALE_FALLBACK
            self._pinky_palm_scale = 1.0
        self.last_debug: dict[str, float | str] = {}

        # Every returned qpos follows the cross-process SDK order owned by
        # utils.schema, independent of the backend model's internal order.
        self.sdk_joint_names = XHAND_SDK_JOINT_NAMES

        self.load_retargeter()

    def load_retargeter(self):
        # Import only the selected DexPilot backend.  TAG deployments should
        # not load dex-retargeting, its native dependencies, or model assets.
        import yaml
        from dex_retargeting.retargeting_config import RetargetingConfig

        from dexmani_real.teleop.dexpilot_prior import build_dexpilot_retargeting

        config_path = ASSET_DIR / "retargeting" / f"xhand_{self.hand_type}_{self.retargeting_type}.yml"

        with open(str(config_path), "r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        cfg = yaml_config["retargeting"]

        # The YAML list is required by dex-retargeting, but it may not define a
        # competing qpos order. Fail closed if it drifts from the schema.
        configured_joint_names = tuple(cfg.get("target_joint_names", ()))
        if configured_joint_names != XHAND_SDK_JOINT_NAMES:
            raise ValueError(
                "DexPilot target_joint_names must exactly match the canonical "
                "XHand SDK joint order"
            )

        if self._dexpilot_config is not None:
            cfg.update(
                scaling_factor=float(self._dexpilot_config.scaling_factor),
                low_pass_alpha=float(self._dexpilot_config.low_pass_alpha),
                project_dist=float(self._dexpilot_config.project_dist_m),
                escape_dist=float(self._dexpilot_config.escape_dist_m),
            )

        self._prior_weight = float(self._dexpilot_config.prior_weight) if self._dexpilot_config is not None else 0.0
        prior_mask = _SDK_FLEXION_MASK  # target order == SDK order

        # Build with the human-flexion prior (in-repo subclass; same path as LeFranX)
        RetargetingConfig.set_default_urdf_dir(str(ASSET_DIR / "robots"))
        self.retargeter = build_dexpilot_retargeting(
            RetargetingConfig.from_dict(cfg),
            prior_weight=self._prior_weight,
            prior_mask=prior_mask,
        )

        self.indices = self.retargeter.optimizer.target_link_human_indices

        retargeter_joint_names = self.retargeter.optimizer.robot.dof_joint_names
        self.retargeted_joint_order = np.array(
            [retargeter_joint_names.index(name) for name in self.sdk_joint_names]
        ).astype(int)
        self.inverse_retargeted_joint_order = np.argsort(self.retargeted_joint_order)

    @property
    def low_pass_alpha(self) -> float:
        """Current LPFilter alpha (new-value weight: 1.0 = pass-through, →0 = freeze).

        The YAML fallback is 0.6; production injects the resolved runtime value.
        """
        return float(self.retargeter.filter.alpha)

    @low_pass_alpha.setter
    def low_pass_alpha(self, value: float) -> None:
        """Tune the LPFilter smoothing strength at runtime."""
        self.retargeter.filter.alpha = float(value)

    def _build_ref_value(self, hand_joint_pos: np.ndarray) -> np.ndarray:
        """Build reference value from hand landmarks for retargeting.

        Applies adaptive_retargeting_xhand (pinky chain scaling) before
        computing origin→task difference vectors.
        """
        # Scale the pinky chain on landmarks before computing ref vectors.  This
        # directly modifies PIP/DIP/TIP positions along the MCP→TIP chain,
        # compensating for human-robot finger length differences.
        scaled_landmarks = adaptive_retargeting_xhand(
            hand_joint_pos,
            scale=self._pinky_scale,
            palm_scale=self._pinky_palm_scale,
        )

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
        valid, reason = validate_landmarks(landmarks)
        if not valid:
            logger.warning("VR landmarks rejected (%s) — holding hand position", reason)
            return None

        # ── Coordinate transform: operator → MANO ──
        try:
            wrist_rot = _estimate_palm_frame(landmarks)
            mano_landmarks = landmarks @ wrist_rot @ _OPERATOR2MANO_RIGHT
        except (ValueError, TypeError, np.linalg.LinAlgError):
            logger.warning("Coordinate transform failed — holding hand position")
            return None

        start_time = time.time()

        # Human-flexion prior reference (before pinky scaling).
        if self._prior_weight > 0:
            self.retargeter.optimizer.set_prior_reference(_human_flexion_sdk_reference(mano_landmarks))

        ref_value = self._build_ref_value(mano_landmarks)
        qpos = self.retargeter.retarget(ref_value, fixed_qpos=self.fixed_joint_values)

        if qpos is None:
            logger.warning("Retargeting returned None.")
            return None

        qpos_arr = np.asarray(qpos, dtype=float)

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

        Resets the SLSQP warm-start seed, the LPFilter accumulator, and the
        DexPilot projection indicators.

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

        # ── Smart warm-start: seed with actual hardware pose ──
        # Without this, last_qpos is joint_limits.mean(1) — a neutral pose that
        # can be far from the operator's current hand shape.  SLSQP must then
        # converge from neutral → actual, costing more iterations (and wall time)
        # on the first frame.  Seeding with the real hardware position makes the
        # first-frame optimization nearly trivial.
        if initial_qpos is not None and initial_qpos.shape == HAND_JOINT_SHAPE:
            qpos = np.asarray(initial_qpos, dtype=np.float32)
            if np.all(np.isfinite(qpos)):
                # Remap from canonical hardware/SDK joint order to
                # retargeter internal order before subsetting by pin2target.
                qpos_retargeter = qpos[self.inverse_retargeted_joint_order]
                idx = self.retargeter.optimizer.idx_pin2target
                self.retargeter.last_qpos = qpos_retargeter[idx]
            else:
                logger.warning("initial_qpos contains NaN/Inf — falling back to neutral seed")


# MediaPipe fingertip landmark indices

_FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=np.intp)


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
            → (12,) SDK-order qpos

    Parameters
    ----------
    hand_type:
        ``"right"`` (default).
    debug:
        If True, log per-frame retargeting timing.
    """

    def __init__(
        self,
        hand_type: str = "right",
        debug: bool = False,
        fingertip_link_names: tuple[str, ...] | None = None,
        tag_config: Any | None = None,
    ) -> None:
        from scipy.spatial.transform import Rotation

        from dexmani_real.config.defaults import hand as hand_d
        from dexmani_real.config.defaults import tag_retargeting as default_tag_cfg
        from dexmani_real.teleop.tag_retargeting.optimizer import HandOptimizer
        from dexmani_real.teleop.tag_retargeting.pin_grad import validate_fingertip_frame_names

        runtime_urdf_path: str | None = None
        if isinstance(tag_config, _TAGRuntimeOverrides):
            runtime_urdf_path = tag_config.urdf_path
            tag_config = tag_config.config
        tag_cfg = default_tag_cfg if tag_config is None else tag_config

        # ── Load URDF, read joint limits ──
        resolved_urdf_path = (
            str(ASSET_DIR / "robots" / "xhand" / f"xhand_{hand_type}.urdf")
            if runtime_urdf_path is None
            else runtime_urdf_path
        )
        resolved_tip_names = validate_fingertip_frame_names(
            hand_d.fingertip_link_names if fingertip_link_names is None else fingertip_link_names
        )
        model = pin_loading(resolved_urdf_path)
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
        self.sdk_joint_names = XHAND_SDK_JOINT_NAMES
        self._mapping_model_to_sdk = np.array(
            [model_names.index(name) for name in self.sdk_joint_names], dtype=np.intp
        )
        self._mapping_sdk_to_model = np.argsort(self._mapping_model_to_sdk)

        # ── Optimizer ──
        # NLopt bounds are the URDF mechanical range, not the operator-set
        # anti-clogging command floor. The measured warm-start pose is real
        # hardware state (e.g. ~4.4° below a 5° command floor) and must not be
        # projected into a stricter box; the command floor is applied later by
        # clipping the published command.

        self._optimizer = HandOptimizer(
            urdf_path=resolved_urdf_path,
            fingertip_frame_names=list(resolved_tip_names),
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
            prior_weight=tag_cfg.prior_weight,
            prior_mask=_SDK_FLEXION_MASK[self._mapping_sdk_to_model],
        )

        # ── Pre-computed transforms (avoid per-frame allocation) ──
        self._R_mano_to_urdf: np.ndarray = Rotation.from_euler("xyz", tag_cfg.mano_to_urdf_euler).as_matrix()
        # MANO and URDF both use +Z for finger extension.
        self._pinky_scale = float(tag_cfg.pinky_scale)
        self._pinky_palm_scale = float(tag_cfg.pinky_palm_scale)
        self._prior_weight = float(tag_cfg.prior_weight)

        # ── State & debug ──
        self._last_raw_qpos: np.ndarray | None = None
        self.debug = bool(debug)

        logger.info(
            "TAGHandRetargeter ready (urdf=%s, mano→urdf=%s)",
            resolved_urdf_path,
            tag_cfg.mano_to_urdf_euler,
        )

    # ── Public API (compatible with XHandRetargeter) ────────────

    def retarget(self, landmarks: np.ndarray | None) -> np.ndarray | None:
        """Retarget VR landmarks (operator-frame, 21×3) to XHand joint qpos (12,).

        Returns None on any failure — caller falls back to previous hand command.
        """
        if landmarks is None:
            return None
        valid, reason = validate_landmarks(landmarks)
        if not valid:
            logger.warning("TAGHandRetargeter: landmarks rejected (%s) — holding position", reason)
            return None

        t0 = time.perf_counter() if self.debug else 0.0

        # 1. Coordinate transform: operator → MANO (reuse existing helpers)
        try:
            wrist_rot = _estimate_palm_frame(landmarks)
            mano = landmarks @ wrist_rot @ _OPERATOR2MANO_RIGHT
        except (ValueError, np.linalg.LinAlgError):
            logger.warning("TAGHandRetargeter: coordinate transform failed — holding position")
            return None

        # 2. Human-flexion prior reference (before pinky scaling — the
        #    reference is the operator's actual flexion, not the robot-scaled chain)
        q_prior_model = None
        if self._prior_weight > 0:
            q_prior_model = _human_flexion_sdk_reference(mano)[self._mapping_sdk_to_model]

        # 3. Pinky chain scaling (reuse existing)
        mano = adaptive_retargeting_xhand(
            mano,
            scale=self._pinky_scale,
            palm_scale=self._pinky_palm_scale,
        )

        # 4. Extract 5 fingertip positions → wrist-centered → rotate to URDF frame
        tips = mano[_FINGERTIP_INDICES].copy()  # (5, 3) in MANO frame
        tips -= mano[0]  # center at wrist
        tips_urdf = tips @ self._R_mano_to_urdf.T  # (5, 3) in URDF frame

        # 5. Two-stage NLopt optimization
        try:
            qpos_model = self._optimizer.solve(tips_urdf, q_prior=q_prior_model)  # (12,) in Pinocchio model order
        except Exception:
            logger.warning("TAGHandRetargeter: optimizer.solve() crashed — holding position", exc_info=True)
            return None

        if qpos_model is None:
            return None

        # 5. Joint order remap: model order → canonical SDK order
        qpos_sdk = qpos_model[self._mapping_model_to_sdk]
        self._last_raw_qpos = qpos_sdk.copy()

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
        # Clear last raw output state
        self._last_raw_qpos = None

        if initial_qpos is not None and initial_qpos.shape == HAND_JOINT_SHAPE and np.all(np.isfinite(initial_qpos)):
            # SDK order → model order for optimizer warm-start
            qpos_model = initial_qpos[self._mapping_sdk_to_model]
            self._optimizer.reset(qpos_model)
        else:
            self._optimizer.reset(None)

    @property
    def last_raw_qpos(self) -> np.ndarray | None:
        """Latest SDK-order retarget output."""
        return self._last_raw_qpos.copy() if self._last_raw_qpos is not None else None


# ── Lazy Pinocchio URDF loader (avoids import at module level) ──


def pin_loading(urdf_path: str):
    """Load a URDF with Pinocchio FreeFlyer model.

    Extracted so TAGHandRetargeter.__init__ doesn't need a module-level
    pinocchio import (consistent with the project's lazy-SDK-import pattern).
    """
    import pinocchio as pin

    return pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
