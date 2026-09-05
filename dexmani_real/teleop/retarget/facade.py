"""VR-to-XHand retargeting: TAG (in-repo NLopt) and DexPilot (dex_retargeting).

One landmark-space adaptation compensates the human-robot kinematic mismatch:
``adaptive_retargeting_xhand`` scales the pinky chain (MCP→PIP→DIP→TIP) by a
constant per-backend ``pinky_scale`` (plus an optional palm-baseline offset).
"""

from __future__ import annotations

__all__ = [
    "XHandRetargeter",
    "TAGHandRetargeter",
    "adaptive_retargeting_xhand",
    "validate_landmarks",
]

import time
from typing import Any

import nlopt
import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.robot_spec import HAND_JOINT_SHAPE, XHAND_SDK_JOINT_NAMES
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


_PINKY_MCP = 17
_PINKY_PIP = 18
_PINKY_DIP = 19
_PINKY_TIP = 20

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

# Human flexion feature index → SDK joint index; unmapped joints are excluded.
_HUMAN_FLEXION_JOINT_PAIRS = (
    (0, 0),
    (1, 2),  # thumb CMC→bend, MCP+IP→rota2
    (2, 4),
    (3, 5),  # index MCP→j1, PIP+DIP→j2
    (4, 6),
    (5, 7),  # middle
    (6, 8),
    (7, 9),  # ring
    (8, 10),
    (9, 11),  # pinky
)
_SDK_FLEXION_MASK = np.zeros(12, dtype=np.float64)
for _feature_index, _sdk_index in _HUMAN_FLEXION_JOINT_PAIRS:
    _SDK_FLEXION_MASK[_sdk_index] = 1.0


def _human_flexion_rad(landmarks: np.ndarray) -> np.ndarray:
    """Human flexion reference (10,) from one (21, 3) landmark frame.

    Rotation-invariant per-finger angles: thumb CMC + MCP+IP, then per finger
    (index/mid/ring/pinky) MCP + PIP+DIP — the same feature order as
    ``_HUMAN_FLEXION_JOINT_PAIRS``.
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
            ang[joint_index] = np.arccos(
                np.clip(np.sum(first * second) / denominator, -1.0, 1.0)
            )
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


# Right-hand operator→MANO rotation; Unity→FLU conversion occurs in the VR receiver.
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
    palm_sine = float(
        np.linalg.norm(np.cross(index_basis, pinky_basis))
        / (index_length * pinky_length)
    )
    if palm_sine < 0.1:
        return False, "palm basis is collinear"
    shortest_bone = min(
        float(np.linalg.norm(points[child] - points[parent]))
        for parent, child in _CONTIGUOUS_BONES
    )
    if shortest_bone < 0.002:
        return False, "a retargeting bone is shorter than 2 mm"
    return True, ""


def _estimate_palm_frame(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """Estimate a palm coordinate frame (3×3 rotation matrix) from 21 hand landmarks.

    Uses wrist + index MCP + middle MCP to fit a palm plane via SVD, then
    constructs a right-handed orthonormal frame with x pointing from middle
    MCP to wrist. The caller must first run ``validate_landmarks()``.

    """
    keypoint_3d_array = np.asarray(keypoint_3d_array, dtype=np.float64)
    if keypoint_3d_array.shape != (21, 3):
        raise ValueError(
            f"keypoint_3d_array must have shape (21, 3), got {keypoint_3d_array.shape}"
        )

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

    # Keep the lateral axis sign consistent with index→middle.
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1.0
        z *= -1.0

    return np.stack([x, normal, z], axis=1)


def adaptive_retargeting_xhand(
    landmarks: np.ndarray,
    *,
    scale: float,
    palm_scale: float,
) -> np.ndarray:
    """Scale the pinky chain and optional wrist-to-MCP baseline on a copy.

    Args:
        landmarks: (21, 3) array in MANO coordinate space.
        scale: Uniform MCP→TIP segment scale.
        palm_scale: Wrist→MCP baseline scale; 1.0 is a no-op.

    Returns:
        (21, 3) array with pinky chain scaled (new copy, input unchanged).
    """
    landmarks = np.asarray(landmarks, dtype=np.float64).copy()
    if landmarks.shape != (21, 3) or not np.all(np.isfinite(landmarks)):
        raise ValueError("landmarks must be a finite (21, 3) array")
    raw = landmarks.copy()

    # Scale the wrist→MCP baseline independently.
    if palm_scale != 1.0:
        landmarks[_PINKY_MCP] = (
            landmarks[0] + (raw[_PINKY_MCP] - landmarks[0]) * palm_scale
        )

    # Apply uniform scaling along MCP→PIP→DIP→TIP.
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
        dexpilot_config: Any,
        fixed_joint_values: np.ndarray | None = None,
        hand_type: str = "right",
        retargeting_type: str = "dexpilot",
        debug_adapters: bool = False,
    ):
        self.hand_type = hand_type
        self.retargeting_type = retargeting_type
        self.fixed_joint_values = (
            np.array([]) if fixed_joint_values is None else np.array(fixed_joint_values)
        )
        self.debug_adapters = bool(debug_adapters)
        self._dexpilot_config = dexpilot_config
        self._pinky_scale = float(dexpilot_config.pinky_scale)
        self._pinky_palm_scale = float(dexpilot_config.pinky_palm_scale)
        self.last_debug: dict[str, float | str] = {}

        # Keep the public output in the schema-owned SDK order.
        self.sdk_joint_names = XHAND_SDK_JOINT_NAMES

        self.load_retargeter()

    def load_retargeter(self):
        import yaml
        from dex_retargeting.retargeting_config import RetargetingConfig

        from dexmani_real.teleop.retarget.dexpilot import build_dexpilot_retargeting

        config_path = (
            ASSET_DIR
            / "retargeting"
            / f"xhand_{self.hand_type}_{self.retargeting_type}.yml"
        )

        with open(str(config_path), "r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        cfg = yaml_config["retargeting"]

        # Require the YAML list without allowing it to redefine schema qpos order.
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

        self._prior_weight = (
            float(self._dexpilot_config.prior_weight)
            if self._dexpilot_config is not None
            else 0.0
        )
        prior_mask = _SDK_FLEXION_MASK  # target order == SDK order

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
        """Current LPFilter alpha (1.0 passes through; 0.0 freezes)."""
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
        scaled_landmarks = adaptive_retargeting_xhand(
            hand_joint_pos,
            scale=self._pinky_scale,
            palm_scale=self._pinky_palm_scale,
        )

        origin_indices = self.indices[0, :]
        task_indices = self.indices[1, :]

        ref_value = (
            scaled_landmarks[task_indices, :] - scaled_landmarks[origin_indices, :]
        )
        return ref_value

    def retarget(self, landmarks: np.ndarray | None) -> np.ndarray | None:
        """Retarget raw VR landmarks (operator-frame, 21x3) to XHand joint qpos.

        Handles coordinate transform (operator → MANO), input validation,
        and NLP optimization. Expected invalid input or solver roundoff returns
        ``None``; unexpected optimizer errors propagate to the session owner.
        """
        if landmarks is None:
            return None
        valid, reason = validate_landmarks(landmarks)
        if not valid:
            logger.warning("VR landmarks rejected (%s) — holding hand position", reason)
            return None

        try:
            wrist_rot = _estimate_palm_frame(landmarks)
            mano_landmarks = landmarks @ wrist_rot @ _OPERATOR2MANO_RIGHT
        except (ValueError, TypeError, np.linalg.LinAlgError):
            logger.warning("Coordinate transform failed — holding hand position")
            return None

        start_time = time.time()

        # Use the unscaled landmarks for the human-flexion prior.
        if self._prior_weight > 0:
            self.retargeter.optimizer.set_prior_reference(
                _human_flexion_sdk_reference(mano_landmarks)
            )

        ref_value = self._build_ref_value(mano_landmarks)
        try:
            qpos = self.retargeter.retarget(
                ref_value, fixed_qpos=self.fixed_joint_values
            )
        except nlopt.RoundoffLimited:
            logger.warning("Retargeting roundoff limit reached — holding hand position")
            return None

        if qpos is None:
            logger.warning("Retargeting returned None.")
            return None

        qpos_arr = np.asarray(qpos, dtype=float)

        qpos_arr = qpos_arr[self.retargeted_joint_order]

        if self.debug_adapters:
            self.last_debug = {
                "retarget_ms": 1000 * (time.time() - start_time),
                "adaptives": "pinky_chain_scaling",
            }
            logger.info("retarget_debug: %s", self.last_debug)

        return qpos_arr

    def reset(self, initial_qpos: np.ndarray | None = None) -> None:
        """Reset optimizer, filter, projection state, and optional warm start."""
        self.retargeter.reset()
        if self.retargeter.filter is not None:
            self.retargeter.filter.reset()
        self.retargeter.optimizer.projected[:] = False

        # Seed the optimizer with the current hardware pose when available.
        if initial_qpos is not None and initial_qpos.shape == HAND_JOINT_SHAPE:
            qpos = np.asarray(initial_qpos, dtype=np.float32)
            if np.all(np.isfinite(qpos)):
                # Convert SDK order to the optimizer's internal order.
                qpos_retargeter = qpos[self.inverse_retargeted_joint_order]
                idx = self.retargeter.optimizer.idx_pin2target
                self.retargeter.last_qpos = qpos_retargeter[idx]
            else:
                logger.warning(
                    "initial_qpos contains NaN/Inf — falling back to neutral seed"
                )


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
        fingertip_link_names: tuple[str, ...],
        tag_config: Any,
        urdf_path: str,
        hand_type: str = "right",
        debug: bool = False,
    ) -> None:
        from scipy.spatial.transform import Rotation

        from dexmani_real.teleop.retarget.pin_grad import validate_fingertip_frame_names
        from dexmani_real.teleop.retarget.tag_optimizer import HandOptimizer

        resolved_urdf_path = str(urdf_path)
        resolved_tip_names = validate_fingertip_frame_names(fingertip_link_names)
        model = pin_loading(resolved_urdf_path)
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
            finger_lengths_robot=np.array(
                tag_config.robot_finger_lengths, dtype=np.float64
            ),
            finger_lengths_human=np.array(
                tag_config.human_finger_lengths, dtype=np.float64
            ),
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
            prior_weight=tag_config.prior_weight,
            prior_mask=_SDK_FLEXION_MASK[self._mapping_sdk_to_model],
        )

        self._R_mano_to_urdf: np.ndarray = Rotation.from_euler(
            "xyz", tag_config.mano_to_urdf_euler
        ).as_matrix()
        self._pinky_scale = float(tag_config.pinky_scale)
        self._pinky_palm_scale = float(tag_config.pinky_palm_scale)
        self._prior_weight = float(tag_config.prior_weight)

        self._last_raw_qpos: np.ndarray | None = None
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
                "TAGHandRetargeter: landmarks rejected (%s) — holding position", reason
            )
            return None

        t0 = time.perf_counter() if self.debug else 0.0

        try:
            wrist_rot = _estimate_palm_frame(landmarks)
            mano = landmarks @ wrist_rot @ _OPERATOR2MANO_RIGHT
        except (ValueError, np.linalg.LinAlgError):
            logger.warning(
                "TAGHandRetargeter: coordinate transform failed — holding position"
            )
            return None

        # Use the unscaled landmarks for the human-flexion prior.
        q_prior_model = None
        if self._prior_weight > 0:
            q_prior_model = _human_flexion_sdk_reference(mano)[
                self._mapping_sdk_to_model
            ]

        mano = adaptive_retargeting_xhand(
            mano,
            scale=self._pinky_scale,
            palm_scale=self._pinky_palm_scale,
        )

        tips = mano[_FINGERTIP_INDICES].copy()  # (5, 3) in MANO frame
        tips -= mano[0]  # center at wrist
        tips_urdf = tips @ self._R_mano_to_urdf.T  # (5, 3) in URDF frame

        qpos_model = self._optimizer.solve(
            tips_urdf, q_prior=q_prior_model
        )  # (12,) in Pinocchio model order

        if qpos_model is None:
            return None

        qpos_sdk = qpos_model[self._mapping_model_to_sdk]
        self._last_raw_qpos = qpos_sdk.copy()

        if self.debug:
            dt_ms = 1000.0 * (time.perf_counter() - t0)
            logger.info("TAGHandRetargeter: retarget %.2f ms", dt_ms)

        return qpos_sdk

    def reset(self, initial_qpos: np.ndarray | None = None) -> None:
        """Reset optimizer state and optional SDK-order warm start."""
        self._last_raw_qpos = None

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

    @property
    def last_raw_qpos(self) -> np.ndarray | None:
        """Latest SDK-order retarget output."""
        return self._last_raw_qpos.copy() if self._last_raw_qpos is not None else None


def pin_loading(urdf_path: str):
    """Load a URDF with Pinocchio FreeFlyer model.

    Extracted so TAGHandRetargeter.__init__ doesn't need a module-level
    pinocchio import (consistent with the project's lazy-SDK-import pattern).
    """
    import pinocchio as pin

    return pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
