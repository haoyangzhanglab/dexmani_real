"""VR-to-XHand retargeting via dex_retargeting + adaptive finger scaling.

Two landmark-space adaptations compensate for human-robot kinematic mismatches
before reference vectors are computed:

1. adaptive_retargeting_thumb — scales thumb_tip to match XHand mechanical thumb
   length (~23% longer than MANO model).  Adaptive: more when extended, less
   when curled — drives rot2 (IP flexion) toward zero in neutral poses while
   preserving curl range for opposition/grip gestures.

2. adaptive_retargeting_xhand (LeFranX) — scales pinky chain (MCP→PIP→DIP→TIP)
   based on extension state, compensating for human-robot finger length
   differences.  Uniform scaling along the kinematic chain.

Both operate on MANO-space landmarks and modify disjoint landmark indices
(thumb: 4, pinky: 18-20), so order is irrelevant.

Refs:
  LeFranX vr_hand_detector_adapter.py:27-84 (pinky)
  CL-20260701 thumb FK analysis: robot wrist→thumb_tip 0.161m vs MANO 0.131m
"""

from __future__ import annotations

__all__ = ["XHandRetargeter", "adaptive_retargeting_thumb", "adaptive_retargeting_xhand"]

import time

import numpy as np
from dex_retargeting.retargeting_config import RetargetingConfig
from dexmani_real import ASSET_DIR
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# ── Pinky landmark indices (MediaPipe convention) ──
_PINKY_MCP = 17
_PINKY_PIP = 18
_PINKY_DIP = 19
_PINKY_TIP = 20

# Pinky adaptive scaling — calibrated to actual MANO-space pinky
# MCP→TIP distances observed across 5 teleop recordings (500+ frames).
# LeFranX original values (_MIN=0.03, _MAX=0.10) were tuned for a
# different MANO space where the pinky reaches 0.10 m at full extension.
# Our MANO space maxes out at ~0.073 m, so we calibrate the range to match.
#
# Parameters selected via offline grid search on real recording data
# (20260701_161732, 3431 frames): sweep of max_scale ∈ [2.0, 2.2, 2.4, 2.6]
# with calibrated _MIN/_MAX.  Chose 2.4 as the best balance between
# straightening the extended pinky and preserving curl range.

_PINKY_MIN_EXTENSION = 0.030  # P5 of MANO pinky MCP→TIP (fully curled)
_PINKY_MAX_EXTENSION = 0.073  # P95 of MANO pinky MCP→TIP (fully extended)
_PINKY_BASE_SCALE = 1.2  # minimum scaling for curled positions
_PINKY_MAX_SCALE = 2.4  # maximum scaling for extended positions (calibrated)


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


# ── Thumb adaptive scaling parameters ──
# XHand thumb mechanical length (wrist→thumb_tip) is ~0.161 m at neutral
# (bend=0, rot1=0, rot2=0).  The MANO model averages ~0.131 m — a ~23 %
# mismatch.  Without compensation the DexPilot optimizer uses rot2 (thumb IP
# flexion) purely to shorten the kinematic chain, keeping it at ~1.1–1.3 rad
# (63–75°) regardless of the human thumb state.
#
# Adaptive scaling: when the human thumb is extended (long wrist→tip),
# scale the target up toward the robotʼs mechanical length so rot2 can drop
# toward zero.  When the human thumb is curled, scale conservatively to
# preserve the optimizerʼs ability to increase rot2 for opposition / grip.

_THUMB_MIN_EXTENSION = 0.117  # P5  of human wrist→thumb_tip (curled)
_THUMB_MAX_EXTENSION = 0.139  # P95 of human wrist→thumb_tip (extended)
_THUMB_BASE_SCALE = 1.05  # minimum scaling for curled thumb
_THUMB_MAX_SCALE = 1.25  # maximum scaling for extended thumb


def adaptive_retargeting_thumb(landmarks: np.ndarray) -> np.ndarray:
    """Scale thumb tip to match XHand mechanical thumb length.

    The DexPilot reference vectors only use thumb_tip (landmark 4) — not
    the intermediate thumb joints (CMC=1, MCP=2, IP=3).  Scaling just the
    tip is sufficient to shift the wrist→thumb_tip distance into the
    robotʼs achievable range at low rot2.

    Operates on MANO-space landmarks.  The pinky scaling
    (adaptive_retargeting_xhand) operates on landmarks 18–20, so the two
    are independent and can be composed in any order.

    Args:
        landmarks: (21, 3) array in MANO coordinate space.

    Returns:
        (21, 3) array with thumb_tip scaled (new copy, input unchanged).
    """
    landmarks = landmarks.copy()

    wrist = landmarks[0]
    thumb_tip = landmarks[4]

    extension = float(np.linalg.norm(thumb_tip - wrist))
    extension_ratio = np.clip(
        (extension - _THUMB_MIN_EXTENSION) / (_THUMB_MAX_EXTENSION - _THUMB_MIN_EXTENSION),
        0.0,
        1.0,
    )
    scale = _THUMB_BASE_SCALE + (_THUMB_MAX_SCALE - _THUMB_BASE_SCALE) * extension_ratio
    landmarks[4] = wrist + (thumb_tip - wrist) * scale

    return landmarks


class XHandRetargeter:
    def __init__(
        self,
        fixed_joint_values: np.ndarray | None = None,
        hand_type: str = "right",
        retargeting_type: str = "dexpilot",
        debug_adapters: bool = False,
    ):
        self.hand_type = hand_type
        self.retargeting_type = retargeting_type
        self.fixed_joint_values = np.array([]) if fixed_joint_values is None else np.array(fixed_joint_values)
        self.debug_adapters = bool(debug_adapters)
        self.last_debug = {}

        self.sapien_joint_names = [
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

        RetargetingConfig.set_default_urdf_dir(str(ASSET_DIR / "robots"))
        self.retargeter = RetargetingConfig.load_from_file(str(config_path)).build()
        self.indices = self.retargeter.optimizer.target_link_human_indices

        retargeter_joint_names = self.retargeter.optimizer.robot.dof_joint_names
        self.retargeted_joint_order = np.array(
            [retargeter_joint_names.index(name) for name in self.sapien_joint_names]
        ).astype(int)

    @property
    def low_pass_alpha(self) -> float:
        """Current LPFilter alpha (0.0 = disabled, 1.0 = no smoothing)."""
        return float(self.retargeter.filter.alpha)

    @low_pass_alpha.setter
    def low_pass_alpha(self, value: float) -> None:
        """Tune the built-in EMA smoothing strength at runtime.

        dex_retargeting applies LPFilter after NLP optimisation, before returning
        qpos.  Default from YAML config: 0.6 (new-value weight, i.e. 60 % new +
        40 % old per frame at 50 Hz).

        0.0 = pass-through (raw optimizer output)
        0.6 = default (moderate smoothing)
        1.0 = no smoothing (always use latest — equivalent to 0.0)
        """
        self.retargeter.filter.alpha = float(value)

    def _build_ref_value(self, hand_joint_pos: np.ndarray) -> np.ndarray:
        """Build reference value from hand landmarks for retargeting.

        Applies two landmark-space adaptations before computing origin→task
        difference vectors:

        1. adaptive_retargeting_thumb — scale thumb_tip to match robot
           mechanical length, enabling low rot2 in neutral poses.
        2. adaptive_retargeting_xhand (LeFranX) — scale pinky chain for
           human-robot finger length mismatch.
        """
        if self.retargeting_type == "position":
            return hand_joint_pos[self.indices, :]

        # Scale thumb tip to compensate for XHand's ~23% longer mechanical
        # thumb — drives rot2 toward zero when the human thumb is extended
        # while preserving curl range for opposition / grip.
        scaled_landmarks = adaptive_retargeting_thumb(hand_joint_pos)
        # LeFranX: scale pinky chain on landmarks before computing ref vectors.
        # This directly modifies PIP/DIP/TIP positions along the MCP→TIP chain,
        # compensating for human-robot finger length differences.
        scaled_landmarks = adaptive_retargeting_xhand(scaled_landmarks)

        origin_indices = self.indices[0, :]
        task_indices = self.indices[1, :]

        ref_value = scaled_landmarks[task_indices, :] - scaled_landmarks[origin_indices, :]
        return ref_value

    def retarget(self, hand_joint_pos: np.ndarray | None) -> np.ndarray | None:
        if hand_joint_pos is None:
            return None

        start_time = time.time()

        ref_value = self._build_ref_value(hand_joint_pos)
        qpos = self.retargeter.retarget(ref_value, fixed_qpos=self.fixed_joint_values)

        if qpos is None:
            logger.warning("Retargeting returned None.")
            return None

        qpos = np.asarray(qpos, dtype=float)[self.retargeted_joint_order]

        if self.debug_adapters:
            self.last_debug = {
                "retarget_ms": 1000 * (time.time() - start_time),
                "adaptives": "thumb(scale_thumb_tip) + pinky(LeFranX chain scaling)",
            }
            logger.info("retarget_debug: %s", self.last_debug)

        return qpos
