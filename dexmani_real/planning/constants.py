"""Canonical robot-model resources and joint mappings used by planning.

The resource paths are also consumed by lifecycle and provenance code. Keeping
them here ensures that planning, replay, deployment, and recording hash the
same static models.
"""

from __future__ import annotations

from dexmani_real import ASSET_DIR
from dexmani_real.utils.schema import XHAND_SDK_JOINT_NAMES

XHAND_MODEL_DIR = ASSET_DIR / "robots" / "xhand"
# Arm planning model with the hand geometry fixed in its open/home posture.
XARM7_XHAND_COLLISION_URDF_PATH = XHAND_MODEL_DIR / "xarm7_xhand_collision.urdf"
# Full 19-DOF model: seven arm joints followed by twelve right-hand joints.
XARM7_XHAND_RIGHT_URDF_PATH = XHAND_MODEL_DIR / "xarm7_xhand_right.urdf"
XARM7_XHAND_SRDF_PATH = XHAND_MODEL_DIR / "xarm7_xhand.srdf"
# Standalone 12-DOF right-hand model used by hand retargeting/kinematics.
XHAND_RIGHT_URDF_PATH = XHAND_MODEL_DIR / "xhand_right.urdf"

# Hand joint order remap: XHand SDK → URDF / Pinocchio.
_HAND_URDF_JOINT_NAMES: tuple[str, ...] = (
    "right_hand_index_bend_joint",
    "right_hand_index_joint1",
    "right_hand_index_joint2",
    "right_hand_mid_joint1",
    "right_hand_mid_joint2",
    "right_hand_pinky_joint1",
    "right_hand_pinky_joint2",
    "right_hand_ring_joint1",
    "right_hand_ring_joint2",
    "right_hand_thumb_bend_joint",
    "right_hand_thumb_rota_joint1",
    "right_hand_thumb_rota_joint2",
)
HAND_SDK_TO_URDF_IDX: tuple[int, ...] = tuple(
    XHAND_SDK_JOINT_NAMES.index(name) for name in _HAND_URDF_JOINT_NAMES
)
