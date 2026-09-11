"""Canonical robot-model resources and joint mappings used across the repository.

The resource paths are shared by planning, replay, deployment, and recording so
all components use the same static models.
"""

from __future__ import annotations

from dexmani_real import ASSET_DIR

ARM_DOF = 7
HAND_DOF = 12
HAND_FINGER_COUNT = 5
HAND_FINGER_NAMES: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
HAND_FINGER_ORDER_ID = "thumb_index_mid_ring_pinky"
XHAND_TACTILE_SENSOR_FINGER_IDS: tuple[int, ...] = (2, 5, 7, 9, 11)
XHAND_TACTILE_SENSOR_INDEX_BY_FINGER_ID = {
    finger_id: index
    for index, finger_id in enumerate(XHAND_TACTILE_SENSOR_FINGER_IDS)
}
XHAND_FINGERTIP_LINK_NAMES: tuple[str, ...] = (
    "right_hand_thumb_rota_tip",
    "right_hand_index_rota_tip",
    "right_hand_mid_tip",
    "right_hand_ring_tip",
    "right_hand_pinky_tip",
)
if (
    len(HAND_FINGER_NAMES) != HAND_FINGER_COUNT
    or HAND_FINGER_COUNT != 5
    or len(XHAND_TACTILE_SENSOR_FINGER_IDS) != HAND_FINGER_COUNT
    or len(set(XHAND_TACTILE_SENSOR_FINGER_IDS)) != HAND_FINGER_COUNT
    or set(XHAND_TACTILE_SENSOR_INDEX_BY_FINGER_ID.values())
    != set(range(HAND_FINGER_COUNT))
    or len(XHAND_FINGERTIP_LINK_NAMES) != HAND_FINGER_COUNT
):
    raise RuntimeError("XHand finger and tactile sensor orders must match five fingers")
TACTILE_POINTS_PER_FINGER = 120
TACTILE_AXIS_COUNT = 3

# XHand SDK force-representation vocabulary shared by recording, dataset
# artifacts, and deployment semantics: contact_force is the aggregate calc_force
# per finger; tactile_force is the dense raw_force payload.  Both stay in the
# SDK-native numeric scale (SI Newton unverified) and per-finger sensor axes
# (taxel spatial geometry unverified); validity comes from provenance flags,
# never from payload magnitude.
CONTACT_FORCE_REPRESENTATION = "xhand_sdk_calc_force_fx_fy_fz_bias_corrected"
TACTILE_FORCE_REPRESENTATION = "xhand_sdk_raw_force_fx_fy_fz_bias_corrected"
TACTILE_FORCE_SENSOR_ORDER = "xhand_sdk_sensor_data_order"
TACTILE_FORCE_POINT_ORDER = "xhand_sdk_sensor_data_raw_force_order"
TACTILE_FORCE_AXIS_LABELS = "fx_fy_fz"
XHAND_SDK_NATIVE_UNKNOWN_SI_UNIT = "xhand_sdk_native_unknown_si"
XHAND_SENSOR_NATIVE_AXES_FRAME = "xhand_sensor_native_axes_per_finger"

ARM_JOINT_SHAPE = (ARM_DOF,)
HAND_JOINT_SHAPE = (HAND_DOF,)
ARM_EE_SHAPE = (9,)
HAND_TACTILE_SUM_SHAPE = (HAND_FINGER_COUNT, TACTILE_AXIS_COUNT)
HAND_TACTILE_FORCE_SHAPE = (
    HAND_FINGER_COUNT,
    TACTILE_POINTS_PER_FINGER,
    TACTILE_AXIS_COUNT,
)
HAND_FINGERTIP_SHAPE = (HAND_FINGER_COUNT, 3)

# Canonical order for every cross-process XHand joint vector. Retargeting,
# planning, recording, and the device boundary all consume this SDK order.
XHAND_SDK_JOINT_NAMES: tuple[str, ...] = (
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
)
if (
    len(XHAND_SDK_JOINT_NAMES) != HAND_DOF
    or len(set(XHAND_SDK_JOINT_NAMES)) != HAND_DOF
):
    raise RuntimeError("XHand SDK joint names must be unique and match HAND_DOF")

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
