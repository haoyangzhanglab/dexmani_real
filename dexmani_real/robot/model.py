"""机器人模型、SDK 关节顺序和额定机械限位。"""

from __future__ import annotations

from dexmani_real import ASSET_DIR

XARM7_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))

ARM_DOF = 7
HAND_DOF = 12
HAND_FINGER_COUNT = 5
XHAND_TACTILE_SENSOR_FINGER_IDS: tuple[int, ...] = (2, 5, 7, 9, 11)
XHAND_TACTILE_SENSOR_INDEX_BY_FINGER_ID = {
    finger_id: index for index, finger_id in enumerate(XHAND_TACTILE_SENSOR_FINGER_IDS)
}
XHAND_FINGERTIP_LINK_NAMES: tuple[str, ...] = (
    "right_hand_thumb_rota_tip",
    "right_hand_index_rota_tip",
    "right_hand_mid_tip",
    "right_hand_ring_tip",
    "right_hand_pinky_tip",
)
TACTILE_POINTS_PER_FINGER = 120
TACTILE_AXIS_COUNT = 3

ARM_JOINT_SHAPE = (ARM_DOF,)
HAND_JOINT_SHAPE = (HAND_DOF,)
HAND_TACTILE_SUM_SHAPE = (HAND_FINGER_COUNT, TACTILE_AXIS_COUNT)
HAND_TACTILE_FORCE_SHAPE = (
    HAND_FINGER_COUNT,
    TACTILE_POINTS_PER_FINGER,
    TACTILE_AXIS_COUNT,
)
HAND_FINGERTIP_SHAPE = (HAND_FINGER_COUNT, 3)


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

XHAND_MODEL_DIR = ASSET_DIR / "robots" / "xhand"
# 七关节规划模型中的手固定在 open/home 姿态。
XARM7_XHAND_COLLISION_URDF_PATH = XHAND_MODEL_DIR / "xarm7_xhand_collision.urdf"
XARM7_XHAND_RIGHT_URDF_PATH = XHAND_MODEL_DIR / "xarm7_xhand_right.urdf"
XARM7_XHAND_SRDF_PATH = XHAND_MODEL_DIR / "xarm7_xhand.srdf"
XHAND_RIGHT_URDF_PATH = XHAND_MODEL_DIR / "xhand_right.urdf"


XHAND_URDF_JOINT_NAMES: tuple[str, ...] = (
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
    XHAND_SDK_JOINT_NAMES.index(name) for name in XHAND_URDF_JOINT_NAMES
)


XARM7_HARD_LOWER = (
    -6.28318530718,
    -2.059,
    -6.28318530718,
    -0.19198,
    -6.28318530718,
    -1.69297,
    -6.28318530718,
)
XARM7_HARD_UPPER = (
    6.28318530718,
    2.0944,
    6.28318530718,
    3.927,
    6.28318530718,
    3.14159265359,
    6.28318530718,
)

# 只有这些轴允许 2π 等价角，运行限位仍可能窄于一整圈。
XARM7_EQUIVALENT_JOINT_MASK = (True, False, True, False, True, False, True)


XHAND_RATED_QPOS_MIN_RAD: tuple[float, ...] = (
    0.0,
    -0.698,
    0.0,
    -0.174,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


XHAND_RATED_QPOS_MAX_RAD: tuple[float, ...] = (
    1.832,
    1.745,
    1.745,
    0.174,
    1.919,
    1.919,
    1.919,
    1.919,
    1.919,
    1.919,
    1.919,
    1.919,
)
