"""机器人模型、SDK 关节顺序和设备有效限位。"""

from __future__ import annotations

from math import pi

from dexmani_real import ASSET_DIR

XARM7_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))

ARM_DOF = 7
HAND_DOF = 12
HAND_FINGER_COUNT = 5
XHAND_TACTILE_SCALE_DIVISOR = 255.0
XHAND_TACTILE_SENSOR_FINGER_IDS: tuple[int, ...] = (2, 5, 7, 9, 11)
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
XARM7_XHAND_RIGHT_URDF_PATH = XHAND_MODEL_DIR / "xarm7_xhand_right.urdf"
XARM7_XHAND_SRDF_PATH = XHAND_MODEL_DIR / "xarm7_xhand.srdf"


# 本项目 XS1305 的有效控制边界（rad），非所有 xArm7 的通用机械范围。
# 来源：xArm-Python-SDK v1.18.4，xarm/x3/xarm.py 的型号选择及
# xarm/core/config/x_config.py 的 JOINT_LIMITS[7][71305]；不含 SDK 判断容差。
XARM7_HARD_LOWER = (
    -2 * pi, -2.042035, -2 * pi, -0.10472,
    -2 * pi, -1.692969, -2 * pi,
)
XARM7_HARD_UPPER = (
    2 * pi, 2.024581, 2 * pi, 3.92699,
    2 * pi, pi, 2 * pi,
)

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
