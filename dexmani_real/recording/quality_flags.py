"""10-bit quality flags for per-frame data validation.

Bit 6 (formerly CAMERA_OK) was removed — it was defined but never set,
and no camera health check existed in the control loop. Re-add when
a camera frame timestamp/freshness check is implemented.
"""

from __future__ import annotations

__all__ = [
    "QualityFlags",
    "ALL_GOOD_MASK",
    "TRACKING_OK",
    "IK_SUCCESS",
    "RETARGET_OK",
    "RETARGET_VALID",
    "JOINT_JUMP_OK",
    "IN_WORKSPACE",
    "ARM_TORQUE_OK",
    "HAND_CURRENT_OK",
    "HAND_TEMP_OK",
    "HAND_COMM_OK",
]

TRACKING_OK = 1 << 0       # bit 0:  VR tracking valid
IK_SUCCESS = 1 << 1        # bit 1:  IK solve success
RETARGET_OK = 1 << 2       # bit 2:  hand retargeting success
RETARGET_VALID = 1 << 3    # bit 3:  retargeting result within physiological range
JOINT_JUMP_OK = 1 << 4     # bit 4:  joint jump within limits
IN_WORKSPACE = 1 << 5      # bit 5:  EEF within workspace
ARM_TORQUE_OK = 1 << 7     # bit 7:  arm torque within normal range
HAND_CURRENT_OK = 1 << 8   # bit 8:  hand current within normal range
HAND_TEMP_OK = 1 << 9      # bit 9:  hand temperature normal
HAND_COMM_OK = 1 << 10     # bit 10: hand communication normal (no board error)

ALL_GOOD_MASK = TRACKING_OK | IK_SUCCESS | RETARGET_OK | RETARGET_VALID | \
    JOINT_JUMP_OK | IN_WORKSPACE | ARM_TORQUE_OK | HAND_CURRENT_OK | \
    HAND_TEMP_OK | HAND_COMM_OK


class QualityFlags:
    """Builder for per-frame quality flags (10 active bits, uint16)."""

    def __init__(self) -> None:
        self.flags: int = 0

    def set(self, bit: int, ok: bool) -> QualityFlags:
        if ok:
            self.flags |= bit
        return self

    def get(self) -> int:
        return self.flags

    def reset(self) -> None:
        self.flags = 0

    @staticmethod
    def is_all_good(flags: int) -> bool:
        return (flags & ALL_GOOD_MASK) == ALL_GOOD_MASK

    @staticmethod
    def describe(flags: int) -> list[str]:
        names = [
            ("TRACKING_OK", TRACKING_OK),
            ("IK_SUCCESS", IK_SUCCESS),
            ("RETARGET_OK", RETARGET_OK),
            ("RETARGET_VALID", RETARGET_VALID),
            ("JOINT_JUMP_OK", JOINT_JUMP_OK),
            ("IN_WORKSPACE", IN_WORKSPACE),
            ("ARM_TORQUE_OK", ARM_TORQUE_OK),
            ("HAND_CURRENT_OK", HAND_CURRENT_OK),
            ("HAND_TEMP_OK", HAND_TEMP_OK),
            ("HAND_COMM_OK", HAND_COMM_OK),
        ]
        failed = [name for name, bit in names if not (flags & bit)]
        return failed
