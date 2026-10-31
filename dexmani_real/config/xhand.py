"""XHand 通信、关节范围、增益与回零参数。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, radians

import numpy as np

from dexmani_real.robot.commands import validate_hand_target
from dexmani_real.robot.model import (
    HAND_DOF,
    XHAND_RATED_QPOS_MAX_RAD,
    XHAND_RATED_QPOS_MIN_RAD,
)

SERIAL_SETTLE_S = 1.0
INITIAL_STATE_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class HandConfig:
    enabled: bool = True

    # 设备连接；device_name=None 时由 SDK 枚举设备。
    comm_type: str = "serial"
    device_name: str | None = "/dev/ttyUSB0"
    baudrate: int = 3_000_000
    device_id: int = 0
    ethercat_slave_position: int = -1

    # SDK 顺序：拇指 3 轴、食指 3 轴，其余每指 2 轴。
    home_qpos_deg: tuple[float, ...] = (
        30.0, 55.33, 10.0,  # 拇指
        0.17, 1.08, 5.0,    # 食指
        1.25, 5.0,          # 中指
        1.33, 5.0,          # 无名指
        1.33, 5.0,          # 小指
    )

    qpos_min_rad: tuple[float, ...] = (
        0.0, -0.698, radians(10.0),  # 拇指
        -0.174, 0.0, radians(5.0),   # 食指
        0.0, radians(5.0),           # 中指
        0.0, radians(5.0),           # 无名指
        0.0, radians(5.0),           # 小指
    )
    qpos_max_rad: tuple[float, ...] = XHAND_RATED_QPOS_MAX_RAD

    kp: tuple[int, ...] = (
        100, 100, 100,  # 拇指
        120, 100, 100,  # 食指
        100, 100,       # 中指
        100, 100,       # 无名指
        100, 100,       # 小指
    )
    ki: int = 0
    kd: int = 0

    # Mode 3 输出范围为 0–400，400 表示 100%，单位不是 mA。
    tor_max: tuple[int, ...] = (
        360, 300, 300,  # 拇指
        360, 300, 300,  # 食指
        300, 300,       # 中指
        300, 300,       # 无名指
        300, 300,       # 小指
    )

    feedback_max_age_s: float = 4 / 30
    # 未经硬件定标的工程起点；达到该连续发送 CRC 次数即使普通位置发送失败。
    send_crc_limit: int = 5

    home_timeout_s: float = 2.0
    home_tolerance_deg: float = 5.0

    def validate(self) -> None:
        if self.comm_type not in ("ethercat", "serial"):
            raise ValueError("hand comm_type must be 'ethercat' or 'serial'")
        if self.device_name is not None:
            if not isinstance(self.device_name, str) or not self.device_name.strip():
                raise ValueError("hand device_name must be a non-empty string or null")
            if self.comm_type == "ethercat" and "/" in self.device_name:
                raise ValueError(
                    "hand device_name for EtherCAT must be an interface name, not a device path; "
                    "set hand.device_name to the interface name or null for discovery"
                )
        communication_field = (
            ("baudrate", 1) if self.comm_type == "serial" else ("ethercat_slave_position", -1)
        )
        for name, minimum in (
            communication_field, ("device_id", 0), ("send_crc_limit", 1), ("ki", 0), ("kd", 0),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"hand {name} must be an integer >= {minimum}")
        self.validate_home_pose()
        if len(self.kp) != HAND_DOF or any(
            type(value) is not int or value <= 0 for value in self.kp
        ):
            raise ValueError("hand kp must contain twelve positive integer gains")
        if len(self.tor_max) != HAND_DOF or any(
            type(value) is not int or not 0 <= value <= 400 for value in self.tor_max
        ):
            raise ValueError(
                "hand tor_max must contain twelve integers in [0, 400] (Mode 3 output scale)"
            )
        for name in (
            "feedback_max_age_s",
            "home_timeout_s",
            "home_tolerance_deg",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"hand {name} must be finite and positive")

    def validate_home_pose(self) -> None:
        """手部禁用时仍须验证规划使用的固定姿态。"""
        bounds = np.asarray((self.qpos_min_rad, self.qpos_max_rad), dtype=np.float64)
        if bounds.shape != (2, HAND_DOF) or not np.isfinite(bounds).all():
            raise ValueError("hand command limits must each contain 12 finite values")
        lower, upper = bounds
        if np.any(lower >= upper):
            raise ValueError("hand command limits must have lower < upper")
        if np.any(lower < XHAND_RATED_QPOS_MIN_RAD) or np.any(upper > XHAND_RATED_QPOS_MAX_RAD):
            raise ValueError("hand command limits exceed the rated device envelope")
        validate_hand_target(np.deg2rad(self.home_qpos_deg), lower, upper)
