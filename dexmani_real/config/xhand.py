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

CONNECT_TIMEOUT_S = 2.0
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
    # 显式 HOME 在操作范围内部留出余量，5° 到位误差不会跨过默认边界。
    home_qpos_deg: tuple[float, ...] = (
        30.0, 55.33, 20.0,  # 拇指
        0.17, 10.0, 15.0,   # 食指
        10.0, 15.0,         # 中指
        10.0, 15.0,         # 无名指
        10.0, 15.0,         # 小指
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

    feedback_max_age_s: float = 0.5
    home_timeout_s: float = 2.0
    home_tolerance_deg: float = 5.0

    def __post_init__(self) -> None:
        # 配置保留原生整数表示，SDK 参数及 JSON/YAML 归档无需各自适配 NumPy。
        for name in ("baudrate", "device_id", "ethercat_slave_position", "ki", "kd"):
            value = getattr(self, name)
            if isinstance(value, np.integer):
                object.__setattr__(self, name, int(value))
        for name in ("kp", "tor_max"):
            object.__setattr__(self, name, tuple(
                int(value) if isinstance(value, np.integer) else value
                for value in getattr(self, name)
            ))

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
            communication_field, ("device_id", 0), ("ki", 0), ("kd", 0),
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"hand {name} must be an integer >= {minimum}")
        self.validate_limits()
        if len(self.kp) != HAND_DOF or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.kp
        ):
            raise ValueError("hand kp must contain twelve positive integer gains")
        if len(self.tor_max) != HAND_DOF or any(
            not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 400
            for value in self.tor_max
        ):
            raise ValueError(
                "hand tor_max must contain twelve integers in [0, 400] (Mode 3 output scale)"
            )
        if not isfinite(self.feedback_max_age_s) or self.feedback_max_age_s <= 0:
            raise ValueError("hand feedback_max_age_s must be finite and positive")

    def validate_limits(self) -> None:
        """实验范围必须包含在硬件额定范围内。"""
        bounds = np.asarray((self.qpos_min_rad, self.qpos_max_rad), dtype=np.float64)
        if bounds.shape != (2, HAND_DOF) or not np.isfinite(bounds).all():
            raise ValueError("hand command limits must each contain 12 finite values")
        lower, upper = bounds
        if np.any(lower >= upper):
            raise ValueError("hand command limits must have lower < upper")
        if np.any(lower < XHAND_RATED_QPOS_MIN_RAD) or np.any(upper > XHAND_RATED_QPOS_MAX_RAD):
            raise ValueError("hand command limits exceed the rated device envelope")

    def validate_home_pose(self) -> None:
        self.validate_limits()
        validate_hand_target(np.deg2rad(self.home_qpos_deg), self.qpos_min_rad, self.qpos_max_rad)

    def validate_home(self) -> None:
        self.validate_home_pose()
        for value in (self.home_timeout_s, self.home_tolerance_deg):
            if not isfinite(value) or value <= 0:
                raise ValueError("hand HOME timeout and tolerance must be finite and positive")
