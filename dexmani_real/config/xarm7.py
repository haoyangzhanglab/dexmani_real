"""xArm7 连接、运动限位、伺服与回零参数。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, pi, radians
from typing import cast

import numpy as np

from dexmani_real.robot.model import ARM_DOF, XARM7_HARD_LOWER, XARM7_HARD_UPPER


@dataclass(frozen=True)
class HomingConfig:
    via_qpos: tuple[float, ...] | None = None
    max_jerk_deg_per_s3: float | None = None
    collision_margin_m: float | None = None
    tracking_error_rad: float | None = None
    hand_hold_tolerance_rad: float | None = None
    max_lateness_s: float | None = None
    timeout_s: float = 60.0
    max_speed_deg_s: float = 45.0
    max_acceleration_deg_per_s2: float = 900.0

    @property
    def max_speed_rad_per_s(self) -> float:
        return radians(self.max_speed_deg_s)

    @property
    def max_acceleration_rad_per_s2(self) -> float:
        return radians(self.max_acceleration_deg_per_s2)

    def validate(self) -> None:
        for name in ("max_jerk_deg_per_s3", "collision_margin_m", "tracking_error_rad",
                     "hand_hold_tolerance_rad", "max_lateness_s"):
            value = getattr(self, name)
            if value is None or not isfinite(value) or value <= 0:
                raise ValueError(f"homing.{name} requires a measured positive value")
        acceleration = self.max_acceleration_rad_per_s2
        if not isfinite(acceleration) or not 0.01 <= acceleration <= 20:
            raise ValueError("homing acceleration must be in [0.01, 20] rad/s²")
        if not isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("homing.timeout_s must be finite and positive")
        speed = self.max_speed_rad_per_s
        if not isfinite(speed) or not 0.0001 <= speed <= pi:
            raise ValueError("homing.max_speed_deg_s must convert to [0.0001, π] rad/s")


@dataclass(frozen=True)
class ArmConfig:
    ip: str = "192.168.1.111"
    collision_sensitivity: int = 0  # 0 关闭控制器碰撞检测。
    tcp_load_mass_kg: float = 1.1
    tcp_load_cog_mm: tuple[float, float, float] = (16.3, 7.9, 109.5)

    # J1–J7，位置和关节范围均使用 rad；运行范围只能在本设备有效限位内收窄。
    home_qpos: tuple[float, ...] = (0.0, 0.041888, 0.001745, 0.417134, -3.138102, 1.195551, 0.0)
    joint_limit_lower: tuple[float, ...] = XARM7_HARD_LOWER
    joint_limit_upper: tuple[float, ...] = XARM7_HARD_UPPER

    max_joint_velocity_deg_per_s: float = 150.0
    max_joint_acceleration_deg_per_s2: float = 900.0
    max_joint_jerk_deg_per_s3: float = 12000.0

    servo_hz: float = 100.0
    feedback_hz: float = 25.0
    # 主机 SDK 查询起点的时效预算；默认覆盖 25 Hz 反馈的四个周期。
    feedback_max_age_s: float = 4 / 25

    # HOME、伺服准备与相机标定共用的实测停稳阈值。
    standstill_position_tolerance_rad: float = 0.002618
    standstill_velocity_tolerance_rad_s: float = 0.03

    homing: HomingConfig = field(default_factory=HomingConfig)

    @property
    def max_joint_velocity_rad_per_s(self) -> float:
        return radians(self.max_joint_velocity_deg_per_s)

    @property
    def max_joint_acceleration_rad_per_s2(self) -> float:
        return radians(self.max_joint_acceleration_deg_per_s2)

    @property
    def max_joint_jerk_rad_per_s3(self) -> float:
        return radians(self.max_joint_jerk_deg_per_s3)

    def validate(self) -> None:
        joint_values = np.asarray(
            (self.joint_limit_lower, self.joint_limit_upper), dtype=np.float64,
        )
        if joint_values.shape != (2, ARM_DOF):
            raise ValueError(f"arm joint limits must each have {ARM_DOF} elements")
        lower, upper = joint_values
        if not np.isfinite(joint_values).all() or np.any(lower >= upper):
            raise ValueError("arm joint limits must be finite and ordered")
        if np.any(lower < XARM7_HARD_LOWER) or np.any(upper > XARM7_HARD_UPPER):
            raise ValueError("operational arm limits must be inside device effective limits")
        if not isinstance(self.ip, str) or not self.ip.strip():
            raise ValueError("arm ip must be non-empty")
        if type(self.collision_sensitivity) is not int or not 0 <= self.collision_sensitivity <= 5:
            raise ValueError("arm collision_sensitivity must be an integer in [0, 5]")
        for name in (
            "servo_hz",
            "feedback_hz",
            "feedback_max_age_s",
            "max_joint_jerk_deg_per_s3",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"arm {name} must be finite and positive")
        if 1 / self.feedback_hz >= self.feedback_max_age_s:
            raise ValueError("feedback period must fit freshness budget")
        self._validate_standstill()
        speed = self.max_joint_velocity_rad_per_s
        if not isfinite(speed) or not 0.0001 <= speed <= pi:
            raise ValueError("max_joint_velocity_deg_per_s must convert to [0.0001, π] rad/s")
        max_acc_rad = self.max_joint_acceleration_rad_per_s2
        if not isfinite(max_acc_rad) or not 0.01 <= max_acc_rad <= 20.0:
            raise ValueError(
                f"max_joint_acceleration_deg_per_s2={self.max_joint_acceleration_deg_per_s2} "
                f"resolves to {max_acc_rad} rad/s², outside the SDK command range [0.01, 20]"
            )
        if not isfinite(self.tcp_load_mass_kg) or self.tcp_load_mass_kg < 0:
            raise ValueError("tcp_load_mass_kg must be finite and nonnegative")
        cog = np.asarray(self.tcp_load_cog_mm, dtype=np.float64)
        if cog.shape != (3,) or not np.isfinite(cog).all():
            raise ValueError("tcp_load_cog_mm must be a finite (3,) vector")

    def _validate_standstill(self) -> None:
        for value in (
            self.standstill_position_tolerance_rad,
            self.standstill_velocity_tolerance_rad_s,
        ):
            if not isfinite(value) or value <= 0:
                raise ValueError("arm standstill thresholds must be finite and positive")

    def validate_home(self) -> None:
        from dexmani_real.robot.commands import validate_arm_target

        self.validate()
        self.homing.validate()
        if cast(float, self.homing.max_lateness_s) >= 1 / self.servo_hz:
            raise ValueError("HOME lateness must be below one servo period")
        if (self.homing.max_speed_deg_s > self.max_joint_velocity_deg_per_s
                or self.homing.max_acceleration_deg_per_s2 > self.max_joint_acceleration_deg_per_s2
                or cast(float, self.homing.max_jerk_deg_per_s3) > self.max_joint_jerk_deg_per_s3):
            raise ValueError("HOME motion limits exceed arm limits")
        if self.homing.via_qpos is not None:
            validate_arm_target(self.homing.via_qpos, self.joint_limit_lower, self.joint_limit_upper)
        validate_arm_target(self.home_qpos, self.joint_limit_lower, self.joint_limit_upper)
