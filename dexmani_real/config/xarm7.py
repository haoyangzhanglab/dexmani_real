"""xArm7 连接、运动限位、伺服与回零参数。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, pi, radians

import numpy as np

from dexmani_real.robot.model import ARM_DOF, XARM7_HARD_LOWER, XARM7_HARD_UPPER

SDK_TIMEOUT_S = 0.020
EXECUTOR_TIMEOUT_S = 3.0
HOME_TARGET_TIMEOUT_S = 0.5
HOME_DWELL_S = 0.30
HOME_CONVERGENCE_TIMEOUT_S = 15.0


@dataclass(frozen=True)
class HomingConfig:
    convergence_rad: float = 0.002618
    max_speed_deg_s: float = 45.0
    velocity_convergence_rad_s: float = 0.03

    @property
    def max_speed_rad_per_s(self) -> float:
        return radians(self.max_speed_deg_s)

    def validate(self) -> None:
        for name in ("convergence_rad", "velocity_convergence_rad_s"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"homing.{name} must be finite and positive")
        speed = self.max_speed_rad_per_s
        if not isfinite(speed) or not 0.0001 <= speed <= pi:
            raise ValueError("homing.max_speed_deg_s must convert to [0.0001, π] rad/s")


@dataclass(frozen=True)
class ArmConfig:
    ip: str = "192.168.1.111"
    collision_sensitivity: int = 0  # 0 关闭控制器碰撞检测。
    tcp_load_mass_kg: float = 1.1
    tcp_load_cog_mm: tuple[float, float, float] = (16.3, 7.9, 109.5)

    # J1–J7，位置和关节范围均使用 rad。
    home_qpos: tuple[float, ...] = (0.0, 0.041888, 0.001745, 0.417134, -3.138102, 1.195551, 0.0)
    joint_limit_lower: tuple[float, ...] = XARM7_HARD_LOWER
    joint_limit_upper: tuple[float, ...] = XARM7_HARD_UPPER

    max_joint_velocity_deg_per_s: float = 150.0
    max_joint_acceleration_deg_per_s2: float = 900.0
    max_joint_jerk_deg_per_s3: float = 12000.0

    servo_hz: float = 100.0
    feedback_hz: float = 25.0
    feedback_max_age_s: float = 4 / 30

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
            (self.home_qpos, self.joint_limit_lower, self.joint_limit_upper), dtype=np.float64,
        )
        if joint_values.shape != (3, ARM_DOF):
            raise ValueError(f"arm home and joint limits must each have {ARM_DOF} elements")
        home, lower, upper = joint_values
        if not np.isfinite(joint_values).all() or np.any(lower >= upper):
            raise ValueError("arm home and joint limits must be finite and ordered")
        if np.any(lower < XARM7_HARD_LOWER) or np.any(upper > XARM7_HARD_UPPER):
            raise ValueError("operational arm limits must be inside mechanical limits")
        if np.any(home < lower) or np.any(home > upper):
            raise ValueError("home_qpos must be within joint limits")
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
        if not (
            self.feedback_hz <= self.servo_hz and 1 / self.feedback_hz < self.feedback_max_age_s
        ):
            raise ValueError("feedback period must fit freshness budget and not exceed servo rate")
        self.homing.validate()
        speed = self.max_joint_velocity_rad_per_s
        if not isfinite(speed) or not 0.0001 <= speed <= pi:
            raise ValueError("max_joint_velocity_deg_per_s must convert to [0.0001, π] rad/s")
        max_acc_rad = self.max_joint_acceleration_rad_per_s2
        if not isfinite(max_acc_rad) or not 0.01 <= max_acc_rad <= 20.0:
            raise ValueError(
                f"max_joint_acceleration_deg_per_s2={self.max_joint_acceleration_deg_per_s2} "
                f"resolves to {max_acc_rad} rad/s², outside the SDK command range [0.01, 20]"
            )
        if not isfinite(self.tcp_load_mass_kg) or self.tcp_load_mass_kg <= 0:
            raise ValueError("tcp_load_mass_kg must be finite and positive")
        cog = np.asarray(self.tcp_load_cog_mm, dtype=np.float64)
        if cog.shape != (3,) or not np.isfinite(cog).all():
            raise ValueError("tcp_load_cog_mm must be a finite (3,) vector")
