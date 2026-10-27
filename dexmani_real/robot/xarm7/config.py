from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dexmani_real.robot.model import ARM_JOINT_SHAPE, XARM7_HARD_LOWER, XARM7_HARD_UPPER

SERVO_MAX_INTERVAL_S = 0.020
SDK_TIMEOUT_S = 0.020
EXECUTOR_TIMEOUT_S = 3.0
HOME_TARGET_TIMEOUT_S = 0.5
HOME_DWELL_S = 0.30
HOME_CONVERGENCE_TIMEOUT_S = 15.0


@dataclass(frozen=True)
class HomingConfig:
    convergence_rad: float = 0.002618
    max_speed_deg_s: float = 30.0
    velocity_convergence_rad_s: float = 0.03

    @property
    def max_speed_rad_per_s(self) -> float:
        return float(np.deg2rad(self.max_speed_deg_s))


@dataclass(frozen=True)
class ArmConfig:
    home_qpos: tuple[float, ...] = (
        0.0,
        0.041888,
        0.001745,
        0.417134,
        -3.138102,
        1.195551,
        0.0,
    )

    joint_limit_lower: tuple[float, ...] = XARM7_HARD_LOWER
    joint_limit_upper: tuple[float, ...] = XARM7_HARD_UPPER

    max_joint_velocity_deg_per_s: float = 135.0
    max_joint_acceleration_deg_per_s2: float = 810.0
    feedback_max_age_s: float = 4 / 30

    servo_hz: float = 100.0
    feedback_hz: float = 25.0
    tracking_error_rad: float = 0.35

    ip: str = "192.168.1.111"

    collision_sensitivity: int = 0  # 0 关闭控制器碰撞检测。

    tcp_load_mass_kg: float = 1.1
    tcp_load_cog_mm: tuple[float, float, float] = (16.3, 7.9, 109.5)

    homing: HomingConfig = field(default_factory=HomingConfig)

    @property
    def max_joint_velocity_rad_per_s(self) -> float:
        return float(np.deg2rad(self.max_joint_velocity_deg_per_s))

    @property
    def max_joint_acceleration_rad_per_s2(self) -> float:
        return float(np.deg2rad(self.max_joint_acceleration_deg_per_s2))

    def validate(self) -> None:
        home = np.asarray(self.home_qpos, dtype=np.float64)
        lower = np.asarray(self.joint_limit_lower, dtype=np.float64)
        upper = np.asarray(self.joint_limit_upper, dtype=np.float64)
        if (
            home.shape != ARM_JOINT_SHAPE
            or lower.shape != ARM_JOINT_SHAPE
            or upper.shape != ARM_JOINT_SHAPE
        ):
            raise ValueError("arm home and joint limits must have 7 elements")
        if not np.all(np.isfinite(np.concatenate((home, lower, upper)))) or np.any(lower >= upper):
            raise ValueError("arm home and joint limits must be finite and ordered")
        if np.any(lower < XARM7_HARD_LOWER) or np.any(upper > XARM7_HARD_UPPER):
            raise ValueError("operational arm limits must be inside mechanical limits")
        if np.any(home < lower) or np.any(home > upper):
            raise ValueError("home_qpos must be within joint limits")
        if not self.ip:
            raise ValueError("arm ip must be non-empty")
        if not (0 <= self.collision_sensitivity <= 5):
            raise ValueError(
                f"collision_sensitivity={self.collision_sensitivity} out of range [0, 5]"
            )
        for name in (
            "servo_hz",
            "feedback_hz",
            "tracking_error_rad",
            "feedback_max_age_s",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"arm {name} must be finite and positive")
        if not (
            self.feedback_hz <= self.servo_hz and 1 / self.feedback_hz < self.feedback_max_age_s
        ):
            raise ValueError("feedback period must fit freshness budget and not exceed servo rate")
        if not 1 / self.servo_hz < SERVO_MAX_INTERVAL_S:
            raise ValueError("servo interval budget must exceed one servo period")

        for value in (self.homing.convergence_rad, self.homing.velocity_convergence_rad_s):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("HOME position/velocity tolerances must be finite and positive")
        for name, speed in (
            ("max_joint_velocity_deg_per_s", self.max_joint_velocity_rad_per_s),
            ("homing.max_speed_deg_s", self.homing.max_speed_rad_per_s),
        ):
            if not (np.isfinite(speed) and 0.0001 <= speed <= np.pi):
                raise ValueError(f"{name} must resolve to an SDK speed in [0.0001, π] rad/s")
        max_acc_rad = self.max_joint_acceleration_rad_per_s2
        if not (np.isfinite(max_acc_rad) and 0.01 <= max_acc_rad <= 20.0):
            raise ValueError(
                f"max_joint_acceleration_deg_per_s2={self.max_joint_acceleration_deg_per_s2} "
                f"resolves to {max_acc_rad} rad/s², outside the SDK command range [0.01, 20]"
            )
        if not np.isfinite(self.tcp_load_mass_kg) or self.tcp_load_mass_kg <= 0:
            raise ValueError("tcp_load_mass_kg must be finite and positive")
        cog = np.asarray(self.tcp_load_cog_mm, dtype=np.float64)
        if cog.shape != (3,) or not np.all(np.isfinite(cog)):
            raise ValueError("tcp_load_cog_mm must be a finite (3,) vector")
