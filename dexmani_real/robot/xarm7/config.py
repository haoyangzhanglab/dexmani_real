"""xArm7 hardware and firmware-planned HOME parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dexmani_real.robot.model import ARM_JOINT_SHAPE, XARM7_HARD_LOWER, XARM7_HARD_UPPER


@dataclass(frozen=True)
class HomingParams:
    """Firmware-planned execution parameters for validated home milestones."""

    convergence_rad: float = 0.002618
    step_interval_s: float = 0.04
    max_speed_deg_s: float = 30.0  # Conservative Mode-0 speed; tuning needs hardware validation.
    # Settling budget for HOME milestones and Mode-1 restoration.
    target_timeout_s: float = 0.5
    velocity_convergence_rad_s: float = 0.03
    dwell_s: float = 0.30
    convergence_timeout_s: float = 15.0
    state_max_age_s: float = 0.5

    def validate(self) -> None:
        values = (
            self.convergence_rad,
            self.step_interval_s,
            self.max_speed_deg_s,
            self.target_timeout_s,
            self.velocity_convergence_rad_s,
            self.dwell_s,
            self.convergence_timeout_s,
            self.state_max_age_s,
        )
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("all homing parameters must be finite and positive")

    @property
    def max_speed_rad_per_s(self) -> float:
        """Maximum Mode-0 homing speed in radians per second."""
        return float(np.deg2rad(self.max_speed_deg_s))


@dataclass(frozen=True)
class ArmParams:
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

    max_joint_velocity_deg_per_s: float = 135.0  # 75% of the 180 deg/s SDK command limit
    # ~14.14 rad/s²; ~71% of the 20 rad/s² SDK limit.
    max_joint_acceleration_deg_per_s2: float = 810.0
    feedback_max_age_s: float = 4 / 30  # host-side feedback freshness budget

    # Host reference generation is independent of teleop/policy observation cadence.
    servo_hz: float = 100.0
    feedback_hz: float = 25.0
    servo_max_interval_s: float = 0.020
    sdk_timeout_s: float = 0.020
    executor_timeout_s: float = 3.0
    tracking_error_rad: float = 0.35

    ip: str = "192.168.1.111"

    table_z_surface_m: float = 0.022
    hand_safety_margin_m: float = 0.05

    collision_sensitivity: int = 0  # 0 disables xArm controller collision detection.

    tcp_load_mass_kg: float = 1.1
    tcp_load_cog_mm: tuple[float, float, float] = (16.3, 7.9, 109.5)

    homing: HomingParams = field(default_factory=HomingParams)

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
        if not np.isfinite(self.table_z_surface_m):
            raise ValueError("table_z_surface_m must be finite")
        if not np.isfinite(self.hand_safety_margin_m) or self.hand_safety_margin_m < 0:
            raise ValueError("hand_safety_margin_m must be finite and non-negative")
        if not (0 <= self.collision_sensitivity <= 5):
            raise ValueError(
                f"collision_sensitivity={self.collision_sensitivity} out of range [0, 5]"
            )
        for name in (
            "servo_hz",
            "feedback_hz",
            "servo_max_interval_s",
            "sdk_timeout_s",
            "executor_timeout_s",
            "tracking_error_rad",
            "feedback_max_age_s",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"arm {name} must be finite and positive")
        if not (
            self.feedback_hz <= self.servo_hz and 1 / self.feedback_hz < self.feedback_max_age_s
        ):
            raise ValueError("feedback period must fit freshness budget and not exceed servo rate")
        if not 1 / self.servo_hz < self.servo_max_interval_s:
            raise ValueError("servo interval budget must exceed one servo period")
        # These physical v/a limits also remain valid for firmware-planned Mode-0 HOME.
        max_speed_rad = self.max_joint_velocity_rad_per_s
        max_acc_rad = self.max_joint_acceleration_rad_per_s2
        if not (np.isfinite(max_speed_rad) and 0.0001 <= max_speed_rad <= np.pi):
            raise ValueError(
                f"max_joint_velocity_deg_per_s={self.max_joint_velocity_deg_per_s} "
                f"resolves to {max_speed_rad} rad/s, outside the SDK command range [0.0001, π]"
            )
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
