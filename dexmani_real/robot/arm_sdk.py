"""Shared xArm SDK surface for the arm servo loop and homing execution.

Holds the pieces both ``arm_loop.py`` (servo loop) and ``homing.py`` (homing
execution) need without either owning them: the resolved ``ArmLoopConfig`` and
the two leaf live-read primitives.  ``arm_loop`` imports from here, and so does
``homing``; neither imports the other for these, so the dependency graph stays
acyclic.  Live-status *composite* helpers (``_read_live_status``,
``_wait_live_status``) remain in ``arm_loop`` because only the servo loop uses
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.config.defaults import arm, safety
from dexmani_real.utils.schema import ARM_JOINT_SHAPE


@dataclass
class ArmLoopConfig:
    """Mode 6 joint online trajectory planning configuration."""

    joint_max_speed_rad_per_s: float = field(
        default_factory=lambda: arm.max_joint_velocity_rad_per_s
    )
    joint_max_acc_rad_per_s2: float = field(
        default_factory=lambda: arm.max_joint_acceleration_rad_per_s2
    )
    arm_loop_hz: float = field(default_factory=lambda: arm.loop_hz)

    joint_limit_lower: tuple[float, ...] = field(
        default_factory=lambda: arm.joint_limit_lower
    )
    joint_limit_upper: tuple[float, ...] = field(
        default_factory=lambda: arm.joint_limit_upper
    )

    tracking_error_warn_rad: float = field(
        default_factory=lambda: arm.tracking_error_warn_rad
    )

    arm_ip: str = field(default_factory=lambda: arm.ip)

    home_qpos: tuple[float, ...] = field(default_factory=lambda: arm.home_qpos)

    collision_sensitivity: int = field(
        default_factory=lambda: arm.collision_sensitivity
    )
    recoverable_errors: frozenset[int] = field(
        default_factory=lambda: arm.recoverable_errors
    )
    collision_fault_errors: frozenset[int] = field(
        default_factory=lambda: arm.collision_fault_errors
    )
    max_consecutive_recoveries: int = field(
        default_factory=lambda: safety.max_consecutive_recoveries
    )

    homing_convergence_rad: float = field(
        default_factory=lambda: arm.homing.convergence_rad
    )
    homing_step_interval_s: float = field(
        default_factory=lambda: arm.homing.step_interval_s
    )
    homing_max_speed_rad_per_s: float = field(
        default_factory=lambda: np.deg2rad(arm.homing.max_speed_deg_s)
    )
    homing_target_timeout_s: float = field(
        default_factory=lambda: arm.homing.target_timeout_s
    )
    homing_velocity_convergence_rad_s: float = field(
        default_factory=lambda: arm.homing.velocity_convergence_rad_s
    )
    homing_dwell_s: float = field(default_factory=lambda: arm.homing.dwell_s)

    tcp_load_mass_kg: float = field(default_factory=lambda: arm.tcp_load_mass_kg)
    tcp_load_cog_mm: tuple[float, float, float] = field(
        default_factory=lambda: arm.tcp_load_cog_mm
    )

    def __post_init__(self) -> None:
        lower = np.asarray(self.joint_limit_lower, dtype=np.float64)
        upper = np.asarray(self.joint_limit_upper, dtype=np.float64)
        home = np.asarray(self.home_qpos, dtype=np.float64)
        if (
            lower.shape != ARM_JOINT_SHAPE
            or upper.shape != ARM_JOINT_SHAPE
            or home.shape != ARM_JOINT_SHAPE
        ):
            raise ValueError(
                f"arm loop joint limits/home must have shape {ARM_JOINT_SHAPE}"
            )
        if not np.all(np.isfinite(np.concatenate((lower, upper, home)))) or np.any(
            lower > upper
        ):
            raise ValueError("arm loop joint limits/home must be finite and ordered")
        if self.recoverable_errors & self.collision_fault_errors:
            raise ValueError(
                "recoverable and collision-fault error codes must be disjoint"
            )
        if self.recoverable_errors != frozenset({24}) or not frozenset(
            {22, 31}
        ).issubset(self.collision_fault_errors):
            raise ValueError(
                "arm loop requires only C24 recoverable and C22/C31 collision-fatal"
            )
        if self.max_consecutive_recoveries <= 0:
            raise ValueError("max_consecutive_recoveries must be positive")
        timing = (
            self.joint_max_speed_rad_per_s,
            self.joint_max_acc_rad_per_s2,
            self.arm_loop_hz,
            self.tracking_error_warn_rad,
            self.homing_convergence_rad,
            self.homing_step_interval_s,
            self.homing_max_speed_rad_per_s,
            self.homing_target_timeout_s,
            self.homing_velocity_convergence_rad_s,
            self.homing_dwell_s,
        )
        if not all(np.isfinite(value) and value > 0 for value in timing):
            raise ValueError(
                "arm loop motion/homing parameters must be finite and positive"
            )
        if not self.arm_ip or not (0 <= self.collision_sensitivity <= 5):
            raise ValueError("arm loop IP/collision sensitivity is invalid")
        if not np.isfinite(self.tcp_load_mass_kg) or self.tcp_load_mass_kg <= 0:
            raise ValueError("arm loop tcp_load_mass_kg must be finite and positive")
        cog = np.asarray(self.tcp_load_cog_mm, dtype=np.float64)
        if cog.shape != (3,) or not np.all(np.isfinite(cog)):
            raise ValueError("arm loop tcp_load_cog_mm must be a finite (3,) vector")

    @classmethod
    def from_runtime(cls, runtime: Any) -> "ArmLoopConfig":
        cfg = runtime.arm
        return cls(
            joint_max_speed_rad_per_s=float(
                np.deg2rad(cfg.max_joint_velocity_deg_per_s)
            ),
            joint_max_acc_rad_per_s2=float(
                np.deg2rad(cfg.max_joint_acceleration_deg_per_s2)
            ),
            arm_loop_hz=float(cfg.loop_hz),
            joint_limit_lower=tuple(cfg.joint_limit_lower),
            joint_limit_upper=tuple(cfg.joint_limit_upper),
            tracking_error_warn_rad=float(cfg.tracking_error_warn_rad),
            arm_ip=str(cfg.ip),
            home_qpos=tuple(cfg.home_qpos),
            collision_sensitivity=int(cfg.collision_sensitivity),
            recoverable_errors=frozenset(int(code) for code in cfg.recoverable_errors),
            collision_fault_errors=frozenset(
                int(code) for code in cfg.collision_fault_errors
            ),
            max_consecutive_recoveries=int(runtime.safety.max_consecutive_recoveries),
            homing_convergence_rad=float(cfg.homing.convergence_rad),
            homing_step_interval_s=float(cfg.homing.step_interval_s),
            homing_max_speed_rad_per_s=float(np.deg2rad(cfg.homing.max_speed_deg_s)),
            homing_target_timeout_s=float(cfg.homing.target_timeout_s),
            homing_velocity_convergence_rad_s=float(
                cfg.homing.velocity_convergence_rad_s
            ),
            homing_dwell_s=float(cfg.homing.dwell_s),
            tcp_load_mass_kg=float(cfg.tcp_load_mass_kg),
            tcp_load_cog_mm=tuple(cfg.tcp_load_cog_mm),
        )


# Controller errors: C24 is recoverable; C22/C31 are immediate collision faults.
def _require_sdk_ok(operation: str, code: Any) -> None:
    """Raise when an xArm setter reports failure without raising."""
    if not isinstance(code, (int, np.integer)) or int(code) != 0:
        raise RuntimeError(f"{operation} failed with SDK code {code!r}")


def _read_live_error_code(arm_api: Any) -> int:
    """Return the live controller error code; raise if the live read fails.

    Unlike the cached ``arm.error_code`` property (updated by a background
    report thread), this synchronously reads ``get_err_warn_code``.  Control
    decisions that follow a setter failure or a homing run must use this and
    treat a raise as a fault — never fall back to the cached value.
    """
    code, values = arm_api.get_err_warn_code()
    _require_sdk_ok("get_err_warn_code", code)
    return int(values[0])
