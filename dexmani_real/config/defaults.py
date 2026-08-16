"""Numeric defaults shared by experiment configs and worker adapters.

Angles are radians unless named ``_deg``; distances are metres, rates are Hz,
and durations are seconds. Runtime YAML overrides these dataclass values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import numpy as np

from dexmani_real.utils.limits import validate_hand_limit_nesting
from dexmani_real.utils.schema import ARM_JOINT_SHAPE

_HEARTBEAT_SUBSYSTEMS = frozenset({"arm", "hand", "policy", "recorder", "vr", "camera", "inference"})
_READINESS_SUBSYSTEMS = frozenset({"arm", "hand", "camera", "recorder", "policy", "vr", "inference"})

# Shared sub-structures


@dataclass(frozen=True)
class HomingParams:
    """Firmware-planned execution parameters for validated home milestones."""

    convergence_rad: float = 0.002618  # final canonical-home tolerance (~0.15°)
    step_interval_s: float = 0.04  # controller-state polling interval
    max_speed_deg_s: float = 30.0  # conservative Mode 0 joint speed; hardware validation required before tuning
    target_timeout_s: float = 0.5  # settling allowance added after distance/speed timing
    velocity_convergence_rad_s: float = 0.03
    dwell_s: float = 0.30
    convergence_timeout_s: float = 15.0
    request_queue_timeout_s: float = 0.2
    state_max_age_s: float = 0.5

    def __post_init__(self) -> None:
        values = (
            self.convergence_rad,
            self.step_interval_s,
            self.max_speed_deg_s,
            self.target_timeout_s,
            self.velocity_convergence_rad_s,
            self.dwell_s,
            self.convergence_timeout_s,
            self.request_queue_timeout_s,
            self.state_max_age_s,
        )
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("all homing parameters must be finite and positive")


@dataclass(frozen=True)
class WorkspaceBounds:
    """EEF workspace bounds in arm-base frame (meters)."""

    x_min: float = 0.25
    x_max: float = 0.72
    y_min: float = -0.50
    y_max: float = 0.50
    z_min: float = 0.05
    z_max: float = 0.50

    def __post_init__(self) -> None:
        bounds = np.asarray(self.as_tuple(), dtype=np.float64)
        if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("workspace bounds must be finite and ordered")

    def as_tuple(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Workspace bounds as ((x_min,x_max), (y_min,y_max), (z_min,z_max))."""
        return (
            (self.x_min, self.x_max),
            (self.y_min, self.y_max),
            (self.z_min, self.z_max),
        )

    def as_array(self) -> "np.ndarray":
        """Workspace bounds as (3,2) np.ndarray — returns a mutable copy."""
        return np.array(
            [[self.x_min, self.x_max], [self.y_min, self.y_max], [self.z_min, self.z_max]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class EMAParams:
    """Cartesian-space EMA smoothing parameters (tuned for the default 16 Hz grid)."""

    alpha_pos: float = 0.6  # τ≈65 ms at 16 Hz
    alpha_rot: float = 0.25  # τ≈223 ms at 16 Hz

    def __post_init__(self) -> None:
        if not (np.isfinite(self.alpha_pos) and np.isfinite(self.alpha_rot)):
            raise ValueError("EMA alphas must be finite")
        if not (0.0 <= self.alpha_pos <= 1.0 and 0.0 <= self.alpha_rot <= 1.0):
            raise ValueError("EMA alphas must be in [0, 1]")


@dataclass(frozen=True)
class VRMappingParams:
    """VR wrist → EEF mapping parameters."""

    pos_scale: float = 1.0
    rot_scale: float = 1.0
    max_delta_rot_rad: float = 3.0  # ~172° total-from-reset rotation cap
    stale_threshold_s: float = 0.5

    def __post_init__(self) -> None:
        values = (self.pos_scale, self.rot_scale, self.max_delta_rot_rad, self.stale_threshold_s)
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("VR mapping scales, delta, and stale threshold must be finite and positive")


@dataclass(frozen=True)
class StaticCollisionBox:
    """One oriented static obstacle in the xArm base frame.

    ``size_xyz_m`` contains full side lengths (not half extents) and
    ``quat_wxyz`` rotates the box-local axes into the base frame.
    """

    name: str = "obstacle"
    center_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size_xyz_m: tuple[float, float, float] = (1.0, 1.0, 1.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or self.name != self.name.strip()
            or self.name == "table"
        ):
            raise ValueError("static collision box name must be non-empty and must not use reserved name 'table'")
        for field_name in ("center_xyz_m", "size_xyz_m", "quat_wxyz"):
            if not isinstance(getattr(self, field_name), tuple):
                raise TypeError(f"static collision box {field_name} must be an immutable tuple")
        center = np.asarray(self.center_xyz_m, dtype=np.float64)
        size = np.asarray(self.size_xyz_m, dtype=np.float64)
        quat = np.asarray(self.quat_wxyz, dtype=np.float64)
        if center.shape != (3,) or size.shape != (3,) or quat.shape != (4,):
            raise ValueError("static collision box center/size/quaternion must have shapes (3,), (3,), and (4,)")
        if not np.all(np.isfinite(np.concatenate((center, size, quat)))):
            raise ValueError("static collision box values must be finite")
        if np.any(size <= 0.0):
            raise ValueError("static collision box size_xyz_m must contain positive full side lengths")
        if not np.isclose(float(np.linalg.norm(quat)), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("static collision box quat_wxyz must be a unit quaternion")


@dataclass(frozen=True)
class TableCollisionConfig:
    """Calibrated table represented as a finite box below an upward plane.

    ``plane_abcd`` uses the robot-base/world convention ``ax+by+cz+d=0``.
    Runtime resolution refreshes it from ``plane_path`` so perception,
    online action validation, and homing use the same calibration artifact.
    """

    enabled: bool = True
    plane_path: str | None = "dexmani_real/config/desk_plane.json"
    plane_abcd: tuple[float, float, float, float] = (0.0, 0.0, 1.0, -0.022)
    size_xy_m: tuple[float, float] = (2.0, 2.0)
    thickness_m: float = 0.04
    # Nominal 10 mm mesh clearance plus the ~10 mm flange-adapter residual: the
    # collision model loads the collision URDF with flange_joint2 at the raw
    # 0.043 m, while the FK/planning path compensates it to 0.033 m (see
    # HandParams.T_eef_handbase_pos_xyz). The collision model applies no
    # equivalent correction, so that residual is folded into this clearance.
    soft_clearance_m: float = 0.02
    allowed_contact_links: tuple[str, ...] = ("link_base",)

    def __post_init__(self) -> None:
        if self.plane_path is not None and (not isinstance(self.plane_path, str) or not self.plane_path.strip()):
            raise ValueError("table plane_path must be a non-empty string or null")
        plane = np.asarray(self.plane_abcd, dtype=np.float64)
        size = np.asarray(self.size_xy_m, dtype=np.float64)
        if plane.shape != (4,) or size.shape != (2,):
            raise ValueError("table plane_abcd and size_xy_m must have shapes (4,) and (2,)")
        if not np.all(np.isfinite(np.concatenate((plane, size)))):
            raise ValueError("table plane and size must be finite")
        normal_norm = float(np.linalg.norm(plane[:3]))
        if normal_norm <= 1e-9 or float(plane[2] / normal_norm) <= 0.0:
            raise ValueError("table plane must have a finite upward-pointing normal")
        if np.any(size <= 0.0) or not np.isfinite(self.thickness_m) or self.thickness_m <= 0.0:
            raise ValueError("table size and thickness must be finite and positive")
        if not np.isfinite(self.soft_clearance_m) or self.soft_clearance_m < 0.0:
            raise ValueError("table soft_clearance_m must be finite and non-negative")
        if not isinstance(self.allowed_contact_links, tuple) or any(
            not isinstance(name, str) or not name.strip() for name in self.allowed_contact_links
        ):
            raise TypeError("table allowed_contact_links must be a tuple of non-empty link names")
        if len(self.allowed_contact_links) != len(set(self.allowed_contact_links)):
            raise ValueError("table allowed_contact_links must be unique")


@dataclass(frozen=True)
class EnvironmentConfig:
    """Calibrated table plus optional static robot-environment geometry."""

    table: TableCollisionConfig = field(default_factory=TableCollisionConfig)
    static_boxes: tuple[StaticCollisionBox, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.table, TableCollisionConfig):
            raise TypeError("environment.table must be a TableCollisionConfig value")
        if not isinstance(self.static_boxes, tuple) or any(
            not isinstance(box, StaticCollisionBox) for box in self.static_boxes
        ):
            raise TypeError("environment.static_boxes must be a tuple of StaticCollisionBox values")
        names = [box.name for box in self.static_boxes]
        if len(names) != len(set(names)):
            raise ValueError("environment.static_boxes names must be unique")


# Arm parameters (xArm7, 7-DOF)


@dataclass(frozen=True)
class ArmParams:
    """xArm7 hardware parameters — single source of truth."""

    # ── Home position (rad) — neutral pose ──
    home_qpos: tuple[float, ...] = (
        0.0,  # J1: 0.0°
        0.041888,  # J2: 2.4°
        0.001745,  # J3: 0.1°
        0.417134,  # J4: 23.9°
        -3.138102,  # J5: -179.8°
        1.195551,  # J6: 68.5°
        0.0,  # J7: 0.0°
    )

    # ── Joint limits (rad) — mirrors xarm7 URDF ──
    # URDF source: assets/robots/xhand/xarm7_xhand_collision.urdf
    joint_limit_lower: tuple[float, ...] = (
        -6.28318530718,
        -2.059,
        -6.28318530718,
        -0.19198,
        -6.28318530718,
        -1.69297,
        -6.28318530718,
    )
    joint_limit_upper: tuple[float, ...] = (
        6.28318530718,
        2.0944,
        6.28318530718,
        3.927,
        6.28318530718,
        3.14159265359,
        6.28318530718,
    )

    # ── Dynamics (Mode 6 firmware) ──
    max_joint_velocity_deg_per_s: float = 120.0  # firmware trajectory speed
    max_joint_acceleration_deg_per_s2: float = 900.0  # firmware trajectory acceleration
    loop_hz: float = 30.0  # arm_loop servo rate

    # ── Connection ──
    ip: str = "192.168.1.111"

    # ── Device identity (validated against the SDK report at connect) ──
    expected_axis: int = 7
    # Explicit model check is opt-in: None means "don't guess the model" (doc §8.5).
    device_profile: str | None = None
    serial_number: str | None = None  # enforced only when configured
    min_firmware: tuple[int, ...] | None = None  # integer-tuple compare via version_number

    # ── Environment ──
    # Compatibility fallback when calibrated environment.table is disabled.
    table_z_surface_m: float = 0.022
    hand_safety_margin_m: float = 0.05

    # ── Safety ──
    tracking_error_warn_rad: float = 0.35  # diagnostic warning threshold
    # UFACTORY semantics: 0 disables detection; 1 is the least-sensitive
    # enabled level and sensitivity increases through level 5.
    collision_sensitivity: int = 1

    # ── TCP load (end-effector mass/COG for firmware dynamics) ──
    # XHand (1.1 kg). COG in tool-flange frame (link_eef) from URDF weighted-COM
    # of all end-effector links; flange_joint2 corrected 0.043→0.033 m per
    # physical measurement.
    tcp_load_mass_kg: float = 1.1
    tcp_load_cog_mm: tuple[float, float, float] = (16.3, 7.9, 109.5)

    # ── Homing ──
    homing: HomingParams = field(default_factory=HomingParams)

    # ── Derived ──
    @property
    def max_joint_velocity_rad_per_s(self) -> float:
        return float(np.deg2rad(self.max_joint_velocity_deg_per_s))

    @property
    def max_joint_acceleration_rad_per_s2(self) -> float:
        return float(np.deg2rad(self.max_joint_acceleration_deg_per_s2))

    def __post_init__(self) -> None:
        home = np.asarray(self.home_qpos, dtype=np.float64)
        lower = np.asarray(self.joint_limit_lower, dtype=np.float64)
        upper = np.asarray(self.joint_limit_upper, dtype=np.float64)
        if home.shape != ARM_JOINT_SHAPE or lower.shape != ARM_JOINT_SHAPE or upper.shape != ARM_JOINT_SHAPE:
            raise ValueError("arm home and joint limits must have 7 elements")
        if not np.all(np.isfinite(np.concatenate((home, lower, upper)))) or np.any(lower > upper):
            raise ValueError("arm home and joint limits must be finite and ordered")
        if np.any(home < lower) or np.any(home > upper):
            raise ValueError("home_qpos must be within joint limits")
        if not self.ip:
            raise ValueError("arm ip must be non-empty")
        if not np.isfinite(self.loop_hz) or self.loop_hz <= 0:
            raise ValueError("arm loop_hz must be finite and positive")
        if not np.isfinite(self.table_z_surface_m):
            raise ValueError("table_z_surface_m must be finite")
        if not np.isfinite(self.hand_safety_margin_m) or self.hand_safety_margin_m < 0:
            raise ValueError("hand_safety_margin_m must be finite and non-negative")
        if not np.isfinite(self.tracking_error_warn_rad) or self.tracking_error_warn_rad <= 0:
            raise ValueError("tracking_error_warn_rad must be finite and positive")
        if not (0 <= self.collision_sensitivity <= 5):
            raise ValueError(f"collision_sensitivity={self.collision_sensitivity} out of range [0, 5]")
        # The Mode 6 command path hard-clamps speed to [0.0001, π] rad/s and
        # mvacc to [0.01, 20] rad/s²; validate in radians (not degrees) so the
        # configured value can never be silently rewritten by the firmware.
        _max_speed_rad = self.max_joint_velocity_rad_per_s
        _max_acc_rad = self.max_joint_acceleration_rad_per_s2
        if not (np.isfinite(_max_speed_rad) and 0.0001 <= _max_speed_rad <= np.pi):
            raise ValueError(
                f"max_joint_velocity_deg_per_s={self.max_joint_velocity_deg_per_s} "
                f"resolves to {_max_speed_rad} rad/s, outside the SDK command range [0.0001, π]"
            )
        if not (np.isfinite(_max_acc_rad) and 0.01 <= _max_acc_rad <= 20.0):
            raise ValueError(
                f"max_joint_acceleration_deg_per_s2={self.max_joint_acceleration_deg_per_s2} "
                f"resolves to {_max_acc_rad} rad/s², outside the SDK command range [0.01, 20]"
            )
        if not isinstance(self.expected_axis, int) or self.expected_axis <= 0:
            raise ValueError("expected_axis must be a positive integer")
        if self.device_profile is not None and not self.device_profile:
            raise ValueError("device_profile must be non-empty when set")
        if self.serial_number is not None and not self.serial_number:
            raise ValueError("serial_number must be non-empty when set")
        if self.min_firmware is not None and (
            not isinstance(self.min_firmware, tuple)
            or not self.min_firmware
            or any(not isinstance(v, int) or v < 0 for v in self.min_firmware)
        ):
            raise ValueError("min_firmware must be a non-empty tuple of non-negative integers")
        if not np.isfinite(self.tcp_load_mass_kg) or self.tcp_load_mass_kg <= 0:
            raise ValueError("tcp_load_mass_kg must be finite and positive")
        cog = np.asarray(self.tcp_load_cog_mm, dtype=np.float64)
        if cog.shape != (3,) or not np.all(np.isfinite(cog)):
            raise ValueError("tcp_load_cog_mm must be a finite (3,) vector")


# Hand parameters (XHand, 12-DOF)

_XHAND_RATED_QPOS_MIN_RAD: tuple[float, ...] = (
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
_XHAND_RATED_QPOS_MAX_RAD: tuple[float, ...] = (
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


@dataclass(frozen=True)
class HandParams:
    """XHand hardware parameters — single source of truth."""

    # Vendor EtherCAT slave position has not been independently validated for
    # this installation. -1 means unknown: close the device and wait for the
    # watchdog, but do not issue a guessed set_firmware_state() request.
    ethercat_slave_position: int = -1

    # ── Home position (deg) — open-hand neutral ──
    home_qpos_deg: tuple[float, ...] = (
        0.0,
        80.66,
        33.2,
        0.0,
        5.11,
        5.0,
        6.53,
        5.0,
        6.76,
        5.0,
        10.13,
        5.0,
    )

    # Rated mechanical ranges from the bundled vendor-derived XHand right-hand
    # URDF. These describe the model/hard-stop envelope and cannot be widened
    # by a runtime override for this device model.
    mechanical_qpos_min_rad: tuple[float, ...] = _XHAND_RATED_QPOS_MIN_RAD
    mechanical_qpos_max_rad: tuple[float, ...] = _XHAND_RATED_QPOS_MAX_RAD

    # Operational command bounds. The distal lower bounds are an operator-set
    # anti-clogging margin (thumb_rota_j2 +10°, index/mid/ring/pinky_j2 +5°)
    # that keeps *commands* away from the 0° hard stop. They are enforced by
    # clipping the command on publish — never by rejecting the action, and never
    # against measured feedback: the hand settles imprecisely (a 5° command
    # lands ~4.6°) and contact can pull the measured angle far from the command.
    qpos_min_rad: tuple[float, ...] = (
        0.0,
        -0.698,
        0.17453292519943295,
        -0.174,
        0.0,
        0.08726646259971647,
        0.0,
        0.08726646259971647,
        0.0,
        0.08726646259971647,
        0.0,
        0.08726646259971647,
    )
    qpos_max_rad: tuple[float, ...] = _XHAND_RATED_QPOS_MAX_RAD
    # Command-to-command rate bound. None disables it: the EtherCAT firmware
    # PID (with per-joint current limits) is the final velocity backstop,
    # mirroring how the arm velocity envelope was removed. When configured,
    # the controller clips the motion to this bound instead of rejecting the
    # whole action.
    max_delta_rad: float | None = None

    # ── Servo gains (PID) and current limit ──
    # Per-joint proportional gains. Index abduction (J3) is raised to 120 to
    # compensate for sideways loading; all other joints use 100. Integral and
    # derivative gains are uniform (0) across all joints.
    kp: tuple[int, ...] = (
        100,
        100,
        100,
        120,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
    )
    ki: int = 0
    kd: int = 0

    # Per-joint torque (current) limit in mA. Index abduction (J3) is raised
    # to 360 mA to handle sideways loading; all other joints use 300 mA.
    tor_max_ma: tuple[int, ...] = (
        300,
        300,
        300,
        360,
        300,
        300,
        300,
        300,
        300,
        300,
        300,
        300,
    )

    loop_hz: float = 30.0

    # ── Homing ──
    # Return-home waits only for the hand worker/SDK to accept the configured
    # home command. It never waits for measured joint convergence.
    home_command_ack_timeout_s: float = 1.0

    # ── Send-error watchdog ──
    send_err_watchdog_count: int = 30  # 1s @ 30Hz

    # ── Hand FK (fingertip positions in world frame) ──
    fingertip_link_names: tuple[str, ...] = (
        "right_hand_thumb_rota_tip",
        "right_hand_index_rota_tip",
        "right_hand_mid_tip",
        "right_hand_ring_tip",
        "right_hand_pinky_tip",
    )
    # Static transform from planning EEF (custom_eef_link) to hand base (right_hand_link).
    # quat = RotY(+π/2) — hand_base Z points palm-forward in URDF; EEF X is link_eef Z.
    #
    # T_eef_handbase_pos breakdown:
    #   URDF raw value  = -0.005 m  (right_hand_mount_joint origin in custom_eef_link)
    #   Physical flange correction = -0.010 m  (URDF 0.043 m → measured 0.033 m, short 10 mm;
    #                              link_eef -Z = custom_eef_link +X, so compensated in -X)
    #   Total            = -0.015 m
    #
    #   NOTE: the collision model applies no equivalent flange correction (it
    #   loads the collision URDF raw), so the same 10 mm residual is folded
    #   into TableCollisionConfig.soft_clearance_m (0.01 → 0.02).
    #
    T_eef_handbase_pos_xyz: tuple[float, float, float] = (-0.015, 0.0, 0.0)
    T_eef_handbase_quat_wxyz: tuple[float, float, float, float] = (0.707107, 0.0, 0.707107, 0.0)

    def __post_init__(self) -> None:
        if self.ethercat_slave_position < -1:
            raise ValueError("hand ethercat_slave_position must be -1 (unknown) or non-negative")
        limit_vectors = (
            self.mechanical_qpos_min_rad,
            self.mechanical_qpos_max_rad,
            self.qpos_min_rad,
            self.qpos_max_rad,
        )
        if len(self.home_qpos_deg) != 12 or any(len(values) != 12 for values in limit_vectors):
            raise ValueError("hand home and joint-limit defaults must have 12 elements")
        command_lower = np.asarray(self.qpos_min_rad, dtype=np.float64)
        command_upper = np.asarray(self.qpos_max_rad, dtype=np.float64)
        validate_hand_limit_nesting(
            command_lower,
            command_upper,
            self.mechanical_qpos_min_rad,
            self.mechanical_qpos_max_rad,
            _XHAND_RATED_QPOS_MIN_RAD,
            _XHAND_RATED_QPOS_MAX_RAD,
            label="hand",
        )
        home_rad = np.deg2rad(np.asarray(self.home_qpos_deg, dtype=np.float64))
        limit_tolerance_rad = 1e-9
        if (
            not np.all(np.isfinite(home_rad))
            or np.any(home_rad < command_lower - limit_tolerance_rad)
            or np.any(home_rad > command_upper + limit_tolerance_rad)
        ):
            raise ValueError("hand home_qpos_deg must be finite and within qpos limits")
        if self.max_delta_rad is not None and (not np.isfinite(self.max_delta_rad) or self.max_delta_rad <= 0):
            raise ValueError("hand max_delta_rad must be finite and > 0 when configured")
        if len(self.kp) != 12 or any(not isinstance(value, int) or value <= 0 for value in self.kp):
            raise ValueError("hand kp must contain twelve positive integer gains")
        if self.ki < 0 or self.kd < 0:
            raise ValueError("hand ki/kd must be non-negative")
        if len(self.tor_max_ma) != 12 or any(
            not isinstance(value, int) or value <= 0 for value in self.tor_max_ma
        ):
            raise ValueError("hand tor_max_ma must contain twelve positive integer mA limits")
        if not np.isfinite(self.loop_hz) or self.loop_hz <= 0:
            raise ValueError("hand loop_hz must be finite and positive")
        if not np.isfinite(self.home_command_ack_timeout_s) or self.home_command_ack_timeout_s <= 0:
            raise ValueError("hand home_command_ack_timeout_s must be finite and positive")
        if self.send_err_watchdog_count <= 0:
            raise ValueError("hand send_err_watchdog_count must be positive")
        if len(self.fingertip_link_names) != 5 or any(not name for name in self.fingertip_link_names):
            raise ValueError("hand fingertip_link_names must contain five non-empty names")
        transform = np.asarray(self.T_eef_handbase_pos_xyz + self.T_eef_handbase_quat_wxyz, dtype=np.float64)
        if transform.shape != (7,) or not np.all(np.isfinite(transform)):
            raise ValueError("hand base transform must contain seven finite values")
        if np.linalg.norm(transform[3:]) <= 1e-12:
            raise ValueError("hand base quaternion must be non-zero")


# Policy / teleop parameters


@dataclass(frozen=True)
class PolicyParams:
    """Policy / teleop parameters — single source of truth."""

    control_hz: float = 16.0
    coordinator_hz: float = 64.0
    action_prepare_timeout_s: float = 0.06
    action_apply_timeout_s: float = 0.75
    arm_state_stale_threshold_s: float = 0.5
    quit_save_timeout_s: float = 30.0
    post_teleop_timeout_s: float = 60.0

    # ── Cartesian EMA ──
    ema: EMAParams = field(default_factory=EMAParams)

    # ── VR mapping ──
    vr_mapping: VRMappingParams = field(default_factory=VRMappingParams)

    # ── Workspace bounds (arm base frame, meters) ──
    workspace: WorkspaceBounds = field(default_factory=WorkspaceBounds)

    # ── Recording ──
    recording_enabled: bool = True
    max_record_duration_s: float = 60.0
    # Quality label only: short, internally consistent episodes are published
    # with min_frames_met=False and left to downstream filtering policy.
    min_record_duration_s: float = 1.0
    episodes_dir: str = "episodes"

    # ── Diagnostics ──
    status_print_interval: int = 16  # status print interval (ticks)
    max_consecutive_errors: int = 10

    # ── VR teleoperation IK ──
    ik_max_pose_error_pos_m: float = 0.02
    ik_max_pose_error_rot_rad: float = np.deg2rad(5.0)
    ik_nullspace_step_rate_deg_s: float = 50.0

    # ── Contact-stall resync ──
    # This is not a table exclusion zone. Near the tabletop, a downward target
    # is resynchronised to measured pose only when the previous Mode-6 joint
    # target has accumulated error and the arm is no longer closing that error.
    contact_stall_enabled: bool = True
    contact_stall_table_context_height_m: float = 0.18
    contact_stall_min_downward_target_m: float = 0.003
    contact_stall_tracking_error_rad: float = 0.18
    contact_stall_max_closing_speed_rad_s: float = 0.05

    # ── Hand retargeting ──
    hand_enabled: bool = True
    hand_retargeting_type: str = "tag"
    hand_ramp_duration_s: float = 0.5  # smoothstep startup ramp, rate-independent
    begin_motion_gate_timeout_s: float = 0.35  # begin voice may delay motion by at most this long
    hand_disconnect_timeout_s: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError(f"control_hz={self.control_hz} must be > 0")
        if not np.isfinite(self.coordinator_hz) or self.coordinator_hz < self.control_hz:
            raise ValueError("coordinator_hz must be finite and >= control_hz")
        timing = (
            self.action_prepare_timeout_s,
            self.action_apply_timeout_s,
            self.arm_state_stale_threshold_s,
            self.quit_save_timeout_s,
            self.post_teleop_timeout_s,
        )
        if not all(np.isfinite(value) and value > 0 for value in timing):
            raise ValueError("policy action, freshness, and operator timeouts must be finite and positive")
        if not (0.0 <= self.ema.alpha_pos <= 1.0):
            raise ValueError(f"ema.alpha_pos={self.ema.alpha_pos} must be in [0, 1]")
        if not (0.0 <= self.ema.alpha_rot <= 1.0):
            raise ValueError(f"ema.alpha_rot={self.ema.alpha_rot} must be in [0, 1]")
        if not np.isfinite(self.hand_ramp_duration_s) or self.hand_ramp_duration_s < 0:
            raise ValueError("hand_ramp_duration_s must be finite and >= 0")
        if not np.isfinite(self.begin_motion_gate_timeout_s) or self.begin_motion_gate_timeout_s < 0:
            raise ValueError("begin_motion_gate_timeout_s must be finite and >= 0")
        if (
            not np.isfinite(self.max_record_duration_s)
            or not np.isfinite(self.min_record_duration_s)
            or self.max_record_duration_s <= 0
            or self.min_record_duration_s < 0
            or self.min_record_duration_s > self.max_record_duration_s
        ):
            raise ValueError("recording durations must be finite, ordered, and non-negative")
        if not self.episodes_dir or self.status_print_interval <= 0 or self.max_consecutive_errors <= 0:
            raise ValueError("policy output path and diagnostic intervals must be valid")
        if (
            not np.isfinite(self.ik_max_pose_error_pos_m)
            or not np.isfinite(self.ik_max_pose_error_rot_rad)
            or not np.isfinite(self.ik_nullspace_step_rate_deg_s)
            or self.ik_max_pose_error_pos_m <= 0
            or self.ik_max_pose_error_rot_rad <= 0
            or self.ik_nullspace_step_rate_deg_s <= 0
        ):
            raise ValueError("policy teleop IK limits must be finite and positive")
        if self.hand_retargeting_type not in {"tag", "dexpilot"}:
            raise ValueError("hand_retargeting_type must be 'tag' or 'dexpilot'")
        if not np.isfinite(self.hand_disconnect_timeout_s) or self.hand_disconnect_timeout_s <= 0:
            raise ValueError("hand_disconnect_timeout_s must be finite and positive")
        if not np.isfinite(self.contact_stall_table_context_height_m) or self.contact_stall_table_context_height_m <= 0:
            raise ValueError("contact_stall_table_context_height_m must be finite and > 0")
        if not np.isfinite(self.contact_stall_min_downward_target_m) or self.contact_stall_min_downward_target_m <= 0:
            raise ValueError("contact_stall_min_downward_target_m must be finite and > 0")
        if not np.isfinite(self.contact_stall_tracking_error_rad) or self.contact_stall_tracking_error_rad <= 0:
            raise ValueError("contact_stall_tracking_error_rad must be finite and > 0")
        if (
            not np.isfinite(self.contact_stall_max_closing_speed_rad_s)
            or self.contact_stall_max_closing_speed_rad_s < 0
        ):
            raise ValueError("contact_stall_max_closing_speed_rad_s must be finite and >= 0")


@dataclass(frozen=True)
class KeyboardTeleopParams:
    """Keyboard teleoperation parameters — single source of truth.

    At 30 Hz, the 8 mm translation step requests 0.24 m/s Cartesian target
    motion. Mode 6 following distance is expected at this rate and is exposed
    by the arm command timing diagnostics.
    """

    control_hz: float = 30.0
    delta_pos_m: float = 0.008  # 240 mm/s at 30 Hz
    delta_rpy_rad: float = 0.03  # 1.7 deg/frame, 51 deg/s at 30 Hz
    command_lookahead_frames: int = 5  # bounded 40 mm / 0.15 rad firmware-following lead
    workspace_command_margin_m: float = 0.005  # > ik_max_pose_error_pos_m (0.002) so a clipped target's FK stays in-bounds
    ik_max_pose_error_pos_m: float = 0.002
    ik_max_pose_error_rot_rad: float = np.deg2rad(2.0)
    status_interval_frames: int = 50
    idle_interval_frames: int = 150

    def __post_init__(self) -> None:
        numeric = (
            self.control_hz,
            self.delta_pos_m,
            self.delta_rpy_rad,
            self.workspace_command_margin_m,
            self.ik_max_pose_error_pos_m,
            self.ik_max_pose_error_rot_rad,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("keyboard teleop numeric parameters must be finite")
        if self.control_hz <= 0:
            raise ValueError(f"control_hz={self.control_hz} must be > 0")
        if self.delta_pos_m <= 0:
            raise ValueError(f"delta_pos_m={self.delta_pos_m} must be > 0")
        if self.delta_rpy_rad <= 0:
            raise ValueError(f"delta_rpy_rad={self.delta_rpy_rad} must be > 0")
        if self.command_lookahead_frames <= 0:
            raise ValueError("command_lookahead_frames must be positive")
        if self.workspace_command_margin_m < 0:
            raise ValueError("workspace_command_margin_m must be non-negative")
        if self.ik_max_pose_error_pos_m <= 0 or self.ik_max_pose_error_rot_rad <= 0:
            raise ValueError("keyboard IK pose-error limits must be > 0")
        if self.status_interval_frames <= 0 or self.idle_interval_frames <= 0:
            raise ValueError("keyboard status/idle intervals must be > 0")


# TAG retargeting parameters


@dataclass(frozen=True)
class TAGRetargetingParams:
    """TAG two-stage NLopt hand retargeting parameters (``retargeting_type="tag"``)."""

    # ── Finger length scaling ──
    robot_finger_lengths: tuple[float, ...] = (0.161, 0.208, 0.206, 0.204, 0.145)
    """XHand finger lengths (thumb..pinky, meters). Pinky set equal to human — adaptive_retargeting_xhand
    already handles pinky chain scaling (1.2-2.2x), so finger_scale for pinky must be 1.0 to avoid
    double-compensation."""

    human_finger_lengths: tuple[float, ...] = (0.13, 0.18, 0.19, 0.18, 0.145)
    """Human finger lengths from TAG glove calibration (thumb..pinky, meters)."""

    finger_scale_boost: float = 1.0
    """Multiplier on robot/human length ratio.  1.0 = no extra boost — VR landmarks
    are already at robot scale after MANO transform + adaptive_retargeting_xhand."""

    # ── Coordinate alignment: MANO → XHand URDF frame (Euler XYZ, rad) ──
    mano_to_urdf_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Identity because MANO and URDF both use +Z for finger extension."""

    # ── Stage 1: Global position matching (L-BFGS) ──
    smooth_weight: float = 0.02
    ftol_abs_s1: float = 1e-4
    maxeval_s1: int = 80

    # ── Stage 2: Pinch refinement (SLSQP) ──
    ftol_abs_s2: float = 1e-6
    maxeval_s2: int = 100
    pinch_base_weight: float = 2000.0
    pinch_start_dist_m: float = 0.030
    pinch_full_dist_m: float = 0.008
    pinch_ema_alpha: float = 0.4
    pinch_skip_threshold: float = 0.01
    reg_stage1_weight: float = 1.0
    reg_last_weight: float = 0.8

    def __post_init__(self) -> None:
        robot = np.asarray(self.robot_finger_lengths, dtype=np.float64)
        human = np.asarray(self.human_finger_lengths, dtype=np.float64)
        euler = np.asarray(self.mano_to_urdf_euler, dtype=np.float64)
        if robot.shape != (5,) or human.shape != (5,) or euler.shape != (3,):
            raise ValueError("TAG finger lengths/Euler alignment have invalid shape")
        if not np.all(np.isfinite(np.concatenate((robot, human, euler)))) or np.any(robot <= 0) or np.any(human <= 0):
            raise ValueError("TAG finger lengths/Euler alignment must be finite and lengths positive")
        positive = (
            self.finger_scale_boost,
            self.ftol_abs_s1,
            self.ftol_abs_s2,
            self.pinch_base_weight,
            self.pinch_start_dist_m,
            self.pinch_full_dist_m,
            self.reg_stage1_weight,
            self.reg_last_weight,
        )
        if not all(np.isfinite(value) and value > 0 for value in positive):
            raise ValueError("TAG scales, tolerances, distances, and regularization weights must be positive")
        if not np.isfinite(self.smooth_weight) or self.smooth_weight < 0:
            raise ValueError("TAG smooth_weight must be finite and non-negative")
        if self.maxeval_s1 <= 0 or self.maxeval_s2 <= 0:
            raise ValueError("TAG optimizer maxeval values must be positive")
        if self.pinch_full_dist_m > self.pinch_start_dist_m:
            raise ValueError("TAG pinch_full_dist_m must not exceed pinch_start_dist_m")
        if not (0.0 <= self.pinch_ema_alpha <= 1.0) or not (0.0 <= self.pinch_skip_threshold <= 1.0):
            raise ValueError("TAG pinch EMA/skip thresholds must be in [0, 1]")


# VR receiver parameters


@dataclass(frozen=True)
class VRParams:
    """VR receiver (HTS) parameters."""

    transport: str = "tcp_server"
    host: str = "0.0.0.0"
    port: int = 8000
    hand_side: str = "both"  # "both" needed for HeadFrame

    def __post_init__(self) -> None:
        if not self.transport or not self.host or not self.hand_side:
            raise ValueError("VR transport, host, and hand_side must be non-empty")
        if not (1 <= self.port <= 65535):
            raise ValueError("VR port must be in [1, 65535]")


# Safety parameters


@dataclass(frozen=True)
class SafetyParams:
    """Safety / heartbeat parameters — single source of truth."""

    heartbeat_timeouts: Mapping[str, float] = field(
        default_factory=lambda: {
            "arm": 1.0,
            "hand": 1.0,
            "policy": 1.0,
            "recorder": 2.0,
            "vr": 5.0,
            "camera": 2.0,
            "inference": 5.0,
        }
    )
    readiness_timeouts_s: Mapping[str, float] = field(
        default_factory=lambda: {
            "arm": 15.0,
            "hand": 15.0,
            "camera": 15.0,
            "recorder": 15.0,
            "policy": 120.0,
            "vr": 120.0,
            "inference": 120.0,
        }
    )
    shutdown_timeout_s: float = 65.0

    # Consecutive arm-health failure threshold (feedback/FK reads, arm_loop)
    max_consecutive_arm_health_failures: int = 30  # 1s @ 30Hz → FAULT

    # Supervisor check rate (Main)
    supervisor_hz: float = 10.0

    def __post_init__(self) -> None:
        if not self.heartbeat_timeouts or any(
            not name or not np.isfinite(value) or value <= 0 for name, value in self.heartbeat_timeouts.items()
        ):
            raise ValueError("heartbeat timeout names/values must be non-empty, finite, and > 0")
        if _HEARTBEAT_SUBSYSTEMS - self.heartbeat_timeouts.keys():
            raise ValueError("heartbeat_timeouts is missing a runtime subsystem")
        if not self.readiness_timeouts_s or any(
            not name or not np.isfinite(value) or value <= 0 for name, value in self.readiness_timeouts_s.items()
        ):
            raise ValueError("readiness timeout names/values must be non-empty, finite, and > 0")
        if _READINESS_SUBSYSTEMS - self.readiness_timeouts_s.keys():
            raise ValueError("readiness_timeouts_s is missing a runtime subsystem")
        if not np.isfinite(self.shutdown_timeout_s) or self.shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be finite and positive")
        if self.max_consecutive_arm_health_failures <= 0:
            raise ValueError(
                f"max_consecutive_arm_health_failures={self.max_consecutive_arm_health_failures} must be > 0"
            )
        if not np.isfinite(self.supervisor_hz) or self.supervisor_hz <= 0:
            raise ValueError(f"supervisor_hz={self.supervisor_hz} must be > 0")
        object.__setattr__(self, "heartbeat_timeouts", MappingProxyType(dict(self.heartbeat_timeouts)))
        object.__setattr__(self, "readiness_timeouts_s", MappingProxyType(dict(self.readiness_timeouts_s)))

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Keep the read-only mappings compatible with multiprocessing spawn."""
        return (
            type(self),
            (
                dict(self.heartbeat_timeouts),
                dict(self.readiness_timeouts_s),
                self.shutdown_timeout_s,
                self.max_consecutive_arm_health_failures,
                self.supervisor_hz,
            ),
        )


# Camera parameters


@dataclass(frozen=True)
class CameraParams:
    """Camera / RealSense parameters."""

    serial: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    align_mode: Literal["depth_to_color", "color_to_depth", "none"] = "depth_to_color"
    warmup_frames: int = 10
    max_frame_age_s: float = 0.25
    recording_stall_abort_s: float = 2.0
    # Frames skipped between consecutive reads that still count as normal.
    # 0 (default) derives the threshold from fps/publish_hz, so the intentional
    # publish-rate throttle does not read as a stall; set a positive value to
    # override.  FRAME_GAP is flagged only when the gap exceeds this threshold.
    frame_gap_stall_threshold: int = 0
    ring_maxlen: int = 5
    pointcloud_num_points: int = 2048
    writer_queue_size: int = 8

    @property
    def rgb_shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, 3)

    @property
    def depth_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def pointcloud_shape(self) -> tuple[int, int]:
        return (self.pointcloud_num_points, 6)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("camera width, height, and fps must be > 0")
        if self.align_mode not in ("depth_to_color", "color_to_depth", "none"):
            raise ValueError(f"unsupported camera align_mode={self.align_mode!r}")
        if self.warmup_frames < 0:
            raise ValueError("camera warmup_frames must be >= 0")
        if self.max_frame_age_s <= 0 or self.recording_stall_abort_s <= self.max_frame_age_s:
            raise ValueError("camera stall abort threshold must be greater than max frame age")
        if self.frame_gap_stall_threshold < 0:
            raise ValueError("camera frame_gap_stall_threshold must be >= 0 (0 = derive from fps/publish_hz)")
        if self.ring_maxlen <= 0 or self.pointcloud_num_points <= 0 or self.writer_queue_size <= 0:
            raise ValueError("camera ring, pointcloud, and writer capacities must be > 0")
        if self.serial is not None and not self.serial:
            raise ValueError("camera serial must be non-empty when configured")


# Module-level defaults

arm = ArmParams()
hand = HandParams()
policy = PolicyParams()
keyboard_teleop = KeyboardTeleopParams()
vr = VRParams()
safety = SafetyParams()
camera = CameraParams()
tag_retargeting = TAGRetargetingParams()
environment = EnvironmentConfig()
