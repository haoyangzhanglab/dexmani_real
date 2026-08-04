"""Centralized defaults — single source of truth for all numeric constants.

Organized as frozen dataclasses with snake_case fields, grouped into logical
sub-structures (homing, stale detection, workspace, EMA, etc.).

Module-level singletons (``arm``, ``hand``, ``policy``, ``vr``, ``safety``,
``camera``) provide ergonomic access::

    from dexmani_real.config.defaults import arm, hand, policy

    @dataclass
    class ArmLoopConfig:
        joint_max_speed_rad_per_s: float = field(
            default_factory=lambda: arm.max_joint_velocity_rad_per_s
        )
        homing_convergence_rad: float = field(default_factory=lambda: arm.homing.convergence_rad)

Conventions:
    - All angles in **radians** unless suffixed ``_deg``.
    - All rates in **Hz** (not periods).
    - Workspace bounds in **meters**, arm-base frame.
    - Unit suffixes: ``_deg``, ``_rad``, ``_deg_per_s``, ``_deg_per_s2``,
      ``_s`` (seconds), ``_hz`` (frequency), ``_m`` (meters), ``_count`` (dimensionless).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Shared sub-structures
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class HomingParams:
    """Joint-space linear-interpolation homing parameters."""

    convergence_rad: float = 0.0174533  # ~1°
    step_interval_s: float = 0.04
    max_speed_deg_s: float = 30.0  # linear-interpolation fallback speed
    target_timeout_s: float = 0.2


@dataclass(frozen=True)
class WorkspaceBounds:
    """EEF workspace bounds in arm-base frame (meters)."""

    x_min: float = 0.25
    x_max: float = 0.72
    y_min: float = -0.50
    y_max: float = 0.50
    z_min: float = 0.05
    z_max: float = 0.50

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
class StaleDetectionParams:
    """Qpos freshness detection (driver board lockout guard)."""

    frame_count: int = 15  # frames @ 30Hz → 0.5s
    qpos_delta_rad: float = 1e-4


@dataclass(frozen=True)
class EMAParams:
    """Cartesian-space EMA smoothing parameters (tuned at 16 Hz, dt=62.5ms).

    Tuning history: POS 0.8→0.65→0.6, ROT 0.4→0.3→0.25.
    """

    alpha_pos: float = 0.6  # τ≈65ms — moderate smoothing
    alpha_rot: float = 0.25  # τ≈223ms — heavy smoothing for wrist jitter


@dataclass(frozen=True)
class VRMappingParams:
    """VR wrist → EEF mapping parameters."""

    pos_scale: float = 1.0
    rot_scale: float = 1.0
    max_delta_rot_rad: float = 3.0  # ~172° total-from-reset rotation cap
    stale_threshold_s: float = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Arm parameters (xArm7, 7-DOF)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ArmParams:
    """xArm7 hardware parameters — single source of truth."""

    # ── Home position (rad) — neutral pose ──
    home_qpos: tuple[float, ...] = (
        -0.523599,  # J1: -30.0°
        -0.033161,  # J2: -1.9°
        0.0,        # J3: 0.0°
        0.235619,   # J4: 13.5°
        -3.141593,  # J5: -180.0°
        1.303762,   # J6: 74.7°
        0.0,        # J7: 0.0°
    )

    # ── Joint limits (rad) — mirrors xarm7 URDF ──
    # URDF source: assets/robots/xhand/xarm7_xhand_collision.urdf
    joint_limit_lower: tuple[float, ...] = (
        -6.28318530718, -2.059, -6.28318530718, -0.19198, -6.28318530718, -1.69297, -6.28318530718,
    )
    joint_limit_upper: tuple[float, ...] = (
        6.28318530718, 2.0944, 6.28318530718, 3.927, 6.28318530718, 3.14159265359, 6.28318530718,
    )

    # ── Dynamics (Mode 6 firmware) ──
    max_joint_velocity_deg_per_s: float = 120.0  # firmware trajectory speed
    max_joint_acceleration_deg_per_s2: float = 900.0  # firmware trajectory acceleration
    loop_hz: float = 30.0  # arm_loop servo rate

    # ── Connection ──
    ip: str = "192.168.1.111"

    # ── Safety ──
    tracking_error_warn_rad: float = 0.35  # diagnostic warning threshold
    collision_sensitivity: int = 1  # 0-5, 1 = most sensitive
    recoverable_errors: frozenset[int] = frozenset({22, 24, 31})  # C22/C24/C31

    # ── Homing ──
    homing: HomingParams = field(default_factory=HomingParams)

    # ── Derived ──
    @property
    def max_joint_velocity_rad_per_s(self) -> float:
        return float(np.deg2rad(self.max_joint_velocity_deg_per_s))

    @property
    def max_joint_acceleration_rad_per_s2(self) -> float:
        return float(np.deg2rad(self.max_joint_acceleration_deg_per_s2))


# ═══════════════════════════════════════════════════════════════════════════════
# Hand parameters (XHand, 12-DOF)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class HandParams:
    """XHand hardware parameters — single source of truth."""

    # ── Home position (deg) — open-hand neutral ──
    home_qpos_deg: tuple[float, ...] = (
        0.0, 80.66, 33.2, 0.0, 5.11, 5.0, 6.53, 5.0, 6.76, 5.0, 10.13, 5.0,
    )

    loop_hz: float = 30.0

    # ── Homing ──
    home_settle_timeout_s: float = 3.0
    home_settle_tol_rad: float = 0.06

    # ── Qpos freshness (driver board lockout guard) ──
    stale: StaleDetectionParams = field(default_factory=StaleDetectionParams)

    # ── Send-error watchdog ──
    send_err_watchdog_count: int = 30  # 1s @ 30Hz

    # ── Hand FK (fingertip positions in world frame) ──
    hand_urdf_path: str = ""
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
    #   URDF 原始值   = -0.005 m  (right_hand_mount_joint origin in custom_eef_link)
    #   物理 flange 修正 = -0.010 m  (URDF 0.043 m → 实测 0.033 m，短 10 mm;
    #                              link_eef -Z = custom_eef_link +X，故补在 -X)
    #   合计           = -0.015 m
    #
    # Verified 2026-07-28: URDF-vs-simulation FK = 0.00 mm.
    T_eef_handbase_pos_xyz: tuple[float, float, float] = (-0.015, 0.0, 0.0)
    T_eef_handbase_quat_wxyz: tuple[float, float, float, float] = (0.707107, 0.0, 0.707107, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Policy / teleop parameters
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PolicyParams:
    """Policy / teleop parameters — single source of truth."""

    control_hz: float = 16.0

    # ── Cartesian EMA ──
    ema: EMAParams = field(default_factory=EMAParams)

    # ── VR mapping ──
    vr_mapping: VRMappingParams = field(default_factory=VRMappingParams)

    # ── Workspace bounds (arm base frame, meters) ──
    workspace: WorkspaceBounds = field(default_factory=WorkspaceBounds)

    # ── Recording ──
    max_record_duration_s: float = 60.0
    min_record_duration_s: float = 1.0
    episodes_dir: str = "episodes"

    # ── Diagnostics ──
    status_print_interval: int = 16  # status print interval (ticks)
    max_consecutive_errors: int = 10

    # ── Hand retargeting ──
    hand_enabled: bool = True
    hand_retargeting_type: str = "dexpilot"
    hand_ramp_frame_count: int = 16  # smoothstep ramp (~1s @ 16Hz)
    hand_disconnect_timeout_s: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# VR receiver parameters
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class VRParams:
    """VR receiver (HTS) parameters."""

    transport: str = "tcp_server"
    host: str = "0.0.0.0"
    port: int = 8000
    hand_side: str = "both"  # "both" needed for HeadFrame


# ═══════════════════════════════════════════════════════════════════════════════
# Safety parameters
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SafetyParams:
    """Safety / heartbeat parameters — single source of truth."""

    heartbeat_timeouts: dict[str, float] = field(default_factory=lambda: {
        "arm": 1.0,
        "hand": 1.0,
        "policy": 1.0,
        "vr": 5.0,
        "camera": 2.0,
    })

    # Consecutive recovery escalation threshold (arm_loop)
    max_consecutive_recoveries: int = 30  # 1s @ 30Hz → FAULT

    # Supervisor check rate (Main)
    supervisor_hz: float = 10.0


# ═══════════════════════════════════════════════════════════════════════════════
# Camera parameters
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CameraParams:
    """Camera / RealSense parameters."""

    rgb_shape: tuple[int, int, int] = (480, 848, 3)
    depth_shape: tuple[int, int] = (480, 848)
    ring_maxlen: int = 5


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level singletons — ergonomic access
# ═══════════════════════════════════════════════════════════════════════════════

arm = ArmParams()
hand = HandParams()
policy = PolicyParams()
vr = VRParams()
safety = SafetyParams()
camera = CameraParams()


