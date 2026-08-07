"""Centralized defaults — single source of truth for all numeric constants.

Organized as dataclasses with snake_case fields, grouped into logical
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


@dataclass
class HomingParams:
    """Joint-space linear-interpolation homing parameters."""

    convergence_rad: float = 0.0174533  # ~1°
    step_interval_s: float = 0.04
    max_speed_deg_s: float = 30.0  # linear-interpolation fallback speed
    target_timeout_s: float = 0.2


@dataclass
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


@dataclass
class StaleDetectionParams:
    """Qpos freshness detection (driver board lockout guard)."""

    frame_count: int = 15  # frames @ 30Hz → 0.5s
    qpos_delta_rad: float = 1e-4


@dataclass
class EMAParams:
    """Cartesian-space EMA smoothing parameters (tuned at 16 Hz, dt=62.5ms).

    Tuning history: POS 0.8→0.65→0.6, ROT 0.4→0.3→0.25.
    """

    alpha_pos: float = 0.6  # τ≈65ms — moderate smoothing
    alpha_rot: float = 0.25  # τ≈223ms — heavy smoothing for wrist jitter


@dataclass
class VRMappingParams:
    """VR wrist → EEF mapping parameters."""

    pos_scale: float = 1.0
    rot_scale: float = 1.0
    max_delta_rot_rad: float = 3.0  # ~172° total-from-reset rotation cap
    stale_threshold_s: float = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Arm parameters (xArm7, 7-DOF)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
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

    # ── Environment ──
    table_z_surface_m: float = -0.008  # table top surface Z in arm-base frame (m)
    hand_safety_margin_m: float = 0.05  # conservative EEF-to-fingertip vertical distance (m)

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

    def __post_init__(self):
        if len(self.joint_limit_lower) != 7 or len(self.joint_limit_upper) != 7:
            raise ValueError("joint_limit_lower/upper must have 7 elements")
        if not all(lo <= hi for lo, hi in zip(self.joint_limit_lower, self.joint_limit_upper)):
            raise ValueError("joint_limit_lower must be <= joint_limit_upper")
        if not all(lo <= q <= hi for q, lo, hi in zip(self.home_qpos, self.joint_limit_lower, self.joint_limit_upper)):
            raise ValueError("home_qpos must be within joint limits")
        if not (0 < self.max_joint_velocity_deg_per_s <= 500):
            raise ValueError(f"max_joint_velocity_deg_per_s={self.max_joint_velocity_deg_per_s} out of range (0, 500]")
        if not (0 < self.max_joint_acceleration_deg_per_s2 <= 50000):
            raise ValueError(
                f"max_joint_acceleration_deg_per_s2={self.max_joint_acceleration_deg_per_s2} out of range (0, 50000]"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Hand parameters (XHand, 12-DOF)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class HandParams:
    """XHand hardware parameters — single source of truth."""

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
    #   URDF raw value  = -0.005 m  (right_hand_mount_joint origin in custom_eef_link)
    #   Physical flange correction = -0.010 m  (URDF 0.043 m → measured 0.033 m, short 10 mm;
    #                              link_eef -Z = custom_eef_link +X, so compensated in -X)
    #   Total            = -0.015 m
    #
    # Verified 2026-07-28: URDF-vs-simulation FK = 0.00 mm.
    T_eef_handbase_pos_xyz: tuple[float, float, float] = (-0.015, 0.0, 0.0)
    T_eef_handbase_quat_wxyz: tuple[float, float, float, float] = (0.707107, 0.0, 0.707107, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Policy / teleop parameters
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
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

    def __post_init__(self):
        if self.control_hz <= 0:
            raise ValueError(f"control_hz={self.control_hz} must be > 0")
        if not (0.0 <= self.ema.alpha_pos <= 1.0):
            raise ValueError(f"ema.alpha_pos={self.ema.alpha_pos} must be in [0, 1]")
        if not (0.0 <= self.ema.alpha_rot <= 1.0):
            raise ValueError(f"ema.alpha_rot={self.ema.alpha_rot} must be in [0, 1]")


# ═══════════════════════════════════════════════════════════════════════════════
# VR receiver parameters
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class VRParams:
    """VR receiver (HTS) parameters."""

    transport: str = "tcp_server"
    host: str = "0.0.0.0"
    port: int = 8000
    hand_side: str = "both"  # "both" needed for HeadFrame


# ═══════════════════════════════════════════════════════════════════════════════
# Safety parameters
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SafetyParams:
    """Safety / heartbeat parameters — single source of truth."""

    heartbeat_timeouts: dict[str, float] = field(
        default_factory=lambda: {
            "arm": 1.0,
            "hand": 1.0,
            "policy": 1.0,
            "vr": 5.0,
            "camera": 2.0,
        }
    )

    # Consecutive recovery escalation threshold (arm_loop)
    max_consecutive_recoveries: int = 30  # 1s @ 30Hz → FAULT

    # Supervisor check rate (Main)
    supervisor_hz: float = 10.0

    def __post_init__(self):
        if not all(v > 0 for v in self.heartbeat_timeouts.values()):
            raise ValueError("heartbeat timeouts must be > 0")
        if self.max_consecutive_recoveries <= 0:
            raise ValueError(f"max_consecutive_recoveries={self.max_consecutive_recoveries} must be > 0")
        if self.supervisor_hz <= 0:
            raise ValueError(f"supervisor_hz={self.supervisor_hz} must be > 0")


# ═══════════════════════════════════════════════════════════════════════════════
# Camera parameters
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
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


def load_config_json(path: str) -> None:
    """Override module-level config singletons from a JSON file.

    Mutates the existing singletons in-place (via ``object.__setattr__``)
    so that **all** references — including ``from defaults import arm``
    captured before this call — see the new values.

    Only **flat** (non-dataclass) fields are supported.  Nested dataclass
    fields (``ema``, ``workspace``, ``homing``, etc.) must be updated via
    Python code.

    Example JSON::

        {"arm": {"max_joint_velocity_deg_per_s": 150}, "policy": {"control_hz": 20}}

    Keys are singleton names (``arm``, ``hand``, ``policy``, ``vr``,
    ``safety``, ``camera``).  Values are flat field overrides.
    """
    import dataclasses
    import json
    import sys

    from dexmani_real.utils.log import get_logger

    _log = get_logger(__name__)

    with open(path) as f:
        data = json.load(f)
    mod = sys.modules[__name__]
    for name, overrides in data.items():
        original = getattr(mod, name)
        for k, v in overrides.items():
            if not hasattr(original, k):
                raise TypeError(f"'{type(original).__name__}' has no field '{k}'")
            current = getattr(original, k)
            if dataclasses.is_dataclass(current) and not isinstance(v, type(current)):
                raise TypeError(
                    f"Field '{k}' of '{name}' is a nested dataclass ({type(current).__name__}) — "
                    f"override not supported via JSON (edit defaults.py)"
                )
            setattr(original, k, v)
        _log.info("config: %s overridden with %s", name, list(overrides.keys()))
