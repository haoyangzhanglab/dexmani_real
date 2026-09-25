"""Hardware parameters; angles in radians unless named _deg, lengths in metres."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dexmani_real.robot.model import (
    ARM_JOINT_SHAPE,
    XARM7_HARD_LOWER,
    XARM7_HARD_UPPER,
    XHAND_FINGERTIP_LINK_NAMES,
)
from dexmani_real.utils.limits import validate_hand_limit_nesting

_OBSERVATION_MAX_AGE_PERIODS = 4


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
class HomingParams:
    """Firmware-planned execution parameters for validated home milestones."""

    convergence_rad: float = 0.002618  # final canonical-home tolerance
    step_interval_s: float = 0.04  # controller-state polling interval
    max_speed_deg_s: float = (
        30.0  # conservative Mode 0 joint speed; hardware validation required before tuning
    )
    target_timeout_s: float = 0.5  # settling allowance added after distance/speed timing
    velocity_convergence_rad_s: float = 0.03
    dwell_s: float = 0.30
    convergence_timeout_s: float = 15.0
    request_queue_timeout_s: float = 0.2
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
            self.request_queue_timeout_s,
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
    """xArm7 hardware parameters — single source of truth."""

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
    loop_hz: float = 30.0  # worker command-admission / feedback-update rate

    ip: str = "192.168.1.111"

    table_z_surface_m: float = 0.022
    hand_safety_margin_m: float = 0.05

    collision_sensitivity: int = 1

    tcp_load_mass_kg: float = 1.1
    tcp_load_cog_mm: tuple[float, float, float] = (16.3, 7.9, 109.5)

    homing: HomingParams = field(default_factory=HomingParams)

    @property
    def feedback_max_age_s(self) -> float:
        return _OBSERVATION_MAX_AGE_PERIODS / self.loop_hz

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
        if not np.isfinite(self.loop_hz) or self.loop_hz <= 0:
            raise ValueError("arm loop_hz must be finite and positive")
        if not np.isfinite(self.table_z_surface_m):
            raise ValueError("table_z_surface_m must be finite")
        if not np.isfinite(self.hand_safety_margin_m) or self.hand_safety_margin_m < 0:
            raise ValueError("hand_safety_margin_m must be finite and non-negative")
        if not (0 <= self.collision_sensitivity <= 5):
            raise ValueError(
                f"collision_sensitivity={self.collision_sensitivity} out of range [0, 5]"
            )
        # Mode 6 firmware clamps speed to [0.0001, π] rad/s and acceleration to
        # [0.01, 20] rad/s²; validate in radians before sending.
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
        if not np.isfinite(self.tcp_load_mass_kg) or self.tcp_load_mass_kg <= 0:
            raise ValueError("tcp_load_mass_kg must be finite and positive")
        cog = np.asarray(self.tcp_load_cog_mm, dtype=np.float64)
        if cog.shape != (3,) or not np.all(np.isfinite(cog)):
            raise ValueError("tcp_load_cog_mm must be a finite (3,) vector")


@dataclass(frozen=True)
class HandParams:
    """XHand hardware parameters — single source of truth."""

    ethercat_slave_position: int = -1

    comm_type: str = "serial"
    device_name: str | None = "/dev/ttyUSB0"
    baudrate: int = 3_000_000
    device_id: int = 0
    rs485_post_open_settle_s: float = 1.0

    home_qpos_deg: tuple[float, ...] = (
        30.0,
        55.33,
        10.0,
        0.17,
        1.08,
        5.0,
        1.25,
        5.0,
        1.33,
        5.0,
        1.33,
        5.0,
    )

    mechanical_qpos_min_rad: tuple[float, ...] = _XHAND_RATED_QPOS_MIN_RAD
    mechanical_qpos_max_rad: tuple[float, ...] = _XHAND_RATED_QPOS_MAX_RAD

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

    tor_max_ma: tuple[int, ...] = (
        360,
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

    loop_hz: float = 30.0  # worker command-admission / feedback-update rate
    state_read_failure_timeout_s: float = 1.0

    home_timeout_s: float = 2.0
    home_tolerance_deg: float = 5.0

    fingertip_link_names: tuple[str, ...] = XHAND_FINGERTIP_LINK_NAMES
    # Real adapter is 10 mm thinner than nominal CAD/URDF (-0.005 m mount).
    T_eef_handbase_pos_xyz: tuple[float, float, float] = (-0.015, 0.0, 0.0)
    T_eef_handbase_quat_wxyz: tuple[float, float, float, float] = (
        0.707107,
        0.0,
        0.707107,
        0.0,
    )

    @property
    def feedback_max_age_s(self) -> float:
        return _OBSERVATION_MAX_AGE_PERIODS / self.loop_hz

    def validate(self) -> None:
        if self.ethercat_slave_position < -1:
            raise ValueError("hand ethercat_slave_position must be -1 (unknown) or non-negative")
        if self.comm_type not in ("ethercat", "serial"):
            raise ValueError("hand comm_type must be 'ethercat' or 'serial'")
        if self.device_name is not None and not isinstance(self.device_name, str):
            raise ValueError("hand device_name must be a string or null")
        if not isinstance(self.baudrate, int) or self.baudrate <= 0:
            raise ValueError("hand baudrate must be a positive integer")
        if not isinstance(self.device_id, int) or self.device_id < 0:
            raise ValueError("hand device_id must be a non-negative integer")
        if not np.isfinite(self.rs485_post_open_settle_s) or self.rs485_post_open_settle_s < 0:
            raise ValueError("hand rs485_post_open_settle_s must be finite and non-negative")
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
        for label, lower, upper in (
            ("command", command_lower, command_upper),
            (
                "mechanical",
                np.asarray(self.mechanical_qpos_min_rad, dtype=np.float64),
                np.asarray(self.mechanical_qpos_max_rad, dtype=np.float64),
            ),
        ):
            if np.any(lower >= upper):
                raise ValueError(f"hand {label} limits must have lower < upper")
        home_rad = np.deg2rad(np.asarray(self.home_qpos_deg, dtype=np.float64))
        limit_tolerance_rad = 1e-9
        if (
            not np.all(np.isfinite(home_rad))
            or np.any(home_rad < command_lower - limit_tolerance_rad)
            or np.any(home_rad > command_upper + limit_tolerance_rad)
        ):
            raise ValueError("hand home_qpos_deg must be finite and within qpos limits")
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
        if (
            not np.isfinite(self.state_read_failure_timeout_s)
            or self.state_read_failure_timeout_s <= 0
        ):
            raise ValueError("hand state_read_failure_timeout_s must be finite and positive")
        if not np.isfinite(self.home_timeout_s) or self.home_timeout_s <= 0:
            raise ValueError("hand home_timeout_s must be finite and positive")
        if not np.isfinite(self.home_tolerance_deg) or self.home_tolerance_deg <= 0:
            raise ValueError("hand home_tolerance_deg must be finite and positive")
        if len(self.fingertip_link_names) != 5 or any(
            not name for name in self.fingertip_link_names
        ):
            raise ValueError("hand fingertip_link_names must contain five non-empty names")
        transform = np.asarray(
            self.T_eef_handbase_pos_xyz + self.T_eef_handbase_quat_wxyz,
            dtype=np.float64,
        )
        if transform.shape != (7,) or not np.all(np.isfinite(transform)):
            raise ValueError("hand base transform must contain seven finite values")
        if np.linalg.norm(transform[3:]) <= 1e-12:
            raise ValueError("hand base quaternion must be non-zero")


@dataclass(frozen=True)
class VRParams:
    """VR receiver (HTS) parameters."""

    transport: str = "tcp_server"
    host: str = "0.0.0.0"
    port: int = 8000
    hand_side: str = "both"  # "both" needed for HeadFrame

    def validate(self) -> None:
        if not self.transport or not self.host or not self.hand_side:
            raise ValueError("VR transport, host, and hand_side must be non-empty")
        if not (1 <= self.port <= 65535):
            raise ValueError("VR port must be in [1, 65535]")


@dataclass(frozen=True)
class CameraParams:
    """Camera / RealSense parameters."""

    serial: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    warmup_frames: int = 10
    # Missing frames beyond this interval fail the camera workflow.
    source_stall_timeout_s: float = 2.0
    l515_visual_preset: int = 5
    l515_confidence_threshold: int | None = None
    # Librealsense-owned frameset queue. Keep only a small scheduling cushion:
    # control freshness is more important than retaining historical frames.
    frame_queue_capacity: int = 2
    ring_maxlen: int = 5

    @property
    def max_frame_age_s(self) -> float:
        return _OBSERVATION_MAX_AGE_PERIODS / self.fps

    @property
    def rgb_shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, 3)

    @property
    def depth_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0 or not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("camera width, height, and fps must be > 0")
        if self.warmup_frames < 0:
            raise ValueError("camera warmup_frames must be >= 0")
        if (
            not np.isfinite(self.source_stall_timeout_s)
            or self.source_stall_timeout_s <= self.max_frame_age_s
        ):
            raise ValueError(
                "camera source_stall_timeout_s must be finite and greater than derived max frame age"
            )
        if (
            not isinstance(self.l515_visual_preset, int)
            or isinstance(self.l515_visual_preset, bool)
            or not 0 <= self.l515_visual_preset <= 5
        ):
            raise ValueError("camera l515_visual_preset must be an integer in [0, 5]")
        if self.l515_confidence_threshold is not None and (
            not isinstance(self.l515_confidence_threshold, int)
            or isinstance(self.l515_confidence_threshold, bool)
            or not 0 <= self.l515_confidence_threshold <= 3
        ):
            raise ValueError("camera l515_confidence_threshold must be in [0, 3] or null")
        if self.frame_queue_capacity <= 0 or self.ring_maxlen <= 0:
            raise ValueError("camera ring and writer capacities must be > 0")
        if self.serial is not None and not self.serial:
            raise ValueError("camera serial must be non-empty when configured")
