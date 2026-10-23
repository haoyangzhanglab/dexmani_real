"""XHand parameters and nested rated, mechanical and command limits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dexmani_real.robot.model import XHAND_FINGERTIP_LINK_NAMES
from dexmani_real.utils.geometry import validate_unit_quaternion_wxyz
from dexmani_real.robot.xhand.limits import validate_hand_limit_nesting

XHAND_RATED_QPOS_MIN_RAD: tuple[float, ...] = (
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


XHAND_RATED_QPOS_MAX_RAD: tuple[float, ...] = (
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
    enabled: bool = True
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

    mechanical_qpos_min_rad: tuple[float, ...] = XHAND_RATED_QPOS_MIN_RAD
    mechanical_qpos_max_rad: tuple[float, ...] = XHAND_RATED_QPOS_MAX_RAD

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
    qpos_max_rad: tuple[float, ...] = XHAND_RATED_QPOS_MAX_RAD

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

    # Mode 3 SDK output scale: 0–400, with 400 = 100% output (not mA).
    tor_max: tuple[int, ...] = (
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

    # Maximum age of consumed joint/tactile feedback, measured with monotonic time.
    feedback_max_age_s: float = 4 / 30
    initial_state_timeout_s: float = 1.0  # Only acquisition after connect/session start/reset.

    home_timeout_s: float = 2.0
    home_tolerance_deg: float = 5.0

    fingertip_link_names: tuple[str, ...] = XHAND_FINGERTIP_LINK_NAMES
    # Real-pose FK uses a 10 mm thinner mount (-0.015 m); planning keeps the nominal -0.005 m URDF.
    T_eef_handbase_pos_xyz: tuple[float, float, float] = (-0.015, 0.0, 0.0)
    T_eef_handbase_quat_wxyz: tuple[float, float, float, float] = (
        0.707107,
        0.0,
        0.707107,
        0.0,
    )

    def validate(self) -> None:
        if self.ethercat_slave_position < -1:
            raise ValueError("hand ethercat_slave_position must be -1 (unknown) or non-negative")
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
        if not isinstance(self.baudrate, int) or self.baudrate <= 0:
            raise ValueError("hand baudrate must be a positive integer")
        if not isinstance(self.device_id, int) or self.device_id < 0:
            raise ValueError("hand device_id must be a non-negative integer")
        if not np.isfinite(self.rs485_post_open_settle_s) or self.rs485_post_open_settle_s < 0:
            raise ValueError("hand rs485_post_open_settle_s must be finite and non-negative")
        self.validate_home_pose()
        if len(self.kp) != 12 or any(not isinstance(value, int) or value <= 0 for value in self.kp):
            raise ValueError("hand kp must contain twelve positive integer gains")
        if self.ki < 0 or self.kd < 0:
            raise ValueError("hand ki/kd must be non-negative")
        if len(self.tor_max) != 12 or any(
            type(value) is not int or not 0 <= value <= 400 for value in self.tor_max
        ):
            raise ValueError(
                "hand tor_max must contain twelve integers in [0, 400] (Mode 3 output scale)"
            )
        for name in (
            "feedback_max_age_s",
            "initial_state_timeout_s",
            "home_timeout_s",
            "home_tolerance_deg",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"hand {name} must be finite and positive")
        if len(self.fingertip_link_names) != 5 or any(
            not name for name in self.fingertip_link_names
        ):
            raise ValueError("hand fingertip_link_names must contain five non-empty names")
        position = np.asarray(self.T_eef_handbase_pos_xyz, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("hand base position must be a finite (3,) vector")
        validate_unit_quaternion_wxyz(self.T_eef_handbase_quat_wxyz, name="hand base quaternion")

    def validate_home_pose(self) -> None:
        """Validate the configured collision posture, including when XHand is disconnected."""
        command_lower = np.asarray(self.qpos_min_rad, dtype=np.float64)
        command_upper = np.asarray(self.qpos_max_rad, dtype=np.float64)
        validate_hand_limit_nesting(
            command_lower,
            command_upper,
            self.mechanical_qpos_min_rad,
            self.mechanical_qpos_max_rad,
            XHAND_RATED_QPOS_MIN_RAD,
            XHAND_RATED_QPOS_MAX_RAD,
            label="hand",
        )
        home_rad = np.deg2rad(np.asarray(self.home_qpos_deg, dtype=np.float64))
        limit_tolerance_rad = 1e-9
        if (
            home_rad.shape != (12,)
            or not np.all(np.isfinite(home_rad))
            or np.any(home_rad < command_lower - limit_tolerance_rad)
            or np.any(home_rad > command_upper + limit_tolerance_rad)
        ):
            raise ValueError("hand home_qpos_deg must be finite and within qpos limits")
