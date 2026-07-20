"""Core robot data types — RobotState, RobotAction, RobotInterfaceConfig.

These types are shared across controller, recording, config, and other modules.
They are independent of the RobotInterface orchestration class to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.robot.xarm7 import XArm7Config
from dexmani_real.robot.xhand import XHandConfig

# Per-joint arm torque limits from URDF: J1-J2=50, J3-J5=30, J6-J7=20 Nm.
# Defined here (robot layer) because these are hardware properties, not teleop policy.
# Used by both teleop safety checks and RobotInterface path execution.
_ARM_TORQUE_LIMIT_NM = np.array([50.0, 50.0, 30.0, 30.0, 30.0, 20.0, 20.0])
_ARM_TEMP_LIMIT_C = 70.0  # joint temperature warning threshold (°C)

if TYPE_CHECKING:
    from dexmani_real.planning.collision_config import CollisionConfig
else:
    CollisionConfig = None  # resolved lazily when needed


def _validate_field_shapes(instance, specs: list[tuple[str, tuple]]) -> None:
    """Validate ndarray field shapes for a dataclass instance.

    For each (field_name, expected_shape) in specs:
      - retrieves the field value via getattr
      - skips None-valued fields
      - converts to np.ndarray via np.asarray
      - raises ValueError on shape mismatch

    The error message includes the class name and field name, matching the
    format originally used inline in RobotState/RobotAction.__post_init__.
    """
    cls_name = type(instance).__name__
    for field_name, expected_shape in specs:
        val = getattr(instance, field_name)
        if val is None:
            continue
        arr = np.asarray(val)
        if arr.shape != expected_shape:
            raise ValueError(f"{cls_name}.{field_name} shape mismatch: " f"expected {expected_shape}, got {arr.shape}")


@dataclass
class RobotState:
    """Complete robot state — from RobotInterface.get_state().

    All physical quantities annotated with units in comments.
    """

    # ── Arm joints ──
    arm_qpos: np.ndarray  # (7,)  float64  rad
    arm_qvel: np.ndarray  # (7,)  float64  rad/s
    arm_tau: np.ndarray  # (7,)  float64  N·m (motor current)

    # ── EEF pose (dual representation) ──
    eef_pos: np.ndarray  # (3,)  float64  m
    eef_quat_wxyz: np.ndarray  # (4,)  float64
    eef_rot6d: np.ndarray  # (6,)  float64

    # ── Hand joints ──
    hand_qpos: np.ndarray  # (12,) float64  rad

    # ── Tactile (ref: DexUMI — both combined force and raw array in default mode) ──
    hand_tactile_sum: np.ndarray  # (5,3)     float64  N — per-finger combined force
    hand_tactile_force: np.ndarray  # (5,120,3) float64  N — per-finger raw force array

    # ── Derived (chained FK) ──
    fingertip_pos: np.ndarray  # (5,3) float64  m (world frame)

    # ── Status ──
    arm_connected: bool
    hand_connected: bool
    timestamp: float  # seconds

    def __post_init__(self):
        _validate_field_shapes(
            self,
            [
                ("arm_qpos", (7,)),
                ("arm_qvel", (7,)),
                ("arm_tau", (7,)),
                ("eef_pos", (3,)),
                ("eef_quat_wxyz", (4,)),
                ("eef_rot6d", (6,)),
                ("hand_qpos", (12,)),
                ("hand_tactile_sum", (5, 3)),
                ("hand_tactile_force", (5, 120, 3)),
                ("fingertip_pos", (5, 3)),
            ],
        )


@dataclass
class RobotAction:
    """Action command sent to hardware.

    arm_qpos_cmd / hand_qpos_cmd: final command after joint limit + delta limit.
    """

    arm_qpos_cmd: np.ndarray  # (7,)  float64  rad
    hand_qpos_cmd: np.ndarray  # (12,) float64  rad

    # ── Intent (pre-IK Cartesian EEF target the arm command tracks) ──
    # Populated by TeleopPipeline; recorded so EE-space policies can train.
    target_eef_pos: np.ndarray | None = field(default=None)  # (3,)  float64  m
    target_eef_rot6d: np.ndarray | None = field(default=None)  # (6,)  float64

    def __post_init__(self):
        _validate_field_shapes(
            self,
            [
                ("arm_qpos_cmd", (7,)),
                ("hand_qpos_cmd", (12,)),
                ("target_eef_pos", (3,)),
                ("target_eef_rot6d", (6,)),
            ],
        )


@dataclass
class RobotInterfaceConfig:
    arm: XArm7Config = field(default_factory=XArm7Config)
    hand: XHandConfig = field(default_factory=XHandConfig)

    # Workspace safety
    workspace_bounds: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.28, 0.72],  # x [min, max] m
                [-0.45, 0.45],  # y [min, max] m
                [0.05, 0.5],  # z [min, max] m
            ],
            dtype=np.float64,
        )
    )

    # Hand FK
    hand_urdf_path: str = ""
    fingertip_link_names: list[str] = field(default_factory=list)

    # Static transform from EEF to hand base
    T_eef_handbase_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    T_eef_handbase_quat_wxyz: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )

    # Unified collision configuration.
    # When set, desk safety uses geometric FK (FingertipDeskSafety).
    collision: CollisionConfig | None = None
