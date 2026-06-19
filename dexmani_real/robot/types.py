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

if TYPE_CHECKING:
    from dexmani_real.planning.collision_config import CollisionConfig


@dataclass
class RobotState:
    """Complete robot state — from RobotInterface.get_state().

    All physical quantities annotated with units in comments.
    """

    # ── Arm joints ──
    arm_qpos: np.ndarray          # (7,)  float64  rad
    arm_qvel: np.ndarray          # (7,)  float64  rad/s
    arm_tau: np.ndarray           # (7,)  float64  N·m (motor current)

    # ── EEF pose (dual representation) ──
    eef_pos: np.ndarray           # (3,)  float64  m
    eef_quat_wxyz: np.ndarray     # (4,)  float64
    eef_rot6d: np.ndarray         # (6,)  float64

    # ── Hand joints ──
    hand_qpos: np.ndarray         # (12,) float64  rad
    hand_current: np.ndarray      # (12,) float64  mA

    # ── Tactile ──
    hand_tactile_sum: np.ndarray  # (5,3) float64  N
    hand_tactile_raw: np.ndarray  # (5,120,3) float64

    hand_temperature: np.ndarray  # (12,) float64  °C

    # ── Derived (chained FK) ──
    fingertip_pos: np.ndarray     # (5,3) float64  m (world frame)

    # ── Status ──
    arm_connected: bool
    hand_connected: bool
    hand_error: bool
    timestamp: float              # seconds

    def __post_init__(self):
        for field_name, expected_shape in [
            ("arm_qpos", (7,)),
            ("arm_qvel", (7,)),
            ("arm_tau", (7,)),
            ("eef_pos", (3,)),
            ("eef_quat_wxyz", (4,)),
            ("eef_rot6d", (6,)),
            ("hand_qpos", (12,)),
            ("hand_current", (12,)),
            ("hand_tactile_sum", (5, 3)),
            ("hand_tactile_raw", (5, 120, 3)),
            ("hand_temperature", (12,)),
            ("fingertip_pos", (5, 3)),
        ]:
            val = getattr(self, field_name)
            if val is not None:
                arr = np.asarray(val)
                if arr.shape != expected_shape:
                    raise ValueError(
                        f"RobotState.{field_name} shape mismatch: "
                        f"expected {expected_shape}, got {arr.shape}"
                    )


@dataclass
class RobotAction:
    """Action command sent to hardware.

    arm_qpos_cmd / hand_qpos_cmd: final command after joint limit + delta limit.
    target_eef_pos / target_eef_rot6d: EEF target before IK (optional).
    """

    arm_qpos_cmd: np.ndarray             # (7,)  float64  rad
    hand_qpos_cmd: np.ndarray            # (12,) float64  rad

    target_eef_pos: np.ndarray | None = None    # (3,)  float64  m
    target_eef_rot6d: np.ndarray | None = None  # (6,)  float64

    def __post_init__(self):
        for field_name, expected_shape in [
            ("arm_qpos_cmd", (7,)),
            ("hand_qpos_cmd", (12,)),
        ]:
            val = getattr(self, field_name)
            if val is not None:
                arr = np.asarray(val)
                if arr.shape != expected_shape:
                    raise ValueError(
                        f"RobotAction.{field_name} shape mismatch: "
                        f"expected {expected_shape}, got {arr.shape}"
                    )


@dataclass
class RobotInterfaceConfig:
    arm: XArm7Config = field(default_factory=XArm7Config)
    hand: XHandConfig = field(default_factory=XHandConfig)

    # Workspace safety
    workspace_bounds: np.ndarray = field(
        default_factory=lambda: np.array([
            [0.28, 0.72],  # x [min, max] m
            [-0.45, 0.45],  # y [min, max] m
            [0.05, 0.5],   # z [min, max] m
        ], dtype=np.float64)
    )

    # Environment collision (table at z=0.0 m in world frame)
    add_table_collision: bool = True
    table_z_world: float = 0.0      # table surface height (world frame, meters)
    table_margin_xy: float = 0.15   # extra margin beyond workspace bounds
    table_layers: int = 5           # number of z-layers for solid volume
    table_layer_spacing: float = 0.01  # spacing between z-layers (meters)
    table_xy_resolution: float = 0.02  # point spacing on each layer (meters)
    table_x_min_clearance: float = 0.15  # minimum x distance from origin (protect robot base)

    # Hand FK
    hand_urdf_path: str = ""
    fingertip_link_names: list[str] = field(default_factory=list)

    # Static transform from EEF to hand base
    T_eef_handbase_pos: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    T_eef_handbase_quat_wxyz: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )

    # Unified collision configuration (preferred over individual table_* fields).
    # When set, desk safety uses geometric FK (FingertipDeskSafety).
    # The legacy table_* fields (add_table_collision, table_z_world, etc.) are
    # still supported for backward compatibility but deprecated in favor of
    # CollisionConfig.
    collision: CollisionConfig | None = None
