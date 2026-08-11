"""Core robot data types — RobotState, RobotAction, ArmState, HandState, HandTactile."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dexmani_real.ipc.schema import (
    ARM_JOINT_SHAPE,
    HAND_CONTACT_SHAPE,
    HAND_FINGERTIP_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)


def _validate_field_shapes(instance: object, specs: list[tuple[str, tuple[int, ...]]]) -> None:
    """Validate present NumPy-compatible fields against expected shapes."""
    cls_name = type(instance).__name__
    for field_name, expected_shape in specs:
        val = getattr(instance, field_name)
        if val is None:
            continue
        arr = np.asarray(val)
        if arr.shape != expected_shape:
            raise ValueError(f"{cls_name}.{field_name} shape mismatch: expected {expected_shape}, got {arr.shape}")


@dataclass
class RobotState:
    """Recording state assembled from the arm, hand, and tactile rings."""

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

    # ── Tactile ──
    hand_tactile_sum: np.ndarray  # (5,3) float64 — SDK-scaled, physical unit unverified
    hand_tactile_force: np.ndarray  # (5,120,3) float64 — SDK-scaled, physical unit unverified
    hand_tactile_contact: np.ndarray  # (5,) bool — per-finger contact detection (from detect_contact)
    hand_tipboard_err: np.ndarray  # (12,) int32 — tip board error registers per joint
    hand_commboard_err: np.ndarray  # (12,) int32 — comm board error registers per joint
    hand_jointboard_err: np.ndarray  # (12,) int32 — joint motor-driver board error registers per joint
    hand_qpos_stale: bool  # True when hand_qpos frozen despite active cmd (driver board lockout)

    # ── Derived (chained FK) ──
    fingertip_pos: np.ndarray  # (5,3) float64  m (world frame)

    # ── Status ──
    arm_connected: bool
    hand_connected: bool
    timestamp: float  # seconds

    # ── Hand motor current (optional, for safety gating) ──
    hand_current: np.ndarray | None = None  # (12,) float64 mA — per-motor current

    hand_error_state: bool = False  # True when hand reports hardware errors (board faults)

    arm_last_cmd_seq: int = 0
    arm_last_cmd_queue_latency_s: float = 0.0  # producer -> arm queue receive
    arm_last_cmd_apply_latency_s: float = 0.0  # producer -> successful SDK return
    arm_last_cmd_sdk_duration_s: float = 0.0  # duration of set_servo_angle()
    arm_last_cmd_is_hold: bool = False  # release-edge/hold command rather than motion intent

    def __post_init__(self):
        _validate_field_shapes(
            self,
            [
                ("arm_qpos", ARM_JOINT_SHAPE),
                ("arm_qvel", ARM_JOINT_SHAPE),
                ("arm_tau", ARM_JOINT_SHAPE),
                ("eef_pos", (3,)),
                ("eef_quat_wxyz", (4,)),
                ("eef_rot6d", (6,)),
                ("hand_qpos", HAND_JOINT_SHAPE),
                ("hand_current", HAND_JOINT_SHAPE),
                ("hand_tactile_sum", HAND_TACTILE_SUM_SHAPE),
                ("hand_tactile_force", HAND_TACTILE_FORCE_SHAPE),
                ("hand_tactile_contact", HAND_CONTACT_SHAPE),
                ("hand_tipboard_err", HAND_JOINT_SHAPE),
                ("hand_commboard_err", HAND_JOINT_SHAPE),
                ("hand_jointboard_err", HAND_JOINT_SHAPE),
                ("fingertip_pos", HAND_FINGERTIP_SHAPE),
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
    # Populated by the teleop loop; recorded so EE-space policies can train.
    target_eef_pos: np.ndarray | None = field(default=None)  # (3,)  float64  m
    target_eef_rot6d: np.ndarray | None = field(default=None)  # (6,)  float64

    def __post_init__(self):
        _validate_field_shapes(
            self,
            [
                ("arm_qpos_cmd", ARM_JOINT_SHAPE),
                ("hand_qpos_cmd", HAND_JOINT_SHAPE),
                ("target_eef_pos", (3,)),
                ("target_eef_rot6d", (6,)),
            ],
        )


@dataclass
class ArmState:
    """Documentation view of ``ipc.schema.ARM_STATE_DTYPE``."""

    qpos: np.ndarray  # (7,)  float64  rad
    qvel: np.ndarray  # (7,)  float64  rad/s
    tau: np.ndarray  # (7,)  float64  N·m
    eef_pos: np.ndarray  # (3,)  float64  m  (FK computed by arm_loop via Pinocchio ArmFK)
    eef_rot6d: np.ndarray  # (6,)  float64
    error_code: int
    connected: bool
    mode: int
    tracking_err: float
    last_cmd_seq: int
    last_cmd_created_s: float  # producer time.monotonic(), seconds
    last_cmd_received_s: float  # arm worker time.monotonic(), seconds
    last_cmd_applied_s: float  # successful SDK return, time.monotonic(), seconds
    last_cmd_queue_latency_s: float
    last_cmd_apply_latency_s: float
    last_cmd_sdk_duration_s: float
    last_cmd_is_hold: bool
    source_monotonic_ns: int
    publish_monotonic_ns: int
    state_valid: bool
    timestamp: float


@dataclass
class HandState:
    """Documentation view of ``ipc.schema.HAND_STATE_DTYPE``."""

    qpos: np.ndarray  # (12,) float64  rad
    current: np.ndarray  # (12,) float64  mA
    tactile_sum: np.ndarray  # (5,3) float64 — SDK-scaled, physical unit unverified
    tactile_contact: np.ndarray  # (5,) bool
    error_state: bool
    connected: bool
    qpos_stale: bool
    commboard_err: np.ndarray  # (12,) int32
    jointboard_err: np.ndarray  # (12,) int32
    tipboard_err: np.ndarray  # (12,) int32
    source_monotonic_ns: int
    publish_monotonic_ns: int
    state_valid: bool
    send_healthy: bool
    read_healthy: bool
    timestamp: float


@dataclass
class HandTactile:
    """Documentation view of ``ipc.schema.HAND_TACTILE_DTYPE``."""

    tactile_force: np.ndarray  # (5,120,3) float64 — SDK-scaled, physical unit unverified
    source_monotonic_ns: int
    fresh: bool
    calibrated: bool
    unit_code: int
