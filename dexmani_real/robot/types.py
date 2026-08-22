"""Core robot data types — RobotState, RobotAction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dexmani_real.utils.schema import (
    ARM_JOINT_SHAPE,
    HAND_CONTACT_SHAPE,
    HAND_FINGERTIP_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)


def _validate_field_shapes(
    instance: object, specs: list[tuple[str, tuple[int, ...]]]
) -> None:
    """Validate present NumPy-compatible fields against expected shapes."""
    cls_name = type(instance).__name__
    for field_name, expected_shape in specs:
        val = getattr(instance, field_name)
        if val is None:
            continue
        arr = np.asarray(val)
        if arr.shape != expected_shape:
            raise ValueError(
                f"{cls_name}.{field_name} shape mismatch: expected {expected_shape}, got {arr.shape}"
            )


@dataclass
class RobotState:
    """Recording state assembled from the arm, hand, and tactile rings."""

    arm_qpos: np.ndarray  # (7,)  float64  rad
    arm_qvel: np.ndarray  # (7,)  float64  rad/s
    # xArm SDK current-estimated effort; precise SI unit is unverified.
    arm_tau: np.ndarray  # (7,) float64

    eef_pos: np.ndarray  # (3,)  float64  m
    eef_quat_wxyz: np.ndarray  # (4,)  float64
    eef_rot6d: np.ndarray  # (6,)  float64

    hand_qpos: np.ndarray  # (12,) float64  rad

    hand_tactile_sum: np.ndarray  # (5,3) float64 — SDK-scaled, physical unit unverified
    hand_tactile_force: (
        np.ndarray
    )  # (5,120,3) float64 — SDK-scaled, physical unit unverified
    hand_tactile_contact: (
        np.ndarray
    )  # (5,) bool — per-finger contact detection (from detect_contact)
    hand_tipboard_err: np.ndarray  # (12,) int32 — tip board error registers per joint
    hand_commboard_err: np.ndarray  # (12,) int32 — comm board error registers per joint
    hand_jointboard_err: (
        np.ndarray
    )  # (12,) int32 — joint motor-driver board error registers per joint
    # Recorder provenance for a sample held after a failed hand read.
    hand_qpos_stale: bool

    fingertip_pos: np.ndarray  # (5,3) float64 m (arm-base frame)

    arm_connected: bool
    hand_connected: bool
    timestamp: float  # seconds

    hand_current: np.ndarray | None = None  # (12,) float64 mA — per-motor current

    arm_last_cmd_seq: int = 0
    # Safety/IK fallback endpoint marker; ordinary command quiescence publishes
    # no endpoint.
    arm_last_cmd_is_hold: bool = False

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

    arm_qpos_cmd / hand_qpos_cmd: final command after joint-limit and bounds validation.
    """

    arm_qpos_cmd: np.ndarray  # (7,)  float64  rad
    hand_qpos_cmd: np.ndarray  # (12,) float64  rad

    # Pre-IK Cartesian intent.
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
