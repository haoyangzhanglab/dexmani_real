"""Core robot data types — RobotState, RobotAction, ArmState, HandState, HandTactile."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
            raise ValueError(f"{cls_name}.{field_name} shape mismatch: expected {expected_shape}, got {arr.shape}")


@dataclass
class RobotState:
    """Complete robot state — assembled from arm_state_ring + hand_state_ring.

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
                ("hand_current", (12,)),
                ("hand_tactile_sum", (5, 3)),
                ("hand_tactile_force", (5, 120, 3)),
                ("hand_tactile_contact", (5,)),
                ("hand_tipboard_err", (12,)),
                ("hand_commboard_err", (12,)),
                ("hand_jointboard_err", (12,)),
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
    # Populated by the teleop pipeline (vr_teleop_policy); recorded so EE-space policies can train.
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


# ═══════════════════════════════════════════════════════════════════
# Per-process state types (Phase 2.9 — new architecture)
# ═══════════════════════════════════════════════════════════════════
#
# These dataclasses replace the monolithic RobotState for intra-process
# communication. Each process writes only the fields it owns.
# RobotState is retained for recording compatibility (Policy assembles
# it from ArmState + HandState + HandTactile).


@dataclass
class ArmState:
    """Arm process state — published to arm_state_ring every tick (~265 bytes).

    Matches ARM_STATE_DTYPE in shm/shared_storage.py.
    """

    qpos: np.ndarray  # (7,)  float64  rad
    qvel: np.ndarray  # (7,)  float64  rad/s
    tau: np.ndarray  # (7,)  float64  N·m
    eef_pos: np.ndarray  # (3,)  float64  m  (FK placeholder — Policy owns FK)
    eef_rot6d: np.ndarray  # (6,)  float64
    error_code: int
    connected: bool
    mode: int
    tracking_err: float
    timestamp: float

    @classmethod
    def from_ring(cls, data: np.ndarray) -> "ArmState":
        """Build from a SeqlockRingBuffer read (1-element structured array)."""
        r = data[0]
        return cls(
            qpos=np.asarray(r["qpos"], dtype=np.float64),
            qvel=np.asarray(r["qvel"], dtype=np.float64),
            tau=np.asarray(r["tau"], dtype=np.float64),
            eef_pos=np.asarray(r["eef_pos"], dtype=np.float64),
            eef_rot6d=np.asarray(r["eef_rot6d"], dtype=np.float64),
            error_code=int(r["error_code"]),
            connected=bool(r["connected"]),
            mode=int(r["mode"]),
            tracking_err=float(r["tracking_err"]),
            timestamp=float(r["timestamp"]),
        )


@dataclass
class HandState:
    """Hand process state — published to hand_state_ring every tick (~328 bytes).

    Does NOT include tactile_force — that's in HandTactile on a separate ring.
    Matches HAND_STATE_DTYPE in shm/shared_storage.py.
    """

    qpos: np.ndarray  # (12,) float64  rad
    current: np.ndarray  # (12,) float64  mA
    tactile_sum: np.ndarray  # (5,3) float64  N
    tactile_contact: np.ndarray  # (5,) bool
    error_state: bool
    connected: bool
    qpos_stale: bool
    timestamp: float

    @classmethod
    def from_ring(cls, data: np.ndarray) -> "HandState":
        """Build from a SeqlockRingBuffer read (1-element structured array)."""
        r = data[0]
        return cls(
            qpos=np.asarray(r["qpos"], dtype=np.float64),
            current=np.asarray(r["current"], dtype=np.float64),
            tactile_sum=np.asarray(r["tactile_sum"], dtype=np.float64),
            tactile_contact=np.asarray(r["tactile_contact"], dtype=bool),
            error_state=bool(r["error_state"]),
            connected=bool(r["connected"]),
            qpos_stale=bool(r["qpos_stale"]) if "qpos_stale" in r.dtype.names else False,
            timestamp=float(r["timestamp"]),
        )


@dataclass
class HandTactile:
    """Hand tactile force — published to hand_tactile_ring (sparse, ~14.4KB).

    Only written when tactile_contact[finger] is nonzero.
    """

    tactile_force: np.ndarray  # (5,120,3) float64  N

    @classmethod
    def from_ring(cls, data: np.ndarray) -> "HandTactile":
        """Build from a SeqlockRingBuffer read."""
        return cls(tactile_force=np.asarray(data[0]["tactile_force"], dtype=np.float64))
