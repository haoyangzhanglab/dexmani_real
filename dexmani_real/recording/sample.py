"""Recorded episode state and action samples."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from dexmani_real.ipc.schema import (
    ARM_JOINT_SHAPE,
    HAND_CONTACT_SHAPE,
    HAND_FINGERTIP_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
    nan_array,
)
from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
from dexmani_real.planning.kinematics.fingertip import (
    compute_fingertip_points_xarm_base,
)
from dexmani_real.planning.kinematics.hand_fk import HandKinematics
from dexmani_real.planning.kinematics.pose import rot6d_to_quat_wxyz


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
class EpisodeState:
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
    # Safety/IK fallback endpoint marker; an ordinary pause boundary publishes
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
class EpisodeAction:
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


def build_episode_state(
    arm_state: np.ndarray | None,
    hand_state: np.ndarray | None,
    hand_tactile: np.ndarray | None = None,
    hand_fk: HandKinematics | None = None,
    handbase_position_eef_m: np.ndarray | None = None,
    handbase_quat_eef_wxyz: np.ndarray | None = None,
    timestamp_s: float | None = None,
) -> EpisodeState:
    """Assemble one generic recording state from arm, hand, and tactile rings.

    The returned values retain the raw episode convention: all geometry is in
    the xArm-base frame and feedback validity remains represented by the
    original fields/sentinels.  Both teleoperation and policy evaluation own
    their action semantics; this helper owns only state assembly.
    """
    if arm_state is not None:
        arm = arm_state[0]
        arm_qpos = np.asarray(arm["qpos"], dtype=np.float64)
        arm_qvel = np.asarray(arm["qvel"], dtype=np.float64)
        arm_tau = np.asarray(arm["tau"], dtype=np.float64)
        arm_state_valid = bool(arm["state_valid"])
        if arm_state_valid:
            eef_pos, eef_rot6d = make_arm_fk().compute(arm_qpos)
        else:
            eef_pos = nan_array(3)
            eef_rot6d = nan_array(6)
        arm_connected = bool(arm["connected"])
        arm_last_cmd_seq = int(arm["last_cmd_seq"])
        arm_last_cmd_is_hold = bool(arm["last_cmd_is_hold"])
    else:
        arm_qpos = nan_array(ARM_JOINT_SHAPE)
        arm_qvel = nan_array(ARM_JOINT_SHAPE)
        arm_tau = nan_array(ARM_JOINT_SHAPE)
        eef_pos = nan_array(3)
        eef_rot6d = nan_array(6)
        arm_connected = False
        arm_last_cmd_seq = 0
        arm_last_cmd_is_hold = False
        arm_state_valid = False

    if hand_state is not None:
        hand = hand_state[0]
        hand_qpos = np.asarray(hand["qpos"], dtype=np.float64)
        hand_current = np.asarray(hand["current"], dtype=np.float64)
        hand_tactile_sum = np.asarray(hand["tactile_sum"], dtype=np.float64)
        hand_tactile_contact = np.asarray(hand["tactile_contact"], dtype=bool)
        hand_connected = bool(hand["connected"])
        hand_qpos_stale = bool(hand["qpos_stale"])
        hand_commboard_err = np.asarray(hand["commboard_err"], dtype=np.int32)
        hand_jointboard_err = np.asarray(hand["jointboard_err"], dtype=np.int32)
        hand_tipboard_err = np.asarray(hand["tipboard_err"], dtype=np.int32)
        hand_state_valid = bool(hand["state_valid"])
    else:
        hand_qpos = nan_array(HAND_JOINT_SHAPE)
        hand_current = nan_array(HAND_JOINT_SHAPE)
        hand_tactile_sum = nan_array(HAND_TACTILE_SUM_SHAPE)
        hand_tactile_contact = np.zeros(HAND_CONTACT_SHAPE, dtype=bool)
        hand_connected = False
        hand_qpos_stale = False
        hand_commboard_err = np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)
        hand_jointboard_err = np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)
        hand_tipboard_err = np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)
        hand_state_valid = False

    if hand_tactile is not None:
        hand_tactile_force = np.asarray(
            hand_tactile[0]["tactile_force"], dtype=np.float64
        )
    else:
        hand_tactile_force = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)

    eef_quat_wxyz = (
        rot6d_to_quat_wxyz(eef_rot6d)
        if np.all(np.isfinite(eef_rot6d))
        else np.array([1.0, 0.0, 0.0, 0.0])
    )

    fingertip_pos = nan_array(HAND_FINGERTIP_SHAPE)
    if (
        hand_fk is not None
        and arm_state_valid
        and hand_state_valid
        and hand_connected
        and handbase_position_eef_m is not None
        and handbase_quat_eef_wxyz is not None
    ):
        fingertip_pos = compute_fingertip_points_xarm_base(
            arm_qpos,
            hand_qpos,
            arm_fk=None,
            hand_fk=hand_fk,
            handbase_position_eef_m=handbase_position_eef_m,
            handbase_quat_eef_wxyz=handbase_quat_eef_wxyz,
            eef_position_xarm_base_m=eef_pos,
            eef_rot6d_xarm_base=eef_rot6d,
        )

    return EpisodeState(
        arm_qpos=arm_qpos,
        arm_qvel=arm_qvel,
        arm_tau=arm_tau,
        eef_pos=eef_pos,
        eef_quat_wxyz=eef_quat_wxyz,
        eef_rot6d=eef_rot6d,
        hand_qpos=hand_qpos,
        hand_current=hand_current,
        hand_tactile_sum=hand_tactile_sum,
        hand_tactile_force=hand_tactile_force,
        hand_tactile_contact=hand_tactile_contact,
        hand_tipboard_err=hand_tipboard_err,
        hand_commboard_err=hand_commboard_err,
        hand_jointboard_err=hand_jointboard_err,
        hand_qpos_stale=hand_qpos_stale,
        arm_last_cmd_seq=arm_last_cmd_seq,
        arm_last_cmd_is_hold=arm_last_cmd_is_hold,
        fingertip_pos=fingertip_pos,
        arm_connected=arm_connected,
        hand_connected=hand_connected,
        timestamp=time.perf_counter() if timestamp_s is None else float(timestamp_s),
    )
