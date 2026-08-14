"""Build and publish typed episode samples from teleoperation snapshots."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dexmani_real.utils.schema import (
    ARM_JOINT_SHAPE,
    HAND_CONTACT_SHAPE,
    HAND_FINGERTIP_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)
from dexmani_real.planning import Pose
from dexmani_real.planning.hand_kinematics import HandKinematics
from dexmani_real.planning.pose_utils import compose_pose, normalize_quat_wxyz, quat_wxyz_to_rot6d, rot6d_to_quat_wxyz
from dexmani_real.policy.runtime import ActionCandidate
from dexmani_real.recording.io_process import RecorderClient
from dexmani_real.robot.types import RobotAction, RobotState
from dexmani_real.shm.shared_storage import SharedStorage
from dexmani_real.utils.array_utils import nan_array
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_FRAME_OK = 0
_FRAME_HELD = 1
_FRAME_IK_FAIL = 2
_FRAME_SAFETY_REJECT = 3
_FRAME_RETARGET_FAIL = 4
_OBSERVATION_MAX_SKEW_S = 0.10
_TACTILE_MAX_AGE_NS = 250_000_000
_NS_PER_SECOND = 1_000_000_000


def _stop_recording(
    recorder: RecorderClient | None,
    was_active: bool,
    *,
    save: bool,
    shared: SharedStorage | None = None,
    reason: str = "",
) -> None:
    """Stop recording if active. Non-blocking — poll completion in main loop."""
    if was_active and recorder is not None:
        recorder.stop_episode(success=save, reason=reason)
        if shared is not None:
            shared.is_recording.value = False


def _matching_source_sequence(ring: Any, frame: np.ndarray | None) -> int:
    """Recover the exact verified ring identity for one copied state frame."""
    if frame is None or frame.dtype.names is None or "source_monotonic_ns" not in frame.dtype.names:
        return 0
    source_ns = int(frame["source_monotonic_ns"][0])
    publish_ns = int(frame["publish_monotonic_ns"][0]) if "publish_monotonic_ns" in frame.dtype.names else 0
    try:
        history = ring.get_last_k(ring.maxlen)
    except Exception:
        logger.warning("teleop: source history read failed", exc_info=True)
        return 0
    for data, _ring_publish_ns, sequence in reversed(history):
        if int(data["source_monotonic_ns"][0]) != source_ns:
            continue
        if publish_ns and "publish_monotonic_ns" in (data.dtype.names or ()):
            if int(data["publish_monotonic_ns"][0]) != publish_ns:
                continue
        return int(sequence)
    return 0


def _recording_provenance(
    shared: SharedStorage,
    arm_state: np.ndarray | None,
    hand_state: np.ndarray | None,
    hand_tactile: np.ndarray | None,
    vr_frame: dict | None,
    cam: dict | None,
    *,
    anchor_monotonic_ns: int | None = None,
    action_candidate: ActionCandidate | None = None,
) -> dict[str, object]:
    """Correlate one policy-grid sample with causal sources and send metadata."""
    anchor_ns = time.monotonic_ns() if anchor_monotonic_ns is None else int(anchor_monotonic_ns)
    if anchor_ns <= 0 or anchor_ns > time.monotonic_ns():
        raise ValueError("recording observation anchor must be a positive elapsed grid deadline")

    def _field(frame: np.ndarray | None, name: str) -> int:
        if frame is None or frame.dtype.names is None or name not in frame.dtype.names:
            return 0
        return int(frame[name][0])

    arm_source_ns = _field(arm_state, "source_monotonic_ns")
    arm_publish_ns = _field(arm_state, "publish_monotonic_ns")
    hand_source_ns = _field(hand_state, "source_monotonic_ns")
    hand_publish_ns = _field(hand_state, "publish_monotonic_ns")
    arm_source_sequence = _matching_source_sequence(shared.arm_state_ring, arm_state)
    hand_source_sequence = _matching_source_sequence(shared.hand_state_ring, hand_state)
    vr_source_ns = int(vr_frame.get("local_recv_ns", 0)) if vr_frame is not None else 0
    vr_source_sequence = int(vr_frame.get("ring_sequence", 0)) if vr_frame is not None else 0
    vr_publish_ns = int(vr_frame.get("publish_monotonic_ns", 0)) if vr_frame is not None else 0
    camera_source_ns = int(cam.get("source_monotonic_ns", 0)) if cam is not None else 0
    camera_receive_ns = int(cam.get("receive_monotonic_ns", 0)) if cam is not None else 0
    camera_publish_ns = int(cam.get("publish_monotonic_ns", 0)) if cam is not None else 0
    source_ns = np.array([arm_source_ns, hand_source_ns, vr_source_ns, camera_source_ns], dtype=np.uint64)
    publish_ns = np.array([arm_publish_ns, hand_publish_ns, vr_publish_ns, camera_publish_ns], dtype=np.uint64)
    receive_ns = np.array([arm_publish_ns, hand_publish_ns, vr_source_ns, camera_receive_ns], dtype=np.uint64)

    source_valid = np.array(
        [
            arm_source_ns > 0 and arm_source_sequence > 0 and _field(arm_state, "state_valid") == 1,
            hand_source_ns > 0 and hand_source_sequence > 0 and _field(hand_state, "state_valid") == 1,
            vr_source_ns > 0 and vr_source_sequence > 0,
            camera_source_ns > 0 and bool(cam.get("camera_fresh", False)) if cam is not None else False,
        ],
        dtype=bool,
    )
    time_valid = (
        (source_ns > 0)
        & (receive_ns > 0)
        & (publish_ns > 0)
        & (source_ns <= receive_ns)
        & (receive_ns <= publish_ns)
        & (publish_ns <= anchor_ns)
    )
    source_valid &= time_valid
    ages_s = np.full(4, np.nan, dtype=np.float64)
    ages_s[source_valid] = (anchor_ns - source_ns[source_valid].astype(np.int64)) / _NS_PER_SECOND
    valid_times = source_ns[source_valid]
    newest_source_ns = int(np.max(valid_times)) if valid_times.size else 0
    skew_s = np.full(4, np.nan, dtype=np.float64)
    if newest_source_ns:
        skew_s[source_valid] = (newest_source_ns - source_ns[source_valid].astype(np.int64)) / _NS_PER_SECOND
    required_mask = source_valid[[0, 2, 3]]
    if hand_state is not None:
        required_mask = np.concatenate([required_mask, source_valid[1:2]])
    observation_valid = bool(np.all(required_mask)) and bool(np.nanmax(skew_s, initial=0.0) <= _OBSERVATION_MAX_SKEW_S)

    action_id = action_candidate.action_id if action_candidate is not None else 0
    observation_id = (
        action_candidate.observation_id
        if action_candidate is not None
        else int(vr_frame.get("ring_sequence", 0)) if vr_frame is not None else 0
    )
    if observation_id <= 0:
        observation_id = anchor_ns

    # Fire-and-forget worker status is not recorded: there is no ACK path to
    # make those values meaningful on this control tick.
    tactile_source_ns = _field(hand_tactile, "source_monotonic_ns")
    tactile_fresh = (
        _field(hand_tactile, "fresh") == 1
        and 0 < tactile_source_ns <= anchor_ns
        and anchor_ns - tactile_source_ns <= _TACTILE_MAX_AGE_NS
    )
    return {
        "observation_id": observation_id,
        "observation_anchor_monotonic_ns": anchor_ns,
        "arm_source_sequence": arm_source_sequence,
        "hand_source_sequence": hand_source_sequence,
        "vr_source_sequence": vr_source_sequence,
        "camera_source_sequence": int(cam.get("ring_sequence", 0)) if cam is not None else 0,
        "arm_source_monotonic_ns": arm_source_ns,
        "hand_source_monotonic_ns": hand_source_ns,
        "vr_source_monotonic_ns": vr_source_ns,
        "camera_source_monotonic_ns": camera_source_ns,
        "arm_publish_monotonic_ns": arm_publish_ns,
        "hand_publish_monotonic_ns": hand_publish_ns,
        "vr_publish_monotonic_ns": vr_publish_ns,
        "camera_publish_monotonic_ns": camera_publish_ns,
        "observation_source_receive_monotonic_ns": receive_ns,
        "observation_source_age_s": ages_s,
        "observation_source_skew_s": skew_s,
        "observation_history_valid_mask": source_valid[:, None],
        "observation_valid": observation_valid,
        "observation_skew_s": float(np.nanmax(skew_s, initial=0.0)),
        "action_id": action_id,
        "action_created_monotonic_ns": action_candidate.created_monotonic_ns if action_candidate is not None else 0,
        "action_target_monotonic_ns": action_candidate.target_monotonic_ns if action_candidate is not None else 0,
        "action_valid_until_monotonic_ns": (
            action_candidate.valid_until_monotonic_ns if action_candidate is not None else 0
        ),
        "action_queued": action_candidate is not None,
        "tactile_fresh": tactile_fresh,
        "tactile_source_monotonic_ns": tactile_source_ns,
        "tactile_calibrated": _field(hand_tactile, "calibrated") == 1,
        "tactile_unit_code": _field(hand_tactile, "unit_code"),
        "pointcloud_source_point_count": (int(cam.get("pointcloud_source_point_count", 0)) if cam is not None else 0),
        "pointcloud_valid_depth_ratio": float(cam.get("valid_depth_ratio", np.nan)) if cam is not None else np.nan,
        "pointcloud_padding_count": int(cam.get("pointcloud_padding_count", 0)) if cam is not None else 0,
    }


def _record_held(
    recorder: RecorderClient | None,
    arm_state: np.ndarray | None,
    hold_arm: np.ndarray,
    hold_hand: np.ndarray,
    vr_frame: dict | None,
    cam: dict | None,
    *,
    hand_state: np.ndarray | None = None,
    hand_tactile: np.ndarray | None = None,
    frame_status: int = _FRAME_HELD,
    retarget_ok: bool = False,
    arm_qpos_sent: np.ndarray | None = None,
    diagnostics: dict | None = None,
    target_eef_pos: np.ndarray | None = None,
    target_eef_rot6d: np.ndarray | None = None,
    hand_fk=None,
    T_eef_handbase_pos: np.ndarray | None = None,
    T_eef_handbase_quat_wxyz: np.ndarray | None = None,
    observation_anchor_monotonic_ns: int | None = None,
    shared: SharedStorage | None = None,
    action_candidate: ActionCandidate | None = None,
) -> None:
    """Record an active safety-fallback frame and its optional hold command.

    Ordinary ``CommandQuiescence`` never calls this helper: a command-silent
    pause emits neither an actuator action nor a recording sample.

    Args:
        arm_qpos_sent: Last command actually queued to arm_action_q.
            Persists the exact command sent so held-frame samples stay consistent.
        diagnostics: Per-frame diagnostics (tracking_error, ik_solve_time_ms, etc.).
        target_eef_pos/rot6d: Last valid IK target — prevents NaN gaps in
            ``action_arm_ee`` in the recorded sample.
        action_candidate: Exact hold candidate published for this observation,
            or ``None`` when the grid intentionally emitted no new command.
    """
    if recorder is None:
        return
    if vr_frame is None:
        vr_frame = {
            "wrist_pos": np.full(3, np.nan),
            "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
            "landmarks": np.full((21, 3), np.nan),
        }
    action = RobotAction(
        arm_qpos_cmd=hold_arm,
        hand_qpos_cmd=hold_hand,
        target_eef_pos=target_eef_pos.copy() if target_eef_pos is not None else None,
        target_eef_rot6d=target_eef_rot6d.copy() if target_eef_rot6d is not None else None,
    )
    state = _build_robot_state(
        arm_state,
        hand_state,
        hand_tactile,
        hk=hand_fk,
        T_eef_handbase_pos=T_eef_handbase_pos,
        T_eef_handbase_quat_wxyz=T_eef_handbase_quat_wxyz,
        timestamp_s=(None if observation_anchor_monotonic_ns is None else int(observation_anchor_monotonic_ns) / 1e9),
    )
    signals: dict[str, object] = {
        "ik_ok": False,
        "ik_attempted": frame_status != _FRAME_HELD,
        "retarget_ok": retarget_ok,
        "held": True,
        "flag_safety_reject": frame_status == _FRAME_SAFETY_REJECT,
        "frame_status": frame_status,
    }
    if shared is not None:
        signals.update(
            _recording_provenance(
                shared,
                arm_state,
                hand_state,
                hand_tactile,
                vr_frame,
                cam,
                anchor_monotonic_ns=observation_anchor_monotonic_ns,
                action_candidate=action_candidate,
            )
        )
    recorder.add_frame(
        state,
        action,
        vr_frame,
        camera_frame=cam,
        signals=signals,
        arm_qpos_sent=arm_qpos_sent,
        diagnostics=diagnostics,
    )


def _record_frame(
    recorder: RecorderClient | None,
    arm_state: np.ndarray | None,
    hand_state: np.ndarray | None,
    arm_cmd: np.ndarray,
    hand_cmd: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    vr_frame: dict | None,
    cam: dict | None,
    ik_solve_time_ms: float,
    target_pos_before_clamp: np.ndarray,
    hand_tactile: np.ndarray | None = None,
    *,
    retarget_ok: bool = False,
    frame_status: int = _FRAME_OK,
    target_eef_pos_raw: np.ndarray | None = None,
    target_eef_rot6d_raw: np.ndarray | None = None,
    action_arm_joint_raw: np.ndarray | None = None,
    action_hand_joint_raw: np.ndarray | None = None,
    policy_map_time_ms: float = np.nan,
    hand_retarget_time_ms: float = np.nan,
    policy_compute_time_ms: float = np.nan,
    hand_fk=None,
    T_eef_handbase_pos: np.ndarray | None = None,
    T_eef_handbase_quat_wxyz: np.ndarray | None = None,
    observation_anchor_monotonic_ns: int | None = None,
    shared: SharedStorage | None = None,
    action_candidate: ActionCandidate | None = None,
) -> None:
    """Record a normal (active teleop) frame.

    Args:
        target_quat: EMA-smoothed IK target quaternion (wxyz), NOT raw VR wrist.
            This is what the IK solver actually tracked.
    """
    if recorder is None:
        return
    action = RobotAction(
        arm_qpos_cmd=arm_cmd,
        hand_qpos_cmd=hand_cmd,
        target_eef_pos=target_pos.copy(),
        target_eef_rot6d=quat_wxyz_to_rot6d(normalize_quat_wxyz(target_quat)),
    )
    state = _build_robot_state(
        arm_state,
        hand_state,
        hand_tactile,
        hk=hand_fk,
        T_eef_handbase_pos=T_eef_handbase_pos,
        T_eef_handbase_quat_wxyz=T_eef_handbase_quat_wxyz,
        timestamp_s=(None if observation_anchor_monotonic_ns is None else int(observation_anchor_monotonic_ns) / 1e9),
    )
    head_quat = vr_frame.get("head_quat_wxyz") if vr_frame is not None else None
    _vr = (
        vr_frame
        if vr_frame is not None
        else {
            "wrist_pos": np.full(3, np.nan),
            "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
            "landmarks": np.full((21, 3), np.nan),
        }
    )
    signals: dict[str, object] = {
        "ik_ok": True,
        "ik_attempted": True,
        "retarget_ok": retarget_ok,
        "held": False,
        "flag_safety_reject": frame_status == _FRAME_SAFETY_REJECT,
        "frame_status": frame_status,
        "action_arm_joint_raw": (
            np.asarray(action_arm_joint_raw, dtype=np.float64) if action_arm_joint_raw is not None else arm_cmd.copy()
        ),
    }
    if shared is not None:
        signals.update(
            _recording_provenance(
                shared,
                arm_state,
                hand_state,
                hand_tactile,
                vr_frame,
                cam,
                anchor_monotonic_ns=observation_anchor_monotonic_ns,
                action_candidate=action_candidate,
            )
        )
    recorder.add_frame(
        state,
        action,
        _vr,
        camera_frame=cam,
        signals=signals,
        arm_qpos_sent=arm_cmd.copy(),
        diagnostics={
            "tracking_error": (
                float(arm_state["tracking_err"][0])
                if arm_state is not None and "tracking_err" in arm_state.dtype.names
                else 0.0
            ),
            "ik_solve_time_ms": ik_solve_time_ms,
            "target_pos_before_clamp": target_pos_before_clamp,
            "head_quat_wxyz": head_quat if head_quat is not None else np.full(4, np.nan),
            "target_eef_pos_raw": (
                np.asarray(target_eef_pos_raw, dtype=np.float64)
                if target_eef_pos_raw is not None
                else np.full(3, np.nan)
            ),
            "target_eef_rot6d_raw": (
                np.asarray(target_eef_rot6d_raw, dtype=np.float64)
                if target_eef_rot6d_raw is not None
                else np.full(6, np.nan)
            ),
            "action_hand_joint_raw": (
                np.asarray(action_hand_joint_raw, dtype=np.float64)
                if action_hand_joint_raw is not None
                else hand_cmd.copy()
            ),
            "policy_map_time_ms": policy_map_time_ms,
            "hand_retarget_time_ms": hand_retarget_time_ms,
            "transition_check_time_ms": 0.0,  # reserved (SafetyGate collision/transition removed 2026-08-12)
            "policy_compute_time_ms": policy_compute_time_ms,
        },
    )


def _build_robot_state(
    arm_state: np.ndarray | None,
    hand_state: np.ndarray | None,
    hand_tactile: np.ndarray | None = None,
    hk: HandKinematics | None = None,
    T_eef_handbase_pos: np.ndarray | None = None,
    T_eef_handbase_quat_wxyz: np.ndarray | None = None,
    timestamp_s: float | None = None,
) -> RobotState:
    """Build a RobotState from ring data for recording.

    Reads arm_state_ring + hand_state_ring + hand_tactile_ring and assembles
    a complete RobotState.  Computes world-frame fingertip positions via hand FK
    chain (handbase -> fingertip -> world via EEF transform).

    Hand hardware/error flags are forwarded to RobotState for recording.
    ``qpos_stale`` remains a reserved false compatibility field; runtime
    freshness comes from the source timestamp and read/state validity.
    """
    if arm_state is not None:
        r = arm_state[0]
        arm_qpos = np.asarray(r["qpos"], dtype=np.float64)
        arm_qvel = np.asarray(r["qvel"], dtype=np.float64)
        arm_tau = np.asarray(r["tau"], dtype=np.float64)
        eef_pos = np.asarray(r["eef_pos"], dtype=np.float64)
        eef_rot6d = np.asarray(r["eef_rot6d"], dtype=np.float64)
        arm_connected = bool(r["connected"])
        arm_last_cmd_seq = int(r["last_cmd_seq"]) if "last_cmd_seq" in r.dtype.names else 0
        arm_last_cmd_queue_latency_s = (
            float(r["last_cmd_queue_latency_s"]) if "last_cmd_queue_latency_s" in r.dtype.names else 0.0
        )
        arm_last_cmd_apply_latency_s = (
            float(r["last_cmd_apply_latency_s"]) if "last_cmd_apply_latency_s" in r.dtype.names else 0.0
        )
        arm_last_cmd_sdk_duration_s = (
            float(r["last_cmd_sdk_duration_s"]) if "last_cmd_sdk_duration_s" in r.dtype.names else 0.0
        )
        arm_last_cmd_is_hold = bool(r["last_cmd_is_hold"]) if "last_cmd_is_hold" in r.dtype.names else False
    else:
        arm_qpos = nan_array(ARM_JOINT_SHAPE)
        arm_qvel = nan_array(ARM_JOINT_SHAPE)
        arm_tau = nan_array(ARM_JOINT_SHAPE)
        eef_pos = nan_array(3)
        eef_rot6d = nan_array(6)
        arm_connected = False
        arm_last_cmd_seq = 0
        arm_last_cmd_queue_latency_s = 0.0
        arm_last_cmd_apply_latency_s = 0.0
        arm_last_cmd_sdk_duration_s = 0.0
        arm_last_cmd_is_hold = False

    if hand_state is not None:
        h = hand_state[0]
        hand_qpos = np.asarray(h["qpos"], dtype=np.float64)
        hand_current = np.asarray(h["current"], dtype=np.float64)
        hand_tactile_sum = np.asarray(h["tactile_sum"], dtype=np.float64)
        hand_tactile_contact = np.asarray(h["tactile_contact"], dtype=bool)
        hand_connected = bool(h["connected"])
        hand_qpos_stale = bool(h["qpos_stale"]) if "qpos_stale" in h.dtype.names else False
        hand_error_state = bool(h["error_state"]) if "error_state" in h.dtype.names else False
        # Board error registers — per-joint hardware fault indicators.
        hand_commboard_err = (
            np.asarray(h["commboard_err"], dtype=np.int32)
            if "commboard_err" in h.dtype.names
            else np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)
        )
        hand_jointboard_err = (
            np.asarray(h["jointboard_err"], dtype=np.int32)
            if "jointboard_err" in h.dtype.names
            else np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)
        )
        hand_tipboard_err = (
            np.asarray(h["tipboard_err"], dtype=np.int32)
            if "tipboard_err" in h.dtype.names
            else np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)
        )
    else:
        hand_qpos = nan_array(HAND_JOINT_SHAPE)
        hand_current = nan_array(HAND_JOINT_SHAPE)
        hand_tactile_sum = nan_array(HAND_TACTILE_SUM_SHAPE)
        hand_tactile_contact = np.zeros(HAND_CONTACT_SHAPE, dtype=bool)
        hand_connected = False
        hand_qpos_stale = False
        hand_error_state = False
        hand_commboard_err = np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)
        hand_jointboard_err = np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)
        hand_tipboard_err = np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)

    # Tactile force comes from the separate sparse ring.
    if hand_tactile is not None:
        hand_tactile_force = np.asarray(hand_tactile[0]["tactile_force"], dtype=np.float64)
    else:
        hand_tactile_force = np.zeros(HAND_TACTILE_FORCE_SHAPE, dtype=np.float64)

    _eef_finite = np.all(np.isfinite(eef_rot6d))
    eef_quat_wxyz = rot6d_to_quat_wxyz(eef_rot6d) if _eef_finite else np.array([1.0, 0.0, 0.0, 0.0])

    # Compute world-frame fingertip positions via hand FK.
    fingertip_pos = nan_array(HAND_FINGERTIP_SHAPE)
    if hk is not None and hk.is_ready() and hand_connected and np.all(np.isfinite(hand_qpos)):
        tips_in_handbase = hk.compute_tip_positions_in_handbase(hand_qpos)
        if np.all(np.isfinite(tips_in_handbase)) and _eef_finite and np.all(np.isfinite(eef_pos)):
            T_world_eef = Pose(p=eef_pos, q=eef_quat_wxyz)
            T_eef_handbase = Pose(
                p=T_eef_handbase_pos if T_eef_handbase_pos is not None else np.zeros(3),
                q=T_eef_handbase_quat_wxyz if T_eef_handbase_quat_wxyz is not None else np.array([1.0, 0.0, 0.0, 0.0]),
            )
            T_world_handbase = compose_pose(T_world_eef, T_eef_handbase)
            tips_world = np.zeros(HAND_FINGERTIP_SHAPE, dtype=np.float64)
            _id_quat = np.array([1.0, 0.0, 0.0, 0.0])
            for i in range(5):
                T_world_tip = compose_pose(T_world_handbase, Pose(p=tips_in_handbase[i], q=_id_quat))
                tips_world[i] = T_world_tip.p
            fingertip_pos = tips_world

    return RobotState(
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
        hand_error_state=hand_error_state,
        arm_last_cmd_seq=arm_last_cmd_seq,
        arm_last_cmd_queue_latency_s=arm_last_cmd_queue_latency_s,
        arm_last_cmd_apply_latency_s=arm_last_cmd_apply_latency_s,
        arm_last_cmd_sdk_duration_s=arm_last_cmd_sdk_duration_s,
        arm_last_cmd_is_hold=arm_last_cmd_is_hold,
        fingertip_pos=fingertip_pos,
        arm_connected=arm_connected,
        hand_connected=hand_connected,
        timestamp=time.perf_counter() if timestamp_s is None else float(timestamp_s),
    )
