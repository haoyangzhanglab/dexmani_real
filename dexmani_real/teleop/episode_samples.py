"""Build and publish typed episode samples from teleoperation snapshots."""

from __future__ import annotations

import time
from typing import Mapping

import numpy as np

from dexmani_real.control.action import ActionCandidate
from dexmani_real.ipc.channels import RuntimeChannels
from dexmani_real.planning.kinematics.pose import (
    normalize_quat_wxyz,
    quat_wxyz_to_rot6d,
    rot6d_to_quat_wxyz,
)
from dexmani_real.recording.client import RecorderClient
from dexmani_real.recording.sample import EpisodeAction, build_episode_state

FRAME_OK = 0
_FRAME_HELD = 1
FRAME_IK_FAIL = 2
FRAME_SAFETY_REJECT = 3
FRAME_RETARGET_FAIL = 4
RECORDING_TACTILE_MAX_AGE_NS = 250_000_000
_NS_PER_SECOND = 1_000_000_000


def stop_recording(
    recorder: RecorderClient | None,
    was_active: bool,
    *,
    save: bool,
    shared: RuntimeChannels | None = None,
    reason: str = "",
) -> None:
    """Stop recording if active. Non-blocking — poll completion in main loop."""
    if was_active and recorder is not None:
        recorder.stop_episode(success=save, reason=reason)
        if shared is not None:
            shared.is_recording.value = False


def _recording_provenance(
    arm_state: np.ndarray | None,
    hand_state: np.ndarray | None,
    hand_tactile: np.ndarray | None,
    vr_frame: dict | None,
    cam: dict | None,
    *,
    anchor_monotonic_ns: int | None = None,
    arm_ring_sequence: int = 0,
    hand_ring_sequence: int = 0,
    action_candidate: ActionCandidate | None = None,
    max_observation_skew_s: float,
) -> dict[str, object]:
    """Correlate one policy-grid sample with causal sources and send metadata."""
    anchor_ns = (
        time.monotonic_ns() if anchor_monotonic_ns is None else int(anchor_monotonic_ns)
    )
    if anchor_ns <= 0 or anchor_ns > time.monotonic_ns():
        raise ValueError(
            "recording observation anchor must be a positive elapsed grid deadline"
        )

    def _field(frame: np.ndarray | None, name: str) -> int:
        if frame is None or frame.dtype.names is None or name not in frame.dtype.names:
            return 0
        return int(frame[name][0])

    arm_source_ns = _field(arm_state, "source_monotonic_ns")
    arm_publish_ns = _field(arm_state, "publish_monotonic_ns")
    hand_source_ns = _field(hand_state, "source_monotonic_ns")
    hand_publish_ns = _field(hand_state, "publish_monotonic_ns")
    arm_source_sequence = int(arm_ring_sequence)
    hand_source_sequence = int(hand_ring_sequence)
    vr_source_ns = int(vr_frame.get("recv_ts_ns", 0)) if vr_frame is not None else 0
    vr_source_sequence = (
        int(vr_frame.get("ring_sequence", 0)) if vr_frame is not None else 0
    )
    vr_publish_ns = (
        int(vr_frame.get("publish_monotonic_ns", 0)) if vr_frame is not None else 0
    )
    camera_source_ns = int(cam.get("source_monotonic_ns", 0)) if cam is not None else 0
    camera_receive_ns = (
        int(cam.get("receive_monotonic_ns", 0)) if cam is not None else 0
    )
    camera_publish_ns = (
        int(cam.get("publish_monotonic_ns", 0)) if cam is not None else 0
    )
    source_ns = np.array(
        [arm_source_ns, hand_source_ns, vr_source_ns, camera_source_ns], dtype=np.uint64
    )
    publish_ns = np.array(
        [arm_publish_ns, hand_publish_ns, vr_publish_ns, camera_publish_ns],
        dtype=np.uint64,
    )
    receive_ns = np.array(
        [arm_publish_ns, hand_publish_ns, vr_source_ns, camera_receive_ns],
        dtype=np.uint64,
    )

    source_valid = np.array(
        [
            arm_source_ns > 0
            and arm_source_sequence > 0
            and _field(arm_state, "state_valid") == 1,
            hand_source_ns > 0
            and hand_source_sequence > 0
            and _field(hand_state, "state_valid") == 1,
            vr_source_ns > 0 and vr_source_sequence > 0,
            (
                camera_source_ns > 0 and bool(cam.get("camera_fresh", False))
                if cam is not None
                else False
            ),
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
    ages_s[source_valid] = (
        anchor_ns - source_ns[source_valid].astype(np.int64)
    ) / _NS_PER_SECOND
    valid_times = source_ns[source_valid]
    newest_source_ns = int(np.max(valid_times)) if valid_times.size else 0
    skew_s = np.full(4, np.nan, dtype=np.float64)
    if newest_source_ns:
        skew_s[source_valid] = (
            newest_source_ns - source_ns[source_valid].astype(np.int64)
        ) / _NS_PER_SECOND
    required_mask = source_valid[[0, 2, 3]]
    if hand_state is not None:
        required_mask = np.concatenate([required_mask, source_valid[1:2]])
    observation_valid = bool(np.all(required_mask)) and bool(
        np.nanmax(skew_s, initial=0.0) <= float(max_observation_skew_s)
    )

    action_id = action_candidate.action_id if action_candidate is not None else 0
    observation_id = (
        action_candidate.observation_id
        if action_candidate is not None
        else int(vr_frame.get("ring_sequence", 0)) if vr_frame is not None else 0
    )
    if observation_id <= 0:
        observation_id = anchor_ns

    # Fire-and-forget worker status is omitted because there is no same-tick ACK.
    tactile_source_ns = _field(hand_tactile, "source_monotonic_ns")
    tactile_fresh = (
        _field(hand_tactile, "fresh") == 1
        and 0 < tactile_source_ns <= anchor_ns
        and anchor_ns - tactile_source_ns <= RECORDING_TACTILE_MAX_AGE_NS
    )
    return {
        "observation_id": observation_id,
        "observation_anchor_monotonic_ns": anchor_ns,
        "arm_source_sequence": arm_source_sequence,
        "hand_source_sequence": hand_source_sequence,
        "vr_source_sequence": vr_source_sequence,
        "camera_source_sequence": (
            int(cam.get("ring_sequence", 0)) if cam is not None else 0
        ),
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
        "hand_accepted_target_action_id": _field(
            hand_state, "accepted_target_action_id"
        ),
        "action_id": action_id,
        "action_created_monotonic_ns": (
            action_candidate.created_monotonic_ns if action_candidate is not None else 0
        ),
        "action_target_monotonic_ns": (
            action_candidate.target_monotonic_ns if action_candidate is not None else 0
        ),
        "action_valid_until_monotonic_ns": (
            action_candidate.valid_until_monotonic_ns
            if action_candidate is not None
            else 0
        ),
        "action_queued": action_candidate is not None,
        "tactile_fresh": tactile_fresh,
        "tactile_source_monotonic_ns": tactile_source_ns,
        "tactile_calibrated": _field(hand_tactile, "calibrated") == 1,
        "tactile_unit_code": _field(hand_tactile, "unit_code"),
        "pointcloud_valid_depth_ratio": (
            float(cam.get("valid_depth_ratio", np.nan)) if cam is not None else np.nan
        ),
    }


def record_held(
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
    arm_ring_sequence: int = 0,
    hand_ring_sequence: int = 0,
    shared: RuntimeChannels | None = None,
    action_candidate: ActionCandidate | None = None,
    control_run_generation: int,
    max_observation_skew_s: float,
    policy_observation: Mapping[str, object] | None = None,
) -> None:
    """Record an active safety-fallback frame and its optional hold command.

    A command-silent pause never calls this helper: it emits neither an
    actuator action nor a recording sample.

    Args:
        arm_qpos_sent: Last arm target published in the coupled command record.
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
    action = EpisodeAction(
        arm_qpos_cmd=hold_arm,
        hand_qpos_cmd=hold_hand,
        target_eef_pos=target_eef_pos.copy() if target_eef_pos is not None else None,
        target_eef_rot6d=(
            target_eef_rot6d.copy() if target_eef_rot6d is not None else None
        ),
    )
    state = build_episode_state(
        arm_state,
        hand_state,
        hand_tactile,
        hand_fk=hand_fk,
        handbase_position_eef_m=T_eef_handbase_pos,
        handbase_quat_eef_wxyz=T_eef_handbase_quat_wxyz,
        timestamp_s=(
            None
            if observation_anchor_monotonic_ns is None
            else int(observation_anchor_monotonic_ns) / 1e9
        ),
    )
    signals: dict[str, object] = {
        "ik_ok": False,
        "ik_attempted": frame_status != _FRAME_HELD,
        "retarget_ok": retarget_ok,
        "held": True,
        "flag_safety_reject": frame_status == FRAME_SAFETY_REJECT,
        "frame_status": frame_status,
    }
    if shared is not None:
        signals.update(
            _recording_provenance(
                arm_state,
                hand_state,
                hand_tactile,
                vr_frame,
                cam,
                anchor_monotonic_ns=observation_anchor_monotonic_ns,
                arm_ring_sequence=arm_ring_sequence,
                hand_ring_sequence=hand_ring_sequence,
                action_candidate=action_candidate,
                max_observation_skew_s=max_observation_skew_s,
            )
        )
    if policy_observation is not None:
        signals.update(policy_observation)
    recorder.add_frame(
        state,
        action,
        vr_frame,
        camera_frame=cam,
        signals=signals,
        arm_qpos_sent=arm_qpos_sent,
        diagnostics=diagnostics,
        control_run_generation=control_run_generation,
    )


def record_frame(
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
    frame_status: int = FRAME_OK,
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
    arm_ring_sequence: int = 0,
    hand_ring_sequence: int = 0,
    shared: RuntimeChannels | None = None,
    action_candidate: ActionCandidate | None = None,
    control_run_generation: int,
    max_observation_skew_s: float,
    policy_observation: Mapping[str, object] | None = None,
) -> None:
    """Record a normal (active teleop) frame.

    Args:
        target_quat: EMA-smoothed IK target quaternion (wxyz), NOT raw VR wrist.
            This is what the IK solver actually tracked.
    """
    if recorder is None:
        return
    action = EpisodeAction(
        arm_qpos_cmd=arm_cmd,
        hand_qpos_cmd=hand_cmd,
        target_eef_pos=target_pos.copy(),
        target_eef_rot6d=quat_wxyz_to_rot6d(normalize_quat_wxyz(target_quat)),
    )
    state = build_episode_state(
        arm_state,
        hand_state,
        hand_tactile,
        hand_fk=hand_fk,
        handbase_position_eef_m=T_eef_handbase_pos,
        handbase_quat_eef_wxyz=T_eef_handbase_quat_wxyz,
        timestamp_s=(
            None
            if observation_anchor_monotonic_ns is None
            else int(observation_anchor_monotonic_ns) / 1e9
        ),
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
        "flag_safety_reject": frame_status == FRAME_SAFETY_REJECT,
        "frame_status": frame_status,
        "action_arm_joint_raw": (
            np.asarray(action_arm_joint_raw, dtype=np.float64)
            if action_arm_joint_raw is not None
            else arm_cmd.copy()
        ),
    }
    if shared is not None:
        signals.update(
            _recording_provenance(
                arm_state,
                hand_state,
                hand_tactile,
                vr_frame,
                cam,
                anchor_monotonic_ns=observation_anchor_monotonic_ns,
                arm_ring_sequence=arm_ring_sequence,
                hand_ring_sequence=hand_ring_sequence,
                action_candidate=action_candidate,
                max_observation_skew_s=max_observation_skew_s,
            )
        )
    if policy_observation is not None:
        signals.update(policy_observation)
    recorder.add_frame(
        state,
        action,
        _vr,
        camera_frame=cam,
        signals=signals,
        arm_qpos_sent=arm_cmd.copy(),
        control_run_generation=control_run_generation,
        diagnostics={
            "tracking_error": (
                float(arm_state["tracking_err"][0])
                if arm_state is not None and "tracking_err" in arm_state.dtype.names
                else 0.0
            ),
            "ik_solve_time_ms": ik_solve_time_ms,
            "target_pos_before_clamp": target_pos_before_clamp,
            "head_quat_wxyz": (
                head_quat if head_quat is not None else np.full(4, np.nan)
            ),
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
            "transition_check_time_ms": 0.0,
            "policy_compute_time_ms": policy_compute_time_ms,
        },
    )
