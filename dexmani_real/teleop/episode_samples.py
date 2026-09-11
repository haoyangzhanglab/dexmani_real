"""Build and publish typed episode samples from teleoperation snapshots."""

from __future__ import annotations

import time

import numpy as np

from dexmani_real.ipc.channels import RuntimeChannels
from dexmani_real.planning.kinematics.pose import (
    normalize_quat_wxyz,
    quat_wxyz_to_rot6d,
)
from dexmani_real.recording.client import RecorderClient
from dexmani_real.recording.sample import EpisodeAction, build_episode_state

FRAME_OK = 0
_FRAME_HELD = 1
FRAME_IK_FAIL = 2
FRAME_SAFETY_REJECT = 3
FRAME_RETARGET_FAIL = 4
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
        recorder.stop_episode(save=save, reason=reason)
        if shared is not None:
            shared.is_recording.value = False


def _recording_provenance(
    arm_state: np.ndarray | None,
    hand_state: np.ndarray | None,
    vr_frame: dict | None,
    cam: dict | None,
    *,
    anchor_monotonic_ns: int | None = None,
    max_observation_skew_s: float,
) -> dict[str, object]:
    """Correlate one policy-grid sample with causal source timestamps."""
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
    vr_source_ns = int(vr_frame.get("recv_ts_ns", 0)) if vr_frame is not None else 0
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
            arm_source_ns > 0 and _field(arm_state, "state_valid") == 1,
            hand_source_ns > 0 and _field(hand_state, "state_valid") == 1,
            vr_source_ns > 0,
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
    valid_times = source_ns[source_valid]
    source_span_s = (
        (int(np.max(valid_times)) - int(np.min(valid_times))) / _NS_PER_SECOND
        if valid_times.size
        else np.inf
    )
    required_mask = source_valid[[0, 2, 3]]
    if hand_state is not None:
        required_mask = np.concatenate([required_mask, source_valid[1:2]])
    observation_valid = bool(np.all(required_mask)) and source_span_s <= float(
        max_observation_skew_s
    )

    return {
        "observation_anchor_monotonic_ns": anchor_ns,
        "arm_source_monotonic_ns": arm_source_ns,
        "hand_source_monotonic_ns": hand_source_ns,
        "vr_source_monotonic_ns": vr_source_ns,
        "camera_source_monotonic_ns": camera_source_ns,
        "observation_valid": observation_valid,
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
    frame_status: int = _FRAME_HELD,
    arm_qpos_sent: np.ndarray,
    action_queued: bool = False,
    target_eef_pos: np.ndarray | None = None,
    target_eef_rot6d: np.ndarray | None = None,
    observation_anchor_monotonic_ns: int | None = None,
    shared: RuntimeChannels | None = None,
    max_observation_skew_s: float,
) -> None:
    """Record an active safety-fallback frame and its optional hold command.

    A command-silent pause never calls this helper: it emits neither an
    actuator action nor a recording sample.

    Args:
        arm_qpos_sent: Last arm target published in the coupled command record.
            Persists the exact command sent so held-frame samples stay consistent.
        target_eef_pos/rot6d: Last valid IK target — prevents NaN gaps in
            ``action_arm_ee`` in the recorded sample.
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
        timestamp_s=(
            None
            if observation_anchor_monotonic_ns is None
            else int(observation_anchor_monotonic_ns) / 1e9
        ),
    )
    signals: dict[str, object] = {
        "action_queued": action_queued,
        "frame_status": frame_status,
        "tracking_error": (
            float(arm_state["tracking_err"][0])
            if arm_state is not None and "tracking_err" in arm_state.dtype.names
            else np.nan
        ),
    }
    if shared is not None:
        signals.update(
            _recording_provenance(
                arm_state,
                hand_state,
                vr_frame,
                cam,
                anchor_monotonic_ns=observation_anchor_monotonic_ns,
                max_observation_skew_s=max_observation_skew_s,
            )
        )
    recorder.add_frame(
        state,
        action,
        vr_frame,
        camera_frame=cam,
        signals=signals,
        arm_qpos_sent=arm_qpos_sent,
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
    *,
    frame_status: int = FRAME_OK,
    observation_anchor_monotonic_ns: int | None = None,
    shared: RuntimeChannels | None = None,
    max_observation_skew_s: float,
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
        timestamp_s=(
            None
            if observation_anchor_monotonic_ns is None
            else int(observation_anchor_monotonic_ns) / 1e9
        ),
    )
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
        "frame_status": frame_status,
        "action_queued": True,
        "tracking_error": (
            float(arm_state["tracking_err"][0])
            if arm_state is not None and "tracking_err" in arm_state.dtype.names
            else np.nan
        ),
    }
    if shared is not None:
        signals.update(
            _recording_provenance(
                arm_state,
                hand_state,
                vr_frame,
                cam,
                anchor_monotonic_ns=observation_anchor_monotonic_ns,
                max_observation_skew_s=max_observation_skew_s,
            )
        )
    recorder.add_frame(
        state,
        action,
        _vr,
        camera_frame=cam,
        signals=signals,
        arm_qpos_sent=arm_cmd.copy(),
    )
