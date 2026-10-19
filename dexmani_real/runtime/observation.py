"""One immutable observation per control step, retaining each source's host clock."""

import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


def sample_is_fresh(timestamp_ns, max_age_s, now_ns=None):
    now = time.monotonic_ns() if now_ns is None else now_ns
    return 0 < int(timestamp_ns) <= now and now - int(timestamp_ns) <= int(max_age_s * 1e9)


def feedback_deadline_ns(row, runtime, *, include_vr=False):
    """Exclusive deadline for feedback used to prepare a target; recheck expiry at each SDK send boundary."""
    deadlines = [int(row.arm["timestamp_ns"][0]) + int(runtime.arm.feedback_max_age_s * 1e9) + 1]
    if row.hand is not None:
        deadlines.append(
            hand_feedback_time_ns(row.hand) + int(runtime.hand.feedback_max_age_s * 1e9)
        )
    if include_vr:
        deadlines.append(
            int(row.vr["recv_ts_ns"]) + int(runtime.control.vr_mapping.stale_threshold_s * 1e9) + 1
        )
    return min(deadlines)


def hand_feedback_time_ns(hand):
    """Oldest required hand source, without conflating tactile age with joint time."""
    return min(
        int(hand[name][0])
        for name in (
            "timestamp_ns",
            "tactile_aggregate_timestamp_ns",
            "tactile_dense_timestamp_ns",
        )
    )


def read_vr_frame(runtime_state):
    result = runtime_state.sensors.vr_ring.read_latest(use_last_good=True)
    if result is None:
        return None
    record = result[0][0]
    return {
        **{name: record[name] for name in record.dtype.names},
        "ring_sequence": result[2],
    }


def read_camera_frame(runtime_state, sequence=None):
    ring = runtime_state.sensors.camera_ring
    result = ring.read_latest() if sequence is None else ring.read_sequence(sequence)
    if result is None:
        return None
    data, publication_ns, sequence = result
    frame = data[0]
    header = frame["header"]
    return {
        "rgb": frame["rgb"],
        "depth": frame["depth"],
        "ring_sequence": sequence,
        "published_ns": publication_ns,
        **{name: int(header[name]) for name in header.dtype.names},
    }


def freeze(value):
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif isinstance(value, dict):
        value = MappingProxyType({k: freeze(v) for k, v in value.items()})
    return value


@dataclass(frozen=True)
class ObservationRow:
    arm: np.ndarray
    hand: np.ndarray | None
    camera: Mapping | None
    vr: Mapping | None
    point_cloud: np.ndarray | None
    observation_timestamp_ns: int
    pointcloud_timestamp_ns: int = 0
    pointcloud_camera_sequence: int = -1
    control_timestamp_ns: int = -1
    pointcloud_camera_received_ns: int = -1
    pointcloud_camera_published_ns: int = -1
    pointcloud_started_ns: int = -1
    pointcloud_published_ns: int = -1


def read_observation(
    runtime_state,
    runtime,
    robot,
    *,
    require_hand=True,
    require_camera=False,
    require_vr=False,
    require_pointcloud=False,
    require_rgb_cloud_identity=False,
    issues=None,
    wait_for_hand=False,
    cancel_requested=None,
):
    unavailable = []
    control_ns = time.monotonic_ns()
    state = (
        robot.read_initial_state(cancel_requested=cancel_requested)
        if wait_for_hand
        else robot.read_state()
    )
    arm, hand = state.arm, state.hand if require_hand else None
    cloud_result = (
        runtime_state.sensors.pointcloud_ring.read_latest(use_last_good=True)
        if require_pointcloud
        else None
    )
    if (require_hand and hand is None) or (require_pointcloud and cloud_result is None):
        unavailable.append("feedback_or_pointcloud_unavailable")
    cloud = cloud_result[0][0] if cloud_result else None
    if require_rgb_cloud_identity:
        camera = (
            read_camera_frame(runtime_state, int(cloud["source_camera_sequence"]))
            if cloud is not None
            else None
        )
    else:
        camera = read_camera_frame(runtime_state) if require_camera else None
    vr = read_vr_frame(runtime_state) if require_vr else None
    if (require_camera or require_rgb_cloud_identity) and camera is None:
        unavailable.append("camera_source_unavailable_or_unmatched")
    if require_vr and vr is None:
        unavailable.append("vr_unavailable")
    now = time.monotonic_ns()
    if not sample_is_fresh(arm["timestamp_ns"][0], runtime.arm.feedback_max_age_s, now):
        unavailable.append("arm_feedback_stale")
    if hand is not None:
        hand_ns = hand_feedback_time_ns(hand)
        if not 0 < hand_ns <= now < hand_ns + int(runtime.hand.feedback_max_age_s * 1e9):
            unavailable.append("hand_feedback_stale")
    if camera is not None and not sample_is_fresh(
        camera["timestamp_ns"], runtime.camera.max_frame_age_s, now
    ):
        unavailable.append("camera_source_stale")
    if cloud is not None and not sample_is_fresh(
        cloud["timestamp_ns"], runtime.camera.max_frame_age_s, now
    ):
        unavailable.append("pointcloud_stale")
    if vr is not None and not sample_is_fresh(
        vr["recv_ts_ns"], runtime.control.vr_mapping.stale_threshold_s, now
    ):
        unavailable.append("vr_stale")
    if unavailable:
        if issues is not None:
            issues.extend(unavailable)
        return None
    row = ObservationRow(
        arm=freeze(arm),
        hand=freeze(hand),
        camera=freeze(camera),
        vr=freeze(vr),
        point_cloud=freeze(cloud["point_cloud"]) if cloud is not None else None,
        observation_timestamp_ns=now,
        pointcloud_timestamp_ns=int(cloud["timestamp_ns"]) if cloud is not None else 0,
        pointcloud_camera_sequence=(
            int(cloud["source_camera_sequence"]) if cloud is not None else -1
        ),
        control_timestamp_ns=control_ns,
        pointcloud_camera_received_ns=int(cloud["camera_received_ns"]) if cloud is not None else -1,
        pointcloud_camera_published_ns=(
            int(cloud["camera_published_ns"]) if cloud is not None else -1
        ),
        pointcloud_started_ns=int(cloud["processing_started_ns"]) if cloud is not None else -1,
        pointcloud_published_ns=int(cloud_result[1]) if cloud_result is not None else -1,
    )

    robot.update_observation_health(row)
    return row


def observation_timing(row):
    """Host observation and pipeline clocks in monotonic nanoseconds."""
    return dict(
        observation_ns=int(row.observation_timestamp_ns),
        control_ns=int(row.control_timestamp_ns),
        arm_ns=int(row.arm["timestamp_ns"][0]),
        hand_ns=int(row.hand["timestamp_ns"][0]) if row.hand is not None else -1,
        camera_ns=int(row.camera["timestamp_ns"]) if row.camera is not None else -1,
        pointcloud_ns=int(row.pointcloud_timestamp_ns) if row.pointcloud_timestamp_ns > 0 else -1,
        camera_sequence=int(row.camera["ring_sequence"]) if row.camera is not None else -1,
        pointcloud_camera_sequence=int(row.pointcloud_camera_sequence),
        **{
            f"camera_{name}": int(row.camera[name]) if row.camera is not None else -1
            for name in ("received_ns", "published_ns", "depth_frame_number", "color_frame_number")
        },
        pointcloud_camera_received_ns=int(row.pointcloud_camera_received_ns),
        pointcloud_camera_published_ns=int(row.pointcloud_camera_published_ns),
        pointcloud_started_ns=int(row.pointcloud_started_ns),
        pointcloud_published_ns=int(row.pointcloud_published_ns),
    )
