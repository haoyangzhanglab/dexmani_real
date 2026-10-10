"""One current copied observation per control step; local rows own temporal history."""

import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


def sample_is_fresh(timestamp_ns, max_age_s, now_ns=None):
    now = time.monotonic_ns() if now_ns is None else now_ns
    return 0 < int(timestamp_ns) <= now and now - int(timestamp_ns) <= int(max_age_s * 1e9)


def feedback_deadline_ns(row, runtime, *, include_vr=False):
    """Exclusive send deadline of the feedback actually used to prepare a target.

    Sampling owns rejection of missing/future timestamps; expiration during
    preparation is enforced again immediately before each target SDK call.
    """
    deadlines = [int(row.arm["timestamp_ns"][0]) + int(runtime.arm.feedback_max_age_s * 1e9) + 1]
    if row.hand is not None:
        deadlines.append(
            int(row.hand["timestamp_ns"][0]) + int(runtime.hand.feedback_max_age_s * 1e9) + 1
        )
    if include_vr:
        deadlines.append(
            int(row.vr["recv_ts_ns"]) + int(runtime.policy.vr_mapping.stale_threshold_s * 1e9) + 1
        )
    return min(deadlines)


def read_vr_frame(shared):
    result = shared.sensors.vr_ring.read_latest()
    if result is None:
        return None
    record = result[0][0]
    return {
        **{name: record[name].copy() for name in record.dtype.names},
        "ring_sequence": result[2],
    }


def read_camera_frame(shared, sequence=None):
    ring = shared.sensors.camera_ring
    result = ring.read_latest_uncached() if sequence is None else ring.read_sequence(sequence)
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


def _freeze(value):
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif isinstance(value, dict):
        value = MappingProxyType({k: _freeze(v) for k, v in value.items()})
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
    shared,
    runtime,
    robot,
    *,
    require_hand=True,
    require_camera=False,
    require_vr=False,
    require_pointcloud=False,
    require_rgb_cloud_identity=False,
    issues=None,
):
    def unavailable(reason):
        if issues is not None:
            issues.append(reason)
        return None

    control_ns = time.monotonic_ns()
    state = robot.read_state()
    if state is None:
        return unavailable("feedback_unavailable")
    arm, hand = state.arm, state.hand if require_hand else None
    cloud_result = shared.sensors.pointcloud_ring.read_latest() if require_pointcloud else None
    if (
        arm is None
        or (require_hand and hand is None)
        or (require_pointcloud and cloud_result is None)
    ):
        return unavailable("feedback_or_pointcloud_unavailable")
    cloud = cloud_result[0][0] if cloud_result else None
    if require_rgb_cloud_identity:
        camera = (
            read_camera_frame(shared, int(cloud["source_camera_sequence"]))
            if cloud is not None
            else None
        )
    else:
        camera = read_camera_frame(shared) if require_camera else None
    vr = read_vr_frame(shared) if require_vr else None
    if (require_camera or require_rgb_cloud_identity) and camera is None:
        return unavailable("camera_source_unavailable_or_unmatched")
    if require_vr and vr is None:
        return unavailable("vr_unavailable")
    now = time.monotonic_ns()
    if not sample_is_fresh(arm["timestamp_ns"][0], runtime.arm.feedback_max_age_s, now):
        return unavailable("arm_feedback_stale")
    if hand is not None and not sample_is_fresh(
        hand["timestamp_ns"][0], runtime.hand.feedback_max_age_s, now
    ):
        return unavailable("hand_feedback_stale")
    if camera is not None and not sample_is_fresh(
        camera["timestamp_ns"], runtime.camera.max_frame_age_s, now
    ):
        return unavailable("camera_source_stale")
    if cloud is not None and not sample_is_fresh(
        cloud["timestamp_ns"], runtime.camera.max_frame_age_s, now
    ):
        return unavailable("pointcloud_stale")
    if vr is not None and not sample_is_fresh(
        vr["recv_ts_ns"], runtime.policy.vr_mapping.stale_threshold_s, now
    ):
        return unavailable("vr_stale")
    return ObservationRow(
        _freeze(arm),
        _freeze(hand),
        _freeze(camera),
        _freeze(vr),
        _freeze(cloud["point_cloud"]) if cloud is not None else None,
        now,
        int(cloud["timestamp_ns"]) if cloud is not None else 0,
        int(cloud["source_camera_sequence"]) if cloud is not None else -1,
        control_ns,
        int(cloud["camera_received_ns"]) if cloud is not None else -1,
        int(cloud["camera_published_ns"]) if cloud is not None else -1,
        int(cloud["processing_started_ns"]) if cloud is not None else -1,
        int(cloud_result[1]) if cloud_result is not None else -1,
    )


def observation_timing(row):
    """Source and pipeline clocks; SDK times retain their explicitly recorded domains."""
    return dict(
        observation_ns=int(row.observation_timestamp_ns),
        control_ns=int(row.control_timestamp_ns),
        arm_ns=int(row.arm["timestamp_ns"][0]),
        hand_ns=int(row.hand["timestamp_ns"][0]) if row.hand is not None else -1,
        camera_ns=int(row.camera["timestamp_ns"]) if row.camera is not None else -1,
        pointcloud_ns=int(row.pointcloud_timestamp_ns) if row.pointcloud_timestamp_ns > 0 else -1,
        camera_sequence=int(row.camera["ring_sequence"]) if row.camera is not None else -1,
        pointcloud_camera_sequence=int(row.pointcloud_camera_sequence),
        **{f"camera_{name}": int(row.camera[name]) if row.camera is not None else -1
           for name in ("received_ns", "published_ns", "depth_sdk_ns", "color_sdk_ns",
                        "depth_timestamp_domain", "color_timestamp_domain",
                        "depth_frame_number", "color_frame_number")},
        pointcloud_camera_received_ns=int(row.pointcloud_camera_received_ns),
        pointcloud_camera_published_ns=int(row.pointcloud_camera_published_ns),
        pointcloud_started_ns=int(row.pointcloud_started_ns),
        pointcloud_published_ns=int(row.pointcloud_published_ns),
    )
