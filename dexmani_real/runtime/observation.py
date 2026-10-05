"""One current copied observation per control step; local rows own temporal history."""

import time
from collections import deque
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
    result = shared.vr_ring.read_latest()
    if result is None:
        return None
    record = result[0][0]
    return {
        **{name: record[name].copy() for name in record.dtype.names},
        "ring_sequence": result[2],
    }


def read_camera_frame(shared, sequence=None):
    if sequence is None:
        result = shared.camera_ring.read_latest()
        if result is None:
            return None
        header, rgb, depth, sequence = result
    else:
        result = shared.camera_ring.read_sequence(sequence)
        if result is None:
            return None
        header, rgb, depth = result["header"], result["rgb"], result["depth"]
    return {
        "rgb": rgb,
        "depth": depth,
        "ring_sequence": sequence,
        "timestamp_ns": int(header["timestamp_ns"][0]),
        "depth_frame_number": int(header["depth_frame_number"][0]),
        "color_frame_number": int(header["color_frame_number"][0]),
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
    pointcloud_camera_sequence: int = 0


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
):
    state = robot.read_state()
    if state is None:
        return None
    arm, hand = state.arm, state.hand if require_hand else None
    cloud_result = shared.pointcloud_ring.read_latest() if require_pointcloud else None
    if (
        arm is None
        or (require_hand and hand is None)
        or (require_pointcloud and cloud_result is None)
    ):
        return None
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
    if ((require_camera or require_rgb_cloud_identity) and camera is None) or (
        require_vr and vr is None
    ):
        return None
    now = time.monotonic_ns()
    if not sample_is_fresh(arm["timestamp_ns"][0], runtime.arm.feedback_max_age_s, now):
        return None
    if hand is not None and not sample_is_fresh(
        hand["timestamp_ns"][0], runtime.hand.feedback_max_age_s, now
    ):
        return None
    if camera is not None and not sample_is_fresh(
        camera["timestamp_ns"], runtime.camera.max_frame_age_s, now
    ):
        return None
    if cloud is not None and not sample_is_fresh(
        cloud["timestamp_ns"], runtime.camera.max_frame_age_s, now
    ):
        return None
    if vr is not None and not sample_is_fresh(
        vr["recv_ts_ns"], runtime.policy.vr_mapping.stale_threshold_s, now
    ):
        return None
    return ObservationRow(
        _freeze(arm),
        _freeze(hand),
        _freeze(camera),
        _freeze(vr),
        _freeze(cloud["point_cloud"]) if cloud is not None else None,
        now,
        int(cloud["timestamp_ns"]) if cloud is not None else 0,
        int(cloud["source_camera_sequence"]) if cloud is not None else 0,
    )


class ObservationHistory:
    def __init__(self, n_obs_steps):
        self.rows = deque(maxlen=n_obs_steps)

    def clear(self):
        self.rows.clear()

    def append(self, row):
        self.rows.append(row)

    def ready_rows(self):
        return tuple(self.rows) if len(self.rows) == self.rows.maxlen else ()
