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


def read_observation(
    shared,
    runtime,
    *,
    require_hand=True,
    require_camera=False,
    require_vr=False,
    require_pointcloud=False,
    require_rgb_cloud_identity=False,
):
    arm_result = shared.arm_state_ring.read_latest()
    hand_result = shared.hand_state_ring.read_latest() if require_hand else None
    cloud_result = shared.pointcloud_ring.read_latest() if require_pointcloud else None
    if (
        arm_result is None
        or (require_hand and hand_result is None)
        or (require_pointcloud and cloud_result is None)
    ):
        return None
    arm = arm_result[0]
    hand = hand_result[0] if hand_result else None
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
    )


class ObservationHistory:
    def __init__(self, n_obs_steps, control_dt_s):
        self.rows = deque(maxlen=n_obs_steps)
        self.max_gap_ns = int(2 * control_dt_s * 1e9)

    def clear(self):
        self.rows.clear()

    def append(self, row):
        if (
            self.rows
            and row.observation_timestamp_ns - self.rows[-1].observation_timestamp_ns
            > self.max_gap_ns
        ):
            self.clear()
        self.rows.append(row)

    def padded(self):
        if not self.rows:
            return ()
        return (self.rows[0],) * (self.rows.maxlen - len(self.rows)) + tuple(self.rows)
