"""按需读取传感器快照，保留各来源时间与相机帧对应关系。"""

import time
from dataclasses import dataclass

import numpy as np

from dexmani_real.ipc.sensors import read_camera_frame, read_vr_frame
from dexmani_real.utils.timing import sample_is_fresh


@dataclass(frozen=True)
class ObservationNeeds:
    camera_fields: tuple[str, ...] = ()
    point_cloud: bool = False
    vr: bool = False
    required_tactile: tuple[str, ...] = ()

    @property
    def camera_worker(self):
        return bool(self.camera_fields) or self.point_cloud

    def union(self, other):
        return ObservationNeeds(
            camera_fields=tuple(
                name for name in ("rgb", "depth")
                if name in self.camera_fields or name in other.camera_fields
            ),
            point_cloud=self.point_cloud or other.point_cloud,
            vr=self.vr or other.vr,
            required_tactile=tuple(
                channel for channel in ("aggregate", "dense")
                if channel in self.required_tactile or channel in other.required_tactile
            ),
        )


@dataclass(frozen=True)
class Observation:
    arm: np.ndarray
    hand: np.ndarray | None
    camera: dict | None
    vr: dict | None
    point_cloud: np.ndarray | None
    observation_timestamp_ns: int
    pointcloud_timestamp_ns: int = 0
    control_timestamp_ns: int = -1


def read_observation(channels, robot, *, needs=ObservationNeeds()):
    control_ns = time.monotonic_ns()
    state = robot.read_state()
    cloud_result = None
    if needs.point_cloud:
        fields = ("timestamp_ns", "source_camera_sequence", "point_cloud", *needs.camera_fields)
        cloud_result = channels.pointcloud_ring.read(fields=fields)
    cloud = cloud_result[0] if cloud_result else None
    camera = None
    if needs.camera_fields:
        if needs.point_cloud:
            if cloud is not None:
                camera = {
                    "timestamp_ns": int(cloud["timestamp_ns"]),
                    "ring_sequence": int(cloud["source_camera_sequence"]),
                    **{name: cloud[name] for name in needs.camera_fields},
                }
        else:
            camera = read_camera_frame(channels, fields=needs.camera_fields)
    vr = read_vr_frame(channels) if needs.vr else None
    return Observation(
        arm=state.arm, hand=state.hand,
        camera=camera, vr=vr,
        point_cloud=cloud["point_cloud"] if cloud is not None else None,
        observation_timestamp_ns=time.monotonic_ns(), control_timestamp_ns=control_ns,
        pointcloud_timestamp_ns=int(cloud["timestamp_ns"]) if cloud is not None else 0,
    )


def sources_available(row, needs):
    """只区分尚无完整快照；已有快照的过期仍由 TTL 检查负责。"""
    return (not needs.vr or row.vr is not None) and (
        not needs.camera_fields or row.camera is not None
    ) and (not needs.point_cloud or row.point_cloud is not None)


def missing_tactile_channels(row, channels):
    """SDK 已校验内部快照的形状和数值；这里只检查真实来源，年龄由消费者检查。"""
    missing = []
    for channel in channels:
        if row.hand is None:
            missing.append(channel)
            continue
        stamp = int(row.hand[f"tactile_{channel}_timestamp_ns"][0])
        if stamp <= 0 or stamp > row.observation_timestamp_ns:
            missing.append(channel)
    return missing


def check_observation(row, runtime, needs, *, require_hand=True, now_ns=None, allow_tactile_hold=False):
    """按消费者需求检查来源；发布前可重新检查长计算期间的年龄。"""
    now = time.monotonic_ns() if now_ns is None else now_ns
    issues = []
    stamp = int(row.arm["timestamp_ns"][0])
    if not sample_is_fresh(stamp, runtime.arm.feedback_max_age_s, now):
        issues.append("arm_feedback_stale")
    if require_hand and row.hand is None:
        issues.append("hand_feedback_unavailable")
    if require_hand and row.hand is not None:
        names = ("timestamp_ns",) if allow_tactile_hold else (
            "timestamp_ns", *(f"tactile_{c}_timestamp_ns" for c in needs.required_tactile)
        )
        for name in names:
            stamp = int(row.hand[name][0])
            if not sample_is_fresh(stamp, runtime.hand.feedback_max_age_s, now):
                issues.append(f"hand_{name}_unavailable_or_stale")
    if require_hand and allow_tactile_hold:
        issues.extend(f"hand_tactile_{channel}_unavailable"
                      for channel in missing_tactile_channels(row, needs.required_tactile))
    if needs.camera_fields:
        if row.camera is None:
            issues.append("camera_source_unavailable_or_unmatched")
        else:
            stamp = int(row.camera["timestamp_ns"])
            if not needs.point_cloud and not sample_is_fresh(
                stamp, runtime.camera.max_frame_age_s, now,
            ):
                issues.append("camera_source_stale")
    if needs.point_cloud:
        if row.point_cloud is None:
            issues.append("pointcloud_unavailable")
        else:
            stamp = row.pointcloud_timestamp_ns
            if not sample_is_fresh(stamp, runtime.camera.max_frame_age_s, now):
                issues.append("pointcloud_stale")
    if needs.vr:
        if row.vr is None:
            issues.append("vr_unavailable")
        else:
            stamp = int(row.vr["recv_ts_ns"])
            age = runtime.teleop.vr_stale_threshold_s
            if not sample_is_fresh(stamp, age, now):
                issues.append("vr_stale")
    return issues
