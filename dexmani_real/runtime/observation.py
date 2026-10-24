"""One immutable consumption snapshot; consumers admit physical sources independently."""

import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

from dexmani_real.robot.commands import RobotError
from dexmani_real.robot.xhand.feedback import hand_feedback_time_ns
from dexmani_real.utils.timing import sample_is_fresh


@dataclass(frozen=True)
class ObservationNeeds:
    camera: Literal["none", "rgb", "depth", "rgbd"] = "none"
    point_cloud: bool = False
    vr: bool = False
    required_tactile: tuple[str, ...] = ()
    camera_static: bool = False

    @property
    def camera_worker(self):
        return self.camera != "none" or self.point_cloud or self.camera_static

    def union(self, other):
        rgb = self.camera in ("rgb", "rgbd") or other.camera in ("rgb", "rgbd")
        depth = self.camera in ("depth", "rgbd") or other.camera in ("depth", "rgbd")
        return ObservationNeeds(
            camera="rgbd" if rgb and depth else "rgb" if rgb else "depth" if depth else "none",
            point_cloud=self.point_cloud or other.point_cloud,
            vr=self.vr or other.vr,
            required_tactile=tuple(
                channel for channel in ("aggregate", "dense")
                if channel in self.required_tactile or channel in other.required_tactile
            ),
            camera_static=self.camera_static or other.camera_static,
        )


def read_vr_frame(channels):
    result = channels.vr_ring.read_latest(use_last_good=True)
    if result is None:
        return None
    record = result[0][0]
    return {**{name: record[name] for name in record.dtype.names}, "ring_sequence": result[2]}


def read_camera_frame(channels, sequence=None, *, payload="rgbd"):
    fields = {"rgb": ("rgb",), "depth": ("depth",), "rgbd": ("rgb", "depth")}[payload]
    result = channels.camera_ring.read_fields(("header", *fields), sequence=sequence)
    if result is None:
        return None
    data, _, sequence = result
    return {
        **{name: data[name] for name in fields},
        "ring_sequence": sequence,
        "timestamp_ns": int(data["header"]["timestamp_ns"]),
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
    control_timestamp_ns: int = -1
    camera_sequence: int = -1


def read_observation(
    channels, robot, *, needs=ObservationNeeds(),
    wait_for_hand=False, cancel_requested=None,
):
    control_ns = time.monotonic_ns()
    state = (
        robot.read_initial_state(cancel_requested=cancel_requested)
        if wait_for_hand else robot.read_state()
    )
    cloud_result = (
        channels.pointcloud_ring.read_latest(use_last_good=True) if needs.point_cloud else None
    )
    cloud = cloud_result[0][0] if cloud_result else None
    camera = None
    if needs.camera != "none":
        if needs.point_cloud:
            if cloud is not None:
                camera = read_camera_frame(
                    channels, int(cloud["source_camera_sequence"]), payload=needs.camera,
                )
        else:
            camera = read_camera_frame(channels, payload=needs.camera)
    vr = read_vr_frame(channels) if needs.vr else None
    return ObservationRow(
        arm=freeze(state.arm), hand=freeze(state.hand),
        camera=freeze(camera), vr=freeze(vr),
        point_cloud=freeze(cloud["point_cloud"]) if cloud is not None else None,
        observation_timestamp_ns=time.monotonic_ns(), control_timestamp_ns=control_ns,
        pointcloud_timestamp_ns=int(cloud["timestamp_ns"]) if cloud is not None else 0,
        camera_sequence=(
            int(cloud["source_camera_sequence"]) if cloud is not None
            else int(camera["ring_sequence"]) if camera is not None else -1
        ),
    )


def observation_issues(row, runtime, needs, *, require_hand=True, now_ns=None):
    now = time.monotonic_ns() if now_ns is None else now_ns
    issues = []
    if not sample_is_fresh(row.arm["timestamp_ns"][0], runtime.arm.feedback_max_age_s, now):
        issues.append("arm_feedback_stale")
    if require_hand and row.hand is None:
        issues.append("hand_feedback_unavailable")
    if require_hand and row.hand is not None:
        for name in ("timestamp_ns", *(f"tactile_{c}_timestamp_ns" for c in needs.required_tactile)):
            stamp = int(row.hand[name][0])
            if not 0 < stamp <= now < stamp + int(runtime.hand.feedback_max_age_s * 1e9):
                issues.append(f"hand_{name}_unavailable_or_stale")
    if needs.camera != "none":
        if row.camera is None:
            issues.append("camera_source_unavailable_or_unmatched")
        elif not sample_is_fresh(row.camera["timestamp_ns"], runtime.camera.max_frame_age_s, now):
            issues.append("camera_source_stale")
    if needs.point_cloud:
        if row.point_cloud is None or not np.isfinite(row.point_cloud).all():
            issues.append("pointcloud_unavailable_or_nonfinite")
        elif not sample_is_fresh(row.pointcloud_timestamp_ns, runtime.camera.max_frame_age_s, now):
            issues.append("pointcloud_stale")
    if needs.vr:
        if row.vr is None:
            issues.append("vr_unavailable")
        elif not sample_is_fresh(row.vr["recv_ts_ns"], runtime.teleop.vr_mapping.stale_threshold_s, now):
            issues.append("vr_stale")
    return issues


def feedback_deadline_ns(row, runtime, *, needs=ObservationNeeds()):
    deadlines = [int(row.arm["timestamp_ns"][0]) + int(runtime.arm.feedback_max_age_s * 1e9) + 1]
    if row.hand is not None:
        deadlines.append(
            hand_feedback_time_ns(row.hand, needs.required_tactile)
            + int(runtime.hand.feedback_max_age_s * 1e9)
        )
    if needs.camera != "none":
        deadlines.append(int(row.camera["timestamp_ns"]) + int(runtime.camera.max_frame_age_s * 1e9))
    if needs.point_cloud:
        deadlines.append(row.pointcloud_timestamp_ns + int(runtime.camera.max_frame_age_s * 1e9))
    if needs.vr:
        deadlines.append(int(row.vr["recv_ts_ns"]) + int(runtime.teleop.vr_mapping.stale_threshold_s * 1e9) + 1)
    return min(deadlines)


def control_deadline_ns(row, runtime, needs, *, now_ns=None):
    now = time.monotonic_ns() if now_ns is None else now_ns
    return min(
        now + int(runtime.arm.feedback_max_age_s * 1e9),
        feedback_deadline_ns(row, runtime, needs=needs),
    )


def service_feedback(robot, runtime):
    """Coordinator preparation/HOME service; only motion feedback renews its bounded health."""
    needs = ObservationNeeds()
    row = read_observation(None, robot, needs=needs)
    issues = observation_issues(row, runtime, needs, require_hand=runtime.hand.enabled)
    if issues:
        raise RobotError("preparation feedback: " + "; ".join(issues))
    robot.arm.update_health(health_until_ns=control_deadline_ns(row, runtime, needs))
    return row
