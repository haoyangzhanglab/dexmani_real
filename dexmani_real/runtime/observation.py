"""One immutable observation per control step, retaining each source's host clock."""

import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

from dexmani_real.robot.xhand.feedback import hand_feedback_time_ns
from dexmani_real.utils.timing import sample_is_fresh


@dataclass(frozen=True)
class ObservationNeeds:
    """Coordinator copies and required sources, shared with device assembly."""

    camera: Literal["none", "header", "rgb", "depth", "rgbd"] = "none"
    point_cloud: bool = False
    vr: bool = False
    required_tactile: tuple[str, ...] = ("aggregate", "dense")

    @classmethod
    def for_policy(cls, fields, *, record_raw):
        fields = set(fields)
        return cls(
            camera=(
                "rgbd" if record_raw or {"rgb", "depth"} <= fields
                else "rgb" if "rgb" in fields
                else "depth" if "depth" in fields
                else "header" if fields & {"camera_intrinsic", "camera_extrinsic"}
                else "none"
            ),
            point_cloud="point_cloud" in fields,
            required_tactile=tuple(
                channel
                for field, channel in (
                    ("finger_tactile_force", "aggregate"),
                    ("taxel_tactile_force", "dense"),
                )
                if record_raw or field in fields
            ),
        )

    @property
    def camera_worker(self):
        # Cloud-only still needs a producer, but no coordinator RGB-D copy.
        return self.camera != "none" or self.point_cloud

    @property
    def paired_camera(self):
        return self.camera != "none" and self.point_cloud


def feedback_deadline_ns(row, runtime, *, include_vr=False):
    """Exclusive deadline for feedback used to prepare a target; recheck expiry at each SDK send boundary."""
    deadlines = [int(row.arm["timestamp_ns"][0]) + int(runtime.arm.feedback_max_age_s * 1e9) + 1]
    if row.hand is not None:
        deadlines.append(
            hand_feedback_time_ns(row.hand, row.required_tactile)
            + int(runtime.hand.feedback_max_age_s * 1e9)
        )
    if include_vr:
        deadlines.append(
            int(row.vr["recv_ts_ns"]) + int(runtime.teleop.vr_mapping.stale_threshold_s * 1e9) + 1
        )
    return min(deadlines)


def read_vr_frame(channels):
    result = channels.vr_ring.read_latest(use_last_good=True)
    if result is None:
        return None
    record = result[0][0]
    return {
        **{name: record[name] for name in record.dtype.names},
        "ring_sequence": result[2],
    }


def read_camera_frame(channels, sequence=None, *, payload="rgbd"):
    ring = channels.camera_ring
    fields = {"header": (), "rgb": ("rgb",), "depth": ("depth",), "rgbd": ("rgb", "depth")}[payload]
    result = ring.read_fields(("header", *fields), sequence=sequence)
    if result is None:
        return None
    data, _, sequence = result
    frame = data[0]
    return {
        **{name: frame[name] for name in fields},
        "ring_sequence": sequence,
        "timestamp_ns": int(frame["header"]["timestamp_ns"]),
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
    required_tactile: tuple[str, ...] = ("aggregate", "dense")


def read_observation(
    channels,
    runtime,
    robot,
    *,
    needs=ObservationNeeds(),
    require_hand=True,
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
        channels.pointcloud_ring.read_latest(use_last_good=True)
        if needs.point_cloud
        else None
    )
    if (require_hand and hand is None) or (needs.point_cloud and cloud_result is None):
        unavailable.append("feedback_or_pointcloud_unavailable")
    cloud = cloud_result[0][0] if cloud_result else None
    if needs.paired_camera:
        camera = (
            read_camera_frame(channels, int(cloud["source_camera_sequence"]), payload=needs.camera)
            if cloud is not None
            else None
        )
    else:
        camera = (
            read_camera_frame(channels, payload=needs.camera)
            if needs.camera != "none" else None
        )
    vr = read_vr_frame(channels) if needs.vr else None
    if needs.camera != "none" and camera is None:
        unavailable.append("camera_source_unavailable_or_unmatched")
    if needs.vr and vr is None:
        unavailable.append("vr_unavailable")
    now = time.monotonic_ns()
    if not sample_is_fresh(arm["timestamp_ns"][0], runtime.arm.feedback_max_age_s, now):
        unavailable.append("arm_feedback_stale")
    if hand is not None:
        hand_ns = hand_feedback_time_ns(hand, needs.required_tactile)
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
        vr["recv_ts_ns"], runtime.teleop.vr_mapping.stale_threshold_s, now
    ):
        unavailable.append("vr_stale")
    if unavailable:
        if issues is not None:
            issues.extend(unavailable)
        return None
    return ObservationRow(
        arm=freeze(arm),
        hand=freeze(hand),
        camera=freeze(camera),
        vr=freeze(vr),
        point_cloud=freeze(cloud["point_cloud"]) if cloud is not None else None,
        observation_timestamp_ns=now,
        pointcloud_timestamp_ns=int(cloud["timestamp_ns"]) if cloud is not None else 0,
        control_timestamp_ns=control_ns,
        required_tactile=needs.required_tactile,
    )


def upstream_deadline_ns(row, runtime):
    """Compute the executor upstream fence from original source times, not processing time."""
    deadlines = []
    if row.camera is not None:
        deadlines.append(
            int(row.camera["timestamp_ns"]) + int(runtime.camera.max_frame_age_s * 1e9)
        )
    if row.pointcloud_timestamp_ns > 0:
        deadlines.append(
            row.pointcloud_timestamp_ns + int(runtime.camera.max_frame_age_s * 1e9)
        )
    if row.vr is not None:
        deadlines.append(
            int(row.vr["recv_ts_ns"])
            + int(runtime.teleop.vr_mapping.stale_threshold_s * 1e9)
        )
    # Arm freshness is independently checked against the executor's live snapshot.
    return min(deadlines) if deadlines else 2**63 - 1
