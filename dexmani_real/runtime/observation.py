"""按需读取传感器快照，保留各来源时间与相机帧对应关系。"""

import json
import logging
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.robot.commands import RobotError
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


def snapshot_camera_parameters(channels, config, calibration, *, require_extrinsics=False):
    """在 ready 边界复制相机静态参数，之后不再依赖 worker 或共享通道。"""
    if not channels.camera_ready.is_set():
        raise ValueError("camera parameters require a ready camera")
    serial = channels.camera_serial.value.decode("utf-8").strip()
    if config.serial is not None and serial != config.serial:
        raise ValueError("active camera serial differs from configuration")
    color = CameraIntrinsics(**json.loads(channels.camera_intrinsics.value.decode("utf-8")))
    if (color.height, color.width) != (config.height, config.width):
        raise ValueError("camera intrinsics differ from configured resolution")
    transform = None
    if calibration is not None:
        try:
            transform = calibration.for_serial(serial)
        except KeyError:
            if require_extrinsics:
                raise
            logging.getLogger(__name__).warning("No camera extrinsics for active serial %s", serial)
    if require_extrinsics and transform is None:
        raise ValueError("camera consumer requires current extrinsics")
    return CameraSnapshot(serial, color, channels.camera_depth_scale.value, transform)


def read_vr_frame(channels):
    result = channels.vr_ring.read()
    if result is None:
        return None
    record, sequence = result
    return {**{name: record[name] for name in record.dtype.names}, "ring_sequence": sequence}


def read_camera_frame(channels, sequence=None, *, payload="rgbd"):
    fields = {"rgb": ("rgb",), "depth": ("depth",), "rgbd": ("rgb", "depth")}[payload]
    result = channels.camera_ring.read(
        fields=("timestamp_ns", "color_frame_number", *fields), sequence=sequence,
    )
    if result is None:
        return None
    data, sequence = result
    return {
        **{name: data[name] for name in fields},
        "ring_sequence": sequence,
        "timestamp_ns": int(data["timestamp_ns"]),
        "color_frame_number": int(data["color_frame_number"]),
    }


def freeze(value):
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif isinstance(value, dict):
        value = MappingProxyType({k: freeze(v) for k, v in value.items()})
    return value


@dataclass(frozen=True)
class Observation:
    arm: np.ndarray
    hand: np.ndarray | None
    camera: Mapping | None
    vr: Mapping | None
    point_cloud: np.ndarray | None
    observation_timestamp_ns: int
    pointcloud_timestamp_ns: int = 0
    control_timestamp_ns: int = -1


def read_observation(
    channels, robot, *, needs=ObservationNeeds(),
    wait_for_hand=False, cancel_requested=None,
):
    control_ns = time.monotonic_ns()
    state = (
        robot.read_initial_state(cancel_requested=cancel_requested)
        if wait_for_hand else robot.read_state()
    )
    cloud_result = channels.pointcloud_ring.read() if needs.point_cloud else None
    cloud = cloud_result[0] if cloud_result else None
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
    return Observation(
        arm=state.arm, hand=state.hand,
        camera=freeze(camera), vr=freeze(vr),
        point_cloud=freeze(cloud["point_cloud"]) if cloud is not None else None,
        observation_timestamp_ns=time.monotonic_ns(), control_timestamp_ns=control_ns,
        pointcloud_timestamp_ns=int(cloud["timestamp_ns"]) if cloud is not None else 0,
    )


def check_observation(row, runtime, needs, *, require_hand=True, now_ns=None):
    """返回来源问题及绝对提交期限；接收与提交保留各自的纳秒边界。"""
    now = time.monotonic_ns() if now_ns is None else now_ns
    issues = []
    stamp = int(row.arm["timestamp_ns"][0])
    deadlines = [stamp + int(runtime.arm.feedback_max_age_s * 1e9) + 1]
    if not sample_is_fresh(stamp, runtime.arm.feedback_max_age_s, now):
        issues.append("arm_feedback_stale")
    if require_hand and row.hand is None:
        issues.append("hand_feedback_unavailable")
    if row.hand is not None:
        for name in ("timestamp_ns", *(f"tactile_{c}_timestamp_ns" for c in needs.required_tactile)):
            stamp = int(row.hand[name][0])
            deadline = stamp + int(runtime.hand.feedback_max_age_s * 1e9)
            deadlines.append(deadline)
            if require_hand and not 0 < stamp <= now < deadline:
                issues.append(f"hand_{name}_unavailable_or_stale")
    if needs.camera != "none":
        if row.camera is None:
            issues.append("camera_source_unavailable_or_unmatched")
        else:
            stamp = int(row.camera["timestamp_ns"])
            deadlines.append(stamp + int(runtime.camera.max_frame_age_s * 1e9))
            if not sample_is_fresh(stamp, runtime.camera.max_frame_age_s, now):
                issues.append("camera_source_stale")
    if needs.point_cloud:
        if row.point_cloud is None or not np.isfinite(row.point_cloud).all():
            issues.append("pointcloud_unavailable_or_nonfinite")
        else:
            stamp = row.pointcloud_timestamp_ns
            deadlines.append(stamp + int(runtime.camera.max_frame_age_s * 1e9))
            if not sample_is_fresh(stamp, runtime.camera.max_frame_age_s, now):
                issues.append("pointcloud_stale")
    if needs.vr:
        if row.vr is None:
            issues.append("vr_unavailable")
        else:
            stamp = int(row.vr["recv_ts_ns"])
            age = runtime.teleop.vr_stale_threshold_s
            deadlines.append(stamp + int(age * 1e9) + 1)
            if not sample_is_fresh(stamp, age, now):
                issues.append("vr_stale")
    return issues, None if issues else min(deadlines)


def control_deadline_ns(feedback_until_ns, arm_config, *, now_ns=None):
    now = time.monotonic_ns() if now_ns is None else now_ns
    return min(
        now + int(arm_config.feedback_max_age_s * 1e9),
        feedback_until_ns,
    )


def service_feedback(robot, runtime):
    """准备与 HOME 期间，仅由有效运动反馈更新控制期限。"""
    needs = ObservationNeeds()
    row = read_observation(None, robot, needs=needs)
    issues, feedback_until_ns = check_observation(
        row, runtime, needs, require_hand=runtime.hand.enabled,
    )
    if issues:
        raise RobotError("preparation feedback: " + "; ".join(issues))
    robot.arm.update_health(health_until_ns=control_deadline_ns(feedback_until_ns, runtime.arm))
    return row
