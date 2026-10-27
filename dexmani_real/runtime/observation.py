"""按需读取传感器快照，保留各来源时间与相机帧对应关系。"""

import time
from dataclasses import dataclass

import numpy as np

from dexmani_real.ipc.sensors import read_camera_frame, read_vr_frame
from dexmani_real.robot.commands import RobotError
from dexmani_real.utils.timing import sample_is_fresh


@dataclass(frozen=True)
class ObservationNeeds:
    camera_fields: tuple[str, ...] = ()
    point_cloud: bool = False
    vr: bool = False
    required_tactile: tuple[str, ...] = ()
    camera_static: bool = False

    @property
    def camera_worker(self):
        return bool(self.camera_fields) or self.point_cloud or self.camera_static

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
            camera_static=self.camera_static or other.camera_static,
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
    cloud_result = channels.pointcloud_ring.read() if needs.point_cloud else None
    cloud = cloud_result[0] if cloud_result else None
    camera = None
    if needs.camera_fields:
        if needs.point_cloud:
            if cloud is not None:
                camera = read_camera_frame(
                    channels, int(cloud["source_camera_sequence"]), fields=needs.camera_fields,
                )
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


def check_observation(row, runtime, needs, *, require_hand=True, now_ns=None):
    """返回来源问题及绝对提交期限；接收与提交保留各自的纳秒边界。"""
    now = time.monotonic_ns() if now_ns is None else now_ns
    issues = []
    stamp = int(row.arm["timestamp_ns"][0])
    arm_age_ns = int(runtime.arm.feedback_max_age_s * 1e9)
    deadlines = [stamp + arm_age_ns + 1, now + arm_age_ns]
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
    if needs.camera_fields:
        if row.camera is None:
            issues.append("camera_source_unavailable_or_unmatched")
        else:
            stamp = int(row.camera["timestamp_ns"])
            deadlines.append(stamp + int(runtime.camera.max_frame_age_s * 1e9))
            if not sample_is_fresh(stamp, runtime.camera.max_frame_age_s, now):
                issues.append("camera_source_stale")
    if needs.point_cloud:
        if row.point_cloud is None:
            issues.append("pointcloud_unavailable")
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


def service_feedback(robot, runtime):
    """准备与 HOME 期间，仅由有效运动反馈更新控制期限。"""
    needs = ObservationNeeds()
    row = read_observation(None, robot, needs=needs)
    issues, feedback_until_ns = check_observation(
        row, runtime, needs, require_hand=runtime.hand.enabled,
    )
    if issues:
        raise RobotError("preparation feedback: " + "; ".join(issues))
    robot.arm.update_health(health_until_ns=feedback_until_ns)
    return row
