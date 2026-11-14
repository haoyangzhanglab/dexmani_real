"""按需读取传感器快照，保留各来源时间与相机帧对应关系。"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dexmani_real.ipc.sensors import read_camera_frame
from dexmani_real.utils.timing import sample_is_fresh

_RECOVERABLE_HOI_ISSUES = frozenset({
    "hoi_unavailable",
    "hoi_pose_invalid",
    "hoi_stale",
    "hoi_joint_alignment_unavailable",
})
_HOI_METADATA_FIELDS = ("episode_id", "timestamp_ns", "source_camera_sequence", "valid")


@dataclass(frozen=True)
class ObservationNeeds:
    camera_fields: tuple[str, ...] = ()
    hoi: bool = False
    hoi_joint_devices: tuple[str, ...] = ()
    point_cloud: bool = False
    vr: bool = False
    required_tactile: tuple[str, ...] = ()

    @property
    def camera_worker(self):
        return bool(self.camera_fields) or self.point_cloud or self.hoi

    def union(self, other):
        return ObservationNeeds(
            camera_fields=tuple(
                name
                for name in ("rgb", "depth")
                if name in self.camera_fields or name in other.camera_fields
            ),
            hoi=self.hoi or other.hoi,
            hoi_joint_devices=tuple(
                name
                for name in ("arm", "hand")
                if name in self.hoi_joint_devices or name in other.hoi_joint_devices
            ),
            point_cloud=self.point_cloud or other.point_cloud,
            vr=self.vr or other.vr,
            required_tactile=tuple(
                channel
                for channel in ("aggregate", "dense")
                if channel in self.required_tactile or channel in other.required_tactile
            ),
        )


RAW_RECORDING_NEEDS = ObservationNeeds(
    camera_fields=("rgb", "depth"),
    required_tactile=("aggregate", "dense"),
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
    hoi: dict | None = None
    aligned_joints: dict | None = None


def read_observation(channels, robot, *, needs: ObservationNeeds, joint_gap_s=0.10) -> Observation:
    control_ns = time.monotonic_ns()
    state = robot.read_state()
    cloud_result = None
    if needs.point_cloud:
        fields = ("timestamp_ns", "source_camera_sequence", "point_cloud", *needs.camera_fields)
        cloud_result = channels.pointcloud_ring.read(fields=fields)
    cloud = cloud_result[0] if cloud_result else None
    hoi = None
    if needs.hoi:
        hoi = _read_hoi(channels, fields=("object_pointcloud", *needs.camera_fields))
    camera = None
    if needs.camera_fields:
        if needs.hoi or needs.point_cloud:
            source = hoi if needs.hoi else cloud
            if source is not None:
                camera = {
                    "timestamp_ns": int(source["timestamp_ns"]),
                    "ring_sequence": int(source["source_camera_sequence"]),
                    **{name: source[name] for name in needs.camera_fields},
                }
        else:
            camera = read_camera_frame(channels, fields=needs.camera_fields)
    vr_result = channels.vr_ring.read() if needs.vr else None
    aligned_joints = None
    if needs.hoi_joint_devices and hoi is not None:
        aligned_joints = robot.aligned_joints(int(hoi["timestamp_ns"]), joint_gap_s)
    return Observation(
        arm=state.arm,
        hand=state.hand,
        hoi=hoi,
        aligned_joints=aligned_joints,
        camera=camera,
        vr=vr_result[0] if vr_result is not None else None,
        point_cloud=cloud["point_cloud"] if cloud is not None else None,
        observation_timestamp_ns=time.monotonic_ns(),
        control_timestamp_ns=control_ns,
        pointcloud_timestamp_ns=int(cloud["timestamp_ns"]) if cloud is not None else 0,
    )


def check_device_feedback(row, runtime, *, now_ns=None):
    """设备事实；长计算后使用新的当前时间重新检查。"""
    now = time.monotonic_ns() if now_ns is None else now_ns
    issues = []
    if not sample_is_fresh(int(row.arm["timestamp_ns"][0]), runtime.arm.feedback_max_age_s, now):
        issues.append("arm_feedback_stale")
    if runtime.hand.enabled:
        if row.hand is None:
            issues.append("hand_feedback_unavailable")
        elif not sample_is_fresh(int(row.hand["timestamp_ns"][0]), runtime.hand.feedback_max_age_s, now):
            issues.append("hand_timestamp_ns_unavailable_or_stale")
    return issues


def check_observation(row, runtime, needs, *, now_ns=None):
    """模型或录制所需模态，不重复检查关节反馈。"""
    now = time.monotonic_ns() if now_ns is None else now_ns
    issues: list[str] = []
    for channel in needs.required_tactile:
        stamp = int(row.hand[f"tactile_{channel}_timestamp_ns"][0]) if row.hand is not None else 0
        if not sample_is_fresh(stamp, runtime.hand.feedback_max_age_s, now):
            issues.append(f"hand_tactile_{channel}_timestamp_ns_unavailable_or_stale")
    # read_observation 已让派生快照携带所需同源图像，只检查对应点云/HOI 来源。
    if needs.camera_fields and not (needs.point_cloud or needs.hoi):
        if row.camera is None:
            issues.append("camera_source_unavailable_or_unmatched")
        else:
            stamp = int(row.camera["timestamp_ns"])
            if not sample_is_fresh(stamp, runtime.camera.max_frame_age_s, now):
                issues.append("camera_source_stale")
    if needs.point_cloud:
        if row.point_cloud is None:
            issues.append("pointcloud_unavailable")
        else:
            stamp = row.pointcloud_timestamp_ns
            if not sample_is_fresh(stamp, runtime.camera.max_frame_age_s, now):
                issues.append("pointcloud_stale")
    if needs.hoi:
        issues.extend(_hoi_issues(row.hoi, runtime, now))
        if needs.hoi_joint_devices and row.aligned_joints is None:
            issues.append("hoi_joint_alignment_unavailable")
    if needs.vr:
        if row.vr is None:
            issues.append("vr_unavailable")
        else:
            stamp = int(row.vr["recv_ts_ns"])
            age = runtime.teleop.vr_stale_threshold_s
            if not sample_is_fresh(stamp, age, now):
                issues.append("vr_stale")
    return issues


def _read_hoi(channels, *, fields=()):
    """完整观测与状态读取共用请求边界；请求由父控制流程独占写入。"""
    request = channels.hoi_request.read()
    if request is None or int(request[0]["episode_id"]) == 0:
        return None
    result = channels.hoi_ring.read(fields=(*_HOI_METADATA_FIELDS, *fields))
    if result is None:
        return None
    data, _ = result
    command, _ = request
    if (data["episode_id"] != command["episode_id"]
            or data["timestamp_ns"] <= command["after_ns"]
            or data["source_camera_sequence"] <= command["after_sequence"]):
        return None
    return data


def _hoi_issues(hoi, runtime, now_ns):
    if hoi is None:
        return ["hoi_unavailable"]
    issues = []
    if not hoi["valid"]:
        issues.append("hoi_pose_invalid")
    if not sample_is_fresh(int(hoi["timestamp_ns"]), runtime.execution.hoi_max_age_s, now_ns):
        issues.append("hoi_stale")
    return issues


def check_hoi_status(channels, runtime, *, now_ns=None) -> list[str]:
    """仅读取当前请求及 HOI 元数据，不读取机器人、点云或图像。"""
    hoi = _read_hoi(channels)
    now = time.monotonic_ns() if now_ns is None else now_ns
    return _hoi_issues(hoi, runtime, now)


def is_recoverable_hoi_failure(issues: Sequence[str]) -> bool:
    """只有非空且完全由 HOI 恢复问题组成的列表可以暂停重试。"""
    return bool(issues) and all(issue in _RECOVERABLE_HOI_ISSUES for issue in issues)
