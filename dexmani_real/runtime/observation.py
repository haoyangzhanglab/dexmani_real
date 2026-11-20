"""按需读取传感器快照，保留各来源时间与相机帧对应关系。"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dexmani_real.config.perception import DEFAULT_HOI_MAX_JOINT_GAP_S
from dexmani_real.ipc.sensors import read_camera_frame
from dexmani_real.robot.xhand.feedback import expire_tactile, tactile_validity
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


RAW_RECORDING_NEEDS = ObservationNeeds(
    camera_fields=("rgb", "depth"),
)


@dataclass(frozen=True)
class Observation:
    arm: np.ndarray
    hand: np.ndarray | None
    camera: dict | None
    vr: dict | None
    point_cloud: np.ndarray | None
    observation_timestamp_ns: int
    pointcloud_timestamp_ns: int | None = None
    control_timestamp_ns: int = -1
    hoi: dict | None = None
    aligned_joints: dict | None = None


def read_observation(
    channels, robot, *, needs: ObservationNeeds,
    joint_gap_s=DEFAULT_HOI_MAX_JOINT_GAP_S, include_visual_payloads=True,
) -> Observation:
    """按需读取快照；就绪检查可只读取视觉来源元数据。"""
    control_ns = time.monotonic_ns()
    state = robot.read_state()
    camera_fields = needs.camera_fields if include_visual_payloads else ()
    cloud = hoi = None
    if needs.hoi:
        hoi = _read_hoi(
            channels, fields=("object_pointcloud", *camera_fields) if include_visual_payloads else (),
        )
    elif needs.point_cloud:
        cloud_fields = ("point_cloud", *camera_fields) if include_visual_payloads else ()
        cloud_result = channels.pointcloud_ring.read(
            fields=("timestamp_ns", "source_camera_sequence", *cloud_fields),
        )
        cloud = cloud_result[0] if cloud_result is not None else None
    camera = None
    if needs.camera_fields:
        if needs.hoi or needs.point_cloud:
            source = hoi if needs.hoi else cloud
            if source is not None:
                camera = {
                    "timestamp_ns": int(source["timestamp_ns"]),
                    "ring_sequence": int(source["source_camera_sequence"]),
                    **{name: source[name] for name in camera_fields},
                }
        else:
            camera = read_camera_frame(channels, fields=camera_fields)
    vr_result = channels.vr_ring.read() if needs.vr else None
    aligned_joints = None
    if needs.hoi_joint_devices and hoi is not None and hoi["valid"]:
        aligned_joints = robot.aligned_joints(int(hoi["timestamp_ns"]), joint_gap_s)
    observed_ns = time.monotonic_ns()
    hand = expire_tactile(state.hand, max_age_s=robot.hand_config.tactile_max_age_s, now_ns=observed_ns)
    return Observation(
        arm=state.arm,
        hand=hand,
        hoi=hoi,
        aligned_joints=aligned_joints,
        camera=camera,
        vr=vr_result[0] if vr_result is not None else None,
        point_cloud=cloud["point_cloud"] if cloud is not None and include_visual_payloads else None,
        observation_timestamp_ns=observed_ns,
        control_timestamp_ns=control_ns,
        pointcloud_timestamp_ns=int(cloud["timestamp_ns"]) if cloud is not None else None,
    )


def check_device_feedback(row, runtime, *, now_ns=None):
    """设备事实；长计算后使用新的当前时间重新检查。"""
    now = time.monotonic_ns() if now_ns is None else now_ns
    issues = []
    if not sample_is_fresh(row.arm["timestamp_ns"][0], runtime.arm.feedback_max_age_s, now):
        issues.append("arm_feedback_stale")
    if runtime.hand.enabled:
        if row.hand is None:
            issues.append("hand_feedback_unavailable")
        elif not sample_is_fresh(row.hand["timestamp_ns"][0], runtime.hand.feedback_max_age_s, now):
            issues.append("hand_timestamp_ns_unavailable_or_stale")
    return issues


def check_observation(row, runtime, needs, *, now_ns=None):
    """模型或录制所需模态，不重复检查关节反馈。"""
    now = time.monotonic_ns() if now_ns is None else now_ns
    issues: list[str] = []
    for channel in needs.required_tactile:
        valid = tactile_validity(row.hand, channel, max_age_s=runtime.hand.tactile_max_age_s, now_ns=now)
        if not valid.all():
            issues.append(f"hand_tactile_{channel}_unavailable_or_stale: fingers={np.flatnonzero(~valid).tolist()}")
    # 派生图像与对应的点云或 HOI 观测共用来源时间。
    if needs.hoi:
        issues.extend(_hoi_issues(row.hoi, runtime, now))
        if needs.hoi_joint_devices and row.aligned_joints is None:
            issues.append("hoi_joint_alignment_unavailable")
    elif needs.point_cloud:
        if row.pointcloud_timestamp_ns is None:
            issues.append("pointcloud_unavailable")
        elif not sample_is_fresh(row.pointcloud_timestamp_ns, runtime.camera.max_frame_age_s, now):
            issues.append("pointcloud_stale")
    elif needs.camera_fields:
        if row.camera is None:
            issues.append("camera_source_unavailable_or_unmatched")
        elif not sample_is_fresh(row.camera["timestamp_ns"], runtime.camera.max_frame_age_s, now):
            issues.append("camera_source_stale")
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
    if not sample_is_fresh(hoi["timestamp_ns"], runtime.execution.hoi_max_age_s, now_ns):
        issues.append("hoi_stale")
    return issues


def check_hoi_status(channels, runtime) -> list[str]:
    """仅读取当前请求及 HOI 元数据，不读取机器人、点云或图像。"""
    hoi = _read_hoi(channels)
    return _hoi_issues(hoi, runtime, time.monotonic_ns())


def is_recoverable_hoi_failure(issues: Sequence[str]) -> bool:
    """只有非空且完全由 HOI 恢复问题组成的列表可以暂停重试。"""
    return bool(issues) and all(issue in _RECOVERABLE_HOI_ISSUES for issue in issues)
