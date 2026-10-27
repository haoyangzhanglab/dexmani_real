"""从已校验的传感快照构造策略观测。"""

import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from dexmani_real.geometry.camera import camera_parameters_cv
from dexmani_real.geometry.kinematics import (
    ArmFK,
    HandFK,
    eef_pose_sequence,
)
from dexmani_real.geometry.pose import quat_wxyz_to_rotmat
from dexmani_real.perception.hand import HandSurface, hand_geometry_sequence
from dexmani_real.robot.model import XHAND_FINGERTIP_LINK_NAMES
from dexmani_real.runtime.observation import ObservationNeeds, freeze

FIELD_DEPENDENCIES = {
    "joint_state": ("arm", "hand"),
    "eef_pose": ("arm", "arm_fk"),
    "fingertip_xyz": ("arm", "hand", "arm_fk", "hand_fk", "fingertips"),
    "hand_pointcloud": ("arm", "hand", "arm_fk", "hand_fk", "hand_cloud"),
    "finger_tactile_force": ("aggregate",),
    "taxel_tactile_force": ("dense",),
    "point_cloud": ("cloud",),
    "rgb": ("camera", "rgb"),
    "depth": ("camera", "depth"),
    "camera_intrinsic": ("camera_static",),
    "camera_extrinsic": ("camera_static",),
}


def policy_dependencies(fields):
    unknown = set(fields) - FIELD_DEPENDENCIES.keys()
    if unknown:
        raise ValueError(f"Real has no live producer for requested modalities: {sorted(unknown)}")
    return frozenset(source for name in fields for source in FIELD_DEPENDENCIES[name])


def policy_needs(dependencies):
    rgb, depth = "rgb" in dependencies, "depth" in dependencies
    return ObservationNeeds(
        camera="rgbd" if rgb and depth else "rgb" if rgb else "depth" if depth else "none",
        point_cloud="cloud" in dependencies,
        required_tactile=tuple(c for c in ("aggregate", "dense") if c in dependencies),
        camera_static="camera_static" in dependencies,
    )


@dataclass(frozen=True)
class ObservationKinematics:
    arm_fk: ArmFK
    hand_fk: HandFK | None = None
    handbase_position_eef_m: tuple[float, float, float] | None = None
    handbase_rotation_eef: np.ndarray | None = None
    hand_cloud: HandSurface | None = None
    row_features: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyObservationRow:
    observation_timestamp_ns: int
    source_timestamp_ns: int
    values: Mapping


def policy_source_timestamp(row, dependencies):
    times = []
    for source, stamp in (
        ("arm", int(row.arm["timestamp_ns"][0])),
        ("hand", int(row.hand["timestamp_ns"][0])),
        ("aggregate", int(row.hand["tactile_aggregate_timestamp_ns"][0])),
        ("dense", int(row.hand["tactile_dense_timestamp_ns"][0])),
        ("cloud", row.pointcloud_timestamp_ns),
    ):
        if source in dependencies:
            times.append(stamp)
    if "camera" in dependencies:
        times.append(int(row.camera["timestamp_ns"]))
    # 静态输入仍使用本轮机械臂和手的真实反馈时间。
    return min(times or (int(row.arm["timestamp_ns"][0]), int(row.hand["timestamp_ns"][0])))


def project_policy_row(row, fields, dependencies):
    values = {}
    if "arm" in dependencies:
        values["arm_qpos"] = row.arm["qpos"][0].copy()
    if "hand" in dependencies:
        values["hand_qpos"] = row.hand["qpos"][0].copy()
    for name in fields:
        if name == "finger_tactile_force":
            # 独立复制，避免聚合触觉视图持有仅供录制的稠密载荷。
            values[name] = row.hand["tactile_aggregate"][0].copy()
        elif name == "taxel_tactile_force":
            values[name] = row.hand["tactile_dense"][0]
        elif name == "point_cloud":
            values[name] = row.point_cloud
        elif name in {"rgb", "depth"}:
            values[name] = row.camera[name]
    return PolicyObservationRow(
        row.observation_timestamp_ns, policy_source_timestamp(row, dependencies), freeze(values),
    )


def build_observation_kinematics(dependencies, *, fingertip_link_names, hand, hand_pointcloud):
    """按需构造 FK 工作区，由推理线程独占。"""
    if "arm_fk" not in dependencies:
        return None
    if "hand_fk" not in dependencies:
        return ObservationKinematics(ArmFK())
    hand_fk = HandFK(
        fingertip_link_names if "fingertips" in dependencies else XHAND_FINGERTIP_LINK_NAMES,
    )
    hand_cloud = None
    if "hand_cloud" in dependencies:
        hand_cloud = HandSurface(hand_pointcloud, hand_fk)
    return ObservationKinematics(
        ArmFK(), hand_fk,
        tuple(hand.T_eef_handbase_pos_xyz),
        quat_wxyz_to_rotmat(hand.T_eef_handbase_quat_wxyz),
        hand_cloud,
    )


def build_observation_camera(camera, fields, *, depth_config=None):
    requested = set(fields) & {"depth", "camera_intrinsic", "camera_extrinsic"}
    if not requested:
        return {}
    if "depth" in requested and not math.isclose(
        camera.depth_scale_m, depth_config["depth_scale_m"], rel_tol=1e-9, abs_tol=0.0,
    ):
        raise ValueError("Live depth scale differs from the saved raw-count input recipe")
    requested.discard("depth")
    if not requested:
        return {}
    parameters = camera_parameters_cv(
        camera.color,
        camera.T_xarm_base_from_color if "camera_extrinsic" in requested else None,
    )
    return {name: parameters[name] for name in requested}


def build_policy_observation(rows, fields, dependencies, *, kinematics=None, camera_parameters=None):
    arrays = {}
    if "joint_state" in fields:
        arrays["joint_state"] = np.stack([
            np.concatenate((r.values["arm_qpos"], r.values["hand_qpos"])) for r in rows
        ]).astype(np.float32)
    for name in fields:
        sources = FIELD_DEPENDENCIES[name]
        if "camera_static" in sources:
            values = camera_parameters[name]
            arrays[name] = np.broadcast_to(values, (len(rows), values.size)).copy()
        elif name != "joint_state" and "arm_fk" not in sources:
            arrays[name] = np.stack([r.values[name] for r in rows])

    if "arm_fk" in dependencies:
        cache = kinematics.row_features
        identities = {r.observation_timestamp_ns for r in rows}
        for identity in tuple(cache):
            if identity not in identities:
                del cache[identity]
        new_rows = tuple({
            r.observation_timestamp_ns: r for r in rows if r.observation_timestamp_ns not in cache
        }.values())
        features = {}
        if new_rows:
            arm = np.stack([r.values["arm_qpos"] for r in new_rows])
            poses = eef_pose_sequence(arm, arm_fk=kinematics.arm_fk)
            if "eef_pose" in fields:
                features["eef_pose"] = poses.astype(np.float32)
            if "hand_fk" in dependencies:
                features.update(hand_geometry_sequence(
                    poses, np.stack([r.values["hand_qpos"] for r in new_rows]),
                    hand_fk=kinematics.hand_fk,
                    handbase_position_eef_m=kinematics.handbase_position_eef_m,
                    handbase_rotation_eef=kinematics.handbase_rotation_eef,
                    hand_cloud=kinematics.hand_cloud,
                    include_fingertips="fingertips" in dependencies,
                ))
        for index, row in enumerate(new_rows):
            cache[row.observation_timestamp_ns] = {
                name: value[index].copy() for name, value in features.items()
            }
        for name in fields:
            if "arm_fk" in FIELD_DEPENDENCIES[name]:
                arrays[name] = np.stack([cache[r.observation_timestamp_ns][name] for r in rows])

    return {name: arrays[name] for name in fields}
