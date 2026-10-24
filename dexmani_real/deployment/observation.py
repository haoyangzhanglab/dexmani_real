"""Fixed policy dependencies and projections of admitted physical rows."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from dexmani_real.planning.kinematics.arm_fk import ArmFK, compute_eef_pose_history_xarm_base
from dexmani_real.planning.kinematics.fingertip import compute_hand_geometry_history_xarm_base
from dexmani_real.planning.kinematics.hand_fk import HandKinematics
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
    return frozenset(source for field in fields for source in FIELD_DEPENDENCIES[field])


def policy_needs(fields):
    unknown = set(fields) - FIELD_DEPENDENCIES.keys()
    if unknown:
        raise ValueError(f"Real has no live producer for requested modalities: {sorted(unknown)}")
    dependencies = policy_dependencies(fields)
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
    hand_fk: HandKinematics | None = None
    handbase_position_eef_m: tuple[float, float, float] | None = None
    handbase_quat_eef_wxyz: tuple[float, float, float, float] | None = None
    hand_cloud: object | None = None
    row_features: dict | None = None


@dataclass(frozen=True)
class PolicyObservationRow:
    observation_timestamp_ns: int
    source_times: tuple[int, ...]
    values: Mapping
    camera_sequence: int


def policy_source_times(row, fields):
    dependencies = policy_dependencies(fields)
    times = []
    for source, stamp in (
        ("arm", int(row.arm["timestamp_ns"][0])),
        ("hand", int(row.hand["timestamp_ns"][0])),
        ("aggregate", int(row.hand["tactile_aggregate_timestamp_ns"][0])),
        ("dense", int(row.hand["tactile_dense_timestamp_ns"][0])),
        ("cloud", row.pointcloud_timestamp_ns),
        ("camera", int(row.camera["timestamp_ns"]) if row.camera is not None else 0),
    ):
        if source in dependencies:
            times.append(stamp)
    # Static parameters do not expire, but a decision still uses real control feedback.
    return tuple(times) or (int(row.arm["timestamp_ns"][0]), int(row.hand["timestamp_ns"][0]))


def project_policy_row(row, fields):
    values = {
        "arm_qpos": row.arm["qpos"][0].copy(),
        "hand_qpos": row.hand["qpos"][0].copy(),
    }
    for name in fields:
        if name == "finger_tactile_force":
            # A structured-field view would retain the recording-only dense payload.
            values[name] = row.hand["tactile_aggregate"][0].copy()
        elif name == "taxel_tactile_force":
            values[name] = row.hand["tactile_dense"][0]
        elif name == "point_cloud":
            values[name] = row.point_cloud
        elif name in {"rgb", "depth"}:
            values[name] = row.camera[name]
    return PolicyObservationRow(
        row.observation_timestamp_ns, policy_source_times(row, fields), freeze(values), row.camera_sequence,
    )


def build_policy_observation(rows, policy_info, *, kinematics=None, camera_parameters=None):
    requested = set(policy_info.observation_fields)
    dependencies = policy_dependencies(requested)
    arrays = {}
    if "joint_state" in requested:
        arrays["joint_state"] = np.stack([
            np.concatenate((r.values["arm_qpos"], r.values["hand_qpos"])) for r in rows
        ]).astype(np.float32)
    for name in requested:
        sources = FIELD_DEPENDENCIES[name]
        if "camera_static" in sources:
            values = camera_parameters[name]
            arrays[name] = np.broadcast_to(values, (len(rows), values.size)).copy()
        elif name != "joint_state" and "arm_fk" not in sources:
            arrays[name] = np.stack([r.values[name] for r in rows])

    if "arm_fk" in dependencies:
        cache = kinematics.row_features
        if cache is None:
            new_rows = rows
        else:
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
            poses = compute_eef_pose_history_xarm_base(arm, arm_fk=kinematics.arm_fk)
            if "eef_pose" in requested:
                features["eef_pose"] = poses.astype(np.float32)
            if "hand_fk" in dependencies:
                features.update(compute_hand_geometry_history_xarm_base(
                    poses, np.stack([r.values["hand_qpos"] for r in new_rows]),
                    hand_fk=kinematics.hand_fk,
                    handbase_position_eef_m=kinematics.handbase_position_eef_m,
                    handbase_quat_eef_wxyz=kinematics.handbase_quat_eef_wxyz,
                    hand_cloud=kinematics.hand_cloud,
                    include_fingertips="fingertips" in dependencies,
                ))
        if cache is None:
            arrays.update(features)
        else:
            for index, row in enumerate(new_rows):
                cache[row.observation_timestamp_ns] = {
                    name: value[index].copy() for name, value in features.items()
                }
            for name in requested:
                if "arm_fk" in FIELD_DEPENDENCIES[name]:
                    arrays[name] = np.stack([cache[r.observation_timestamp_ns][name] for r in rows])

    result = {}
    for name in policy_info.observation_fields:
        values = np.ascontiguousarray(arrays[name])
        if values.dtype.kind == "f" and not np.isfinite(values).all():
            raise ValueError(f"Nonfinite policy observation {name}")
        result[name] = values
    return result
