"""Project real control-row history into the requested Policy observation mapping."""

from dataclasses import dataclass

import numpy as np

from dexmani_real.planning.kinematics.arm_fk import (
    ArmFK,
    compute_eef_pose_history_xarm_base,
)
from dexmani_real.planning.kinematics.fingertip import compute_hand_geometry_history_xarm_base
from dexmani_real.planning.kinematics.hand_fk import HandKinematics


@dataclass(frozen=True)
class ObservationKinematics:
    arm_fk: ArmFK
    hand_fk: HandKinematics | None = None
    handbase_position_eef_m: tuple[float, float, float] | None = None
    handbase_quat_eef_wxyz: tuple[float, float, float, float] | None = None
    hand_cloud: object | None = None


def policy_observation_issue(row, fields):
    """Check each real row, including bootstrap and slots using an existing plan."""
    if row is None or row.hand is None:
        return "required_observation_unavailable"
    if not np.isfinite(row.arm["qpos"]).all() or not np.isfinite(row.hand["qpos"]).all():
        return "control_feedback_nonfinite"
    for name in fields:
        if name in {"finger_tactile_force", "taxel_tactile_force"}:
            channel = "aggregate" if name == "finger_tactile_force" else "dense"
            if not np.isfinite(row.hand[f"tactile_{channel}"]).all():
                return f"required_{name}_unavailable"
        elif name == "point_cloud":
            if row.point_cloud is None or not np.isfinite(row.point_cloud).all():
                return "required_point_cloud_unavailable_or_nonfinite"
        elif name in {"rgb", "depth", "camera_intrinsic", "camera_extrinsic"}:
            if row.camera is None:
                return f"required_{name}_unavailable"
            if name not in {"rgb", "depth"}:
                continue
            image = row.camera.get(name)
            if (
                not isinstance(image, np.ndarray)
                or image.dtype != (np.uint8 if name == "rgb" else np.uint16)
                or image.ndim != (3 if name == "rgb" else 2)
                or (name == "rgb" and image.shape[2] != 3)
                or min(image.shape[:2]) <= 0
            ):
                return f"required_{name}_invalid"
    return None


def build_policy_observation(rows, policy_info, *, kinematics=None, camera_parameters=None):
    """Stack admitted rows and validate numerical transforms."""
    requested = set(policy_info.observation_fields)
    arrays = {}
    if requested & {"joint_state", "eef_pose", "fingertip_xyz", "hand_pointcloud"}:
        # Keep measured precision through FK, as in the Raw -> canonical path.
        arm = np.stack([r.arm["qpos"][0] for r in rows])
    if requested & {"joint_state", "fingertip_xyz", "hand_pointcloud"}:
        hand = np.stack([r.hand["qpos"][0] for r in rows])
    if "joint_state" in requested:
        arrays["joint_state"] = np.concatenate((arm, hand), axis=1).astype(np.float32)
    for name, field in (
        ("finger_tactile_force", "tactile_aggregate"),
        ("taxel_tactile_force", "tactile_dense"),
    ):
        if name in requested:
            arrays[name] = np.stack([r.hand[field][0] for r in rows])
    if "point_cloud" in requested:
        arrays["point_cloud"] = np.stack([r.point_cloud for r in rows])
    for name in requested & {"rgb", "depth"}:
        arrays[name] = np.stack([r.camera[name] for r in rows])
    for name in requested & {"camera_intrinsic", "camera_extrinsic"}:
        if camera_parameters is None or name not in camera_parameters:
            raise ValueError(f"Missing session camera parameter {name}")
        values = camera_parameters[name]
        arrays[name] = np.broadcast_to(values, (len(rows), values.size)).copy()

    if requested & {"eef_pose", "fingertip_xyz", "hand_pointcloud"}:
        if kinematics is None:
            raise ValueError("Requested geometry requires prepared kinematics")
        poses = compute_eef_pose_history_xarm_base(arm, arm_fk=kinematics.arm_fk)
        if "eef_pose" in requested:
            arrays["eef_pose"] = poses.astype(np.float32)
        if requested & {"fingertip_xyz", "hand_pointcloud"}:
            arrays.update(compute_hand_geometry_history_xarm_base(
                poses,
                hand,
                hand_fk=kinematics.hand_fk,
                handbase_position_eef_m=kinematics.handbase_position_eef_m,
                handbase_quat_eef_wxyz=kinematics.handbase_quat_eef_wxyz,
                hand_cloud=kinematics.hand_cloud,
                include_fingertips="fingertip_xyz" in requested,
            ))

    result = {}
    for name in policy_info.observation_fields:
        values = np.ascontiguousarray(arrays[name])
        # Immutable source rows were checked per tick; these casts can still overflow.
        if values.dtype.kind == "f" and not np.isfinite(values).all():
            raise ValueError(f"Nonfinite policy observation {name}")
        result[name] = values
    return result


def policy_source_times(row, fields):
    """Host source times of requested inputs in the latest control row."""
    arm = int(row.arm["timestamp_ns"][0])
    hand = int(row.hand["timestamp_ns"][0])
    timestamps = []
    for name in fields:
        if name in {"joint_state", "fingertip_xyz", "hand_pointcloud"}:
            times = (arm, hand)
        elif name == "eef_pose":
            times = (arm,)
        elif name in {"finger_tactile_force", "taxel_tactile_force"}:
            channel = "aggregate" if name == "finger_tactile_force" else "dense"
            times = (int(row.hand[f"tactile_{channel}_timestamp_ns"][0]),)
        elif name == "point_cloud":
            times = (row.pointcloud_timestamp_ns,)
        elif name in {"rgb", "depth", "camera_intrinsic", "camera_extrinsic"}:
            # Static calibration stays tied to a real camera source, never the query clock.
            times = (int(row.camera["timestamp_ns"]),)
        else:
            raise ValueError(f"No source clock for {name}")
        timestamps.extend(times)
    return tuple(timestamps)
