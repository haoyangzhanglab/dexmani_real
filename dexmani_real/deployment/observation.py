"""Project real control-row history into the requested Policy observation mapping."""

from dataclasses import dataclass

import numpy as np

from dexmani_real.planning.kinematics.arm_fk import (
    ArmFK,
    compute_eef_pose_history_xarm_base,
)
from dexmani_real.planning.kinematics.fingertip import compute_fingertip_history_xarm_base
from dexmani_real.planning.kinematics.hand_fk import HandKinematics
from dexmani_real.robot.xhand.config import HandParams


@dataclass(frozen=True)
class ObservationKinematics:
    arm_fk: ArmFK
    hand_fk: HandKinematics | None = None
    mount: HandParams | None = None


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
        elif name in {"rgb", "camera_intrinsic", "camera_extrinsic"}:
            if row.camera is None:
                return f"required_{name}_unavailable"
            if name != "rgb":
                continue
            image = row.camera.get("rgb")
            if (
                not isinstance(image, np.ndarray)
                or image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[2] != 3
                or min(image.shape[:2]) <= 0
            ):
                return f"required_{name}_invalid"
    return None


def build_policy_observation(rows, policy_info, *, kinematics=None, camera_parameters=None):
    """Stack admitted rows and validate numerical transforms."""
    requested = set(policy_info.observation_fields)
    arrays = {}
    if requested & {"joint_state", "eef_pose", "fingertip_xyz"}:
        # Keep measured precision through FK, as in the Raw -> canonical path.
        arm = np.stack([r.arm["qpos"][0] for r in rows])
    if requested & {"joint_state", "fingertip_xyz"}:
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
    if "rgb" in requested:
        arrays["rgb"] = np.stack([r.camera["rgb"] for r in rows])
    for name in requested & {"camera_intrinsic", "camera_extrinsic"}:
        if camera_parameters is None or name not in camera_parameters:
            raise ValueError(f"Missing session camera parameter {name}")
        values = camera_parameters[name]
        arrays[name] = np.broadcast_to(values, (len(rows), values.size)).copy()

    if requested & {"eef_pose", "fingertip_xyz"}:
        poses = compute_eef_pose_history_xarm_base(arm, arm_fk=kinematics.arm_fk)
        if "eef_pose" in requested:
            arrays["eef_pose"] = poses.astype(np.float32)
        if "fingertip_xyz" in requested:
            mount = kinematics.mount
            arrays["fingertip_xyz"] = compute_fingertip_history_xarm_base(
                poses,
                hand,
                hand_fk=kinematics.hand_fk,
                handbase_position_eef_m=mount.T_eef_handbase_pos_xyz,
                handbase_quat_eef_wxyz=mount.T_eef_handbase_quat_wxyz,
            )

    result = {}
    for name in policy_info.observation_fields:
        values = np.ascontiguousarray(arrays[name])
        # Immutable source rows were checked per tick; these casts can still overflow.
        if name in {"joint_state", "eef_pose"} and not np.isfinite(values).all():
            raise ValueError(f"Nonfinite policy observation {name}")
        result[name] = values
    return result


def policy_sources(row, fields):
    """Host source times for each requested field in the latest control row."""
    arm = int(row.arm["timestamp_ns"][0])
    hand = int(row.hand["timestamp_ns"][0])
    sources = {}
    for name in fields:
        if name in {"joint_state", "fingertip_xyz"}:
            times = (arm, hand)
        elif name == "eef_pose":
            times = (arm,)
        elif name in {"finger_tactile_force", "taxel_tactile_force"}:
            channel = "aggregate" if name == "finger_tactile_force" else "dense"
            times = (int(row.hand[f"tactile_{channel}_timestamp_ns"][0]),)
        elif name == "point_cloud":
            times = (row.pointcloud_timestamp_ns,)
        elif name in {"rgb", "camera_intrinsic", "camera_extrinsic"}:
            # Static calibration stays tied to a real camera source, never the query clock.
            times = (int(row.camera["timestamp_ns"]),)
        else:
            raise ValueError(f"No source clock for {name}")
        sources[name] = times
    return sources
