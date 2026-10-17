"""Project real control-row history into the requested Policy observation mapping."""

from dataclasses import dataclass
import json
from typing import Any

import numpy as np

from dexmani_real.config.hardware import HandParams
from dexmani_real.planning.kinematics.arm_fk import (
    ArmFK,
    compute_eef_pose_history_xarm_base,
    make_arm_fk,
)
from dexmani_real.planning.kinematics.fingertip import compute_fingertip_history_xarm_base
from dexmani_real.planning.kinematics.hand_fk import HandKinematics
from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH
from dexmani_real.sensor.camera.geometry import RGBDGeometry, camera_parameters_cv


@dataclass(frozen=True)
class ObservationKinematics:
    arm_fk: ArmFK
    hand_fk: HandKinematics | None = None
    mount: HandParams | None = None


def build_observation_kinematics(policy_info: Any, runtime: Any):
    """Build only requested FK resources; EEF and fingertips share aligned qpos history and canonical ArmFK."""
    requested = set(policy_info.observation_fields)
    needs_hand_fk = "fingertip_xyz" in requested
    if not needs_hand_fk and "eef_pose" not in requested:
        return None
    if not needs_hand_fk:
        return ObservationKinematics(make_arm_fk())
    hand_fk = HandKinematics(
        str(XHAND_RIGHT_URDF_PATH),
        list(policy_info.fingertip_link_names),
    )
    return ObservationKinematics(make_arm_fk(), hand_fk, runtime.hand)


def build_observation_camera(sensors, camera_config, fields, calibration):
    """Snapshot requested K/E once after the fixed camera worker becomes ready."""
    requested = set(fields) & {"camera_intrinsic", "camera_extrinsic"}
    if not requested:
        return {}
    if sensors is None or not sensors.camera_ready.is_set():
        raise ValueError("Policy camera parameters require a ready camera")
    serial = sensors.camera_serial.value.decode("utf-8").strip()
    if not serial or (camera_config.serial is not None and serial != camera_config.serial):
        raise ValueError("Policy camera parameters require the active camera serial to match configuration")
    geometry = RGBDGeometry.from_dict(json.loads(sensors.camera_geometry.value.decode("utf-8")))
    color = geometry.color
    if (color.height, color.width) != (camera_config.height, camera_config.width):
        raise ValueError("Policy color intrinsics do not match the configured RGB resolution")
    if sensors.camera_ring.dtype["rgb"].shape != (color.height, color.width, 3):
        raise ValueError("Policy color intrinsics do not match the camera RGB payload")
    transform = None
    if "camera_extrinsic" in requested:
        if calibration is None:
            raise ValueError("camera_extrinsic requires current camera calibration")
        transform = calibration.for_serial(serial)
    parameters = camera_parameters_cv(color, transform)
    return {name: parameters[name] for name in requested}


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
            image = row.camera["rgb"]
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                    or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0):
                return f"required_{name}_invalid"
    return None


def build_policy_observation(rows, policy_info, *, kinematics=None, camera_parameters=None):
    """Stack frozen rows already admitted by policy_observation_issue; validate new numerical transforms."""
    requested = set(policy_info.observation_fields)
    arrays = {}
    if requested & {"joint_state", "eef_pose", "fingertip_xyz"}:
        # Keep measured precision through FK, as in the Raw -> canonical path.
        arm = np.stack([r.arm["qpos"][0] for r in rows])
    if requested & {"joint_state", "fingertip_xyz"}:
        hand = np.stack([r.hand["qpos"][0] for r in rows])
    if "joint_state" in requested:
        arrays["joint_state"] = np.concatenate((arm, hand), axis=1).astype(np.float32)
    for name, field in (("finger_tactile_force", "tactile_aggregate"), ("taxel_tactile_force", "tactile_dense")):
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
        arm_fk, hand_fk, cfg = kinematics.arm_fk, kinematics.hand_fk, kinematics.mount
        poses = compute_eef_pose_history_xarm_base(arm, arm_fk=arm_fk)
        if "eef_pose" in requested:
            arrays["eef_pose"] = poses.astype(np.float32)
        if "fingertip_xyz" in requested:
            arrays["fingertip_xyz"] = compute_fingertip_history_xarm_base(
                arm,
                hand,
                hand_fk=hand_fk,
                handbase_position_eef_m=np.asarray(cfg.T_eef_handbase_pos_xyz),
                handbase_quat_eef_wxyz=np.asarray(cfg.T_eef_handbase_quat_wxyz),
                arm_fk=arm_fk,
                eef_pose_history=poses,
            )
    result = {}
    for name in policy_info.observation_fields:
        values = np.ascontiguousarray(arrays[name])
        if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
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


def decision_is_fresh(sources, now_ns, max_age_s):
    return bool(sources) and all(
        0 < timestamp <= now_ns and now_ns - timestamp <= int(max_age_s * 1e9)
        for times in sources.values()
        for timestamp in times
    )
