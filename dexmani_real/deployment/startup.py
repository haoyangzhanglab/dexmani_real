"""Check deployment inputs before outputs, model loading or device connections."""

from __future__ import annotations

from typing import Any

from dexmani_real.config.experiment import validate_control_rate, validate_robot_config
from dexmani_real.sensor.pointcloud.config import PointCloudConfig, resolve_saved_pointcloud_recipe


SUPPORTED_OBSERVATION_FIELDS = frozenset(
    {
        "joint_state",
        "point_cloud",
        "rgb",
        "finger_tactile_force",
        "fingertip_xyz",
        "eef_pose",
        "taxel_tactile_force",
        "camera_intrinsic",
        "camera_extrinsic",
    }
)


def validate_policy_inputs(info: Any, runtime: Any) -> PointCloudConfig | None:
    """Check physical capability before allocating IPC or connecting devices."""
    if not runtime.control.hand_enabled:
        raise ValueError("Dexterous policy deployment requires hand_enabled=true")
    names = set(info.observation_fields)
    unsupported = names - SUPPORTED_OBSERVATION_FIELDS
    if unsupported:
        raise ValueError(
            f"Real has no live producer for requested modalities: {sorted(unsupported)}"
        )
    if info.action_mode not in {"joint", "eef"}:
        raise ValueError(f"unsupported Real action mode: {info.action_mode!r}")
    if "fingertip_xyz" in names:
        links = info.fingertip_link_names
        if (
            not isinstance(links, (list, tuple))
            or len(links) != 5
            or any(not isinstance(link, str) or not link.strip() for link in links)
            or len(set(links)) != 5
        ):
            raise ValueError("Live fingertip_xyz requires five distinct non-empty link names")
    if "point_cloud" in names:
        return resolve_saved_pointcloud_recipe(info.pointcloud_config)
    return None


def validate_deployment_config(
    info: Any, runtime: Any, *, record_raw: bool
) -> PointCloudConfig | None:
    """Check both deployment entries before outputs, models or devices; no calibration I/O."""
    info.validate_execution_mode(runtime.execution.execution_mode)
    runtime.execution.validate(info)
    history_steps = info.hand_history_steps
    if history_steps is not None:
        if type(history_steps) is not int or history_steps < 1:
            raise ValueError("hand_history_steps must be a positive integer or None")
        if (
            info.action_mode != "joint"
            or info.control_action_dim != 19
            or "joint_state" not in info.observation_fields
            or (info.horizon, info.n_obs_steps, info.n_action_steps, runtime.execution.action_steps)
            != (1, 1, 1, 1)
        ):
            raise ValueError("LAMP hand history requires joint, joint_state, H=Nobs=C=A_exec=1")
    cloud_recipe = validate_policy_inputs(info, runtime)
    validate_robot_config(runtime)
    validate_control_rate(runtime, 1.0 / info.control_dt_s)
    if record_raw or set(info.observation_fields) & {
        "point_cloud",
        "rgb",
        "camera_intrinsic",
        "camera_extrinsic",
    }:
        runtime.camera.validate()
    return cloud_recipe
