"""Check deployment inputs before outputs, model loading or device connections."""

from __future__ import annotations

import math
from typing import Any

from dexmani_real.config.experiment import validate_control_rate, validate_robot_config
from dexmani_real.sensor.pointcloud.config import PointCloudConfig, resolve_saved_pointcloud_recipe


def validate_max_running_s(max_running_s: float | None) -> float | None:
    """Validate the cooperative duration budget from RUNNING admission."""
    if max_running_s is None:
        return None
    if isinstance(max_running_s, bool):
        raise TypeError("max_running_s must be a finite positive number or None")
    value = float(max_running_s)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("max_running_s must be finite and positive")
    return value


def validate_num_episodes(num_episodes: Any) -> int | None:
    """Count episodes that entered RUNNING once they end; None waits for operator exit."""
    if num_episodes is None:
        return None
    if type(num_episodes) is not int:
        raise TypeError("num_episodes must be a positive integer or None")
    if num_episodes < 1:
        raise ValueError("num_episodes must be a positive integer")
    return num_episodes


def validate_execution_config(info, execution) -> None:
    """Validate Real scheduling; the loaded Policy checks its model's mode support."""
    if execution.execution_mode not in {"sync", "async", "rtc"}:
        raise ValueError("execution_mode must be sync, async or rtc")
    dt = info.control_dt_s
    if (
        isinstance(dt, bool)
        or not isinstance(dt, (int, float))
        or not math.isfinite(dt)
        or dt <= 0
        or not math.isfinite(dt * 1e9)
        or int(dt * 1e9) <= 0
    ):
        raise ValueError("Real deployment requires finite positive control_dt_s of at least 1 ns")

    tolerance = execution.start_arm_home_tolerance_deg
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("start_arm_home_tolerance_deg must be finite and positive")
    for name in ("max_decision_age_s", "max_wait_s"):
        value = getattr(execution, name)
        if isinstance(value, bool) or value is None or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive experimental budget")

    prediction_steps = info.horizon - info.n_obs_steps + 1
    action_steps = execution.action_steps
    if type(action_steps) is not int:
        raise ValueError("Resolve effective A_exec at the deployment entry before validation")
    if not 1 <= action_steps <= prediction_steps:
        raise ValueError("execution requires 1 <= A_exec <= P=H-N+1")
    if execution.execution_mode != "sync":
        prefetch_steps = execution.prefetch_steps
        if (
            type(prefetch_steps) is not int
            or not 1 <= prefetch_steps <= action_steps
            or action_steps + prefetch_steps > prediction_steps
        ):
            raise ValueError(
                "async/rtc require 1 <= d <= A_exec and A_exec+d <= P; "
                f"A_exec={action_steps}, P={prediction_steps}, d={prefetch_steps}"
            )

    if execution.execution_mode == "rtc":
        guidance_cap = execution.rtc_guidance_cap
        if (
            isinstance(guidance_cap, bool)
            or guidance_cap is None
            or not math.isfinite(guidance_cap)
            or guidance_cap < 0
        ):
            raise ValueError("rtc_guidance_cap requires an explicit finite nonnegative value")


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


def validate_deployment_config(info: Any, runtime: Any, *, needs) -> PointCloudConfig | None:
    """Check Real constraints on inspected Policy metadata; no calibration I/O."""
    validate_execution_config(info, runtime.execution)
    if info.hand_history_steps is not None:
        if (
            info.action_mode != "joint"
            or "joint_state" not in info.observation_fields
            or (info.horizon, info.n_obs_steps, info.n_action_steps, runtime.execution.action_steps)
            != (1, 1, 1, 1)
        ):
            raise ValueError("LAMP hand history requires joint, joint_state, H=Nobs=C=A_exec=1")
    cloud_recipe = validate_policy_inputs(info, runtime)
    validate_robot_config(runtime)
    validate_control_rate(runtime, 1.0 / info.control_dt_s)
    if needs.camera_worker:
        runtime.camera.validate()
    return cloud_recipe
