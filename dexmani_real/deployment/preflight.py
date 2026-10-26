"""Check deployment inputs before outputs, model loading or device connections."""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any

from dexmani_real.perception.config import HandCloudConfig, PointCloudConfig
from dexmani_real.runtime.preflight import validate_control_rate, validate_robot_config


def _restore_cloud_config(recipe, config_type):
    if not isinstance(recipe, dict) or recipe.keys() != {f.name for f in fields(config_type)}:
        raise ValueError(
            f"Saved {config_type.__name__} is incomplete or obsolete; "
            "re-export the dataset and retrain the checkpoint"
        )
    return config_type(**recipe)


def validate_execution_config(info, execution) -> None:
    """Validate saved mode support and Real scheduling before model/device startup."""
    if execution.execution_mode not in {"sync", "async", "rtc"}:
        raise ValueError("execution_mode must be sync, async or rtc")
    if execution.execution_mode not in info.supported_execution_modes:
        raise NotImplementedError(f"Policy does not support {execution.execution_mode} execution")
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

    prediction_steps = info.future_steps
    action_steps = execution.action_steps
    if type(action_steps) is not int:
        raise ValueError("Resolve effective A_exec at the deployment entry before validation")
    if not 1 <= action_steps <= prediction_steps:
        raise ValueError("execution requires 1 <= A_exec <= policy future_steps")
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


def validate_deployment_config(
    info: Any, runtime: Any, *, needs,
) -> tuple[PointCloudConfig | None, HandCloudConfig | None]:
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
    if not runtime.hand.enabled:
        raise ValueError("Dexterous policy deployment requires hand.enabled=true")
    names = set(info.observation_fields)
    if "fingertip_xyz" in names:
        links = info.fingertip_link_names
        if (
            not isinstance(links, (list, tuple))
            or len(links) != 5
            or any(not isinstance(link, str) or not link.strip() for link in links)
            or len(set(links)) != 5
        ):
            raise ValueError("Live fingertip_xyz requires five distinct non-empty link names")
    if "depth" in names and tuple(info.depth_config["shape"]) != runtime.camera.depth_shape:
        raise ValueError("Live camera shape differs from the saved depth input")
    recipes = (
        _restore_cloud_config(info.pointcloud_config, PointCloudConfig) if "point_cloud" in names else None,
        _restore_cloud_config(info.hand_pointcloud_config, HandCloudConfig)
        if "hand_pointcloud" in names else None,
    )
    validate_robot_config(runtime)
    validate_control_rate(runtime, 1.0 / info.control_dt_s)
    if needs.camera_worker:
        runtime.camera.validate()
    return recipes
