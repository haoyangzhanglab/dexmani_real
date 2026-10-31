"""解析训练参数与真机配置，在模型和设备启动前校验。"""

from __future__ import annotations

import math
from dataclasses import fields, replace
from typing import TYPE_CHECKING

from dexmani_real.config.deployment import ExecutionConfig
from dexmani_real.config.perception import HandCloudConfig, PointCloudConfig
from dexmani_real.runtime.preflight import validate_control_rate, validate_robot_config

if TYPE_CHECKING:
    from dexmani_real.config.real import RealConfig
    from dexmani_real.runtime.observation import ObservationNeeds


def _restore_cloud_config(recipe, config_type):
    names = {field.name for field in fields(config_type)}
    if not isinstance(recipe, dict) or set(recipe) != names:
        raise ValueError(
            f"Saved {config_type.__name__} must contain exactly these fields: {sorted(names)}"
        )
    return config_type(**recipe)


def validate_execution_config(info, execution: ExecutionConfig) -> None:
    if execution.mode not in {"sync", "async", "rtc"}:
        raise ValueError("execution.mode must be sync, async or rtc")
    if execution.mode not in info.supported_execution_modes:
        raise NotImplementedError(f"Policy does not support {execution.mode} execution")

    dt = info.control_dt_s
    if (
        type(dt) not in (int, float)
        or not math.isfinite(dt * 1e9)
        or int(dt * 1e9) <= 0
    ):
        raise ValueError("Real deployment requires finite positive control_dt_s of at least 1 ns")
    prediction_steps = info.future_steps
    action_steps = execution.action_steps
    if type(action_steps) is not int or not 1 <= action_steps <= prediction_steps:
        raise ValueError("execution requires integer 1 <= A_exec <= policy future_steps")
    if execution.mode != "sync":
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
    if execution.mode == "rtc":
        cap = execution.rtc_guidance_cap
        if (
            (type(cap) is not int and type(cap) is not float)
            or not math.isfinite(cap) or cap < 0
        ):
            raise ValueError("rtc_guidance_cap requires an explicit finite nonnegative value")


def resolve_deployment_config(
    info, runtime: RealConfig, needs: ObservationNeeds,
) -> RealConfig:
    from dexmani_policy.common.config import require_tactile_source

    source = require_tactile_source(info.observation_fields, info.tactile_source)
    if source == "sim":
        raise ValueError("Real deployment requires real-source tactile observations")
    control_dim = {"joint": 19, "eef": 21}[info.action_mode]
    if info.control_action_dim != control_dim:
        raise ValueError(f"Real {info.action_mode} deployment requires control_action_dim={control_dim}")
    action_steps = runtime.execution.action_steps
    execution = replace(
        runtime.execution,
        action_steps=info.n_action_steps if action_steps is None else action_steps,
    )
    validate_execution_config(info, execution)
    names = set(info.observation_fields)
    if not runtime.hand.enabled:
        raise ValueError("Dexterous policy deployment requires hand.enabled=true")
    if "fingertip_xyz" in names:
        links = info.fingertip_link_names
        if (
            not isinstance(links, (list, tuple))
            or len(links) != 5
            or any(not isinstance(link, str) or not link.strip() for link in links)
            or len(set(links)) != 5
        ):
            raise ValueError("Live fingertip_xyz requires five distinct non-empty link names")
    if "depth" in names:
        recipe = info.depth_config
        if not isinstance(recipe, dict) or "shape" not in recipe or "depth_scale_m" not in recipe:
            raise ValueError("Real depth input requires saved shape and depth_scale_m")
        scale = recipe["depth_scale_m"]
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Saved depth_scale_m must be finite and positive")
        if tuple(recipe["shape"]) != runtime.camera.depth_shape:
            raise ValueError("Live camera shape differs from the saved depth input")

    runtime = replace(
        runtime,
        execution=execution,
        pointcloud=(
            _restore_cloud_config(info.pointcloud_config, PointCloudConfig)
            if "point_cloud" in names else runtime.pointcloud
        ),
        hand_pointcloud=(
            _restore_cloud_config(info.hand_pointcloud_config, HandCloudConfig)
            if "hand_pointcloud" in names else runtime.hand_pointcloud
        ),
    )
    validate_robot_config(runtime)
    validate_control_rate(runtime, 1.0 / info.control_dt_s)
    if needs.camera_worker:
        runtime.camera.validate()
    return runtime
