"""模型加载后解析有效真机配置，在设备连接前校验。"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from dexmani_real.config.deployment import ExecutionConfig
from dexmani_real.config.perception import HandCloudConfig, PointCloudConfig
from dexmani_real.runtime.preflight import validate_control_rate, validate_robot_config

if TYPE_CHECKING:
    from dexmani_real.config.real import RealConfig


def validate_execution_config(info, execution: ExecutionConfig) -> None:
    if execution.mode not in {"sync", "async", "rtc"}:
        raise ValueError("execution.mode must be sync, async or rtc")
    if execution.mode == "rtc" and not info.supports_rtc:
        raise NotImplementedError("Policy does not support RTC")

    if "object_pointcloud" in info.observation_fields:
        age = execution.hoi_max_age_s
        if type(age) not in (int, float) or not math.isfinite(age) or age <= 0:
            raise ValueError("hoi_max_age_s requires a finite positive value")

    dt = execution.control_dt_s
    if dt is None:
        raise ValueError(
            "Checkpoint has no control period; set CONFIG.execution.control_dt_s "
            "in the experiment file passed via --config"
        )
    if (
        type(dt) not in (int, float)
        or not math.isfinite(dt * 1e9)
        or int(dt * 1e9) <= 0
    ):
        raise ValueError(f"execution.control_dt_s must be finite, positive and at least 1 ns; got {dt!r}")
    prediction_steps = info.future_steps
    action_steps = execution.action_steps
    if type(action_steps) is not int or not 1 <= action_steps <= prediction_steps:
        raise ValueError(
            f"execution.action_steps={action_steps!r} requires an integer in "
            f"[1, Checkpoint future_steps={prediction_steps}]"
        )
    if info.requires_prev_action and (
        execution.mode != "sync" or action_steps != info.n_action_steps
    ):
        raise ValueError("ODE requires sync execution with action_steps == n_action_steps")
    if execution.mode != "sync":
        prefetch_steps = execution.prefetch_steps
        if (
            type(prefetch_steps) is not int
            or not 1 <= prefetch_steps <= action_steps
        ):
            raise ValueError(
                "async/rtc require integer 1 <= d <= A_exec; "
                f"A_exec={action_steps}, P={prediction_steps}, d={prefetch_steps}"
            )
        if prefetch_steps >= prediction_steps:
            raise ValueError("async/rtc require d < P so a prediction tail remains at handoff")
    if execution.mode == "rtc":
        cap = execution.rtc_guidance_cap
        if (
            (type(cap) is not int and type(cap) is not float)
            or not math.isfinite(cap) or cap < 0
        ):
            raise ValueError("rtc_guidance_cap requires a finite nonnegative value")


def resolve_deployment_config(info, runtime: RealConfig) -> RealConfig:
    control_dim = {"joint": 19, "eef": 21}[info.action_mode]
    if info.control_action_dim != control_dim:
        raise ValueError(f"Real {info.action_mode} deployment requires control_action_dim={control_dim}")
    action_steps = runtime.execution.action_steps
    execution = replace(
        runtime.execution,
        action_steps=info.n_action_steps if action_steps is None else action_steps,
        control_dt_s=(info.control_dt_s if runtime.execution.control_dt_s is None
                      else runtime.execution.control_dt_s),
    )
    validate_execution_config(info, execution)
    names = set(info.observation_fields)
    if not runtime.hand.enabled:
        raise ValueError("Dexterous policy deployment requires hand.enabled=true")
    geometry = {}
    for field, name, config_type in (
        ("point_cloud", "pointcloud", PointCloudConfig),
        ("hand_pointcloud", "hand_pointcloud", HandCloudConfig),
    ):
        if field not in names:
            continue
        recipe_name = f"{name}_config"
        recipe = getattr(info, recipe_name)
        if recipe is None:
            raise ValueError(f"{field} requires Checkpoint {recipe_name}; use a Checkpoint containing the training recipe")
        try:
            config = config_type(**recipe)
            config.validate()
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"Checkpoint {recipe_name}: {exc}") from exc
        geometry[name] = config
    if "fingertip_xyz" in names and info.fingertip_link_names is None:
        raise ValueError("fingertip_xyz requires Checkpoint fingertip_link_names in model input order")
    if "depth" in names:
        recipe = info.depth_config
        if not isinstance(recipe, dict) or "shape" not in recipe or "depth_scale_m" not in recipe:
            raise ValueError("Checkpoint depth_config requires saved shape and depth_scale_m")
        scale = recipe["depth_scale_m"]
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Checkpoint depth_config.depth_scale_m={scale!r} must be finite and positive")
        if tuple(recipe["shape"]) != runtime.camera.image_shape:
            raise ValueError(
                f"Checkpoint depth_config.shape={tuple(recipe['shape'])} (height, width), "
                f"but RealConfig camera.image_shape={runtime.camera.image_shape}; "
                "set camera.height and camera.width in the experiment file passed via --config; "
                "depth is not resized"
            )

    runtime = replace(
        runtime,
        execution=execution,
        pointcloud=cast(PointCloudConfig, geometry.get("pointcloud", runtime.pointcloud)),
        hand_pointcloud=cast(HandCloudConfig, geometry.get("hand_pointcloud", runtime.hand_pointcloud)),
    )
    validate_robot_config(runtime)
    if info.action_mode == "eef":
        runtime.environment.validate()
    validate_control_rate(runtime, 1.0 / cast(float, execution.control_dt_s))
    return runtime


def resolve_hoi_inputs(info):
    """解析并加载 HOI 任务模板；模板身份失配在设备连接前失败。"""
    from pathlib import Path

    if "object_pointcloud" not in info.observation_fields:
        return None
    if "point_cloud" in info.observation_fields:
        raise ValueError("HOI 与 point_cloud 不支持混合输入")
    if info.hoi_pcd is None:
        raise ValueError("Checkpoint 缺少 HOI 任务与模板信息")
    name = info.hoi_pcd["task_name"]
    if Path(name).name != name or name in (".", "..") or name != info.task_name:
        raise ValueError("Checkpoint HOI 任务名不匹配")
    root = Path(__file__).resolve().parents[2]
    from dexmani_real.geometry.hoi import load_task_template
    task = root / "assets/pointcloud_template/object" / name
    template, manifest = load_task_template(task)
    if info.hoi_pcd != manifest:
        raise ValueError("Checkpoint HOI 任务或模板哈希不匹配")
    return task, template
