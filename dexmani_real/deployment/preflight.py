"""模型加载后解析有效真机配置，在设备连接前校验。"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from dexmani_real.config.deployment import ExecutionConfig
from dexmani_real.runtime.preflight import validate_control_rate, validate_robot_config

if TYPE_CHECKING:
    from dexmani_real.config.real import RealConfig


def validate_execution_config(info, execution: ExecutionConfig) -> None:
    if execution.mode not in {"sync", "async", "rtc"}:
        raise ValueError("execution.mode must be sync, async or rtc")
    if execution.mode == "rtc" and not info.supports_rtc:
        raise NotImplementedError("Policy does not support RTC")

    if "object_pointcloud" in info.observation_fields and set(info.observation_fields) & {
        "hand_pointcloud", "joint_state", "fingertip_xyz"
    }:
        from dexmani_real.utils.timing import validate_hoi_skew_budget
        validate_hoi_skew_budget(execution.hoi_max_skew_s)
    if "object_pointcloud" in info.observation_fields:
        age = execution.hoi_max_age_s
        if type(age) not in (int, float) or not math.isfinite(age) or age <= 0:
            raise ValueError("hoi_max_age_s requires a finite positive value")

    dt = execution.control_dt_s
    if (
        dt is None or type(dt) not in (int, float)
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
    if not runtime.hand.enabled:
        raise ValueError("Dexterous policy deployment requires hand.enabled=true")
    if "point_cloud" in info.observation_fields:
        runtime.pointcloud.validate()
    runtime = replace(runtime, execution=execution)
    validate_robot_config(runtime)
    if info.action_mode == "eef":
        runtime.environment.validate()
    validate_control_rate(runtime, 1.0 / cast(float, execution.control_dt_s))
    return runtime


def resolve_hoi_inputs(info):
    """解析并加载 HOI 任务模板；模板错误在设备连接前失败。"""
    from pathlib import Path

    if "object_pointcloud" not in info.observation_fields:
        return None
    if "point_cloud" in info.observation_fields:
        raise ValueError("HOI 与 point_cloud 不支持混合输入")
    name = info.task_name
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("非法任务名")
    root = Path(__file__).resolve().parents[2]
    from dexmani_real.geometry.hoi import load_task_template
    task = root / "assets/pointcloud_template/object" / name
    return task, load_task_template(task)
