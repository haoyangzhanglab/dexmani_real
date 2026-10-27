"""解析训练参数与真机配置，在模型和设备启动前校验。"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING

from dexmani_real.perception.config import HandCloudConfig, PointCloudConfig
from dexmani_real.runtime.preflight import validate_control_rate, validate_robot_config

if TYPE_CHECKING:
    from dexmani_real.config import ExperimentConfig
    from dexmani_real.runtime.observation import ObservationNeeds

DEFAULT_NUM_EPISODES = 20
DEFAULT_MAX_RUNNING_S = 60.0


@dataclass(frozen=True)
class ExecutionConfig:
    # async 可继续使用旧预测的有效尾部，action_steps 是名义执行窗口。
    action_steps: int | None = None
    execution_mode: str = "sync"
    # 开始策略的容差独立于 HOME 的物理收敛容差。
    start_arm_home_tolerance_deg: float = 0.5
    # 时效与等待预算不代表物理停止延迟。
    max_decision_age_s: float = 1.0
    max_wait_s: float = 2.0
    prefetch_steps: int | None = None
    rtc_guidance_cap: float | None = None


def _restore_cloud_config(recipe, config_type):
    if not isinstance(recipe, dict) or recipe.keys() != {f.name for f in fields(config_type)}:
        raise ValueError(
            f"Saved {config_type.__name__} is incomplete or obsolete; "
            "re-export the dataset and retrain the checkpoint"
        )
    return config_type(**recipe)


def validate_execution_config(info, execution: ExecutionConfig) -> None:
    if execution.execution_mode not in {"sync", "async", "rtc"}:
        raise ValueError("execution_mode must be sync, async or rtc")
    if execution.execution_mode not in info.supported_execution_modes:
        raise NotImplementedError(f"Policy does not support {execution.execution_mode} execution")

    dt = info.control_dt_s
    if (
        type(dt) not in (int, float)
        or not math.isfinite(dt * 1e9)
        or int(dt * 1e9) <= 0
    ):
        raise ValueError("Real deployment requires finite positive control_dt_s of at least 1 ns")
    for name in ("start_arm_home_tolerance_deg", "max_decision_age_s", "max_wait_s"):
        value = getattr(execution, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    prediction_steps = info.future_steps
    action_steps = execution.action_steps
    if type(action_steps) is not int or not 1 <= action_steps <= prediction_steps:
        raise ValueError("execution requires integer 1 <= A_exec <= policy future_steps")
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
        cap = execution.rtc_guidance_cap
        if (
            (type(cap) is not int and type(cap) is not float)
            or not math.isfinite(cap) or cap < 0
        ):
            raise ValueError("rtc_guidance_cap requires an explicit finite nonnegative value")


def resolve_deployment_config(
    info, runtime: ExperimentConfig, needs: ObservationNeeds,
) -> ExperimentConfig:
    action_steps = runtime.execution.action_steps
    execution = replace(
        runtime.execution,
        action_steps=info.n_action_steps if action_steps is None else action_steps,
    )
    validate_execution_config(info, execution)
    names = set(info.observation_fields)
    if info.hand_history_steps is not None and (
        info.action_mode != "joint"
        or "joint_state" not in names
        or (info.horizon, info.n_obs_steps, info.n_action_steps, execution.action_steps)
        != (1, 1, 1, 1)
    ):
        raise ValueError("LAMP hand history requires joint, joint_state, H=Nobs=C=A_exec=1")
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
    if "depth" in names and tuple(info.depth_config["shape"]) != runtime.camera.depth_shape:
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
