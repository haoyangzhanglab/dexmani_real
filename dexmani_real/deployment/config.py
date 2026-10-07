"""Validate policy requirements against current physical and numerical configuration.

Saved Policy config supplies modalities, cadence and the numerical cloud recipe.
Current Real config owns physical capability; this module imports neither Policy nor Torch.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dexmani_real.config.pointcloud import PointCloudConfig

_SUPPORTED_OBSERVATION_FIELDS = frozenset(
    {
        "joint_state",
        "point_cloud",
        "rgb",
        "contact_force",
        "fingertip_points",
        "eef_pose",
        "tactile_force",
    }
)


def with_action_steps(config: dict, n_action_steps: int | None) -> dict:
    """Override only deployment chunk length, preserving the training snapshot."""
    result = deepcopy(config)
    if n_action_steps is None:
        return result
    if type(n_action_steps) is not int or n_action_steps < 1:
        raise ValueError("n_action_steps must be a positive integer")
    agent = result["agent"]
    if agent["n_obs_steps"] - 1 + n_action_steps > agent["horizon"]:
        raise ValueError("n_obs_steps - 1 + n_action_steps must not exceed horizon")
    agent["n_action_steps"] = n_action_steps
    result["n_action_steps"] = n_action_steps
    return result


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


def validate_policy_runtime_compatibility(info: Any, runtime: Any) -> PointCloudConfig | None:
    """Check physical capability before allocating IPC or connecting devices."""
    if not runtime.policy.hand_enabled:
        raise ValueError("Dexterous policy deployment requires hand_enabled=true")
    control_dt_ns(info)
    names = set(info.observation_fields)
    unsupported = names - _SUPPORTED_OBSERVATION_FIELDS
    if unsupported:
        raise ValueError(
            f"Real has no live producer for requested modalities: {sorted(unsupported)}"
        )
    if info.action_mode not in {"joint", "eef"}:
        raise ValueError(f"unsupported Real action mode: {info.action_mode!r}")
    if "fingertip_points" in names:
        links = info.fingertip_link_names
        if (
            not isinstance(links, (list, tuple))
            or len(links) != 5
            or any(not isinstance(link, str) or not link.strip() for link in links)
            or len(set(links)) != 5
        ):
            raise ValueError("Live fingertip_points requires five distinct non-empty link names")
    if "point_cloud" in names:
        from dexmani_real.config.pointcloud import PointCloudConfig

        recipe = info.pointcloud_config
        if not isinstance(recipe, dict):
            raise ValueError("Live point_cloud requires saved numerical parameters")
        missing = {field.name for field in fields(PointCloudConfig)} - recipe.keys()
        if missing:
            raise ValueError(f"Saved pointcloud recipe missing fields: {sorted(missing)}")
        return PointCloudConfig.from_dict(recipe)
    return None


def control_dt_ns(info):
    dt = info.control_dt_s
    if (
        isinstance(dt, bool)
        or not isinstance(dt, (int, float))
        or not math.isfinite(dt)
        or dt <= 0
        or not math.isfinite(dt * 1e9)
        or int(dt * 1e9) <= 0
    ):
        raise ValueError(
            "Real deployment requires saved real_runtime.dt with positive control_dt_s"
        )
    return int(dt * 1e9)


@dataclass(frozen=True)
class PolicyRuntimeConfig:
    """Local model loading arguments; config includes explicit deployment overrides."""

    config: dict
    info: Any
    device: str
    seed: int


def validate_num_episodes(num_episodes: Any) -> int:
    """Validate the run-owner episode budget (episodes to run, not saves)."""
    if isinstance(num_episodes, bool) or type(num_episodes) is not int:
        raise TypeError("num_episodes must be a positive integer")
    if num_episodes < 1:
        raise ValueError("num_episodes must be a positive integer")
    return int(num_episodes)


@dataclass(frozen=True)
class RolloutRecordingConfig:
    """Recording inputs for a physical rollout, with an absolute session data_dir.

    The CLI validates paths before the session starts. Episode counts and budgets
    belong to lifecycle configuration. Runtime records technical stop reasons;
    task success is judged offline from raw episodes.
    """

    data_dir: str
    task_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.data_dir, str) or not self.data_dir.strip():
            raise ValueError("rollout data_dir must be a non-empty path")
        raw_data_dir = Path(self.data_dir)
        resolved_data_dir = raw_data_dir.resolve(strict=False)
        if resolved_data_dir == resolved_data_dir.parent:
            raise ValueError("rollout data_dir must not be a filesystem root")
        if not isinstance(self.task_label, str) or not self.task_label.strip():
            raise ValueError("rollout task_label must be non-empty")
        if self.task_label != self.task_label.strip():
            raise ValueError("rollout task_label must not have surrounding whitespace")
        object.__setattr__(self, "data_dir", str(resolved_data_dir))


@dataclass(frozen=True)
class ExecutionConfig:
    execution_mode: str = "sync"
    max_decision_age_s: float | None = None
    max_wait_s: float | None = None
    max_tick_lateness_s: float | None = None
    prefetch_steps: int | None = None
    rtc_guidance_cap: float | None = None

    def validate(self, info, *, max_running_s=None):
        dt = control_dt_ns(info)
        max_running_s = validate_max_running_s(max_running_s)
        for name in ("n_obs_steps", "horizon", "n_action_steps"):
            value = getattr(info, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if info.n_obs_steps > info.horizon:
            raise ValueError("n_obs_steps must not exceed horizon")
        if self.execution_mode not in {"sync", "async", "rtc"}:
            raise ValueError("execution_mode must be sync, async or rtc")
        for name in ("max_decision_age_s", "max_wait_s", "max_tick_lateness_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or value is None
                or not math.isfinite(value)
                or value < 0
                or (value == 0 and name != "max_tick_lateness_s")
            ):
                raise ValueError(f"{name} requires an explicit finite positive experimental budget")
        lateness = int(self.max_tick_lateness_s * 1e9)
        if lateness >= dt:
            raise ValueError("max_tick_lateness_s must be smaller than control_dt_s")
        p = info.horizon - info.n_obs_steps + 1
        a = info.n_action_steps
        if not 1 <= a <= p:
            raise ValueError("sync requires 1 <= A <= P=H-N+1")
        if self.execution_mode != "sync":
            d = self.prefetch_steps
            if type(d) is not int or not 1 <= d <= a or a + d > p:
                raise ValueError(f"async/rtc require 1 <= d <= A and A+d <= P; A={a}, P={p}, d={d}")
        if self.execution_mode == "rtc":
            beta = self.rtc_guidance_cap
            if isinstance(beta, bool) or beta is None or not math.isfinite(beta) or beta < 0:
                raise ValueError("rtc_guidance_cap requires an explicit finite nonnegative value")
        _validate_wait_lower_bound(info.n_obs_steps * dt, self, max_running_s, "Static")
        age = a * dt - lateness
        if self.execution_mode != "sync":
            age = max(age, (a + self.prefetch_steps - 1) * dt - lateness)
        if int(self.max_decision_age_s * 1e9) < age:
            raise ValueError("Static grid lower bound exceeds max_decision_age_s")
        return self


def _validate_wait_lower_bound(bound, execution, max_running_s, source):
    if int(execution.max_wait_s * 1e9) <= bound:
        raise ValueError(f"{source} first-action grid lower bound reaches max_wait_s")
    if max_running_s is not None and int(max_running_s * 1e9) <= bound:
        raise ValueError(f"{source} first-action grid lower bound reaches max_running_s")


def validate_warmup_budget(durations, execution_config, info, *, max_running_s=None):
    """Necessary grid bounds only; measured samples are not realtime guarantees."""
    dt = control_dt_ns(info)
    lateness = int(execution_config.max_tick_lateness_s * 1e9)
    age = int(execution_config.max_decision_age_s * 1e9)
    for path in ("bootstrap", "steady"):
        samples = durations[path]
        if not samples or any(not math.isfinite(i) or i < 0 for i in samples):
            raise ValueError(f"Invalid {path} warmup durations")
    for sample in durations["bootstrap"]:
        inference = int(sample * 1e9)
        k = inference // dt + 1
        _validate_wait_lower_bound(
            (info.n_obs_steps - 1 + k) * dt, execution_config, max_running_s, "Measured sample"
        )
        if max(inference, k * dt - lateness) + (info.n_action_steps - 1) * dt > age:
            raise ValueError("Measured bootstrap sample cannot fit max_decision_age_s")
    if execution_config.execution_mode != "sync":
        d = execution_config.prefetch_steps
        for sample in durations["steady"]:
            inference = int(sample * 1e9)
            if inference > d * dt + lateness:
                raise ValueError("Measured steady sample cannot fit handoff deadline")
            if max(inference, (info.n_action_steps + d - 1) * dt - lateness) > age:
                raise ValueError("Measured steady sample cannot fit max_decision_age_s")
