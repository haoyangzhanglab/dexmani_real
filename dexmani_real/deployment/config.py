"""Validate policy requirements against current physical and numerical configuration.

Saved Policy config supplies modalities, cadence and the numerical cloud recipe.
Current Real config owns physical capability; this module imports neither Policy nor Torch.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    dt = info.control_dt_s
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Real deployment requires saved real_runtime with positive control_dt_s")
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
        recipe = info.pointcloud_config
        if not isinstance(recipe, dict):
            raise ValueError("Live point_cloud requires saved numerical parameters")
        return PointCloudConfig.from_dict(recipe)
    return None


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

    def validate(self, info):
        if self.execution_mode not in {"sync", "async", "rtc"}:
            raise ValueError("execution_mode must be sync, async or rtc")
        for name in ("max_decision_age_s", "max_wait_s", "max_tick_lateness_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or value is None or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} requires an explicit finite positive experimental budget")
        if self.max_tick_lateness_s >= info.control_dt_s:
            raise ValueError("max_tick_lateness_s must be smaller than control_dt_s")
        p = info.horizon - info.n_obs_steps + 1
        a = info.n_action_steps
        if not 1 <= a <= p:
            raise ValueError("sync requires 1 <= A <= P=H-N+1")
        if self.execution_mode != "sync":
            d = self.prefetch_steps
            if type(d) is not int or not 1 <= d <= a or a + d > p:
                raise ValueError(f"async/rtc require 1 <= d <= A and A+d <= P; A={a}, P={p}, d={d}")
            if self.max_decision_age_s <= (a + d - 1) * info.control_dt_s:
                raise ValueError("decision age budget cannot cover the asynchronous A+d slots")
        elif self.max_decision_age_s <= (a - 1) * info.control_dt_s:
            raise ValueError("decision age budget cannot cover the synchronous A slots")
        if self.execution_mode == "rtc":
            beta = self.rtc_guidance_cap
            if isinstance(beta, bool) or beta is None or not math.isfinite(beta) or beta < 0:
                raise ValueError("rtc_guidance_cap requires an explicit finite nonnegative value")
        return self
