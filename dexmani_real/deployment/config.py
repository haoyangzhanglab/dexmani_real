"""Validate policy requirements against current physical and numerical configuration.

Saved Policy config supplies modalities, cadence and the numerical cloud recipe.
Current Real config owns physical capability; this module imports neither Policy nor Torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
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


def resolve_execution_config(execution, info, n_action_steps=None):
    """Resolve A_exec once at the Real entry; never rewrite saved model config."""
    value = info.n_action_steps if n_action_steps is None else n_action_steps
    if type(value) is not int or value < 1:
        raise ValueError("A_exec must be a positive integer")
    return replace(execution, action_steps=value)


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
    """Local model loading arguments; config retains saved model semantics."""

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
    """Output directory and task label for optional Raw rollout recording."""

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
    """Policy startup admission and execution budgets."""

    # Effective A_exec, resolved from CLI or saved default at the Real entry.
    action_steps: int | None = None
    execution_mode: str = "sync"
    # Policy may start near HOME without relaxing the physical HOME convergence.
    start_arm_home_tolerance_deg: float = 0.5
    # Per-face allowance for the final FK target; nominal command bounds stay fixed.
    eef_workspace_margin_m: float = 0.005
    # Recheck these admission budgets against Policy cadence and measured costs;
    # they do not bound the physical stop response time.
    max_decision_age_s: float = 1.0
    max_wait_s: float = 2.0
    max_tick_lateness_s: float = 0.03
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
        tolerance = self.start_arm_home_tolerance_deg
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance)
            or tolerance <= 0
        ):
            raise ValueError("start_arm_home_tolerance_deg must be finite and positive")
        margin = self.eef_workspace_margin_m
        if (
            isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not math.isfinite(margin)
            or margin < 0
        ):
            raise ValueError("eef_workspace_margin_m must be finite and nonnegative")
        for name in ("max_decision_age_s", "max_wait_s", "max_tick_lateness_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or value is None
                or not math.isfinite(value)
                or value < 0
                or (value == 0 and name != "max_tick_lateness_s")
            ):
                bound = "nonnegative" if name == "max_tick_lateness_s" else "positive"
                raise ValueError(f"{name} must be a finite {bound} experimental budget")
        lateness = int(self.max_tick_lateness_s * 1e9)
        if lateness >= dt:
            raise ValueError("max_tick_lateness_s must be smaller than control_dt_s")
        p = info.horizon - info.n_obs_steps + 1
        a = self.action_steps
        if type(a) is not int:
            raise ValueError("Resolve effective A_exec at the deployment entry before validation")
        if not 1 <= a <= p:
            raise ValueError("execution requires 1 <= A_exec <= P=H-N+1")
        if self.execution_mode != "sync":
            d = self.prefetch_steps
            if type(d) is not int or not 1 <= d <= a or a + d > p:
                raise ValueError(f"async/rtc require 1 <= d <= A_exec and A_exec+d <= P; A_exec={a}, P={p}, d={d}")
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
    """Model-only necessary bounds; input/prefix preparation and margin also need
    time inside the nominal d*dt window. Owner lateness never extends it.
    """
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
        if max(inference, k * dt - lateness) + (execution_config.action_steps - 1) * dt > age:
            raise ValueError("Measured bootstrap sample cannot fit max_decision_age_s")
    if execution_config.execution_mode != "sync":
        d = execution_config.prefetch_steps
        for sample in durations["steady"]:
            inference = int(sample * 1e9)
            if inference >= d * dt:
                raise ValueError("Model-only sample leaves no time for input/prefix preparation and handoff margin")
            if max(inference, (execution_config.action_steps + d - 1) * dt - lateness) > age:
                raise ValueError("Measured steady sample cannot fit max_decision_age_s")
