"""Policy execution windows and runtime budgets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


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
    """None waits for operator exit; a limit counts begun episodes, not saves."""
    if num_episodes is None:
        return None
    if type(num_episodes) is not int:
        raise TypeError("num_episodes must be a positive integer or None")
    if num_episodes < 1:
        raise ValueError("num_episodes must be a positive integer")
    return num_episodes


@dataclass(frozen=True)
class ExecutionConfig:
    """Policy action horizon and runtime execution budgets."""

    # Effective A_exec, resolved from CLI, YAML or saved default at the Real entry.
    action_steps: int | None = None
    execution_mode: str = "sync"
    # Policy may start near HOME without relaxing the physical HOME convergence.
    start_arm_home_tolerance_deg: float = 0.5
    # Runtime age/wait budgets do not bound the physical stop response time.
    max_decision_age_s: float = 1.0
    max_wait_s: float = 2.0
    prefetch_steps: int | None = None
    rtc_guidance_cap: float | None = None

    def validate(self, info) -> None:
        """Validate Real budgets after PolicyInfo checks the execution mode."""
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

        for name in ("n_obs_steps", "horizon"):
            value = getattr(info, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if info.n_obs_steps > info.horizon:
            raise ValueError("n_obs_steps must not exceed horizon")

        tolerance = self.start_arm_home_tolerance_deg
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance)
            or tolerance <= 0
        ):
            raise ValueError("start_arm_home_tolerance_deg must be finite and positive")
        for name in ("max_decision_age_s", "max_wait_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or value is None or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive experimental budget")

        prediction_steps = info.horizon - info.n_obs_steps + 1
        action_steps = self.action_steps
        if type(action_steps) is not int:
            raise ValueError("Resolve effective A_exec at the deployment entry before validation")
        if not 1 <= action_steps <= prediction_steps:
            raise ValueError("execution requires 1 <= A_exec <= P=H-N+1")
        if self.execution_mode != "sync":
            prefetch_steps = self.prefetch_steps
            if (
                type(prefetch_steps) is not int
                or not 1 <= prefetch_steps <= action_steps
                or action_steps + prefetch_steps > prediction_steps
            ):
                raise ValueError(
                    "async/rtc require 1 <= d <= A_exec and A_exec+d <= P; "
                    f"A_exec={action_steps}, P={prediction_steps}, d={prefetch_steps}"
                )

        if self.execution_mode == "rtc":
            guidance_cap = self.rtc_guidance_cap
            if (
                isinstance(guidance_cap, bool)
                or guidance_cap is None
                or not math.isfinite(guidance_cap)
                or guidance_cap < 0
            ):
                raise ValueError("rtc_guidance_cap requires an explicit finite nonnegative value")
