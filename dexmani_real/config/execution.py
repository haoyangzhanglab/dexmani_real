"""Declared policy execution windows and budgets, validated by deployment preflight."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionConfig:
    """Policy action horizon and runtime execution budgets."""

    # Effective A_exec, resolved from CLI, YAML or saved default at the Real entry.
    # Ordinary async uses this nominal replan length; late inference may consume more old actions.
    action_steps: int | None = None
    execution_mode: str = "sync"
    # Policy may start near HOME without relaxing the physical HOME convergence.
    start_arm_home_tolerance_deg: float = 0.5
    # Runtime age/wait budgets do not bound the physical stop response time.
    max_decision_age_s: float = 1.0
    max_wait_s: float = 2.0
    prefetch_steps: int | None = None
    rtc_guidance_cap: float | None = None
