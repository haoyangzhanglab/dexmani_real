"""Validate policy requirements against actuator worker service rates.

Saved Policy config supplies modalities, cadence and the numerical cloud recipe.
Current Real config owns physical capability; this module imports neither Policy nor Torch.
"""

from __future__ import annotations

import math
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
    """Check physical capability before allocating IPC or starting workers."""
    if not runtime.policy.hand_enabled:
        raise ValueError("Dexterous policy deployment requires hand_enabled=true")
    dt = info.control_dt_s
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Real deployment requires saved real_runtime with positive control_dt_s")
    names = set(info.observation_fields)
    if not names <= _SUPPORTED_OBSERVATION_FIELDS or "joint_state" not in names:
        raise ValueError("Real supports only configured modalities including joint_state")
    policy_hz = 1.0 / dt
    limiting_hz = min(runtime.arm.loop_hz, runtime.hand.loop_hz)
    if policy_hz > limiting_hz and not math.isclose(policy_hz, limiting_hz, rel_tol=1e-9):
        raise ValueError(f"Policy rate {policy_hz:g} Hz exceeds worker rate {limiting_hz:g} Hz")
    return PointCloudConfig.from_dict(info.pointcloud_config) if "point_cloud" in names else None


def validate_recording_budget(info, max_running_s):
    from dexmani_real.recording.recorder import HARD_MAX_RECORD_FRAMES

    duration = validate_max_running_s(max_running_s)
    if duration is None:
        raise ValueError("Recorded evaluation requires a finite duration")
    rows = math.ceil(duration / info.control_dt_s)
    if rows >= HARD_MAX_RECORD_FRAMES:
        raise ValueError(f"Recording requests {rows} rows; require rows < {HARD_MAX_RECORD_FRAMES}")


@dataclass(frozen=True)
class PolicyRuntimeConfig:
    """Spawn arguments; config is the already-read saved experiment snapshot."""

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

    The CLI validates paths before workers start. Episode counts and budgets
    belong to lifecycle configuration. Runtime records technical stop reasons;
    task success is judged offline from raw episodes.
    """

    data_dir: str
    task_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.data_dir, str) or not self.data_dir.strip():
            raise ValueError("rollout data_dir must be a non-empty path")
        raw_data_dir = Path(self.data_dir)
        if not raw_data_dir.is_absolute():
            raise ValueError("rollout data_dir must be absolute")
        resolved_data_dir = raw_data_dir.resolve(strict=False)
        if resolved_data_dir == resolved_data_dir.parent:
            raise ValueError("rollout data_dir must not be a filesystem root")
        if not isinstance(self.task_label, str) or not self.task_label.strip():
            raise ValueError("rollout task_label must be non-empty")
        if self.task_label != self.task_label.strip():
            raise ValueError("rollout task_label must not have surrounding whitespace")
        object.__setattr__(self, "data_dir", str(resolved_data_dir))
