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
from dexmani_real.dataset.contracts import (
    CANONICAL_MODALITY_SEMANTICS,
    FINGERTIP_KINEMATIC_MODEL,
)

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
    unsupported = names - _SUPPORTED_OBSERVATION_FIELDS
    if unsupported:
        raise ValueError(
            f"Real has no live producer for requested modalities: {sorted(unsupported)}"
        )
    if "joint_state" not in names:
        raise ValueError("Real deployment requires the joint_state observation")
    contracts = info.modality_contracts
    if not isinstance(contracts, dict):
        raise ValueError("Real deployment requires saved modality_contracts")
    action_key = {"joint": "action", "eef": "action_ee"}.get(info.action_mode)
    if action_key is None:
        raise ValueError(f"unsupported Real action mode: {info.action_mode!r}")
    for name in names | {action_key}:
        contract = contracts.get(name)
        if not isinstance(contract, dict):
            raise ValueError(f"Required modality {name!r} lacks its saved training contract")
        for key, expected in CANONICAL_MODALITY_SEMANTICS[name].items():
            if contract.get(key) != expected:
                raise ValueError(f"Live modality {name!r} cannot provide the saved {key}")
    if "fingertip_points" in names:
        recipe = contracts["fingertip_points"].get("recipe", {})
        if not isinstance(recipe, dict):
            raise ValueError("Live fingertip_points requires a saved FK recipe")
        links = recipe.get("fingertip_link_names")
        if (
            recipe.get("kinematic_model") != FINGERTIP_KINEMATIC_MODEL
            or recipe.get("mount_source") != "raw_episode"
            or not isinstance(links, list)
            or len(links) != 5
            or any(not isinstance(link, str) or not link for link in links)
            or len(set(links)) != 5
        ):
            raise ValueError("Live fingertip_points cannot provide the saved FK representation")
    policy_hz = 1.0 / dt
    limiting_hz = min(runtime.arm.loop_hz, runtime.hand.loop_hz)
    if policy_hz > limiting_hz and not math.isclose(policy_hz, limiting_hz, rel_tol=1e-9):
        raise ValueError(f"Policy rate {policy_hz:g} Hz exceeds worker rate {limiting_hz:g} Hz")
    if "point_cloud" in names:
        recipe = contracts["point_cloud"].get("recipe")
        if not isinstance(recipe, dict) or set(recipe) != set(PointCloudConfig().to_dict()):
            raise ValueError("Live point_cloud requires a complete saved numerical recipe")
        return PointCloudConfig.from_dict(recipe)
    return None


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
