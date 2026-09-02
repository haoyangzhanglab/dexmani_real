"""Strict physical learned-policy run intent loaded from one YAML profile."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from dexmani_real.deployment.task_scene import TaskSceneCard, load_task_scene_card

_PROFILE_V1_KEYS = frozenset(
    {
        "schema_version",
        "experiment_dir",
        "runtime_config",
        "deployment_config",
        "device",
        "seed",
        "hand_acknowledged",
        "expected_checkpoint_sha256",
        "max_running_seconds",
        "acknowledgement_timeout_seconds",
        "max_published_endpoints",
    }
)
_PROFILE_V2_KEYS = _PROFILE_V1_KEYS | {"task_scene_card"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class PhysicalRunProfile:
    """Operator-owned physical intent; model dimensions remain artifact-owned."""

    source_path: Path
    experiment_dir: Path
    runtime_config: Path | None
    deployment_config: Path | None
    device: str
    seed: int
    hand_acknowledged: bool
    expected_checkpoint_sha256: str
    max_running_seconds: float
    acknowledgement_timeout_seconds: float
    max_published_endpoints: int
    task_scene_card: TaskSceneCard | None


def _require_path(
    value: Any, *, name: str, profile_dir: Path, optional: bool
) -> Path | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        qualifier = "a path string or null" if optional else "a non-empty path string"
        raise TypeError(f"profile {name} must be {qualifier}")
    path = Path(value)
    if not path.is_absolute():
        path = profile_dir / path
    return path.resolve(strict=False)


def _require_positive_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"profile {name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"profile {name} must be a finite positive number")
    return result


def load_physical_run_profile(path: str | Path) -> PhysicalRunProfile:
    """Load one exact-key physical profile with profile-relative paths.

    Schema v1 remains the H4-only historical profile. Schema v2 adds a frozen
    task scene card; ``run`` requires it before any hardware lifecycle starts.
    """
    source_path = Path(path).resolve(strict=True)
    if source_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("physical profile path must use a .yaml or .yml suffix")
    try:
        with source_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"physical profile is invalid YAML: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise TypeError("physical profile root must be a mapping")
    schema_version = loaded.get("schema_version")
    if isinstance(schema_version, bool) or schema_version not in {1, 2}:
        raise ValueError("physical profile schema_version must be 1 or 2")
    expected_keys = _PROFILE_V1_KEYS if schema_version == 1 else _PROFILE_V2_KEYS
    actual_keys = frozenset(str(key) for key in loaded)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unknown = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"physical profile keys mismatch: missing={missing}, unknown={unknown}"
        )
    device = loaded["device"]
    if not isinstance(device, str) or not device or device != device.strip():
        raise ValueError("profile device must be a non-empty trimmed string")
    seed = loaded["seed"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 2**31 - 1
    ):
        raise ValueError("profile seed must be an integer in [0, 2**31 - 1]")
    hand_acknowledged = loaded["hand_acknowledged"]
    if hand_acknowledged is not True:
        raise ValueError("physical profile hand_acknowledged must be true")
    expected_sha256 = loaded["expected_checkpoint_sha256"]
    if (
        not isinstance(expected_sha256, str)
        or _SHA256_RE.fullmatch(expected_sha256) is None
    ):
        raise ValueError(
            "profile expected_checkpoint_sha256 must be a lowercase 64-hex digest"
        )
    endpoints = loaded["max_published_endpoints"]
    if isinstance(endpoints, bool) or not isinstance(endpoints, int) or endpoints <= 0:
        raise ValueError("profile max_published_endpoints must be a positive integer")
    profile_dir = source_path.parent
    experiment_dir = _require_path(
        loaded["experiment_dir"],
        name="experiment_dir",
        profile_dir=profile_dir,
        optional=False,
    )
    assert experiment_dir is not None
    task_scene_card: TaskSceneCard | None = None
    if schema_version == 2:
        scene_card_path = _require_path(
            loaded["task_scene_card"],
            name="task_scene_card",
            profile_dir=profile_dir,
            optional=False,
        )
        assert scene_card_path is not None
        task_scene_card = load_task_scene_card(scene_card_path)
    return PhysicalRunProfile(
        source_path=source_path,
        experiment_dir=experiment_dir,
        runtime_config=_require_path(
            loaded["runtime_config"],
            name="runtime_config",
            profile_dir=profile_dir,
            optional=True,
        ),
        deployment_config=_require_path(
            loaded["deployment_config"],
            name="deployment_config",
            profile_dir=profile_dir,
            optional=True,
        ),
        device=device,
        seed=seed,
        hand_acknowledged=hand_acknowledged,
        expected_checkpoint_sha256=expected_sha256,
        max_running_seconds=_require_positive_number(
            loaded["max_running_seconds"], name="max_running_seconds"
        ),
        acknowledgement_timeout_seconds=_require_positive_number(
            loaded["acknowledgement_timeout_seconds"],
            name="acknowledgement_timeout_seconds",
        ),
        max_published_endpoints=endpoints,
        task_scene_card=task_scene_card,
    )
