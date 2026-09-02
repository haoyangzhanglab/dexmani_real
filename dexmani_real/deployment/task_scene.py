"""Operator-owned scene card for one bounded physical task run.

The learned-policy artifact says nothing about the physical object or target
that happens to be in front of the camera.  A task scene card binds those
operator facts to the run profile and gives the passive diagnostics collector
four explicitly chosen image/point-cloud milestones.  It is deliberately
separate from the model/runtime projection: changing a scene description must
never alter a learned action or a safety limit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_CARD_KEYS = frozenset(
    {
        "schema_version",
        "task_name",
        "object_description",
        "object_start_description",
        "target_description",
        "success_criterion",
        "phase_endpoint_indices",
    }
)
_PHASE_NAMES = ("approach", "grasp", "lift", "place")


@dataclass(frozen=True)
class TaskSceneCard:
    """Immutable operator declarations and capture milestones for one task."""

    source_path: Path
    sha256: str
    task_name: str
    object_description: str
    object_start_description: str
    target_description: str
    success_criterion: str
    phase_endpoint_indices: tuple[tuple[str, int], ...]

    @property
    def phase_indices(self) -> dict[str, int]:
        return dict(self.phase_endpoint_indices)

    def validate_for_task(
        self, *, task_name: str, max_published_endpoints: int
    ) -> None:
        """Check task identity and that all selected captures fit the bound."""
        if self.task_name != task_name:
            raise ValueError(
                "task scene card task_name does not match the policy artifact: "
                f"{self.task_name!r} != {task_name!r}"
            )
        if max_published_endpoints <= 1:
            raise ValueError("task scene card requires a multi-endpoint task bound")
        previous = 0
        for phase, index in self.phase_endpoint_indices:
            if not 1 <= index <= max_published_endpoints:
                raise ValueError(
                    f"scene card {phase} endpoint index {index} is outside "
                    f"1..{max_published_endpoints}"
                )
            if index <= previous:
                raise ValueError(
                    "scene card phase endpoint indices must be strictly increasing"
                )
            previous = index

    def provenance(self) -> dict[str, object]:
        """Return the compact immutable identity bound into the task receipt."""
        return {
            "path": str(self.source_path),
            "sha256": self.sha256,
            "task_name": self.task_name,
            "phase_endpoint_indices": dict(self.phase_endpoint_indices),
        }


def _require_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"task scene card {name} must be non-empty trimmed text")
    return value


def load_task_scene_card(path: str | Path) -> TaskSceneCard:
    """Load an exact-key JSON scene card and bind its source digest."""
    source_path = Path(path).resolve(strict=True)
    if source_path.suffix.lower() != ".json":
        raise ValueError("task scene card path must use a .json suffix")
    payload = source_path.read_bytes()
    try:
        loaded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"task scene card is invalid JSON: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise TypeError("task scene card root must be a JSON object")
    actual_keys = frozenset(str(key) for key in loaded)
    if actual_keys != _CARD_KEYS:
        missing = sorted(_CARD_KEYS - actual_keys)
        unknown = sorted(actual_keys - _CARD_KEYS)
        raise ValueError(
            f"task scene card keys mismatch: missing={missing}, unknown={unknown}"
        )
    if loaded["schema_version"] != 1 or isinstance(loaded["schema_version"], bool):
        raise ValueError("task scene card schema_version must be exactly 1")
    phase_values = loaded["phase_endpoint_indices"]
    if not isinstance(phase_values, Mapping):
        raise TypeError("task scene card phase_endpoint_indices must be an object")
    if frozenset(str(key) for key in phase_values) != frozenset(_PHASE_NAMES):
        raise ValueError(
            "task scene card phase_endpoint_indices must contain exactly "
            f"{list(_PHASE_NAMES)}"
        )
    phases: list[tuple[str, int]] = []
    for phase in _PHASE_NAMES:
        index = phase_values[phase]
        if isinstance(index, bool) or not isinstance(index, int) or index <= 0:
            raise ValueError(
                f"task scene card {phase} endpoint index must be a positive integer"
            )
        phases.append((phase, index))
    return TaskSceneCard(
        source_path=source_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        task_name=_require_text(loaded["task_name"], name="task_name"),
        object_description=_require_text(
            loaded["object_description"], name="object_description"
        ),
        object_start_description=_require_text(
            loaded["object_start_description"], name="object_start_description"
        ),
        target_description=_require_text(
            loaded["target_description"], name="target_description"
        ),
        success_criterion=_require_text(
            loaded["success_criterion"], name="success_criterion"
        ),
        phase_endpoint_indices=tuple(phases),
    )
