"""Narrow formal-evaluation contract shared across deployment layers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

from dexmani_real.recording.recorder import normalize_provenance_metadata


EVALUATION_MAX_FRAMES_STOP_REASON = "eval:invalid:max_frames"


class EvaluationOutcome(IntEnum):
    """The only operator-visible outcome labels for a formal rollout."""

    NONE = 0
    SUCCESS = 1
    FAILURE = 2
    INVALID = 3


def evaluation_outcome_stop_reason(outcome: EvaluationOutcome) -> str:
    """Return the stable recorder reason for an explicit operator outcome."""
    reasons = {
        EvaluationOutcome.SUCCESS: "eval:success:operator",
        EvaluationOutcome.FAILURE: "eval:failure:operator",
        EvaluationOutcome.INVALID: "eval:invalid:operator",
    }
    try:
        return reasons[outcome]
    except KeyError as exc:
        raise ValueError("NONE is not a terminal evaluation outcome") from exc


@dataclass(frozen=True)
class PolicyEvaluationConfig:
    """Resolved formal-evaluation inputs, separate from ordinary deployment.

    ``data_dir`` is already an isolated absolute output directory.  The CLI
    owns selector/path validation and checkpoint hashing before workers start;
    this pickle-safe object carries only the resulting evaluation contract.
    """

    data_dir: str
    task_label: str
    operator: str
    max_running_s: float
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data_dir, str) or not self.data_dir.strip():
            raise ValueError("evaluation data_dir must be a non-empty path")
        raw_data_dir = Path(self.data_dir)
        if not raw_data_dir.is_absolute():
            raise ValueError("evaluation data_dir must be absolute")
        resolved_data_dir = raw_data_dir.resolve(strict=False)
        if resolved_data_dir == resolved_data_dir.parent:
            raise ValueError("evaluation data_dir must not be a filesystem root")
        for field_name in ("task_label", "operator"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"evaluation {field_name} must be non-empty")
            if value != value.strip():
                raise ValueError(
                    f"evaluation {field_name} must not have surrounding whitespace"
                )
        if isinstance(self.max_running_s, bool):
            raise TypeError("evaluation max_running_s must be a finite positive number")
        timeout_s = float(self.max_running_s)
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("evaluation max_running_s must be finite and positive")
        object.__setattr__(self, "data_dir", str(resolved_data_dir))
        object.__setattr__(self, "max_running_s", timeout_s)
        object.__setattr__(
            self,
            "provenance",
            normalize_provenance_metadata(self.provenance),
        )


__all__ = [
    "EVALUATION_MAX_FRAMES_STOP_REASON",
    "EvaluationOutcome",
    "PolicyEvaluationConfig",
    "evaluation_outcome_stop_reason",
]
