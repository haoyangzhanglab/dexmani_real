"""Physical rollout recording inputs and task outcome metadata."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

from dexmani_real.recording.recorder import normalize_provenance_metadata
from dexmani_real.utils.atomic_io import atomic_json_dump

EVALUATION_MAX_FRAMES_STOP_REASON = "eval:invalid:max_frames"


class EvaluationOutcome(IntEnum):
    """Operator-visible task outcomes, independent of storage commitment."""

    NONE = 0
    SUCCESS = 1
    FAILURE = 2
    INVALID = 3
    STOPPED = 4


def evaluation_outcome_stop_reason(outcome: EvaluationOutcome) -> str:
    """Return the stable recorder reason for an explicit operator outcome."""
    reasons = {
        EvaluationOutcome.SUCCESS: "eval:success:operator",
        EvaluationOutcome.FAILURE: "eval:failure:operator",
        EvaluationOutcome.INVALID: "eval:invalid:operator",
        EvaluationOutcome.STOPPED: "run:stopped:operator",
    }
    try:
        return reasons[outcome]
    except KeyError as exc:
        raise ValueError("NONE is not a terminal evaluation outcome") from exc


@dataclass(frozen=True)
class RolloutRecordingConfig:
    """Resolved recording inputs shared by physical run and eval.

    ``data_dir`` is already an isolated absolute output directory.  The CLI
    owns selector/path validation and checkpoint hashing before workers start;
    this pickle-safe object carries only recording and result inputs.
    """

    data_dir: str
    task_label: str
    operator: str
    max_running_s: float
    mode: str = "eval"
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"run", "eval"}:
            raise ValueError("recorded rollout mode must be run or eval")
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
    "RolloutRecordingConfig",
    "evaluation_outcome_stop_reason",
    "write_rollout_result",
]


def write_rollout_result(
    config: RolloutRecordingConfig,
    episode_path: str | Path,
    *,
    outcome: EvaluationOutcome,
    stop_reason: str,
    duration_s: float,
    metrics: Mapping[str, int | float],
    saved: bool,
) -> Path:
    """Write task outcome separately from the raw storage commit flag."""
    path = Path(episode_path)
    # Failed storage still leaves the rollout's result, without pretending raw exists.
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        **dict(config.provenance),
        "mode": config.mode,
        "outcome": outcome.name.lower(),
        "stop_reason": stop_reason,
        "eval_seed": int(config.provenance["eval_seed"]),
        "duration_s": duration_s,
        "raw_saved": bool(saved),
        "metrics": metrics,
    }
    return atomic_json_dump(payload, path / "result.json", indent=2, ensure_ascii=False)
