"""CollectionConfig — data collection parameters (simplified)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

# Default max frames per recording episode — single source of truth
# used by both CollectionConfig and EpisodeRecorder.
DEFAULT_MAX_RECORD_FRAMES: int = 10000

__all__ = ["CollectionConfig", "DEFAULT_MAX_RECORD_FRAMES"]


@dataclass
class CollectionConfig:
    """Parameters governing teleop data collection lifecycle."""

    task_label: str = "teleop"
    operator: str = ""
    tags: list[str] = field(default_factory=list)

    max_frames: int = DEFAULT_MAX_RECORD_FRAMES
    min_frames: int = 50

    # Drop the first N frames of each episode to skip the begin-transition pose
    # noise (0.2s at 50Hz). Set 0 to record from the first frame.
    skip_initial_frames: int = 10

    auto_stop_on_max_frames: bool = True
    save_sidecar_json: bool = True

    # Run DataValidator on the finished .h5 in the writer thread at stop.
    validate_on_stop: bool = False

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
