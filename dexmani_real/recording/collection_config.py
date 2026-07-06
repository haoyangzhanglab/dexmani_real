"""CollectionConfig — data collection parameters (simplified)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from dexmani_real.config.pipeline_config import DEFAULT_MAX_RECORD_FRAMES

__all__ = ["CollectionConfig", "DEFAULT_MAX_RECORD_FRAMES"]


@dataclass
class CollectionConfig:
    """Parameters governing teleop data collection lifecycle."""

    task_label: str = "teleop"
    operator: str = ""
    tags: list[str] = field(default_factory=list)

    max_frames: int = DEFAULT_MAX_RECORD_FRAMES
    min_frames: int = 50

    camera_enabled: bool = True
    camera_recovery_enabled: bool = True
    camera_max_age_s: float = 0.5

    record_all_frames: bool = True
    auto_stop_on_max_frames: bool = True
    save_sidecar_json: bool = True

    # When True, all non-camera streams are aligned to a unified dt=20ms time grid
    # at record time via TimestampAlignedBuffer and flushed to HDF5 in bulk at
    # episode stop.  Camera frames are stored per-frame as before (too large for
    # pre-allocation).  When False (default), every frame is appended to HDF5
    # individually with raw timestamps.
    use_timestamp_buffer: bool = False

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
