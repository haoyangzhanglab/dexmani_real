"""CollectionConfig — data collection parameters.

Single config object for the teleop collection pipeline. Serialized into
HDF5 /meta for full reproducibility of recording sessions.

Ref: data collection loop design — Phase 2.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from dexmani_real.config.pipeline_config import DEFAULT_MAX_RECORD_FRAMES

__all__ = ["CollectionConfig", "DEFAULT_MAX_RECORD_FRAMES"]


@dataclass
class CollectionConfig:
    """Parameters governing teleop data collection lifecycle."""

    # ── Episode metadata ──
    task_label: str = "teleop"
    operator: str = ""
    tags: list[str] = field(default_factory=list)

    # ── Recording constraints ──
    max_frames: int = DEFAULT_MAX_RECORD_FRAMES  # hard cap per episode
    min_frames: int = 50                          # minimum frames for valid episode

    # ── Camera ──
    camera_enabled: bool = True
    camera_recovery_enabled: bool = True  # auto-restart crashed camera daemon
    camera_max_age_s: float = 0.5         # CAMERA_OK freshness threshold

    # ── Quality gating ──
    min_quality_ratio: float = 0.6  # minimum ratio of quality-OK frames
    record_all_frames: bool = True   # record even low-quality frames (filter offline)

    # ── Auto-stop triggers ──
    auto_stop_on_max_frames: bool = True
    auto_stop_on_quality_drop: bool = False       # abort if quality drops too low
    quality_drop_threshold: float = 0.5            # minimum quality ratio before triggering
    quality_drop_streak: int = 100                 # consecutive low-quality frames to trigger

    # ── Episode annotation (post-recording) ──
    annotate_on_save: bool = True   # run EpisodeAnnotator on stop_episode (sidecar JSON)
    success_default: bool = True    # default success flag (can override per-episode)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
