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
    camera_max_age_s: float = 0.5         # camera frame freshness threshold (seconds)

    # ── Recording ──
    record_all_frames: bool = True   # record all frames

    # ── Auto-stop triggers ──
    auto_stop_on_max_frames: bool = True

    # ── Episode file classification (Phase 1.2) ──
    # After stop_episode(), files are moved to success_dir / failure_dir
    # based on the classification parameter.  None disables move (backward compat).
    # Ref: T-Rex data_writer.py move_episode_files().
    success_dir: str | None = None   # e.g. "data/success"
    failure_dir: str | None = None   # e.g. "data/failure"

    # ── Sidecar JSON (Phase 1.2) ──
    # When True, writes an episode_NNN.json sidecar next to the HDF5 file.
    save_sidecar_json: bool = True

    # ── Pre-record buffer (Phase 3.1) ──
    # Seconds of frames to buffer before the Record key is pressed.
    # When > 0, InMemoryFrameBuffer runs a ring buffer in pre-record mode,
    # flushing buffered frames to HDF5 on start_episode().
    pre_record_duration_s: float = 0.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
