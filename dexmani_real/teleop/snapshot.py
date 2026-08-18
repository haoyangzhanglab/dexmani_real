"""Camera freshness tracking for the recording grid.

The causal ring readers live in ``shm/causal_reader.py``; compatibility aliases
remain here for existing teleop imports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from dexmani_real.shm.causal_reader import (
    read_arm_state_causal as _read_arm_state,
    read_camera_frame_causal as _read_camera_frame,
    read_hand_state_causal as _read_hand_state,
    read_hand_tactile_causal as _read_hand_tactile,
    read_vr_frame_causal as _read_vr_frame,
)

__all__ = [
    "CameraFreshnessTracker",
    "_read_arm_state",
    "_read_camera_frame",
    "_read_hand_state",
    "_read_hand_tactile",
    "_read_vr_frame",
]


@dataclass
class CameraFreshnessTracker:
    """Classify camera frames on the recording grid and detect source stalls."""

    max_age_s: float
    abort_after_s: float
    episode_started_s: float = 0.0
    last_ring_sequence: int = 0
    last_frame_number: int = 0
    stale_since_s: float | None = None

    def reset(self, episode_started_s: float) -> None:
        self.episode_started_s = float(episode_started_s)
        self.last_ring_sequence = 0
        self.last_frame_number = 0
        self.stale_since_s = None

    def observe(self, frame: dict | None, now_s: float | None = None) -> tuple[dict | None, bool]:
        now = time.monotonic() if now_s is None else float(now_s)
        fresh = False
        if frame is not None:
            sequence = int(frame.get("ring_sequence", 0))
            frame_number = int(frame.get("frame_number", 0))
            source_ns = int(frame.get("source_monotonic_ns", 0))
            source_s = source_ns / 1e9 if source_ns > 0 else float(frame.get("capture_monotonic_s", np.nan))
            age_s = max(0.0, now - source_s) if np.isfinite(source_s) else float("inf")
            is_new = (
                sequence > 0
                and sequence != self.last_ring_sequence
                and frame_number > 0
                and frame_number != self.last_frame_number
            )
            after_episode_start = np.isfinite(source_s) and source_s >= self.episode_started_s
            healthy = int(frame.get("camera_health", 1)) == 0
            fresh = is_new and after_episode_start and healthy and age_s <= self.max_age_s
            if sequence > 0 and sequence != self.last_ring_sequence:
                self.last_ring_sequence = sequence
                self.last_frame_number = frame_number
            frame["camera_age_s"] = age_s
            frame["camera_fresh"] = fresh

        if fresh:
            self.stale_since_s = None
        elif self.stale_since_s is None:
            self.stale_since_s = now

        stalled = self.stale_since_s is not None and now - self.stale_since_s >= self.abort_after_s
        return frame, stalled
