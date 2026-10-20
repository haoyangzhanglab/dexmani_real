"""Camera capture parameters and derived host freshness budget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OBSERVATION_MAX_AGE_PERIODS = 4


@dataclass(frozen=True)
class CameraParams:
    serial: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    warmup_frames: int = 10
    source_stall_timeout_s: float = 2.0
    l515_visual_preset: int = 5
    l515_confidence_threshold: int | None = None
    # Keep the SDK queue small to favor fresh feedback over frame retention.
    frame_queue_capacity: int = 2
    ring_maxlen: int = 5

    @property
    def max_frame_age_s(self) -> float:
        return OBSERVATION_MAX_AGE_PERIODS / self.fps

    @property
    def rgb_shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, 3)

    @property
    def depth_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def validate(self) -> None:
        for name in ("width", "height", "fps", "frame_queue_capacity", "ring_maxlen"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"camera {name} must be a positive integer")
        if type(self.warmup_frames) is not int or self.warmup_frames < 0:
            raise ValueError("camera warmup_frames must be a nonnegative integer")
        if (
            not np.isfinite(self.source_stall_timeout_s)
            or self.source_stall_timeout_s <= self.max_frame_age_s
        ):
            raise ValueError(
                "camera source_stall_timeout_s must be finite and greater than derived max frame age"
            )
        if (
            not isinstance(self.l515_visual_preset, int)
            or isinstance(self.l515_visual_preset, bool)
            or not 0 <= self.l515_visual_preset <= 5
        ):
            raise ValueError("camera l515_visual_preset must be an integer in [0, 5]")
        if self.l515_confidence_threshold is not None and (
            not isinstance(self.l515_confidence_threshold, int)
            or isinstance(self.l515_confidence_threshold, bool)
            or not 0 <= self.l515_confidence_threshold <= 3
        ):
            raise ValueError("camera l515_confidence_threshold must be in [0, 3] or null")
        if self.serial is not None and (
            not isinstance(self.serial, str) or not self.serial.strip()
        ):
            raise ValueError("camera serial must be a non-empty string or None")
