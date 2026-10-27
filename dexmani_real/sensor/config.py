from __future__ import annotations

from dataclasses import dataclass

OBSERVATION_MAX_AGE_PERIODS = 4


@dataclass(frozen=True)
class CameraConfig:
    serial: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    l515_visual_preset: int = 5
    l515_confidence_threshold: int | None = None

    @property
    def max_frame_age_s(self) -> float:
        return OBSERVATION_MAX_AGE_PERIODS / self.fps

    @property
    def depth_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def validate(self) -> None:
        for name in ("width", "height", "fps"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"camera {name} must be a positive integer")
        if type(self.l515_visual_preset) is not int or not 0 <= self.l515_visual_preset <= 5:
            raise ValueError("camera l515_visual_preset must be an integer in [0, 5]")
        if self.l515_confidence_threshold is not None and (
            type(self.l515_confidence_threshold) is not int
            or not 0 <= self.l515_confidence_threshold <= 3
        ):
            raise ValueError("camera l515_confidence_threshold must be in [0, 3] or null")
        if self.serial is not None and (
            not isinstance(self.serial, str) or not self.serial.strip()
        ):
            raise ValueError("camera serial must be a non-empty string or None")


@dataclass(frozen=True)
class VRConfig:
    transport: str = "tcp_server"
    host: str = "0.0.0.0"
    port: int = 8000

    def validate(self) -> None:
        if not self.transport or not self.host:
            raise ValueError("VR transport and host must be non-empty")
        if not (1 <= self.port <= 65535):
            raise ValueError("VR port must be in [1, 65535]")
