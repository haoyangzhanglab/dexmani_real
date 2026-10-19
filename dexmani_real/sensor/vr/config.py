"""HTS receiver transport parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VRParams:
    """HTS transport for right-hand observations and cached head tracking."""

    transport: str = "tcp_server"
    host: str = "0.0.0.0"
    port: int = 8000

    def validate(self) -> None:
        if not self.transport or not self.host:
            raise ValueError("VR transport and host must be non-empty")
        if not (1 <= self.port <= 65535):
            raise ValueError("VR port must be in [1, 65535]")
