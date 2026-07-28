"""Protocol definitions for camera drivers.

Enables dependency injection in ``CameraProcess``: ``RealSense`` already
satisfies ``CameraDriver``, and the protocol allows mock-camera integration
tests and future non-RealSense backends without touching the capture loop.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class CameraDriver(Protocol):
    """Structural interface for camera backends.

    ``CameraProcess._run()`` depends on this protocol, not on the concrete
    ``RealSense`` class, so tests can inject a mock and non-RealSense
    backends (ZED, OAK-D, …) can slot in without forking the process logic.

    All attributes / methods below reflect the subset of ``RealSense`` that
    ``_run()`` actually calls.
    """

    # -- attributes set after connect() --
    active_serial: str | None
    K: np.ndarray | None

    # -- lifecycle --
    def connect(self) -> bool:
        """Open the device pipeline.  Returns True on success."""
        ...

    def disconnect(self) -> None:
        """Close the pipeline; idempotent."""
        ...

    # -- frame capture --
    def read(self, timeout_ms: int = 5000, *, compute_depth: bool = True) -> Any:
        """Read one (RGB, depth) frame pair.  Blocking up to *timeout_ms*.

        Returns a ``CameraFrame`` (frozen dataclass with ``rgb``, ``depth``,
        ``depth_raw``, ``timestamp``, ``frame_id``, ``host_time``, ``serial``,
        ``K``, ``intr``, ``intrinsics_info``, ``depth_scale``, ``align_mode``,
        ``frame_name`` fields).
        """
        ...

    # -- metadata --
    def get_depth_scale(self) -> float:
        """Depth-unit scale factor (meters per raw unit)."""
        ...

    def get_rays(self) -> np.ndarray | None:
        """Camera rays for pointcloud deprojection, or None if unavailable."""
        ...
