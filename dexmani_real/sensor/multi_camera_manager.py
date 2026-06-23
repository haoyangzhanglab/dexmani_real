"""Multi-camera manager — N camera processes with N shared memory buffers.

Manages multiple RealSense cameras simultaneously, each in its own
crash-isolated process with a dedicated SharedMemoryRingBuffer.

Ref: ManiUniCon multi-camera design (configurable N camera daemons).

HDF5 layout extension:
    Recording with N cameras produces:
      /camera_0/rgb, depth, timestamps
      /camera_1/rgb, depth, timestamps
      ...
    (Backward compatible: single camera uses /camera/ as before.)

Usage:
    mgr = MultiCameraManager([
        CameraProcessConfig(serial="123456", shm_name="dexmani_cam_0"),
        CameraProcessConfig(serial="789012", shm_name="dexmani_cam_1"),
    ])
    mgr.start_all()

    # In controller loop:
    frames = mgr.read_all_latest()
    for name, frame in frames.items():
        if frame is not None:
            recorder.add_frame(..., camera_frame=frame, camera_name=name)

    mgr.stop_all()
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.log import get_logger

logger = get_logger(__name__)


@dataclass
class MultiCameraConfig:
    """Configuration for MultiCameraManager."""

    cameras: list = field(default_factory=list)  # list of CameraProcessConfig
    auto_restart: bool = True  # auto-restart crashed camera processes
    health_check_interval_s: float = 5.0  # how often to check camera health


class MultiCameraManager:
    """Manages multiple camera processes with shared memory.

    Each camera has:
      - A CameraProcess (with use_shm=True)
      - A named SharedMemoryRingBuffer
      - A health status
    """

    def __init__(self, configs: list, cfg: MultiCameraConfig | None = None) -> None:
        """
        Args:
            configs: List of CameraProcessConfig objects (one per camera).
            cfg: Optional MultiCameraConfig for global settings.
        """
        # Import here to avoid circular dependencies
        from dexmani_real.sensor.camera_process import CameraProcess, CameraProcessConfig

        self._configs = configs
        self._cfg = cfg or MultiCameraConfig()

        self._processes: list[CameraProcess] = []
        self._names: list[str] = []
        self._exit_stack: ExitStack | None = None

        self._last_health_check: float = 0.0

        # Build camera processes
        for i, cam_config in enumerate(configs):
            # Ensure shared memory mode is enabled and names are unique
            if isinstance(cam_config, CameraProcessConfig):
                c = cam_config
                c.use_shm = True
                if not c.shm_name or c.shm_name == "dexmani_cam_0":
                    c.shm_name = f"dexmani_cam_{i}"
            else:
                # Assume it's a dict-like object
                c = CameraProcessConfig(
                    camera_name=getattr(cam_config, "camera_name", f"camera_{i}"),
                    serial=getattr(cam_config, "serial", None),
                    hz=getattr(cam_config, "hz", 30.0),
                    warmup_frames=getattr(cam_config, "warmup_frames", 10),
                    timeout_ms=getattr(cam_config, "timeout_ms", 1000),
                    use_shm=True,
                    shm_name=f"dexmani_cam_{i}",
                )

            proc = CameraProcess(c)
            name = c.camera_name or f"camera_{i}"
            self._processes.append(proc)
            self._names.append(name)

        logger.info(
            "MultiCameraManager created: %d camera(s) — %s",
            len(self._processes),
            ", ".join(self._names),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_all(self) -> list[bool]:
        """Start all camera processes.

        Returns a list of booleans indicating success for each camera.
        """
        results = []
        for name, proc in zip(self._names, self._processes):
            ok = proc.start()
            results.append(ok)
            if ok:
                logger.info("  Camera '%s' started.", name)
            else:
                logger.warning("  Camera '%s' failed to start.", name)
        self._last_health_check = time.perf_counter()
        return results

    def stop_all(self, timeout: float = 3.0) -> None:
        """Stop all camera processes."""
        for name, proc in zip(self._names, self._processes):
            try:
                proc.stop(timeout=timeout)
                logger.info("  Camera '%s' stopped.", name)
            except (ValueError, RuntimeError) as e:
                logger.warning("  Camera '%s' stop error: %s", name, e)

    # ------------------------------------------------------------------
    # Context manager (ExitStack — auto-cleanup on scope exit)
    # ------------------------------------------------------------------

    def __enter__(self) -> MultiCameraManager:
        """Start all cameras and register cleanup via ExitStack.

        Usage:
            with MultiCameraManager(configs) as mgr:
                frames = mgr.read_all_latest()
                ...
            # All cameras auto-stopped on scope exit, even on exception.
        """
        self._exit_stack = ExitStack()
        self._exit_stack.callback(self.stop_all)
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop all cameras and release ExitStack resources."""
        if hasattr(self, "_exit_stack") and self._exit_stack is not None:
            self._exit_stack.close()
            self._exit_stack = None
        else:
            self.stop_all()
        return False  # don't suppress exceptions

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read_all_latest(self) -> dict[str, dict | None]:
        """Read the latest frame from all cameras.

        Returns a dict mapping camera name → camera frame dict (or None).
        """
        frames: dict[str, dict | None] = {}
        for name, proc in zip(self._names, self._processes):
            try:
                frames[name] = proc.poll_latest_frame()
            except (ValueError, RuntimeError, OSError):
                logger.debug("Camera '%s' poll failed.", name)
                frames[name] = None
        return frames

    def read_latest(self, camera_name: str) -> dict | None:
        """Read the latest frame from a specific camera by name."""
        for name, proc in zip(self._names, self._processes):
            if name == camera_name:
                try:
                    return proc.poll_latest_frame()
                except (ValueError, RuntimeError, OSError):
                    return None
        return None

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def check_health(self) -> dict[str, bool]:
        """Check health status of all cameras.

        Returns a dict mapping camera name → healthy (bool).
        """
        now = time.perf_counter()
        self._last_health_check = now

        health: dict[str, bool] = {}
        for name, proc in zip(self._names, self._processes):
            healthy = proc.running and not proc.crashed
            health[name] = healthy
            if not healthy and self._cfg.auto_restart:
                self._try_restart(name, proc)
        return health

    def _try_restart(self, name: str, proc) -> None:
        """Attempt to restart a crashed camera process."""
        logger.warning("Camera '%s' unhealthy — attempting restart.", name)
        try:
            proc.stop(timeout=1.0)
        except (ValueError, RuntimeError):
            pass
        time.sleep(0.5)
        if proc.start():
            logger.info("Camera '%s' restarted successfully.", name)
        else:
            logger.error("Camera '%s' restart failed.", name)

    @property
    def health_ok(self) -> bool:
        """True if all cameras are healthy."""
        return all(self.check_health().values())

    @property
    def camera_names(self) -> list[str]:
        return list(self._names)

    @property
    def n_cameras(self) -> int:
        return len(self._processes)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Get detailed status for all cameras."""
        status: dict[str, Any] = {}
        for name, proc in zip(self._names, self._processes):
            status[name] = {
                "running": proc.running,
                "crashed": proc.crashed,
            }
        return status
